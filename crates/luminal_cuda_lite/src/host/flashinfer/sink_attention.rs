//! `SinkAttention` — host op wrapping the FA3/Hopper AttentionSink paged
//! batch-prefill kernel (rung two of the gpt-oss attention ladder; the raw
//! kernel was validated in `fa3.rs`).
//!
//! Runtime inputs (7): `q` (s, nq*hd) bf16, `k_pool`/`v_pool`
//! (num_slots, nkv*hd) bf16 (post-scatter pool states), `kv_indices` (c,)
//! Int (compact page table, page_size 1), `qo_indptr`/`kv_indptr` (r,) Int
//! on DEVICE (read back per execute — no caching in v1; the ripped spike's
//! readback cache was a crash suspect), `sinks` (nq,) F32.
//!
//! Output: (nq, s, hd) F32 — the layout+dtype of the reference chain's
//! attention output point, produced by a fused transpose+upcast kernel.
//! Decode is the same kernel at qo_len=1 (no SM90 decode-with-sink kernel).
//!
//! The rewrite rule (sink_attention.egg) matches the paged gpt-oss sink
//! attention chain and unions this op in; which host-mask Input feeds the
//! chain ("mask_sliding" vs "mask_full") selects window_left. Disable with
//! LUMINAL_DISABLE_SINK_ATTENTION=1.

use std::sync::Arc;

use luminal::{
    egglog_utils::api::{Rule, SortDef, sort},
    egglog_utils::base::{EXPRESSION, F64, OP_KIND},
    egglog_utils::{SerializedEGraph, extract_expr},
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::cudarc::driver::{CudaStream, DevicePtr, result};

use super::super::{DeviceBuffer, HostOp};
use super::jit;
use super::{
    INT_WORKSPACE_SIZE, PAGE_LOCKED_WORKSPACE, PageLockedPtr, bytes_to_i32_vec, cuda_pin_memory,
    flashinfer_workspaces,
};

#[derive(Debug)]
pub struct SinkAttention {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The 's' (batch tokens) dimension expression.
    pub batch_dim: Expression,
    /// Softmax scale; 0.0 = default `1/sqrt(head_dim)`.
    pub sm_scale: f64,
    /// FlashInfer window_left convention (visible previous positions);
    /// -1 = full attention. Selects the swa .so variant at compile time.
    pub window_left: i64,
}

impl Default for SinkAttention {
    fn default() -> Self {
        Self {
            num_qo_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            batch_dim: Expression::default(),
            sm_scale: 0.0,
            window_left: -1,
        }
    }
}

impl EgglogOp for SinkAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "SinkAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("batch_dim", EXPRESSION),
                ("sm_scale", F64),
                ("window_left", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        // q, k_pool, v_pool, kv_indices, qo_indptr, kv_indptr, sinks
        7
    }

    fn rewrites(&self) -> Vec<Rule> {
        // The FA3 kernels are Hopper-only (sm_90a WGMMA/TMA): emit no rules
        // on other architectures so the search never selects the op there.
        if crate::device_compute_major() != 9 {
            return vec![];
        }
        // Kill switch for A/B-ing compile cost and for emergencies.
        if std::env::var("LUMINAL_DISABLE_SINK_ATTENTION").is_ok_and(|v| v == "1") {
            return vec![];
        }
        vec![Rule::raw(include_str!("sink_attention.egg"))]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let num_qo_heads = extract_expr(egraph, kind_children[0], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let num_kv_heads = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let head_dim = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let batch_dim = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let sm_scale: f64 = egraph.enodes[kind_children[4]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let window_left = egraph.enodes[kind_children[5]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            .round() as i64;

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            batch_dim,
            sm_scale,
            window_left,
        };

        // JIT at extract time so the ~45s nvcc cost never lands inside a
        // GA profiling trial (same rationale as FlashInferAttention).
        let _ = jit::ensure_compiled_fa3(head_dim, window_left >= 0);

        // The rule passes the FLAT gather index (proof anchor); recover the
        // compact per-token page table the kernel consumes.
        let flat_idx_node = input_enodes[3];
        let gather_idx = super::find_indptrs::try_find_compact_gather_idx(egraph, flat_idx_node)
            .expect("SinkAttention matched a gather without recoverable compact gather_idx");
        let final_inputs = vec![
            input_enodes[0], // q (bf16)
            input_enodes[1], // k_pool
            input_enodes[2], // v_pool
            gather_idx,      // compact kv_indices
            input_enodes[4], // qo_indptr
            input_enodes[5], // kv_indptr
            input_enodes[6], // sinks (f32)
        ];

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for SinkAttention {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            inputs.len() >= 7,
            "SinkAttention expects 7 inputs, got {}",
            inputs.len()
        );
        let buf = |n: NodeIndex, what: &str| -> anyhow::Result<DeviceBuffer> {
            buffers
                .get(&n)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("SinkAttention: missing buffer for {what}"))
        };
        let q = buf(inputs[0], "q")?;
        let k_pool = buf(inputs[1], "k_pool")?;
        let v_pool = buf(inputs[2], "v_pool")?;
        let kv_indices = buf(inputs[3], "kv_indices")?;
        let qo_indptr_buf = buf(inputs[4], "qo_indptr")?;
        let kv_indptr_buf = buf(inputs[5], "kv_indptr")?;
        let sinks = buf(inputs[6], "sinks")?;
        let out = buf(self_node, "output")?;

        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        // Read the indptrs back to the host: the FA3 plan is host-side.
        // Two tiny DtoH syncs per layer per tick — correctness-first v1.
        let read_device_i32s = |b: DeviceBuffer| -> anyhow::Result<Vec<i32>> {
            let mut host_bytes = vec![0u8; b.len()];
            unsafe {
                result::memcpy_dtoh_async(&mut host_bytes, b.ptr(), stream.cu_stream())?;
            }
            stream.synchronize()?;
            Ok(bytes_to_i32_vec(host_bytes))
        };
        let mut qo_indptr = read_device_i32s(qo_indptr_buf)?;
        let mut kv_indptr = read_device_i32s(kv_indptr_buf)?;
        anyhow::ensure!(
            qo_indptr.len() == kv_indptr.len() && qo_indptr.len() >= 2,
            "SinkAttention: malformed indptrs (qo len {}, kv len {})",
            qo_indptr.len(),
            kv_indptr.len()
        );
        let batch_size = qo_indptr.len() - 1;
        let nnz_qo = *qo_indptr.last().unwrap() as usize;
        let total_pages = *kv_indptr.last().unwrap() as usize;
        anyhow::ensure!(
            kv_indices.len() >= total_pages * std::mem::size_of::<i32>(),
            "SinkAttention: kv_indices buffer smaller than kv_indptr total"
        );
        // page_size = 1: per-sequence kv length in tokens == pages.
        let mut kv_len_arr: Vec<i32> = kv_indptr.windows(2).map(|w| w[1] - w[0]).collect();

        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = flashinfer_workspaces(stream);
        let page_locked = PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
            assert_eq!(status, 0, "Failed to allocate page-locked workspace");
            let cuda_status = cuda_pin_memory(ptr, INT_WORKSPACE_SIZE);
            assert_eq!(cuda_status, 0, "Failed to pin memory");
            PageLockedPtr(ptr as *mut u8)
        });

        let mut plan_info = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = unsafe {
            (lib.prefill_plan)(
                float_ws_ptr as *mut std::ffi::c_void,
                super::FLOAT_WORKSPACE_SIZE,
                int_ws_ptr as *mut std::ffi::c_void,
                page_locked.0 as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                qo_indptr.as_mut_ptr(),
                kv_indptr.as_mut_ptr(),
                kv_len_arr.as_mut_ptr(),
                nnz_qo as i32,
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                /*page_size=*/ 1,
                /*causal=*/ 1,
                cu_stream,
                plan_info.as_mut_ptr(),
                &mut plan_info_len,
            )
        };
        anyhow::ensure!(plan_ret == 0, "SinkAttention: fa3 plan failed ({plan_ret})");

        // Kernel-native (s, heads, dim) bf16 scratch, then fused
        // transpose+upcast into the (heads, s, dim) F32 output buffer.
        let temp_bytes = (nnz_qo * self.num_qo_heads * self.head_dim * 2).max(1);
        let temp = unsafe { stream.alloc::<u8>(temp_bytes)? };
        let temp_ptr = temp.device_ptr(stream).0;

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let run_ret = unsafe {
            (lib.prefill_run)(
                int_ws_ptr as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info_len,
                q.ptr() as *mut std::ffi::c_void,
                k_pool.ptr() as *mut std::ffi::c_void,
                v_pool.ptr() as *mut std::ffi::c_void,
                kv_indices.ptr() as *mut i32,
                sinks.ptr() as *mut f32,
                temp_ptr as *mut std::ffi::c_void,
                std::ptr::null_mut(), // lse
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                /*page_size=*/ 1,
                jit::FlashInferDType::Bf16 as i32,
                sm_scale,
                self.window_left as i32,
                /*causal=*/ 1,
                cu_stream,
            )
        };
        anyhow::ensure!(run_ret == 0, "SinkAttention: fa3 run failed ({run_ret})");

        let tr_ret = unsafe {
            (lib.transpose_output_f32)(
                temp_ptr as *const std::ffi::c_void,
                out.ptr() as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.head_dim as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(tr_ret == 0, "SinkAttention: output transpose failed");

        // The temp scratch must outlive the async kernels enqueued above.
        stream.synchronize()?;
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4 // F32 output
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("SinkAttention")
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_ref::{
        RefSeq, assert_close, deterministic_f32, reference_attention, round_to_bf16, to_bf16_bytes,
    };
    use super::*;
    use crate::cudarc::driver::{CudaContext, DevicePtr};

    /// Shares the pinned plan scratch with the fa3 kernel tests — serialize.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const HEAD_DIM: usize = 64;
    const BF16_RTOL: f32 = 3e-2;
    const BF16_ATOL: f32 = 3e-3;

    fn hopper_gpu() -> bool {
        if CudaContext::new(0).is_err() {
            return false;
        }
        crate::device_compute_major() == 9
    }

    /// Drive one case through SinkAttention::execute (the op boundary, not
    /// the raw lib) and compare against the CPU oracle. K/V live in a slot
    /// pool larger than the context, with sequences' tokens scattered.
    fn run_op_case(label: &str, seqs: &[RefSeq], window_left: i64, sinks_base: f32, seed: u64) {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let (nq, nkv) = (64usize, 8usize);
        let total_qo: usize = seqs.iter().map(|s| s.qo_len).sum();
        let total_kv: usize = seqs.iter().map(|s| s.kv_len).sum();

        let q = round_to_bf16(&deterministic_f32(total_qo * nq * HEAD_DIM, seed));
        let k = round_to_bf16(&deterministic_f32(total_kv * nkv * HEAD_DIM, seed + 1));
        let v = round_to_bf16(&deterministic_f32(total_kv * nkv * HEAD_DIM, seed + 2));
        let sinks: Vec<f32> = (0..nq).map(|h| sinks_base + h as f32 * 0.13).collect();

        // Oracle output is (s, heads, dim); the op emits (heads, s, dim).
        let want_bhd = reference_attention(
            &q,
            &k,
            &v,
            &sinks,
            seqs,
            nq,
            nkv,
            HEAD_DIM,
            window_left,
            1.0 / (HEAD_DIM as f32).sqrt(),
        );
        let mut want = vec![0.0f32; want_bhd.len()];
        for b in 0..total_qo {
            for h in 0..nq {
                let src = &want_bhd[(b * nq + h) * HEAD_DIM..][..HEAD_DIM];
                want[(h * total_qo + b) * HEAD_DIM..][..HEAD_DIM].copy_from_slice(src);
            }
        }

        // Slot pool with a scattered layout (pool 2x the context; logical
        // token i at slot 2i+1 — non-contiguous like the real engine).
        let row = nkv * HEAD_DIM;
        let num_slots = total_kv * 2;
        let mut k_pool = vec![0.0f32; num_slots * row];
        let mut v_pool = vec![0.0f32; num_slots * row];
        let mut kv_indices_host = Vec::with_capacity(total_kv);
        for i in 0..total_kv {
            let slot = 2 * i + 1;
            k_pool[slot * row..(slot + 1) * row].copy_from_slice(&k[i * row..][..row]);
            v_pool[slot * row..(slot + 1) * row].copy_from_slice(&v[i * row..][..row]);
            kv_indices_host.push(slot as i32);
        }

        let mut qo_indptr: Vec<i32> = vec![0];
        let mut kv_indptr: Vec<i32> = vec![0];
        for s in seqs {
            qo_indptr.push(qo_indptr.last().unwrap() + s.qo_len as i32);
            kv_indptr.push(kv_indptr.last().unwrap() + s.kv_len as i32);
        }

        let up_bytes = |b: &[u8]| stream.clone_htod(b).unwrap();
        let d_q = up_bytes(&to_bf16_bytes(&q));
        let d_k = up_bytes(&to_bf16_bytes(&k_pool));
        let d_v = up_bytes(&to_bf16_bytes(&v_pool));
        let d_idx = stream.clone_htod(&kv_indices_host).unwrap();
        let d_qo = stream.clone_htod(&qo_indptr).unwrap();
        let d_kv = stream.clone_htod(&kv_indptr).unwrap();
        let d_sinks = stream.clone_htod(&sinks).unwrap();
        let d_out = up_bytes(&vec![0u8; total_qo * nq * HEAD_DIM * 4]);

        let inputs: Vec<NodeIndex> = (0..7).map(NodeIndex::new).collect();
        let self_node = NodeIndex::new(99);
        let mut buffers: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        buffers.insert(
            inputs[0],
            DeviceBuffer::new(d_q.device_ptr(&stream).0, d_q.len()),
        );
        buffers.insert(
            inputs[1],
            DeviceBuffer::new(d_k.device_ptr(&stream).0, d_k.len()),
        );
        buffers.insert(
            inputs[2],
            DeviceBuffer::new(d_v.device_ptr(&stream).0, d_v.len()),
        );
        buffers.insert(
            inputs[3],
            DeviceBuffer::new(d_idx.device_ptr(&stream).0, d_idx.len() * 4),
        );
        buffers.insert(
            inputs[4],
            DeviceBuffer::new(d_qo.device_ptr(&stream).0, d_qo.len() * 4),
        );
        buffers.insert(
            inputs[5],
            DeviceBuffer::new(d_kv.device_ptr(&stream).0, d_kv.len() * 4),
        );
        buffers.insert(
            inputs[6],
            DeviceBuffer::new(d_sinks.device_ptr(&stream).0, d_sinks.len() * 4),
        );
        buffers.insert(
            self_node,
            DeviceBuffer::new(d_out.device_ptr(&stream).0, d_out.len()),
        );

        let op = SinkAttention {
            num_qo_heads: nq,
            num_kv_heads: nkv,
            head_dim: HEAD_DIM,
            batch_dim: 's'.into(),
            sm_scale: 0.0,
            window_left,
        };
        op.execute(&stream, self_node, &inputs, &buffers, &FxHashMap::default())
            .unwrap();
        stream.synchronize().unwrap();

        let got: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&stream.clone_dtoh(&d_out).unwrap()).to_vec();
        assert_close(&got, &want, BF16_RTOL, BF16_ATOL, label);
    }

    #[test]
    fn sink_attention_op_prefill() {
        if !hopper_gpu() {
            return;
        }
        run_op_case(
            "op prefill 64/8",
            &[RefSeq {
                qo_len: 16,
                kv_len: 48,
            }],
            -1,
            0.5,
            101,
        );
    }

    #[test]
    fn sink_attention_op_ragged_decode() {
        if !hopper_gpu() {
            return;
        }
        run_op_case(
            "op ragged decode",
            &[
                RefSeq {
                    qo_len: 1,
                    kv_len: 61,
                },
                RefSeq {
                    qo_len: 16,
                    kv_len: 80,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 250,
                },
            ],
            -1,
            0.8,
            202,
        );
    }

    #[test]
    fn sink_attention_op_sliding_window() {
        if !hopper_gpu() {
            return;
        }
        run_op_case(
            "op sliding window",
            &[
                RefSeq {
                    qo_len: 4,
                    kv_len: 300,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 200,
                },
            ],
            127,
            0.5,
            303,
        );
    }
}
