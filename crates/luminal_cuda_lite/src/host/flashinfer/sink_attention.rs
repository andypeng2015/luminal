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

/// Grow-only device scratch (q transpose in, kernel output out), reused
/// across calls instead of a per-call alloc + trailing sync. Stream-ordered
/// reuse on the same stream is safe (each execute's indptr-readback sync
/// drains prior consumers); on a stream change (tests) the old buffers are
/// leaked rather than dropped, since their context may be gone. Bounded by
/// the largest tick, so leaking on replace/grow costs nothing real.
static SCRATCH: std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>)>> =
    std::sync::Mutex::new(None);

/// Split-KV decode scratch (same grow-only, stream-keyed, leak-on-replace
/// idiom as SCRATCH). Sections, 256-B aligned: expanded q [P·H·D bf16],
/// partial outputs [P·H·D bf16], partial LSEs [P·H f32], token map [P i32],
/// merge indptr [B+1 i32], fake sinks [H × -1e30 f32].
static SPLIT_SCRATCH: std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>)>> =
    std::sync::Mutex::new(None);

/// Split-KV decode schedule: how one pure-decode tick's sequences partition
/// into KV chunks. All host-side; built once per tick from the cached
/// indptrs. Chunk k of sequence b covers a CONTIGUOUS range of the original
/// kv_indices buffer (page_size = 1), so only new indptrs are needed —
/// kv_indices is never rewritten.
struct SplitSchedule {
    /// [0, 1, .., P] — one single-token query row per chunk.
    qo_indptr: Vec<i32>,
    /// [P+1] chunk boundaries into the ORIGINAL kv_indices buffer.
    kv_indptr: Vec<i32>,
    /// [P] per-chunk KV length in tokens (== pages at page_size 1).
    kv_len_arr: Vec<i32>,
    /// [P] chunk → source sequence row (pure decode: q row == seq index).
    token_map: Vec<i32>,
    /// [B+1] chunk ranges per sequence, for the merge kernel.
    merge_indptr: Vec<i32>,
}

fn iceil(a: i32, b: i32) -> i32 {
    debug_assert!(a >= 0 && b > 0);
    (a + b - 1) / b
}

const SPLIT_CHUNK_ALIGN: i32 = 128;
const SPLIT_MIN_CHUNK: i32 = 256;
const SPLIT_MAX_PER_SEQ: i32 = 32;

fn split_kv_min() -> i32 {
    std::env::var("LUMINAL_SPLIT_KV_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}

/// Decide whether (and how) to split a tick's KV. Pure host function:
/// `sm_count` is a parameter so tests don't need a GPU. Returns None when the
/// tick isn't pure decode, the longest context is below the threshold, or
/// splitting wouldn't create any extra parallelism (P == B) — which is how
/// splitting naturally shuts off at high concurrency, where batch × kv-heads
/// CTAs already fill the GPU.
fn compute_split_schedule(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    sm_count: i32,
    num_kv_heads: i32,
) -> Option<SplitSchedule> {
    if std::env::var("LUMINAL_DISABLE_SPLIT_KV").is_ok_and(|v| v == "1") {
        return None;
    }
    let b = qo_indptr.len().checked_sub(1)?;
    // w[1] - w[0] == 1 for all query_index_pointers means we are in a decode only
    // situation, so no need to do splitting
    if b == 0 || !qo_indptr.windows(2).all(|w| w[1] - w[0] == 1) {
        return None;
    }
    let kv_len: Vec<i32> = kv_indptr.windows(2).map(|w| w[1] - w[0]).collect();
    if kv_len.iter().any(|&l| l <= 0) {
        return None;
    }
    if *kv_len.iter().max().unwrap() < split_kv_min() {
        return None;
    }
    // Global chunk size targeting ~2 work items per SM once each chunk fans
    // out across the KV heads.
    let total_kv = *kv_indptr.last().unwrap();
    let p_target = iceil(2 * sm_count, num_kv_heads.max(1)).max(1);
    let chunk = iceil(iceil(total_kv, p_target), SPLIT_CHUNK_ALIGN)
        .saturating_mul(SPLIT_CHUNK_ALIGN)
        .max(SPLIT_MIN_CHUNK);

    let mut qo = vec![0i32];
    let mut kvp = vec![0i32];
    let mut kv_len_arr = Vec::new();
    let mut token_map = Vec::new();
    let mut merge_indptr = vec![0i32];
    for (bi, &len) in kv_len.iter().enumerate() {
        let splits = iceil(len, chunk).clamp(1, SPLIT_MAX_PER_SEQ);
        // Ceil-division: every chunk (incl. the last) has >= 1 token, so no
        // partial can produce a -inf LSE.
        let per = iceil(len, splits);
        let base = kv_indptr[bi];
        for j in 0..splits {
            let end = ((j + 1) * per).min(len);
            kvp.push(base + end);
            kv_len_arr.push(end - j * per);
            qo.push(*qo.last().unwrap() + 1);
            token_map.push(bi as i32);
        }
        merge_indptr.push(*qo.last().unwrap());
    }
    if token_map.len() == b {
        return None; // one chunk per seq: nothing gained
    }
    Some(SplitSchedule {
        qo_indptr: qo,
        kv_indptr: kvp,
        kv_len_arr,
        token_map,
        merge_indptr,
    })
}

/// Byte offsets of the SPLIT_SCRATCH sections for a given schedule size.
/// Deterministic in (p, b, h, d).
struct SplitOffsets {
    q_exp: usize,
    partial_out: usize,
    partial_lse: usize,
    token_map: usize,
    merge_indptr: usize,
    neg_sinks: usize,
    total: usize,
}

fn split_offsets(p: usize, b: usize, h: usize, d: usize) -> SplitOffsets {
    let align = |x: usize| (x + 255) & !255;
    let q_exp = 0;
    let partial_out = q_exp + align(p * h * d * 2);
    let partial_lse = partial_out + align(p * h * d * 2);
    let token_map = partial_lse + align(p * h * 4);
    let merge_indptr = token_map + align(p * 4);
    let neg_sinks = merge_indptr + align((b + 1) * 4);
    let total = neg_sinks + align(h * 4);
    SplitOffsets {
        q_exp,
        partial_out,
        partial_lse,
        token_map,
        merge_indptr,
        neg_sinks,
        total,
    }
}

/// Grow-only, stream-keyed device buffer (SCRATCH idiom): returns the base
/// pointer, reallocating (and leaking the old buffer — its context may be
/// gone and in-flight kernels may still read it) when the stream changes or
/// the requested size outgrows it. Callers sync before growth matters; the
/// refresh path always syncs first.
fn grow_only_buffer(
    slot: &std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>)>>,
    stream: &Arc<CudaStream>,
    bytes: usize,
) -> anyhow::Result<u64> {
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    let stream_key = stream.cu_stream() as usize;
    let needs_new = !matches!(&*guard,
        Some((key, buf)) if *key == stream_key && buf.len() >= bytes);
    if needs_new {
        stream.synchronize()?;
        let buf = unsafe { stream.alloc::<u8>(bytes.next_power_of_two().max(256))? };
        if let Some((_, old)) = guard.take() {
            std::mem::forget(old);
        }
        *guard = Some((stream_key, buf));
    }
    Ok(guard.as_ref().unwrap().1.device_ptr(stream).0)
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn device_sm_count() -> i32 {
    static SM_COUNT: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *SM_COUNT.get_or_init(|| {
        crate::cudarc::driver::CudaContext::new(0)
            .ok()
            .and_then(|ctx| {
                ctx.attribute(
                    crate::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
                .ok()
            })
            .unwrap_or(132) // H100 SXM
    })
}

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

        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = flashinfer_workspaces(stream);

        // Read the indptrs back to the host: the FA3 plan is host-side and
        // runs once per execute. The first read's synchronize also drains
        // the previous execute's async traffic out of the shared pinned plan
        // buffer and the scratch pools before this call rewrites them.
        // (A per-tick plan cache existed briefly — d6d022be — but its
        // tick-detection machinery wasn't worth ~2.6ms/tick at this stage;
        // the commit is the reference if it earns its way back.)
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

        let page_locked = PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
            assert_eq!(status, 0, "Failed to allocate page-locked workspace");
            let cuda_status = cuda_pin_memory(ptr, INT_WORKSPACE_SIZE);
            assert_eq!(cuda_status, 0, "Failed to pin memory");
            PageLockedPtr(ptr as *mut u8)
        });

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };

        // ── Split-KV decode path (full-attention layers, pure-decode ticks
        // with long context): run the kernel over an expanded one-entry-per-
        // chunk batch with FAKE sinks (-1e30 ⇒ the per-chunk finalize adds
        // exactly 0 to the denominator) and non-null LSE, then merge with
        // the REAL sink injected exactly once. Only one plan is live per
        // execute, so the shared workspaces serve both paths. ──
        if self.window_left < 0 {
            if let Some(sched) = compute_split_schedule(
                &qo_indptr,
                &kv_indptr,
                device_sm_count(),
                self.num_kv_heads as i32,
            ) {
                let p_total = sched.token_map.len();
                let offs = split_offsets(p_total, batch_size, self.num_qo_heads, self.head_dim);
                let base = grow_only_buffer(&SPLIT_SCRATCH, stream, offs.total)?;
                // Pageable host memory: the async HtoD degrades to a
                // synchronous copy — fine for ~9 KiB.
                let neg_sinks = vec![-1e30f32; self.num_qo_heads];
                unsafe {
                    let htod = |off: usize, bytes: &[u8]| -> anyhow::Result<()> {
                        result::memcpy_htod_async(base + off as u64, bytes, stream.cu_stream())?;
                        Ok(())
                    };
                    htod(offs.token_map, bytemuck_i32(&sched.token_map))?;
                    htod(offs.merge_indptr, bytemuck_i32(&sched.merge_indptr))?;
                    htod(offs.neg_sinks, bytemuck_f32(&neg_sinks))?;
                }

                let mut qo = sched.qo_indptr;
                let mut kvp = sched.kv_indptr;
                let mut kvl = sched.kv_len_arr;
                let mut plan_info = [0i64; 16];
                let mut plan_info_len: i32 = 0;
                let plan_ret = unsafe {
                    (lib.prefill_plan)(
                        float_ws_ptr as *mut std::ffi::c_void,
                        super::FLOAT_WORKSPACE_SIZE,
                        int_ws_ptr as *mut std::ffi::c_void,
                        page_locked.0 as *mut std::ffi::c_void,
                        INT_WORKSPACE_SIZE,
                        qo.as_mut_ptr(),
                        kvp.as_mut_ptr(),
                        kvl.as_mut_ptr(),
                        p_total as i32,
                        p_total as i32,
                        self.num_qo_heads as i32,
                        self.num_kv_heads as i32,
                        /*page_size=*/ 1,
                        /*causal=*/ 1,
                        cu_stream,
                        plan_info.as_mut_ptr(),
                        &mut plan_info_len,
                    )
                };
                anyhow::ensure!(
                    plan_ret == 0,
                    "SinkAttention: split plan failed ({plan_ret})"
                );

                let exp_ret = unsafe {
                    (lib.expand_q_bf16)(
                        q.ptr() as *const std::ffi::c_void,
                        (base + offs.q_exp as u64) as *mut std::ffi::c_void,
                        (base + offs.token_map as u64) as *const i32,
                        p_total as i32,
                        nnz_qo as i32,
                        self.num_qo_heads as i32,
                        self.head_dim as i32,
                        cu_stream,
                    )
                };
                anyhow::ensure!(exp_ret == 0, "SinkAttention split: expand_q failed");

                let run_ret = unsafe {
                    (lib.prefill_run)(
                        int_ws_ptr as *mut std::ffi::c_void,
                        plan_info.as_mut_ptr(),
                        plan_info_len,
                        (base + offs.q_exp as u64) as *mut std::ffi::c_void,
                        k_pool.ptr() as *mut std::ffi::c_void,
                        v_pool.ptr() as *mut std::ffi::c_void,
                        kv_indices.ptr() as *mut i32,
                        (base + offs.neg_sinks as u64) as *mut f32,
                        (base + offs.partial_out as u64) as *mut std::ffi::c_void,
                        (base + offs.partial_lse as u64) as *mut f32,
                        p_total as i32,
                        self.num_qo_heads as i32,
                        self.num_kv_heads as i32,
                        /*page_size=*/ 1,
                        jit::FlashInferDType::Bf16 as i32,
                        sm_scale,
                        /*window_left=*/ -1,
                        /*causal=*/ 1,
                        cu_stream,
                    )
                };
                anyhow::ensure!(
                    run_ret == 0,
                    "SinkAttention split: fa3 run failed ({run_ret})"
                );

                let merge_ret = unsafe {
                    (lib.merge_sink_f32)(
                        (base + offs.partial_out as u64) as *const std::ffi::c_void,
                        (base + offs.partial_lse as u64) as *const f32,
                        (base + offs.merge_indptr as u64) as *const i32,
                        sinks.ptr() as *const f32,
                        out.ptr() as *mut std::ffi::c_void,
                        batch_size as i32,
                        self.num_qo_heads as i32,
                        self.head_dim as i32,
                        cu_stream,
                    )
                };
                anyhow::ensure!(merge_ret == 0, "SinkAttention split: merge failed");
                return Ok(());
            }
        }

        // ── Single-pass path ──
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

        // Kernel-native (s, heads, dim) bf16 scratch (front half: transposed
        // q in, back half: kernel out), from the grow-only pool. Reuse is
        // stream-ordered; the refresh-path sync above covers growth.
        let temp_bytes = (nnz_qo * self.num_qo_heads * self.head_dim * 2).max(1);
        let mut scratch_guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let stream_key = stream.cu_stream() as usize;
        let needs_new = !matches!(&*scratch_guard,
            Some((key, buf)) if *key == stream_key && buf.len() >= 2 * temp_bytes);
        if needs_new {
            stream.synchronize()?; // in-flight users of the old scratch
            let buf = unsafe { stream.alloc::<u8>((2 * temp_bytes).next_power_of_two())? };
            if let Some((_, old)) = scratch_guard.take() {
                std::mem::forget(old); // context may be gone; never free
            }
            *scratch_guard = Some((stream_key, buf));
        }
        let base_ptr = scratch_guard.as_ref().unwrap().1.device_ptr(stream).0;
        let (q_temp_ptr, temp_ptr) = (base_ptr, base_ptr + temp_bytes as u64);

        // The graph's q is (heads, s, dim) — the same heads-major layout
        // world as the output point — but the kernel reads token-major
        // (s, heads, dim) q. The layouts are byte-identical at s == 1
        // (decode), which is how this survived every single-token path;
        // prefill (s > 1) needs the transpose.
        let qtr_ret = unsafe {
            (lib.transpose_q_bf16)(
                q.ptr() as *const std::ffi::c_void,
                q_temp_ptr as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.head_dim as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(
            qtr_ret == 0,
            "SinkAttention: q transpose failed ({qtr_ret})"
        );

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
                q_temp_ptr as *mut std::ffi::c_void,
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

        // No trailing sync: scratch is pooled (never freed), so the enqueued
        // kernels own it under stream ordering; the next tick's plan-refresh
        // path syncs before touching shared host buffers or growing scratch.
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

    /// Shares the pinned plan scratch with the fa3 kernel tests — one lock
    /// across BOTH test modules (a module-local lock still races the other
    /// module's plans on the shared staging buffer).
    use super::super::TEST_LOCK;

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

        // The op contract takes GRAPH-layout q — (heads, s, dim), the layout
        // the compiled graph produces — while the oracle and kernel work in
        // token-major. Feeding heads-major here is what pins the op's
        // internal q transpose (identical layouts at s == 1 would hide it).
        let mut q_hbd = vec![0.0f32; q.len()];
        for b in 0..total_qo {
            for h in 0..nq {
                let src = &q[(b * nq + h) * HEAD_DIM..][..HEAD_DIM];
                q_hbd[(h * total_qo + b) * HEAD_DIM..][..HEAD_DIM].copy_from_slice(src);
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
        let d_q = up_bytes(&to_bf16_bytes(&q_hbd));
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

    /// Host-only: chunk scheduling invariants. No GPU needed, but takes
    /// TEST_LOCK because the split env knobs are process-global and the GPU
    /// tests read them.
    #[test]
    fn split_schedule_host() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Ragged multi-seq split: chunks contiguous, lengths sum, no empties.
        let qo = [0, 1, 2];
        let kv = [0, 600, 1300];
        let s = compute_split_schedule(&qo, &kv, 132, 8).expect("should split");
        let p = s.token_map.len();
        assert_eq!(s.qo_indptr, (0..=p as i32).collect::<Vec<_>>());
        assert_eq!(s.kv_indptr.len(), p + 1);
        assert_eq!(*s.kv_indptr.last().unwrap(), 1300);
        assert!(s.kv_len_arr.iter().all(|&l| l > 0), "empty chunk");
        for k in 0..p {
            assert_eq!(
                s.kv_indptr[k + 1] - s.kv_indptr[k],
                s.kv_len_arr[k],
                "chunk {k} not contiguous"
            );
        }
        // Per-seq chunk lengths reassemble each sequence exactly.
        assert_eq!(s.merge_indptr, {
            let mut m = vec![0i32];
            for b in 0..2 {
                let n = s.token_map.iter().filter(|&&t| t == b).count() as i32;
                m.push(m.last().unwrap() + n);
            }
            m
        });
        for b in 0..2 {
            let sum: i32 = (s.merge_indptr[b] as usize..s.merge_indptr[b + 1] as usize)
                .map(|k| s.kv_len_arr[k])
                .sum();
            assert_eq!(sum, kv[b + 1] - kv[b], "seq {b} chunk lengths");
        }
        assert!(s.token_map.windows(2).all(|w| w[0] <= w[1]));

        // Below threshold: no split.
        assert!(compute_split_schedule(&[0, 1], &[0, 400], 132, 8).is_none());
        // Not pure decode: no split.
        assert!(compute_split_schedule(&[0, 2, 3], &[0, 600, 1300], 132, 8).is_none());
        // One chunk per seq (P == B): no split (tiny GPU → huge chunks).
        assert!(compute_split_schedule(&[0, 1], &[0, 600], 1, 8).is_none());
        // Splits-per-seq cap at 32 with ragged tail.
        let s = compute_split_schedule(&[0, 1], &[0, 100_001], 132, 8).expect("should split");
        assert_eq!(s.token_map.len(), 32);
        assert!(s.kv_len_arr.iter().all(|&l| l > 0));
        assert_eq!(s.kv_len_arr.iter().sum::<i32>(), 100_001);
    }

    /// Split-KV decode at the op boundary: pure-decode tick, long ragged
    /// contexts forcing multiple chunks per sequence, REAL sinks (0.8 base —
    /// a per-chunk sink would inflate the denominator ~Nx and blow the
    /// tolerance), vs the unchanged CPU oracle. Then the same case with the
    /// kill switch proves the single-pass path agrees.
    #[test]
    fn sink_attention_op_split_decode() {
        if !hopper_gpu() {
            return;
        }
        let seqs = [
            RefSeq {
                qo_len: 1,
                kv_len: 300,
            },
            RefSeq {
                qo_len: 1,
                kv_len: 700,
            },
            RefSeq {
                qo_len: 1,
                kv_len: 1500,
            },
        ];
        // env knobs are process-global; all readers hold TEST_LOCK (which
        // run_op_case takes), so scope the overrides to this test.
        unsafe { std::env::set_var("LUMINAL_SPLIT_KV_MIN", "256") };
        run_op_case("op split decode", &seqs, -1, 0.8, 404);
        run_op_case(
            "op split decode (B=1, kv=2048)",
            &[RefSeq {
                qo_len: 1,
                kv_len: 2048,
            }],
            -1,
            0.8,
            405,
        );
        unsafe { std::env::set_var("LUMINAL_DISABLE_SPLIT_KV", "1") };
        run_op_case("op split decode (single-pass control)", &seqs, -1, 0.8, 404);
        unsafe {
            std::env::remove_var("LUMINAL_DISABLE_SPLIT_KV");
            std::env::remove_var("LUMINAL_SPLIT_KV_MIN");
        }
    }
}
