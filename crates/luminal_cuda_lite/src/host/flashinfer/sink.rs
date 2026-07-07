//! FlashInfer attention with gpt-oss attention sinks (multi-sequence,
//! explicit-indptr, prefill-kernel-only).
//!
//! Differences from [`super::FlashInferAttention`]:
//! - Per-head learned **sink** logit folded into the softmax denominator
//!   (kernel-side `AttentionSink` variant in wrapper.cu).
//! - Always takes **explicit qo/kv indptrs** (multi-sequence continuous
//!   batching); never derives single-sequence indptrs.
//! - Always plans/runs through the **BatchPrefill** path — the decode kernels
//!   have no sink hook, and prefill with qo_len=1 rows is upstream's own
//!   sink-decode strategy.
//! - Never absorbed into CUDA-graph decode capture (to_host.rs downcasts to
//!   `FlashInferAttention` only), so it runs as a live host op each step.
//!
//! Runtime inputs (extract() order):
//!   0: Q          (s, H*D)      16-bit   memory layout (s, heads, dim)
//!   1: K_pool     (slots, KV)   16-bit
//!   2: V_pool     (slots, KV)   16-bit
//!   3: gather_idx (c,)          Int      compact slot indices
//!   4: qo_indptr  (r,)          Int      r = n_seqs + 1
//!   5: kv_indptr  (r,)          Int
//!   6: sinks      (heads,)      16-bit   per-qo-head sink logits
//! The additive mask in the egg rule is a proof anchor only and is dropped.
//!
//! Per-step sharing: all layers of a model step see the same indptr buffers,
//! so the device→host indptr readback (which syncs the stream) and the
//! host-side plan run once per (step, window flavor) and are cached. The
//! execution epoch that scopes the cache is bumped by `CudaRuntime::execute`.
//! Each cached plan owns a PRIVATE int workspace: plans write scheduling
//! metadata at plan-determined offsets, and interleaving the full/window
//! flavors through the shared workspace would clobber it.

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

use luminal::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span},
        *,
    },
};

use crate::{
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr, result},
    host::{DeviceBuffer, HostOp},
};

use super::{
    FLOAT_WORKSPACE_SIZE, INT_WORKSPACE_SIZE, bytes_to_i32_vec, find_indptrs,
    flashinfer_workspaces, jit, jit::FlashInferDType, page_locked_workspace,
};

/// Bumped once per `CudaRuntime::execute` so per-step caches can tell steps
/// apart without any notion of time.
pub(crate) static EXEC_EPOCH: AtomicU64 = AtomicU64::new(0);

pub(crate) fn bump_exec_epoch() {
    EXEC_EPOCH.fetch_add(1, Ordering::Relaxed);
}

#[derive(Debug)]
pub struct FlashInferSinkAttention {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub batch_dim: Expression,
    pub dtype: DType,
    /// Softmax scale; 0.0 = default `1/sqrt(head_dim)`.
    pub sm_scale: f64,
    /// FlashInfer `window_left`; -1 = no window. gpt-oss sliding layers use
    /// W-1 = 127.
    pub window_left: i64,
}

impl Default for FlashInferSinkAttention {
    fn default() -> Self {
        Self {
            num_qo_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            page_size: 0,
            batch_dim: Expression::default(),
            dtype: DType::Bf16,
            sm_scale: 0.0,
            window_left: -1,
        }
    }
}

// ── per-step caches ──────────────────────────────────────────────────────

/// One prepared (planned) sink-attention configuration. Owns its int
/// workspace and device-side indptr/indices buffers.
struct PreparedSink {
    lib: &'static jit::FlashInferLib,
    plan_info: Vec<i64>,
    /// Private scheduling-metadata workspace (see module docs).
    _int_workspace: CudaSlice<u8>,
    int_workspace_ptr: u64,
    float_workspace_ptr: u64,
    _dev_qo_indptr: CudaSlice<i32>,
    dev_qo_indptr_ptr: u64,
    _dev_kv_indptr: CudaSlice<i32>,
    dev_kv_indptr_ptr: u64,
    _indices: CudaSlice<i32>,
    indices_ptr: u64,
    _last_page_len: CudaSlice<i32>,
    last_page_len_ptr: u64,
    _temp_output: CudaSlice<u8>,
    temp_output_ptr: u64,
    total_q_tokens: usize,
    batch_size: usize,
    c: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    head_dim: usize,
    kv_dim: usize,
    dtype: FlashInferDType,
    sm_scale_bits: u32,
    window_left: i32,
}

// SAFETY: owns CUDA allocations + a process-lifetime library handle; all
// launches are serialized on the caller's stream.
unsafe impl Send for PreparedSink {}
unsafe impl Sync for PreparedSink {}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SinkPlanKey {
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    total_q_tokens: usize,
    c: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    dtype: FlashInferDType,
    sm_scale_bits: u32,
    window_left: i32,
}

/// Plan cache. Within a step: 2 misses (full + sliding flavors), 34 hits.
/// Bounded: cleared whenever it grows past a small cap (indptr contents churn
/// every step during serving, so cross-step reuse is limited to steady decode
/// where contexts grow — those keys differ too; the win is within-step).
static SINK_PLAN_CACHE: OnceLock<Mutex<HashMap<SinkPlanKey, Arc<PreparedSink>>>> = OnceLock::new();
const SINK_PLAN_CACHE_CAP: usize = 16;

/// Indptr readback cache: all 36 layers of a step share the same indptr
/// buffers, so only the first does the dtoh + stream sync.
static INDPTR_READBACK: OnceLock<Mutex<Option<IndptrReadback>>> = OnceLock::new();

struct IndptrReadback {
    epoch: u64,
    qo_ptr: u64,
    kv_ptr: u64,
    r: usize,
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
}

fn read_indptrs(
    stream: &Arc<CudaStream>,
    qo_ptr: u64,
    kv_ptr: u64,
    r: usize,
) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
    let epoch = EXEC_EPOCH.load(Ordering::Relaxed);
    let cache = INDPTR_READBACK.get_or_init(Default::default);
    {
        let guard = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("indptr readback lock poisoned"))?;
        if let Some(rb) = guard.as_ref()
            && rb.epoch == epoch
            && rb.qo_ptr == qo_ptr
            && rb.kv_ptr == kv_ptr
            && rb.r == r
        {
            return Ok((rb.qo_indptr.clone(), rb.kv_indptr.clone()));
        }
    }
    let read = |ptr: u64| -> anyhow::Result<Vec<i32>> {
        let mut host_bytes = vec![0u8; r * std::mem::size_of::<i32>()];
        unsafe {
            result::memcpy_dtoh_async(&mut host_bytes, ptr, stream.cu_stream())?;
        }
        Ok(bytes_to_i32_vec(host_bytes))
    };
    let qo_bytes = read(qo_ptr)?;
    let kv_bytes = read(kv_ptr)?;
    stream.synchronize()?;
    let (qo_indptr, kv_indptr) = (qo_bytes, kv_bytes);
    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("indptr readback lock poisoned"))?;
    *guard = Some(IndptrReadback {
        epoch,
        qo_ptr,
        kv_ptr,
        r,
        qo_indptr: qo_indptr.clone(),
        kv_indptr: kv_indptr.clone(),
    });
    Ok((qo_indptr, kv_indptr))
}

impl FlashInferSinkAttention {
    pub(crate) fn graph_inputs(&self) -> usize {
        7
    }

    fn prepare(
        &self,
        stream: &Arc<CudaStream>,
        key: SinkPlanKey,
    ) -> anyhow::Result<Arc<PreparedSink>> {
        let cache = SINK_PLAN_CACHE.get_or_init(Default::default);
        {
            let guard = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("sink plan cache lock poisoned"))?;
            if let Some(p) = guard.get(&key) {
                return Ok(p.clone());
            }
        }

        let lib = jit::ensure_compiled(key.head_dim, key.window_left >= 0);
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let batch_size = key.qo_indptr.len() - 1;
        let kv_dim = key.num_kv_heads * key.head_dim;

        // Private int workspace per plan (see module docs); float workspace
        // (split-KV scratch, consumed within each serialized run) is shared.
        let int_workspace = unsafe { stream.alloc::<u8>(INT_WORKSPACE_SIZE)? };
        let int_workspace_ptr = int_workspace.device_ptr(stream).0;
        let (_float_ws, float_workspace_ptr, _shared_int_ws, _shared_int_ptr) =
            flashinfer_workspaces(stream);
        let page_locked = page_locked_workspace();

        let mut qo_indptr_host = key.qo_indptr.clone();
        let mut kv_indptr_host = key.kv_indptr.clone();

        let mut plan_info_buf = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = unsafe {
            (lib.prefill_plan)(
                float_workspace_ptr as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_workspace_ptr as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                page_locked as *mut std::ffi::c_void,
                qo_indptr_host.as_mut_ptr(),
                kv_indptr_host.as_mut_ptr(),
                key.total_q_tokens as i32,
                batch_size as i32,
                key.num_qo_heads as i32,
                key.num_kv_heads as i32,
                key.page_size as i32,
                key.head_dim as i32,
                key.dtype as i32,
                key.window_left,
                cu_stream,
                plan_info_buf.as_mut_ptr(),
                &mut plan_info_len,
            )
        };
        if plan_ret != 0 {
            anyhow::bail!("FlashInfer sink prefill plan failed with error code {plan_ret}");
        }

        let dev_qo_indptr = stream.clone_htod(&qo_indptr_host)?;
        let dev_qo_indptr_ptr = dev_qo_indptr.device_ptr(stream).0;
        let dev_kv_indptr = stream.clone_htod(&kv_indptr_host)?;
        let dev_kv_indptr_ptr = dev_kv_indptr.device_ptr(stream).0;
        let indices = unsafe { stream.alloc::<i32>(key.c.max(1))? };
        let indices_ptr = indices.device_ptr(stream).0;
        let last_page_len = stream.clone_htod(&vec![1i32; batch_size.max(1)])?;
        let last_page_len_ptr = last_page_len.device_ptr(stream).0;
        let temp_output_bytes =
            (key.total_q_tokens * key.num_qo_heads * key.head_dim * key.dtype.size_of()).max(1);
        let temp_output = unsafe { stream.alloc::<u8>(temp_output_bytes)? };
        let temp_output_ptr = temp_output.device_ptr(stream).0;

        let prepared = Arc::new(PreparedSink {
            lib,
            plan_info: plan_info_buf[..plan_info_len as usize].to_vec(),
            _int_workspace: int_workspace,
            int_workspace_ptr,
            float_workspace_ptr,
            _dev_qo_indptr: dev_qo_indptr,
            dev_qo_indptr_ptr,
            _dev_kv_indptr: dev_kv_indptr,
            dev_kv_indptr_ptr,
            _indices: indices,
            indices_ptr,
            _last_page_len: last_page_len,
            last_page_len_ptr,
            _temp_output: temp_output,
            temp_output_ptr,
            total_q_tokens: key.total_q_tokens,
            batch_size,
            c: key.c,
            num_qo_heads: key.num_qo_heads,
            num_kv_heads: key.num_kv_heads,
            page_size: key.page_size,
            head_dim: key.head_dim,
            kv_dim,
            dtype: key.dtype,
            sm_scale_bits: key.sm_scale_bits,
            window_left: key.window_left,
        });

        let mut guard = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("sink plan cache lock poisoned"))?;
        if guard.len() >= SINK_PLAN_CACHE_CAP {
            guard.clear();
        }
        guard.insert(key, prepared.clone());
        Ok(prepared)
    }
}

impl PreparedSink {
    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        gather_idx: u64,
        sinks: u64,
        output: u64,
    ) -> anyhow::Result<()> {
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        // Fill kv_indices (page table, page_size=1) from the compact gather.
        if self.c > 0 {
            unsafe {
                (self.lib.extract_slot_indices)(
                    gather_idx as *const i32,
                    self.indices_ptr as *mut i32,
                    self.c as i32,
                    self.kv_dim as i32,
                    cu_stream,
                );
            }
        }

        // (batch, heads, dim) == (heads, batch, dim) byte-identically at s=1.
        let direct_output = self.total_q_tokens == 1;
        let run_output_ptr = if direct_output {
            output
        } else {
            self.temp_output_ptr
        };

        let mut plan_info = self.plan_info.clone();
        let run_ret = unsafe {
            (self.lib.prefill_sink_run)(
                self.float_workspace_ptr as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                self.int_workspace_ptr as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info.len() as i32,
                q as *mut std::ffi::c_void,
                k_cache as *mut std::ffi::c_void,
                v_cache as *mut std::ffi::c_void,
                self.dev_qo_indptr_ptr as *mut i32,
                self.dev_kv_indptr_ptr as *mut i32,
                self.indices_ptr as *mut i32,
                self.last_page_len_ptr as *mut i32,
                sinks as *mut std::ffi::c_void,
                run_output_ptr as *mut std::ffi::c_void,
                self.total_q_tokens as i32,
                self.batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                self.page_size as i32,
                self.head_dim as i32,
                self.dtype as i32,
                f32::from_bits(self.sm_scale_bits),
                self.window_left,
                cu_stream,
            )
        };
        if run_ret != 0 {
            anyhow::bail!("FlashInfer sink run failed with error code {run_ret}");
        }

        if !direct_output {
            unsafe {
                (self.lib.transpose_output)(
                    self.temp_output_ptr as *const std::ffi::c_void,
                    output as *mut std::ffi::c_void,
                    self.total_q_tokens as i32,
                    self.num_qo_heads as i32,
                    self.head_dim as i32,
                    self.dtype as i32,
                    cu_stream,
                );
            }
        }

        Ok(())
    }
}

impl EgglogOp for FlashInferSinkAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FlashInferSinkAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("page_size", EXPRESSION),
                ("batch_dim", EXPRESSION),
                ("dtype", DTYPE),
                ("sm_scale", F64),
                ("window_left", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        // Q, K_pool, V_pool, flat gather_idx, qo_indptr, kv_indptr, sinks,
        // plus the proof-only mask; extract() drops the mask and compacts the
        // gather index.
        8
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Rules live in flashinfer_attention.egg (shipped by
        // FlashInferAttention::rewrites) because they share its staged
        // relations; this op only declares its sort.
        vec![]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let mut exec0 = |i: usize| {
            extract_expr(egraph, kind_children[i], expr_cache)
                .unwrap()
                .exec(&FxHashMap::default())
                .unwrap()
        };
        let num_qo_heads = exec0(0);
        let num_kv_heads = exec0(1);
        let head_dim = exec0(2);
        let page_size = exec0(3);
        let batch_dim = extract_expr(egraph, kind_children[4], expr_cache).unwrap();
        let dtype = extract_dtype(egraph, kind_children[5]);
        let sm_scale: f64 = egraph.enodes[kind_children[6]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let window_left = egraph.enodes[kind_children[7]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            .round() as i64;
        let fi_dtype = FlashInferDType::from_dtype(dtype);
        assert!(
            fi_dtype.is_some_and(|d| d.supports_prefill()),
            "FlashInferSinkAttention requires a 16-bit dtype, got {dtype:?}"
        );

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            batch_dim,
            dtype,
            sm_scale,
            window_left,
        };

        // Pay the nvcc cost at extract time, not in the GA profiling loop.
        let _ = jit::ensure_compiled(head_dim, window_left >= 0);

        let flat_idx_node = input_enodes[3];
        let gather_idx = find_indptrs::try_find_compact_gather_idx(egraph, flat_idx_node)
            .expect("FlashInferSinkAttention matched a gather without recoverable compact gather_idx");
        let final_inputs = vec![
            input_enodes[0], // Q
            input_enodes[1], // K_pool
            input_enodes[2], // V_pool
            gather_idx,      // compact gather_idx
            input_enodes[4], // qo_indptr
            input_enodes[5], // kv_indptr
            input_enodes[6], // sinks
        ];

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for FlashInferSinkAttention {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() != 7 {
            anyhow::bail!(
                "FlashInferSinkAttention expects 7 inputs, got {}",
                inputs.len()
            );
        }
        let total_q_tokens = self
            .batch_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FlashInferSinkAttention batch_dim is unresolved"))?;
        let c = *dyn_map
            .get(&'c')
            .ok_or_else(|| anyhow::anyhow!("FlashInferSinkAttention requires dynamic dim 'c'"))?;
        let r = *dyn_map
            .get(&'r')
            .ok_or_else(|| anyhow::anyhow!("FlashInferSinkAttention requires dynamic dim 'r'"))?;
        if r < 2 {
            anyhow::bail!("FlashInferSinkAttention requires r >= 2 (n_seqs + 1), got {r}");
        }

        let get_buf = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("FlashInferSinkAttention missing {name} buffer for {node:?}")
            })
        };
        let q_buf = get_buf("Q", inputs[0])?;
        let k_buf = get_buf("K_cache", inputs[1])?;
        let v_buf = get_buf("V_cache", inputs[2])?;
        let gather_idx_buf = get_buf("gather_idx", inputs[3])?;
        let qo_indptr_buf = get_buf("qo_indptr", inputs[4])?;
        let kv_indptr_buf = get_buf("kv_indptr", inputs[5])?;
        let sinks_buf = get_buf("sinks", inputs[6])?;
        let out_buf = get_buf("output", self_node)?;

        let dtype = FlashInferDType::from_dtype(self.dtype)
            .filter(|d| d.supports_prefill())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "FlashInferSinkAttention requires a 16-bit dtype, got {:?}",
                    self.dtype
                )
            })?;

        // Once per step (readback cache): pull the batch layout to the host.
        let (qo_indptr, kv_indptr) =
            read_indptrs(stream, qo_indptr_buf.ptr(), kv_indptr_buf.ptr(), r)?;
        let n_seqs = r - 1;
        let q_total = *qo_indptr.last().unwrap_or(&0);
        let kv_total = *kv_indptr.last().unwrap_or(&0);
        if q_total as usize != total_q_tokens || kv_total as usize != c {
            anyhow::bail!(
                "FlashInferSinkAttention indptr mismatch: qo ends at {q_total} (s={total_q_tokens}), kv ends at {kv_total} (c={c})"
            );
        }
        let _ = n_seqs;

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let key = SinkPlanKey {
            qo_indptr,
            kv_indptr,
            total_q_tokens,
            c,
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            page_size: self.page_size,
            dtype,
            sm_scale_bits: sm_scale.to_bits(),
            window_left: self.window_left as i32,
        };
        let prepared = self.prepare(stream, key)?;

        // One-shot confirmation that the FlashInfer sink path was selected
        // (vs. the HLIR masked-dense fallback), for smoke/bench runs.
        static SELECTED_ONCE: std::sync::Once = std::sync::Once::new();
        SELECTED_ONCE.call_once(|| {
            eprintln!(
                "[flashinfer-sink] selected: s={total_q_tokens} c={c} window_left={}",
                self.window_left
            );
        });

        let _span = span!(
            Level::TRACE,
            "FlashInferSinkAttention",
            total_q_tokens,
            c,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        .entered();
        prepared.enqueue(
            stream,
            q_buf.ptr(),
            k_buf.ptr(),
            v_buf.ptr(),
            gather_idx_buf.ptr(),
            sinks_buf.ptr(),
            out_buf.ptr(),
        )
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("FlashInferSinkAttention")
    }
}
