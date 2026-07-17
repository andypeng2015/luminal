//! Fused MoE decode path (see decode.cu): two regular stream-ordered
//! launches — phase 1 gate_up GEMV + SwiGLU, phase 2 down GEMV + routed
//! sum — for the small-batch regime where the grouped GEMM chain is
//! padding-dominated.
//!
//! Output contract: writes `out` f32 `[seq, hidden]` = the complete MoE block
//! output (bias + routed-weight sum applied). Needs only one scratch buffer:
//! `hidden` f32 `[seq*top_k, inter]`. Reads x as f32 directly (no cast
//! kernel) and the routing tensors in place (no align kernels).

use std::sync::{Arc, OnceLock};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
};

const SOURCE: &str = include_str!("decode.cu");
const BLOCK_THREADS: u32 = 256;
// A block-cooperative split-K variant was built, benched, and removed:
// warp-per-row won at every pair count once the __constant__-LUT replay
// serialization was fixed (131.6 vs 151.2 us at s=1; 1875 vs 2161 at
// pairs=64).

struct DecodeKernel {
    _module: Arc<CudaModule>,
    /// Phase 1 (gate/up + SwiGLU) and phase 2 (down + mix) as regular
    /// launches; stream order enforces the phase boundary, so grids are
    /// unbounded (one warp per task, no residency cap). Both are the
    /// row-blocked R=4 kernels; see the ncu note in decode.cu.
    phase1: CudaFunction,
    phase2: CudaFunction,
}

// Process-wide cache (NOT per-instance: ops are cloned during GA profiling).
static KERNEL: OnceLock<DecodeKernel> = OnceLock::new();

fn kernel(stream: &Arc<CudaStream>) -> &'static DecodeKernel {
    KERNEL.get_or_init(|| {
        let image = compile_module_image_for_current_device(stream.context(), SOURCE)
            .expect("moe decode kernel should compile");
        let module = stream
            .context()
            .load_module(image)
            .expect("moe decode module should load");
        let phase1 = module
            .load_function("moe_phase1_r4")
            .expect("moe_phase1_r4 should exist");
        let phase2 = module
            .load_function("moe_phase2_r4")
            .expect("moe_phase2_r4 should exist");
        DecodeKernel {
            _module: module,
            phase1,
            phase2,
        }
    })
}

/// Force the process-wide NVRTC compile of this module (idempotent). Called
/// from FusedMoE's extract-time warmup so the compile cost lands outside
/// timed profiling trials.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernel(stream);
}

/// Rows-per-warp for the GEMV kernels. The r1 (one output per warp) and r2
/// variants were deleted after r4 won the bench at every seq 1..128 (2-3x
/// r1); resurrect from git if a future arch moves the tradeoff.
const GEMV_ROWS: usize = 4;

/// The MoE block for `seq` tokens as two stream-ordered launches (phase 1
/// gate/up+SwiGLU, phase 2 down+mix); split beat the old cooperative single
/// launch by 5-16% across seq 1..16 tokens. All pointers are device
/// addresses; see decode.cu for layouts. Dims must be multiples of 32
/// (which also makes them multiples of GEMV_ROWS).
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_decode(
    stream: &Arc<CudaStream>,
    x_ptr: u64,
    gu_q_ptr: u64,
    gu_scale_ptr: u64,
    gu_bias_ptr: u64,
    dn_q_ptr: u64,
    dn_scale_ptr: u64,
    dn_bias_ptr: u64,
    topk_ids_ptr: u64,
    topk_w_ptr: u64,
    hidden_scratch_ptr: u64,
    out_ptr: u64,
    hidden_dim: usize,
    inter: usize,
    top_k: usize,
    seq: usize,
    idx_row_stride: usize,
    alpha: f32,
    limit: f32,
) -> anyhow::Result<()> {
    // 32-aligned dims are also exactly what the kernel's uint4 weight loads
    // need: row stride k/2 bytes divisible by 16.
    anyhow::ensure!(
        hidden_dim.is_multiple_of(32) && inter.is_multiple_of(32),
        "decode GEMV requires 32-aligned dims (e8m0 group width): hidden={hidden_dim}, inter={inter}"
    );
    anyhow::ensure!(idx_row_stride >= top_k, "idx_row_stride must be >= top_k");
    if seq == 0 || top_k == 0 {
        return Ok(());
    }

    let rows = GEMV_ROWS; // dims % 32 == 0 guarantees % GEMV_ROWS == 0
    let k = kernel(stream);
    let (p1, p2) = (&k.phase1, &k.phase2);
    // Split launch (measured 5-16% faster than the cooperative single launch
    // across seq 1..16): one warp per task, unbounded grid, phase order
    // enforced by the stream.
    let warps_per_block = (BLOCK_THREADS / 32) as usize;
    let grid = |tasks: usize| LaunchConfig {
        grid_dim: (tasks.div_ceil(warps_per_block).max(1) as u32, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let (h, i, tk, s, stride) = (
        hidden_dim as i32,
        inter as i32,
        top_k as i32,
        seq as i32,
        idx_row_stride as i32,
    );
    unsafe {
        stream
            .launch_builder(p1)
            .arg(&x_ptr)
            .arg(&gu_q_ptr)
            .arg(&gu_scale_ptr)
            .arg(&gu_bias_ptr)
            .arg(&topk_ids_ptr)
            .arg(&hidden_scratch_ptr)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&s)
            .arg(&stride)
            .arg(&alpha)
            .arg(&limit)
            .launch(grid(seq * top_k * inter / rows))?;
        stream
            .launch_builder(p2)
            .arg(&dn_q_ptr)
            .arg(&dn_scale_ptr)
            .arg(&dn_bias_ptr)
            .arg(&topk_ids_ptr)
            .arg(&topk_w_ptr)
            .arg(&hidden_scratch_ptr)
            .arg(&out_ptr)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&s)
            .arg(&stride)
            .launch(grid(seq * hidden_dim / rows))?;
    }
    Ok(())
}
