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

#[cfg(test)]
mod tests {
    use super::super::test_ref::{Lcg, to_bf16_bytes};
    use super::*;
    use crate::cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

    /// Kernel iteration harness: times the fused decode kernel at the real
    /// gpt-oss decode workload (E=128, H=I=2880, top-4, s=1) with synthetic
    /// weights. Comparison targets (same box, vLLM Marlin probe):
    /// 52 us/layer graph-captured, 89 us eager. Run explicitly:
    ///   cargo test -p luminal_cuda_lite decode_bench -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark, run explicitly"]
    fn decode_bench_real_dims() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let (e_cnt, hidden, inter, top_k, seq) = (128usize, 2880usize, 2880usize, 4usize, 1usize);
        let gate_up_n = 2 * inter;
        let mut rng = Lcg(9);

        let fill =
            |n: usize, rng: &mut Lcg| -> Vec<u8> { (0..n).map(|_| rng.below(256) as u8).collect() };
        let gu_q = fill(e_cnt * gate_up_n * hidden / 2, &mut rng);
        let gu_s: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 32)
            .map(|_| 124 + rng.below(6) as u8)
            .collect();
        let dn_q = fill(e_cnt * hidden * inter / 2, &mut rng);
        let dn_s: Vec<u8> = (0..e_cnt * hidden * inter / 32)
            .map(|_| 124 + rng.below(6) as u8)
            .collect();
        let gu_b: Vec<f32> = (0..e_cnt * gate_up_n).map(|_| 0.01).collect();
        let dn_b: Vec<f32> = (0..e_cnt * hidden).map(|_| 0.01).collect();
        let x: Vec<f32> = (0..seq * hidden)
            .map(|_| (rng.below(200) as f32 - 100.0) / 100.0)
            .collect();
        let ids: Vec<i32> = vec![3, 71, 15, 120];
        let w: Vec<f32> = vec![0.4, 0.3, 0.2, 0.1];

        let up = |b: &[u8]| stream.clone_htod(b).unwrap();
        let ptr = |b: &CudaSlice<u8>| b.device_ptr(&stream).0;
        let d_gu_q = up(&gu_q);
        let d_gu_s = up(&gu_s);
        let d_gu_b = up(&to_bf16_bytes(&gu_b));
        let d_dn_q = up(&dn_q);
        let d_dn_s = up(&dn_s);
        let d_dn_b = up(&to_bf16_bytes(&dn_b));
        let _d_x = up(bytemuck::cast_slice::<f32, u8>(&x));
        let _d_ids = up(bytemuck::cast_slice::<i32, u8>(&ids));
        let _d_w = up(bytemuck::cast_slice::<f32, u8>(&w));
        let _d_hid = stream.alloc_zeros::<u8>(seq * top_k * inter * 4).unwrap();
        let _d_out = stream.alloc_zeros::<u8>(seq * hidden * 4).unwrap();

        // {warp, block} x seq sweep. Buffers sized for the largest seq; each
        // cell re-launches with its own seq (ids/weights repeat per token).
        let max_seq = 16usize;
        let ids_all: Vec<i32> = (0..max_seq).flat_map(|_| ids.iter().copied()).collect();
        let w_all: Vec<f32> = (0..max_seq).flat_map(|_| w.iter().copied()).collect();
        let x_all: Vec<f32> = (0..max_seq * hidden)
            .map(|_| (rng.below(200) as f32 - 100.0) / 100.0)
            .collect();
        let d_ids_all = up(bytemuck::cast_slice::<i32, u8>(&ids_all));
        let d_w_all = up(bytemuck::cast_slice::<f32, u8>(&w_all));
        let d_x_all = up(bytemuck::cast_slice::<f32, u8>(&x_all));
        let d_hid_all = stream
            .alloc_zeros::<u8>(max_seq * top_k * inter * 4)
            .unwrap();
        let d_out_all = stream.alloc_zeros::<u8>(max_seq * hidden * 4).unwrap();

        eprintln!("seq  pairs   us/launch   GB/s(weights)   x36 (ms)");
        {
            let name = "";
            for s_i in [1usize, 2, 4, 8, 16] {
                let launch = || {
                    fused_moe_decode(
                        &stream,
                        ptr(&d_x_all),
                        ptr(&d_gu_q),
                        ptr(&d_gu_s),
                        ptr(&d_gu_b),
                        ptr(&d_dn_q),
                        ptr(&d_dn_s),
                        ptr(&d_dn_b),
                        ptr(&d_ids_all),
                        ptr(&d_w_all),
                        ptr(&d_hid_all),
                        ptr(&d_out_all),
                        hidden,
                        inter,
                        top_k,
                        s_i,
                        top_k,
                        1.702,
                        7.0,
                    )
                    .unwrap()
                };
                for _ in 0..20 {
                    launch();
                }
                stream.synchronize().unwrap();
                let iters = 200;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    launch();
                }
                stream.synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                // distinct expert rows touched (worst case: all pairs distinct)
                let bytes = (s_i * top_k).min(e_cnt)
                    * (gate_up_n * hidden / 2
                        + gate_up_n * hidden / 32
                        + hidden * inter / 2
                        + hidden * inter / 32);
                let gbps = bytes as f64 / (us * 1e-6) / 1e9;
                eprintln!(
                    "{name}     {s_i:>3}  {:>5}   {us:>8.1}   {gbps:>10.0}      {:>6.2}",
                    s_i * top_k,
                    us * 36.0 / 1e3
                );
            }
        }
        eprintln!("Marlin same-box reference @ s=1: 52 us captured / 89 us eager");
    }
}
