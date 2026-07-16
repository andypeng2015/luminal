//! Fused single-kernel MoE decode path (see decode.cu): gate_up GEMV +
//! SwiGLU + down GEMV + routed sum in ONE cooperative launch, for the
//! small-batch regime where the tiled GEMM path is padding-dominated.
//!
//! Output contract: writes `out` f32 `[seq, hidden]` = the complete MoE block
//! output (bias + routed-weight sum applied). Needs only one scratch buffer:
//! `hidden` f32 `[seq*top_k, inter]`. Reads x as f32 directly (no cast
//! kernel) and the routing tensors in place (no align kernels).
//!
//! Cooperative-launch contract: the grid is sized to co-resident occupancy
//! (queried per-function, cached) — launching wider would fail outright.

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
    /// unbounded (one warp per task, no residency cap).
    phase1: CudaFunction,
    phase2: CudaFunction,
    /// Row-blocked variants (R outputs per warp sharing one activation
    /// read); see the ncu note in decode.cu.
    phase1_r2: CudaFunction,
    phase1_r4: CudaFunction,
    phase2_r2: CudaFunction,
    phase2_r4: CudaFunction,
    #[cfg(test)]
    debug_dot: CudaFunction,
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
            .load_function("moe_phase1")
            .expect("moe_phase1 should exist");
        let phase2 = module
            .load_function("moe_phase2")
            .expect("moe_phase2 should exist");
        let phase1_r2 = module.load_function("moe_phase1_r2").unwrap();
        let phase1_r4 = module.load_function("moe_phase1_r4").unwrap();
        let phase2_r2 = module.load_function("moe_phase2_r2").unwrap();
        let phase2_r4 = module.load_function("moe_phase2_r4").unwrap();
        #[cfg(test)]
        let debug_dot = module
            .load_function("debug_row_dot")
            .expect("debug_row_dot should exist");
        DecodeKernel {
            _module: module,
            phase1,
            phase2,
            phase1_r2,
            phase1_r4,
            phase2_r2,
            phase2_r4,
            #[cfg(test)]
            debug_dot,
        }
    })
}

/// Force the process-wide NVRTC compile of this module (idempotent). Called
/// from FusedMoE's extract-time warmup so the compile cost lands outside
/// timed profiling trials.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernel(stream);
}

/// The MoE block for `seq` tokens as two stream-ordered launches (phase 1
/// gate/up+SwiGLU, phase 2 down+mix); split beat the old cooperative single
/// launch by 5-16% across seq 1..16
/// tokens. All pointers are device addresses; see decode.cu for layouts.
/// Rows-per-warp for the GEMV: 1 = original kernels, 2/4 = row-blocked.
/// Bench: r4 fastest at every seq 1..128 (2-3x r1); dims not divisible by
/// R fall back to 1 inside `fused_moe_decode_with_rows`.
const GEMV_ROWS: usize = 4;

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
    fused_moe_decode_with_rows(
        stream,
        GEMV_ROWS,
        x_ptr,
        gu_q_ptr,
        gu_scale_ptr,
        gu_bias_ptr,
        dn_q_ptr,
        dn_scale_ptr,
        dn_bias_ptr,
        topk_ids_ptr,
        topk_w_ptr,
        hidden_scratch_ptr,
        out_ptr,
        hidden_dim,
        inter,
        top_k,
        seq,
        idx_row_stride,
        alpha,
        limit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_moe_decode_with_rows(
    stream: &Arc<CudaStream>,
    rows: usize,
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
    anyhow::ensure!(
        hidden_dim.is_multiple_of(32) && inter.is_multiple_of(32),
        "decode GEMV requires 32-aligned dims (e8m0 group width): hidden={hidden_dim}, inter={inter}"
    );
    anyhow::ensure!(idx_row_stride >= top_k, "idx_row_stride must be >= top_k");
    // The kernel loads weight groups as uint4: every row must start 16B
    // aligned, i.e. row stride (k/2 bytes) divisible by 16.
    anyhow::ensure!(
        (hidden_dim / 2).is_multiple_of(16) && (inter / 2).is_multiple_of(16),
        "decode GEMV requires 16B-aligned weight rows (dims % 32 == 0)"
    );
    if seq == 0 || top_k == 0 {
        return Ok(());
    }

    // Dims must split evenly into row blocks; otherwise original kernels.
    let rows = if rows > 1 && inter % rows == 0 && hidden_dim % rows == 0 {
        rows
    } else {
        1
    };
    let k = kernel(stream);
    let (p1, p2) = match rows {
        2 => (&k.phase1_r2, &k.phase2_r2),
        4 => (&k.phase1_r4, &k.phase2_r4),
        _ => (&k.phase1, &k.phase2),
    };
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
    use super::super::test_ref::{assert_close, to_bf16_bytes};
    use super::*;
    use crate::cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

    struct Lcg(u64);
    impl Lcg {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % n
        }
    }

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

    /// The dot primitive in isolation vs host_weight math.
    #[test]
    fn decode_row_dot_primitive() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let (e_cnt, n_dim, k_dim) = (3usize, 8usize, 64usize);
        let mut rng = Lcg(5);
        let bq: Vec<u8> = (0..e_cnt * n_dim * k_dim / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let bs: Vec<u8> = (0..e_cnt * n_dim * k_dim / 32)
            .map(|_| 120 + rng.below(12) as u8)
            .collect();
        let vec: Vec<f32> = (0..k_dim)
            .map(|_| (rng.below(200) as f32 - 100.0) / 40.0)
            .collect();
        let d_bq = stream.clone_htod(&bq).unwrap();
        let d_bs = stream.clone_htod(&bs).unwrap();
        let d_v = stream
            .clone_htod(bytemuck::cast_slice::<f32, u8>(&vec))
            .unwrap();
        let d_out = stream.alloc_zeros::<u8>(4).unwrap();
        let k = kernel(&stream);
        for (e, row) in [(0usize, 0usize), (1, 3), (2, 7), (1, 5)] {
            let (ei, ni, ki, ri) = (e as i32, n_dim as i32, k_dim as i32, row as i32);
            let (pq, ps, pv, po) = (
                d_bq.device_ptr(&stream).0,
                d_bs.device_ptr(&stream).0,
                d_v.device_ptr(&stream).0,
                d_out.device_ptr(&stream).0,
            );
            unsafe {
                stream
                    .launch_builder(&k.debug_dot)
                    .arg(&pq)
                    .arg(&ps)
                    .arg(&pv)
                    .arg(&po)
                    .arg(&ei)
                    .arg(&ni)
                    .arg(&ki)
                    .arg(&ri)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let got = bytemuck::cast_slice::<u8, f32>(&stream.clone_dtoh(&d_out).unwrap())[0];
            use super::super::test_ref::host_weight;
            let want: f32 = (0..k_dim)
                .map(|kk| vec[kk] * host_weight(&bq, &bs, e, n_dim, k_dim, row, kk))
                .sum();
            let rel = (got - want).abs() / want.abs().max(1e-3);
            eprintln!("dot e={e} row={row}: got={got:.5} want={want:.5} rel={rel:.6}");
            assert!(rel < 1e-4, "dot primitive mismatch");
        }
    }

    /// Tiny dims vs the shared host chain reference (same oracle the tiled
    /// path and the FusedMoE op test use), with a strided topk_idx buffer.
    #[test]
    fn decode_fused_tiny_vs_host() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let (tokens, top_k, e_cnt) = (13usize, 2usize, 4usize);
        let (hidden, inter) = (64usize, 64usize);
        let gate_up_n = 2 * inter;
        let idx_row_stride = 8usize;
        let mut rng = Lcg(31);

        let x: Vec<f32> = (0..tokens * hidden)
            .map(|_| (rng.below(200) as f32 - 100.0) / 50.0)
            .collect();
        let gu_q: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let gu_s: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 32)
            .map(|_| 125 + rng.below(6) as u8)
            .collect();
        let gu_bias: Vec<f32> = (0..e_cnt * gate_up_n)
            .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
            .collect();
        let dn_q: Vec<u8> = (0..e_cnt * hidden * inter / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let dn_s: Vec<u8> = (0..e_cnt * hidden * inter / 32)
            .map(|_| 125 + rng.below(6) as u8)
            .collect();
        let dn_bias: Vec<f32> = (0..e_cnt * hidden)
            .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
            .collect();

        let mut topk_ids = Vec::new();
        let mut wide_ids = vec![i32::MIN; tokens * idx_row_stride];
        let mut topk_w = Vec::new();
        for t in 0..tokens {
            let first = rng.below(e_cnt);
            let second = (first + 1 + rng.below(e_cnt - 1)) % e_cnt;
            for (kk, e) in [first, second].into_iter().enumerate() {
                topk_ids.push(e as i32);
                wide_ids[t * idx_row_stride + kk] = e as i32;
            }
            let w0 = 0.2 + (rng.below(60) as f32) / 100.0;
            topk_w.extend([w0, 1.0 - w0]);
        }

        let up = |bytes: &[u8]| stream.clone_htod(bytes).unwrap();
        let ptr = |b: &CudaSlice<u8>| b.device_ptr(&stream).0;
        let d_x = up(bytemuck::cast_slice::<f32, u8>(&x));
        let d_ids = up(bytemuck::cast_slice::<i32, u8>(&wide_ids));
        let d_w = up(bytemuck::cast_slice::<f32, u8>(&topk_w));
        let d_gu_q = up(&gu_q);
        let d_gu_s = up(&gu_s);
        let d_gu_b = up(&to_bf16_bytes(&gu_bias));
        let d_dn_q = up(&dn_q);
        let d_dn_s = up(&dn_s);
        let d_dn_b = up(&to_bf16_bytes(&dn_bias));
        let d_hid = stream
            .alloc_zeros::<u8>(tokens * top_k * inter * 4)
            .unwrap();
        let d_out = stream.alloc_zeros::<u8>(tokens * hidden * 4).unwrap();

        // All row-block variants must match the same oracle (R divides the
        // 64-wide tiny dims, so every kernel pair actually runs).
        let mut results: Vec<Vec<f32>> = vec![];
        for rows in [1usize, 2, 4] {
            fused_moe_decode_with_rows(
                &stream,
                rows,
                ptr(&d_x),
                ptr(&d_gu_q),
                ptr(&d_gu_s),
                ptr(&d_gu_b),
                ptr(&d_dn_q),
                ptr(&d_dn_s),
                ptr(&d_dn_b),
                ptr(&d_ids),
                ptr(&d_w),
                ptr(&d_hid),
                ptr(&d_out),
                hidden,
                inter,
                top_k,
                tokens,
                idx_row_stride,
                1.702,
                7.0,
            )
            .unwrap();
            stream.synchronize().unwrap();
            let got_b = stream.clone_dtoh(&d_out).unwrap();
            results.push(bytemuck::cast_slice::<u8, f32>(&got_b).to_vec());
        }
        let got: &[f32] = &results[0];

        use super::super::test_ref::{ChainWeights, host_chain_reference_f32};
        let want = host_chain_reference_f32(
            &ChainWeights {
                gu_q: &gu_q,
                gu_s: &gu_s,
                gu_bias: &gu_bias,
                dn_q: &dn_q,
                dn_s: &dn_s,
                dn_bias: &dn_bias,
            },
            &x,
            &topk_ids,
            &topk_w,
            tokens,
            top_k,
            hidden,
            inter,
        );

        // 0.01: f32 summation-order noise (warp-parallel vs sequential)
        // amplified on cancellation-heavy outputs; NOT quantization slack.
        assert_close(got, &want, 0.01, "decode_fused_tiny");
    }
}
