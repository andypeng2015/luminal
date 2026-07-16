//! Launch wrapper + tests for `moe.cu`'s `fused_moe_mxfp4_gemm`: the grouped
//! MoE GEMM consuming the `align` building block's outputs (tokens gathered
//! to resident MXFP4 weights). Semantics follow the Triton spec-of-record at
//! `harness/fused_moe_mxfp4_triton.py`; this CUDA version stages f32 tiles in
//! shared memory and accumulates with scalar fmaf (more precise than the
//! Triton kernel, which rounds scaled weights to bf16 before `tl.dot`).
//! Tensor-core mma, GROUP_M swizzle, and occupancy tuning are the planned
//! Phase-B upgrades — this is the correctness kernel.
//!
//! Verification is oracle-based: `harness/moe_gemm_oracle.py` runs the Triton
//! reference on pinned inputs (including host-computed align metadata, so
//! placement order is deterministic) and dumps fixtures that the tests here
//! replay through the CUDA kernel.

use std::sync::{Arc, OnceLock};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
};

const MOE_GEMM_SRC: &str = include_str!("moe.cu");

/// Tile sizes fixed inside moe.cu. `BM` must equal the `block_size` used when
/// running `align::moe_align_block_size` (the expert_ids-per-tile coupling).
pub const BM: usize = 64;
pub const BN: usize = 64;

struct MoeGemmKernels {
    _module: Arc<CudaModule>,
    gemm: CudaFunction,
}

/// Process-wide kernel cache.
static KERNELS: OnceLock<MoeGemmKernels> = OnceLock::new();

/// Force the process-wide NVRTC compile of this module (idempotent). Called
/// from host-op prewarm so the compile cost lands in the untimed prebuild
/// phase, never inside a profiling trial or first real execution.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernels(stream);
}

fn kernels(stream: &Arc<CudaStream>) -> &'static MoeGemmKernels {
    KERNELS.get_or_init(|| {
        let ptx = compile_module_image_for_current_device(stream.context(), MOE_GEMM_SRC)
            .expect("moe_gemm NVRTC compile failed");
        let module = stream
            .context()
            .load_module(ptx)
            .expect("moe_gemm module load failed");
        MoeGemmKernels {
            gemm: module.load_function("fused_moe_mxfp4_gemm").unwrap(),
            _module: module,
        }
    })
}

/// Launch the fused MXFP4 grouped GEMM.
///
/// Pointers are device addresses; layouts (all contiguous):
///   a           bf16 [num_tokens, k]
///   b_q         u8   [num_experts, n, k/2]  (lo nibble = even k)
///   b_scale     u8   [num_experts, n, k/32] (e8m0)
///   c           bf16 [num_pairs, n]
///   bias        bf16 [num_experts, n]; pass 0 for none
///   topk_weights f32 [num_pairs]
///   sorted/expert/num_post: the align outputs (align run with block_size==BM)
///
/// `em` = length of sorted_token_ids; `num_valid_tokens` = num_pairs (the
/// sentinel value in sorted ids).
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_mxfp4_gemm(
    stream: &Arc<CudaStream>,
    a_ptr: u64,
    b_q_ptr: u64,
    b_scale_ptr: u64,
    c_ptr: u64,
    bias_ptr: u64,
    topk_weights_ptr: u64,
    sorted_token_ids_ptr: u64,
    expert_ids_ptr: u64,
    num_tokens_post_padded_ptr: u64,
    n: usize,
    k: usize,
    em: usize,
    num_valid_tokens: usize,
    top_k: usize,
    mul_routed_weight: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        k % 64 == 0,
        "K must be a multiple of BK=64 (no K-tail masking beyond it)"
    );
    // em may be ragged (align capacity = numel + E*(BM-1)): tiles past the
    // padded total early-exit BEFORE touching sorted_token_ids, and the
    // padded total is always a BM multiple, so the ragged tail is never read.
    let kf = kernels(stream);
    let (n_i, k_i) = (n as i32, k as i32);
    let nvt = num_valid_tokens as i64;
    let (top_k_i, mrw_i) = (top_k as i32, mul_routed_weight as i32);
    let grid = (
        (n as u32).div_ceil(BN as u32),
        (em as u32).div_ceil(BM as u32),
        1,
    );
    let mut b = stream.launch_builder(&kf.gemm);
    b.arg(&a_ptr)
        .arg(&b_q_ptr)
        .arg(&b_scale_ptr)
        .arg(&c_ptr)
        .arg(&bias_ptr)
        .arg(&topk_weights_ptr)
        .arg(&sorted_token_ids_ptr)
        .arg(&expert_ids_ptr)
        .arg(&num_tokens_post_padded_ptr)
        .arg(&n_i)
        .arg(&k_i)
        .arg(&nvt)
        .arg(&top_k_i)
        .arg(&mrw_i);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: grid,
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0, // static smem only
        })?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::test_ref::{ChainWeights, assert_close, host_chain_reference, to_bf16_bytes};
    use super::super::{align, moe_ops};
    use super::*;
    use crate::cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

    /// Tiny deterministic RNG for test data.
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

    #[test]
    fn moe_gemm_chain_tiny() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let ptr = |b: &CudaSlice<u8>| b.device_ptr(&stream).0;

        let (tokens, top_k, e_cnt) = (13usize, 2usize, 4usize);
        let (hidden, inter) = (64usize, 64usize); // K=64, gate_up N=128, down N=64
        let gate_up_n = 2 * inter;
        let num_pairs = tokens * top_k;
        let mut rng = Lcg(7);

        // Inputs
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
        let mut topk_ids = Vec::with_capacity(num_pairs);
        let mut topk_w = Vec::with_capacity(num_pairs);
        for _ in 0..tokens {
            let first = rng.below(e_cnt);
            let second = (first + 1 + rng.below(e_cnt - 1)) % e_cnt;
            topk_ids.extend([first as i32, second as i32]);
            let w0 = 0.2 + (rng.below(60) as f32) / 100.0;
            topk_w.extend([w0, 1.0 - w0]);
        }

        // ---- Device chain ----
        let d_x = stream
            .memcpy_stod(bytemuck::cast_slice::<f32, u8>(&x))
            .unwrap();
        let d_xb = stream.alloc_zeros::<u8>(tokens * hidden * 2).unwrap();
        moe_ops::f32_to_bf16(&stream, ptr(&d_x), ptr(&d_xb), tokens * hidden).unwrap();

        let d_ids = stream
            .memcpy_stod(bytemuck::cast_slice::<i32, u8>(&topk_ids))
            .unwrap();
        let bufs = align::MoeAlignBuffers::alloc(&stream, num_pairs, e_cnt, BM).unwrap();
        align::moe_align_block_size(
            &stream,
            ptr(&d_ids),
            num_pairs,
            top_k,
            top_k,
            e_cnt,
            BM,
            &bufs,
        )
        .unwrap();

        let d_gu_q = stream.memcpy_stod(&gu_q).unwrap();
        let d_gu_s = stream.memcpy_stod(&gu_s).unwrap();
        let d_gu_b = stream.memcpy_stod(&to_bf16_bytes(&gu_bias)).unwrap();
        let d_dn_q = stream.memcpy_stod(&dn_q).unwrap();
        let d_dn_s = stream.memcpy_stod(&dn_s).unwrap();
        let d_dn_b = stream.memcpy_stod(&to_bf16_bytes(&dn_bias)).unwrap();
        let d_tw = stream
            .memcpy_stod(bytemuck::cast_slice::<f32, u8>(&topk_w))
            .unwrap();

        let em = bufs.max_num_tokens_padded;
        let d_gu_out = stream.alloc_zeros::<u8>(num_pairs * gate_up_n * 2).unwrap();
        fused_moe_mxfp4_gemm(
            &stream,
            ptr(&d_xb),
            ptr(&d_gu_q),
            ptr(&d_gu_s),
            ptr(&d_gu_out),
            ptr(&d_gu_b),
            ptr(&d_tw),
            ptr(&bufs.sorted_token_ids),
            ptr(&bufs.expert_ids),
            ptr(&bufs.num_tokens_post_pad),
            gate_up_n,
            hidden,
            em,
            num_pairs,
            top_k,
            false,
        )
        .unwrap();

        let d_hid = stream.alloc_zeros::<u8>(num_pairs * inter * 2).unwrap();
        moe_ops::swiglu_interleaved(
            &stream,
            ptr(&d_gu_out),
            ptr(&d_hid),
            num_pairs,
            inter,
            1.702,
            7.0,
        )
        .unwrap();

        // Down GEMM consumes per-pair rows: top_k=1 makes pair==row.
        let d_dn_out = stream.alloc_zeros::<u8>(num_pairs * hidden * 2).unwrap();
        fused_moe_mxfp4_gemm(
            &stream,
            ptr(&d_hid),
            ptr(&d_dn_q),
            ptr(&d_dn_s),
            ptr(&d_dn_out),
            ptr(&d_dn_b),
            ptr(&d_tw),
            ptr(&bufs.sorted_token_ids),
            ptr(&bufs.expert_ids),
            ptr(&bufs.num_tokens_post_pad),
            hidden,
            inter,
            em,
            num_pairs,
            1,
            true,
        )
        .unwrap();

        let d_out = stream.alloc_zeros::<u8>(tokens * hidden * 4).unwrap();
        moe_ops::moe_sum(&stream, ptr(&d_dn_out), ptr(&d_out), tokens, top_k, hidden).unwrap();
        stream.synchronize().unwrap();
        let got_b = stream.memcpy_dtov(&d_out).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&got_b);

        // ---- Host reference (moe_naive math) ----
        let want = host_chain_reference(
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
        assert_close(got, &want, 0.05, "chain_tiny");
    }
}
