//! Launch wrapper + tests for `moe.cu`'s `fused_moe_mxfp4_gemm_mma`: the
//! grouped MoE GEMM consuming the `align` building block's outputs (tokens
//! gathered to resident MXFP4 weights). Semantics follow the Triton
//! spec-of-record at `harness/fused_moe_mxfp4_triton.py`; this CUDA version
//! uses tensor cores (BM=16 m16n8k16 tiles, register dequant, one packed fp4
//! byte per B operand register).
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
/// Tensor-core kernel tiles (fixed inside moe.cu). `BM_MMA` must equal the
/// `block_size` used when running `align::moe_align_block_size`.
pub const BM_MMA: usize = 16;
pub const BN_MMA: usize = 64;
/// The mma kernel stages the whole-K e8m0 strip in smem: K/32 <= 96.
pub const MMA_MAX_K: usize = 3072;

struct MoeGemmKernels {
    _module: Arc<CudaModule>,
    gemm_mma: CudaFunction,
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
            gemm_mma: module.load_function("fused_moe_mxfp4_gemm_mma").unwrap(),
            _module: module,
        }
    })
}

/// Launch the tensor-core (m16n8k16) variant. Same contract as
/// `fused_moe_mxfp4_gemm`, with the align step run at block_size == BM_MMA
/// and two extra static limits: K % 64 == 0 and K <= MMA_MAX_K (the whole-K
/// e8m0 strip is staged in shared memory once per block).
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_mxfp4_gemm_mma(
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
    anyhow::ensure!(k % 64 == 0, "K must be a multiple of BK=64");
    anyhow::ensure!(
        k <= MMA_MAX_K,
        "K={k} exceeds the mma kernel's staged-scale limit {MMA_MAX_K}"
    );
    let kf = kernels(stream);
    let (n_i, k_i) = (n as i32, k as i32);
    let nvt = num_valid_tokens as i64;
    let (top_k_i, mrw_i) = (top_k as i32, mul_routed_weight as i32);
    let grid = (
        (n as u32).div_ceil(BN_MMA as u32),
        (em as u32).div_ceil(BM_MMA as u32),
        1,
    );
    let mut b = stream.launch_builder(&kf.gemm_mma);
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

    /// Host oracle for ONE grouped GEMM (order-independent: the kernel
    /// writes per-pair rows, so align placement doesn't matter).
    #[allow(clippy::too_many_arguments)]
    fn host_gemm_reference(
        x: &[f32], // [tokens, k] (pre-bf16-rounded)
        bq: &[u8],
        bs: &[u8],
        bias: Option<&[f32]>,
        topk_ids: &[i32],
        topk_w: &[f32],
        tokens: usize,
        top_k: usize,
        n_dim: usize,
        k_dim: usize,
        mul_routed_weight: bool,
    ) -> Vec<f32> {
        use super::super::test_ref::{bf16_bits_to_f32, f32_to_bf16_bits, host_weight};
        let bf = |v: f32| bf16_bits_to_f32(f32_to_bf16_bits(v));
        let num_pairs = tokens * top_k;
        let mut c = vec![0.0f32; num_pairs * n_dim];
        for p in 0..num_pairs {
            let t = p / top_k;
            let e = topk_ids[p] as usize;
            let rw = if mul_routed_weight { topk_w[p] } else { 1.0 };
            for nn in 0..n_dim {
                let mut acc = 0.0f32;
                for kk in 0..k_dim {
                    acc += bf(x[t * k_dim + kk]) * host_weight(bq, bs, e, n_dim, k_dim, nn, kk);
                }
                if let Some(b) = bias {
                    acc += bf(b[e * n_dim + nn]);
                }
                c[p * n_dim + nn] = bf(acc * rw);
            }
        }
        c
    }

    /// T4: the tensor-core kernel in isolation. Cases: random ragged tokens
    /// (padded tiles + expert==-1 blocks live), one-hot x (fragment-mapping
    /// bugs become (k,n)-coordinate-diagnosable), widened e8m0 range, k-loop
    /// (K=128), routed-weight epilogue, multi n-tile (N=128).
    #[test]
    fn moe_gemm_mma_tiny() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let ptr = |b: &CudaSlice<u8>| b.device_ptr(&stream).0;

        struct Case {
            label: &'static str,
            tokens: usize,
            k_dim: usize,
            n_dim: usize,
            one_hot: bool,
            scale_lo: usize,
            scale_hi: usize,
            mul_routed: bool,
            bias: bool,
        }
        let cases = [
            Case {
                label: "mma random",
                tokens: 13,
                k_dim: 64,
                n_dim: 128,
                one_hot: false,
                scale_lo: 125,
                scale_hi: 131,
                mul_routed: false,
                bias: true,
            },
            Case {
                label: "mma one-hot",
                tokens: 16,
                k_dim: 64,
                n_dim: 64,
                one_hot: true,
                scale_lo: 127,
                scale_hi: 128,
                mul_routed: false,
                bias: false,
            },
            Case {
                label: "mma wide-scale k128 rw",
                tokens: 40,
                k_dim: 128,
                n_dim: 128,
                one_hot: false,
                scale_lo: 100,
                scale_hi: 160,
                mul_routed: true,
                bias: true,
            },
        ];
        for cs in cases {
            let (top_k, e_cnt) = (2usize, 4usize);
            let num_pairs = cs.tokens * top_k;
            let mut rng = Lcg(cs.tokens as u64 * 31 + 5);

            let x: Vec<f32> = if cs.one_hot {
                let mut v = vec![0.0f32; cs.tokens * cs.k_dim];
                for t in 0..cs.tokens {
                    v[t * cs.k_dim + (t * 7 + 3) % cs.k_dim] = 1.0;
                }
                v
            } else {
                (0..cs.tokens * cs.k_dim)
                    .map(|_| (rng.below(200) as f32 - 100.0) / 50.0)
                    .collect()
            };
            let bq: Vec<u8> = (0..e_cnt * cs.n_dim * cs.k_dim / 2)
                .map(|_| rng.below(256) as u8)
                .collect();
            let bs: Vec<u8> = (0..e_cnt * cs.n_dim * cs.k_dim / 32)
                .map(|_| (cs.scale_lo + rng.below(cs.scale_hi - cs.scale_lo)) as u8)
                .collect();
            let bias: Vec<f32> = (0..e_cnt * cs.n_dim)
                .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
                .collect();
            let mut topk_ids = Vec::with_capacity(num_pairs);
            let mut topk_w = Vec::with_capacity(num_pairs);
            for _ in 0..cs.tokens {
                let first = rng.below(e_cnt);
                let second = (first + 1 + rng.below(e_cnt - 1)) % e_cnt;
                topk_ids.extend([first as i32, second as i32]);
                let w0 = 0.2 + (rng.below(60) as f32) / 100.0;
                topk_w.extend([w0, 1.0 - w0]);
            }

            let d_x = stream.memcpy_stod(&to_bf16_bytes(&x)).unwrap();
            let d_ids = stream
                .memcpy_stod(bytemuck::cast_slice::<i32, u8>(&topk_ids))
                .unwrap();
            let d_w = stream
                .memcpy_stod(bytemuck::cast_slice::<f32, u8>(&topk_w))
                .unwrap();
            let d_bq = stream.memcpy_stod(&bq).unwrap();
            let d_bs = stream.memcpy_stod(&bs).unwrap();
            let d_bias = stream.memcpy_stod(&to_bf16_bytes(&bias)).unwrap();
            let d_c = stream.alloc_zeros::<u8>(num_pairs * cs.n_dim * 2).unwrap();

            let bufs = align::MoeAlignBuffers::alloc(&stream, num_pairs, e_cnt, BM_MMA).unwrap();
            align::moe_align_block_size(
                &stream,
                ptr(&d_ids),
                num_pairs,
                top_k,
                top_k,
                e_cnt,
                BM_MMA,
                &bufs,
            )
            .unwrap();

            fused_moe_mxfp4_gemm_mma(
                &stream,
                ptr(&d_x),
                ptr(&d_bq),
                ptr(&d_bs),
                ptr(&d_c),
                if cs.bias { ptr(&d_bias) } else { 0 },
                ptr(&d_w),
                ptr(&bufs.sorted_token_ids),
                ptr(&bufs.expert_ids),
                ptr(&bufs.num_tokens_post_pad),
                cs.n_dim,
                cs.k_dim,
                bufs.max_num_tokens_padded,
                num_pairs,
                top_k,
                cs.mul_routed,
            )
            .unwrap();
            stream.synchronize().unwrap();

            let got_b = stream.memcpy_dtov(&d_c).unwrap();
            let got: Vec<f32> = super::super::test_ref::from_bf16_bytes(&got_b);
            let x_rounded: Vec<f32> = {
                use super::super::test_ref::{bf16_bits_to_f32, f32_to_bf16_bits};
                x.iter()
                    .map(|&v| bf16_bits_to_f32(f32_to_bf16_bits(v)))
                    .collect()
            };
            let want = host_gemm_reference(
                &x_rounded,
                &bq,
                &bs,
                if cs.bias { Some(&bias) } else { None },
                &topk_ids,
                &topk_w,
                cs.tokens,
                top_k,
                cs.n_dim,
                cs.k_dim,
                cs.mul_routed,
            );
            assert_close(&got, &want, 0.05, cs.label);
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
        let bufs = align::MoeAlignBuffers::alloc(&stream, num_pairs, e_cnt, BM_MMA).unwrap();
        align::moe_align_block_size(
            &stream,
            ptr(&d_ids),
            num_pairs,
            top_k,
            top_k,
            e_cnt,
            BM_MMA,
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
        fused_moe_mxfp4_gemm_mma(
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
        fused_moe_mxfp4_gemm_mma(
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
