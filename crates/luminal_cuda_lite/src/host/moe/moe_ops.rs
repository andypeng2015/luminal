//! Elementwise helper blocks around the fused MoE GEMM (`moe.cu`):
//! `f32_to_bf16`, the PURE interleaved clamped SwiGLU (no bias — the GEMM's
//! `has_bias` epilogue owns bias now), and `moe_sum` (plain top-k sum; the
//! routing weight was already applied by the GEMM's `mul_routed_weight`
//! epilogue). See `moe_gemm.rs` for the pipeline these compose into.

use std::sync::{Arc, OnceLock};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
};

const MOE_OPS_SRC: &str = include_str!("moe_ops.cu");

struct MoeOpsKernels {
    _module: Arc<CudaModule>,
    f32_to_bf16: CudaFunction,
    swiglu: CudaFunction,
    moe_sum: CudaFunction,
}

/// Process-wide kernel cache (compile once per process; the per-instance
/// OnceLock anti-pattern recompiles NVRTC per clone).
static KERNELS: OnceLock<MoeOpsKernels> = OnceLock::new();

/// Force the process-wide NVRTC compile of this module (idempotent). Called
/// from host-op prewarm so the compile cost lands in the untimed prebuild
/// phase, never inside a profiling trial or first real execution.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernels(stream);
}

fn kernels(stream: &Arc<CudaStream>) -> &'static MoeOpsKernels {
    KERNELS.get_or_init(|| {
        let ptx = compile_module_image_for_current_device(stream.context(), MOE_OPS_SRC)
            .expect("moe_ops NVRTC compile failed");
        let module = stream
            .context()
            .load_module(ptx)
            .expect("moe_ops module load failed");
        MoeOpsKernels {
            f32_to_bf16: module.load_function("f32_to_bf16").unwrap(),
            swiglu: module.load_function("swiglu_interleaved").unwrap(),
            moe_sum: module.load_function("moe_sum").unwrap(),
            _module: module,
        }
    })
}

fn cfg_1d(total: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((total as u32).div_ceil(256).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Cast `n` f32 elements to bf16.
pub fn f32_to_bf16(
    stream: &Arc<CudaStream>,
    in_ptr: u64,
    out_ptr: u64,
    n: usize,
) -> anyhow::Result<()> {
    let k = kernels(stream);
    let n_ll = n as i64;
    let mut b = stream.launch_builder(&k.f32_to_bf16);
    b.arg(&in_ptr).arg(&out_ptr).arg(&n_ll);
    unsafe { b.launch(cfg_1d(n))? };
    Ok(())
}

/// gpt-oss interleaved clamped SwiGLU (pure activation, no bias):
/// `gu [rows, 2*inter]` bf16 -> `hid [rows, inter]` bf16.
pub fn swiglu_interleaved(
    stream: &Arc<CudaStream>,
    gu_ptr: u64,
    hid_ptr: u64,
    rows: usize,
    inter: usize,
    alpha: f32,
    limit: f32,
) -> anyhow::Result<()> {
    let k = kernels(stream);
    let rows_ll = rows as i64;
    let inter_i = inter as i32;
    let mut b = stream.launch_builder(&k.swiglu);
    b.arg(&gu_ptr)
        .arg(&hid_ptr)
        .arg(&rows_ll)
        .arg(&inter_i)
        .arg(&alpha)
        .arg(&limit);
    unsafe { b.launch(cfg_1d(rows * inter))? };
    Ok(())
}

/// Sum the top_k per-pair rows per token: `per_pair [tokens, top_k, n]` bf16
/// -> `out [tokens, n]` f32.
pub fn moe_sum(
    stream: &Arc<CudaStream>,
    per_pair_ptr: u64,
    out_ptr: u64,
    tokens: usize,
    top_k: usize,
    n: usize,
) -> anyhow::Result<()> {
    let k = kernels(stream);
    let tokens_ll = tokens as i64;
    let (top_k_i, n_i) = (top_k as i32, n as i32);
    let mut b = stream.launch_builder(&k.moe_sum);
    b.arg(&per_pair_ptr)
        .arg(&out_ptr)
        .arg(&tokens_ll)
        .arg(&top_k_i)
        .arg(&n_i);
    unsafe { b.launch(cfg_1d(tokens * n))? };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_ref::{
        bf16_bits_to_f32, f32_to_bf16_bits, from_bf16_bytes, to_bf16_bytes,
    };
    use super::*;
    use crate::cudarc::driver::{CudaContext, DevicePtr};

    #[test]
    fn moe_ops_helpers() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let ptr = |b: &crate::cudarc::driver::CudaSlice<u8>| b.device_ptr(&stream).0;

        // f32 -> bf16
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.37).collect();
        let d_in = stream
            .memcpy_stod(bytemuck::cast_slice::<f32, u8>(&xs))
            .unwrap();
        let d_out = stream.alloc_zeros::<u8>(xs.len() * 2).unwrap();
        f32_to_bf16(&stream, ptr(&d_in), ptr(&d_out), xs.len()).unwrap();
        stream.synchronize().unwrap();
        let got = from_bf16_bytes(&stream.memcpy_dtov(&d_out).unwrap());
        for (g, x) in got.iter().zip(&xs) {
            assert_eq!(*g, bf16_bits_to_f32(f32_to_bf16_bits(*x)), "cast mismatch");
        }

        // swiglu_interleaved vs host math (incl. clamp boundaries)
        let (rows, inter) = (7usize, 33usize);
        let gu: Vec<f32> = (0..rows * 2 * inter)
            .map(|i| ((i * 2654435761usize % 1000) as f32 - 500.0) / 40.0) // spans ±12.5 -> clamps hit
            .collect();
        let gu_b = to_bf16_bytes(&gu);
        let d_gu = stream.memcpy_stod(&gu_b).unwrap();
        let d_hid = stream.alloc_zeros::<u8>(rows * inter * 2).unwrap();
        swiglu_interleaved(&stream, ptr(&d_gu), ptr(&d_hid), rows, inter, 1.702, 7.0).unwrap();
        stream.synchronize().unwrap();
        let got = from_bf16_bytes(&stream.memcpy_dtov(&d_hid).unwrap());
        for r in 0..rows {
            for j in 0..inter {
                let gate = bf16_bits_to_f32(f32_to_bf16_bits(gu[r * 2 * inter + 2 * j])).min(7.0);
                let up = bf16_bits_to_f32(f32_to_bf16_bits(gu[r * 2 * inter + 2 * j + 1]))
                    .clamp(-7.0, 7.0);
                let want = (up + 1.0) * (gate / (1.0 + (-1.702f32 * gate).exp()));
                let g = got[r * inter + j];
                assert!(
                    (g - want).abs() <= want.abs().max(1.0) * 0.02,
                    "swiglu ({r},{j}): got {g} want {want}"
                );
            }
        }

        // moe_sum vs host
        let (tokens, top_k, n) = (5usize, 4usize, 17usize);
        let pp: Vec<f32> = (0..tokens * top_k * n)
            .map(|i| ((i * 40503usize % 200) as f32 - 100.0) / 8.0)
            .collect();
        let d_pp = stream.memcpy_stod(&to_bf16_bytes(&pp)).unwrap();
        let d_sum = stream.alloc_zeros::<u8>(tokens * n * 4).unwrap();
        moe_sum(&stream, ptr(&d_pp), ptr(&d_sum), tokens, top_k, n).unwrap();
        stream.synchronize().unwrap();
        let got_b = stream.memcpy_dtov(&d_sum).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&got_b);
        for t in 0..tokens {
            for col in 0..n {
                let want: f32 = (0..top_k)
                    .map(|k| bf16_bits_to_f32(f32_to_bf16_bits(pp[(t * top_k + k) * n + col])))
                    .sum();
                let g = got[t * n + col];
                assert!(
                    (g - want).abs() <= want.abs().max(1.0) * 1e-3,
                    "moe_sum ({t},{col}): got {g} want {want}"
                );
            }
        }
    }
}
