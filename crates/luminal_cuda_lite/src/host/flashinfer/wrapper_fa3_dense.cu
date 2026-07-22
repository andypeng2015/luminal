// FA3/Hopper (SM90) dense bidirectional single-prefill for luminal_cuda.
//
// Fused softmax(Q·K^T·scale)·V over plain strided Q/K/V tensors — no KV
// cache, no mask, no plan phase. The Hopper SinglePrefillParams carry
// independent n/h strides for Q, K, V AND O, so the kernel both reads
// head-interleaved projection views and writes luminal's head-major
// [heads, qo_len, head_dim] output directly (no transpose pass).
//
// JIT-compiled at runtime with -DLUMINAL_HEAD_DIM=N and -arch=sm_90a
// (WGMMA/TMA — Hopper only). f16/bf16, fp32 accumulate.
//
// This TU is a hand-rendered equivalent of what FlashInfer's Python JIT
// generates from csrc/single_prefill_sm90_customize_config.jinja for the
// default (StandardAttention) variant, minus the tvm-ffi shell.

#ifndef LUMINAL_HEAD_DIM
#error "LUMINAL_HEAD_DIM must be defined (e.g. -DLUMINAL_HEAD_DIM=128)"
#endif

// The SM90 single prefill only supports head_dim 64 / 128 / 256.
#if LUMINAL_HEAD_DIM != 64 && LUMINAL_HEAD_DIM != 128 && LUMINAL_HEAD_DIM != 256
#error "FA3 single prefill supports LUMINAL_HEAD_DIM of 64, 128, or 256 only"
#endif

#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>

#include "wrapper_fa3_dense.h"

#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace flashinfer;

constexpr uint32_t HEAD_DIM = LUMINAL_HEAD_DIM;

// dtype codes shared with the Rust side (jit.rs / mod.rs).
constexpr int LUMINAL_DTYPE_F16 = 1;
constexpr int LUMINAL_DTYPE_BF16 = 2;

template <typename T>
static int fa3_dense_run_t(
    T* q, T* k, T* v, T* o,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    long long q_stride_n, long long q_stride_h,
    long long k_stride_n, long long k_stride_h,
    long long v_stride_n, long long v_stride_h,
    long long o_stride_n, long long o_stride_h,
    float sm_scale,
    cudaStream_t stream)
{
    using TC = cutlass_dtype_t<T>;
    using Params = SinglePrefillParams<TC, TC, TC>;
    Params params;
    std::memset(&params, 0, sizeof(params));
    params.q_ptr = reinterpret_cast<TC*>(q);
    params.k_ptr = reinterpret_cast<TC*>(k);
    params.v_ptr = reinterpret_cast<TC*>(v);
    params.o_ptr = reinterpret_cast<TC*>(o);
    params.lse_ptr = nullptr;
    params.additional_params.logits_soft_cap = 0.0f;
    params.additional_params.sm_scale = sm_scale;
    params.additional_params.scale_q = nullptr;
    params.additional_params.scale_k = nullptr;
    params.additional_params.scale_v = nullptr;
    params.q_stride_n = q_stride_n;
    params.k_stride_n = k_stride_n;
    params.v_stride_n = v_stride_n;
    params.o_stride_n = o_stride_n;
    params.q_stride_h = q_stride_h;
    params.k_stride_h = k_stride_h;
    params.v_stride_h = v_stride_h;
    params.o_stride_h = o_stride_h;
    params.qo_len = qo_len;
    params.kv_len = kv_len;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_qo_heads / num_kv_heads;
    params.window_left = -1;  // unused: LEFT_SLIDING_WINDOW=false below
    params.causal = false;

    cudaError_t status = SinglePrefillWithKVCacheDispatched<
        HEAD_DIM, HEAD_DIM, MaskMode::kNone, /*LEFT_SLIDING_WINDOW=*/false,
        StandardAttention, Params>(params, stream);
    return (int)status;
}

extern "C" {

int flashinfer_fa3_dense_run(
    void* q, void* k, void* v, void* output,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    long long q_stride_n, long long q_stride_h,
    long long k_stride_n, long long k_stride_h,
    long long v_stride_n, long long v_stride_h,
    long long o_stride_n, long long o_stride_h,
    int dtype, float sm_scale,
    cudaStream_t stream)
{
    switch (dtype) {
        case LUMINAL_DTYPE_F16:
            return fa3_dense_run_t<half>(
                (half*)q, (half*)k, (half*)v, (half*)output,
                qo_len, kv_len, num_qo_heads, num_kv_heads,
                q_stride_n, q_stride_h, k_stride_n, k_stride_h,
                v_stride_n, v_stride_h, o_stride_n, o_stride_h,
                sm_scale, stream);
        case LUMINAL_DTYPE_BF16:
            return fa3_dense_run_t<__nv_bfloat16>(
                (__nv_bfloat16*)q, (__nv_bfloat16*)k, (__nv_bfloat16*)v,
                (__nv_bfloat16*)output,
                qo_len, kv_len, num_qo_heads, num_kv_heads,
                q_stride_n, q_stride_h, k_stride_n, k_stride_h,
                v_stride_n, v_stride_h, o_stride_n, o_stride_h,
                sm_scale, stream);
        default:
            return -1;  // WGMMA is 16-bit only
    }
}

}  // extern "C"
