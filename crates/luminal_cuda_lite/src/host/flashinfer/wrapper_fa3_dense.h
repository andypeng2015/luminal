#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// FA3/Hopper (SM90) dense bidirectional single prefill (f16 / bf16 only).
// Fused softmax(Q·K^T·scale)·V over plain strided tensors — no KV cache, no
// mask, no plan phase. All strides are in elements; head_dim is contiguous
// in every tensor, and every stride must keep the 16-byte TMA alignment
// (stride * elem_size % 16 == 0). Output strides are free, so the kernel
// can write head-major [heads, qo_len, head_dim] directly.
// dtype codes: 1 = f16, 2 = bf16. Returns 0 on success.
int flashinfer_fa3_dense_run(
    void* q, void* k, void* v, void* output,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    long long q_stride_n, long long q_stride_h,
    long long k_stride_n, long long k_stride_h,
    long long v_stride_n, long long v_stride_h,
    long long o_stride_n, long long o_stride_h,
    int dtype, float sm_scale,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
