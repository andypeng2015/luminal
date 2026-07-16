#include <cuda_bf16.h>

// Elementwise helpers around the fused MoE GEMM (see moe.cu). All are
// deliberately simple scalar kernels: none of them is a bottleneck next to
// the grouped GEMMs, and simple means testable.

extern "C" __global__ void f32_to_bf16(
    const float* __restrict__ in_, __nv_bfloat16* __restrict__ out, long long n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in_[i]);
}

// gpt-oss interleaved clamped SwiGLU, PURE activation (no bias: the GEMM's
// has_bias epilogue owns the bias). gate = col 2j, up = col 2j+1:
//   gate' = min(gate, limit); up' = clamp(up, -limit, limit)
//   out   = (up' + 1) * gate' * sigmoid(alpha * gate')
extern "C" __global__ void swiglu_interleaved(
    const __nv_bfloat16* __restrict__ gu,  // [rows, 2*inter]
    __nv_bfloat16* __restrict__ hid,       // [rows, inter]
    long long rows, int inter, float alpha, float limit
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * (long long)inter;
    if (i >= total) return;
    long long r = i / inter;
    int j = (int)(i - r * inter);
    const __nv_bfloat16* row = gu + r * (long long)(2 * inter);
    float gate = __bfloat162float(row[2 * j]);
    float up = __bfloat162float(row[2 * j + 1]);
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
    float glu = gate / (1.0f + expf(-alpha * gate));
    hid[i] = __float2bfloat16((up + 1.0f) * glu);
}

// Sum the top_k per-pair rows of each token (routing weights were already
// applied by the GEMM's mul_routed_weight epilogue). Scalar version of
// vLLM's moe_sum. Contiguous input [tokens, top_k, n]; f32 output.
extern "C" __global__ void moe_sum(
    const __nv_bfloat16* __restrict__ per_pair,  // [tokens, top_k, n]
    float* __restrict__ out,                     // [tokens, n]
    long long tokens, int top_k, int n
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = tokens * (long long)n;
    if (i >= total) return;
    long long t = i / n;
    int col = (int)(i - t * n);
    const __nv_bfloat16* base = per_pair + (t * top_k) * (long long)n + col;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        acc += __bfloat162float(base[(long long)k * n]);
    }
    out[i] = acc;
}
