// Fused single-kernel MoE decode path: gate_up GEMV + clamped interleaved
// SwiGLU + down GEMV + routed-weight sum, one cooperative launch per step.
//
// At decode (s*top_k pairs small) the MoE degenerates to per-expert
// matrix-vector products with a hard global dependency in the middle: every
// down-row dot reads the ENTIRE hidden vector, so down work cannot start
// until all gate/up dots finished across all blocks. grid.sync() (cooperative
// launch) is that barrier; the launcher must size the grid to co-resident
// occupancy (see decode.rs).
//
// Replaces, per step: f32->bf16 cast, both align kernels, both tiled GEMMs,
// swiglu and moe_sum — six launches and their padding waste (tiled BM=64
// blocks carry ~4 live rows at s=1) become one launch where every weight
// byte read feeds a real output. Weight layout/decode math identical to
// moe.cu: blocks [E, N, K/2] (lo nibble = even k), e8m0 scales [E, N, K/32],
// bf16 biases.
//
// Work mapping: warp-per-output-element, grid-stride over flat task ids.
//   phase 1: task = pair * inter + j  -> hidden[pair][j]  (two dots + swiglu)
//   phase 2: task = t * hidden_dim + r -> out[t][r]       (top_k dots + sum)
// x is read as f32 directly (the op's input dtype — the cast kernel is not
// needed on this path); x/hidden stay L2-hot (KBs vs the ~52MB weight sweep).

#include <cuda_bf16.h>


__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// One scale-group's contribution to a row dot: 16 packed weight bytes
// (one aligned uint4) x 32 activation floats (eight float4), decoded from
// registers. Shared by both work mappings below.
__device__ __forceinline__ float mxfp4_group_dot(
    const unsigned char* __restrict__ qrow,
    const unsigned char* __restrict__ srow,
    int g,
    const float* __restrict__ vec,
    const float* __restrict__ lut  // SHARED-memory copy: __constant__
                                   // serializes on divergent per-nibble
                                   // indices (up to 32-way replay per read)
) {
    // e8m0 IS the IEEE-754 exponent field: 2^(byte-127) == bits(byte<<23).
    // Exact for byte in [1,254] (real gpt-oss scales sit near 124-130);
    // avoids an SFU exp2f per group.
    const float scale = __uint_as_float((unsigned int)srow[g] << 23);
    const uint4 q = *reinterpret_cast<const uint4*>(qrow + g * 16);
    const float4* v4 = reinterpret_cast<const float4*>(vec + g * 32);
    const unsigned int words[4] = {q.x, q.y, q.z, q.w};
    float gacc = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const unsigned int word = words[w];
        const float4 a = v4[2 * w];
        const float4 b = v4[2 * w + 1];
        gacc = fmaf(lut[(word >> 0) & 0xF], a.x, gacc);
        gacc = fmaf(lut[(word >> 4) & 0xF], a.y, gacc);
        gacc = fmaf(lut[(word >> 8) & 0xF], a.z, gacc);
        gacc = fmaf(lut[(word >> 12) & 0xF], a.w, gacc);

        gacc = fmaf(lut[(word >> 16) & 0xF], b.x, gacc);
        gacc = fmaf(lut[(word >> 20) & 0xF], b.y, gacc);
        gacc = fmaf(lut[(word >> 24) & 0xF], b.z, gacc);
        gacc = fmaf(lut[(word >> 28) & 0xF], b.w, gacc);
    }
    return gacc * scale;
}

// Stage the LUT into shared memory once per block.
#define STAGE_LUT(name)                       \
    __shared__ float name[16];                \
    if (threadIdx.x < 16) {                   \
        name[threadIdx.x] = FP4_LUT[threadIdx.x]; \
    }                                         \
    __syncthreads();

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;
}

// Warp-parallel dequant dot: row (n) of expert e against a dense f32 vector.
// K % 32 == 0 (one e8m0 group = 32 fp4 values = 16 bytes).
__device__ __forceinline__ float mxfp4_row_dot(
    const unsigned char* __restrict__ b_q,
    const unsigned char* __restrict__ b_scale,
    long long expert,
    int n_dim,
    int k_dim,
    int row,
    const float* __restrict__ vec,
    int lane,
    const float* __restrict__ lut
) {
    const unsigned char* qrow = b_q + (expert * n_dim + row) * (long long)(k_dim / 2);
    const unsigned char* srow = b_scale + (expert * n_dim + row) * (long long)(k_dim / 32);
    float acc = 0.0f;
    for (int g = lane; g < k_dim / 32; g += 32) {
        acc += mxfp4_group_dot(qrow, srow, g, vec, lut);
    }
    return warp_reduce_sum(acc); // valid in lane 0 only
}

// ─────────────────────────────────────────────────────────────────────────
// The MoE decode path: two regular launches, phase order enforced by stream
// order. This replaced a single cooperative kernel with a grid.sync()
// between the phases: the barrier required every block to be co-resident,
// capping the grid and forcing multi-wave grid-strides — measured 5-16%
// slower across seq 1..16 (406 -> 466 GB/s weight-read at seq=16). Split
// kernels launch one warp per task with unbounded grids.
// ─────────────────────────────────────────────────────────────────────────

// Phase 1: gate/up dequant dots + clamped-SwiGLU epilogue -> hidden scratch.
extern "C" __global__ void moe_phase1(
    unsigned long long x_ptr,
    unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr,
    unsigned long long gu_bias_ptr,
    unsigned long long topk_ids_ptr,
    unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride,
    float alpha, float limit
) {
    const float* x = (const float*)x_ptr;
    const unsigned char* gu_q = (const unsigned char*)gu_q_ptr;
    const unsigned char* gu_scale = (const unsigned char*)gu_scale_ptr;
    const __nv_bfloat16* gu_bias = (const __nv_bfloat16*)gu_bias_ptr;
    const int* topk_ids = (const int*)topk_ids_ptr;
    float* hidden = (float*)hidden_ptr;
    STAGE_LUT(slut)
    const int lane = threadIdx.x % 32;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int total_warps = (gridDim.x * blockDim.x) / 32;
    const int num_pairs = seq * top_k;
    const int gate_up_n = 2 * inter;
    for (long long task = warp_global; task < (long long)num_pairs * inter; task += total_warps) {
        const int pair = (int)(task / inter);
        const int j = (int)(task % inter);
        const int t = pair / top_k;
        const long long e = topk_ids[(long long)t * idx_row_stride + pair % top_k];
        const float* xt = x + (long long)t * hidden_dim;
        const float gate_dot = mxfp4_row_dot(gu_q, gu_scale, e, gate_up_n, hidden_dim, 2 * j, xt, lane, slut);
        const float up_dot = mxfp4_row_dot(gu_q, gu_scale, e, gate_up_n, hidden_dim, 2 * j + 1, xt, lane, slut);
        if (lane == 0) {
            float gate = gate_dot + __bfloat162float(gu_bias[e * gate_up_n + 2 * j]);
            float up = up_dot + __bfloat162float(gu_bias[e * gate_up_n + 2 * j + 1]);
            gate = fminf(gate, limit);
            up = fminf(fmaxf(up, -limit), limit);
            const float sig = 1.0f / (1.0f + expf(-alpha * gate));
            hidden[(long long)pair * inter + j] = (up + 1.0f) * gate * sig;
        }
    }
}

// Phase 2: down dequant dots + in-register weighted mix over top_k.
extern "C" __global__ void moe_phase2(
    unsigned long long dn_q_ptr,
    unsigned long long dn_scale_ptr,
    unsigned long long dn_bias_ptr,
    unsigned long long topk_ids_ptr,
    unsigned long long topk_w_ptr,
    unsigned long long hidden_ptr,
    unsigned long long out_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride
) {
    const unsigned char* dn_q = (const unsigned char*)dn_q_ptr;
    const unsigned char* dn_scale = (const unsigned char*)dn_scale_ptr;
    const __nv_bfloat16* dn_bias = (const __nv_bfloat16*)dn_bias_ptr;
    const int* topk_ids = (const int*)topk_ids_ptr;
    const float* topk_w = (const float*)topk_w_ptr;
    float* hidden = (float*)hidden_ptr;
    float* out = (float*)out_ptr;
    STAGE_LUT(slut)
    const int lane = threadIdx.x % 32;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int total_warps = (gridDim.x * blockDim.x) / 32;
    for (long long task = warp_global; task < (long long)seq * hidden_dim; task += total_warps) {
        const int t = (int)(task / hidden_dim);
        const int r = (int)(task % hidden_dim);
        float acc = 0.0f;
        for (int kk = 0; kk < top_k; ++kk) {
            const long long e = topk_ids[(long long)t * idx_row_stride + kk];
            const int pair = t * top_k + kk;
            const float dot =
                mxfp4_row_dot(dn_q, dn_scale, e, hidden_dim, inter, r, hidden + (long long)pair * inter, lane, slut);
            if (lane == 0) {
                const float w = topk_w[(long long)t * top_k + kk];
                acc = fmaf(w, dot + __bfloat162float(dn_bias[e * hidden_dim + r]), acc);
            }
        }
        if (lane == 0) {
            out[(long long)t * hidden_dim + r] = acc;
        }
    }
}

// Test-only: expose the dot primitive for one (expert,row) so the wrapper's
// tests can verify it in isolation.
extern "C" __global__ void debug_row_dot(
    unsigned long long b_q_ptr,
    unsigned long long b_scale_ptr,
    unsigned long long vec_ptr,
    unsigned long long out_ptr,
    int expert,
    int n_dim,
    int k_dim,
    int row
) {
    STAGE_LUT(slut)
    const float r = mxfp4_row_dot(
        (const unsigned char*)b_q_ptr,
        (const unsigned char*)b_scale_ptr,
        expert,
        n_dim,
        k_dim,
        row,
        (const float*)vec_ptr,
        (int)(threadIdx.x % 32),
        slut
    );
    if (threadIdx.x == 0) {
        *(float*)out_ptr = r;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Row-blocked variants: R output elements per warp sharing ONE activation
// read. ncu on the R=1 kernels (seq=1 real dims) showed DRAM at 11% while
// the L1/TEX pipe sat at 96%: each warp re-reads 128 activation bytes per
// 16 weight bytes (8:1), so the LSU pipe saturates serving L1 hits and
// throttles the DRAM weight stream. Blocking R rows per warp cuts the
// activation traffic R-fold and runs R independent weight streams.
// ─────────────────────────────────────────────────────────────────────────

// One scale-group step for R rows: activation float4s loaded ONCE into
// registers, dotted against R packed weight rows.
template <int R>
__device__ __forceinline__ void mxfp4_group_dot_rows(
    const unsigned char* __restrict__ b_q,
    const unsigned char* __restrict__ b_scale,
    long long expert,
    int n_dim,
    int k_dim,
    int row0, // rows row0 .. row0+R-1
    int g,
    const float* __restrict__ vec,
    const float* __restrict__ lut,
    float* __restrict__ acc // [R]
) {
    const float4* v4 = reinterpret_cast<const float4*>(vec + g * 32);
    const float4 a0 = v4[0], a1 = v4[1], a2 = v4[2], a3 = v4[3];
    const float4 a4 = v4[4], a5 = v4[5], a6 = v4[6], a7 = v4[7];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const long long row = row0 + r;
        const unsigned char* qrow = b_q + (expert * n_dim + row) * (long long)(k_dim / 2);
        const unsigned char* srow = b_scale + (expert * n_dim + row) * (long long)(k_dim / 32);
        const float scale = __uint_as_float((unsigned int)srow[g] << 23);
        const uint4 q = *reinterpret_cast<const uint4*>(qrow + g * 16);
        const unsigned int words[4] = {q.x, q.y, q.z, q.w};
        float gacc = 0.0f;
        const float4 av[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned int word = words[w];
            const float4 a = av[2 * w];
            const float4 b = av[2 * w + 1];
            gacc = fmaf(lut[(word >> 0) & 0xF], a.x, gacc);
            gacc = fmaf(lut[(word >> 4) & 0xF], a.y, gacc);
            gacc = fmaf(lut[(word >> 8) & 0xF], a.z, gacc);
            gacc = fmaf(lut[(word >> 12) & 0xF], a.w, gacc);
            gacc = fmaf(lut[(word >> 16) & 0xF], b.x, gacc);
            gacc = fmaf(lut[(word >> 20) & 0xF], b.y, gacc);
            gacc = fmaf(lut[(word >> 24) & 0xF], b.z, gacc);
            gacc = fmaf(lut[(word >> 28) & 0xF], b.w, gacc);
        }
        acc[r] = fmaf(gacc, scale, acc[r]);
    }
}

// Phase 1, row-blocked: warp task = (pair, j-block of R hidden columns);
// 2R weight rows (gate/up interleaved) share each activation read.
template <int R>
__device__ __forceinline__ void phase1_body(
    unsigned long long x_ptr, unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,
    unsigned long long topk_ids_ptr, unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride,
    float alpha, float limit
) {
    const float* x = (const float*)x_ptr;
    const unsigned char* gu_q = (const unsigned char*)gu_q_ptr;
    const unsigned char* gu_scale = (const unsigned char*)gu_scale_ptr;
    const __nv_bfloat16* gu_bias = (const __nv_bfloat16*)gu_bias_ptr;
    const int* topk_ids = (const int*)topk_ids_ptr;
    float* hidden = (float*)hidden_ptr;
    STAGE_LUT(slut)
    const int lane = threadIdx.x % 32;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int total_warps = (gridDim.x * blockDim.x) / 32;
    const int jblocks = inter / R;
    const int gate_up_n = 2 * inter;
    const long long total = (long long)seq * top_k * jblocks;
    for (long long task = warp_global; task < total; task += total_warps) {
        const int pair = (int)(task / jblocks);
        const int j0 = (int)(task % jblocks) * R;
        const int t = pair / top_k;
        const long long e = topk_ids[(long long)t * idx_row_stride + pair % top_k];
        const float* xt = x + (long long)t * hidden_dim;
        float acc[2 * R];
#pragma unroll
        for (int r = 0; r < 2 * R; ++r) acc[r] = 0.0f;
        for (int g = lane; g < hidden_dim / 32; g += 32) {
            mxfp4_group_dot_rows<2 * R>(
                gu_q, gu_scale, e, gate_up_n, hidden_dim, 2 * j0, g, xt, slut, acc);
        }
#pragma unroll
        for (int r = 0; r < 2 * R; ++r) acc[r] = warp_reduce_sum(acc[r]);
        if (lane == 0) {
#pragma unroll
            for (int jj = 0; jj < R; ++jj) {
                float gate = acc[2 * jj] + __bfloat162float(gu_bias[e * gate_up_n + 2 * (j0 + jj)]);
                float up = acc[2 * jj + 1] + __bfloat162float(gu_bias[e * gate_up_n + 2 * (j0 + jj) + 1]);
                gate = fminf(gate, limit);
                up = fminf(fmaxf(up, -limit), limit);
                const float sig = 1.0f / (1.0f + expf(-alpha * gate));
                hidden[(long long)pair * inter + j0 + jj] = (up + 1.0f) * gate * sig;
            }
        }
    }
}

// Phase 2, row-blocked: warp task = (token, r-block of R output rows);
// each of the top_k hidden vectors is read once per group for R rows.
template <int R>
__device__ __forceinline__ void phase2_body(
    unsigned long long dn_q_ptr, unsigned long long dn_scale_ptr,
    unsigned long long dn_bias_ptr, unsigned long long topk_ids_ptr,
    unsigned long long topk_w_ptr, unsigned long long hidden_ptr,
    unsigned long long out_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride
) {
    const unsigned char* dn_q = (const unsigned char*)dn_q_ptr;
    const unsigned char* dn_scale = (const unsigned char*)dn_scale_ptr;
    const __nv_bfloat16* dn_bias = (const __nv_bfloat16*)dn_bias_ptr;
    const int* topk_ids = (const int*)topk_ids_ptr;
    const float* topk_w = (const float*)topk_w_ptr;
    float* hidden = (float*)hidden_ptr;
    float* out = (float*)out_ptr;
    STAGE_LUT(slut)
    const int lane = threadIdx.x % 32;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int total_warps = (gridDim.x * blockDim.x) / 32;
    const int rblocks = hidden_dim / R;
    const long long total = (long long)seq * rblocks;
    for (long long task = warp_global; task < total; task += total_warps) {
        const int t = (int)(task / rblocks);
        const int r0 = (int)(task % rblocks) * R;
        float mix[R];
#pragma unroll
        for (int r = 0; r < R; ++r) mix[r] = 0.0f;
        for (int kk = 0; kk < top_k; ++kk) {
            const long long e = topk_ids[(long long)t * idx_row_stride + kk];
            const int pair = t * top_k + kk;
            const float* hv = hidden + (long long)pair * inter;
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            for (int g = lane; g < inter / 32; g += 32) {
                mxfp4_group_dot_rows<R>(dn_q, dn_scale, e, hidden_dim, inter, r0, g, hv, slut, acc);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = warp_reduce_sum(acc[r]);
            if (lane == 0) {
                const float w = topk_w[(long long)t * top_k + kk];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    mix[r] = fmaf(w, acc[r] + __bfloat162float(dn_bias[e * hidden_dim + r0 + r]), mix[r]);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) out[(long long)t * hidden_dim + r0 + r] = mix[r];
        }
    }
}

extern "C" __global__ void moe_phase1_r2(
    unsigned long long x_ptr, unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,
    unsigned long long topk_ids_ptr, unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride,
    float alpha, float limit
) {
    phase1_body<2>(x_ptr, gu_q_ptr, gu_scale_ptr, gu_bias_ptr, topk_ids_ptr,
                   hidden_ptr, hidden_dim, inter, top_k, seq, idx_row_stride,
                   alpha, limit);
}

extern "C" __global__ void moe_phase1_r4(
    unsigned long long x_ptr, unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,
    unsigned long long topk_ids_ptr, unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride,
    float alpha, float limit
) {
    phase1_body<4>(x_ptr, gu_q_ptr, gu_scale_ptr, gu_bias_ptr, topk_ids_ptr,
                   hidden_ptr, hidden_dim, inter, top_k, seq, idx_row_stride,
                   alpha, limit);
}

extern "C" __global__ void moe_phase2_r2(
    unsigned long long dn_q_ptr, unsigned long long dn_scale_ptr,
    unsigned long long dn_bias_ptr, unsigned long long topk_ids_ptr,
    unsigned long long topk_w_ptr, unsigned long long hidden_ptr,
    unsigned long long out_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride
) {
    phase2_body<2>(dn_q_ptr, dn_scale_ptr, dn_bias_ptr, topk_ids_ptr,
                   topk_w_ptr, hidden_ptr, out_ptr, hidden_dim, inter, top_k,
                   seq, idx_row_stride);
}

extern "C" __global__ void moe_phase2_r4(
    unsigned long long dn_q_ptr, unsigned long long dn_scale_ptr,
    unsigned long long dn_bias_ptr, unsigned long long topk_ids_ptr,
    unsigned long long topk_w_ptr, unsigned long long hidden_ptr,
    unsigned long long out_ptr,
    int hidden_dim, int inter, int top_k, int seq, int idx_row_stride
) {
    phase2_body<4>(dn_q_ptr, dn_scale_ptr, dn_bias_ptr, topk_ids_ptr,
                   topk_w_ptr, hidden_ptr, out_ptr, hidden_dim, inter, top_k,
                   seq, idx_row_stride);
}
