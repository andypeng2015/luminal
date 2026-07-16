#include <cuda_bf16.h>

#define BM 64
#define BN 64
#define BK 64
#define TM 4
#define TN 4
// 256 threads = (BM/TM) * (BN/TN)

__constant__ float FP4_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void fused_moe_mxfp4_gemm(
    const __nv_bfloat16* __restrict__ a,        // [num_tokens, K]
    const unsigned char* __restrict__ b_q,      // [E, N, K/2], lo nibble = even k
    const unsigned char* __restrict__ b_scale,  // [E, N, K/32], e8m0
    __nv_bfloat16* __restrict__ c,              // [num_pairs, N]
    const __nv_bfloat16* __restrict__ bias,     // [E, N] or nullptr
    const float* __restrict__ topk_weights,     // [num_pairs]
    const int* __restrict__ sorted_token_ids,   // [EM]
    const int* __restrict__ expert_ids,         // [EM / BM]
    const int* __restrict__ num_tokens_post_padded,
    int N, int K, long long num_valid_tokens,
    int top_k, int mul_routed_weight
) {
    const int pid_m = blockIdx.y;
    const int pid_n = blockIdx.x;
    if (pid_m * BM >= *num_tokens_post_padded) return;

    __shared__ int   s_ids[BM];
    __shared__ float s_a[BM][BK + 1];
    __shared__ float s_b[BK][BN];

    const int tid = threadIdx.x;
    const int trow = tid / (BN / TN);            // 0..15
    const int tcol = tid % (BN / TN);            // 0..15

    if (tid < BM) s_ids[tid] = sorted_token_ids[pid_m * BM + tid];
    __syncthreads();

    const long long expert = expert_ids[pid_m];
    const int n0 = pid_n * BN;

    if (expert == -1) {
        for (int i = tid; i < BM * BN; i += blockDim.x) {
            const int r = i / BN, col = n0 + i % BN;
            const long long pair = s_ids[r];
            if (pair < num_valid_tokens && col < N)
                c[pair * N + col] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned char* bq_e = b_q     + expert * (long long)N * (K / 2);
    const unsigned char* bs_e = b_scale + expert * (long long)N * (K / 32);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // stage A: [BM, BK] bf16 -> f32, zero masked rows / K tail
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            const int r = i / BK, kk = i % BK;
            const long long pair = s_ids[r];
            float v = 0.0f;
            if (pair < num_valid_tokens && k0 + kk < K)
                v = __bfloat162float(a[(pair / top_k) * (long long)K + k0 + kk]);
            s_a[r][kk] = v;
        }
        // stage B: each packed byte read once, decoded to two f32 rows
        for (int i = tid; i < (BK / 2) * BN; i += blockDim.x) {
            const int kb = i / BN, col = i % BN;
            const int n = n0 + col, k = k0 + 2 * kb;
            float lo = 0.0f, hi = 0.0f;
            if (n < N && k < K) {
                const unsigned char byte = bq_e[(long long)n * (K / 2) + k / 2];
                const float sc = exp2f((float)bs_e[(long long)n * (K / 32) + k / 32] - 127.0f);
                lo = FP4_LUT[byte & 0xF] * sc;
                if (k + 1 < K) hi = FP4_LUT[byte >> 4] * sc;
            }
            s_b[2 * kb][col] = lo;
            s_b[2 * kb + 1][col] = hi;
        }
        __syncthreads();

        #pragma unroll 8
        for (int kk = 0; kk < BK; kk++) {
            float ar[TM], br[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) ar[i] = s_a[trow * TM + i][kk];
            #pragma unroll
            for (int j = 0; j < TN; j++) br[j] = s_b[kk][tcol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int r = trow * TM + i;
        const long long pair = s_ids[r];
        if (pair >= num_valid_tokens) continue;
        const float rw = mul_routed_weight ? topk_weights[pair] : 1.0f;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int col = n0 + tcol * TN + j;
            if (col >= N) continue;
            float v = acc[i][j];
            if (bias) v += __bfloat162float(bias[expert * (long long)N + col]);
            v *= rw;
            c[pair * (long long)N + col] = __float2bfloat16(v);
        }
    }
}
