#include <cuda_bf16.h>
// (The SIMT f32 correctness kernel that previously lived here lost the
// forced-kernel bench to the mma variant at every seq in 16..128 and was
// deleted — see the moe_gemm.rs bench and the dispatch commit message.)

__constant__ float FP4_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ─────────────────────────────────────────────────────────────────────────
// Tensor-core variant: BM=16 expert blocks, one m16n8k16 accumulator tile
// per warp. The fp4 layout is mma-native: for m16n8k16 row.col, one packed
// fp4 byte (two adjacent k values of one weight row) is exactly one .b32
// B-operand register — weights stay packed in smem and dequant happens in
// registers at fragment-load time, so the <=16-token weight reuse happens
// inside the mma with no expanded bf16 B tile. Dequant goes through a
// 16-entry SHARED-memory LUT (broadcast lds): __constant__ with
// lane-divergent nibble indices serializes into replays (see decode.rs).
// Requires K % 64 == 0 and K <= 3072 (whole-K scale strip staged once).
// ─────────────────────────────────────────────────────────────────────────

#define MMA_BM 16
#define MMA_BN 64
#define MMA_BK 64
#define MMA_MAX_KSC 96 // K/32 <= 96 (K <= 3072)

extern "C" __global__ void fused_moe_mxfp4_gemm_mma(
    const __nv_bfloat16* __restrict__ a,        // [num_tokens, K]
    const unsigned char* __restrict__ b_q,      // [E, N, K/2], lo nibble = even k
    const unsigned char* __restrict__ b_scale,  // [E, N, K/32], e8m0
    __nv_bfloat16* __restrict__ c,              // [num_pairs, N]
    const __nv_bfloat16* __restrict__ bias,     // [E, N] or nullptr
    const float* __restrict__ topk_weights,     // [num_pairs]
    const int* __restrict__ sorted_token_ids,   // [EM]
    const int* __restrict__ expert_ids,         // [EM / MMA_BM]
    const int* __restrict__ num_tokens_post_padded,
    int N, int K, long long num_valid_tokens,
    int top_k, int mul_routed_weight
) {
    const int pid_m = blockIdx.y;
    const int pid_n = blockIdx.x;
    if (pid_m * MMA_BM >= *num_tokens_post_padded) return;

    __shared__ int s_ids[MMA_BM];
    __shared__ __nv_bfloat16 s_a[MMA_BM][MMA_BK + 8]; // +8: bank-spread pad
    __shared__ unsigned char s_bq[MMA_BN][MMA_BK / 2 + 4];
    __shared__ unsigned char s_sc[MMA_BN][MMA_MAX_KSC];
    __shared__ __nv_bfloat16 s_lut[16];

    const int tid = threadIdx.x; // 256 threads = 8 warps
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int gid = lane >> 2; // 0..7
    const int tig = lane & 3;  // 0..3

    const long long expert = expert_ids[pid_m];
    const int n0 = pid_n * MMA_BN;

    if (tid < MMA_BM) s_ids[tid] = sorted_token_ids[pid_m * MMA_BM + tid];
    if (tid < 16) s_lut[tid] = __float2bfloat16(FP4_LUT[tid]);
    __syncthreads();

    if (expert == -1) {
        for (int i = tid; i < MMA_BM * MMA_BN; i += blockDim.x) {
            const int r = i / MMA_BN, col = n0 + i % MMA_BN;
            const long long pair = s_ids[r];
            if (pair < num_valid_tokens && col < N)
                c[pair * N + col] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned char* bq_e = b_q + expert * (long long)N * (K / 2);
    const unsigned char* bs_e = b_scale + expert * (long long)N * (K / 32);

    // Whole-K e8m0 strip for this n-tile, staged once per block: every scale
    // byte is globally read exactly once (each (expert, n-strip) block is
    // unique in the grid).
    const int ksc = K / 32;
    for (int i = tid; i < MMA_BN * ksc; i += blockDim.x) {
        const int n = i / ksc, g = i % ksc;
        s_sc[n][g] = (n0 + n < N) ? bs_e[(long long)(n0 + n) * ksc + g] : 127;
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int nl = 8 * warp + gid; // this lane's B column within the tile

    for (int k0 = 0; k0 < K; k0 += MMA_BK) {
        __syncthreads(); // previous iteration's frag reads done
        // A tile: token-gathered rows, zero-filled for padding pairs.
        for (int i = tid; i < MMA_BM * MMA_BK; i += blockDim.x) {
            const int r = i / MMA_BK, kk = i % MMA_BK;
            const long long pair = s_ids[r];
            __nv_bfloat16 v = __float2bfloat16(0.0f);
            if (pair < num_valid_tokens) {
                const long long token = pair / top_k;
                v = a[token * K + k0 + kk];
            }
            s_a[r][kk] = v;
        }
        // B tile: raw packed fp4 bytes, 32 per row.
        for (int i = tid; i < MMA_BN * (MMA_BK / 2); i += blockDim.x) {
            const int n = i / (MMA_BK / 2), bb = i % (MMA_BK / 2);
            s_bq[n][bb] =
                (n0 + n < N) ? bq_e[(long long)(n0 + n) * (K / 2) + k0 / 2 + bb] : 0;
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < 4; ++s) {
            // A fragments (m16k16 row-major): rows gid/gid+8, k pairs
            // {2tig, 2tig+1} and {+8, +9} within this k16 step.
            const unsigned a0 = *(const unsigned*)&s_a[gid][16 * s + 2 * tig];
            const unsigned a1 = *(const unsigned*)&s_a[gid + 8][16 * s + 2 * tig];
            const unsigned a2 = *(const unsigned*)&s_a[gid][16 * s + 2 * tig + 8];
            const unsigned a3 = *(const unsigned*)&s_a[gid + 8][16 * s + 2 * tig + 8];

            // B fragments (k16n8 col-major): column nl, k pairs {2tig,2tig+1}
            // and {2tig+8,2tig+9} — one packed byte each.
            const unsigned char byte0 = s_bq[nl][8 * s + tig];
            const unsigned char byte1 = s_bq[nl][8 * s + tig + 4];
            // One 32-wide e8m0 group covers this whole k16 step.
            const unsigned scbits = ((unsigned)s_sc[nl][(k0 + 16 * s) >> 5]) << 7;
            __nv_bfloat162 sc2;
            *(unsigned*)&sc2 = scbits | (scbits << 16);
            __nv_bfloat162 b0 = __halves2bfloat162(s_lut[byte0 & 0xF], s_lut[byte0 >> 4]);
            __nv_bfloat162 b1 = __halves2bfloat162(s_lut[byte1 & 0xF], s_lut[byte1 >> 4]);
            b0 = __hmul2(b0, sc2);
            b1 = __hmul2(b1, sc2);
            const unsigned rb0 = *(const unsigned*)&b0;
            const unsigned rb1 = *(const unsigned*)&b1;

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(rb0), "r"(rb1));
        }
    }

    // Epilogue: lane holds rows gid/gid+8, columns 2tig/2tig+1 of its warp's
    // 8-column strip. Bias + optional routing weight, packed bf16x2 store.
    const int ncol = n0 + 8 * warp + 2 * tig;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = gid + 8 * h;
        const long long pair = s_ids[r];
        if (pair >= num_valid_tokens) continue;
        const float rw = mul_routed_weight ? topk_weights[pair] : 1.0f;
        float v0 = acc[2 * h + 0];
        float v1 = acc[2 * h + 1];
        if (bias) {
            v0 += __bfloat162float(bias[expert * N + ncol]);
            v1 += __bfloat162float(bias[expert * N + ncol + 1]);
        }
        v0 *= rw;
        v1 *= rw;
        if (ncol + 1 < N) {
            const __nv_bfloat162 out2 =
                __halves2bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
            *(__nv_bfloat162*)&c[pair * N + ncol] = out2;
        } else if (ncol < N) {
            c[pair * N + ncol] = __float2bfloat16(v0);
        }
    }
}
