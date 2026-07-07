//! Fused MXFP4 grouped-MoE HostOp for gpt-oss.
//!
//! Like [`super::GLUMoE`], but the expert weights stay **MXFP4-packed** in
//! resident memory and a single gathered expert slice is dequantized inside the
//! op (into a reused scratch buffer) right before each per-expert cuBLASLt
//! matmul. This avoids materializing the `[s, k, gate_up_dim, hidden]` /
//! `[s, k, hidden, intermediate]` dequantized weight tensors in the captured
//! graph arena (the ~17 GB blowup; see LUM-645). It also implements gpt-oss's
//! clamped interleaved SwiGLU and the per-expert gate_up / down biases.
//!
//! Inputs (graph edges, in order):
//!   0: x               [seq, hidden]                 F32
//!   1: topk_indices    [seq, k]                      Int
//!   2: topk_values     [seq, k]                      F32
//!   3: gate_up_blocks  [E, gate_up_dim, hidden/2]    U8   (MXFP4 packed)
//!   4: gate_up_scales  [E, gate_up_dim, hidden/32]   U8   (e8m0)
//!   5: gate_up_bias    [E, gate_up_dim]              BF16
//!   6: down_blocks     [E, hidden, intermediate/2]   U8   (MXFP4 packed)
//!   7: down_scales     [E, hidden, intermediate/32]  U8   (e8m0)
//!   8: down_bias       [E, hidden]                   BF16
//!
//! Output: [seq, hidden] F32. `gate_up_dim = 2 * intermediate` (interleaved
//! gate/up lanes). top-k weights are applied directly (no normalization).

use std::sync::{Arc, OnceLock};

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    host::{DeviceBuffer, HostOp},
};

use super::{buf_ptr, slice_ptr};

const MXFP4_BLOCK: usize = 32;
// Used only by the reference activation in the unit test (the kernel inlines
// these as literals in its NVRTC source).
#[cfg(test)]
const SWIGLU_LIMIT: f32 = 7.0;
#[cfg(test)]
const SWIGLU_ALPHA: f32 = 1.702;

/// Compiled kernels for the fused MXFP4 MoE.
///
/// Cached PROCESS-WIDE (same pattern as flashinfer's FLASHINFER_LIBS): the
/// CUDA source is a fixed string identical for every op instance, so one
/// NVRTC compile serves the whole process. A per-instance cache here is a
/// serious footgun — compile-time search extracts a FRESH op instance for
/// every candidate graph, and a per-instance lazy compile (~1.2s) then lands
/// in each candidate's first execute, dominating profiling. Single-context
/// assumption, same as the flashinfer statics.
struct MoeKernels {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    f32_to_bf16: CudaFunction,
    /// Per-(token,expert)-pair GEMV path (small batches / kill switch).
    gemv_gu: CudaFunction,
    gemv_dn: CudaFunction,
    /// Grouped tensor-core GEMM path (pairs sorted by expert, vLLM-style).
    align: CudaFunction,
    gemm_gu: CudaFunction,
    gemm_dn: CudaFunction,
    zero_f32: CudaFunction,
    /// Shared by both paths.
    glu: CudaFunction,
    reduce: CudaFunction,
}

/// Tile sizes of the grouped GEMM kernels (must match the CUDA source).
const GEMM_BM: usize = 16;
const GEMM_BN: usize = 64;
/// Max experts supported by moe_align's shared-memory arrays.
const ALIGN_MAX_E: usize = 256;

pub struct GLUMoEMXFP4 {
    /// Hidden dim H (= gate_up matmul K, = down matmul output rows).
    hidden: Expression,
    /// Expert intermediate dim I (= down matmul K). gate_up_dim = 2*I.
    intermediate: Expression,
    /// Number of experts summed per token (top_k).
    output_k: Expression,
}

static MOE_KERNELS: OnceLock<MoeKernels> = OnceLock::new();

impl Default for GLUMoEMXFP4 {
    fn default() -> Self {
        Self {
            hidden: Expression::default(),
            intermediate: Expression::default(),
            output_k: Expression::default(),
        }
    }
}

impl std::fmt::Debug for GLUMoEMXFP4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLUMoEMXFP4")
            .field("hidden", &self.hidden)
            .field("intermediate", &self.intermediate)
            .field("output_k", &self.output_k)
            .finish()
    }
}

impl Clone for GLUMoEMXFP4 {
    fn clone(&self) -> Self {
        Self {
            hidden: self.hidden,
            intermediate: self.intermediate,
            output_k: self.output_k,
        }
    }
}

impl GLUMoEMXFP4 {
    pub(crate) fn new(hidden: Expression, intermediate: Expression, output_k: Expression) -> Self {
        Self {
            hidden,
            intermediate,
            output_k,
        }
    }

    fn get_kernels(&self, stream: &Arc<CudaStream>) -> &'static MoeKernels {
        MOE_KERNELS.get_or_init(|| {
            let src = r#"
#include <cuda_bf16.h>

extern "C" __global__ void f32_to_bf16(unsigned long long in_ptr, unsigned long long out_ptr, int n) {
    const float* in_ = (const float*)in_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in_[i]);
}

// Fused MXFP4 GEMV: y = W @ x, dequantizing W on the fly (no bf16 weight
// materialization). One warp per output row; x staged in shared memory and
// reused by every row in the block; coalesced FP4 byte reads (consecutive lanes
// read consecutive `blocks` bytes, each byte = two columns); warp-shuffle reduce.
// Numerics mirror the old dequant: w = LUT[nib]*2^(sb-127), rounded to bf16 then
// back to f32 before the MAC; x read bf16->f32; accumulate in f32.
//   blocks: u8 [out_dim, in_dim/2]   scales: u8 [out_dim, in_dim/32]   x: bf16 [in_dim]
__device__ __forceinline__ float mxfp4_row_dot(
    const unsigned char* brow, const unsigned char* srow,
    const __nv_bfloat16* xs, int in_dim, int lane
) {
    const float LUT[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                           -0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    int in_half = in_dim >> 1;
    float acc = 0.f;
    for (int t = lane; t < in_half; t += 32) {
        unsigned char byte = brow[t];
        int c_even = t << 1;
        unsigned char sb = srow[c_even >> 5];
        float scale = exp2f((float)((int)sb - 127));
        float w_lo = __bfloat162float(__float2bfloat16(LUT[byte & 0x0F] * scale));
        float w_hi = __bfloat162float(__float2bfloat16(LUT[byte >> 4]   * scale));
        acc += w_lo * __bfloat162float(xs[c_even]);
        acc += w_hi * __bfloat162float(xs[c_even + 1]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    return acc; // valid on lane 0
}

// Batched MoE over ALL (token,expert) pairs per layer: each kernel is launched
// ONCE (grid covers every pair) instead of once per pair, so the CPU issues ~4
// launches/layer instead of 4*seq*top_k. Routing is read on-GPU: for pair p the
// token is t=p/top_k, slot i=p%top_k, and the expert index (clamped to
// [0,num_experts)) + routing weight come from the topk buffers.
__device__ __forceinline__ int pair_expert(
    unsigned long long topk_idx_base, int p, int top_k, int idx_stride, int num_experts
) {
    int t = p / top_k, i = p - t * top_k;
    int e = ((const int*)topk_idx_base)[(long)t * idx_stride + i];
    return e < 0 ? 0 : (e >= num_experts ? num_experts - 1 : e);
}

// Batched gate_up GEMV. grid=(rows/warps, num_pairs). gu_out[p,o]=W[e_p]@x[t_p].
extern "C" __global__ void mxfp4_gemv_batched(
    unsigned long long blocks_base, unsigned long long scales_base,
    unsigned long long blk_stride, unsigned long long sc_stride,
    unsigned long long topk_idx_base, int top_k, int idx_stride, int num_experts,
    unsigned long long x_base, unsigned long long out_base,
    int out_dim, int in_dim
) {
    extern __shared__ __nv_bfloat16 xs[];
    int p = blockIdx.y;
    int t = p / top_k;
    int e = pair_expert(topk_idx_base, p, top_k, idx_stride, num_experts);
    const __nv_bfloat16* x = (const __nv_bfloat16*)(x_base + (unsigned long long)t * in_dim * 2);
    int tid = threadIdx.y * 32 + threadIdx.x, nthreads = blockDim.x * blockDim.y;
    for (int c = tid; c < in_dim; c += nthreads) xs[c] = x[c];
    __syncthreads();
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;
    const unsigned char* brow = (const unsigned char*)(blocks_base + (unsigned long long)e * blk_stride) + (long)row * (in_dim >> 1);
    const unsigned char* srow = (const unsigned char*)(scales_base + (unsigned long long)e * sc_stride) + (long)row * (in_dim >> 5);
    float acc = mxfp4_row_dot(brow, srow, xs, in_dim, threadIdx.x);
    if (threadIdx.x == 0) ((__nv_bfloat16*)out_base)[(long)p * out_dim + row] = __float2bfloat16(acc);
}

// Batched clamped interleaved SwiGLU + per-expert gate_up bias. One thread per
// (pair, j). hid[p,j] = act(gu[p,2j]+bias[2j], gu[p,2j+1]+bias[2j+1]).
extern "C" __global__ void glu_batched(
    unsigned long long gu_base,
    unsigned long long bias_base, unsigned long long bias_stride,
    unsigned long long topk_idx_base, int top_k, int idx_stride, int num_experts,
    unsigned long long out_base, int intermediate, int num_pairs
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)num_pairs * intermediate) return;
    int p = idx / intermediate, j = idx - (long)p * intermediate;
    int e = pair_expert(topk_idx_base, p, top_k, idx_stride, num_experts);
    const __nv_bfloat16* gu = (const __nv_bfloat16*)(gu_base + (unsigned long long)p * (2 * intermediate) * 2);
    const __nv_bfloat16* bias = (const __nv_bfloat16*)(bias_base + (unsigned long long)e * bias_stride);
    float gate = __bfloat162float(gu[2*j])   + __bfloat162float(bias[2*j]);
    float up   = __bfloat162float(gu[2*j+1]) + __bfloat162float(bias[2*j+1]);
    gate = fminf(gate, 7.0f);
    up   = fminf(fmaxf(up, -7.0f), 7.0f);
    float glu = gate / (1.0f + expf(-1.702f * gate));
    ((__nv_bfloat16*)out_base)[(long)p * intermediate + j] = __float2bfloat16((up + 1.0f) * glu);
}

// Batched down GEMV (raw W@hid, no weight/bias). grid=(rows/warps, num_pairs).
// dn_out[p,o] f32 (full precision; the reduce applies weight+bias).
extern "C" __global__ void mxfp4_down_batched(
    unsigned long long blocks_base, unsigned long long scales_base,
    unsigned long long blk_stride, unsigned long long sc_stride,
    unsigned long long topk_idx_base, int top_k, int idx_stride, int num_experts,
    unsigned long long hid_base, unsigned long long out_base,
    int out_dim, int in_dim
) {
    extern __shared__ __nv_bfloat16 xs[];
    int p = blockIdx.y;
    int e = pair_expert(topk_idx_base, p, top_k, idx_stride, num_experts);
    const __nv_bfloat16* x = (const __nv_bfloat16*)(hid_base + (unsigned long long)p * in_dim * 2);
    int tid = threadIdx.y * 32 + threadIdx.x, nthreads = blockDim.x * blockDim.y;
    for (int c = tid; c < in_dim; c += nthreads) xs[c] = x[c];
    __syncthreads();
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;
    const unsigned char* brow = (const unsigned char*)(blocks_base + (unsigned long long)e * blk_stride) + (long)row * (in_dim >> 1);
    const unsigned char* srow = (const unsigned char*)(scales_base + (unsigned long long)e * sc_stride) + (long)row * (in_dim >> 5);
    float acc = mxfp4_row_dot(brow, srow, xs, in_dim, threadIdx.x);
    if (threadIdx.x == 0) ((float*)out_base)[(long)p * out_dim + row] = acc;
}

// Deterministic weighted reduce over the top_k experts (fixed order, no atomics):
// out[t,r] = sum_i topk_w[t,i] * (dn_out[t*top_k+i, r] + dn_bias[e_i, r]).
extern "C" __global__ void moe_reduce(
    unsigned long long dn_out_base,
    unsigned long long dn_bias_base, unsigned long long dn_bias_stride,
    unsigned long long topk_idx_base, unsigned long long topk_vals_base,
    int top_k, int idx_stride, int val_stride, int num_experts,
    unsigned long long out_base, int seq, int hidden
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)seq * hidden) return;
    int t = idx / hidden, r = idx - (long)t * hidden;
    float sum = 0.f;
    for (int i = 0; i < top_k; i++) {
        int e = ((const int*)topk_idx_base)[(long)t * idx_stride + i];
        e = e < 0 ? 0 : (e >= num_experts ? num_experts - 1 : e);
        float w = ((const float*)topk_vals_base)[(long)t * val_stride + i];
        float dn = ((const float*)dn_out_base)[(long)(t * top_k + i) * hidden + r];
        const __nv_bfloat16* bias = (const __nv_bfloat16*)(dn_bias_base + (unsigned long long)e * dn_bias_stride);
        sum += w * (dn + __bfloat162float(bias[r]));
    }
    ((float*)out_base)[idx] = sum;
}

// ───────────────────── grouped-GEMM path (vLLM-style) ─────────────────────
// Sort (token,expert) pairs by expert so each expert's weights are read ONCE
// per step and multiplied against all its tokens with tensor-core MMA, instead
// of one GEMV per pair re-reading the weights (see mxfp4_gemv_batched above).

// Counting sort of pair ids by expert, padded to BLOCK_M per expert.
// One block, two passes over pairs (<= ~4096) + serial prefix over E (<= 256).
// Outputs: sorted_ids [padded, sentinel=num_pairs], expert_ids [per m-block],
// num_post_pad [1]. Expert clamping matches pair_expert().
#define ALIGN_MAX_E 256
extern "C" __global__ void moe_align(
    unsigned long long topk_idx_base, int top_k, int idx_stride, int num_experts,
    int num_pairs, int block_m,
    unsigned long long sorted_ids_base,
    unsigned long long expert_ids_base,
    unsigned long long num_post_pad_base
) {
    __shared__ int cnt[ALIGN_MAX_E];
    __shared__ int off[ALIGN_MAX_E];
    __shared__ int total_sh;
    int* sorted_ids = (int*)sorted_ids_base;
    int* expert_ids = (int*)expert_ids_base;
    int tid = threadIdx.x;

    for (int e = tid; e < num_experts; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = tid; p < num_pairs; p += blockDim.x) {
        int e = pair_expert(topk_idx_base, p, top_k, idx_stride, num_experts);
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int e = 0; e < num_experts; e++) {
            off[e] = c;
            c += ((cnt[e] + block_m - 1) / block_m) * block_m;
        }
        total_sh = c;
        *(int*)num_post_pad_base = c;
    }
    __syncthreads();
    int total = total_sh;
    for (int i = tid; i < total; i += blockDim.x) sorted_ids[i] = num_pairs;
    __syncthreads();
    for (int e = tid; e < num_experts; e += blockDim.x) {
        int nblk = (cnt[e] + block_m - 1) / block_m;
        int b0 = off[e] / block_m;
        for (int b = 0; b < nblk; b++) expert_ids[b0 + b] = e;
        cnt[e] = 0; // reuse as scatter cursor
    }
    __syncthreads();
    for (int p = tid; p < num_pairs; p += blockDim.x) {
        int e = pair_expert(topk_idx_base, p, top_k, idx_stride, num_experts);
        sorted_ids[off[e] + atomicAdd(&cnt[e], 1)] = p;
    }
}

// Grouped MXFP4 GEMM: one block computes a [BM=16, BN=64] output tile for ONE
// expert (expert_ids[blockIdx.x]), gathering A rows via sorted_ids and
// scatter-writing C back in ORIGINAL pair order (so glu_batched / moe_reduce
// stay unchanged). Weights are dequantized to bf16 in shared memory once per
// [BN, BK=64] tile and reused by all BM rows via mma.sync m16n8k16.
// 128 threads = 4 warps; warp w owns output cols [16w, 16w+16).
// A row for pair p is p/row_div (row_div = top_k for gate_up reading x, 1 for
// down reading hid which is per-pair).
#define GEMM_BM 16
#define GEMM_BN 64
#define GEMM_BK 64
// Epilogue modes: how the accumulated [BM, BN] tile leaves the kernel.
//   EPI_GLU:    gate_up GEMM. Each lane's adjacent (col, col+1) accumulator
//               pair IS the interleaved (gate, up) pair for hid column col/2
//               (col is provably even). Fuses the per-expert bias add and the
//               clamped interleaved SwiGLU (limit 7, alpha 1.702, (up+1)*glu)
//               and writes ONE bf16 to hid[pid * (out_dim/2) + col/2] -- the
//               full-width gu_out tensor and the separate glu kernel vanish.
//   EPI_SCATTER: down GEMM. Fuses the per-expert down bias and the top-k
//               routing weight, atomicAdd-accumulating w*(acc+bias) straight
//               into the final [tokens, hidden] f32 output (which must be
//               zeroed first) -- dn_out and the separate reduce vanish.
//               Accumulation order across a token's k pairs is atomic, i.e.
//               NOT bitwise run-to-run deterministic (accepted trade-off).
#define EPI_GLU 0
#define EPI_SCATTER 1

template<int EPI>
__device__ __forceinline__ void mxfp4_gemm_impl(
    unsigned long long blocks_base, unsigned long long scales_base,
    unsigned long long blk_stride, unsigned long long sc_stride,
    unsigned long long sorted_ids_base, unsigned long long expert_ids_base,
    unsigned long long num_post_pad_base,
    int num_pairs, int row_div,
    unsigned long long a_base, unsigned long long c_base,
    int out_dim, int in_dim,
    unsigned long long bias_base, unsigned long long bias_stride,
    unsigned long long topk_vals_base, int val_stride, int top_k
) {
    const float LUT[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                           -0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    int num_post = *(const int*)num_post_pad_base;
    if (blockIdx.x * GEMM_BM >= num_post) return; // overprovisioned grid
    int e = ((const int*)expert_ids_base)[blockIdx.x];

    // Double-buffered pipeline: while stage k computes (dequant + mma), stage
    // k+1's global loads are in flight via cp.async. A-tiles land in smA
    // directly (bf16, no transform); B's raw fp4 bytes land in a small staging
    // buffer and are dequanted to smB at consume time (smem->smem, fast).
    // +8 halves of padding per row: fragment loads hit distinct banks.
    __shared__ __nv_bfloat16 smA[2][GEMM_BM][GEMM_BK + 8];
    __shared__ __nv_bfloat16 smB[GEMM_BN][GEMM_BK + 8];
    __shared__ unsigned char smRawB[2][GEMM_BN][GEMM_BK / 2]; // 32 fp4 bytes/row/stage
    __shared__ int spair[GEMM_BM];

    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int grp = lane >> 2, q = lane & 3;

    if (tid < GEMM_BM) {
        spair[tid] = ((const int*)sorted_ids_base)[blockIdx.x * GEMM_BM + tid];
    }
    __syncthreads();

    const unsigned char* Bblk = (const unsigned char*)(blocks_base + (unsigned long long)e * blk_stride);
    const unsigned char* Bsc  = (const unsigned char*)(scales_base + (unsigned long long)e * sc_stride);
    int n0 = blockIdx.y * GEMM_BN;
    int in_half = in_dim >> 1;
    int in_sc = in_dim >> 5;

    // Per-thread copy slots (constant across stages).
    int a_m = tid >> 3, a_seg = tid & 7;           // A: 16 rows x 8 16B segs
    int a_pid = spair[a_m];
    const __nv_bfloat16* a_row = (a_pid < num_pairs)
        ? (const __nv_bfloat16*)(a_base) + (long)(a_pid / row_div) * in_dim
        : (const __nv_bfloat16*)0;
    int b_row = tid >> 1, b_g = tid & 1;           // B: 64 rows x 2 16B granules
    const unsigned char* b_src_row = Bblk + (long)(n0 + b_row) * in_half + b_g * 16;
    // Padding rows: zero their smA slots once (cp.async below only overwrites
    // valid rows; zeros persist for invalid ones across all stages).
    if (a_pid >= num_pairs) {
        *(uint4*)&smA[0][a_m][a_seg * 8] = uint4{0u, 0u, 0u, 0u};
        *(uint4*)&smA[1][a_m][a_seg * 8] = uint4{0u, 0u, 0u, 0u};
    }
    __syncthreads();

    // 16-byte async copy helper (sm_80+).
    #define CP_ASYNC16(dst_smem, src_gmem)                                        \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::            \
            "r"((unsigned)__cvta_generic_to_shared(dst_smem)), "l"(src_gmem))
    #define CP_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
    #define CP_WAIT1() asm volatile("cp.async.wait_group 1;\n" ::)
    #define CP_WAIT0() asm volatile("cp.async.wait_group 0;\n" ::)

    int n_kb = in_dim / GEMM_BK;
    // Prologue: issue stage 0 loads.
    if (a_pid < num_pairs) CP_ASYNC16(&smA[0][a_m][a_seg * 8], a_row + a_seg * 8);
    CP_ASYNC16(&smRawB[0][b_row][b_g * 16], b_src_row);
    CP_COMMIT();

    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};

    for (int kb = 0; kb < n_kb; kb++) {
        int cur = kb & 1, nxt = cur ^ 1;
        // Issue stage kb+1 loads (overlap with this stage's compute).
        if (kb + 1 < n_kb) {
            if (a_pid < num_pairs)
                CP_ASYNC16(&smA[nxt][a_m][a_seg * 8], a_row + (kb + 1) * GEMM_BK + a_seg * 8);
            CP_ASYNC16(&smRawB[nxt][b_row][b_g * 16], b_src_row + (kb + 1) * 32);
            CP_COMMIT();
            CP_WAIT1(); // stage kb complete; kb+1 may still be in flight
        } else {
            CP_WAIT0();
        }
        __syncthreads();

        // Dequant this stage's raw B bytes (smem -> smem) into smB.
        {
            unsigned char sb = Bsc[(long)(n0 + b_row) * in_sc + kb * 2 + b_g];
            float scale = exp2f((float)((int)sb - 127));
            const unsigned char* rb = &smRawB[cur][b_row][b_g * 16];
            #pragma unroll
            for (int by = 0; by < 16; by++) {
                unsigned char b = rb[by];
                smB[b_row][b_g * 32 + by * 2]     = __float2bfloat16(LUT[b & 0x0F] * scale);
                smB[b_row][b_g * 32 + by * 2 + 1] = __float2bfloat16(LUT[b >> 4]   * scale);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < GEMM_BK / 16; ks++) {
            unsigned a0 = *(const unsigned*)&smA[cur][grp    ][ks * 16 + 2 * q];
            unsigned a1 = *(const unsigned*)&smA[cur][grp + 8][ks * 16 + 2 * q];
            unsigned a2 = *(const unsigned*)&smA[cur][grp    ][ks * 16 + 2 * q + 8];
            unsigned a3 = *(const unsigned*)&smA[cur][grp + 8][ks * 16 + 2 * q + 8];
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int n = warp * 16 + j * 8 + grp;
                unsigned b0 = *(const unsigned*)&smB[n][ks * 16 + 2 * q];
                unsigned b1 = *(const unsigned*)&smB[n][ks * 16 + 2 * q + 8];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    // Epilogue: lane holds D rows {grp, grp+8}, cols {2q, 2q+1} within each
    // warp's j-th n8 column; rows scatter back to ORIGINAL pair order.
    const __nv_bfloat16* ebias =
        (const __nv_bfloat16*)(bias_base + (unsigned long long)e * bias_stride);
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            int m = grp + 8 * h;
            int pid = spair[m];
            if (pid >= num_pairs) continue;
            long col = (long)n0 + warp * 16 + j * 8 + 2 * q;
            if (EPI == EPI_GLU) {
                // (col, col+1) = the interleaved (gate, up) pair of hid col/2.
                float gate = acc[j][2 * h]     + __bfloat162float(ebias[col]);
                float up   = acc[j][2 * h + 1] + __bfloat162float(ebias[col + 1]);
                gate = fminf(gate, 7.0f);
                up   = fminf(fmaxf(up, -7.0f), 7.0f);
                float glu = gate / (1.0f + expf(-1.702f * gate));
                ((__nv_bfloat16*)c_base)[(long)pid * (out_dim >> 1) + (col >> 1)] =
                    __float2bfloat16((up + 1.0f) * glu);
            } else {
                // Weighted scatter-accumulate into the final [tokens, hidden]
                // f32 output (zeroed beforehand).
                int t = pid / top_k, slot = pid - t * top_k;
                float w = ((const float*)topk_vals_base)[(long)t * val_stride + slot];
                float* out_row = (float*)c_base + (long)t * out_dim;
                atomicAdd(&out_row[col],
                          w * (acc[j][2 * h] + __bfloat162float(ebias[col])));
                atomicAdd(&out_row[col + 1],
                          w * (acc[j][2 * h + 1] + __bfloat162float(ebias[col + 1])));
            }
        }
    }
}

extern "C" __global__ void zero_f32(unsigned long long ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ((float*)ptr)[i] = 0.0f;
}

extern "C" __global__ void mxfp4_gemm_gu(
    unsigned long long blocks_base, unsigned long long scales_base,
    unsigned long long blk_stride, unsigned long long sc_stride,
    unsigned long long sorted_ids_base, unsigned long long expert_ids_base,
    unsigned long long num_post_pad_base,
    int num_pairs, int row_div,
    unsigned long long a_base, unsigned long long c_base,
    int out_dim, int in_dim,
    unsigned long long bias_base, unsigned long long bias_stride,
    unsigned long long topk_vals_base, int val_stride, int top_k
) {
    mxfp4_gemm_impl<EPI_GLU>(blocks_base, scales_base, blk_stride, sc_stride,
        sorted_ids_base, expert_ids_base, num_post_pad_base,
        num_pairs, row_div, a_base, c_base, out_dim, in_dim,
        bias_base, bias_stride, topk_vals_base, val_stride, top_k);
}

extern "C" __global__ void mxfp4_gemm_dn(
    unsigned long long blocks_base, unsigned long long scales_base,
    unsigned long long blk_stride, unsigned long long sc_stride,
    unsigned long long sorted_ids_base, unsigned long long expert_ids_base,
    unsigned long long num_post_pad_base,
    int num_pairs, int row_div,
    unsigned long long a_base, unsigned long long c_base,
    int out_dim, int in_dim,
    unsigned long long bias_base, unsigned long long bias_stride,
    unsigned long long topk_vals_base, int val_stride, int top_k
) {
    mxfp4_gemm_impl<EPI_SCATTER>(blocks_base, scales_base, blk_stride, sc_stride,
        sorted_ids_base, expert_ids_base, num_post_pad_base,
        num_pairs, row_div, a_base, c_base, out_dim, in_dim,
        bias_base, bias_stride, topk_vals_base, val_stride, top_k);
}
"#;
            let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            MoeKernels {
                f32_to_bf16: module.load_function("f32_to_bf16").unwrap(),
                gemv_gu: module.load_function("mxfp4_gemv_batched").unwrap(),
                gemv_dn: module.load_function("mxfp4_down_batched").unwrap(),
                align: module.load_function("moe_align").unwrap(),
                gemm_gu: module.load_function("mxfp4_gemm_gu").unwrap(),
                gemm_dn: module.load_function("mxfp4_gemm_dn").unwrap(),
                zero_f32: module.load_function("zero_f32").unwrap(),
                glu: module.load_function("glu_batched").unwrap(),
                reduce: module.load_function("moe_reduce").unwrap(),
                module,
            }
        })
    }
}

impl EgglogOp for GLUMoEMXFP4 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "GLUMoEMXFP4",
            &[
                ("hidden", EXPRESSION),
                ("intermediate", EXPRESSION),
                ("output_k", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                (
                    (= ?e (Op (GLUMoEMXFP4 ?hidden ?intermediate ?output_k) ?inputs))
                )
                (
                    (set (dtype ?e) (F32))
                )
                :ruleset dtype_prop
            )",
            ),
            Rule::raw(include_str!["glumoe_mxfp4_rewrite.egg"]),
        ]
    }

    fn n_inputs(&self) -> usize {
        9
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let hidden = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let intermediate = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let output_k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let extracted = GLUMoEMXFP4::new(hidden, intermediate, output_k);
        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for GLUMoEMXFP4 {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 9 {
            anyhow::bail!("GLUMoEMXFP4 expected 9 inputs, got {}", inputs.len());
        }

        let hidden = self
            .hidden
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 hidden unresolved"))?;
        let intermediate = self
            .intermediate
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 intermediate unresolved"))?;
        let top_k = self
            .output_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 top_k unresolved"))?;
        if hidden == 0 || intermediate == 0 {
            anyhow::bail!("GLUMoEMXFP4 zero dims: hidden={hidden}, intermediate={intermediate}");
        }
        if top_k == 0 {
            return Ok(());
        }
        if hidden % MXFP4_BLOCK != 0 || intermediate % MXFP4_BLOCK != 0 {
            anyhow::bail!(
                "GLUMoEMXFP4 dims must be multiples of {MXFP4_BLOCK}: hidden={hidden}, intermediate={intermediate}"
            );
        }
        let gate_up_dim = 2 * intermediate;

        let output_bytes = self
            .output_bytes()
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 output bytes unresolved"))?;
        if output_bytes % (hidden * 4) != 0 {
            anyhow::bail!("GLUMoEMXFP4 output bytes {output_bytes} not divisible by hidden*4");
        }
        let seq = output_bytes / (hidden * 4);
        if seq == 0 {
            return Ok(());
        }

        let get_buffer = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers
                .get(&node)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 missing {name} buffer for {node:?}"))
        };

        let x_buf = get_buffer("x", inputs[0])?;
        let topk_idx_buf = get_buffer("topk indices", inputs[1])?;
        let topk_vals_buf = get_buffer("topk values", inputs[2])?;
        let gu_blocks_buf = get_buffer("gate_up blocks", inputs[3])?;
        let gu_scales_buf = get_buffer("gate_up scales", inputs[4])?;
        let gu_bias_buf = get_buffer("gate_up bias", inputs[5])?;
        let dn_blocks_buf = get_buffer("down blocks", inputs[6])?;
        let dn_scales_buf = get_buffer("down scales", inputs[7])?;
        let dn_bias_buf = get_buffer("down bias", inputs[8])?;
        let output_buf = get_buffer("output", self_node)?;

        // Per-expert byte strides into the packed weight tensors.
        let gu_blocks_stride = gate_up_dim * (hidden / 2); // u8
        let gu_scales_stride = gate_up_dim * (hidden / MXFP4_BLOCK); // u8
        let gu_bias_stride = gate_up_dim * 2; // bf16
        let dn_blocks_stride = hidden * (intermediate / 2);
        let dn_scales_stride = hidden * (intermediate / MXFP4_BLOCK);
        let dn_bias_stride = hidden * 2;

        if gu_blocks_stride == 0 || gu_blocks_buf.len() % gu_blocks_stride != 0 {
            anyhow::bail!(
                "GLUMoEMXFP4 gate_up blocks buffer {} not a multiple of per-expert stride {gu_blocks_stride}",
                gu_blocks_buf.len()
            );
        }
        let num_experts = gu_blocks_buf.len() / gu_blocks_stride;
        if num_experts == 0 {
            anyhow::bail!("GLUMoEMXFP4 has no experts");
        }

        let x_ptr = buf_ptr(x_buf, stream);
        let gu_blocks_ptr = buf_ptr(gu_blocks_buf, stream);
        let gu_scales_ptr = buf_ptr(gu_scales_buf, stream);
        let gu_bias_ptr = buf_ptr(gu_bias_buf, stream);
        let dn_blocks_ptr = buf_ptr(dn_blocks_buf, stream);
        let dn_scales_ptr = buf_ptr(dn_scales_buf, stream);
        let dn_bias_ptr = buf_ptr(dn_bias_buf, stream);
        let output_ptr = buf_ptr(output_buf, stream);

        let kernels = self.get_kernels(stream);
        let (f32_to_bf16_fn, gemv_fn, down_fn, act_fn, reduce_fn) = (
            &kernels.f32_to_bf16,
            &kernels.gemv_gu,
            &kernels.gemv_dn,
            &kernels.glu,
            &kernels.reduce,
        );

        // Clean non-perturbing timers (LUMINAL_MOE_PROF2): routing (host DtoH
        // sync) vs the kernel-launch loop, no per-phase syncs added.
        let prof2 = std::env::var_os("LUMINAL_MOE_PROF2").is_some();
        let rt0 = std::time::Instant::now();

        // GPU-side routing: pass the topk index/value BUFFERS into the kernels;
        // each kernel reads its slot's expert index + weight from device memory
        // (clamped to [0, num_experts)) and offsets the per-expert weights itself.
        // No DtoH copy => no per-layer host sync. Match read_routing's indexing:
        // the buffer row stride may exceed top_k (gating output width), so derive
        // it from the buffer length (i32/f32 = 4 bytes) rather than assuming top_k.
        let topk_idx_base = buf_ptr(topk_idx_buf, stream);
        let topk_vals_base = buf_ptr(topk_vals_buf, stream);
        let idx_stride = (topk_idx_buf.len() / 4)
            .checked_div(seq)
            .filter(|s| *s >= top_k)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 bad topk index buffer length"))?;
        let val_stride = (topk_vals_buf.len() / 4)
            .checked_div(seq)
            .filter(|s| *s >= top_k)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 bad topk value buffer length"))?;
        let routing_ms = if prof2 { rt0.elapsed().as_secs_f64() * 1e3 } else { 0.0 };
        let lp0 = std::time::Instant::now();

        // One launch per kernel over ALL (token,expert) pairs this step.
        let num_pairs = seq * top_k;
        // Reused scratch (freed at end of execute): per-pair gate_up / hidden /
        // down buffers. No dequantized weights (fused GEMVs read FP4 directly).
        let x_bf16 = unsafe { stream.alloc::<u8>(seq * hidden * 2)? };
        let hid = unsafe { stream.alloc::<u8>(num_pairs * intermediate * 2)? };
        let xbf16_ptr = slice_ptr(&x_bf16, stream);
        let hid_ptr = slice_ptr(&hid, stream);

        let launch = |f: &CudaFunction, n: usize, args: &[KernelArg]| -> anyhow::Result<()> {
            let blocks = (n as u32).div_ceil(256);
            let mut b = stream.launch_builder(f);
            for a in args {
                match a {
                    KernelArg::U64(v) => b.arg(v),
                    KernelArg::I32(v) => b.arg(v),
                    KernelArg::F32(v) => b.arg(v),
                };
            }
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            }
            Ok(())
        };

        // Batched MXFP4 GEMV launch: one warp per output row (grid.x), one grid.y
        // per (token,expert) pair; x staged in shared per block.
        const GEMV_WARPS: u32 = 8;
        let launch_gemv = |f: &CudaFunction,
                           out_dim: usize,
                           in_dim: usize,
                           pairs: usize,
                           args: &[KernelArg]|
         -> anyhow::Result<()> {
            let blocks = (out_dim as u32).div_ceil(GEMV_WARPS);
            let mut b = stream.launch_builder(f);
            for a in args {
                match a {
                    KernelArg::U64(v) => b.arg(v),
                    KernelArg::I32(v) => b.arg(v),
                    KernelArg::F32(v) => b.arg(v),
                };
            }
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (blocks, pairs as u32, 1),
                    block_dim: (32, GEMV_WARPS, 1),
                    shared_mem_bytes: (in_dim * 2) as u32,
                })?;
            }
            Ok(())
        };

        // Grouped-GEMM (tensor core) vs per-pair GEMV dispatch. The GEMM path
        // sorts pairs by expert so each expert's weights are read once per
        // step; it needs tile-divisible dims and E within moe_align's smem.
        // LUMINAL_MOE_GEMV=1 forces the old path (kill switch / decode A-B);
        // LUMINAL_MOE_GEMM=1 forces the new one regardless of batch size.
        let gemm_ok = hidden % GEMM_BN == 0
            && intermediate % GEMM_BN == 0
            && num_experts <= ALIGN_MAX_E;
        let min_pairs = std::env::var("LUMINAL_MOE_GEMM_MIN_PAIRS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64);
        let use_gemm = gemm_ok
            && std::env::var_os("LUMINAL_MOE_GEMV").is_none()
            && (num_pairs >= min_pairs || std::env::var_os("LUMINAL_MOE_GEMM").is_some());

        // GEMM-path scratch: pair ids sorted by expert (padded to GEMM_BM per
        // expert), the expert of each m-block, and the padded total. The GEMM
        // grid is overprovisioned to the worst-case block count and early-exits
        // on the device-side total, so no DtoH sync is needed.
        let max_padded = (num_pairs + num_experts * (GEMM_BM - 1)).min(num_pairs * GEMM_BM);
        let max_m_blocks = max_padded.div_ceil(GEMM_BM);
        let (sorted_ids_ptr, expert_ids_ptr, num_post_pad_ptr, _gemm_scratch) = if use_gemm {
            let sorted_ids = unsafe { stream.alloc::<u8>(max_padded * 4)? };
            let expert_ids = unsafe { stream.alloc::<u8>(max_m_blocks * 4)? };
            let num_post_pad = unsafe { stream.alloc::<u8>(4)? };
            let ptrs = (
                slice_ptr(&sorted_ids, stream),
                slice_ptr(&expert_ids, stream),
                slice_ptr(&num_post_pad, stream),
            );
            (ptrs.0, ptrs.1, ptrs.2, Some((sorted_ids, expert_ids, num_post_pad)))
        } else {
            (0, 0, 0, None)
        };

        // GEMV-path-only intermediates (the fused GEMM path writes hid and the
        // final output directly from the GEMM epilogues).
        let (gu_out_ptr, dn_out_ptr, _gemv_scratch) = if !use_gemm {
            let gu_out = unsafe { stream.alloc::<u8>(num_pairs * gate_up_dim * 2)? };
            let dn_out = unsafe { stream.alloc::<u8>(num_pairs * hidden * 4)? };
            let ptrs = (slice_ptr(&gu_out, stream), slice_ptr(&dn_out, stream));
            (ptrs.0, ptrs.1, Some((gu_out, dn_out)))
        } else {
            (0u64, 0u64, None)
        };

        // Grouped GEMM launch: grid (m-blocks, out_dim/BN), 128 threads.
        #[allow(clippy::too_many_arguments)]
        let launch_gemm = |f: &CudaFunction,
                           out_dim: usize,
                           in_dim: usize,
                           row_div: usize,
                           blocks_ptr: u64,
                           scales_ptr: u64,
                           blk_stride: usize,
                           sc_stride: usize,
                           a_ptr: u64,
                           c_ptr: u64,
                           bias_ptr: u64,
                           bias_stride: usize|
         -> anyhow::Result<()> {
            let mut b = stream.launch_builder(f);
            let args = [
                KernelArg::U64(blocks_ptr),
                KernelArg::U64(scales_ptr),
                KernelArg::U64(blk_stride as u64),
                KernelArg::U64(sc_stride as u64),
                KernelArg::U64(sorted_ids_ptr),
                KernelArg::U64(expert_ids_ptr),
                KernelArg::U64(num_post_pad_ptr),
                KernelArg::I32(num_pairs as i32),
                KernelArg::I32(row_div as i32),
                KernelArg::U64(a_ptr),
                KernelArg::U64(c_ptr),
                KernelArg::I32(out_dim as i32),
                KernelArg::I32(in_dim as i32),
                KernelArg::U64(bias_ptr),
                KernelArg::U64(bias_stride as u64),
                KernelArg::U64(topk_vals_base),
                KernelArg::I32(val_stride as i32),
                KernelArg::I32(top_k as i32),
            ];
            for a in &args {
                match a {
                    KernelArg::U64(v) => b.arg(v),
                    KernelArg::I32(v) => b.arg(v),
                    KernelArg::F32(v) => b.arg(v),
                };
            }
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (max_m_blocks as u32, (out_dim / GEMM_BN) as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0, // static smem only
                })?;
            }
            Ok(())
        };

        // x F32 -> BF16
        launch(
            f32_to_bf16_fn,
            seq * hidden,
            &[
                KernelArg::U64(x_ptr),
                KernelArg::U64(xbf16_ptr),
                KernelArg::I32((seq * hidden) as i32),
            ],
        )?;

        // ONE launch per kernel over ALL (token,expert) pairs — was 4*seq*top_k
        // launches/layer; profiling showed that CPU launch issue dominated a
        // batched step (75% at s=8). Routing is read on-GPU from the topk buffers.
        let num_experts_i32 = num_experts as i32;
        let top_k_i32 = top_k as i32;
        let idx_stride_i32 = idx_stride as i32;
        let val_stride_i32 = val_stride as i32;

        if use_gemm {
            // Sort pair ids by expert (single block; pairs and E are small).
            let mut b = stream.launch_builder(&kernels.align);
            let args = [
                KernelArg::U64(topk_idx_base),
                KernelArg::I32(top_k_i32),
                KernelArg::I32(idx_stride_i32),
                KernelArg::I32(num_experts_i32),
                KernelArg::I32(num_pairs as i32),
                KernelArg::I32(GEMM_BM as i32),
                KernelArg::U64(sorted_ids_ptr),
                KernelArg::U64(expert_ids_ptr),
                KernelArg::U64(num_post_pad_ptr),
            ];
            for a in &args {
                match a {
                    KernelArg::U64(v) => b.arg(v),
                    KernelArg::I32(v) => b.arg(v),
                    KernelArg::F32(v) => b.arg(v),
                };
            }
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            }
            // gate_up grouped GEMM with FUSED bias + clamped SwiGLU epilogue:
            // writes activated hid [num_pairs, intermediate] bf16 directly
            // (no full-width gu_out, no separate glu launch).
            launch_gemm(
                &kernels.gemm_gu,
                gate_up_dim,
                hidden,
                top_k,
                gu_blocks_ptr,
                gu_scales_ptr,
                gu_blocks_stride,
                gu_scales_stride,
                xbf16_ptr,
                hid_ptr,
                gu_bias_ptr,
                gu_bias_stride,
            )?;
        } else {
            // gate_up GEMV over all pairs -> gu_out [num_pairs, gate_up_dim] (bf16)
            launch_gemv(
                gemv_fn,
                gate_up_dim,
                hidden,
                num_pairs,
                &[
                    KernelArg::U64(gu_blocks_ptr),
                    KernelArg::U64(gu_scales_ptr),
                    KernelArg::U64(gu_blocks_stride as u64),
                    KernelArg::U64(gu_scales_stride as u64),
                    KernelArg::U64(topk_idx_base),
                    KernelArg::I32(top_k_i32),
                    KernelArg::I32(idx_stride_i32),
                    KernelArg::I32(num_experts_i32),
                    KernelArg::U64(xbf16_ptr),
                    KernelArg::U64(gu_out_ptr),
                    KernelArg::I32(gate_up_dim as i32),
                    KernelArg::I32(hidden as i32),
                ],
            )?;
        }

        if use_gemm {
            // Zero the final output, then down grouped GEMM with FUSED
            // per-expert bias + top-k routing weight, atomicAdd-scattering
            // straight into output [seq, hidden] f32 (no dn_out, no reduce).
            launch(
                &kernels.zero_f32,
                seq * hidden,
                &[
                    KernelArg::U64(output_ptr),
                    KernelArg::I32((seq * hidden) as i32),
                ],
            )?;
            launch_gemm(
                &kernels.gemm_dn,
                hidden,
                intermediate,
                1,
                dn_blocks_ptr,
                dn_scales_ptr,
                dn_blocks_stride,
                dn_scales_stride,
                hid_ptr,
                output_ptr,
                dn_bias_ptr,
                dn_bias_stride,
            )?;
        } else {
            // GEMV path: separate SwiGLU, down GEMV, and deterministic reduce.
            // clamped interleaved SwiGLU + gate_up bias over all pairs -> hid
            launch(
                act_fn,
                num_pairs * intermediate,
                &[
                    KernelArg::U64(gu_out_ptr),
                    KernelArg::U64(gu_bias_ptr),
                    KernelArg::U64(gu_bias_stride as u64),
                    KernelArg::U64(topk_idx_base),
                    KernelArg::I32(top_k_i32),
                    KernelArg::I32(idx_stride_i32),
                    KernelArg::I32(num_experts_i32),
                    KernelArg::U64(hid_ptr),
                    KernelArg::I32(intermediate as i32),
                    KernelArg::I32(num_pairs as i32),
                ],
            )?;
            // down GEMV (raw W@hid) over all pairs -> dn_out [num_pairs, hidden] (f32)
            launch_gemv(
                down_fn,
                hidden,
                intermediate,
                num_pairs,
                &[
                    KernelArg::U64(dn_blocks_ptr),
                    KernelArg::U64(dn_scales_ptr),
                    KernelArg::U64(dn_blocks_stride as u64),
                    KernelArg::U64(dn_scales_stride as u64),
                    KernelArg::U64(topk_idx_base),
                    KernelArg::I32(top_k_i32),
                    KernelArg::I32(idx_stride_i32),
                    KernelArg::I32(num_experts_i32),
                    KernelArg::U64(hid_ptr),
                    KernelArg::U64(dn_out_ptr),
                    KernelArg::I32(hidden as i32),
                    KernelArg::I32(intermediate as i32),
                ],
            )?;
        }

        if !use_gemm {
        // deterministic weighted reduce over top_k experts -> output [seq, hidden]
        launch(
            reduce_fn,
            seq * hidden,
            &[
                KernelArg::U64(dn_out_ptr),
                KernelArg::U64(dn_bias_ptr),
                KernelArg::U64(dn_bias_stride as u64),
                KernelArg::U64(topk_idx_base),
                KernelArg::U64(topk_vals_base),
                KernelArg::I32(top_k_i32),
                KernelArg::I32(idx_stride_i32),
                KernelArg::I32(val_stride_i32),
                KernelArg::I32(num_experts_i32),
                KernelArg::U64(output_ptr),
                KernelArg::I32(seq as i32),
                KernelArg::I32(hidden as i32),
            ],
        )?;
        }

        // No per-layer sync: same-stream ordering + the step-end get_f32 cover
        // correctness; scratch frees stream-ordered via the async pool.
        if prof2 {
            eprintln!(
                "MOE2 seq={seq} pairs={num_pairs} path={} routing={routing_ms:.3} loop={:.3}",
                if use_gemm { "gemm" } else { "gemv" },
                lp0.elapsed().as_secs_f64() * 1e3
            );
        }

        Ok(())
    }

    fn output_size(&self) -> Expression {
        Expression::from('s') * self.hidden
    }

    fn output_bytes(&self) -> Expression {
        Expression::from('s') * self.hidden * 4
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("GLUMoEMXFP4")
    }
}

enum KernelArg {
    U64(u64),
    I32(i32),
    F32(f32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cudarc::driver::{CudaContext, DevicePtr};

    // FP4 E2M1 value table (mirrors quant.rs FP4_E2M1_LUT).
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// Build one expert's packed bytes where every weight element decodes to the
    /// FP4 value at `nibble` (both nibbles per byte), with all e8m0 scales = 1.0.
    fn packed_const(out_dim: usize, in_dim: usize, nibble: u8) -> (Vec<u8>, Vec<u8>) {
        let byte = (nibble << 4) | nibble;
        let blocks = vec![byte; out_dim * (in_dim / 2)];
        let scales = vec![127u8; out_dim * (in_dim / MXFP4_BLOCK)]; // 2^(127-127)=1.0
        (blocks, scales)
    }

    #[test]
    fn glumoe_mxfp4_matches_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let stream = ctx.default_stream();

        let (hidden, intermediate, e, top_k, seq) = (64usize, 64usize, 2usize, 1usize, 1usize);
        let gate_up_dim = 2 * intermediate;

        // Weights: gate_up = 1.0 everywhere (nibble 2), down = 0.5 (nibble 1).
        let mut gu_blocks = Vec::new();
        let mut gu_scales = Vec::new();
        for _ in 0..e {
            let (b, s) = packed_const(gate_up_dim, hidden, 2);
            gu_blocks.extend(b);
            gu_scales.extend(s);
        }
        let mut dn_blocks = Vec::new();
        let mut dn_scales = Vec::new();
        for _ in 0..e {
            let (b, s) = packed_const(hidden, intermediate, 1);
            dn_blocks.extend(b);
            dn_scales.extend(s);
        }
        let gu_bias = vec![0u8; e * gate_up_dim * 2]; // bf16 zeros
        let dn_bias = vec![0u8; e * hidden * 2];

        let x_f32 = vec![0.1f32; seq * hidden];
        let topk_idx = vec![0i32; seq * top_k]; // expert 0
        let topk_val = vec![1.0f32; seq * top_k]; // weight 1.0
        let out_init = vec![0u8; seq * hidden * 4];

        let to_bytes_f32 = |v: &[f32]| bytemuck::cast_slice::<f32, u8>(v).to_vec();
        let to_bytes_i32 = |v: &[i32]| bytemuck::cast_slice::<i32, u8>(v).to_vec();

        // Upload (keep slices alive for the duration of execute).
        let d_x = stream.memcpy_stod(&to_bytes_f32(&x_f32)).unwrap();
        let d_idx = stream.memcpy_stod(&to_bytes_i32(&topk_idx)).unwrap();
        let d_val = stream.memcpy_stod(&to_bytes_f32(&topk_val)).unwrap();
        let d_gub = stream.memcpy_stod(&gu_blocks).unwrap();
        let d_gus = stream.memcpy_stod(&gu_scales).unwrap();
        let d_gubias = stream.memcpy_stod(&gu_bias).unwrap();
        let d_dnb = stream.memcpy_stod(&dn_blocks).unwrap();
        let d_dns = stream.memcpy_stod(&dn_scales).unwrap();
        let d_dnbias = stream.memcpy_stod(&dn_bias).unwrap();
        let d_out = stream.memcpy_stod(&out_init).unwrap();

        let pdb = |s: &crate::cudarc::driver::CudaSlice<u8>| DeviceBuffer::new(s.device_ptr(&stream).0, s.len());
        let mut buffers: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        let nodes: Vec<NodeIndex> = (0..10).map(NodeIndex::new).collect();
        for (n, b) in nodes.iter().zip([
            pdb(&d_x), pdb(&d_idx), pdb(&d_val), pdb(&d_gub), pdb(&d_gus),
            pdb(&d_gubias), pdb(&d_dnb), pdb(&d_dns), pdb(&d_dnbias), pdb(&d_out),
        ]) {
            buffers.insert(*n, b);
        }
        let inputs: Vec<NodeIndex> = nodes[..9].to_vec();
        let self_node = nodes[9];

        let mut dyn_map = FxHashMap::default();
        dyn_map.insert('s', seq);

        let op = GLUMoEMXFP4::new(hidden.into(), intermediate.into(), top_k.into());
        op.execute(&stream, self_node, &inputs, &buffers, &dyn_map)
            .expect("execute failed");

        let out_bytes = stream.memcpy_dtov(&d_out).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&out_bytes);

        // CPU reference (f32).
        let s: f32 = x_f32.iter().take(hidden).sum(); // gate_up_out[j] = sum(x)*1.0
        let gate = s.min(SWIGLU_LIMIT);
        let up = s.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let glu = gate / (1.0 + (-SWIGLU_ALPHA * gate).exp());
        let h_val = (up + 1.0) * glu; // hidden[i]
        let down_out = (intermediate as f32) * h_val * 0.5; // down_w = 0.5
        let expected = 1.0 * down_out; // weight 1.0
        assert_eq!(LUT[2], 1.0);

        let max_rel = out
            .iter()
            .map(|&o| ((o - expected).abs() / expected.abs()).abs())
            .fold(0.0f32, f32::max);
        eprintln!("expected={expected:.4} got[0]={:.4} max_rel={max_rel:.4}", out[0]);
        assert!(
            max_rel < 0.08,
            "GLUMoEMXFP4 output deviates: expected {expected:.4}, got {:?}, max_rel {max_rel:.4}",
            &out[..hidden.min(4)]
        );
    }

    // Discriminating case the constant-everything test can't catch: distinct
    // even/odd nibbles per byte (low nibble -> even col, high -> odd) AND a
    // non-127 e8m0 scale AND parity-dependent x — so a hi/lo nibble swap or a
    // wrong scale index changes the result.
    #[test]
    fn glumoe_mxfp4_mixed_nibbles_and_scale() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let stream = ctx.default_stream();

        let (hidden, intermediate, top_k, seq) = (64usize, 64usize, 1usize, 1usize);
        let gate_up_dim = 2 * intermediate;

        // gate_up: byte 0x21 (low nibble 1 -> even col -> LUT[1]=0.5;
        // high nibble 2 -> odd col -> LUT[2]=1.0), e8m0 scale byte 126 -> 2^-1.
        let gu_blocks = vec![0x21u8; gate_up_dim * (hidden / 2)];
        let gu_scales = vec![126u8; gate_up_dim * (hidden / MXFP4_BLOCK)];
        // down: 0.5 everywhere (nibble 1), scale 1.0.
        let (dn_blocks, dn_scales) = packed_const(hidden, intermediate, 1);
        let gu_bias = vec![0u8; gate_up_dim * 2];
        let dn_bias = vec![0u8; hidden * 2];

        // Parity-dependent x: even cols 0.2, odd cols 0.1.
        let x_f32: Vec<f32> = (0..seq * hidden)
            .map(|c| if c % 2 == 0 { 0.2 } else { 0.1 })
            .collect();
        let topk_idx = vec![0i32; seq * top_k];
        let topk_val = vec![1.0f32; seq * top_k];
        let out_init = vec![0u8; seq * hidden * 4];

        let to_bytes_f32 = |v: &[f32]| bytemuck::cast_slice::<f32, u8>(v).to_vec();
        let to_bytes_i32 = |v: &[i32]| bytemuck::cast_slice::<i32, u8>(v).to_vec();

        let d_x = stream.memcpy_stod(&to_bytes_f32(&x_f32)).unwrap();
        let d_idx = stream.memcpy_stod(&to_bytes_i32(&topk_idx)).unwrap();
        let d_val = stream.memcpy_stod(&to_bytes_f32(&topk_val)).unwrap();
        let d_gub = stream.memcpy_stod(&gu_blocks).unwrap();
        let d_gus = stream.memcpy_stod(&gu_scales).unwrap();
        let d_gubias = stream.memcpy_stod(&gu_bias).unwrap();
        let d_dnb = stream.memcpy_stod(&dn_blocks).unwrap();
        let d_dns = stream.memcpy_stod(&dn_scales).unwrap();
        let d_dnbias = stream.memcpy_stod(&dn_bias).unwrap();
        let d_out = stream.memcpy_stod(&out_init).unwrap();

        let pdb = |s: &crate::cudarc::driver::CudaSlice<u8>| {
            DeviceBuffer::new(s.device_ptr(&stream).0, s.len())
        };
        let mut buffers: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        let nodes: Vec<NodeIndex> = (0..10).map(NodeIndex::new).collect();
        for (n, b) in nodes.iter().zip([
            pdb(&d_x), pdb(&d_idx), pdb(&d_val), pdb(&d_gub), pdb(&d_gus),
            pdb(&d_gubias), pdb(&d_dnb), pdb(&d_dns), pdb(&d_dnbias), pdb(&d_out),
        ]) {
            buffers.insert(*n, b);
        }
        let inputs: Vec<NodeIndex> = nodes[..9].to_vec();
        let self_node = nodes[9];
        let mut dyn_map = FxHashMap::default();
        dyn_map.insert('s', seq);

        let op = GLUMoEMXFP4::new(hidden.into(), intermediate.into(), top_k.into());
        op.execute(&stream, self_node, &inputs, &buffers, &dyn_map)
            .expect("execute failed");
        let out_bytes = stream.memcpy_dtov(&d_out).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&out_bytes);

        // CPU reference. gate_up_out = Σ_c W[o,c]·x[c], W even col = LUT[1]·0.5,
        // odd col = LUT[2]·0.5; x even 0.2, odd 0.1.
        let mut gu = 0f32;
        for c in 0..hidden {
            let w = if c % 2 == 0 { LUT[1] * 0.5 } else { LUT[2] * 0.5 };
            let xc = if c % 2 == 0 { 0.2f32 } else { 0.1f32 };
            gu += w * xc;
        }
        let gate = gu.min(SWIGLU_LIMIT);
        let up = gu.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let glu = gate / (1.0 + (-SWIGLU_ALPHA * gate).exp());
        let h_val = (up + 1.0) * glu;
        let expected = (intermediate as f32) * h_val * 0.5; // down_w 0.5, weight 1.0

        let max_rel = out
            .iter()
            .map(|&o| ((o - expected).abs() / expected.abs()).abs())
            .fold(0.0f32, f32::max);
        eprintln!("mixed: expected={expected:.4} got[0]={:.4} max_rel={max_rel:.4}", out[0]);
        assert!(
            max_rel < 0.08,
            "GLUMoEMXFP4 mixed-nibble output deviates: expected {expected:.4}, got {:?}, max_rel {max_rel:.4}",
            &out[..hidden.min(4)]
        );
    }

    // ───────────────── grouped-GEMM path tests ─────────────────

    /// Serializes LUMINAL_MOE_GEMV / LUMINAL_MOE_GEMM mutation across tests
    /// (cargo test runs tests in parallel threads of one process).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[derive(Clone, Copy)]
    enum Path {
        ForceGemv,
        ForceGemm,
        Default,
    }

    fn bf16_round(v: f32) -> f32 {
        // Round-to-nearest-even to bf16, back to f32 (mirrors __float2bfloat16).
        let bits = v.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        f32::from_bits(rounded & 0xFFFF_0000)
    }

    fn bf16_bytes(v: f32) -> [u8; 2] {
        let bits = bf16_round(v).to_bits();
        [(bits >> 16) as u8, (bits >> 24) as u8]
    }

    /// Tiny deterministic LCG (no rand dep in tests).
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// A full randomized MoE problem with a CPU f32 reference.
    struct MoeProblem {
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        top_k: usize,
        seq: usize,
        x: Vec<f32>,
        topk_idx: Vec<i32>,
        topk_val: Vec<f32>,
        gu_blocks: Vec<u8>,
        gu_scales: Vec<u8>,
        gu_bias: Vec<f32>,
        dn_blocks: Vec<u8>,
        dn_scales: Vec<u8>,
        dn_bias: Vec<f32>,
    }

    impl MoeProblem {
        fn random(seed: u32, hidden: usize, intermediate: usize, e: usize, top_k: usize, seq: usize) -> Self {
            let mut r = Lcg(seed);
            let gate_up_dim = 2 * intermediate;
            // bf16-exact x values in [-0.78, 0.78].
            let x = (0..seq * hidden)
                .map(|_| (r.below(201) as f32 - 100.0) / 128.0)
                .collect();
            // Distinct experts per token.
            assert!(top_k <= e, "MoeProblem::random requires top_k <= num_experts");
            let mut topk_idx = Vec::with_capacity(seq * top_k);
            for _ in 0..seq {
                let mut chosen: Vec<i32> = Vec::new();
                while chosen.len() < top_k {
                    let c = r.below(e as u32) as i32;
                    if !chosen.contains(&c) {
                        chosen.push(c);
                    }
                }
                topk_idx.extend(chosen);
            }
            let topk_val = (0..seq * top_k)
                .map(|_| bf16_round((r.below(100) as f32 + 1.0) / 64.0))
                .collect();
            let rand_packed = |r: &mut Lcg, out: usize, inp: usize| -> (Vec<u8>, Vec<u8>) {
                let blocks = (0..out * inp / 2).map(|_| r.below(256) as u8).collect();
                // e8m0 near 1.0: 2^-3 .. 2^3 keeps magnitudes tame.
                let scales = (0..out * inp / MXFP4_BLOCK)
                    .map(|_| (124 + r.below(7)) as u8)
                    .collect();
                (blocks, scales)
            };
            let mut gu_blocks = Vec::new();
            let mut gu_scales = Vec::new();
            let mut dn_blocks = Vec::new();
            let mut dn_scales = Vec::new();
            for _ in 0..e {
                let (b, s) = rand_packed(&mut r, gate_up_dim, hidden);
                gu_blocks.extend(b);
                gu_scales.extend(s);
                let (b, s) = rand_packed(&mut r, hidden, intermediate);
                dn_blocks.extend(b);
                dn_scales.extend(s);
            }
            let gu_bias = (0..e * gate_up_dim)
                .map(|_| bf16_round((r.below(65) as f32 - 32.0) / 64.0))
                .collect();
            let dn_bias = (0..e * hidden)
                .map(|_| bf16_round((r.below(65) as f32 - 32.0) / 64.0))
                .collect();
            Self {
                hidden,
                intermediate,
                num_experts: e,
                top_k,
                seq,
                x,
                topk_idx,
                topk_val,
                gu_blocks,
                gu_scales,
                gu_bias,
                dn_blocks,
                dn_scales,
                dn_bias,
            }
        }

        /// Dequantized weight element [o, c] of expert e (bf16-rounded like the kernels).
        fn weight(blocks: &[u8], scales: &[u8], e: usize, out: usize, inp: usize, o: usize, c: usize) -> f32 {
            let byte = blocks[e * out * (inp / 2) + o * (inp / 2) + c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let sb = scales[e * out * (inp / MXFP4_BLOCK) + o * (inp / MXFP4_BLOCK) + c / MXFP4_BLOCK];
            bf16_round(LUT[nib as usize] * (sb as f32 - 127.0).exp2())
        }

        /// Full CPU f32 reference, rounding intermediates to bf16 where the
        /// kernels store bf16 (x, gu_out, hid).
        fn cpu_ref(&self) -> Vec<f32> {
            let (h, i, k) = (self.hidden, self.intermediate, self.top_k);
            let gate_up_dim = 2 * i;
            let mut out = vec![0.0f32; self.seq * h];
            for t in 0..self.seq {
                let x_row: Vec<f32> = (0..h).map(|c| bf16_round(self.x[t * h + c])).collect();
                for slot in 0..k {
                    let e = (self.topk_idx[t * k + slot].max(0) as usize).min(self.num_experts - 1);
                    let w = self.topk_val[t * k + slot];
                    // gate_up
                    let gu: Vec<f32> = (0..gate_up_dim)
                        .map(|o| {
                            let dot: f32 = (0..h)
                                .map(|c| Self::weight(&self.gu_blocks, &self.gu_scales, e, gate_up_dim, h, o, c) * x_row[c])
                                .sum();
                            bf16_round(dot)
                        })
                        .collect();
                    // clamped interleaved swiglu + bias
                    let hid: Vec<f32> = (0..i)
                        .map(|j| {
                            let gate = (gu[2 * j] + self.gu_bias[e * gate_up_dim + 2 * j]).min(SWIGLU_LIMIT);
                            let up = (gu[2 * j + 1] + self.gu_bias[e * gate_up_dim + 2 * j + 1])
                                .clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
                            let glu = gate / (1.0 + (-SWIGLU_ALPHA * gate).exp());
                            bf16_round((up + 1.0) * glu)
                        })
                        .collect();
                    // down + bias, weighted accumulate
                    for r in 0..h {
                        let dot: f32 = (0..i)
                            .map(|c| Self::weight(&self.dn_blocks, &self.dn_scales, e, h, i, r, c) * hid[c])
                            .sum();
                        out[t * h + r] += w * (dot + self.dn_bias[e * h + r]);
                    }
                }
            }
            out
        }

        /// Upload, run the op under the given path forcing, download output.
        fn run(&self, path: Path) -> Vec<f32> {
            let ctx = CudaContext::new(0).expect("CUDA device required");
            let stream = ctx.default_stream();
            let gate_up_dim = 2 * self.intermediate;

            let f32b = |v: &[f32]| bytemuck::cast_slice::<f32, u8>(v).to_vec();
            let i32b = |v: &[i32]| bytemuck::cast_slice::<i32, u8>(v).to_vec();
            let bf16b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| bf16_bytes(f)).collect() };

            let d_x = stream.memcpy_stod(&f32b(&self.x)).unwrap();
            let d_idx = stream.memcpy_stod(&i32b(&self.topk_idx)).unwrap();
            let d_val = stream.memcpy_stod(&f32b(&self.topk_val)).unwrap();
            let d_gub = stream.memcpy_stod(&self.gu_blocks).unwrap();
            let d_gus = stream.memcpy_stod(&self.gu_scales).unwrap();
            let d_gubias = stream.memcpy_stod(&bf16b(&self.gu_bias)).unwrap();
            let d_dnb = stream.memcpy_stod(&self.dn_blocks).unwrap();
            let d_dns = stream.memcpy_stod(&self.dn_scales).unwrap();
            let d_dnbias = stream.memcpy_stod(&bf16b(&self.dn_bias)).unwrap();
            let d_out = stream.memcpy_stod(&vec![0u8; self.seq * self.hidden * 4]).unwrap();
            let _ = gate_up_dim;

            let pdb = |s: &crate::cudarc::driver::CudaSlice<u8>| DeviceBuffer::new(s.device_ptr(&stream).0, s.len());
            let mut buffers: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
            let nodes: Vec<NodeIndex> = (0..10).map(NodeIndex::new).collect();
            for (n, b) in nodes.iter().zip([
                pdb(&d_x), pdb(&d_idx), pdb(&d_val), pdb(&d_gub), pdb(&d_gus),
                pdb(&d_gubias), pdb(&d_dnb), pdb(&d_dns), pdb(&d_dnbias), pdb(&d_out),
            ]) {
                buffers.insert(*n, b);
            }
            let inputs: Vec<NodeIndex> = nodes[..9].to_vec();
            let mut dyn_map = FxHashMap::default();
            dyn_map.insert('s', self.seq);

            let op = GLUMoEMXFP4::new(self.hidden.into(), self.intermediate.into(), self.top_k.into());

            let guard = ENV_LOCK.lock().unwrap();
            match path {
                Path::ForceGemv => unsafe { std::env::set_var("LUMINAL_MOE_GEMV", "1") },
                Path::ForceGemm => unsafe { std::env::set_var("LUMINAL_MOE_GEMM", "1") },
                Path::Default => {}
            }
            let result = op.execute(&stream, nodes[9], &inputs, &buffers, &dyn_map);
            unsafe {
                std::env::remove_var("LUMINAL_MOE_GEMV");
                std::env::remove_var("LUMINAL_MOE_GEMM");
            }
            drop(guard);
            result.expect("execute failed");

            let out_bytes = stream.memcpy_dtov(&d_out).unwrap();
            bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec()
        }
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len());
        // Magnitude-aware denominator: near-zero elements of a large-scale
        // output vector have catastrophic-cancellation noise (the fused GEMM
        // path keeps f32 accumulators through the activation while cpu_ref /
        // the GEMV path round intermediates to bf16, so tiny ABSOLUTE
        // differences at the vector's natural scale are expected and blow up
        // pure relative error at zero crossings).
        let scale = want.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let floor = (scale * 0.02).max(1e-3);
        let mut max_rel = 0.0f32;
        let mut worst = 0usize;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let rel = (g - w).abs() / w.abs().max(floor);
            if rel > max_rel {
                max_rel = rel;
                worst = i;
            }
        }
        eprintln!(
            "{label}: max_rel={max_rel:.5} scale={scale:.2} (worst idx {worst}: got {} want {})",
            got[worst], want[worst]
        );
        assert!(max_rel < tol, "{label}: max_rel {max_rel:.5} exceeds {tol}");
    }

    /// Random multi-token / multi-expert problem: GEMV, GEMM, and CPU ref all agree.
    #[test]
    fn glumoe_mxfp4_grouped_multi_expert() {
        if CudaContext::new(0).is_err() {
            eprintln!("no CUDA device; skipping");
            return;
        }
        let p = MoeProblem::random(7, 64, 64, 8, 2, 32); // pairs=64
        let want = p.cpu_ref();
        let gemv = p.run(Path::ForceGemv);
        let gemm = p.run(Path::ForceGemm);
        assert_close(&gemv, &want, 0.08, "gemv vs ref");
        assert_close(&gemm, &want, 0.08, "gemm vs ref");
        // The two paths round differently now (fused GEMM keeps f32 through
        // the activation; GEMV rounds gu to bf16 first), so clamp-boundary
        // crossings legitimately diverge at this test's adversarial weight
        // scales — same tolerance as vs-reference.
        assert_close(&gemm, &gemv, 0.08, "gemm vs gemv");
    }

    /// Routing skew: (a) every pair on one expert (multi-m-block single expert,
    /// all other experts zero-count); (b) tiny s with max padding.
    #[test]
    fn glumoe_mxfp4_routing_skew() {
        if CudaContext::new(0).is_err() {
            eprintln!("no CUDA device; skipping");
            return;
        }
        // (a) all 64 pairs -> expert 3 of 8 (4 m-blocks for one expert).
        let mut p = MoeProblem::random(11, 64, 64, 8, 2, 32);
        for v in p.topk_idx.iter_mut() {
            *v = 3;
        }
        let want = p.cpu_ref();
        assert_close(&p.run(Path::ForceGemm), &want, 0.08, "skew-single-expert gemm vs ref");

        // (b) s=1, k=2, two distinct experts: 2 pairs -> 2 blocks of 16 (max padding).
        let p2 = MoeProblem::random(13, 64, 64, 4, 2, 1);
        let want2 = p2.cpu_ref();
        assert_close(&p2.run(Path::ForceGemm), &want2, 0.08, "max-padding gemm vs ref");
    }

    /// Default dispatch straddles the pairs threshold (64): s=31,k=2 -> 62 pairs
    /// (GEMV) and s=33,k=2 -> 66 pairs (GEMM); both must match their reference.
    #[test]
    fn glumoe_mxfp4_dispatch_threshold() {
        if CudaContext::new(0).is_err() {
            eprintln!("no CUDA device; skipping");
            return;
        }
        for (seq, label) in [(31usize, "below threshold (gemv)"), (33, "above threshold (gemm)")] {
            let p = MoeProblem::random(17, 64, 64, 8, 2, seq);
            let want = p.cpu_ref();
            assert_close(&p.run(Path::Default), &want, 0.08, label);
        }
    }

    /// Launch moe_align directly and validate its invariants, including expert
    /// clamping and a topk row stride wider than top_k.
    #[test]
    fn moe_align_matches_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let stream = ctx.default_stream();
        let (seq, top_k, idx_stride, num_experts, block_m) = (13usize, 3usize, 5usize, 4usize, 16usize);
        let num_pairs = seq * top_k;

        // topk rows padded to stride 5; include out-of-range ids (clamped).
        let mut r = Lcg(23);
        let mut topk_idx = vec![0i32; seq * idx_stride];
        for t in 0..seq {
            for i in 0..top_k {
                topk_idx[t * idx_stride + i] = match r.below(10) {
                    0 => -1,   // clamps to 0
                    1 => 99,   // clamps to num_experts-1
                    v => (v % num_experts as u32) as i32,
                };
            }
            for i in top_k..idx_stride {
                topk_idx[t * idx_stride + i] = 12345; // never read
            }
        }
        let clamp = |e: i32| -> usize { (e.max(0) as usize).min(num_experts - 1) };

        let max_padded = (num_pairs + num_experts * (block_m - 1)).min(num_pairs * block_m);
        let max_m_blocks = max_padded.div_ceil(block_m);

        let d_idx = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&topk_idx)).unwrap();
        let d_sorted = stream.memcpy_stod(&vec![0xAAu8; max_padded * 4]).unwrap();
        let d_experts = stream.memcpy_stod(&vec![0xAAu8; max_m_blocks * 4]).unwrap();
        let d_total = stream.memcpy_stod(&[0u8; 4]).unwrap();

        let op = GLUMoEMXFP4::new(64.into(), 64.into(), top_k.into());
        let kernels = op.get_kernels(&stream);
        let (idx_ptr, _) = d_idx.device_ptr(&stream);
        let (sorted_ptr, _) = d_sorted.device_ptr(&stream);
        let (experts_ptr, _) = d_experts.device_ptr(&stream);
        let (total_ptr, _) = d_total.device_ptr(&stream);
        let (top_k_a, stride_a, experts_a, pairs_a, bm_a) = (
            top_k as i32,
            idx_stride as i32,
            num_experts as i32,
            num_pairs as i32,
            block_m as i32,
        );
        let mut b = stream.launch_builder(&kernels.align);
        b.arg(&idx_ptr)
            .arg(&top_k_a)
            .arg(&stride_a)
            .arg(&experts_a)
            .arg(&pairs_a)
            .arg(&bm_a)
            .arg(&sorted_ptr)
            .arg(&experts_ptr)
            .arg(&total_ptr);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        stream.synchronize().unwrap();

        let total = bytemuck::cast_slice::<u8, i32>(&stream.memcpy_dtov(&d_total).unwrap())[0] as usize;
        let sorted = bytemuck::cast_slice::<u8, i32>(&stream.memcpy_dtov(&d_sorted).unwrap()).to_vec();
        let experts = bytemuck::cast_slice::<u8, i32>(&stream.memcpy_dtov(&d_experts).unwrap()).to_vec();

        assert!(total % block_m == 0 && total <= max_padded, "bad total {total}");
        // Every pair id appears exactly once among non-sentinel slots.
        let mut seen = vec![0u8; num_pairs];
        for (i, &id) in sorted[..total].iter().enumerate() {
            if id == num_pairs as i32 {
                continue; // padding sentinel
            }
            let id = id as usize;
            assert!(id < num_pairs, "sorted id out of range at {i}");
            seen[id] += 1;
            // Block expert consistency: the pair's clamped expert == block's expert.
            let block = i / block_m;
            let t = id / top_k;
            let slot = id % top_k;
            let e = clamp(topk_idx[t * idx_stride + slot]);
            assert_eq!(e as i32, experts[block], "pair {id} in block {block} expert mismatch");
        }
        assert!(seen.iter().all(|&c| c == 1), "pair multiplicity wrong: {seen:?}");
    }

    /// Ground truth at the kernel boundary: launch the fused gate_up GEMM
    /// directly and diff `hid` element-wise vs CPU.
    #[test]
    fn fused_gu_hid_direct() {
        // hidden=64: single K-iter; hidden=192: 3 K-iters — exercises the
        // cp.async double-buffered pipeline.
        for hidden in [64usize, 192] {
            fused_gu_hid_direct_at(hidden);
        }
    }

    fn fused_gu_hid_direct_at(hidden: usize) {
        let Ok(ctx) = CudaContext::new(0) else { return; };
        let stream = ctx.default_stream();
        let (inter, e_cnt, top_k, seq) = (64usize, 1usize, 1usize, 16usize);
        let p = MoeProblem::random(3, hidden, inter, e_cnt, top_k, seq);
        let gate_up_dim = 2 * inter;
        let num_pairs = seq * top_k;

        // Device buffers
        let to_bf16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| bf16_bytes(f)).collect() };
        let x_bf16: Vec<u8> = to_bf16(&p.x);
        let d_x = stream.memcpy_stod(&x_bf16).unwrap();
        let d_blk = stream.memcpy_stod(&p.gu_blocks).unwrap();
        let d_sc = stream.memcpy_stod(&p.gu_scales).unwrap();
        let d_bias = stream.memcpy_stod(&to_bf16(&p.gu_bias)).unwrap();
        let d_vals = stream.memcpy_stod(bytemuck::cast_slice::<f32, u8>(&p.topk_val)).unwrap();
        // sorted ids: identity (k=1, E=1 → all pairs expert 0, one m-block)
        let sorted: Vec<i32> = (0..num_pairs as i32).collect();
        let d_sorted = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&sorted)).unwrap();
        let d_experts = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&[0i32])).unwrap();
        let d_total = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&[num_pairs as i32])).unwrap();
        let d_hid = stream.memcpy_stod(&vec![0u8; num_pairs * inter * 2]).unwrap();

        let op = GLUMoEMXFP4::new(hidden.into(), inter.into(), top_k.into());
        let kernels = op.get_kernels(&stream);
        use crate::cudarc::driver::DevicePtr;
        let ptr = |b: &crate::cudarc::driver::CudaSlice<u8>| b.device_ptr(&stream).0;
        let args_u64 = [
            ptr(&d_blk), ptr(&d_sc),
            (gate_up_dim * (hidden / 2)) as u64, (gate_up_dim * (hidden / 32)) as u64,
            ptr(&d_sorted), ptr(&d_experts), ptr(&d_total),
        ];
        let (np_i, rd_i) = (num_pairs as i32, top_k as i32);
        let (a_p, c_p) = (ptr(&d_x), ptr(&d_hid));
        let (od_i, id_i) = (gate_up_dim as i32, hidden as i32);
        let (bias_p, bias_st) = (ptr(&d_bias), (gate_up_dim * 2) as u64);
        let (vals_p, vs_i, tk_i) = (ptr(&d_vals), top_k as i32, top_k as i32);
        let mut b = stream.launch_builder(&kernels.gemm_gu);
        b.arg(&args_u64[0]).arg(&args_u64[1]).arg(&args_u64[2]).arg(&args_u64[3])
            .arg(&args_u64[4]).arg(&args_u64[5]).arg(&args_u64[6])
            .arg(&np_i).arg(&rd_i).arg(&a_p).arg(&c_p).arg(&od_i).arg(&id_i)
            .arg(&bias_p).arg(&bias_st).arg(&vals_p).arg(&vs_i).arg(&tk_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, (gate_up_dim / GEMM_BN) as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            }).unwrap();
        }
        stream.synchronize().unwrap();
        let hid_bytes = stream.memcpy_dtov(&d_hid).unwrap();
        let hid_gpu: Vec<f32> = hid_bytes.chunks_exact(2)
            .map(|c| bf16_bits_to_f32_local(u16::from_le_bytes([c[0], c[1]]))).collect();

        // CPU: per pair p (= token t, k=1), per hid col c
        let mut worst = (0.0f32, 0usize);
        for pid in 0..num_pairs {
            let t = pid;
            for c in 0..inter {
                let gate_dot: f32 = (0..hidden).map(|kk|
                    MoeProblem::weight(&p.gu_blocks, &p.gu_scales, 0, gate_up_dim, hidden, 2*c, kk)
                    * bf16_round(p.x[t*hidden+kk])).sum();
                let up_dot: f32 = (0..hidden).map(|kk|
                    MoeProblem::weight(&p.gu_blocks, &p.gu_scales, 0, gate_up_dim, hidden, 2*c+1, kk)
                    * bf16_round(p.x[t*hidden+kk])).sum();
                let gate = (gate_dot + p.gu_bias[2*c]).min(7.0);
                let up = (up_dot + p.gu_bias[2*c+1]).clamp(-7.0, 7.0);
                let want = (up + 1.0) * (gate / (1.0 + (-1.702f32*gate).exp()));
                let got = hid_gpu[pid*inter + c];
                let r = (got - want).abs() / want.abs().max(1e-3);
                if r > worst.0 { worst = (r, pid*inter + c); }
            }
        }
        eprintln!("fused gu hid: max_rel={:.5} worst idx {} got {} ", worst.0, worst.1, hid_gpu[worst.1]);
    }

    fn bf16_bits_to_f32_local(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

    /// Ground truth for the fused down epilogue: known hid in, direct launch,
    /// compare output vs CPU.
    #[test]
    fn fused_dn_direct() {
        let Ok(ctx) = CudaContext::new(0) else { return; };
        let stream = ctx.default_stream();
        let (hidden, inter, top_k, seq) = (64usize, 64usize, 1usize, 16usize);
        let p = MoeProblem::random(7, hidden, inter, 1, top_k, seq);
        let num_pairs = seq * top_k;

        // Known hid: bf16-exact smallish values.
        let mut r = Lcg(99);
        let hid: Vec<f32> = (0..num_pairs * inter)
            .map(|_| bf16_round((r.below(101) as f32 - 50.0) / 64.0))
            .collect();
        let to_bf16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| bf16_bytes(f)).collect() };

        let d_hid = stream.memcpy_stod(&to_bf16(&hid)).unwrap();
        let d_blk = stream.memcpy_stod(&p.dn_blocks).unwrap();
        let d_sc = stream.memcpy_stod(&p.dn_scales).unwrap();
        let d_bias = stream.memcpy_stod(&to_bf16(&p.dn_bias)).unwrap();
        let d_vals = stream.memcpy_stod(bytemuck::cast_slice::<f32, u8>(&p.topk_val)).unwrap();
        let sorted: Vec<i32> = (0..num_pairs as i32).collect();
        let d_sorted = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&sorted)).unwrap();
        let d_experts = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&[0i32])).unwrap();
        let d_total = stream.memcpy_stod(bytemuck::cast_slice::<i32, u8>(&[num_pairs as i32])).unwrap();
        let d_out = stream.memcpy_stod(&vec![0u8; seq * hidden * 4]).unwrap();

        let op = GLUMoEMXFP4::new(hidden.into(), inter.into(), top_k.into());
        let kernels = op.get_kernels(&stream);
        use crate::cudarc::driver::DevicePtr;
        let ptr = |b: &crate::cudarc::driver::CudaSlice<u8>| b.device_ptr(&stream).0;
        let (blk_st, sc_st) = ((hidden * (inter / 2)) as u64, (hidden * (inter / 32)) as u64);
        let (np_i, rd_i) = (num_pairs as i32, 1i32);
        let (a_p, c_p) = (ptr(&d_hid), ptr(&d_out));
        let (od_i, id_i) = (hidden as i32, inter as i32);
        let (bias_p, bias_st) = (ptr(&d_bias), (hidden * 2) as u64);
        let (vals_p, vs_i, tk_i) = (ptr(&d_vals), top_k as i32, top_k as i32);
        let (blk_p, sc_p, sort_p, exp_p, tot_p) =
            (ptr(&d_blk), ptr(&d_sc), ptr(&d_sorted), ptr(&d_experts), ptr(&d_total));
        let mut b = stream.launch_builder(&kernels.gemm_dn);
        b.arg(&blk_p).arg(&sc_p).arg(&blk_st).arg(&sc_st)
            .arg(&sort_p).arg(&exp_p).arg(&tot_p)
            .arg(&np_i).arg(&rd_i).arg(&a_p).arg(&c_p).arg(&od_i).arg(&id_i)
            .arg(&bias_p).arg(&bias_st).arg(&vals_p).arg(&vs_i).arg(&tk_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, (hidden / GEMM_BN) as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            }).unwrap();
        }
        stream.synchronize().unwrap();
        let out_bytes = stream.memcpy_dtov(&d_out).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&out_bytes);

        let mut worst = (0.0f32, 0usize);
        for t in 0..seq {
            for rr in 0..hidden {
                let dot: f32 = (0..inter).map(|c|
                    MoeProblem::weight(&p.dn_blocks, &p.dn_scales, 0, hidden, inter, rr, c)
                    * hid[t*inter + c]).sum();
                let want = p.topk_val[t] * (dot + p.dn_bias[rr]);
                let got = out[t*hidden + rr];
                let rel = (got - want).abs() / want.abs().max(1e-3);
                if rel > worst.0 { worst = (rel, t*hidden + rr); }
            }
        }
        eprintln!("fused dn: max_rel={:.5} worst idx {} got {}", worst.0, worst.1, out[worst.1]);
    }
}
