#include <cub/block/block_scan.cuh>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32
#define ALIGN_THREADS 1024

// Standard-path align kernel. Launched with <<<2, ALIGN_THREADS, smem>>>:
// blockIdx.x == 1 only initializes sorted_token_ids with the sentinel (numel)
// and exits; blockIdx.x == 0 histograms, scans, and fills expert_ids. This is
// safe because this kernel never reads sorted_token_ids (upstream trick).
// topk_ids reads are strided: pair i lives at row i/top_k, column i%top_k of
// a row-major buffer whose row stride (idx_row_stride >= top_k) may exceed
// top_k — e.g. when the ids are the first top_k columns of a full [s, E]
// argsort tensor. Contiguous [s, top_k] callers pass idx_row_stride == top_k.
extern "C" __global__ void moe_align_block_size_kernel(
    unsigned long long topk_ids_ptr,          // const int* [numel strided rows]
    unsigned long long sorted_token_ids_ptr,  // int* [max_num_tokens_padded]
    unsigned long long expert_ids_ptr,        // int* [max_num_m_blocks]
    unsigned long long total_tokens_post_pad_ptr,  // int* [1]
    int num_experts,
    int padded_num_experts,
    int experts_per_warp,
    int block_size,
    long long numel,
    unsigned long long cumsum_ptr,            // int* [num_experts + 1]
    int max_num_tokens_padded,
    int max_num_m_blocks,
    int top_k,
    int idx_row_stride
) {
    const int* topk_ids = (const int*)topk_ids_ptr;
    int* sorted_token_ids = (int*)sorted_token_ids_ptr;
    int* expert_ids = (int*)expert_ids_ptr;
    int* total_tokens_post_pad = (int*)total_tokens_post_pad_ptr;
    int* cumsum = (int*)cumsum_ptr;

    extern __shared__ int shared_counts[];

    if (blockIdx.x % 2) {
        for (long long it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
            sorted_token_ids[it] = (int)numel;
        }
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int my_expert_start = warp_id * experts_per_warp;
    for (int i = 0; i < experts_per_warp; ++i) {
        if (my_expert_start + i < padded_num_experts) {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }
    __syncthreads();

    // Histogram over the flattened pairs (atomicAdd into warp-partitioned
    // shared counts, as upstream).
    for (long long i = threadIdx.x; i < numel; i += blockDim.x) {
        int expert_id = topk_ids[(i / top_k) * idx_row_stride + (i % top_k)];
        if (expert_id >= num_experts) continue;
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
    __syncthreads();

    // Exclusive scan over block-padded per-expert counts (as upstream).
    using BlockScan = cub::BlockScan<int, ALIGN_THREADS>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int expert_id = threadIdx.x;
    if (expert_id < num_experts) {
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);

    if (expert_id <= num_experts) {
        cumsum[expert_id] = cumsum_val;
    }
    if (expert_id == num_experts) {
        total_tokens_post_pad[0] = cumsum_val;
    }
    __syncthreads();

    if (threadIdx.x < num_experts) {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // Fill remaining expert_ids with -1 (inactive).
    const long long fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    for (long long i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
        expert_ids[i] = -1;
    }
}

// Placement: each pair claims its slot with an atomicAdd on its expert's
// cumsum cell and writes its FLATTENED PAIR INDEX. Within-expert order is
// therefore unspecified. Grid is (1, blocks_y); tid/stride as upstream.
extern "C" __global__ void count_and_sort_expert_tokens_kernel(
    unsigned long long topk_ids_ptr,          // const int* [numel strided rows]
    unsigned long long sorted_token_ids_ptr,  // int* [max_num_tokens_padded]
    unsigned long long cumsum_buffer_ptr,     // int* [num_experts + 1] (mutated!)
    long long numel,
    int num_experts,
    int top_k,
    int idx_row_stride
) {
    const int* topk_ids = (const int*)topk_ids_ptr;
    int* sorted_token_ids = (int*)sorted_token_ids_ptr;
    int* cumsum_buffer = (int*)cumsum_buffer_ptr;

    const long long tid = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    const long long stride = (long long)blockDim.x * gridDim.y;

    for (long long i = tid; i < numel; i += stride) {
        int expert_id = topk_ids[(i / top_k) * idx_row_stride + (i % top_k)];
        if (expert_id >= num_experts) continue;
        int rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = (int)i;
    }
}
