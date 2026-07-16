//! `moe_align_block_size`: expert-grouped, block-padded (token, expert) pair
//! ordering for grouped MoE GEMMs — a standalone CUDA building block, not
//! wired into any graph op.
//!
//! Adapted from vLLM `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`
//! (vllm-project/vllm @ fbc9ba6d303b97d38a3f8b420a90de161b709aff), standard
//! two-kernel path only. Deviations from the source, all deliberate:
//!   - The LoRA / `token_mask` / `model_offset` / `expert_map` machinery is
//!     dropped — the upstream standard `__global__` wrappers already pass
//!     null/0 for all of it; the `expert_id >= num_experts` skip guard stays.
//!   - Pointers cross the ABI as `unsigned long long` (house style).
//!
//! The `cub::BlockScan` cumsum is kept verbatim from upstream — CUB compiles
//! fine under this crate's NVRTC path (probed on CUDA 12.8 with the distro
//! CUB in /usr/include).
//!
//! Output contract (identical to vLLM):
//!   - `sorted_token_ids[max_num_tokens_padded]`: flattened PAIR indices
//!     (token = pair / top_k), grouped by expert, each expert's run padded to
//!     `block_size` with the sentinel `numel`. Order WITHIN an expert's run
//!     is unspecified (atomic placement).
//!   - `expert_ids[max_num_m_blocks]`: owning expert per block; `-1` past the
//!     padded total.
//!   - `num_tokens_post_pad[1]`: Σ_e ceil(count_e / block_size) · block_size.
//!   - capacity: `max_num_tokens_padded = numel + num_experts*(block_size-1)`.

use std::sync::{Arc, OnceLock};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{
        CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    },
};

const WARP_SIZE: usize = 32;
/// One thread per (padded) expert in the scan phase; upstream requires
/// `padded_num_experts < 1024` for the same reason.
const ALIGN_THREADS: usize = 1024;

const MOE_ALIGN_SRC: &str = include_str!("align.cu");

struct MoeAlignKernels {
    _module: Arc<CudaModule>,
    align: CudaFunction,
    count_sort: CudaFunction,
}

/// Process-wide kernel cache: compiled once per process, shared by every
/// caller. (A per-instance OnceLock here would recompile ~seconds of NVRTC
/// per clone — the GLUMoE anti-pattern.)
static KERNELS: OnceLock<MoeAlignKernels> = OnceLock::new();

/// Force the process-wide NVRTC compile of this module (idempotent). Called
/// from host-op prewarm so the compile cost lands in the untimed prebuild
/// phase, never inside a profiling trial or first real execution.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernels(stream);
}

fn kernels(stream: &Arc<CudaStream>) -> &'static MoeAlignKernels {
    KERNELS.get_or_init(|| {
        let ptx = compile_module_image_for_current_device(stream.context(), MOE_ALIGN_SRC)
            .expect("moe_align NVRTC compile failed");
        let module = stream
            .context()
            .load_module(ptx)
            .expect("moe_align module load failed");
        let align = module.load_function("moe_align_block_size_kernel").unwrap();
        let count_sort = module
            .load_function("count_and_sort_expert_tokens_kernel")
            .unwrap();
        MoeAlignKernels {
            _module: module,
            align,
            count_sort,
        }
    })
}

/// Device buffers implementing the vLLM output-capacity contract.
pub struct MoeAlignBuffers {
    pub sorted_token_ids: CudaSlice<u8>, // int32 [max_num_tokens_padded]
    pub expert_ids: CudaSlice<u8>,       // int32 [max_num_m_blocks]
    pub num_tokens_post_pad: CudaSlice<u8>, // int32 [1]
    pub cumsum: CudaSlice<u8>,           // int32 [num_experts + 1]
    pub max_num_tokens_padded: usize,
    pub max_num_m_blocks: usize,
}

impl MoeAlignBuffers {
    pub fn alloc(
        stream: &Arc<CudaStream>,
        num_pairs: usize,
        num_experts: usize,
        block_size: usize,
    ) -> anyhow::Result<Self> {
        let max_num_tokens_padded = num_pairs + num_experts * (block_size - 1);
        let max_num_m_blocks = max_num_tokens_padded.div_ceil(block_size);
        Ok(Self {
            sorted_token_ids: unsafe { stream.alloc::<u8>(max_num_tokens_padded * 4)? },
            expert_ids: unsafe { stream.alloc::<u8>(max_num_m_blocks * 4)? },
            num_tokens_post_pad: unsafe { stream.alloc::<u8>(4)? },
            cumsum: unsafe { stream.alloc::<u8>((num_experts + 1) * 4)? },
            max_num_tokens_padded,
            max_num_m_blocks,
        })
    }
}

/// Launch the standard two-kernel `moe_align_block_size` path (upstream host
/// dispatch geometry: align `<<<2, 1024, smem>>>`, placement grid `(1, y)` of
/// 256-thread blocks).
///
/// `topk_ids_ptr` is a device pointer to `num_pairs` contiguous int32 expert
/// ids (row-major `[tokens, top_k]` flattened). Requires
/// `padded_num_experts < 1024` (one scan lane per expert), as upstream.
/// `topk_ids` is read with a row stride: pair `i` = row `i/top_k`, column
/// `i%top_k` of a row-major Int buffer with `idx_row_stride >= top_k` columns
/// (contiguous callers pass `idx_row_stride == top_k`; graph callers bind the
/// full argsort tensor and derive the stride from the buffer length).
#[allow(clippy::too_many_arguments)]
pub fn moe_align_block_size(
    stream: &Arc<CudaStream>,
    topk_ids_ptr: u64,
    num_pairs: usize,
    top_k: usize,
    idx_row_stride: usize,
    num_experts: usize,
    block_size: usize,
    out: &MoeAlignBuffers,
) -> anyhow::Result<()> {
    let padded_num_experts = num_experts.div_ceil(WARP_SIZE) * WARP_SIZE;
    anyhow::ensure!(
        padded_num_experts < ALIGN_THREADS,
        "moe_align_block_size requires padded_num_experts < {ALIGN_THREADS}, got {padded_num_experts}"
    );
    anyhow::ensure!(block_size > 0, "block_size must be > 0");
    anyhow::ensure!(
        top_k > 0 && num_pairs % top_k == 0,
        "num_pairs must be a multiple of top_k"
    );
    anyhow::ensure!(idx_row_stride >= top_k, "idx_row_stride must be >= top_k");

    let k = kernels(stream);
    let ptr = |b: &CudaSlice<u8>| b.device_ptr(stream).0;

    let num_warps = padded_num_experts.div_ceil(WARP_SIZE);
    let shared_mem_bytes = (num_warps * WARP_SIZE * 4) as u32;

    let sorted_ptr = ptr(&out.sorted_token_ids);
    let expert_ids_ptr = ptr(&out.expert_ids);
    let total_ptr = ptr(&out.num_tokens_post_pad);
    let cumsum_ptr = ptr(&out.cumsum);
    let (ne, pne, epw, bs) = (
        num_experts as i32,
        padded_num_experts as i32,
        WARP_SIZE as i32,
        block_size as i32,
    );
    let numel = num_pairs as i64;
    let (tk_i, stride_i) = (top_k as i32, idx_row_stride as i32);
    let (mntp, mnmb) = (
        out.max_num_tokens_padded as i32,
        out.max_num_m_blocks as i32,
    );

    unsafe {
        stream
            .launch_builder(&k.align)
            .arg(&topk_ids_ptr)
            .arg(&sorted_ptr)
            .arg(&expert_ids_ptr)
            .arg(&total_ptr)
            .arg(&ne)
            .arg(&pne)
            .arg(&epw)
            .arg(&bs)
            .arg(&numel)
            .arg(&cumsum_ptr)
            .arg(&mntp)
            .arg(&mnmb)
            .arg(&tk_i)
            .arg(&stride_i)
            .launch(LaunchConfig {
                grid_dim: (2, 1, 1),
                block_dim: (ALIGN_THREADS as u32, 1, 1),
                shared_mem_bytes,
            })?;
    }

    let block_threads = 256usize;
    let blocks_y = num_pairs.div_ceil(block_threads).clamp(1, 65535) as u32;
    unsafe {
        stream
            .launch_builder(&k.count_sort)
            .arg(&topk_ids_ptr)
            .arg(&sorted_ptr)
            .arg(&cumsum_ptr)
            .arg(&numel)
            .arg(&ne)
            .arg(&tk_i)
            .arg(&stride_i)
            .launch(LaunchConfig {
                grid_dim: (1, blocks_y, 1),
                block_dim: (block_threads as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cudarc::driver::CudaContext;

    /// Invariant checks over downloaded GPU outputs (O(P+E) host passes; no
    /// CPU mirror of the algorithm). Within-expert order is unspecified and
    /// deliberately NOT checked.
    fn check_invariants(
        topk_ids: &[i32],
        sorted: &[i32],
        expert_ids: &[i32],
        num_post_pad: i32,
        num_experts: usize,
        block_size: usize,
        label: &str,
    ) {
        let numel = topk_ids.len();
        // Host histogram (single pass).
        let mut counts = vec![0usize; num_experts];
        for &e in topk_ids {
            assert!(
                (0..num_experts as i32).contains(&e),
                "{label}: bad input expert {e}"
            );
            counts[e as usize] += 1;
        }
        // Expected padded total.
        let expected_total: usize = counts
            .iter()
            .map(|c| c.div_ceil(block_size) * block_size)
            .sum();
        assert_eq!(
            num_post_pad as usize, expected_total,
            "{label}: num_tokens_post_pad"
        );

        // Segment boundaries implied by the padded counts, in expert order.
        let mut boundaries = vec![0usize; num_experts + 1];
        for e in 0..num_experts {
            boundaries[e + 1] = boundaries[e] + counts[e].div_ceil(block_size) * block_size;
        }

        // expert_ids: owning expert per block within the padded region, -1 after.
        for b in 0..expert_ids.len() {
            let block_start = b * block_size;
            if block_start < expected_total {
                let owner = (0..num_experts)
                    .find(|&e| block_start >= boundaries[e] && block_start < boundaries[e + 1])
                    .unwrap() as i32;
                assert_eq!(expert_ids[b], owner, "{label}: expert_ids[{b}]");
            } else {
                assert_eq!(
                    expert_ids[b], -1,
                    "{label}: expert_ids[{b}] past padded total"
                );
            }
        }

        // Every pair index appears exactly once, inside its own expert's segment;
        // padding slots hold the sentinel (numel).
        let mut seen = vec![false; numel];
        for (slot, &v) in sorted.iter().enumerate().take(expected_total) {
            if v as usize == numel {
                continue; // padding
            }
            let pair = v as usize;
            assert!(
                pair < numel,
                "{label}: slot {slot} holds out-of-range pair {pair}"
            );
            assert!(!seen[pair], "{label}: pair {pair} appears twice");
            seen[pair] = true;
            let e = topk_ids[pair] as usize;
            assert!(
                (boundaries[e]..boundaries[e + 1]).contains(&slot),
                "{label}: pair {pair} (expert {e}) at slot {slot}, segment {}..{}",
                boundaries[e],
                boundaries[e + 1]
            );
        }
        assert!(
            seen.iter().all(|&s| s),
            "{label}: some pairs missing from sorted output"
        );
        // Per-expert real-vs-padding accounting.
        for e in 0..num_experts {
            let real = sorted[boundaries[e]..boundaries[e + 1]]
                .iter()
                .filter(|&&v| (v as usize) < numel)
                .count();
            assert_eq!(real, counts[e], "{label}: expert {e} real-slot count");
        }
        // Sentinel fill beyond the padded region (kernel initialized the whole buffer).
        for (slot, &v) in sorted.iter().enumerate().skip(expected_total) {
            assert_eq!(
                v as usize, numel,
                "{label}: slot {slot} beyond padded total not sentinel"
            );
        }
    }

    fn run_case(
        stream: &Arc<CudaStream>,
        topk_ids: &[i32],
        num_experts: usize,
        block_size: usize,
        label: &str,
    ) {
        let d_topk = stream
            .memcpy_stod(bytemuck::cast_slice::<i32, u8>(topk_ids))
            .unwrap();
        let bufs = MoeAlignBuffers::alloc(stream, topk_ids.len(), num_experts, block_size).unwrap();
        let topk_ptr = d_topk.device_ptr(stream).0;
        moe_align_block_size(
            stream,
            topk_ptr,
            topk_ids.len(),
            1,
            1,
            num_experts,
            block_size,
            &bufs,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let sorted_b: Vec<u8> = stream.memcpy_dtov(&bufs.sorted_token_ids).unwrap();
        let experts_b: Vec<u8> = stream.memcpy_dtov(&bufs.expert_ids).unwrap();
        let total_b: Vec<u8> = stream.memcpy_dtov(&bufs.num_tokens_post_pad).unwrap();
        let sorted: &[i32] = bytemuck::cast_slice(&sorted_b);
        let experts: &[i32] = bytemuck::cast_slice(&experts_b);
        let total: i32 = bytemuck::cast_slice::<u8, i32>(&total_b)[0];
        check_invariants(
            topk_ids,
            sorted,
            experts,
            total,
            num_experts,
            block_size,
            label,
        );
    }

    /// Deterministic LCG so tests are reproducible without rand.
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

    // ids taken from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
    #[test]
    fn moe_align_happy_path() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        // The worked example from vLLM's moe_align_block_size.py docstring:
        //   topk_ids = [[2,3,4], [1,2,4], [1,3,4], [1,2,3]], block_size = 4,
        //   num_experts = 4.
        // The docstring's expert ids are 1-indexed, but the kernel contract
        // is 0-indexed (ids must be < num_experts), so every id here is
        // shifted down by 1 to make the example run with its stated 4
        // experts.
        let top_k_ids = [1, 2, 3, 0, 1, 3, 0, 2, 3, 0, 1, 2];
        run_case(&stream, &top_k_ids, 4, 4, "docstring happy path");
    }

    #[test]
    fn moe_align_invariants() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();

        // gpt-oss shape: E=128, k=4, uniform-random routing.
        for &tokens in &[1usize, 16, 256, 2048] {
            let mut rng = Lcg(42 + tokens as u64);
            let topk: Vec<i32> = (0..tokens * 4).map(|_| rng.below(128) as i32).collect();
            for &bs in &[16usize, 64, 128] {
                run_case(
                    &stream,
                    &topk,
                    128,
                    bs,
                    &format!("gptoss t={tokens} bs={bs}"),
                );
            }
        }

        // Generic shapes.
        let mut rng = Lcg(7);
        let topk: Vec<i32> = (0..37 * 2).map(|_| rng.below(8) as i32).collect();
        run_case(&stream, &topk, 8, 16, "E=8 k=2 t=37");
        let topk: Vec<i32> = (0..100 * 8).map(|_| rng.below(256) as i32).collect();
        run_case(&stream, &topk, 256, 64, "E=256 k=8 t=100");
    }

    #[test]
    fn moe_align_edge_cases() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();

        // All pairs to one expert (max padding on one segment, all others empty).
        let topk = vec![5i32; 200];
        run_case(&stream, &topk, 128, 16, "all-one-expert");

        // Fewer pairs than one block.
        let topk = vec![0i32, 3, 3, 7];
        run_case(&stream, &topk, 8, 16, "pairs<block");

        // Counts exactly on block boundaries (no padding needed anywhere).
        let mut topk = Vec::new();
        for e in 0..4i32 {
            topk.extend(std::iter::repeat_n(e, 32));
        }
        run_case(&stream, &topk, 8, 16, "exact-blocks");

        // Invariants stable across repeated runs (atomic order may differ;
        // the contract must not).
        let mut rng = Lcg(99);
        let topk: Vec<i32> = (0..512 * 4).map(|_| rng.below(128) as i32).collect();
        for i in 0..3 {
            run_case(&stream, &topk, 128, 16, &format!("repeat-{i}"));
        }
    }

    /// Strided topk_ids: real ids in the first top_k columns of wider rows
    /// (the [s, E] argsort-tensor binding); the padding columns hold values
    /// the kernels must never read.
    #[test]
    fn moe_align_strided_rows() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let (tokens, top_k, row_stride, num_experts, block_size) =
            (37usize, 2usize, 8usize, 8usize, 16usize);
        let mut rng = Lcg(11);
        let mut wide = vec![i32::MIN; tokens * row_stride]; // poison padding
        let mut flat = Vec::with_capacity(tokens * top_k);
        for t in 0..tokens {
            for k in 0..top_k {
                let e = rng.below(num_experts) as i32;
                wide[t * row_stride + k] = e;
                flat.push(e);
            }
        }
        let d_wide = stream
            .memcpy_stod(bytemuck::cast_slice::<i32, u8>(&wide))
            .unwrap();
        let bufs = MoeAlignBuffers::alloc(&stream, flat.len(), num_experts, block_size).unwrap();
        let wide_ptr = d_wide.device_ptr(&stream).0;
        moe_align_block_size(
            &stream,
            wide_ptr,
            flat.len(),
            top_k,
            row_stride,
            num_experts,
            block_size,
            &bufs,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let sorted_b: Vec<u8> = stream.memcpy_dtov(&bufs.sorted_token_ids).unwrap();
        let experts_b: Vec<u8> = stream.memcpy_dtov(&bufs.expert_ids).unwrap();
        let total_b: Vec<u8> = stream.memcpy_dtov(&bufs.num_tokens_post_pad).unwrap();
        check_invariants(
            &flat,
            bytemuck::cast_slice(&sorted_b),
            bytemuck::cast_slice(&experts_b),
            bytemuck::cast_slice::<u8, i32>(&total_b)[0],
            num_experts,
            block_size,
            "strided-rows",
        );
    }
}
