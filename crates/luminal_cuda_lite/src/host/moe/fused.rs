//! FusedMoE: the gpt-oss MXFP4 MoE as one host op — a stream-ordered pair of
//! launches (see `decode.cu`): phase 1 gate_up dequant-GEMV + clamped SwiGLU,
//! phase 2 down projection + top-k weighted mix, for ALL batch sizes (the
//! tiled GEMM chain was deleted after a forced-path A/B, and the cooperative
//! single-launch variant after the split A/B; see comments in `execute`).
//!
//! Installed by the union-only rewrite in `fused_moe_rewrite.egg`, whose LHS
//! is the model's dense reference spelling (`moe_naive`). Enforcement of the
//! fused choice is the candidate memory filter: the raw naive spelling
//! materializes expert-dense tensors no genome can afford.
//!
//! Inputs (graph edges, in order):
//!   0: x            [s, hidden]               F32
//!   1: topk_idx     [s, >=top_k] row-major    Int  (argsort tensor; row
//!      stride derived from the buffer length, GLUMoE-style)
//!   2: topk_weights [s, top_k]                F32
//!   3: gu_blocks    [E, 2*inter, hidden/2]    U8 (packed fp4, lo nibble = even k)
//!   4: gu_scales    [E, 2*inter, hidden/32]   U8 (e8m0)
//!   5: gu_bias      [E, 2*inter]              BF16
//!   6: dn_blocks    [E, hidden, inter/2]      U8
//!   7: dn_scales    [E, hidden, inter/32]     U8
//!   8: dn_bias      [E, hidden]               BF16
//!
//! Output: [s, hidden] F32.
//!
//! `num_experts` is derived from the weight buffer lengths (not op metadata).
//! No host readback, no mid-execute synchronize: every launch is
//! stream-ordered, and the kernels live in process-wide caches (the blocks'
//! statics), so cloning this op during GA profiling costs nothing.

use std::sync::Arc;

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
    cudarc::driver::CudaStream,
    host::{DeviceBuffer, HostOp},
};

use super::{align, decode, moe_gemm, moe_ops};

const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;

/// Force the process-wide NVRTC compiles of the three block modules, once.
/// Extraction has no runtime stream, so this binds the device's primary
/// context itself; if no device is available the compile stays lazy (first
/// execute will pay it, as before).
fn ensure_kernels_compiled() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if let Ok(ctx) = crate::cudarc::driver::CudaContext::new(0) {
            let stream = ctx.default_stream();
            decode::warm(&stream);
            super::moe_ops::warm(&stream);
            super::align::warm(&stream);
            super::moe_gemm::warm(&stream);
        }
    });
}

/// Which MoE implementation `execute` runs. Grouped = the expert-grouped
/// (tokens-to-weights) GEMM chain; Gemv = the per-token fused decode kernel.
/// `LUMINAL_MOE_FORCE_PATH=gemv|grouped` overrides for A/B and rollback
/// (gemv = full rollback to the pre-grouped behavior). Unset: grouped when
/// num_pairs > LUMINAL_MOE_GEMM_MIN_PAIRS (default 64) — the bench crossover
/// (gemv 1.80/3.48/7.00/14.20 ms vs mma 1.90/2.69/3.68/4.36 ms at
/// seq 16/32/64/128, i.e. pairs 64/128/256/512).
#[derive(Clone, Copy, PartialEq, Debug)]
enum MoePath {
    Gemv,
    Grouped,
}

fn moe_gemm_min_pairs() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("LUMINAL_MOE_GEMM_MIN_PAIRS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

fn forced_moe_path() -> Option<MoePath> {
    static FORCED: std::sync::OnceLock<Option<MoePath>> = std::sync::OnceLock::new();
    *FORCED.get_or_init(
        || match std::env::var("LUMINAL_MOE_FORCE_PATH").as_deref() {
            Ok("gemv") => Some(MoePath::Gemv),
            Ok("grouped") => Some(MoePath::Grouped),
            Ok(other) => {
                eprintln!("FusedMoE: unknown LUMINAL_MOE_FORCE_PATH={other:?} ignored");
                None
            }
            Err(_) => None,
        },
    )
}

/// The expert-grouped (tokens-to-weights) chain: cast x to bf16, sort
/// (token, expert) pairs by expert on-GPU, gate_up GEMM (bias in epilogue),
/// SwiGLU, down GEMM (routing weight in epilogue), top-k sum. All
/// stream-ordered; no DtoH, no mid-execute sync. Requires
/// hidden % 64 == 0 && intermediate % 64 == 0 (the GEMMs' K % BK contract);
/// the mma kernel additionally needs K <= moe_gemm::MMA_MAX_K (both K's are
/// 2880 for gpt-oss). Standalone fn (not a method) so tests drive it
/// directly instead of mutating the process-global LUMINAL_MOE_FORCE_PATH.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_grouped(
    stream: &Arc<CudaStream>,
    x_ptr: u64,
    gu_blocks: u64,
    gu_scales: u64,
    gu_bias: u64,
    dn_blocks: u64,
    dn_scales: u64,
    dn_bias: u64,
    topk_idx: u64,
    topk_vals: u64,
    out_ptr: u64,
    hidden: usize,
    intermediate: usize,
    top_k: usize,
    seq: usize,
    num_experts: usize,
    idx_row_stride: usize,
) -> anyhow::Result<()> {
    let num_pairs = seq * top_k;
    let gate_up_n = 2 * intermediate;
    let bm = moe_gemm::BM_MMA;
    #[allow(clippy::too_many_arguments)]
    let gemm = |a: u64,
                bq: u64,
                bs: u64,
                c: u64,
                bias: u64,
                tw: u64,
                sids: u64,
                eids: u64,
                npp: u64,
                n: usize,
                k: usize,
                em: usize,
                nvt: usize,
                tk: usize,
                mrw: bool|
     -> anyhow::Result<()> {
        moe_gemm::fused_moe_mxfp4_gemm_mma(
            stream, a, bq, bs, c, bias, tw, sids, eids, npp, n, k, em, nvt, tk, mrw,
        )
    };

    // Scratch (freed at end of execute; arena packing via extra_buffer_nodes
    // is a later refinement).
    let x_bf16 = unsafe { stream.alloc::<u8>(seq * hidden * 2)? };
    let align_bufs = align::MoeAlignBuffers::alloc(stream, num_pairs, num_experts, bm)?;
    let gu_out = unsafe { stream.alloc::<u8>(num_pairs * gate_up_n * 2)? };
    let hid = unsafe { stream.alloc::<u8>(num_pairs * intermediate * 2)? };
    let dn_out = unsafe { stream.alloc::<u8>(num_pairs * hidden * 2)? };

    let sptr = |b: &crate::cudarc::driver::CudaSlice<u8>| -> u64 {
        use crate::cudarc::driver::DevicePtr;
        b.device_ptr(stream).0
    };

    moe_ops::f32_to_bf16(stream, x_ptr, sptr(&x_bf16), seq * hidden)?;

    align::moe_align_block_size(
        stream,
        topk_idx,
        num_pairs,
        top_k,
        idx_row_stride,
        num_experts,
        bm,
        &align_bufs,
    )?;

    let em = align_bufs.max_num_tokens_padded;
    gemm(
        sptr(&x_bf16),
        gu_blocks,
        gu_scales,
        sptr(&gu_out),
        gu_bias,
        topk_vals,
        sptr(&align_bufs.sorted_token_ids),
        sptr(&align_bufs.expert_ids),
        sptr(&align_bufs.num_tokens_post_pad),
        gate_up_n,
        hidden,
        em,
        num_pairs,
        top_k,
        false,
    )?;

    moe_ops::swiglu_interleaved(
        stream,
        sptr(&gu_out),
        sptr(&hid),
        num_pairs,
        intermediate,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
    )?;

    // Down GEMM consumes per-pair rows (top_k=1 indexing) and applies the
    // routing weight in its epilogue.
    gemm(
        sptr(&hid),
        dn_blocks,
        dn_scales,
        sptr(&dn_out),
        dn_bias,
        topk_vals,
        sptr(&align_bufs.sorted_token_ids),
        sptr(&align_bufs.expert_ids),
        sptr(&align_bufs.num_tokens_post_pad),
        hidden,
        intermediate,
        em,
        num_pairs,
        1,
        true,
    )?;

    moe_ops::moe_sum(stream, sptr(&dn_out), out_ptr, seq, top_k, hidden)?;
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct FusedMoE {
    hidden: Expression,
    intermediate: Expression,
    top_k: Expression,
}

impl EgglogOp for FusedMoE {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedMoE",
            &[
                ("hidden", EXPRESSION),
                ("intermediate", EXPRESSION),
                ("top_k", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                (
                    (= ?e (Op (FusedMoE ?hidden ?intermediate ?top_k) ?inputs))
                )
                (
                    (set (dtype ?e) (F32))
                )
                :ruleset dtype_prop
            )",
            ),
            Rule::raw(include_str!["fused_moe_rewrite.egg"]),
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
        let top_k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let extracted = FusedMoE {
            hidden,
            intermediate,
            top_k,
        };
        // Compile the block kernels now (flashinfer precedent: JIT at extract,
        // not first execute) so the ~650ms of NVRTC — align.cu's CUB headers
        // dominate at ~390ms — never lands inside a timed profiling trial.
        ensure_kernels_compiled();
        (
            LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>),
            input_enodes,
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for FusedMoE {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 9 {
            anyhow::bail!("FusedMoE expected 9 inputs, got {}", inputs.len());
        }

        let hidden = self
            .hidden
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE hidden dim is unresolved"))?;
        let intermediate = self
            .intermediate
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE intermediate dim is unresolved"))?;
        let top_k = self
            .top_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE top_k is unresolved"))?;
        if top_k == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            hidden % 32 == 0 && intermediate % 32 == 0,
            "FusedMoE dims must be multiples of 32 (e8m0 group width): hidden={hidden}, intermediate={intermediate}"
        );
        let gate_up_n = 2 * intermediate;

        let output_bytes = self
            .output_bytes()
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE output byte size is unresolved"))?;
        anyhow::ensure!(
            output_bytes % (hidden * 4) == 0,
            "FusedMoE output bytes {output_bytes} not divisible by row bytes {}",
            hidden * 4
        );
        let seq = output_bytes / (hidden * 4);
        if seq == 0 {
            return Ok(());
        }
        let num_pairs = seq * top_k;

        let get_buffer = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("FusedMoE missing {name} buffer for LLIR node {node:?}")
            })
        };
        let x_buf = get_buffer("x", inputs[0])?;
        let topk_idx_buf = get_buffer("topk indices", inputs[1])?;
        let topk_vals_buf = get_buffer("topk weights", inputs[2])?;
        let gu_blocks_buf = get_buffer("gate_up blocks", inputs[3])?;
        let gu_scales_buf = get_buffer("gate_up scales", inputs[4])?;
        let gu_bias_buf = get_buffer("gate_up bias", inputs[5])?;
        let dn_blocks_buf = get_buffer("down blocks", inputs[6])?;
        let dn_scales_buf = get_buffer("down scales", inputs[7])?;
        let dn_bias_buf = get_buffer("down bias", inputs[8])?;
        let output_buf = get_buffer("output", self_node)?;

        // num_experts is derived from the resident weight buffers.
        let gu_stride = gate_up_n * hidden / 2;
        anyhow::ensure!(
            gu_stride > 0 && gu_blocks_buf.len() % gu_stride == 0,
            "FusedMoE gate_up blocks len {} not a multiple of per-expert stride {gu_stride}",
            gu_blocks_buf.len()
        );
        let num_experts = gu_blocks_buf.len() / gu_stride;
        anyhow::ensure!(num_experts > 0, "FusedMoE derived zero experts");
        let checks: [(&str, usize, usize); 5] = [
            (
                "gate_up scales",
                gu_scales_buf.len(),
                num_experts * gate_up_n * hidden / 32,
            ),
            (
                "gate_up bias",
                gu_bias_buf.len(),
                num_experts * gate_up_n * 2,
            ),
            (
                "down blocks",
                dn_blocks_buf.len(),
                num_experts * hidden * intermediate / 2,
            ),
            (
                "down scales",
                dn_scales_buf.len(),
                num_experts * hidden * intermediate / 32,
            ),
            ("down bias", dn_bias_buf.len(), num_experts * hidden * 2),
        ];
        for (name, have, want) in checks {
            anyhow::ensure!(
                have >= want,
                "FusedMoE {name} buffer too small: {have} < {want}"
            );
        }
        anyhow::ensure!(
            x_buf.len() >= seq * hidden * 4,
            "FusedMoE x buffer too small: {} < {}",
            x_buf.len(),
            seq * hidden * 4
        );
        anyhow::ensure!(
            output_buf.len() >= output_bytes,
            "FusedMoE output buffer too small: {} < {output_bytes}",
            output_buf.len()
        );

        // Row strides derived from buffer lengths (the rewrite binds the full
        // [s, E] argsort tensor as topk_idx; weights come from the softmax as
        // [s, top_k], but derive anyway).
        let derive_stride = |name: &str, len_bytes: usize| -> anyhow::Result<usize> {
            anyhow::ensure!(
                len_bytes.is_multiple_of(4) && (len_bytes / 4).is_multiple_of(seq),
                "FusedMoE {name} buffer len {len_bytes} not divisible into {seq} rows"
            );
            let stride = len_bytes / 4 / seq;
            anyhow::ensure!(
                stride >= top_k,
                "FusedMoE {name} row stride {stride} < top_k {top_k}"
            );
            Ok(stride)
        };
        let idx_row_stride = derive_stride("topk index", topk_idx_buf.len())?;
        let vals_row_stride = derive_stride("topk weights", topk_vals_buf.len())?;
        anyhow::ensure!(
            vals_row_stride == top_k,
            "FusedMoE topk weights must be contiguous [s, top_k]; got row stride {vals_row_stride}"
        );

        let span = tracing::span!(
            tracing::Level::TRACE,
            "FusedMoE",
            seq,
            num_pairs,
            num_experts,
            hidden,
            intermediate
        );
        let _guard = span.enter();

        // Path history: the ORIGINAL tiled GEMM chain (SIMT f32 inner loop,
        // fixed BM=64) lost a forced-path A/B at 36L (decode TPOT 43.2 vs
        // 87.5 ms; 35-token prefill 295 vs 400 ms) and was deleted in
        // 46c4aa45. nsys measurement (2026-07-16) then showed the GEMV costs
        // ~112-122us/token at ALL batch sizes (no cross-token weight reuse,
        // ~445GB/s) — 71% of a 64-token prefill tick — so the grouped chain
        // is restored as the prefill path, with a tensor-core inner loop
        // replacing the SIMT one that actually lost that A/B.
        let path = forced_moe_path().unwrap_or({
            if num_pairs > moe_gemm_min_pairs() {
                MoePath::Grouped
            } else {
                MoePath::Gemv
            }
        });
        // Both grouped GEMMs need K % 64 (gate_up K=hidden, down K=inter).
        if path == MoePath::Grouped && hidden % 64 == 0 && intermediate % 64 == 0 {
            return execute_grouped(
                stream,
                x_buf.ptr(),
                gu_blocks_buf.ptr(),
                gu_scales_buf.ptr(),
                gu_bias_buf.ptr(),
                dn_blocks_buf.ptr(),
                dn_scales_buf.ptr(),
                dn_bias_buf.ptr(),
                topk_idx_buf.ptr(),
                topk_vals_buf.ptr(),
                output_buf.ptr(),
                hidden,
                intermediate,
                top_k,
                seq,
                num_experts,
                idx_row_stride,
            );
        }

        let hidden_scratch = unsafe { stream.alloc::<u8>(num_pairs * intermediate * 4)? };
        let hs_ptr = {
            use crate::cudarc::driver::DevicePtr;
            hidden_scratch.device_ptr(stream).0
        };
        decode::fused_moe_decode(
            stream,
            x_buf.ptr(),
            gu_blocks_buf.ptr(),
            gu_scales_buf.ptr(),
            gu_bias_buf.ptr(),
            dn_blocks_buf.ptr(),
            dn_scales_buf.ptr(),
            dn_bias_buf.ptr(),
            topk_idx_buf.ptr(),
            topk_vals_buf.ptr(),
            hs_ptr,
            output_buf.ptr(),
            hidden,
            intermediate,
            top_k,
            seq,
            idx_row_stride,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
        )?;
        Ok(())
    }

    fn output_size(&self) -> Expression {
        // [seq, hidden] F32; 's' is the seq dim by model convention (GLUMoE
        // does the same).
        Expression::from('s') * self.hidden
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4 // F32
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("FusedMoE")
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_ref::{
        ChainWeights, assert_close, host_chain_reference, host_chain_reference_f32, to_bf16_bytes,
    };
    use super::*;
    use crate::cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

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

    /// Execute the op directly (hand-built buffers map) at tiny dims and
    /// compare against the shared host chain reference. The topk index buffer
    /// is deliberately WIDER than top_k ([s, 8] with poisoned padding) to
    /// exercise the row-stride derivation the graph binding relies on.
    #[test]
    fn fused_moe_op_tiny() {
        run_op_case(13, false);
    }

    #[test]
    fn fused_moe_op_large_batch() {
        // 40 tokens x top_k 2 = 80 pairs: grid-strides past one resident
        // wave (the old tiled regime), same kernel.
        run_op_case(40, false);
    }

    /// T2: the restored expert-grouped chain at the op boundary — 40 tokens
    /// (80 pairs, multiple BM blocks) and 13 tokens (single ragged block).
    #[test]
    fn fused_moe_op_grouped_tiny() {
        run_op_case(40, true);
        run_op_case(13, true);
    }

    /// Build device buffers for a synthetic MoE of the given dims; returns
    /// (host copies for oracles, device slices in fused-op input order).
    #[allow(clippy::type_complexity)]
    fn synth_moe(
        stream: &Arc<CudaStream>,
        rng: &mut Lcg,
        tokens: usize,
        top_k: usize,
        e_cnt: usize,
        hidden: usize,
        inter: usize,
    ) -> (
        (
            Vec<f32>,
            Vec<u8>,
            Vec<u8>,
            Vec<f32>,
            Vec<u8>,
            Vec<u8>,
            Vec<f32>,
            Vec<i32>,
            Vec<f32>,
        ),
        Vec<CudaSlice<u8>>,
    ) {
        let gate_up_n = 2 * inter;
        let x: Vec<f32> = (0..tokens * hidden)
            .map(|_| (rng.below(200) as f32 - 100.0) / 50.0)
            .collect();
        let gu_q: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let gu_s: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 32)
            .map(|_| 121 + rng.below(8) as u8)
            .collect();
        let gu_bias: Vec<f32> = (0..e_cnt * gate_up_n)
            .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
            .collect();
        let dn_q: Vec<u8> = (0..e_cnt * hidden * inter / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let dn_s: Vec<u8> = (0..e_cnt * hidden * inter / 32)
            .map(|_| 121 + rng.below(8) as u8)
            .collect();
        let dn_bias: Vec<f32> = (0..e_cnt * hidden)
            .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
            .collect();
        let mut topk_ids = Vec::with_capacity(tokens * top_k);
        let mut topk_w = Vec::with_capacity(tokens * top_k);
        for _ in 0..tokens {
            let mut picked = [usize::MAX; 8];
            let mut wsum = 0.0f32;
            let mut ws = [0.0f32; 8];
            for k in 0..top_k {
                let mut e = rng.below(e_cnt);
                while picked[..k].contains(&e) {
                    e = rng.below(e_cnt);
                }
                picked[k] = e;
                ws[k] = 0.1 + (rng.below(90) as f32) / 100.0;
                wsum += ws[k];
            }
            for k in 0..top_k {
                topk_ids.push(picked[k] as i32);
                topk_w.push(ws[k] / wsum);
            }
        }
        let up = |bytes: &[u8]| stream.clone_htod(bytes).unwrap();
        let dev = vec![
            up(bytemuck::cast_slice::<f32, u8>(&x)),
            up(bytemuck::cast_slice::<i32, u8>(&topk_ids)),
            up(bytemuck::cast_slice::<f32, u8>(&topk_w)),
            up(&gu_q),
            up(&gu_s),
            up(&to_bf16_bytes(&gu_bias)),
            up(&dn_q),
            up(&dn_s),
            up(&to_bf16_bytes(&dn_bias)),
        ];
        (
            (
                x, gu_q, gu_s, gu_bias, dn_q, dn_s, dn_bias, topk_ids, topk_w,
            ),
            dev,
        )
    }

    /// T5: the mma chain at the real contraction size (K = 2880 on both
    /// GEMMs, 45 BK steps, 90-byte scale rows) — the tiny cases can't see
    /// scale-strip indexing bugs past group 2.
    #[test]
    fn moe_mma_real_k() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let ptr = |b: &CudaSlice<u8>| b.device_ptr(&stream).0;
        let (tokens, top_k, e_cnt, hidden, inter) = (16usize, 4usize, 8usize, 2880usize, 2880usize);
        let mut rng = Lcg(97);
        let (host, dev) = synth_moe(&stream, &mut rng, tokens, top_k, e_cnt, hidden, inter);
        let (x, gu_q, gu_s, gu_bias, dn_q, dn_s, dn_bias, topk_ids, topk_w) = host;
        let d_out = stream.alloc_zeros::<u8>(tokens * hidden * 4).unwrap();
        super::execute_grouped(
            &stream,
            ptr(&dev[0]),
            ptr(&dev[3]),
            ptr(&dev[4]),
            ptr(&dev[5]),
            ptr(&dev[6]),
            ptr(&dev[7]),
            ptr(&dev[8]),
            ptr(&dev[1]),
            ptr(&dev[2]),
            ptr(&d_out),
            hidden,
            inter,
            top_k,
            tokens,
            e_cnt,
            top_k,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let got_b = stream.clone_dtoh(&d_out).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&got_b);
        let weights = ChainWeights {
            gu_q: &gu_q,
            gu_s: &gu_s,
            gu_bias: &gu_bias,
            dn_q: &dn_q,
            dn_s: &dn_s,
            dn_bias: &dn_bias,
        };
        let want = host_chain_reference(
            &weights, &x, &topk_ids, &topk_w, tokens, top_k, hidden, inter,
        );
        assert_close(got, &want, 0.05, "mma real-K chain");
    }

    /// T6: the decision bench — GEMV vs SIMT-grouped vs MMA-grouped at real
    /// dims across the seq range the engine actually produces. Prints
    /// ms/call and effective weight-streaming GB/s. Not an assertion.
    #[test]
    #[ignore = "benchmark, run explicitly"]
    fn moe_bench_gemv_vs_grouped() {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();
        let ptr = |b: &CudaSlice<u8>| b.device_ptr(&stream).0;
        let (top_k, e_cnt, hidden, inter) = (4usize, 128usize, 2880usize, 2880usize);
        let gate_up_n = 2 * inter;
        let mut rng = Lcg(1234);
        let max_seq = 128usize;
        let (_, dev) = synth_moe(&stream, &mut rng, max_seq, top_k, e_cnt, hidden, inter);
        let d_out = stream.alloc_zeros::<u8>(max_seq * hidden * 4).unwrap();
        let scratch = stream
            .alloc_zeros::<u8>(max_seq * top_k * inter * 4)
            .unwrap();

        // Per-pair expert-weight bytes (fp4 + scales), for the no-reuse
        // traffic model; grouped traffic is bounded by touched-experts
        // instead — printed per unique-expert sweep for context.
        let per_expert_bytes = (gate_up_n * hidden / 2 + gate_up_n * hidden / 32)
            + (hidden * inter / 2 + hidden * inter / 32);

        let iters = 50;
        for &seq in &[16usize, 32, 64, 128] {
            let num_pairs = seq * top_k;
            let mut run = |label: &str, f: &dyn Fn()| {
                for _ in 0..5 {
                    f();
                }
                stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f();
                }
                stream.synchronize().unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
                // GEMV traffic: every pair re-reads its expert. Grouped
                // floor: each of <=E touched experts read once.
                let gemv_gb = (num_pairs * per_expert_bytes) as f64 / 1e9;
                let floor_gb = (e_cnt.min(num_pairs) * per_expert_bytes) as f64 / 1e9;
                println!(
                    "seq {seq:>4} {label:<6} {ms:>8.3} ms | gemv-model {:>6.0} GB/s | floor-model {:>6.0} GB/s",
                    gemv_gb / (ms / 1e3),
                    floor_gb / (ms / 1e3),
                );
            };
            run("gemv", &|| {
                decode::fused_moe_decode(
                    &stream,
                    ptr(&dev[0]),
                    ptr(&dev[3]),
                    ptr(&dev[4]),
                    ptr(&dev[5]),
                    ptr(&dev[6]),
                    ptr(&dev[7]),
                    ptr(&dev[8]),
                    ptr(&dev[1]),
                    ptr(&dev[2]),
                    ptr(&scratch),
                    ptr(&d_out),
                    hidden,
                    inter,
                    top_k,
                    seq,
                    top_k,
                    SWIGLU_ALPHA,
                    SWIGLU_LIMIT,
                )
                .unwrap()
            });
            {
                run("mma", &|| {
                    super::execute_grouped(
                        &stream,
                        ptr(&dev[0]),
                        ptr(&dev[3]),
                        ptr(&dev[4]),
                        ptr(&dev[5]),
                        ptr(&dev[6]),
                        ptr(&dev[7]),
                        ptr(&dev[8]),
                        ptr(&dev[1]),
                        ptr(&dev[2]),
                        ptr(&d_out),
                        hidden,
                        inter,
                        top_k,
                        seq,
                        e_cnt,
                        top_k,
                    )
                    .unwrap()
                });
            }
        }
    }

    fn run_op_case(tokens: usize, grouped: bool) {
        let Ok(ctx) = CudaContext::new(0) else { return };
        let stream = ctx.default_stream();

        let (top_k, e_cnt) = (2usize, 4usize);
        let (hidden, inter) = (64usize, 64usize);
        let gate_up_n = 2 * inter;
        let idx_row_stride = 8usize; // wider than top_k, like the argsort tensor
        let mut rng = Lcg(31);

        let x: Vec<f32> = (0..tokens * hidden)
            .map(|_| (rng.below(200) as f32 - 100.0) / 50.0)
            .collect();
        let gu_q: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let gu_s: Vec<u8> = (0..e_cnt * gate_up_n * hidden / 32)
            .map(|_| 125 + rng.below(6) as u8)
            .collect();
        let gu_bias: Vec<f32> = (0..e_cnt * gate_up_n)
            .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
            .collect();
        let dn_q: Vec<u8> = (0..e_cnt * hidden * inter / 2)
            .map(|_| rng.below(256) as u8)
            .collect();
        let dn_s: Vec<u8> = (0..e_cnt * hidden * inter / 32)
            .map(|_| 125 + rng.below(6) as u8)
            .collect();
        let dn_bias: Vec<f32> = (0..e_cnt * hidden)
            .map(|_| (rng.below(100) as f32 - 50.0) / 25.0)
            .collect();

        let mut topk_ids = Vec::with_capacity(tokens * top_k);
        let mut wide_ids = vec![i32::MIN; tokens * idx_row_stride]; // poison padding
        let mut topk_w = Vec::with_capacity(tokens * top_k);
        for t in 0..tokens {
            let first = rng.below(e_cnt);
            let second = (first + 1 + rng.below(e_cnt - 1)) % e_cnt;
            for (k, e) in [first, second].into_iter().enumerate() {
                topk_ids.push(e as i32);
                wide_ids[t * idx_row_stride + k] = e as i32;
            }
            let w0 = 0.2 + (rng.below(60) as f32) / 100.0;
            topk_w.extend([w0, 1.0 - w0]);
        }

        // Upload everything; build the buffers map with synthetic node ids.
        let up = |bytes: &[u8]| stream.clone_htod(bytes).unwrap();
        let d_x = up(bytemuck::cast_slice::<f32, u8>(&x));
        let d_ids = up(bytemuck::cast_slice::<i32, u8>(&wide_ids));
        let d_w = up(bytemuck::cast_slice::<f32, u8>(&topk_w));
        let d_gu_q = up(&gu_q);
        let d_gu_s = up(&gu_s);
        let d_gu_b = up(&to_bf16_bytes(&gu_bias));
        let d_dn_q = up(&dn_q);
        let d_dn_s = up(&dn_s);
        let d_dn_b = up(&to_bf16_bytes(&dn_bias));
        let d_out = stream.alloc_zeros::<u8>(tokens * hidden * 4).unwrap();

        let dev = |b: &CudaSlice<u8>| DeviceBuffer::new(b.device_ptr(&stream).0, b.len());
        let slices = [
            &d_x, &d_ids, &d_w, &d_gu_q, &d_gu_s, &d_gu_b, &d_dn_q, &d_dn_s, &d_dn_b,
        ];
        let mut buffers: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        let inputs: Vec<NodeIndex> = (0..9).map(NodeIndex::new).collect();
        for (i, s) in slices.iter().enumerate() {
            buffers.insert(NodeIndex::new(i), dev(s));
        }
        let self_node = NodeIndex::new(99);
        buffers.insert(self_node, dev(&d_out));

        let op = FusedMoE {
            hidden: hidden.into(),
            intermediate: inter.into(),
            top_k: top_k.into(),
        };
        let mut dyn_map = FxHashMap::default();
        dyn_map.insert('s', tokens);
        if grouped {
            // Drive the grouped chain directly: env-var path forcing would
            // race other tests (process-global).
            super::execute_grouped(
                &stream,
                d_x.device_ptr(&stream).0,
                d_gu_q.device_ptr(&stream).0,
                d_gu_s.device_ptr(&stream).0,
                d_gu_b.device_ptr(&stream).0,
                d_dn_q.device_ptr(&stream).0,
                d_dn_s.device_ptr(&stream).0,
                d_dn_b.device_ptr(&stream).0,
                d_ids.device_ptr(&stream).0,
                d_w.device_ptr(&stream).0,
                d_out.device_ptr(&stream).0,
                hidden,
                inter,
                top_k,
                tokens,
                e_cnt,
                idx_row_stride,
            )
            .unwrap();
        } else {
            op.execute(&stream, self_node, &inputs, &buffers, &dyn_map)
                .unwrap();
        }
        stream.synchronize().unwrap();

        let got_b = stream.clone_dtoh(&d_out).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&got_b);
        let weights = ChainWeights {
            gu_q: &gu_q,
            gu_s: &gu_s,
            gu_bias: &gu_bias,
            dn_q: &dn_q,
            dn_s: &dn_s,
            dn_bias: &dn_bias,
        };
        // The GEMV path keeps f32 internally; the grouped chain rounds
        // through bf16 between stages — each gets the matching oracle. An
        // undirected op.execute call routes by the num_pairs threshold, so
        // large batches are expected to take the grouped path (this is the
        // dispatch's op-boundary coverage).
        let expect_grouped = grouped || tokens * 2 > super::moe_gemm_min_pairs();
        if expect_grouped {
            let want = host_chain_reference(
                &weights, &x, &topk_ids, &topk_w, tokens, top_k, hidden, inter,
            );
            assert_close(got, &want, 0.05, "fused_moe_op_grouped");
        } else {
            let want = host_chain_reference_f32(
                &weights, &x, &topk_ids, &topk_w, tokens, top_k, hidden, inter,
            );
            assert_close(got, &want, 0.01, "fused_moe_op");
        }
    }
}
