//! gpt-oss-120b model graph with **paged** attention.
//!
//! Identical to `examples/gpt_oss` except the KV cache is a flat bf16 slot
//! pool `(num_slots, KV_DIM)` shared across sequences (paged attention, like
//! `examples/paged_llama`) instead of a dense `(N_KV_HEADS, max_seq, HEAD_DIM)`
//! per-sequence cache. New KV is written to slots with [`scatter_rows`] and the
//! attention context is read with [`gather_rows`]; the causal / sliding-window
//! mask is built IN-GRAPH from per-request `qo_indptr`/`kv_indptr` boundaries
//! (the FlashInfer indptr spelling), which also gives cross-sequence isolation
//! for batched decoding.
//!
//! gpt-oss specifics are preserved: MXFP4 MoE, attention sinks, alternating
//! sliding-window / full attention, YaRN RoPE, biases.

use luminal::{dtype::DType, graph::Graph, prelude::GraphTensor, shape::Expression};
use luminal_nn::{gather_rows, scatter_rows};

use crate::quant::{Mxfp4Experts, unpack_mxfp4};

// gpt-oss-120b hyperparameters (config.json).
//
// The model has 36 layers. On a single 80 GB H100 the full model's ~63 GB of
// resident MXFP4 weights plus luminal's working arena currently OOMs (see the
// gpt_oss README / LUM-645); `GPTOSS_LAYERS` caps the layer count to fit while
// bringing up / validating the pipeline.
pub fn layers() -> usize {
    std::env::var("GPTOSS_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(36)
        .min(36)
}
pub const HIDDEN: usize = 2880;
pub const HEAD_DIM: usize = 64;
pub const N_HEADS: usize = 64;
pub const N_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // 8
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // 4096
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 512
pub const VOCAB_SIZE: usize = 201088;
pub const RMS_NORM_EPS: f32 = 1e-5;
pub const NUM_EXPERTS: usize = 128;
pub const TOP_K: usize = 4;
pub const EXPERT_DIM: usize = 2880; // expert intermediate_size
pub const GATE_UP_OUT: usize = EXPERT_DIM * 2; // 5760
pub const SLIDING_WINDOW: usize = 128;
pub const SWIGLU_LIMIT: f32 = 7.0;
pub const SWIGLU_ALPHA: f32 = 1.702;

/// Number of rotary frequency pairs.
pub const ROPE_DIM: usize = HEAD_DIM / 2; // 32

/// Flat paged KV cache: one bf16 `(num_slots, KV_DIM)` slot pool per layer.
pub struct PagedKVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
}

impl PagedKVCache {
    pub fn new(cx: &mut Graph, num_slots: usize) -> Self {
        let mut k_caches = Vec::with_capacity(layers());
        let mut v_caches = Vec::with_capacity(layers());
        for l in 0..layers() {
            k_caches.push(
                cx.named_tensor(format!("kv_cache.{l}.k"), (num_slots, KV_DIM))
                    .as_dtype(DType::Bf16),
            );
            v_caches.push(
                cx.named_tensor(format!("kv_cache.{l}.v"), (num_slots, KV_DIM))
                    .as_dtype(DType::Bf16),
            );
        }
        Self { k_caches, v_caches }
    }
}

pub struct GptOss {
    embedding: GraphTensor,
    inv_freq: GraphTensor,
    lut_lo: GraphTensor,
    lut_hi: GraphTensor,
    mscale: f32,
    layers: Vec<GptOssLayer>,
    final_norm: GraphTensor,
    lm_head: GraphTensor,
}

/// Per-step paging inputs threaded through every layer.
#[derive(Clone, Copy)]
pub struct PagingInputs {
    pub pos_ids: GraphTensor,     // (s,) Int — absolute positions for RoPE
    pub scatter_idx: GraphTensor, // (s,) Int — slots to write new KV
    pub gather_idx: GraphTensor,  // (c,) Int — slots to read for context
    pub qo_indptr: GraphTensor,   // (r,) Int — cumulative query counts per request
    pub kv_indptr: GraphTensor,   // (r,) Int — cumulative context lengths per request
}

impl GptOss {
    pub fn init(cx: &mut Graph, mscale: f32) -> Self {
        Self {
            embedding: bf16_weight(cx, "model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            // YaRN inv_freq + MXFP4 nibble LUTs are computed on host, uploaded at runtime.
            inv_freq: persist(cx, "rope.inv_freq", ROPE_DIM),
            lut_lo: bf16_weight(cx, "fp4.lut_lo", 256),
            lut_hi: bf16_weight(cx, "fp4.lut_hi", 256),
            mscale,
            layers: (0..layers()).map(|l| GptOssLayer::init(cx, l)).collect(),
            final_norm: bf16_weight(cx, "model.norm.weight", HIDDEN),
            lm_head: bf16_weight(cx, "lm_head.weight", (VOCAB_SIZE, HIDDEN)),
        }
    }

    /// The persistent `rope.inv_freq` tensor; set its data from [`yarn_inv_freq`].
    pub fn rope_inv_freq(&self) -> GraphTensor {
        self.inv_freq
    }

    /// The persistent MXFP4 nibble lookup tables; set from [`crate::quant::fp4_byte_luts`].
    pub fn fp4_luts(&self) -> (GraphTensor, GraphTensor) {
        (self.lut_lo, self.lut_hi)
    }

    pub fn forward(
        &self,
        token_ids: GraphTensor,
        paging: PagingInputs,
        kv_cache: &PagedKVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = token_embedding(self.embedding, token_ids);
        let mut cache_outputs = Vec::with_capacity(layers());
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                self.inv_freq,
                self.mscale,
                self.lut_lo,
                self.lut_hi,
                paging,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }
        let normed = rms_norm(x, self.final_norm);
        let logits = normed.matmul(self.lm_head.cast(DType::F32).t());
        (logits, cache_outputs)
    }
}

struct GptOssLayer {
    layer_idx: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sinks: GraphTensor, // [N_HEADS] bf16
    input_norm: GraphTensor,
    post_attn_norm: GraphTensor,
    router: Linear,
    gate_up: Mxfp4Experts,
    down: Mxfp4Experts,
}

impl GptOssLayer {
    fn init(cx: &mut Graph, l: usize) -> Self {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        Self {
            layer_idx: l,
            q_proj: Linear::init(cx, &p("self_attn.q_proj"), Q_DIM, HIDDEN),
            k_proj: Linear::init(cx, &p("self_attn.k_proj"), KV_DIM, HIDDEN),
            v_proj: Linear::init(cx, &p("self_attn.v_proj"), KV_DIM, HIDDEN),
            o_proj: Linear::init(cx, &p("self_attn.o_proj"), HIDDEN, Q_DIM),
            sinks: bf16_weight(cx, p("self_attn.sinks"), N_HEADS),
            input_norm: bf16_weight(cx, p("input_layernorm.weight"), HIDDEN),
            post_attn_norm: bf16_weight(cx, p("post_attention_layernorm.weight"), HIDDEN),
            router: Linear::init(cx, &p("mlp.router"), NUM_EXPERTS, HIDDEN),
            gate_up: Mxfp4Experts::new(
                &p("mlp.experts.gate_up_proj"),
                NUM_EXPERTS,
                GATE_UP_OUT,
                HIDDEN,
                cx,
            ),
            down: Mxfp4Experts::new(
                &p("mlp.experts.down_proj"),
                NUM_EXPERTS,
                HIDDEN,
                EXPERT_DIM,
                cx,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        mut x: GraphTensor,
        inv_freq: GraphTensor,
        mscale: f32,
        lut_lo: GraphTensor,
        lut_hi: GraphTensor,
        paging: PagingInputs,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        // Attention
        let x_attn = rms_norm(x, self.input_norm);
        let q = self.q_proj.forward(x_attn);
        let k = self.k_proj.forward(x_attn);
        let v = self.v_proj.forward(x_attn);

        let q_rope = rope(q, paging.pos_ids, inv_freq, mscale, N_HEADS);
        let k_rope = rope(k, paging.pos_ids, inv_freq, mscale, N_KV_HEADS);

        // Even layers use sliding-window attention, odd layers full attention.
        let window = self
            .layer_idx
            .is_multiple_of(2)
            .then_some(SLIDING_WINDOW);
        let (attn_out, k_cache_out, v_cache_out) = paged_attention(
            q_rope,
            k_rope,
            v,
            self.sinks,
            paging,
            k_cache_in,
            v_cache_in,
            window,
        );
        x += self.o_proj.forward(attn_out);

        // MoE FFN
        let x_mlp = rms_norm(x, self.post_attn_norm);
        let mlp_out = self.moe(x_mlp, lut_lo, lut_hi);
        (x + mlp_out, k_cache_out, v_cache_out)
    }

    /// gpt-oss MoE: top-4 routing (softmax after topk) + interleaved clamped
    /// SwiGLU experts. Only the selected experts' packed MXFP4 bytes are
    /// gathered and unpacked in-graph (see [`crate::quant`]), so the per-layer
    /// transient scales with `TOP_K`, not the expert count.
    fn moe(&self, x: GraphTensor, lut_lo: GraphTensor, lut_hi: GraphTensor) -> GraphTensor {
        let n = x.dims().len(); // 2 for [s, H]

        // Router: logits = x @ W^T + b, then top-4 and softmax over the 4.
        let router_logits = self.router.forward(x); // [s, E]
        let top_k_indices = router_logits.topk_indexes(TOP_K, n - 1); // [s, k] Int
        let k_expr = Expression::from(TOP_K);
        let row_offsets = x.graph().iota(
            Expression::from('z') / k_expr * NUM_EXPERTS,
            top_k_indices.dims(),
        );
        let top_k_logits = router_logits.gather(row_offsets + top_k_indices); // [s, k]
        let top_k_weights = top_k_logits.softmax(n - 1); // softmax over k

        // gate_up: gather + unpack selected experts -> [s,k,5760,H], then
        // [s,k,1,H] @ [s,k,H,5760] -> [s,k,5760].
        let gate_up_bytes = gather_experts(x, top_k_indices, self.gate_up.blocks); // [s,k,5760,H/2]
        let gate_up_scales = gather_experts(x, top_k_indices, self.gate_up.scales); // [s,k,5760,H/32]
        let gate_up_w = unpack_mxfp4(gate_up_bytes, gate_up_scales, lut_lo, lut_hi); // [s,k,5760,H] bf16
        let x_bf = x.cast(DType::Bf16);
        let x_exp = x_bf.expand_dim(n - 1, TOP_K).unsqueeze(n); // [s,k,1,H]
        let mut gate_up_out = x_exp.matmul(gate_up_w.transpose(2, 3)).squeeze(n); // [s,k,5760] bf16
        gate_up_out += gather_expert_bias(top_k_indices, self.gate_up.bias);

        // Interleaved clamped SwiGLU: gate = [...,::2], up = [...,1::2].
        let gu2 = gate_up_out.split_dims(n, 2); // [s,k,expert_dim,2]
        let gate = gu2.slice((.., .., .., ..1)).squeeze(n + 1); // even lanes [s,k,expert_dim]
        let up = gu2.slice((.., .., .., 1..)).squeeze(n + 1); // odd lanes
        let gate = gate.minimum_f32(SWIGLU_LIMIT); // clamp(max=limit)
        let up = up.clip(-SWIGLU_LIMIT, SWIGLU_LIMIT); // clamp(-limit, limit)
        let glu = gate * (gate * SWIGLU_ALPHA).sigmoid();
        let hidden = (up + 1.0) * glu; // [s,k,expert_dim]

        // down: gather + unpack -> [s,k,H,expert_dim], then
        // [s,k,1,expert_dim] @ [s,k,expert_dim,H] -> [s,k,H].
        let down_bytes = gather_experts(x, top_k_indices, self.down.blocks); // [s,k,H,expert_dim/2]
        let down_scales = gather_experts(x, top_k_indices, self.down.scales); // [s,k,H,expert_dim/32]
        let down_w = unpack_mxfp4(down_bytes, down_scales, lut_lo, lut_hi); // [s,k,H,expert_dim]
        let hidden_exp = hidden.unsqueeze(2); // [s,k,1,expert_dim]
        let mut down_out = hidden_exp.matmul(down_w.transpose(2, 3)).squeeze(2); // [s,k,H] bf16
        down_out += gather_expert_bias(top_k_indices, self.down.bias);

        // Weighted sum over k experts (in F32).
        let down_f = down_out.cast(DType::F32);
        let mut w_exp = top_k_weights.unsqueeze(top_k_weights.dims().len()); // [s,k,1]
        w_exp.shape.expand(down_f.dims());
        (down_f * w_exp).sum(n - 1) // [s, H]
    }
}

/// A bf16 linear layer (`weight (out, in)` + `bias (out)`), computed in F32.
struct Linear {
    weight: GraphTensor,
    bias: GraphTensor,
}

impl Linear {
    fn init(cx: &mut Graph, prefix: &str, out_dim: usize, in_dim: usize) -> Self {
        Self {
            weight: bf16_weight(cx, format!("{prefix}.weight"), (out_dim, in_dim)),
            bias: bf16_weight(cx, format!("{prefix}.bias"), out_dim),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let y = x.matmul(self.weight.cast(DType::F32).t());
        let b = self.bias.cast(DType::F32);
        y + b.expand_lhs(&y.dims()[..y.dims().len() - 1])
    }
}

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

fn bf16_weight(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).as_dtype(DType::Bf16).persist()
}

/// RMSNorm computed in F32; the bf16 weight is cast up at point of use.
fn rms_norm(x: GraphTensor, weight: GraphTensor) -> GraphTensor {
    let normed = x.std_norm(x.shape.last_axis(), RMS_NORM_EPS);
    normed
        * weight
            .cast(DType::F32)
            .expand_lhs(&x.dims()[..x.dims().len() - 1])
}

fn token_embedding(embedding: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
    let seq = token_ids.dims1();
    let gathered = embedding.gather(
        (token_ids * HIDDEN).expand_dim(1, HIDDEN)
            + token_ids.graph().arange(HIDDEN).expand_dim(0, seq),
    );
    gathered.cast(DType::F32)
}

/// Gather expert weight matrices: weights `[E, d1, d2]`, indices `[s, k]` ->
/// `[s, k, d1, d2]`. Indices stay Int (expert flat offsets exceed 2^24).
fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (i, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(i, *dim);
    }
    weights.gather(exp_base + exp_within)
}

/// Gather expert bias vectors: bias `[E, d]`, indices `[s, k]` -> `[s, k, d]`.
fn gather_expert_bias(top_k_indices: GraphTensor, bias: GraphTensor) -> GraphTensor {
    let (_, d) = bias.dims2();
    let base = top_k_indices * d; // [s, k]
    let within = top_k_indices.graph().iota(Expression::from('z'), d); // [d]
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d); // [s, k, d]
    let exp_within = within
        .expand_dim(0, base.dims()[0])
        .expand_dim(1, base.dims()[1]);
    bias.gather(exp_base + exp_within) // already Bf16
}

/// YaRN RoPE. `inv_freq` is the host-computed `[ROPE_DIM]` frequency vector and
/// `mscale` the YaRN attention scaling. Rotation is NeoX-style (rotate halves).
fn rope(
    mut input: GraphTensor,
    pos_ids: GraphTensor,
    inv_freq: GraphTensor,
    mscale: f32,
    n_heads: usize,
) -> GraphTensor {
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1); // [n_heads, seq, HEAD_DIM]

    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freq.cast(DType::F32).expand_dim(0, 1)); // [seq, ROPE_DIM]

    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    let cos = (emb.cos() * mscale).expand_dim(0, n_heads);
    let sin = (emb.sin() * mscale).expand_dim(0, n_heads);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

/// Map an indptr vector `(r,)` to a per-row request index over `n` rows: row
/// `i` belongs to request `j` iff `indptr[j] <= i < indptr[j+1]`. Exact HLIR
/// spelling from `luminal_cuda_lite`'s FlashInfer tests
/// (`test_indptr_to_request_idx`) so the ops stay rule-matchable.
fn indptr_to_request_idx(indptr: GraphTensor, n: Expression) -> GraphTensor {
    let graph = indptr.graph();
    let r = indptr.dims1();
    let indices = graph.arange(n).expand_dim(1, r);
    let indptr_2d = indptr.expand_dim(0, n);
    let ge = indptr_2d.le(indices).cast(DType::Int);
    ge.sum(1).cast(DType::Int) - 1
}

/// Build the additive `(s, c)` attention mask in-graph from the per-request
/// indptrs (FlashInfer `test_compute_attn_mask` spelling): a query attends a
/// context slot iff same request && causal (`c_local_pos <= q_abs_pos`), and —
/// for sliding-window layers — the slot is within the window
/// (`q_abs_pos - c_local_pos < W`, spelled gemma-style as
/// `c_local_pos < q_pos - (W-1)` blocked, so `W-1` shows as a plain constant).
/// Allowed → 0, blocked → -1e10 (or -2e10 when both terms block).
fn compute_attn_mask(
    q_pos: GraphTensor,
    qo_indptr: GraphTensor,
    kv_indptr: GraphTensor,
    c: Expression,
    window: Option<usize>,
) -> GraphTensor {
    let graph = q_pos.graph();
    let s = q_pos.dims1();
    let q_request = indptr_to_request_idx(qo_indptr, s);
    let c_request = indptr_to_request_idx(kv_indptr, c);
    let c_arange = graph.arange(c);
    let c_kv_start = kv_indptr.gather(c_request);
    let c_local_pos = c_arange - c_kv_start;
    let q_req_2d = q_request.expand_dim(1, c);
    let c_req_2d = c_request.expand_dim(0, s);
    let same = q_req_2d.eq(c_req_2d);
    let c_pos_2d = c_local_pos.expand_dim(0, s);
    let qp_2d = q_pos.expand_dim(1, c);
    let causal = c_pos_2d.le(qp_2d);
    let allowed = same.cast(DType::F32) * causal.cast(DType::F32);
    let mask = allowed * 1e10 - 1e10;
    if let Some(w) = window {
        // Sliding window: block context positions older than q_pos - (W-1)
        // (gemma spelling; the "W-1" constant is FlashInfer's window_left).
        let q_f = q_pos.cast(DType::F32);
        let win_lo = q_f - (w - 1) as f32;
        let c_local_f = c_local_pos.cast(DType::F32);
        let too_old = c_local_f.expand_dim(0, s).lt(win_lo.expand_dim(1, c));
        mask + too_old.cast(DType::F32) * -1e10
    } else {
        mask
    }
}

/// Paged attention for one layer: scatter new KV into the bf16 slot pools,
/// gather the context, GQA attention with an in-graph indptr-derived mask and
/// attention sinks. Mirrors the qwen3_moe HLIR spelling (bf16 chain, `* 1.0`
/// contiguous-materialization anchors) so the FlashInfer rewrites can match;
/// the HLIR chain remains the correct fallback.
///
/// - `q_rope` (s, Q_DIM) F32, `k_rope`/`v` (s, KV_DIM) F32 — cast to bf16 here
/// - `sinks` (N_HEADS,) bf16 (raw, no F32 cast)
/// - `k_cache`/`v_cache` (num_slots, KV_DIM) bf16
/// - `window`: `Some(W)` for sliding-window layers, `None` for full attention
#[allow(clippy::too_many_arguments)]
fn paged_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    sinks: GraphTensor, // [N_HEADS] bf16
    paging: PagingInputs,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    window: Option<usize>,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let s = q_rope.dims()[0];
    let ctx = paging.gather_idx.dims()[0];

    // Write the new tokens' KV into their slots (bf16), then read the context.
    let k_cache_out = scatter_rows(
        k_rope.cast(DType::Bf16),
        paging.scatter_idx,
        k_cache,
        KV_DIM,
    );
    let v_cache_out = scatter_rows(v.cast(DType::Bf16), paging.scatter_idx, v_cache, KV_DIM);
    let k = gather_rows(k_cache_out, paging.gather_idx, KV_DIM); // [ctx, KV_DIM] bf16
    let v_ctx = gather_rows(v_cache_out, paging.gather_idx, KV_DIM); // [ctx, KV_DIM] bf16

    // Reshape to heads + GQA broadcast (qwen3_moe spelling): `* 1.0` forces
    // contiguous materialization — the FlashInfer rules anchor on Mul(x, 1.0).
    let q = (q_rope.cast(DType::Bf16) * 1.0)
        .split_dims(1, HEAD_DIM)
        .transpose(0, 1); // [N_HEADS, s, HEAD_DIM]
    let k = k.split_dims(1, HEAD_DIM).permute((1, 2, 0)); // [N_KV_HEADS, HEAD_DIM, ctx]
    let v_3d = v_ctx.split_dims(1, HEAD_DIM).transpose(0, 1); // [N_KV_HEADS, ctx, HEAD_DIM]
    let k = k.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0; // [N_HEADS, HEAD_DIM, ctx]
    let v_3d = v_3d.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0; // [N_HEADS, ctx, HEAD_DIM]

    let scores = q.matmul(k) / (HEAD_DIM as f32).sqrt(); // [N_HEADS, s, ctx] bf16

    // Additive mask built in-graph from the indptrs (same-request isolation +
    // causal + optional sliding window), cast to the scores dtype.
    let mask = compute_attn_mask(
        paging.pos_ids,
        paging.qo_indptr,
        paging.kv_indptr,
        ctx,
        window,
    );
    let masked = scores + mask.cast(DType::Bf16).expand_dim(0, N_HEADS);

    // Sink-augmented softmax: the per-head learned logit enters the denominator
    // only (it never contributes to the V matmul). Simplified spelling — the
    // softmax is shift-invariant, so skipping `.maximum(sink)` in the row max
    // gives identical results while keeping the max/exp/sum chain canonical.
    let row_max = masked.max(2); // [N_HEADS, s]
    let num = (masked - row_max.expand_dim(2, ctx)).exp(); // [N_HEADS, s, ctx]
    let sink_term = (sinks.expand_dim(1, s) - row_max).exp(); // [N_HEADS, s]
    let denom = num.sum(2) + sink_term; // [N_HEADS, s]
    let attn = num / denom.expand_dim(2, ctx);

    let out = attn.matmul(v_3d).transpose(0, 1).merge_dims(1, 2); // [s, Q_DIM] bf16
    (out.cast(DType::F32), k_cache_out, v_cache_out)
}

/// Host-side YaRN inv_freq + attention scaling, matching HF
/// `_compute_yarn_parameters` for gpt-oss.
pub fn yarn_inv_freq() -> (Vec<f32>, f32) {
    let dim = HEAD_DIM as f32;
    let base = 150000f32;
    let factor = 32f32;
    let orig_max = 4096f32;
    let beta_fast = 32f32;
    let beta_slow = 1f32;

    let find_dim = |num_rot: f32| {
        (dim * (orig_max / (num_rot * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * base.ln())
    };
    let low = find_dim(beta_fast).floor().max(0.0);
    let high = find_dim(beta_slow).ceil().min(dim - 1.0);
    let denom = (high - low).max(1e-3);

    let mut inv_freq = Vec::with_capacity(ROPE_DIM);
    for i in 0..ROPE_DIM {
        let pos_freq = base.powf((2 * i) as f32 / dim);
        let inv_extrap = 1.0 / pos_freq;
        let inv_interp = 1.0 / (factor * pos_freq);
        let ramp = (((i as f32) - low) / denom).clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;
        inv_freq.push(inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor);
    }
    let mscale = 0.1 * factor.ln() + 1.0;
    (inv_freq, mscale)
}
