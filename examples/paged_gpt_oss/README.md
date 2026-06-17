# paged_gpt_oss (paged attention, MXFP4)

A **paged-attention** variant of the [`gpt_oss`](../gpt_oss) example: same gpt-oss-120b model
(MXFP4 MoE, attention sinks, alternating sliding-window/full attention, YaRN RoPE), but with a
flat slot-pool KV cache shared across sequences via a CPU page table — like
[`paged_llama`](../paged_llama). Validated to produce the **identical** tokens as the dense
`gpt_oss` example (which matches HuggingFace).

```
cargo run --release -p paged_gpt_oss
```

## How paging works here

- The KV cache is one flat `(num_slots, KV_DIM)` pool per layer. A CPU `PageTable` maps each
  sequence to a list of physical slots; slots are allocated in position order.
- Per step the host builds `scatter_idx` (slots for the new tokens), `gather_idx` (all context
  slots for the sequences in the batch), `pos_ids`, and **two additive masks** — full-causal and
  causal+sliding-window — that also enforce **cross-sequence isolation**.
- The graph uses `luminal_nn::scatter_rows` / `gather_rows` to write/read the cache, then runs GQA
  attention with attention sinks and the precomputed mask (each layer picks full or sliding). See
  `paged_attention` in `src/model.rs`.

## Demo phases

1. Prefill sequence A (token by token).
2. Decode A.
3. Prefill sequence B into the **same** slot pool.
4. **Supersequence decode**: A and B decode together in one `s=2` batched step; the masks keep
   each query attending only to its own sequence.

## Memory note (important)

gpt-oss's MoE expert-gather memory scales with the batch dimension `s` — a full-prompt prefill at
`s=35` estimates ~239 GB. So, unlike `paged_llama`, this example **prefills token-by-token (s=1)**
and shows batching as an `s=2` supersequence decode. The `s=2` bucket roughly doubles the `s=1`
working set, which does not fit beside ~63 GB of weights at the full 36 layers on an 80 GB GPU.

Env knobs:

- `PAGED_BATCH` (default 1) — run the multi-sequence batching demo (phases 3–4, needs the `s=2`
  bucket). Set `PAGED_BATCH=0` for a full-model **single-sequence** run.
- `GPTOSS_LAYERS` (default 36) — cap layers to fit / iterate (output only correct at 36).
- `GPTOSS_MEM_CAP_GIB` (default 14), `NUM_SLOTS` (default 4096), and the inherited
  `LUMINAL_CUBLASLT_WORKSPACE_MB` (set to 2 by the example).

Recommended: the multi-sequence demo at a reduced `GPTOSS_LAYERS` (fits comfortably), and
`PAGED_BATCH=0` at full 36 layers for the single-sequence HF-matching run (same stochastic ~80 GB
fit as the dense example — re-run on OOM). The robust fix is cross-layer arena reuse in
`luminal_cuda_lite` (tracked in LUM-645).

## Validation

Run both examples at the same `GPTOSS_LAYERS` and compare the first-token argmax — they match
(e.g. `16809` at 8 layers), confirming the paged attention is numerically equivalent to the dense
path. The dense `gpt_oss` already matches HF at 36 layers (`200005` → `<|channel|>analysis...`).
