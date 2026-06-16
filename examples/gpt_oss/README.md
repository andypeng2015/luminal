# gpt-oss-120b (MXFP4)

Runs [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) on luminal's CUDA
backend with the MoE experts kept in **MXFP4** and dequantized in-graph. Validated against
HuggingFace `transformers`: greedy decoding matches token-for-token.

```
cargo run --release -p gpt_oss
```

The weights (~61 GB, 15 shards) download from HuggingFace on first run.

## Architecture

Implements the full gpt-oss-120b model: 36 layers, hidden 2880, 64 query / 8 KV heads
(head_dim 64), 128 experts top-4, vocab 201088. Notable pieces vs. a vanilla MoE transformer:

- **MXFP4 experts** — `*_blocks` (U8 packed fp4) + `*_scales` (e8m0) + `*_bias`, gathered and
  unpacked in-graph (see `quant.rs`).
- **Attention sinks** — a per-head learned logit folded into the softmax denominator.
- **Alternating attention** — sliding window (128) on even layers, full on odd.
- **YaRN RoPE** — inv_freq + attention scaling computed on the host.
- **Interleaved clamped SwiGLU** — `gate = x[..., ::2]`, `up = x[..., 1::2]`, clamp ±7,
  `out = (up + 1) · gate · σ(1.702 · gate)`.
- **Top-4 routing** with softmax applied *after* topk; biases on q/k/v/o, router, experts.

## FP4 on non-Blackwell GPUs

`F4E2M1` / `F8UE8M0` conversion runs correctly on sm_90 (H100) via CUDA 12.8's header software
fallbacks — no Blackwell hardware required. `bin/phase0_cast.rs` verifies this:

```
cargo run --release -p gpt_oss --bin phase0_cast
```

## Memory

The full model is a **marginal** fit on an 80 GB GPU: ~63 GB resident MXFP4 weights + a ~10–14 GB
intermediate arena + cuBLASLt workspaces lands at ~77–80 GB. The arena allocation falls in
fragmented free space and currently succeeds only intermittently (~1 in 9 runs) — **if you hit a
`CUDA_ERROR_OUT_OF_MEMORY` at the arena alloc, just re-run.** When it allocates, the model runs to
completion and matches HuggingFace token-for-token. The robust fix (cross-layer arena reuse /
loop-rolling so the arena shrinks to ~1 layer) is tracked in LUM-645. On an H200 (141 GB) or
multi-GPU it fits with room to spare.

Env knobs:

- `GPTOSS_MEM_CAP_GIB` (default 14) — intermediate-buffer arena cap.
- `GPTOSS_LAYERS` (default 36) — cap the layer count to fit on smaller GPUs (output is only
  correct at 36; fewer layers is for bring-up/perf only).
- `LUMINAL_CUBLASLT_WORKSPACE_MB` (default set to 2 by this example) — per-matmul cuBLASLt
  workspace; the default 32 MB × ~200 matmuls would add ~6 GB.

Prefill is done one token at a time (s=1) to keep the expert-gather index tensors small.

## Validation harness

`../../../harness/golden.py` dumps the HF reference (logits + greedy continuation) for the same
harmony prompt; `../../../harness/dequant_ref.py` checks the MXFP4 unpack against the raw bytes.
