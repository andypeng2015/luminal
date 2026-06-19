use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use paged_gpt_oss::hf::prepare_hf_model;
use paged_gpt_oss::model::*;
use paged_gpt_oss::quant::fp4_byte_luts;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;

// Harmony special tokens (o200k_harmony).
const EOS_RETURN: u32 = 200002; // <|return|>
const END_TOKEN: u32 = 200007; // <|end|>

/// Minimal harmony-format chat prompt for a single user turn (matches gpt_oss).
fn harmony_prompt(user_prompt: &str) -> String {
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Reasoning: low<|end|>\
         <|start|>user<|message|>{user_prompt}<|end|>\
         <|start|>assistant"
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ─── Page table (CPU-side slot manager, per paged_llama) ───

struct PageTable {
    tables: Vec<Vec<usize>>,
    next_free_slot: usize,
}

impl PageTable {
    fn new() -> Self {
        Self {
            tables: vec![],
            next_free_slot: 0,
        }
    }
    fn new_sequence(&mut self) -> usize {
        let id = self.tables.len();
        self.tables.push(vec![]);
        id
    }
    fn allocate(&mut self, seq_id: usize, n: usize) {
        let slots: Vec<usize> = (self.next_free_slot..self.next_free_slot + n).collect();
        self.next_free_slot += n;
        self.tables[seq_id].extend_from_slice(&slots);
    }
    fn context_slots(&self, seq_id: usize) -> &[usize] {
        &self.tables[seq_id]
    }
    fn context_len(&self, seq_id: usize) -> usize {
        self.tables[seq_id].len()
    }
}

/// Per-batch host tensors: scatter/gather slot indices, query positions, and the
/// two additive masks (full causal, and causal+sliding-window). Slots for a
/// sequence are allocated in position order, so a context slot's index within
/// its sequence equals that token's absolute position.
struct Batch {
    scatter_idx: Vec<i32>,
    gather_idx: Vec<i32>,
    q_pos: Vec<i32>,
    mask_full: Vec<f32>,
    mask_sliding: Vec<f32>,
    total_s: usize,
    total_c: usize,
}

fn build_batch(entries: &[(usize, Vec<usize>)], page_table: &PageTable) -> Batch {
    let total_s: usize = entries.iter().map(|(_, pos)| pos.len()).sum();

    let mut gather_idx: Vec<i32> = vec![];
    let mut ctx_ranges: Vec<(usize, usize)> = vec![];
    for (seq_id, _) in entries {
        let start = gather_idx.len();
        let slots = page_table.context_slots(*seq_id);
        gather_idx.extend(slots.iter().map(|&s| s as i32));
        ctx_ranges.push((start, slots.len()));
    }
    let total_c = gather_idx.len();

    let mut scatter_idx: Vec<i32> = vec![];
    let mut q_pos: Vec<i32> = vec![];
    for (seq_id, positions) in entries {
        let ctx_len = page_table.context_len(*seq_id);
        let n_new = positions.len();
        let slots = page_table.context_slots(*seq_id);
        scatter_idx.extend(slots[ctx_len - n_new..].iter().map(|&s| s as i32));
        q_pos.extend(positions.iter().map(|&p| p as i32));
    }

    // Masks default to -1e30 (blocked); a query attends only within its own
    // sequence's context range (cross-sequence isolation), causally, and — for
    // the sliding mask — within the window.
    let mut mask_full = vec![-1e30f32; total_s * total_c];
    let mut mask_sliding = vec![-1e30f32; total_s * total_c];
    let mut q_offset = 0;
    for (entry_idx, (_, positions)) in entries.iter().enumerate() {
        let (ctx_start, ctx_len) = ctx_ranges[entry_idx];
        for (qi, &abs_pos) in positions.iter().enumerate() {
            for ci in 0..ctx_len {
                if ci <= abs_pos {
                    let idx = (q_offset + qi) * total_c + (ctx_start + ci);
                    mask_full[idx] = 0.0;
                    if abs_pos - ci < SLIDING_WINDOW {
                        mask_sliding[idx] = 0.0;
                    }
                }
            }
        }
        q_offset += positions.len();
    }

    Batch {
        scatter_idx,
        gather_idx,
        q_pos,
        mask_full,
        mask_sliding,
        total_s,
        total_c,
    }
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

fn logits_row(all_logits: &[f32], row_idx: usize) -> &[f32] {
    &all_logits[row_idx * VOCAB_SIZE..(row_idx + 1) * VOCAB_SIZE]
}

struct Inputs {
    input: GraphTensor,
    pos_ids: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    mask_full: GraphTensor,
    mask_sliding: GraphTensor,
}

#[allow(clippy::too_many_arguments)]
fn run(
    cx: &mut Graph,
    runtime: &mut CudaRuntime,
    inp: &Inputs,
    logits: GraphTensor,
    kv_cache: &PagedKVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
    tokens: Vec<i32>,
    batch: &Batch,
) -> Vec<f32> {
    runtime.set_data(inp.input, tokens);
    runtime.set_data(inp.pos_ids, batch.q_pos.clone());
    runtime.set_data(inp.scatter_idx, batch.scatter_idx.clone());
    runtime.set_data(inp.gather_idx, batch.gather_idx.clone());
    runtime.set_data(inp.mask_full, batch.mask_full.clone());
    runtime.set_data(inp.mask_sliding, batch.mask_sliding.clone());
    cx.set_dim('s', batch.total_s);
    cx.set_dim('c', batch.total_c);
    runtime.execute(&cx.dyn_map);
    let all = runtime.get_f32(logits);
    // Round-trip KV cache: feed updated buffers back as inputs for the next step.
    for (i, (k_out, v_out)) in cache_outputs.iter().enumerate() {
        let k_buf = runtime.remove_buffer(*k_out);
        let v_buf = runtime.remove_buffer(*v_out);
        runtime.set_buffer(kv_cache.k_caches[i], k_buf);
        runtime.set_buffer(kv_cache.v_caches[i], v_buf);
    }
    all[..batch.total_s * VOCAB_SIZE].to_vec()
}

fn main() {
    // Small per-matmul cuBLASLt workspace by default (one per matmul node; the
    // 32 MiB default × ~200 matmuls would add ~6 GB). See gpt_oss README.
    if std::env::var("LUMINAL_CUBLASLT_WORKSPACE_MB").is_err() {
        unsafe { std::env::set_var("LUMINAL_CUBLASLT_WORKSPACE_MB", "2") };
    }

    let num_slots = env_usize("NUM_SLOTS", 4096);
    let gen_tokens = env_usize("GEN_TOKENS", 24);
    let prompt_a = "What is the capital of France?";
    let prompt_b = "Name three primary colors.";

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Anti-fragmentation reserve. The intermediate arena (~7-8 GiB at s=1, 36
    // layers) is allocated lazily during the search below, *after* the ~63 GB
    // weight load has churned and fragmented free VRAM — so even with ~17 GiB
    // free there may be no contiguous block big enough and the arena's
    // cudaMalloc OOMs (a fragmentation lottery). Reserve a contiguous block now,
    // while VRAM is empty, and free it just before the arena is allocated so the
    // arena reuses that clean hole. The real fix is a pooled allocator /
    // cross-layer arena reuse (LUM-645). Set PAGED_ARENA_RESERVE_GIB=0 to disable.
    let reserve_gib = env_usize("PAGED_ARENA_RESERVE_GIB", 10);
    let mut arena_reserve =
        (reserve_gib > 0).then(|| stream.alloc_zeros::<u8>(reserve_gib << 30).unwrap());

    let (model_dir, shard_paths) = prepare_hf_model().expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let encode = |p: &str| -> Vec<u32> {
        tokenizer
            .encode(harmony_prompt(p).as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec()
    };
    let tokens_a = encode(prompt_a);
    let tokens_b = encode(prompt_b);
    println!(
        "Prompt A: {} tokens | Prompt B: {} tokens",
        tokens_a.len(),
        tokens_b.len()
    );

    let (inv_freq_vals, mscale) = yarn_inv_freq();
    let (lut_lo_f32, lut_hi_f32) = fp4_byte_luts();
    let lut_lo_vals: Vec<half::bf16> = lut_lo_f32
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    let lut_hi_vals: Vec<half::bf16> = lut_hi_f32
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();

    // ─── Build graph ───
    let mut cx = Graph::default();
    let inp = Inputs {
        input: cx.named_tensor("input", 's').as_dtype(DType::Int),
        pos_ids: cx.named_tensor("pos_ids", 's').as_dtype(DType::Int),
        scatter_idx: cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int),
        gather_idx: cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int),
        mask_full: cx.named_tensor("mask_full", ('s', 'c')),
        mask_sliding: cx.named_tensor("mask_sliding", ('s', 'c')),
    };
    let kv_cache = PagedKVCache::new(&mut cx, num_slots);
    let model = GptOss::init(&mut cx, mscale);
    let inv_freq_id = model.rope_inv_freq();
    let (lut_lo_id, lut_hi_id) = model.fp4_luts();
    let paging = PagingInputs {
        pos_ids: inp.pos_ids,
        scatter_idx: inp.scatter_idx,
        gather_idx: inp.gather_idx,
        mask_full: inp.mask_full,
        mask_sliding: inp.mask_sliding,
    };
    let (logits, cache_outputs) = model.forward(inp.input, paging, &kv_cache);
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    // gpt-oss's MoE expert-gather memory scales with `s`, so (unlike paged_llama)
    // we keep `s` small: prefill is done token-by-token (s=1) and the multi-
    // sequence batching is shown as an s=2 supersequence decode. The s=2 bucket
    // roughly doubles the s=1 working set, which does not fit beside ~63 GB of
    // weights at full 36 layers, so the batching demo is gated by PAGED_BATCH
    // (default on; set PAGED_BATCH=0 for a full-model single-sequence run).
    let batch_demo = env_usize("PAGED_BATCH", 1) == 1;
    let buckets: &[DimBucket] = if batch_demo {
        &[DimBucket::new(1, 1), DimBucket::new(2, 2)]
    } else {
        &[DimBucket::new(1, 1)]
    };
    let build_options = CompileOptions::default().dim_buckets('s', buckets);
    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>(build_options);

    println!("Loading weights ({} shards)...", shard_paths.len());
    let cap_gib = env_usize("GPTOSS_MEM_CAP_GIB", 14);
    let mut runtime = CudaRuntime::initialize(stream.clone()).with_max_memory_gib(cap_gib);

    let set_consts = |rt: &mut CudaRuntime| {
        rt.set_data(inv_freq_id, inv_freq_vals.clone());
        rt.set_data(lut_lo_id, lut_lo_vals.clone());
        rt.set_data(lut_hi_id, lut_hi_vals.clone());
    };
    let cache_bytes = num_slots * KV_DIM * std::mem::size_of::<f32>();
    let zero_cache = |rt: &mut CudaRuntime| {
        for i in 0..kv_cache.k_caches.len() {
            rt.set_zeros(kv_cache.k_caches[i], cache_bytes);
            rt.set_zeros(kv_cache.v_caches[i], cache_bytes);
        }
    };

    // Allocate the persistent KV-cache pool and constant buffers BEFORE loading
    // the weights. They then sit "below" the weights in VRAM, leaving the large
    // post-weight free region contiguous for the intermediate arena + CUDA-graph
    // capture. Doing this *after* the weight load (the obvious order) instead
    // splinters that region, so the ~7.7 GiB arena's cudaMalloc OOMs even with
    // ~18 GiB free — this is the one structural difference from the dense
    // gpt_oss example, which has no separate KV buffers and fits. See LUM-645.
    set_consts(&mut runtime);
    zero_cache(&mut runtime);

    for p in &shard_paths {
        runtime.load_safetensors(&cx, p.to_str().unwrap());
    }

    println!("Compiling...");
    // Compile the context dim large enough to cover prompt + generation so the
    // intermediate arena is allocated once (no mid-prefill realloc).
    let (ss, sc) = (if batch_demo { 2 } else { 1 }, env_usize("COMPILE_C", 64));
    cx.set_dim('s', ss);
    cx.set_dim('c', sc);
    runtime.set_data(inp.input, vec![1i32; ss]);
    runtime.set_data(inp.pos_ids, vec![0i32; ss]);
    runtime.set_data(inp.scatter_idx, (0..ss as i32).collect::<Vec<_>>());
    runtime.set_data(inp.gather_idx, (0..sc as i32).collect::<Vec<_>>());
    runtime.set_data(inp.mask_full, vec![0.0f32; ss * sc]);
    runtime.set_data(inp.mask_sliding, vec![0.0f32; ss * sc]);
    set_consts(&mut runtime);
    // Free the reserved contiguous block right before the arena is allocated
    // (during search) so the arena reuses this clean hole instead of OOMing in
    // the fragmented free space left by the weight load.
    drop(arena_reserve.take());
    stream.synchronize().unwrap();
    let search_options = CompileOptions::default().search_graph_limit(1);
    runtime = cx.search(runtime, search_options);
    set_consts(&mut runtime);
    zero_cache(&mut runtime);

    let mut page_table = PageTable::new();

    // Prefill a sequence one token at a time into the page table's slots,
    // returning the argmax of the final token's logits.
    let prefill = |cx: &mut Graph,
                   rt: &mut CudaRuntime,
                   pt: &mut PageTable,
                   seq: usize,
                   toks: &[u32]|
     -> u32 {
        let mut last = 0u32;
        for &tok in toks {
            let pos = pt.context_len(seq);
            pt.allocate(seq, 1);
            let batch = build_batch(&[(seq, vec![pos])], pt);
            let lg = run(
                cx,
                rt,
                &inp,
                logits,
                &kv_cache,
                &cache_outputs,
                vec![tok as i32],
                &batch,
            );
            last = argmax(logits_row(&lg, 0));
        }
        last
    };

    // ═══ Phase 1: prefill sequence A (token by token) ═══
    println!("\n══ Phase 1: prefill A ({} tokens) ══", tokens_a.len());
    let seq_a = page_table.new_sequence();
    let t = std::time::Instant::now();
    let mut next_a = prefill(&mut cx, &mut runtime, &mut page_table, seq_a, &tokens_a);
    println!(
        "  prefill {:.0} ms | [A] first token {next_a} {:?}",
        t.elapsed().as_secs_f64() * 1e3,
        tokenizer.decode(&[next_a], false).unwrap()
    );

    // ═══ Phase 2: decode A ═══
    println!("\n══ Phase 2: decode A ══\n[A] ");
    let mut decode_times = vec![];
    let mut a_ids = vec![next_a];
    for _ in 0..gen_tokens {
        if next_a == EOS_RETURN || next_a == END_TOKEN {
            break;
        }
        let t = std::time::Instant::now();
        let pos = page_table.context_len(seq_a);
        page_table.allocate(seq_a, 1);
        let batch = build_batch(&[(seq_a, vec![pos])], &page_table);
        let lg = run(
            &mut cx,
            &mut runtime,
            &inp,
            logits,
            &kv_cache,
            &cache_outputs,
            vec![next_a as i32],
            &batch,
        );
        decode_times.push(t.elapsed());
        next_a = argmax(logits_row(&lg, 0));
        a_ids.push(next_a);
        print!("{}", tokenizer.decode(&[next_a], false).unwrap());
        std::io::stdout().flush().unwrap();
    }
    println!();
    println!("A_DECODE_IDS: {a_ids:?}");
    if decode_times.len() > 1 {
        let avg = decode_times.iter().skip(1).sum::<Duration>() / (decode_times.len() - 1) as u32;
        println!("  Avg TPOT: {:.1} ms", avg.as_secs_f64() * 1e3);
    }

    // ═══ Phase 3+4: multi-sequence batching (gated by PAGED_BATCH) ═══
    if batch_demo {
        // Phase 3: add sequence B (prefill token by token)
        println!(
            "\n══ Phase 3: prefill B ({} tokens), sharing the slot pool ══",
            tokens_b.len()
        );
        let seq_b = page_table.new_sequence();
        let mut next_b = prefill(&mut cx, &mut runtime, &mut page_table, seq_b, &tokens_b);
        println!(
            "  [B] first token {:?}",
            tokenizer.decode(&[next_b], false).unwrap()
        );

        // ═══ Phase 4: supersequence decode (A + B together, s=2) ═══
        // Both sequences decode in one batched step from the shared slot pool; the
        // precomputed masks keep each query attending only to its own sequence.
        println!("\n══ Phase 4: supersequence decode A+B ══");
        let mut text_a = String::new();
        let mut text_b = String::new();
        let mut a_super_ids = vec![];
        for _ in 0..gen_tokens {
            let a_done = next_a == EOS_RETURN || next_a == END_TOKEN;
            let b_done = next_b == EOS_RETURN || next_b == END_TOKEN;
            if a_done && b_done {
                break;
            }
            let pa = page_table.context_len(seq_a);
            let pb = page_table.context_len(seq_b);
            page_table.allocate(seq_a, 1);
            page_table.allocate(seq_b, 1);
            let batch = build_batch(&[(seq_a, vec![pa]), (seq_b, vec![pb])], &page_table);
            let lg = run(
                &mut cx,
                &mut runtime,
                &inp,
                logits,
                &kv_cache,
                &cache_outputs,
                vec![next_a as i32, next_b as i32],
                &batch,
            );
            next_a = argmax(logits_row(&lg, 0));
            next_b = argmax(logits_row(&lg, 1));
            if !a_done {
                a_super_ids.push(next_a);
                text_a += &tokenizer.decode(&[next_a], false).unwrap();
            }
            if !b_done {
                text_b += &tokenizer.decode(&[next_b], false).unwrap();
            }
        }
        println!("[A] ...{text_a}");
        println!("[B] ...{text_b}");
        println!("A_SUPER_IDS: {a_super_ids:?}");
    } // end batch_demo

    println!(
        "\nPage table: {} / {num_slots} slots used",
        page_table.next_free_slot
    );
}
