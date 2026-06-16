mod hf;
mod model;
mod quant;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use quant::fp4_byte_luts;
use rand::{SeedableRng, rngs::SmallRng};
use rustc_hash::FxHashSet;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;

const SEARCH_SEED: u64 = 0;

// Harmony special tokens (o200k_harmony).
const EOS_RETURN: u32 = 200002; // <|return|>
const END_TOKEN: u32 = 200007; // <|end|>

/// Minimal harmony-format chat prompt for a single user turn.
fn harmony_prompt(user_prompt: &str) -> String {
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Reasoning: low<|end|>\
         <|start|>user<|message|>{user_prompt}<|end|>\
         <|start|>assistant"
    )
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

fn main() {
    // The full 36-layer model peaks near the 80 GB H100 limit. The per-matmul
    // cuBLASLt workspaces (one per matmul node) dominate the overhead, so cap
    // them small by default unless the user overrides it.
    if std::env::var("LUMINAL_CUBLASLT_WORKSPACE_MB").is_err() {
        unsafe { std::env::set_var("LUMINAL_CUBLASLT_WORKSPACE_MB", "2") };
    }

    let max_seq_len = 1024;
    let gen_tokens = 24;
    let search_graphs = 1;
    let prompt = "What is the capital of France?";

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let (model_dir, shard_paths) = prepare_hf_model().expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let chat_prompt = harmony_prompt(prompt);
    let prompt_tokens = tokenizer
        .encode(chat_prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();

    // Host-computed YaRN rotary frequencies + attention scaling.
    let (inv_freq_vals, mscale) = yarn_inv_freq();

    // Build graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let model = GptOss::init(&mut cx, mscale);
    let inv_freq_id = model.rope_inv_freq();
    let (lut_lo_id, lut_hi_id) = model.fp4_luts();
    let (lut_lo_f32, lut_hi_f32) = fp4_byte_luts();
    let lut_lo_vals: Vec<half::bf16> = lut_lo_f32
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    let lut_hi_vals: Vec<half::bf16> = lut_hi_f32
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    let (logits, cache_outputs) = model.forward(input, pos_ids, &kv_cache);
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    // ~61 GB of weights stay resident, leaving little room for intermediates.
    // The expert-weight gather index tensors scale with the sequence length
    // (`[s, k, out, in]` Int), so a multi-token prefill bucket would need tens
    // of GB. Run the whole model at s=1 (token-by-token prefill) to keep the
    // per-layer transient near the ~13 GB dense expert dequant, which fits.
    let build_options = CompileOptions::default().dim_buckets('s', &[DimBucket::new(1, 1)]);
    let search_s = 1;

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>(build_options);

    println!("Loading weights ({} shards)...", shard_paths.len());
    // ~61 GB of packed MXFP4 weights stay resident, so cap the intermediate
    // arena to force per-layer buffer reuse (each layer's expert dequant +
    // gather is the transient peak; it is freed and reused by the next layer).
    let cap_gib: usize = std::env::var("GPTOSS_MEM_CAP_GIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(14);
    let mut runtime = CudaRuntime::initialize(stream).with_max_memory_gib(cap_gib);
    for p in &shard_paths {
        runtime.load_safetensors(&cx, p.to_str().unwrap());
    }
    runtime.set_data(inv_freq_id, inv_freq_vals.clone());
    runtime.set_data(lut_lo_id, lut_lo_vals.clone());
    runtime.set_data(lut_hi_id, lut_hi_vals.clone());

    let cache_bytes = N_KV_HEADS * max_seq_len * HEAD_DIM * std::mem::size_of::<f32>();
    for i in 0..kv_cache.k_caches.len() {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Compiling...");
    cx.set_dim('s', search_s);
    cx.set_dim('p', 0);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(pos_ids, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(inv_freq_id, inv_freq_vals.clone());
    runtime.set_data(lut_lo_id, lut_lo_vals.clone());
    runtime.set_data(lut_hi_id, lut_hi_vals.clone());
    let mut rng = SmallRng::seed_from_u64(SEARCH_SEED);
    let search_options = CompileOptions::default().search_graph_limit(search_graphs);
    runtime = cx.search_with_rng(runtime, search_options, &mut rng);

    runtime.set_data(inv_freq_id, inv_freq_vals.clone());
    runtime.set_data(lut_lo_id, lut_lo_vals.clone());
    runtime.set_data(lut_hi_id, lut_hi_vals.clone());
    for i in 0..kv_cache.k_caches.len() {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Prompt: {prompt}");
    print!("Response: ");
    std::io::stdout().flush().unwrap();

    let mut fwd_durations = vec![];
    let mut seen_tokens = FxHashSet::default();

    // Prefill, one token at a time (keeps the expert-gather transients tiny).
    let prefill_start = std::time::Instant::now();
    let mut prev_seq = 0usize;
    let mut last_logits: Vec<f32> = Vec::new();
    for &tok in &prompt_tokens {
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![tok as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);
        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
        }
        prev_seq += 1;
        last_logits = runtime.get_f32(logits);
    }
    let prefill_duration = prefill_start.elapsed();

    let mut next_token = argmax(&last_logits[..VOCAB_SIZE]);
    println!("\n[first-token argmax: {next_token}]");
    print!("{}", tokenizer.decode(&[next_token], false).unwrap());
    std::io::stdout().flush().unwrap();
    seen_tokens.insert(next_token);

    // Decode loop (greedy)
    for _ in 1..gen_tokens {
        let start = std::time::Instant::now();
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![next_token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);
        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
        }
        prev_seq += 1;

        let logits_data = runtime.get_f32(logits);
        next_token = argmax(&logits_data[..VOCAB_SIZE]);
        seen_tokens.insert(next_token);
        if next_token == EOS_RETURN || next_token == END_TOKEN {
            break;
        }
        print!("{}", tokenizer.decode(&[next_token], false).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    println!(
        "  TTFT: {:.2} ms ({} prompt tokens)",
        prefill_duration.as_secs_f64() * 1e3,
        prompt_tokens.len()
    );
    if fwd_durations.len() > 1 {
        println!(
            "  TPOT: {:.2} ms",
            (fwd_durations.iter().skip(1).sum::<Duration>() / (fwd_durations.len() - 1) as u32)
                .as_secs_f64()
                * 1e3
        );
    }
}
