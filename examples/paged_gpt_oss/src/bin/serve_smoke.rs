//! Smoke test for the continuous-batching engine: drive two concurrent requests
//! and print their decoded outputs.
//!
//! With GPTOSS_LAYERS=8, request 1's first generated token must be 16809
//! (matches the demo's known-good first-token argmax for "capital of France").

use std::collections::HashMap;

use luminal_cuda_lite::cudarc::driver::CudaContext;
use paged_gpt_oss::{
    chat::harmony_prompt,
    engine::{Engine, EngineConfig},
    hf::prepare_hf_model,
    scheduler::{Request, SamplingParams},
};
use tokenizers::Tokenizer;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let (model_dir, shards) = prepare_hf_model().expect("Failed to prepare model");
    let tok = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let encode = |s: &str| -> Vec<i32> {
        tok.encode(s, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect()
    };

    let cfg = EngineConfig {
        max_batch: env_usize("MAX_BATCH", 4),
        max_prefill: env_usize("MAX_PREFILL", 64),
        kv_capacity: env_usize("KV_CAPACITY", 512),
        mem_cap_gib: env_usize("GPTOSS_MEM_CAP_GIB", 14),
    };
    let mut engine = Engine::load(stream, &shards, cfg);

    let prompts = [
        (1u64, "What is the capital of France?"),
        (2u64, "Name three primary colors."),
    ];
    let gen_tokens = env_usize("GEN_TOKENS", 24);
    for (id, p) in prompts {
        engine.add_request(Request {
            id,
            prompt: encode(&harmony_prompt(p)),
            params: SamplingParams {
                max_tokens: gen_tokens,
                ignore_eos: false,
            },
        });
    }

    let mut out: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut first: HashMap<u64, u32> = HashMap::new();
    let mut steps = 0;
    let t0 = std::time::Instant::now();
    while engine.has_work() {
        steps += 1;
        assert!(steps < 10_000, "runaway");
        let o = engine.step();
        for (id, t) in o.emitted {
            first.entry(id).or_insert(t);
            out.entry(id).or_default().push(t);
        }
        for f in &o.finished {
            println!("[finish] req {} {:?}", f.id, f.reason);
        }
    }
    println!("steps={steps} elapsed={:?}", t0.elapsed());
    for (id, p) in prompts {
        let ids = out.get(&id).cloned().unwrap_or_default();
        let text = tok.decode(&ids, false).unwrap_or_default();
        println!(
            "\n=== req {id} ({p}) | first_token={:?} | {} tokens ===\n{text}",
            first.get(&id),
            ids.len()
        );
    }
}
