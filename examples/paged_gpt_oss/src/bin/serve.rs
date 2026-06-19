//! OpenAI-compatible inference server for paged gpt-oss-120b (MXFP4) on luminal.
//!
//! Run:  PORT=8000 KV_CAPACITY=2048 MAX_BATCH=16 cargo run --release -p paged_gpt_oss --bin serve
//! Then point InferenceX's benchmark_serving.py --backend openai at it.

use std::sync::Arc;

use luminal_cuda_lite::cudarc::driver::CudaContext;
use paged_gpt_oss::{
    engine::EngineConfig,
    hf::prepare_hf_model,
    server::{AppState, router, spawn_engine},
};
use tokenizers::Tokenizer;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[tokio::main]
async fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let (model_dir, shards) = prepare_hf_model().expect("Failed to prepare model");
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());

    let cfg = EngineConfig {
        max_batch: env_usize("MAX_BATCH", 16),
        max_prefill: env_usize("MAX_PREFILL", 512),
        kv_capacity: env_usize("KV_CAPACITY", 4096),
        mem_cap_gib: env_usize("GPTOSS_MEM_CAP_GIB", 14),
    };
    println!(
        "[serve] config: max_batch={} max_prefill={} kv_capacity={}",
        cfg.max_batch, cfg.max_prefill, cfg.kv_capacity
    );
    let engine = spawn_engine(stream, shards, cfg);

    let state = AppState {
        engine,
        tokenizer,
        model: std::env::var("MODEL_NAME").unwrap_or_else(|_| "openai/gpt-oss-120b".into()),
        default_max_tokens: env_usize("DEFAULT_MAX_TOKENS", 128),
    };
    let app = router(state);

    let port = env_usize("PORT", 8000);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("[serve] listening on http://{addr} (loading model in background; /health is 503 until ready)");
    axum::serve(listener, app).await.unwrap();
}
