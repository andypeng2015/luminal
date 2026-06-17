//! HuggingFace download helpers for openai/gpt-oss-120b.
//!
//! Unlike `qwen3_moe`, we do **not** combine or pre-dequantize anything. The
//! MoE experts are stored as MXFP4 (`*_blocks` U8 packed fp4 + `*_scales` U8
//! e8m0) already stacked over the expert axis, and every other weight is BF16.
//! luminal's `load_safetensors` copies raw bytes by name, and our declared
//! logical dtypes (`F4E2M1` / `F8UE8M0` / `Bf16`) have matching byte counts, so
//! we download the original shards and load them directly (flux2-style).

use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf};

pub const REPO_ID: &str = "openai/gpt-oss-120b";

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Download (or resolve from cache) the tokenizer and all safetensors shards.
/// Returns `(model_dir, shard_paths)` where `shard_paths` are in shard order.
pub fn prepare_hf_model() -> Result<(PathBuf, Vec<PathBuf>), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(REPO_ID.to_string());

    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();

    let index_path = repo.get("model.safetensors.index.json")?;
    let raw = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&raw)?;

    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    let mut shard_paths = Vec::with_capacity(shard_files.len());
    for f in &shard_files {
        shard_paths.push(repo.get(f)?);
    }
    Ok((model_dir, shard_paths))
}
