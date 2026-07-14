//! Dump the 1-layer HLIR egglog program (no GPU needed) for rewrite-rule
//! work. Run:
//!   cargo run --release -p gpt_oss --bin dump_egglog -- --layers 1 > dump.txt

#[path = "../hf.rs"]
mod hf;
#[path = "../model.rs"]
mod model;
#[path = "../quant.rs"]
mod quant;

use luminal::prelude::*;
use model::*;

fn main() {
    let max_seq_len = 2048;
    let args: Vec<String> = std::env::args().collect();
    let n_layers: usize = args
        .iter()
        .position(|a| a == "--layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let (_, mscale) = yarn_inv_freq();
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq_len, n_layers);
    let model = GptOss::init(&mut cx, mscale, n_layers);
    let (logits, cache_outputs) = model.forward(input, pos_ids, &kv_cache);
    logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }
    let (program, _root) = luminal::egglog_utils::hlir_to_egglog(&cx);
    println!("{program}");
}
