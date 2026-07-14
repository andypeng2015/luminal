//! Library surface for the unpaged gpt-oss example so tests can build the
//! real model graph (the binaries use #[path] includes of the same files).

pub mod hf;
pub mod model;
pub mod quant;

#[cfg(test)]
mod search_space_tests {
    use crate::model::*;
    use luminal::prelude::*;
    use luminal_cuda_lite::runtime::CudaRuntime;

    /// The naive (dense reference) MoE spelling is only viable through the
    /// FusedMoE fusion — raw candidates materialize expert-dense tensors the
    /// memory filter rejects. Assert the fused_moe_rewrite rules fire on the
    /// real model graph.
    #[test]
    fn fused_moe_rule_fires_in_search_space() {
        let n_layers = 2;
        let (_, mscale) = yarn_inv_freq();
        let mut cx = Graph::default();
        let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
        let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
        let kv_cache = KVCache::new(&mut cx, 256, n_layers);
        let model = GptOss::init(&mut cx, mscale, n_layers);
        let (logits, cache_outputs) = model.forward(input, pos_ids, &kv_cache);
        logits.output();
        for (k_out, v_out) in &cache_outputs {
            k_out.output();
            v_out.output();
        }

        let options = CompileOptions::default().dim_buckets(
            's',
            &[DimBucket::new(1, 1), DimBucket::new(2, 64).representative(16)],
        );
        cx.build_search_space::<CudaRuntime>(options);

        let egraph = cx.egraph().expect("egraph missing");
        let count = |what: &str| {
            egraph
                .enodes
                .values()
                .filter(|(name, _)| name.contains(what))
                .count()
        };
        let fused_nodes = count("FusedMoE");
        eprintln!("FusedMoE enodes: {fused_nodes}");
        // NB: egglog sort names, not Rust struct names (cublaslt is lowercase)
        eprintln!("cublaslt enodes: {}", count("cublaslt"));
        eprintln!("GenericMatmul enodes: {}", count("GenericMatmul"));
        assert!(
            fused_nodes > 0,
            "fused_moe_rewrite did not fire on the real model graph"
        );
    }
}
