//! Library surface for the paged gpt-oss serving engine. The `main.rs` demo and
//! the `serve` binary share these modules.

pub mod hf;
pub mod model;
pub mod quant;

pub mod batch;
pub mod chat;
pub mod engine;
pub mod kv_alloc;
pub mod scheduler;
pub mod server;

#[cfg(test)]
mod search_space_tests {
    use luminal::prelude::*;
    use luminal_cuda_lite::runtime::CudaRuntime;

    use crate::model::{GptOss, PagedKVCache, PagingInputs, yarn_inv_freq};

    /// The FlashInfer sink rule must fire on the REAL model graph (not just
    /// the test replica in luminal_cuda_lite): the saturated search-space
    /// egraph must contain FlashInferSinkAttention enodes, and the search
    /// must see more than a handful of choice sets.
    #[test]
    fn sink_attention_in_real_model_search_space() {
        // SAFETY: test-local env, single-threaded use of the model constructor.
        // Honor an externally-set GPTOSS_LAYERS (e.g. 8 to reproduce the rolled
        // loop body); default to 2.
        if std::env::var_os("GPTOSS_LAYERS").is_none() {
            unsafe { std::env::set_var("GPTOSS_LAYERS", "2") };
        }

        let (_, mscale) = yarn_inv_freq();
        let mut cx = Graph::default();
        let inp_input = cx.named_tensor("input", 's').as_dtype(DType::Int);
        let paging = PagingInputs {
            pos_ids: cx.named_tensor("pos_ids", 's').as_dtype(DType::Int),
            scatter_idx: cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int),
            gather_idx: cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int),
            qo_indptr: cx.named_tensor("qo_indptr", 'r').as_dtype(DType::Int),
            kv_indptr: cx.named_tensor("kv_indptr", 'r').as_dtype(DType::Int),
        };
        let kv_capacity: usize = std::env::var("TEST_KV_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let max_s: usize = std::env::var("TEST_MAX_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(72);
        let kv_cache = PagedKVCache::new(&mut cx, kv_capacity);
        let model = GptOss::init(&mut cx, mscale);
        let (logits, cache_outputs) = model.forward(inp_input, paging, &kv_cache);
        logits.output();
        for (k_out, v_out) in &cache_outputs {
            k_out.output();
            v_out.output();
        }

        let options = CompileOptions::default()
            .dim_buckets('s', &[DimBucket::new(1, max_s).representative(max_s)])
            .dim_buckets(
                'c',
                &[DimBucket::new(1, kv_capacity).representative(kv_capacity)],
            )
            .dim_buckets('r', &[DimBucket::new(2, 10).representative(10)]);
        cx.build_search_space::<CudaRuntime>(options);

        let egraph = cx.egraph().expect("egraph missing");
        let sink_nodes = egraph
            .enodes
            .values()
            .filter(|(name, _)| name.contains("FlashInferSinkAttention"))
            .count();
        let choice_sets = luminal::egglog_utils::count_choice_sets_up_to(egraph, 10_000);
        eprintln!("sink enodes: {sink_nodes}, choice sets (capped 10k): {choice_sets}");
        assert!(
            sink_nodes > 0,
            "FlashInferSinkAttention rule did not fire on the real model graph"
        );

        // With the subsume in the island rules, EVERY extractable genome must
        // route attention through the sink op — extract one and check.
        use rand::SeedableRng;
        let ops = cx.egglog_ops().expect("ops missing");
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let genome = luminal::egglog_utils::random_initial_choice(egraph, &mut rng);
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = luminal::egglog_utils::egglog_to_llir(
            egraph,
            genome,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        let mut names: std::collections::BTreeMap<String, usize> = Default::default();
        for n in llir.node_indices() {
            use luminal_cuda_lite::host::HostOp;
            let name = llir[n]
                .to_dialect::<dyn HostOp>()
                .and_then(|op| op.stats_name())
                .unwrap_or("other");
            *names.entry(name.to_string()).or_default() += 1;
        }
        // Informational only: with union-only rules (matching the other
        // FlashInfer models), whether a single random genome routes through
        // the island is probabilistic — the search samples hundreds.
        eprintln!("extracted op composition: {names:?}");
    }
}
