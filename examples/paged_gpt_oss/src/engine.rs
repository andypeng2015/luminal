//! GPU serving engine: owns the compiled gpt-oss graph + runtime + the
//! continuous-batching scheduler, and runs one step at a time.
//!
//! The graph is compiled once with dim buckets that cover the whole serving
//! envelope — `s` in `1..=max_batch+max_prefill`, `c` in `1..=kv_capacity` — so
//! each step is just `set_dim` + `execute` (a cheap buffer refresh, no
//! recompile) as long as we stay within those bounds. The arena is sized for
//! the bucket maxima, so `max_batch * kv_capacity` is the memory knob.

use std::path::PathBuf;
use std::sync::Arc;

use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaStream, runtime::CudaRuntime};

use crate::batch::{argmax, build_batch, logits_row};
use crate::kv_alloc::SeqId;
use crate::model::{GptOss, KV_DIM, PagedKVCache, PagingInputs, yarn_inv_freq};
use crate::quant::fp4_byte_luts;
use crate::scheduler::{Finished, Request, Scheduler, SchedulerConfig};

pub struct EngineConfig {
    /// Max concurrent running sequences (decode batch ceiling).
    pub max_batch: usize,
    /// Max prompt tokens prefilled in one step.
    pub max_prefill: usize,
    /// Total KV slots (max total context across all live sequences).
    pub kv_capacity: usize,
    /// Search-time max-intermediate-memory cap (GiB).
    pub mem_cap_gib: usize,
}

struct Inputs {
    input: GraphTensor,
    pos_ids: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    mask_full: GraphTensor,
    mask_sliding: GraphTensor,
}

pub struct StepOutcome {
    /// (request id, token) emitted this step.
    pub emitted: Vec<(u64, u32)>,
    /// Requests that finished (or were rejected) this step.
    pub finished: Vec<Finished>,
    /// Whether a model step actually ran (false when idle).
    pub ran: bool,
}

pub struct Engine {
    cx: Graph,
    runtime: CudaRuntime,
    inp: Inputs,
    logits: GraphTensor,
    kv_cache: PagedKVCache,
    cache_outputs: Vec<(GraphTensor, GraphTensor)>,
    scheduler: Scheduler,
}

impl Engine {
    pub fn load(stream: Arc<CudaStream>, shard_paths: &[PathBuf], cfg: EngineConfig) -> Self {
        // Uniform 1-token-per-sequence steps: total_s == #running <= max_batch.
        let max_s = cfg.max_batch.max(1);

        // Anti-fragmentation reserve (as in the demo): grab a contiguous block
        // while VRAM is empty and free it right before the arena is allocated,
        // so the arena gets a clean hole instead of OOMing in the fragmented
        // free space left by the ~63 GB weight load. See LUM-645.
        let reserve_gib = std::env::var("PAGED_ARENA_RESERVE_GIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10usize);
        let mut arena_reserve =
            (reserve_gib > 0).then(|| stream.alloc_zeros::<u8>(reserve_gib << 30).unwrap());

        let (inv_freq_vals, mscale) = yarn_inv_freq();
        let (lut_lo_f32, lut_hi_f32) = fp4_byte_luts();
        let lut_lo_vals: Vec<half::bf16> =
            lut_lo_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let lut_hi_vals: Vec<half::bf16> =
            lut_hi_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();

        let mut cx = Graph::default();
        let inp = Inputs {
            input: cx.named_tensor("input", 's').as_dtype(DType::Int),
            pos_ids: cx.named_tensor("pos_ids", 's').as_dtype(DType::Int),
            scatter_idx: cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int),
            gather_idx: cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int),
            mask_full: cx.named_tensor("mask_full", ('s', 'c')),
            mask_sliding: cx.named_tensor("mask_sliding", ('s', 'c')),
        };
        let kv_cache = PagedKVCache::new(&mut cx, cfg.kv_capacity);
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

        // A SINGLE 's' bucket [1, max_batch] -> one compiled plan / one arena.
        // (Splitting decode vs prefill into two buckets like paged_llama gives
        // each its own ~13 GB arena, which doesn't fit beside 63 GB of weights;
        // the arena is nearly s-independent so one bucket covering 1..max_batch
        // is the same size and fits.) No 'c' bucket — a 'c' range bucket
        // triggered CUDA_ILLEGAL_ADDRESS; context grows via arena re-plan.
        // Also bucket 'c' so the arena is sized for the max context ONCE, up
        // front — otherwise context growth during decode re-plans (free+alloc+
        // sync) the arena every step, which dominates and erases the batching
        // win. The attention-over-context scratch is reused across layers, so a
        // large 'c' bucket only adds a little to the arena.
        let build_options = CompileOptions::default()
            .dim_buckets('s', &[DimBucket::new(1, max_s).representative(max_s)])
            .dim_buckets(
                'c',
                &[DimBucket::new(1, cfg.kv_capacity).representative(cfg.kv_capacity)],
            );
        println!("[engine] building search space (max_s={max_s}, kv={})...", cfg.kv_capacity);
        cx.build_search_space::<CudaRuntime>(build_options);

        let mut runtime =
            CudaRuntime::initialize(stream.clone()).with_max_memory_gib(cfg.mem_cap_gib);
        let set_consts = |rt: &mut CudaRuntime| {
            rt.set_data(inv_freq_id, inv_freq_vals.clone());
            rt.set_data(lut_lo_id, lut_lo_vals.clone());
            rt.set_data(lut_hi_id, lut_hi_vals.clone());
        };
        let cache_bytes = cfg.kv_capacity * KV_DIM * std::mem::size_of::<f32>();
        let zero_cache = |rt: &mut CudaRuntime| {
            for i in 0..kv_cache.k_caches.len() {
                rt.set_zeros(kv_cache.k_caches[i], cache_bytes);
                rt.set_zeros(kv_cache.v_caches[i], cache_bytes);
            }
        };
        // KV pool + consts before weights → contiguous post-weight region (LUM-645).
        set_consts(&mut runtime);
        zero_cache(&mut runtime);
        println!("[engine] loading {} weight shards...", shard_paths.len());
        for p in shard_paths {
            runtime.load_safetensors(&cx, p.to_str().unwrap());
        }

        // Valid dummy inputs for the search profile (s=1 decode shape, c=64 so
        // the initial arena covers short contexts without an immediate re-plan).
        let (s0, c0) = (1usize, 64.min(cfg.kv_capacity));
        cx.set_dim('s', s0);
        cx.set_dim('c', c0);
        runtime.set_data(inp.input, vec![1i32; s0]);
        runtime.set_data(inp.pos_ids, vec![0i32; s0]);
        runtime.set_data(inp.scatter_idx, (0..s0 as i32).collect::<Vec<_>>());
        runtime.set_data(inp.gather_idx, (0..c0 as i32).collect::<Vec<_>>());
        runtime.set_data(inp.mask_full, vec![0.0f32; s0 * c0]);
        runtime.set_data(inp.mask_sliding, vec![0.0f32; s0 * c0]);
        set_consts(&mut runtime);
        // Release the reserved hole right before search allocates the arena.
        drop(arena_reserve.take());
        stream.synchronize().ok();
        println!("[engine] searching...");
        runtime = cx.search(runtime, CompileOptions::default().search_graph_limit(1));
        set_consts(&mut runtime);
        zero_cache(&mut runtime);

        let scheduler = Scheduler::new(
            cfg.kv_capacity,
            SchedulerConfig {
                max_batch: cfg.max_batch,
                max_prefill: cfg.max_prefill,
            },
        );
        println!("[engine] ready");
        Self {
            cx,
            runtime,
            inp,
            logits,
            kv_cache,
            cache_outputs,
            scheduler,
        }
    }

    pub fn add_request(&mut self, req: Request) {
        self.scheduler.add_request(req);
    }

    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Run one continuous-batching step: assemble the batch, run the model,
    /// sample greedily, feed tokens back. Returns what was emitted/finished.
    pub fn step(&mut self) -> StepOutcome {
        let (plan, rejected) = self.scheduler.schedule();
        let Some(plan) = plan else {
            return StepOutcome {
                emitted: vec![],
                finished: rejected,
                ran: false,
            };
        };
        let batch = build_batch(&plan.entries, self.scheduler.allocator());

        // Env-gated per-step profiling (LUMINAL_ENGINE_PROFILE): break the
        // serving step into set_data / execute / get_logits / kv-roundtrip /
        // sample to see what dominates now that the MoE op is fused.
        let prof = std::env::var_os("LUMINAL_ENGINE_PROFILE").is_some();
        let mut mk = std::time::Instant::now();
        let mut lap = |on: bool| -> f64 {
            let d = mk.elapsed();
            mk = std::time::Instant::now();
            if on { d.as_secs_f64() * 1e3 } else { 0.0 }
        };

        self.runtime.set_data(self.inp.input, plan.tokens.clone());
        self.runtime.set_data(self.inp.pos_ids, batch.q_pos.clone());
        self.runtime.set_data(self.inp.scatter_idx, batch.scatter_idx.clone());
        self.runtime.set_data(self.inp.gather_idx, batch.gather_idx.clone());
        self.runtime.set_data(self.inp.mask_full, batch.mask_full.clone());
        self.runtime.set_data(self.inp.mask_sliding, batch.mask_sliding.clone());
        self.cx.set_dim('s', batch.total_s);
        self.cx.set_dim('c', batch.total_c);
        let t_set = lap(prof);
        self.runtime.execute(&self.cx.dyn_map);
        let t_exec = lap(prof);
        let all = self.runtime.get_f32(self.logits);
        let t_logits = lap(prof);

        // Round-trip KV cache: updated output buffers become next step's inputs.
        for (i, (k_out, v_out)) in self.cache_outputs.iter().enumerate() {
            let k_buf = self.runtime.remove_buffer(*k_out);
            let v_buf = self.runtime.remove_buffer(*v_out);
            self.runtime.set_buffer(self.kv_cache.k_caches[i], k_buf);
            self.runtime.set_buffer(self.kv_cache.v_caches[i], v_buf);
        }
        let t_kv = lap(prof);

        let sampled: Vec<(SeqId, u32)> = plan
            .samples
            .iter()
            .map(|&(row, seq)| (seq, argmax(logits_row(&all, row))))
            .collect();
        let t_sample = lap(prof);
        if prof {
            eprintln!(
                "ENGINE_PROF s={} c={} set={t_set:.2} exec={t_exec:.2} logits={t_logits:.2} kv={t_kv:.2} sample={t_sample:.2}",
                batch.total_s, batch.total_c
            );
        }
        let (emitted, mut finished) = self.scheduler.ingest(&sampled);
        finished.splice(0..0, rejected);
        StepOutcome {
            emitted,
            finished,
            ran: true,
        }
    }
}
