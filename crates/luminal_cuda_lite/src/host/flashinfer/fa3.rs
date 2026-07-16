//! FA3/Hopper (SM90) AttentionSink paged batch-prefill — rung one of the
//! gpt-oss attention ladder.
//!
//! Ladder (mirrors the FusedMoE build): (1) launch the target kernel from a
//! Rust test against a standalone CPU oracle ← THIS MODULE, (2) egglog rule,
//! (3) host op, (4) full model. The kernel is FlashInfer's
//! `BatchPrefillWithPagedKVCacheDispatched` from
//! `attention/hopper/prefill_sm90.cuh`, instantiated with the upstream
//! `AttentionSink` variant in `wrapper_fa3.cu` and JIT-compiled by
//! `jit::ensure_compiled_fa3` (-arch=sm_90a).
//!
//! Decode is the same kernel at qo_len=1 — there is no SM90 decode-with-sink
//! kernel; upstream's own sink tests decode through the paged prefill path.
//!
//! What the tests establish that upstream does NOT cover: head_dim 64 (their
//! sink tests are head_dim-128-only) and GQA group_size 8 (they test 1 and 4)
//! — i.e. exactly the gpt-oss-120b shapes.

#[cfg(test)]
mod tests {
    use super::super::jit;
    use super::super::test_ref::{
        RefSeq, assert_close, bf16_bytes_to_f32, deterministic_f32, reference_attention,
        round_to_bf16, to_bf16_bytes,
    };
    use super::super::{
        FLOAT_WORKSPACE_SIZE, INT_WORKSPACE_SIZE, PAGE_LOCKED_WORKSPACE, PageLockedPtr,
        cuda_pin_memory,
    };
    use crate::cudarc::driver::{CudaContext, DevicePtr};

    /// Serializes flashinfer GPU tests: they share the process-wide
    /// page-locked plan scratch (PAGE_LOCKED_WORKSPACE), which concurrent
    /// plan() calls would corrupt (the engine is single-threaded; cargo test
    /// is not). One lock across BOTH test modules — see flashinfer/mod.rs.
    use super::super::TEST_LOCK;

    const HEAD_DIM: usize = 64;
    const DTYPE_BF16: i32 = 2;
    /// bf16 tolerances (same values as the crate's existing FA2 bf16 tests).
    const BF16_RTOL: f32 = 3e-2;
    const BF16_ATOL: f32 = 3e-3;
    /// "No sink": exp(-1e30 - m) == 0 for any finite row max, recovering the
    /// standard softmax — the harness-validation trick.
    const SINKLESS: f32 = -1e30;

    /// gpt-oss layers alternate window_left = 127 (even) and full (odd).
    const GPTOSS_WINDOW_LEFT: i64 = 127;

    struct Case<'a> {
        label: &'a str,
        seqs: &'a [RefSeq],
        num_qo_heads: usize,
        num_kv_heads: usize,
        window_left: i64,
        /// Per-qo-head sink logits.
        sinks: Vec<f32>,
        /// Shuffle logical KV tokens across physical slots (real paging).
        scatter_pages: bool,
        seed: u64,
    }

    fn hopper_gpu() -> bool {
        if CudaContext::new(0).is_err() {
            return false;
        }
        crate::device_compute_major() == 9
    }

    /// Run one case through plan + run and compare against the CPU oracle.
    fn run_case(case: &Case) {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        let total_qo: usize = case.seqs.iter().map(|s| s.qo_len).sum();
        let total_kv: usize = case.seqs.iter().map(|s| s.kv_len).sum();
        let (nq, nkv) = (case.num_qo_heads, case.num_kv_heads);
        let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // bf16-rounded operands so oracle and kernel see identical inputs.
        let q = round_to_bf16(&deterministic_f32(total_qo * nq * HEAD_DIM, case.seed));
        let k = round_to_bf16(&deterministic_f32(total_kv * nkv * HEAD_DIM, case.seed + 1));
        let v = round_to_bf16(&deterministic_f32(total_kv * nkv * HEAD_DIM, case.seed + 2));

        let want = reference_attention(
            &q,
            &k,
            &v,
            &case.sinks,
            case.seqs,
            nq,
            nkv,
            HEAD_DIM,
            case.window_left,
            sm_scale,
        );

        // ── Paged KV pool (page_size = 1: one token per page/slot) ──
        // Logical token i lives at physical slot perm[i]; kv_indices carries
        // the per-token page ids in logical order.
        let perm: Vec<usize> = if case.scatter_pages {
            // Deterministic Fisher-Yates.
            let mut p: Vec<usize> = (0..total_kv).collect();
            let mut state = case.seed.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(99);
            for i in (1..total_kv).rev() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = ((state >> 33) as usize) % (i + 1);
                p.swap(i, j);
            }
            p
        } else {
            (0..total_kv).collect()
        };
        let row = nkv * HEAD_DIM;
        let mut k_pool = vec![0.0f32; total_kv * row];
        let mut v_pool = vec![0.0f32; total_kv * row];
        for logical in 0..total_kv {
            let phys = perm[logical];
            k_pool[phys * row..(phys + 1) * row].copy_from_slice(&k[logical * row..][..row]);
            v_pool[phys * row..(phys + 1) * row].copy_from_slice(&v[logical * row..][..row]);
        }
        let kv_indices_host: Vec<i32> = perm.iter().map(|&p| p as i32).collect();

        // Host plan arrays: qo/kv indptrs (kv counts pages == tokens here).
        // Mutable because the C plan ABI takes *mut (it only reads them).
        let mut qo_indptr_h: Vec<i32> = vec![0];
        let mut kv_indptr_h: Vec<i32> = vec![0];
        let mut kv_len_arr_h: Vec<i32> = Vec::new();
        for s in case.seqs {
            qo_indptr_h.push(qo_indptr_h.last().unwrap() + s.qo_len as i32);
            kv_indptr_h.push(kv_indptr_h.last().unwrap() + s.kv_len as i32);
            kv_len_arr_h.push(s.kv_len as i32);
        }

        // ── Device buffers ──
        let d_q = stream.clone_htod(&to_bf16_bytes(&q)).unwrap();
        let d_k = stream.clone_htod(&to_bf16_bytes(&k_pool)).unwrap();
        let d_v = stream.clone_htod(&to_bf16_bytes(&v_pool)).unwrap();
        let d_kv_indices = stream.clone_htod(&kv_indices_host).unwrap();
        let d_sinks = stream.clone_htod(&case.sinks).unwrap();
        let d_out = stream
            .clone_htod(&vec![0u8; total_qo * nq * HEAD_DIM * 2])
            .unwrap();

        let float_ws = unsafe { stream.alloc::<u8>(FLOAT_WORKSPACE_SIZE).unwrap() };
        let int_ws = unsafe { stream.alloc::<u8>(INT_WORKSPACE_SIZE).unwrap() };
        let page_locked = PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
            assert_eq!(status, 0, "Failed to allocate page-locked workspace");
            let cuda_status = cuda_pin_memory(ptr, INT_WORKSPACE_SIZE);
            assert_eq!(cuda_status, 0, "Failed to pin memory");
            PageLockedPtr(ptr as *mut u8)
        });

        // ── Plan + run ──
        let lib = jit::ensure_compiled_fa3(HEAD_DIM, case.window_left >= 0);
        let mut plan_info = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = unsafe {
            (lib.prefill_plan)(
                float_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                page_locked.0 as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                qo_indptr_h.as_mut_ptr(),
                kv_indptr_h.as_mut_ptr(),
                kv_len_arr_h.as_mut_ptr(),
                total_qo as i32,
                case.seqs.len() as i32,
                nq as i32,
                nkv as i32,
                /*page_size=*/ 1,
                /*causal=*/ 1,
                cu_stream,
                plan_info.as_mut_ptr(),
                &mut plan_info_len,
            )
        };
        assert_eq!(plan_ret, 0, "{}: fa3 plan failed", case.label);

        let run_ret = unsafe {
            (lib.prefill_run)(
                int_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info_len,
                d_q.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_k.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_v.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_kv_indices.device_ptr(&stream).0 as *mut i32,
                d_sinks.device_ptr(&stream).0 as *mut f32,
                d_out.device_ptr(&stream).0 as *mut std::ffi::c_void,
                std::ptr::null_mut(), // lse
                total_qo as i32,
                nq as i32,
                nkv as i32,
                /*page_size=*/ 1,
                DTYPE_BF16,
                sm_scale,
                case.window_left as i32,
                /*causal=*/ 1,
                cu_stream,
            )
        };
        assert_eq!(run_ret, 0, "{}: fa3 run failed", case.label);

        stream.synchronize().unwrap();
        let got = bf16_bytes_to_f32(&stream.clone_dtoh(&d_out).unwrap());
        assert_close(&got, &want, BF16_RTOL, BF16_ATOL, case.label);
    }

    fn uniform_sinks(n: usize, base: f32) -> Vec<f32> {
        // Distinct per-head values so a head-indexing bug can't cancel out.
        (0..n).map(|h| base + h as f32 * 0.13).collect()
    }

    /// 1. Harness validation: sinks at -1e30 reduce the sink softmax to the
    ///    standard one, so any mismatch here is plan/run plumbing, not sink math.
    #[test]
    fn fa3_harness_sinkless_single_seq() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sinkless single-seq",
            seqs: &[RefSeq {
                qo_len: 4,
                kv_len: 8,
            }],
            num_qo_heads: 8,
            num_kv_heads: 8,
            window_left: -1,
            sinks: vec![SINKLESS; 8],
            scatter_pages: false,
            seed: 11,
        });
    }

    /// 2. Real sinks, gpt-oss head config (64 Q / 8 KV, group 8), prefill.
    #[test]
    fn fa3_sink_prefill_single_seq() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sink prefill 64/8",
            seqs: &[RefSeq {
                qo_len: 16,
                kv_len: 48,
            }],
            num_qo_heads: 64,
            num_kv_heads: 8,
            window_left: -1,
            sinks: uniform_sinks(64, 0.5),
            scatter_pages: false,
            seed: 22,
        });
    }

    /// 3. Decode shape: qo_len=1 per sequence, mixed context lengths.
    #[test]
    fn fa3_sink_decode_batch() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sink decode batch",
            seqs: &[
                RefSeq {
                    qo_len: 1,
                    kv_len: 7,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 33,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 128,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 250,
                },
            ],
            num_qo_heads: 64,
            num_kv_heads: 8,
            window_left: -1,
            sinks: uniform_sinks(64, 1.0),
            scatter_pages: false,
            seed: 33,
        });
    }

    /// 4. Ragged batch: prefill chunks and decodes in one launch (the mixed
    ///    tick shape the engine produces).
    #[test]
    fn fa3_sink_ragged_batch() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sink ragged batch",
            seqs: &[
                RefSeq {
                    qo_len: 1,
                    kv_len: 61,
                },
                RefSeq {
                    qo_len: 16,
                    kv_len: 80,
                },
                RefSeq {
                    qo_len: 5,
                    kv_len: 5,
                },
            ],
            num_qo_heads: 64,
            num_kv_heads: 8,
            window_left: -1,
            sinks: uniform_sinks(64, 0.8),
            scatter_pages: false,
            seed: 44,
        });
    }

    /// 5a. Sliding window (gpt-oss even layers): kv_len well past the window
    ///     so truncation actually bites.
    #[test]
    fn fa3_sink_sliding_window() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sink sliding-window 127",
            seqs: &[
                RefSeq {
                    qo_len: 4,
                    kv_len: 300,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 200,
                },
            ],
            num_qo_heads: 64,
            num_kv_heads: 8,
            window_left: GPTOSS_WINDOW_LEFT,
            sinks: uniform_sinks(64, 0.5),
            scatter_pages: false,
            seed: 55,
        });
    }

    /// 5b. Full-attention twin of 5a (same shapes/seed, no window) — pins the
    ///     two .so variants apart.
    #[test]
    fn fa3_sink_full_attention_twin() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sink full-attn twin",
            seqs: &[
                RefSeq {
                    qo_len: 4,
                    kv_len: 300,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 200,
                },
            ],
            num_qo_heads: 64,
            num_kv_heads: 8,
            window_left: -1,
            sinks: uniform_sinks(64, 0.5),
            scatter_pages: false,
            seed: 55,
        });
    }

    /// 6. Scattered pages: shuffled physical slots (the slot-pool reality).
    #[test]
    fn fa3_sink_scattered_pages() {
        if !hopper_gpu() {
            return;
        }
        run_case(&Case {
            label: "sink scattered pages",
            seqs: &[
                RefSeq {
                    qo_len: 8,
                    kv_len: 90,
                },
                RefSeq {
                    qo_len: 1,
                    kv_len: 140,
                },
            ],
            num_qo_heads: 64,
            num_kv_heads: 8,
            window_left: -1,
            sinks: uniform_sinks(64, 0.7),
            scatter_pages: true,
            seed: 66,
        });
    }

    /// 7. Timing at gpt-oss serving shapes (FA3 sink path vs the FA2 default
    ///    wrapper on the identical workload). Informational; run explicitly:
    ///    `cargo test -p luminal_cuda_lite fa3_bench_real_dims -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark, run explicitly"]
    fn fa3_bench_real_dims() {
        if !hopper_gpu() {
            return;
        }
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        // 64-way decode tick over 1k contexts — the c=64 1k/1k serving shape.
        let batch = 64usize;
        let kv_len = 1024usize;
        let (nq, nkv) = (64usize, 8usize);
        let total_qo = batch;
        let total_kv = batch * kv_len;
        let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let q = deterministic_f32(total_qo * nq * HEAD_DIM, 7);
        let kv = deterministic_f32(total_kv * nkv * HEAD_DIM, 8);
        let sinks = uniform_sinks(nq, 0.5);

        let d_q = stream.clone_htod(&to_bf16_bytes(&q)).unwrap();
        let d_k = stream.clone_htod(&to_bf16_bytes(&kv)).unwrap();
        let d_v = stream.clone_htod(&to_bf16_bytes(&kv)).unwrap();
        let kv_indices_host: Vec<i32> = (0..total_kv as i32).collect();
        let d_kv_indices = stream.clone_htod(&kv_indices_host).unwrap();
        let d_sinks = stream.clone_htod(&sinks).unwrap();
        let d_out = stream
            .clone_htod(&vec![0u8; total_qo * nq * HEAD_DIM * 2])
            .unwrap();

        let float_ws = unsafe { stream.alloc::<u8>(FLOAT_WORKSPACE_SIZE).unwrap() };
        let int_ws = unsafe { stream.alloc::<u8>(INT_WORKSPACE_SIZE).unwrap() };
        let page_locked = PAGE_LOCKED_WORKSPACE.get_or_init(|| unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = libc::posix_memalign(&mut ptr, 4096, INT_WORKSPACE_SIZE);
            assert_eq!(status, 0);
            assert_eq!(cuda_pin_memory(ptr, INT_WORKSPACE_SIZE), 0);
            PageLockedPtr(ptr as *mut u8)
        });

        let mut qo_indptr_h: Vec<i32> = (0..=batch as i32).collect();
        let mut kv_indptr_h: Vec<i32> = (0..=batch as i32).map(|i| i * kv_len as i32).collect();
        let mut kv_len_arr_h: Vec<i32> = vec![kv_len as i32; batch];

        // ── FA3 sink path ──
        let lib = jit::ensure_compiled_fa3(HEAD_DIM, false);
        let mut plan_info = [0i64; 16];
        let mut plan_info_len: i32 = 0;
        let plan_ret = unsafe {
            (lib.prefill_plan)(
                float_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                page_locked.0 as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                qo_indptr_h.as_mut_ptr(),
                kv_indptr_h.as_mut_ptr(),
                kv_len_arr_h.as_mut_ptr(),
                total_qo as i32,
                batch as i32,
                nq as i32,
                nkv as i32,
                1,
                1,
                cu_stream,
                plan_info.as_mut_ptr(),
                &mut plan_info_len,
            )
        };
        assert_eq!(plan_ret, 0, "bench: fa3 plan failed");

        let fa3_run = || unsafe {
            let ret = (lib.prefill_run)(
                int_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                plan_info.as_ptr() as *mut i64,
                plan_info_len,
                d_q.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_k.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_v.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_kv_indices.device_ptr(&stream).0 as *mut i32,
                d_sinks.device_ptr(&stream).0 as *mut f32,
                d_out.device_ptr(&stream).0 as *mut std::ffi::c_void,
                std::ptr::null_mut(),
                total_qo as i32,
                nq as i32,
                nkv as i32,
                1,
                DTYPE_BF16,
                sm_scale,
                -1,
                1,
                cu_stream,
            );
            assert_eq!(ret, 0, "bench: fa3 run failed");
        };
        for _ in 0..5 {
            fa3_run();
        }
        stream.synchronize().unwrap();
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            fa3_run();
        }
        stream.synchronize().unwrap();
        let fa3_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // ── FA2 default wrapper on the identical workload (no sinks — the
        // FA2 wrapper has no sink support; this is a kernel-cost comparison,
        // not a numerics one) ──
        let fa2 = jit::ensure_compiled(HEAD_DIM, false);
        let d_qo_indptr = stream.clone_htod(&qo_indptr_h).unwrap();
        let d_kv_indptr = stream.clone_htod(&kv_indptr_h).unwrap();
        let d_last_page_len = stream.clone_htod(&vec![1i32; batch]).unwrap();
        let mut fa2_plan_info = [0i64; 16];
        let mut fa2_plan_info_len: i32 = 0;
        let plan_ret = unsafe {
            (fa2.prefill_plan)(
                float_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                page_locked.0 as *mut std::ffi::c_void,
                qo_indptr_h.as_mut_ptr(),
                kv_indptr_h.as_mut_ptr(),
                total_qo as i32,
                batch as i32,
                nq as i32,
                nkv as i32,
                1,
                HEAD_DIM as i32,
                DTYPE_BF16,
                -1,
                cu_stream,
                fa2_plan_info.as_mut_ptr(),
                &mut fa2_plan_info_len,
            )
        };
        assert_eq!(plan_ret, 0, "bench: fa2 plan failed");
        let fa2_run = || unsafe {
            let ret = (fa2.prefill_run)(
                float_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                FLOAT_WORKSPACE_SIZE,
                int_ws.device_ptr(&stream).0 as *mut std::ffi::c_void,
                fa2_plan_info.as_ptr() as *mut i64,
                fa2_plan_info_len,
                d_q.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_k.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_v.device_ptr(&stream).0 as *mut std::ffi::c_void,
                d_qo_indptr.device_ptr(&stream).0 as *mut i32,
                d_kv_indptr.device_ptr(&stream).0 as *mut i32,
                d_kv_indices.device_ptr(&stream).0 as *mut i32,
                d_last_page_len.device_ptr(&stream).0 as *mut i32,
                d_out.device_ptr(&stream).0 as *mut std::ffi::c_void,
                total_qo as i32,
                batch as i32,
                nq as i32,
                nkv as i32,
                1,
                HEAD_DIM as i32,
                DTYPE_BF16,
                sm_scale,
                -1,
                cu_stream,
            );
            assert_eq!(ret, 0, "bench: fa2 run failed");
        };
        for _ in 0..5 {
            fa2_run();
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            fa2_run();
        }
        stream.synchronize().unwrap();
        let fa2_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        eprintln!(
            "decode tick batch={batch} kv={kv_len} heads {nq}/{nkv} hd{HEAD_DIM}: \
             FA3+sink {fa3_us:.1} us/layer, FA2 (sinkless) {fa2_us:.1} us/layer"
        );
    }

    /// Split-KV merge kernel in isolation: ragged partials per sequence
    /// combined in base-2 LSE space with the REAL sink joining the
    /// denominator exactly once, output written in (heads, B, dim) F32 graph
    /// layout. CPU oracle does the identical combine in f64. Sinks alternate
    /// dominant (+8, exp2 term rules the denominator) and negligible (-20)
    /// per head so both max branches are exercised.
    #[test]
    fn fa3_merge_sink_unit() {
        if !hopper_gpu() {
            eprintln!("skipping fa3_merge_sink_unit: needs Hopper GPU");
            return;
        }
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;
        let lib = jit::ensure_compiled_fa3(HEAD_DIM, false);

        let (b, h, d) = (3usize, 64usize, HEAD_DIM);
        let merge_indptr: Vec<i32> = vec![0, 2, 5, 7];
        let p_total = 7usize;
        // Partial LSEs in [-6, 6] (base-2 domain); partial outputs are
        // normalized attention rows, so bf16 values in [-1, 1].
        let plse: Vec<f32> = deterministic_f32(p_total * h, 11)
            .iter()
            .map(|x| x * 6.0)
            .collect();
        let pv = round_to_bf16(&deterministic_f32(p_total * h * d, 12));
        let sinks: Vec<f32> = (0..h)
            .map(|i| if i % 2 == 0 { 8.0 } else { -20.0 })
            .collect();

        // f64 CPU oracle of the exact combine the kernel documents.
        let mut want = vec![0.0f32; h * b * d];
        for bi in 0..b {
            for hi in 0..h {
                let (p0, p1) = (merge_indptr[bi] as usize, merge_indptr[bi + 1] as usize);
                let log_sink = sinks[hi] as f64 * std::f64::consts::LOG2_E;
                let mut m = log_sink;
                for p in p0..p1 {
                    m = m.max(plse[p * h + hi] as f64);
                }
                let mut den = (log_sink - m).exp2();
                let mut num = vec![0.0f64; d];
                for p in p0..p1 {
                    let w = ((plse[p * h + hi] as f64) - m).exp2();
                    den += w;
                    for (di, acc) in num.iter_mut().enumerate() {
                        *acc += w * pv[(p * h + hi) * d + di] as f64;
                    }
                }
                for di in 0..d {
                    want[(hi * b + bi) * d + di] = (num[di] / den) as f32;
                }
            }
        }

        let d_pv = stream.clone_htod(&to_bf16_bytes(&pv)).unwrap();
        let d_plse = stream.clone_htod(&plse).unwrap();
        let d_indptr = stream.clone_htod(&merge_indptr).unwrap();
        let d_sinks = stream.clone_htod(&sinks).unwrap();
        let d_out = stream.clone_htod(&vec![0u8; h * b * d * 4]).unwrap();
        let ret = unsafe {
            (lib.merge_sink_f32)(
                d_pv.device_ptr(&stream).0 as *const std::ffi::c_void,
                d_plse.device_ptr(&stream).0 as *const f32,
                d_indptr.device_ptr(&stream).0 as *const i32,
                d_sinks.device_ptr(&stream).0 as *const f32,
                d_out.device_ptr(&stream).0 as *mut std::ffi::c_void,
                b as i32,
                h as i32,
                d as i32,
                cu_stream,
            )
        };
        assert_eq!(ret, 0, "merge_sink launch failed");
        stream.synchronize().unwrap();
        let got: Vec<f32> = stream
            .clone_dtoh(&d_out)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_close(&got, &want, BF16_RTOL, BF16_ATOL, "merge sink unit");
    }
}
