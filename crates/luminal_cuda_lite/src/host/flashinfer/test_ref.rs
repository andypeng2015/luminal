//! Standalone CPU reference for gpt-oss-style attention: GQA, causal,
//! optional left sliding window, and per-head attention sinks that join the
//! softmax denominator (never the V accumulation). The oracle for the FA3
//! rung-one kernel tests (same role as `moe/test_ref.rs` for the MoE ladder).
//!
//! All math in f32. Inputs are expected to already be rounded to bf16 so the
//! oracle and the kernel see identical operands; the remaining divergence is
//! accumulation order, which the magnitude-aware `assert_close` absorbs.

#![allow(clippy::needless_range_loop)]

/// One batched sequence: `qo_len` new query rows attending over `kv_len`
/// context tokens (the last `qo_len` of which are the queries themselves —
/// query j sits at absolute position `kv_len - qo_len + j`).
#[derive(Clone, Copy, Debug)]
pub struct RefSeq {
    pub qo_len: usize,
    pub kv_len: usize,
}

/// Reference attention over a batch of sequences.
///
/// - `q`: `[total_qo, num_qo_heads, head_dim]` (concatenated per sequence)
/// - `k`/`v`: `[total_kv, num_kv_heads, head_dim]` in LOGICAL per-sequence
///   order (the test resolves paging before calling the oracle)
/// - `sinks`: `[num_qo_heads]` sink logits (pass ~-1e30 for sinkless softmax)
/// - `window_left`: visible previous positions (+self); -1 = unlimited
///
/// Returns `[total_qo, num_qo_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: &[f32],
    seqs: &[RefSeq],
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_left: i64,
    sm_scale: f32,
) -> Vec<f32> {
    let total_qo: usize = seqs.iter().map(|s| s.qo_len).sum();
    let total_kv: usize = seqs.iter().map(|s| s.kv_len).sum();
    assert_eq!(q.len(), total_qo * num_qo_heads * head_dim, "q size");
    assert_eq!(k.len(), total_kv * num_kv_heads * head_dim, "k size");
    assert_eq!(v.len(), total_kv * num_kv_heads * head_dim, "v size");
    assert_eq!(sinks.len(), num_qo_heads, "sinks size");
    let group_size = num_qo_heads / num_kv_heads;

    let mut out = vec![0.0f32; total_qo * num_qo_heads * head_dim];
    let mut qo_base = 0usize;
    let mut kv_base = 0usize;
    for seq in seqs {
        for j in 0..seq.qo_len {
            let abs_pos = seq.kv_len - seq.qo_len + j;
            for h in 0..num_qo_heads {
                let kv_h = h / group_size;
                let q_row = &q[((qo_base + j) * num_qo_heads + h) * head_dim..][..head_dim];

                // Scores over the visible context window.
                let lo = if window_left >= 0 {
                    abs_pos.saturating_sub(window_left as usize)
                } else {
                    0
                };
                let mut scores = Vec::with_capacity(abs_pos + 1 - lo);
                for i in lo..=abs_pos {
                    let k_row = &k[((kv_base + i) * num_kv_heads + kv_h) * head_dim..][..head_dim];
                    let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                    scores.push(dot * sm_scale);
                }

                // Sink-augmented softmax: the sink logit joins the row max and
                // the denominator but never the V accumulation.
                let sink = sinks[h];
                let m = scores.iter().copied().fold(sink, f32::max);
                let mut denom = (sink - m).exp();
                let probs: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        let p = (s - m).exp();
                        denom += p;
                        p
                    })
                    .collect();

                let o_row = &mut out[((qo_base + j) * num_qo_heads + h) * head_dim..][..head_dim];
                for (idx, &p) in probs.iter().enumerate() {
                    let i = lo + idx;
                    let v_row = &v[((kv_base + i) * num_kv_heads + kv_h) * head_dim..][..head_dim];
                    let w = p / denom;
                    for d in 0..head_dim {
                        o_row[d] += w * v_row[d];
                    }
                }
            }
        }
        qo_base += seq.qo_len;
        kv_base += seq.kv_len;
    }
    out
}

/// Deterministic pseudo-random values in roughly [-0.5, 0.5] (LCG — fixed
/// seeds per AGENTS.md test guidance; no rand dep).
pub fn deterministic_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 % 1000) as f32 / 1000.0 - 0.5
        })
        .collect()
}

/// Round f32 values through bf16 so the CPU oracle sees exactly the operands
/// the kernel reads.
pub fn round_to_bf16(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|&x| half::bf16::from_f32(x).to_f32())
        .collect()
}

pub fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// rtol/atol comparison (numpy-style: fail when `|g-w| > atol + rtol*|w|`).
/// The atol matters for attention outputs: FlashAttention-family kernels
/// round the probability matrix to bf16 before the P·V tensor-core matmul,
/// so near-zero output elements carry ~1e-4 absolute noise that a purely
/// relative check misreads as error. Tolerances follow the crate's existing
/// bf16 FlashInfer tests (RTOL 3e-2, ATOL 3e-3).
pub fn assert_close(got: &[f32], want: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut max_ratio = 0.0f32;
    let mut worst = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ratio = (g - w).abs() / (atol + rtol * w.abs());
        if ratio > max_ratio {
            max_ratio = ratio;
            worst = i;
        }
    }
    eprintln!(
        "{label}: max violation ratio={max_ratio:.4} (worst idx {worst}: got {} want {}, diff {:.2e})",
        got[worst],
        want[worst],
        (got[worst] - want[worst]).abs()
    );
    assert!(
        max_ratio <= 1.0,
        "{label}: |got-want| exceeds atol({atol}) + rtol({rtol})*|want| by {max_ratio}x at idx {worst}"
    );
}
