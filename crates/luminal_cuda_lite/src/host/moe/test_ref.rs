//! Host-side references shared by the MoE kernel and op tests: bf16 helpers,
//! FP4 dequant, the gate_up → SwiGLU → down → weighted-sum chain reference,
//! and the magnitude-aware tolerance check. Test-only (cfg(test) in mod.rs).
#![allow(clippy::needless_range_loop)] // reference code mirrors kernel indexing

/// bf16 helpers shared by the moe block tests.
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}
pub fn from_bf16_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}
pub fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&f| f32_to_bf16_bits(f).to_le_bytes())
        .collect()
}

pub const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub struct ChainWeights<'a> {
    pub gu_q: &'a [u8],
    pub gu_s: &'a [u8],
    pub gu_bias: &'a [f32],
    pub dn_q: &'a [u8],
    pub dn_s: &'a [u8],
    pub dn_bias: &'a [f32],
}

/// Host dequant of one weight element (row n, col k) of expert e.
pub fn host_weight(
    bq: &[u8],
    bs: &[u8],
    e: usize,
    n_dim: usize,
    k_dim: usize,
    n: usize,
    k: usize,
) -> f32 {
    let byte = bq[e * n_dim * (k_dim / 2) + n * (k_dim / 2) + k / 2];
    let nib = if k.is_multiple_of(2) {
        byte & 0xF
    } else {
        byte >> 4
    };
    let sc = bs[e * n_dim * (k_dim / 32) + n * (k_dim / 32) + k / 32];
    FP4_LUT[nib as usize] * (2.0f32).powi(sc as i32 - 127)
}

/// f32 end-to-end variant (no bf16 rounding): the reference for the
/// fused decode path, which keeps f32 internally. The bf16-rounding
/// variant below matches the tiled kernel's internal precision.
#[allow(clippy::too_many_arguments)]
pub fn host_chain_reference_f32(
    w: &ChainWeights<'_>,
    x: &[f32],
    topk_ids: &[i32],
    topk_w: &[f32],
    tokens: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let gate_up_n = 2 * inter;
    let mut want = vec![0.0f32; tokens * hidden];
    for t in 0..tokens {
        for slot in 0..top_k {
            let e = topk_ids[t * top_k + slot] as usize;
            let rw = topk_w[t * top_k + slot];
            let mut hid = vec![0.0f32; inter];
            for j in 0..inter {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for c in 0..hidden {
                    let xv = x[t * hidden + c];
                    gate += xv * host_weight(w.gu_q, w.gu_s, e, gate_up_n, hidden, 2 * j, c);
                    up += xv * host_weight(w.gu_q, w.gu_s, e, gate_up_n, hidden, 2 * j + 1, c);
                }
                gate = (gate + w.gu_bias[e * gate_up_n + 2 * j]).min(7.0);
                up = (up + w.gu_bias[e * gate_up_n + 2 * j + 1]).clamp(-7.0, 7.0);
                hid[j] = (up + 1.0) * gate / (1.0 + (-1.702f32 * gate).exp());
            }
            for r in 0..hidden {
                let mut dot = 0.0f32;
                for c in 0..inter {
                    dot += hid[c] * host_weight(w.dn_q, w.dn_s, e, hidden, inter, r, c);
                }
                want[t * hidden + r] += rw * (dot + w.dn_bias[e * hidden + r]);
            }
        }
    }
    want
}

/// Magnitude-aware comparison (bf16 accumulation-order variance makes
/// near-zero elements of large-scale outputs meaningless in pure relative
/// terms — the lesson from the earlier MoE test work).
/// bf16-rounding end-to-end variant: matches the tiled/grouped GEMM chain's
/// internal precision (bf16 A and intermediates, f32 accumulate). The f32
/// variant above is the reference for the fused decode path.
#[allow(clippy::too_many_arguments)]
pub fn host_chain_reference(
    w: &ChainWeights<'_>,
    x: &[f32],
    topk_ids: &[i32],
    topk_w: &[f32],
    tokens: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let gate_up_n = 2 * inter;
    let bf = |v: f32| bf16_bits_to_f32(f32_to_bf16_bits(v));
    let mut want = vec![0.0f32; tokens * hidden];
    for t in 0..tokens {
        for slot in 0..top_k {
            let e = topk_ids[t * top_k + slot] as usize;
            let rw = topk_w[t * top_k + slot];
            let mut gu = vec![0.0f32; gate_up_n];
            for (nn, g) in gu.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for kk in 0..hidden {
                    acc += bf(x[t * hidden + kk])
                        * host_weight(w.gu_q, w.gu_s, e, gate_up_n, hidden, nn, kk);
                }
                *g = bf(acc + bf(w.gu_bias[e * gate_up_n + nn]));
            }
            let mut hid = vec![0.0f32; inter];
            for (j, h) in hid.iter_mut().enumerate() {
                let gate = gu[2 * j].min(7.0);
                let up = gu[2 * j + 1].clamp(-7.0, 7.0);
                *h = bf((up + 1.0) * (gate / (1.0 + (-1.702f32 * gate).exp())));
            }
            for nn in 0..hidden {
                let mut acc = 0.0f32;
                for (kk, h) in hid.iter().enumerate() {
                    acc += h * host_weight(w.dn_q, w.dn_s, e, hidden, inter, nn, kk);
                }
                want[t * hidden + nn] += bf((acc + bf(w.dn_bias[e * hidden + nn])) * rw);
            }
        }
    }
    want
}

pub fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let floor = (scale * 0.02).max(1e-3);
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(floor);
        if rel > max_rel {
            max_rel = rel;
            worst = i;
        }
    }
    eprintln!(
        "{label}: max_rel={max_rel:.5} scale={scale:.1} (worst idx {worst}: got {} want {})",
        got[worst], want[worst]
    );
    assert!(max_rel < tol, "{label}: max_rel {max_rel} exceeds {tol}");
}

/// Deterministic LCG for test data — shared by every moe test module.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}
