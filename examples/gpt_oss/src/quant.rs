//! MXFP4 expert weights, gathered-then-unpacked in HLIR.
//!
//! gpt-oss-120b stores each MoE layer's experts in MXFP4, stacked over the
//! expert axis. Per projection (gate_up / down) three tensors are stored:
//!
//! | suffix     | on-disk dtype/shape           | declared here             |
//! |------------|-------------------------------|---------------------------|
//! | `_blocks`  | U8  (E, out, in/32, 16)       | U8      (E, out, in/2)    |
//! | `_scales`  | U8  (E, out, in/32)           | F8UE8M0 (E, out, in/32)   |
//! | `_bias`    | BF16 (E, out)                 | Bf16    (E, out)          |
//!
//! ## Why gather-first
//!
//! Only `top_k = 4` of the 128 experts are used per token, but dequantizing the
//! whole stacked `(E, out, in)` weight up front materializes ~15 GB per layer,
//! which does not fit alongside the ~61 GB of resident packed weights on an
//! 80 GB GPU. Instead we gather only the selected experts' **packed bytes**
//! (the expert axis is byte-aligned, so an ordinary gather is valid) and unpack
//! them in-graph. Per-layer transient then scales with `top_k`, not `E`.
//!
//! ## Unpacking without bitwise ops
//!
//! luminal has no tensor bitwise AND/shift, so the two fp4 nibbles in each byte
//! are decoded with a pair of 256-entry lookup tables (`lut_lo`, `lut_hi`)
//! computed on the host: `lut_lo[b] = fp4(b & 0xF)`, `lut_hi[b] = fp4(b >> 4)`.
//! Gathering both tables by the byte value and interleaving the results yields
//! the unpacked fp4 weights. The per-block e8m0 scale (`2^(byte-127)`) is just
//! `F8UE8M0 -> bf16`.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;

/// MXFP4 block size along the input dimension (OCP MX standard).
pub const MXFP4_BLOCK: usize = 32;

/// 16-entry FP4 E2M1 value table.
pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Host-built 256-entry byte -> low/high fp4 nibble value tables.
pub fn fp4_byte_luts() -> (Vec<f32>, Vec<f32>) {
    let mut lo = Vec::with_capacity(256);
    let mut hi = Vec::with_capacity(256);
    for b in 0u32..256 {
        lo.push(FP4_E2M1_LUT[(b & 0x0F) as usize]);
        hi.push(FP4_E2M1_LUT[((b >> 4) & 0x0F) as usize]);
    }
    (lo, hi)
}

/// Persistent handles for one layer's stacked MXFP4 expert projection.
pub struct Mxfp4Experts {
    /// Packed fp4 weights, declared `(E, out, in/2)`, dtype `U8`.
    pub blocks: GraphTensor,
    /// Per-block e8m0 scales, declared `(E, out, in/32)`, dtype `F8UE8M0`.
    pub scales: GraphTensor,
    /// Per-expert output bias, declared `(E, out)`, dtype `Bf16`.
    pub bias: GraphTensor,
}

impl Mxfp4Experts {
    /// `prefix` is the full tensor name without the suffix, e.g.
    /// `"model.layers.0.mlp.experts.gate_up_proj"`. `out_dim`/`in_dim` are the
    /// unpacked weight dims (`in_dim` must be a multiple of [`MXFP4_BLOCK`]).
    pub fn new(
        prefix: &str,
        num_experts: usize,
        out_dim: usize,
        in_dim: usize,
        cx: &mut Graph,
    ) -> Self {
        assert!(
            in_dim.is_multiple_of(MXFP4_BLOCK),
            "in_dim ({in_dim}) must be a multiple of MXFP4 block size ({MXFP4_BLOCK})",
        );
        Self {
            blocks: cx
                .named_tensor(
                    format!("{prefix}_blocks"),
                    (num_experts, out_dim, in_dim / 2),
                )
                .as_dtype(DType::U8)
                .persist(),
            scales: cx
                .named_tensor(
                    format!("{prefix}_scales"),
                    (num_experts, out_dim, in_dim / MXFP4_BLOCK),
                )
                .as_dtype(DType::F8UE8M0)
                .persist(),
            bias: cx
                .named_tensor(format!("{prefix}_bias"), (num_experts, out_dim))
                .as_dtype(DType::Bf16)
                .persist(),
        }
    }
}

/// Unpack already-gathered MXFP4 bytes into dense bf16 weights.
///
/// `bytes`  : gathered packed weights `(.., out, in/2)`, dtype `U8`.
/// `scales` : gathered e8m0 scales `(.., out, in/32)`, dtype `F8UE8M0`.
/// `lut_lo` / `lut_hi` : `[256]` bf16 byte -> nibble value tables.
/// Returns dense bf16 weights `(.., out, in)`.
pub fn unpack_mxfp4(
    bytes: GraphTensor,
    scales: GraphTensor,
    lut_lo: GraphTensor,
    lut_hi: GraphTensor,
) -> GraphTensor {
    let n = bytes.dims().len(); // last axis is in/2
    let g = bytes.cast(DType::Int); // byte values 0..256

    // Decode the two nibbles per byte via the lookup tables.
    let lo = lut_lo.gather(g); // (.., out, in/2) bf16
    let hi = lut_hi.gather(g);

    // Interleave: result[.., 2j] = lo[.., j], result[.., 2j+1] = hi[.., j].
    let inter = lo.unsqueeze(n).concat_along(hi.unsqueeze(n), n); // (.., out, in/2, 2)
    let w = inter.merge_dims(n - 1, n); // (.., out, in)

    // Broadcast each e8m0 block scale to MXFP4_BLOCK consecutive columns.
    // (Cast matches the LUT dtype so the multiply stays in one dtype.)
    let sc = scales
        .cast(lut_lo.dtype)
        .expand_dim(n, MXFP4_BLOCK)
        .merge_dims(n - 1, n); // (.., out, in)

    w * sc
}
