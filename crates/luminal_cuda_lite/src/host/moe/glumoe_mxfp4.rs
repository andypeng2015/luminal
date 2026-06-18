//! Fused MXFP4 grouped-MoE HostOp for gpt-oss.
//!
//! Like [`super::GLUMoE`], but the expert weights stay **MXFP4-packed** in
//! resident memory and a single gathered expert slice is dequantized inside the
//! op (into a reused scratch buffer) right before each per-expert cuBLASLt
//! matmul. This avoids materializing the `[s, k, gate_up_dim, hidden]` /
//! `[s, k, hidden, intermediate]` dequantized weight tensors in the captured
//! graph arena (the ~17 GB blowup; see LUM-645). It also implements gpt-oss's
//! clamped interleaved SwiGLU and the per-expert gate_up / down biases.
//!
//! Inputs (graph edges, in order):
//!   0: x               [seq, hidden]                 F32
//!   1: topk_indices    [seq, k]                      Int
//!   2: topk_values     [seq, k]                      F32
//!   3: gate_up_blocks  [E, gate_up_dim, hidden/2]    U8   (MXFP4 packed)
//!   4: gate_up_scales  [E, gate_up_dim, hidden/32]   U8   (e8m0)
//!   5: gate_up_bias    [E, gate_up_dim]              BF16
//!   6: down_blocks     [E, hidden, intermediate/2]   U8   (MXFP4 packed)
//!   7: down_scales     [E, hidden, intermediate/32]  U8   (e8m0)
//!   8: down_bias       [E, hidden]                   BF16
//!
//! Output: [seq, hidden] F32. `gate_up_dim = 2 * intermediate` (interleaved
//! gate/up lanes). top-k weights are applied directly (no normalization).

use std::sync::{Arc, OnceLock};

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::{
    compile_module_image_for_current_device,
    cudarc::{
        cublaslt::CudaBlasLT,
        driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    },
    host::{DeviceBuffer, HostOp},
    try_create_cublaslt,
};

use super::{MoeDims, WORKSPACE_SIZE, buf_ptr, read_routing, run_moe_experts, slice_ptr};

const MXFP4_BLOCK: usize = 32;
// Used only by the reference activation in the unit test (the kernel inlines
// these as literals in its NVRTC source).
#[cfg(test)]
const SWIGLU_LIMIT: f32 = 7.0;
#[cfg(test)]
const SWIGLU_ALPHA: f32 = 1.702;

pub struct GLUMoEMXFP4 {
    /// Hidden dim H (= gate_up matmul K, = down matmul output rows).
    hidden: Expression,
    /// Expert intermediate dim I (= down matmul K). gate_up_dim = 2*I.
    intermediate: Expression,
    /// Number of experts summed per token (top_k).
    output_k: Expression,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
    /// (module, f32_to_bf16, dequant_mxfp4_expert, glu_clamped_interleaved_bf16, accumulate_bias_scaled_f32)
    module: OnceLock<(
        Arc<CudaModule>,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
    )>,
}

impl Default for GLUMoEMXFP4 {
    fn default() -> Self {
        Self {
            hidden: Expression::default(),
            intermediate: Expression::default(),
            output_k: Expression::default(),
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }
}

impl std::fmt::Debug for GLUMoEMXFP4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLUMoEMXFP4")
            .field("hidden", &self.hidden)
            .field("intermediate", &self.intermediate)
            .field("output_k", &self.output_k)
            .finish()
    }
}

impl Clone for GLUMoEMXFP4 {
    fn clone(&self) -> Self {
        Self {
            hidden: self.hidden,
            intermediate: self.intermediate,
            output_k: self.output_k,
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }
}

impl GLUMoEMXFP4 {
    pub(crate) fn new(hidden: Expression, intermediate: Expression, output_k: Expression) -> Self {
        Self {
            hidden,
            intermediate,
            output_k,
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }

    fn get_cublaslt(&self, stream: &Arc<CudaStream>) -> anyhow::Result<Arc<CudaBlasLT>> {
        if let Some(cublaslt) = self.cublaslt.get() {
            return Ok(cublaslt.clone());
        }
        let created = try_create_cublaslt(stream.clone()).map_err(|message| {
            anyhow::anyhow!("cuBLASLt unavailable on this machine: {message}")
        })?;
        let _ = self.cublaslt.set(created.clone());
        Ok(created)
    }

    #[allow(clippy::type_complexity)]
    fn get_kernels(
        &self,
        stream: &Arc<CudaStream>,
    ) -> &(
        Arc<CudaModule>,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
    ) {
        self.module.get_or_init(|| {
            let src = r#"
#include <cuda_bf16.h>

extern "C" __global__ void f32_to_bf16(unsigned long long in_ptr, unsigned long long out_ptr, int n) {
    const float* in_ = (const float*)in_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in_[i]);
}

// Dequantize ONE expert's MXFP4 weight slice into bf16.
//   blocks: u8 [out_dim, in_dim/2]  (lo nibble -> even col, hi nibble -> odd col)
//   scales: u8 [out_dim, in_dim/32] (e8m0: value = 2^(byte-127))
//   out:    bf16 [out_dim, in_dim]
extern "C" __global__ void dequant_mxfp4_expert(
    unsigned long long blocks_ptr,
    unsigned long long scales_ptr,
    unsigned long long out_ptr,
    int out_dim,
    int in_dim
) {
    const float LUT[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                           -0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    long total = (long)out_dim * (long)in_dim;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int r = (int)(idx / in_dim);
    int c = (int)(idx % in_dim);
    const unsigned char* blocks = (const unsigned char*)blocks_ptr;
    const unsigned char* scales = (const unsigned char*)scales_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int in_half = in_dim >> 1;
    int in_blk = in_dim >> 5;
    unsigned char byte = blocks[(long)r * in_half + (c >> 1)];
    int nib = (c & 1) ? (byte >> 4) : (byte & 0x0F);
    unsigned char sb = scales[(long)r * in_blk + (c >> 5)];
    float scale = exp2f((float)((int)sb - 127));
    out[idx] = __float2bfloat16(LUT[nib] * scale);
}

// gpt-oss clamped interleaved SwiGLU, with the per-expert gate_up bias folded in.
//   gate_up: bf16 [2*intermediate], interleaved: gate = [2i], up = [2i+1]
//   bias:    bf16 [2*intermediate], same interleaving
//   out:     bf16 [intermediate]
extern "C" __global__ void glu_clamped_interleaved_bf16(
    unsigned long long gate_up_ptr,
    unsigned long long bias_ptr,
    unsigned long long out_ptr,
    int intermediate
) {
    const __nv_bfloat16* gate_up = (const __nv_bfloat16*)gate_up_ptr;
    const __nv_bfloat16* bias = (const __nv_bfloat16*)bias_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate) {
        float gate = __bfloat162float(gate_up[2*i])   + __bfloat162float(bias[2*i]);
        float up   = __bfloat162float(gate_up[2*i+1]) + __bfloat162float(bias[2*i+1]);
        gate = fminf(gate, 7.0f);                       // clamp(max=limit)
        up   = fminf(fmaxf(up, -7.0f), 7.0f);           // clamp(-limit, limit)
        float glu = gate / (1.0f + expf(-1.702f * gate));
        out[i] = __float2bfloat16((up + 1.0f) * glu);
    }
}

// out[i] += scale * bias[i]   (f32 accumulator, bf16 bias)
extern "C" __global__ void accumulate_bias_scaled_f32(
    unsigned long long out_ptr,
    unsigned long long bias_ptr,
    int n,
    float scale
) {
    float* out = (float*)out_ptr;
    const __nv_bfloat16* bias = (const __nv_bfloat16*)bias_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += scale * __bfloat162float(bias[i]);
}
"#;
            let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let f32_to_bf16 = module.load_function("f32_to_bf16").unwrap();
            let dequant = module.load_function("dequant_mxfp4_expert").unwrap();
            let activation = module.load_function("glu_clamped_interleaved_bf16").unwrap();
            let accum_bias = module.load_function("accumulate_bias_scaled_f32").unwrap();
            (module, f32_to_bf16, dequant, activation, accum_bias)
        })
    }
}

impl EgglogOp for GLUMoEMXFP4 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "GLUMoEMXFP4",
            &[
                ("hidden", EXPRESSION),
                ("intermediate", EXPRESSION),
                ("output_k", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                (
                    (= ?e (Op (GLUMoEMXFP4 ?hidden ?intermediate ?output_k) ?inputs))
                )
                (
                    (set (dtype ?e) (F32))
                )
                :ruleset dtype_prop
            )",
            ),
            Rule::raw(include_str!["glumoe_mxfp4_rewrite.egg"]),
        ]
    }

    fn n_inputs(&self) -> usize {
        9
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let hidden = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let intermediate = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let output_k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let extracted = GLUMoEMXFP4::new(hidden, intermediate, output_k);
        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for GLUMoEMXFP4 {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 9 {
            anyhow::bail!("GLUMoEMXFP4 expected 9 inputs, got {}", inputs.len());
        }

        let hidden = self
            .hidden
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 hidden unresolved"))?;
        let intermediate = self
            .intermediate
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 intermediate unresolved"))?;
        let top_k = self
            .output_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 top_k unresolved"))?;
        if hidden == 0 || intermediate == 0 {
            anyhow::bail!("GLUMoEMXFP4 zero dims: hidden={hidden}, intermediate={intermediate}");
        }
        if top_k == 0 {
            return Ok(());
        }
        if hidden % MXFP4_BLOCK != 0 || intermediate % MXFP4_BLOCK != 0 {
            anyhow::bail!(
                "GLUMoEMXFP4 dims must be multiples of {MXFP4_BLOCK}: hidden={hidden}, intermediate={intermediate}"
            );
        }
        let gate_up_dim = 2 * intermediate;

        let output_bytes = self
            .output_bytes()
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 output bytes unresolved"))?;
        if output_bytes % (hidden * 4) != 0 {
            anyhow::bail!("GLUMoEMXFP4 output bytes {output_bytes} not divisible by hidden*4");
        }
        let seq = output_bytes / (hidden * 4);
        if seq == 0 {
            return Ok(());
        }

        let get_buffer = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers
                .get(&node)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("GLUMoEMXFP4 missing {name} buffer for {node:?}"))
        };

        let x_buf = get_buffer("x", inputs[0])?;
        let topk_idx_buf = get_buffer("topk indices", inputs[1])?;
        let topk_vals_buf = get_buffer("topk values", inputs[2])?;
        let gu_blocks_buf = get_buffer("gate_up blocks", inputs[3])?;
        let gu_scales_buf = get_buffer("gate_up scales", inputs[4])?;
        let gu_bias_buf = get_buffer("gate_up bias", inputs[5])?;
        let dn_blocks_buf = get_buffer("down blocks", inputs[6])?;
        let dn_scales_buf = get_buffer("down scales", inputs[7])?;
        let dn_bias_buf = get_buffer("down bias", inputs[8])?;
        let output_buf = get_buffer("output", self_node)?;

        // Per-expert byte strides into the packed weight tensors.
        let gu_blocks_stride = gate_up_dim * (hidden / 2); // u8
        let gu_scales_stride = gate_up_dim * (hidden / MXFP4_BLOCK); // u8
        let gu_bias_stride = gate_up_dim * 2; // bf16
        let dn_blocks_stride = hidden * (intermediate / 2);
        let dn_scales_stride = hidden * (intermediate / MXFP4_BLOCK);
        let dn_bias_stride = hidden * 2;

        if gu_blocks_stride == 0 || gu_blocks_buf.len() % gu_blocks_stride != 0 {
            anyhow::bail!(
                "GLUMoEMXFP4 gate_up blocks buffer {} not a multiple of per-expert stride {gu_blocks_stride}",
                gu_blocks_buf.len()
            );
        }
        let num_experts = gu_blocks_buf.len() / gu_blocks_stride;
        if num_experts == 0 {
            anyhow::bail!("GLUMoEMXFP4 has no experts");
        }

        let x_ptr = buf_ptr(x_buf, stream);
        let gu_blocks_ptr = buf_ptr(gu_blocks_buf, stream);
        let gu_scales_ptr = buf_ptr(gu_scales_buf, stream);
        let gu_bias_ptr = buf_ptr(gu_bias_buf, stream);
        let dn_blocks_ptr = buf_ptr(dn_blocks_buf, stream);
        let dn_scales_ptr = buf_ptr(dn_scales_buf, stream);
        let dn_bias_ptr = buf_ptr(dn_bias_buf, stream);
        let output_ptr = buf_ptr(output_buf, stream);

        let cublaslt = self.get_cublaslt(stream)?;
        let (_, f32_to_bf16_fn, dequant_fn, activation_fn, accum_bias_fn) = self.get_kernels(stream);

        // Host-side top-k routing, compacted to a dense top_k stride. top-k
        // weights are applied directly (no normalization).
        let (topk_idx, topk_vals) =
            read_routing(stream, topk_idx_buf, topk_vals_buf, seq, top_k, "GLUMoEMXFP4")?;
        for &expert_idx in &topk_idx {
            if expert_idx < 0 || expert_idx as usize >= num_experts {
                anyhow::bail!("GLUMoEMXFP4 expert index {expert_idx} out of range");
            }
        }

        // Reused scratch (freed at end of this layer's execute).
        let x_bf16 = unsafe { stream.alloc::<u8>(seq * hidden * 2)? };
        let gu_w_deq = unsafe { stream.alloc::<u8>(gate_up_dim * hidden * 2)? };
        let gu_out = unsafe { stream.alloc::<u8>(gate_up_dim * 2)? };
        let hid = unsafe { stream.alloc::<u8>(intermediate * 2)? };
        let dn_w_deq = unsafe { stream.alloc::<u8>(hidden * intermediate * 2)? };
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_SIZE)? };
        let xbf16_ptr = slice_ptr(&x_bf16, stream);
        let gu_w_deq_ptr = slice_ptr(&gu_w_deq, stream);
        let gu_out_ptr = slice_ptr(&gu_out, stream);
        let hid_ptr = slice_ptr(&hid, stream);
        let dn_w_deq_ptr = slice_ptr(&dn_w_deq, stream);
        let ws_ptr = slice_ptr(&workspace, stream);

        let launch = |f: &CudaFunction, n: usize, args: &[KernelArg]| -> anyhow::Result<()> {
            let blocks = (n as u32).div_ceil(256);
            let mut b = stream.launch_builder(f);
            for a in args {
                match a {
                    KernelArg::U64(v) => b.arg(v),
                    KernelArg::I32(v) => b.arg(v),
                    KernelArg::F32(v) => b.arg(v),
                };
            }
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            }
            Ok(())
        };

        // x F32 -> BF16
        launch(
            f32_to_bf16_fn,
            seq * hidden,
            &[
                KernelArg::U64(x_ptr),
                KernelArg::U64(xbf16_ptr),
                KernelArg::I32((seq * hidden) as i32),
            ],
        )?;

        // Per-token expert computation via the shared MoE skeleton. The MXFP4
        // specifics live in the hooks: dequant the gathered expert into reused
        // scratch (gate_up / down), the clamped interleaved SwiGLU + gate_up
        // bias activation, and the weighted down-bias accumulate epilogue.
        let dims = MoeDims {
            seq,
            hidden,
            intermediate,
            gate_up_dim,
            top_k,
        };
        run_moe_experts(
            stream,
            &cublaslt,
            ws_ptr,
            &dims,
            xbf16_ptr,
            output_ptr,
            gu_out_ptr,
            hid_ptr,
            &topk_idx,
            &topk_vals,
            |e| {
                launch(
                    dequant_fn,
                    gate_up_dim * hidden,
                    &[
                        KernelArg::U64(gu_blocks_ptr + (e * gu_blocks_stride) as u64),
                        KernelArg::U64(gu_scales_ptr + (e * gu_scales_stride) as u64),
                        KernelArg::U64(gu_w_deq_ptr),
                        KernelArg::I32(gate_up_dim as i32),
                        KernelArg::I32(hidden as i32),
                    ],
                )?;
                Ok(gu_w_deq_ptr)
            },
            |e, gu, hid_out| {
                launch(
                    activation_fn,
                    intermediate,
                    &[
                        KernelArg::U64(gu),
                        KernelArg::U64(gu_bias_ptr + (e * gu_bias_stride) as u64),
                        KernelArg::U64(hid_out),
                        KernelArg::I32(intermediate as i32),
                    ],
                )
            },
            |e| {
                launch(
                    dequant_fn,
                    hidden * intermediate,
                    &[
                        KernelArg::U64(dn_blocks_ptr + (e * dn_blocks_stride) as u64),
                        KernelArg::U64(dn_scales_ptr + (e * dn_scales_stride) as u64),
                        KernelArg::U64(dn_w_deq_ptr),
                        KernelArg::I32(hidden as i32),
                        KernelArg::I32(intermediate as i32),
                    ],
                )?;
                Ok(dn_w_deq_ptr)
            },
            |_t, e, w, out| {
                launch(
                    accum_bias_fn,
                    hidden,
                    &[
                        KernelArg::U64(out),
                        KernelArg::U64(dn_bias_ptr + (e * dn_bias_stride) as u64),
                        KernelArg::I32(hidden as i32),
                        KernelArg::F32(w),
                    ],
                )
            },
        )?;

        Ok(())
    }

    fn output_size(&self) -> Expression {
        Expression::from('s') * self.hidden
    }

    fn output_bytes(&self) -> Expression {
        Expression::from('s') * self.hidden * 4
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("GLUMoEMXFP4")
    }
}

enum KernelArg {
    U64(u64),
    I32(i32),
    F32(f32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cudarc::driver::{CudaContext, DevicePtr};

    // FP4 E2M1 value table (mirrors quant.rs FP4_E2M1_LUT).
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// Build one expert's packed bytes where every weight element decodes to the
    /// FP4 value at `nibble` (both nibbles per byte), with all e8m0 scales = 1.0.
    fn packed_const(out_dim: usize, in_dim: usize, nibble: u8) -> (Vec<u8>, Vec<u8>) {
        let byte = (nibble << 4) | nibble;
        let blocks = vec![byte; out_dim * (in_dim / 2)];
        let scales = vec![127u8; out_dim * (in_dim / MXFP4_BLOCK)]; // 2^(127-127)=1.0
        (blocks, scales)
    }

    #[test]
    fn glumoe_mxfp4_matches_reference() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        let stream = ctx.default_stream();

        let (hidden, intermediate, e, top_k, seq) = (64usize, 64usize, 2usize, 1usize, 1usize);
        let gate_up_dim = 2 * intermediate;

        // Weights: gate_up = 1.0 everywhere (nibble 2), down = 0.5 (nibble 1).
        let mut gu_blocks = Vec::new();
        let mut gu_scales = Vec::new();
        for _ in 0..e {
            let (b, s) = packed_const(gate_up_dim, hidden, 2);
            gu_blocks.extend(b);
            gu_scales.extend(s);
        }
        let mut dn_blocks = Vec::new();
        let mut dn_scales = Vec::new();
        for _ in 0..e {
            let (b, s) = packed_const(hidden, intermediate, 1);
            dn_blocks.extend(b);
            dn_scales.extend(s);
        }
        let gu_bias = vec![0u8; e * gate_up_dim * 2]; // bf16 zeros
        let dn_bias = vec![0u8; e * hidden * 2];

        let x_f32 = vec![0.1f32; seq * hidden];
        let topk_idx = vec![0i32; seq * top_k]; // expert 0
        let topk_val = vec![1.0f32; seq * top_k]; // weight 1.0
        let out_init = vec![0u8; seq * hidden * 4];

        let to_bytes_f32 = |v: &[f32]| bytemuck::cast_slice::<f32, u8>(v).to_vec();
        let to_bytes_i32 = |v: &[i32]| bytemuck::cast_slice::<i32, u8>(v).to_vec();

        // Upload (keep slices alive for the duration of execute).
        let d_x = stream.memcpy_stod(&to_bytes_f32(&x_f32)).unwrap();
        let d_idx = stream.memcpy_stod(&to_bytes_i32(&topk_idx)).unwrap();
        let d_val = stream.memcpy_stod(&to_bytes_f32(&topk_val)).unwrap();
        let d_gub = stream.memcpy_stod(&gu_blocks).unwrap();
        let d_gus = stream.memcpy_stod(&gu_scales).unwrap();
        let d_gubias = stream.memcpy_stod(&gu_bias).unwrap();
        let d_dnb = stream.memcpy_stod(&dn_blocks).unwrap();
        let d_dns = stream.memcpy_stod(&dn_scales).unwrap();
        let d_dnbias = stream.memcpy_stod(&dn_bias).unwrap();
        let d_out = stream.memcpy_stod(&out_init).unwrap();

        let pdb = |s: &crate::cudarc::driver::CudaSlice<u8>| DeviceBuffer::new(s.device_ptr(&stream).0, s.len());
        let mut buffers: FxHashMap<NodeIndex, DeviceBuffer> = FxHashMap::default();
        let nodes: Vec<NodeIndex> = (0..10).map(NodeIndex::new).collect();
        for (n, b) in nodes.iter().zip([
            pdb(&d_x), pdb(&d_idx), pdb(&d_val), pdb(&d_gub), pdb(&d_gus),
            pdb(&d_gubias), pdb(&d_dnb), pdb(&d_dns), pdb(&d_dnbias), pdb(&d_out),
        ]) {
            buffers.insert(*n, b);
        }
        let inputs: Vec<NodeIndex> = nodes[..9].to_vec();
        let self_node = nodes[9];

        let mut dyn_map = FxHashMap::default();
        dyn_map.insert('s', seq);

        let op = GLUMoEMXFP4::new(hidden.into(), intermediate.into(), top_k.into());
        op.execute(&stream, self_node, &inputs, &buffers, &dyn_map)
            .expect("execute failed");

        let out_bytes = stream.memcpy_dtov(&d_out).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&out_bytes);

        // CPU reference (f32).
        let s: f32 = x_f32.iter().take(hidden).sum(); // gate_up_out[j] = sum(x)*1.0
        let gate = s.min(SWIGLU_LIMIT);
        let up = s.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let glu = gate / (1.0 + (-SWIGLU_ALPHA * gate).exp());
        let h_val = (up + 1.0) * glu; // hidden[i]
        let down_out = (intermediate as f32) * h_val * 0.5; // down_w = 0.5
        let expected = 1.0 * down_out; // weight 1.0
        assert_eq!(LUT[2], 1.0);

        let max_rel = out
            .iter()
            .map(|&o| ((o - expected).abs() / expected.abs()).abs())
            .fold(0.0f32, f32::max);
        eprintln!("expected={expected:.4} got[0]={:.4} max_rel={max_rel:.4}", out[0]);
        assert!(
            max_rel < 0.08,
            "GLUMoEMXFP4 output deviates: expected {expected:.4}, got {:?}, max_rel {max_rel:.4}",
            &out[..hidden.min(4)]
        );
    }
}
