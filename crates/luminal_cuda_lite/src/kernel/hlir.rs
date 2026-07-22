use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{CudaFunctionExt, KernelOp},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use itertools::Itertools;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, Term, app, eq, rule, set, sort, union, v},
        base::{DTYPE, ELIST, EXPRESSION, F64, OP_KIND, SORTS, dtype, ilist, op_term},
        extract_dtype, extract_expr, extract_expr_list,
    },
    hlir::{LessThan, MaxReduce, Mod, Scatter, SumReduce},
    op::*,
    prelude::*,
};

/// Generates CUDA include directives based on the dtypes used in a kernel
pub fn dtype_includes(dtypes: &[DType]) -> String {
    let needs_fp16 = dtypes.iter().any(|d| matches!(d, DType::F16));
    let needs_bf16 = dtypes.iter().any(|d| matches!(d, DType::Bf16));
    let needs_fp8 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0));
    let needs_fp6 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F6E2M3 | DType::F6E3M2));
    let needs_fp4 = dtypes.iter().any(|d| matches!(d, DType::F4E2M1));
    let mut s = String::new();
    if needs_fp16 {
        s.push_str("#include <cuda_fp16.h>\n");
    }
    if needs_bf16 {
        s.push_str("#include <cuda_bf16.h>\n");
    }
    if needs_fp8 {
        s.push_str("#include <cuda_fp8.h>\n");
    }
    if needs_fp6 {
        s.push_str("#include <cuda_fp6.h>\n");
    }
    if needs_fp4 {
        s.push_str("#include <cuda_fp4.h>\n");
    }
    s
}

pub type Ops = (
    KernelMod,
    KernelLessThan,
    KernelIota,
    KernelGather,
    KernelGatherIota,
    KernelGatherIotaPair,
    KernelScatter,
    KernelSumReduce,
    KernelCastSumReduce,
    KernelMaxReduce,
    KernelConstant,
    KernelCast,
    KernelEmbed,
);

/// Build a rewrite that matches an HLIR op, reads dtype(s) from the given source fields,
/// and unions with a kernel op that has the same fields plus the dtype(s) appended.
pub fn kernel_rewrite<H: Default + EgglogOp, L: Default + EgglogOp>() -> Rule {
    let hlir = H::default().sort();
    let llir = L::default().sort();
    let (mut args, hlir_kind_term) = hlir.new_call();
    let inputs = v("?__inputs");
    let hlir_op = op_term(hlir_kind_term, inputs.clone());
    let dt = v("?__dt");
    args.add("dtype", dt.clone());
    let llir_kind_term = llir.call(&args);
    let llir_op = op_term(llir_kind_term, inputs);
    rule(union(hlir_op.clone(), llir_op))
        .fact(eq(dt, dtype(hlir_op)))
        .ruleset("kernel_lower")
}

#[derive(Default, Debug, Clone)]

pub struct KernelMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelMaxReduce {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMax",
            &[
                ("shape", ELIST),
                ("iters", EXPRESSION),
                ("strides", ELIST),
                ("iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<MaxReduce, Self>()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[5]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelMaxReduce {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        // Sub-32-bit float storage reduces through a float accumulator.
        // Exact for max: selection never rounds, and the final store writes a
        // value already representable in the storage dtype.
        let low_precision_storage = matches!(
            self.dtype,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0 | DType::F16 | DType::Bf16
        );
        let accum_dtype = if low_precision_storage {
            "float"
        } else {
            dtype
        };
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");
        let load_value = if low_precision_storage {
            format!("static_cast<float>(in[in_start + {iter_stride_of_i}])")
        } else {
            format!("in[in_start + {iter_stride_of_i}]")
        };

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_max_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        __shared__ {accum_dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        {accum_dtype} max_value = ({accum_dtype})NEG_INF_F;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            max_value = fmaxf(max_value, {load_value});
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            max_value = fmaxf(max_value, __shfl_down_sync(FULL_MASK, max_value, s));
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = max_value;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {accum_dtype} block_max = tid < cnt ? warp_sums[tid] : ({accum_dtype})NEG_INF_F;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_max = fmaxf(block_max, __shfl_down_sync(FULL_MASK, block_max, s));
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})block_max;
            }}
        }}
    }}
}}",
            dtype = dtype,
            accum_dtype = accum_dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            load_value = load_value,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_max_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks
            32.into(),                                      // shmem size
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "MaxReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelSumReduce {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelSum",
            &[
                ("shape", ELIST),
                ("iters", EXPRESSION),
                ("strides", ELIST),
                ("iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<SumReduce, Self>()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            {
                let out_shape =
                    extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
                let iters = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
                let in_stride =
                    extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap();
                let iter_stride = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
                let out_stride =
                    extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap();
                let dtype = extract_dtype(egraph, kind_children[5]);
                LLIROp::new::<dyn KernelOp>(Box::new(Self {
                    out_shape,
                    iters,
                    in_stride,
                    iter_stride,
                    out_stride,
                    dtype,
                }) as Box<dyn KernelOp>)
            },
            input_enodes,
        )
    }
}

impl KernelOp for KernelSumReduce {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        // Sub-32-bit float storage accumulates in float — the reduction
        // analogue of cuBLASLt's COMPUTE_32F_FAST_16BF policy for 16-bit
        // GEMMs (16-bit IO, F32 accumulation, one rounding at the store).
        let uses_fp8_storage = matches!(
            self.dtype,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0 | DType::F16 | DType::Bf16
        );
        let accum_dtype = if uses_fp8_storage { "float" } else { dtype };
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");
        let load_value = if uses_fp8_storage {
            format!("static_cast<float>(in_data[in_start + {iter_stride_of_i}])")
        } else {
            format!("in_data[in_start + {iter_stride_of_i}]")
        };
        let zero = if uses_fp8_storage {
            "0.0f".to_string()
        } else {
            format!("({dtype})0")
        };

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_sum_k({dtype} *out, const {dtype} *in_data{dyn_dims_param}) {{
        __shared__ {accum_dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        {accum_dtype} partial = {zero};
        {accum_dtype} comp = {zero};   // Kahan compensation
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            {accum_dtype} y = {load_value} - comp;
            {accum_dtype} t = partial + y;
            comp = (t - partial) - y;
            partial = t;
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {accum_dtype} block_sum = tid < cnt ? warp_sums[tid] : {zero};

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})block_sum;
            }}
        }}
    }}
}}",
            dtype = dtype,
            accum_dtype = accum_dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            load_value = load_value,
            zero = zero,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_sum_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks (warp-parallel)
            32.into(),                                      // shmem for warp_sums
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "SumReduce"
    }
}

/// Fused `Cast(F32) → SumReduce → Cast(16-bit)` with an F32 accumulator.
///
/// The dtype contract requires accumulation precision to be expressed as
/// explicit casts in HLIR: a 16-bit tensor summed with F32 accumulation is
/// written `x.cast(F32).sum(..).cast(dt)`. Lowered naively that is three
/// kernels and two F32-sized intermediate buffers. This op matches the
/// explicit pattern and unions a single kernel (16-bit loads, F32 Kahan
/// accumulation, 16-bit store) into the outer Cast's eclass — same dtype,
/// same semantics, one kernel.
#[derive(Default, Debug, Clone)]
pub struct KernelCastSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelCastSumReduce {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelCastSum",
            &[
                ("shape", ELIST),
                ("iters", EXPRESSION),
                ("strides", ELIST),
                ("iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // The inner Cast is positionwise (out[z] = (float)in[z]), so the
        // SumReduce's shape/strides over the cast output apply unchanged to
        // the 16-bit input.
        ["F16", "Bf16"]
            .into_iter()
            .map(|dt| {
                Rule::raw(format!(
                    "(rule (
                        (= ?x_cast (Op (Cast ?cast_size (F32)) (ICons ?x (INil))))
                        (= ({dt}) (dtype ?x))
                        (= ?sum (Op (Sum ?shape ?iters ?strides ?iter_stride ?out_strides) (ICons ?x_cast (INil))))
                        (= ?out_cast (Op (Cast ?out_size ({dt})) (ICons ?sum (INil))))
                     ) (
                        (let ?ks (Op (KernelCastSum ?shape ?iters ?strides ?iter_stride ?out_strides ({dt})) (ICons ?x (INil))))
                        (union ?out_cast ?ks)
                        (set (dtype ?ks) ({dt}))
                     ) :ruleset kernel_specialize :name \"kernel-cast-sum-{dt}\")"
                ))
            })
            .collect()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[5]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelCastSumReduce {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void cast_reduce_sum_k({dtype} *out, const {dtype} *in_data{dyn_dims_param}) {{
        __shared__ float warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        float partial = 0.0f;
        float comp = 0.0f;   // Kahan compensation
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            float y = static_cast<float>(in_data[in_start + {iter_stride_of_i}]) - comp;
            float t = partial + y;
            comp = (t - partial) - y;
            partial = t;
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})block_sum;
            }}
        }}
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("cast_reduce_sum_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks (warp-parallel)
            32.into(),                                      // shmem for warp_sums
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CastSumReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelGather {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGather",
            &[
                ("out_shape", ELIST),
                ("index_strides", ELIST),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match HLIR Gather (now in Op format) and rewrite to KernelGather.
        // Mirror the IList pattern used by `Gather`'s own dtype propagation
        // rule (`src/hlir.rs`): use a `?__tail` variable instead of a
        // strict `(INil)` so we don't accidentally fail to match against a
        // Gather Op whose IList tail eclass has been merged with another
        // chain by some unrelated egglog union. Without this the kernel
        // rewrite is silently skipped for some Gathers in deep models
        // (e.g. YOLO's stacked make_contiguous chains).
        let hlir_gather = luminal::hlir::Gather::default().sort();
        let (gather_args, gather_kind_term) = hlir_gather.new_call();
        let indexes = v("?__indexes");
        let data = v("?__data");
        let tail = v("?__tail");
        let gather_inputs = Term::App {
            variant: "ICons".to_string(),
            args: vec![
                indexes.clone(),
                Term::App {
                    variant: "ICons".to_string(),
                    args: vec![data.clone(), tail],
                },
            ],
        };
        let gather_op = op_term(gather_kind_term, gather_inputs);

        let out_strides = SORTS
            .row_major
            .call(("list".to_string(), gather_args["index_shape"].clone()));
        let dt = v("?__dt");
        let kernel_kind_args = [
            ("out_shape".to_string(), gather_args["index_shape"].clone()),
            (
                "index_strides".to_string(),
                gather_args["index_strides"].clone(),
            ),
            ("data_shape".to_string(), gather_args["data_shape"].clone()),
            (
                "data_strides".to_string(),
                gather_args["data_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
            ("dtype".to_string(), dt.clone()),
        ];
        let kernel_kind_term = self.sort().call(kernel_kind_args);
        let kernel_op = op_term(kernel_kind_term, ilist(vec![indexes, data.clone()]));
        vec![
            rule(union(gather_op, kernel_op))
                .fact(eq(dt, dtype(data)))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                index_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[5]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelGather {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let idx_idx = flatten_strides(&self.out_shape, &self.index_stride).to_kernel();
        let data_idx = flatten_strides(&self.data_shape, &self.data_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        {dtype}* out = C + {out_idx};
        const_z = indexes[{idx_idx}];
        *out = data[{data_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Data + indices (indices are always int32)
        (self.output_size() * self.dtype.bits()).ceil_div(8) + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Gather"
    }
}

/// Gather whose index input is a [`KernelIota`]: the index is a pure function
/// of the output position, so it is computed inline instead of materializing
/// an int32 index tensor and reading it back. The frontend's `pad`/`concat`
/// lowering (`gather(iota(index_expr)) * iota(mask_expr)`) hits this on every
/// concat — at diffusion sizes each such gather otherwise writes and re-reads
/// a ~4·N-byte index buffer plus an extra kernel launch.
#[derive(Default, Debug, Clone)]
pub struct KernelGatherIota {
    out_shape: Vec<Expression>,
    index_expr: Expression,
    /// Inline pad mask (1 = unmasked): out-of-range positions read as zero
    /// instead of multiplying by a materialized `iota(mask) cast` tensor.
    mask_expr: Expression,
    mask_stride: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelGatherIota {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGatherIota",
            &[
                ("out_shape", ELIST),
                ("index_expr", EXPRESSION),
                ("mask_expr", EXPRESSION),
                ("mask_strides", ELIST),
                ("index_strides", ELIST),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn seed_priority(&self) -> u8 {
        // Strictly dominates KernelGather + KernelIota in the same e-class
        // (same result, one fewer kernel, no index-buffer traffic); seed it
        // so the small profiled search does not have to rediscover that per
        // gather e-class.
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                    (
                        (= ?g (Op (KernelGather ?out_shape ?index_strides ?data_shape ?data_strides ?out_strides ?dt)
                            (ICons ?idx (ICons ?data ?tail))))
                        (= ?idx (Op (KernelIota ?e ?range) ?iota_inputs))
                    )
                    (
                        (let ?gi (Op (KernelGatherIota ?out_shape ?e (MNum 1) ?index_strides ?index_strides ?data_shape ?data_strides ?out_strides ?dt)
                            (ICons ?data (INil))))
                        (union ?g ?gi)
                        (set (dtype ?gi) ?dt)
                    )
                    :ruleset kernel_lower
                    :name \"gather of iota computes indices inline\"
                )",
            ),
            // The frontend's pad lowering multiplies the gathered tensor by a
            // `cast(iota(mask))` 0/1 tensor. Fold that multiply into the
            // gather: out-of-range positions become predicated zero writes,
            // which also skips their data reads. Alignment is guaranteed by
            // reusing the gather's out_strides as the Mul's a_strides.
            Rule::raw(
                "(rule
                    (
                        (= ?m (Op (Mul ?out_shape ?g_out_strides ?mask_strides ?mul_out_strides)
                            (ICons ?g (ICons ?mask (INil)))))
                        (= ?g (Op (KernelGatherIota ?out_shape ?ie (MNum 1) ?ms0 ?index_strides ?data_shape ?data_strides ?g_out_strides ?dt)
                            (ICons ?data (INil))))
                        (= ?mask (Op (Cast ?csize ?dt) (ICons ?mi (INil))))
                        (= ?mi (Op (KernelIota ?me ?mr) ?iota_inputs))
                    )
                    (
                        (let ?gm (Op (KernelGatherIota ?out_shape ?ie ?me ?mask_strides ?index_strides ?data_shape ?data_strides ?mul_out_strides ?dt)
                            (ICons ?data (INil))))
                        (union ?m ?gm)
                        (set (dtype ?gm) ?dt)
                    )
                    :ruleset kernel_lower
                    :name \"fold pad mask into gather-iota\"
                )",
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                index_expr: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                mask_expr: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                mask_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                index_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, kind_children[5], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, kind_children[6], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[7], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[8]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelGatherIota {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        // The iota buffer at position p holds index_expr(p); the gather reads
        // it at flatten(index_strides)(z), so the inline index is the
        // composition index_expr(flatten(index_strides)(z)).
        let idx_read = flatten_strides(&self.out_shape, &self.index_stride);
        let composed = self.index_expr.substitute('z', idx_read).simplify();
        let mask = (self.mask_expr != Expression::from(1)).then(|| {
            let mask_read = flatten_strides(&self.out_shape, &self.mask_stride);
            self.mask_expr.substitute('z', mask_read).simplify()
        });

        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(composed.dyn_vars())
            .chain(mask.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let data_idx = flatten_strides(&self.data_shape, &self.data_stride).to_kernel();
        // Fast path: static sizes below 2^31 and a contiguous output let the
        // kernel run 32-bit index math (nvcc strength-reduces constant
        // division to multiply-shift; 64-bit division is the bottleneck of
        // the scalar path) and write 16-byte vectors.
        let out_size_static = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_usize();
        let data_size_static = self
            .data_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_usize();
        let out_contiguous = out_size_static.is_some_and(|n| {
            luminal::shape::flat_index_linear_coeff(
                &flatten_strides(&self.out_shape, &self.out_stride),
                n,
            ) == Some(1)
        });
        let vec = (16 / self.dtype.bits().div_ceil(8)).min(8);
        let fast = out_size_static.is_some_and(|n| n < i32::MAX as usize)
            && data_size_static.is_some_and(|n| n < i32::MAX as usize)
            && out_contiguous
            && matches!(self.dtype.bits(), 16 | 32);

        let kernel = if fast {
            let guarded = if let Some(mask) = &mask {
                format!(
                    "if ({}) {{ const_z = {}; v = data[{data_idx}]; }}",
                    mask.to_kernel(),
                    composed.to_kernel(),
                )
            } else {
                format!(
                    "{{ const_z = {}; v = data[{data_idx}]; }}",
                    composed.to_kernel(),
                )
            };
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather_iota({dtype} *C, const {dtype} *data{dyn_dims_param}) {{
        long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * {vec};
        if (base >= {n_elements}) return;
        {dtype} vals[{vec}];
        #pragma unroll
        for (int i = 0; i < {vec}; ++i) {{
            int const_z = (int)base + i;
            {dtype} v = ({dtype})0.0f;
            if (const_z < {n_elements}) {guarded}
            vals[i] = v;
        }}
        if (base + {vec} <= {n_elements}) {{
            *reinterpret_cast<uint4*>(C + base) = *reinterpret_cast<const uint4*>(vals);
        }} else {{
            for (int i = 0; base + i < {n_elements}; ++i) C[base + i] = vals[i];
        }}
    }}
}}",
            )
        } else {
            let body = if let Some(mask) = &mask {
                format!(
                    "if (!({})) {{ *out = ({dtype})0.0f; return; }}
        const_z = {};
        *out = data[{data_idx}];",
                    mask.to_kernel(),
                    composed.to_kernel(),
                )
            } else {
                format!(
                    "const_z = {};
        *out = data[{data_idx}];",
                    composed.to_kernel(),
                )
            };
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather_iota({dtype} *C, const {dtype} *data{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        {dtype}* out = C + {out_idx};
        {body}
    }}
}}",
            )
        };
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather_iota").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        let threads = if fast {
            out_size.ceil_div(vec as i32)
        } else {
            out_size
        };
        (
            func,
            module,
            kernel,
            (threads.ceil_div(256), 1.into(), 1.into()),
            (threads.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.index_expr.dyn_vars())
            .chain(self.mask_expr.dyn_vars())
            .chain(self.mask_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Data only — the index is computed, not read.
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GatherIota"
    }
}

/// Addition of two mask-folded [`KernelGatherIota`]s — the frontend's
/// `concat_along` (pad + pad + add). One kernel, predicated reads: each
/// thread reads only the side(s) whose pad mask covers its position, so a
/// concat costs one data read plus one write instead of two gather kernels,
/// two mask tensors, and an add region.
#[derive(Default, Debug, Clone)]
pub struct KernelGatherIotaPair {
    out_shape: Vec<Expression>,
    out_stride: Vec<Expression>,
    a_index_expr: Expression,
    a_mask_expr: Expression,
    a_mask_stride: Vec<Expression>,
    a_index_stride: Vec<Expression>,
    a_data_shape: Vec<Expression>,
    a_data_stride: Vec<Expression>,
    b_index_expr: Expression,
    b_mask_expr: Expression,
    b_mask_stride: Vec<Expression>,
    b_index_stride: Vec<Expression>,
    b_data_shape: Vec<Expression>,
    b_data_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelGatherIotaPair {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGatherIotaPair",
            &[
                ("out_shape", ELIST),
                ("out_strides", ELIST),
                ("a_index_expr", EXPRESSION),
                ("a_mask_expr", EXPRESSION),
                ("a_mask_strides", ELIST),
                ("a_index_strides", ELIST),
                ("a_data_shape", ELIST),
                ("a_data_strides", ELIST),
                ("b_index_expr", EXPRESSION),
                ("b_mask_expr", EXPRESSION),
                ("b_mask_strides", ELIST),
                ("b_index_strides", ELIST),
                ("b_data_shape", ELIST),
                ("b_data_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn seed_priority(&self) -> u8 {
        // Strictly dominates the two masked gathers plus the add region it
        // replaces (see KernelGatherIota::seed_priority).
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    (= ?s (Op (Add ?out_shape ?sa ?sb ?so) (ICons ?ga (ICons ?gb (INil)))))
                    (= ?ga (Op (KernelGatherIota ?out_shape ?iea ?mea ?msa ?isa ?dsha ?dstra ?sa ?dt)
                        (ICons ?da (INil))))
                    (= ?gb (Op (KernelGatherIota ?out_shape ?ieb ?meb ?msb ?isb ?dshb ?dstrb ?sb ?dt)
                        (ICons ?db (INil))))
                    (!= ?mea (MNum 1))
                    (!= ?meb (MNum 1))
                )
                (
                    (let ?cat (Op (KernelGatherIotaPair ?out_shape ?so
                        ?iea ?mea ?msa ?isa ?dsha ?dstra
                        ?ieb ?meb ?msb ?isb ?dshb ?dstrb ?dt)
                        (ICons ?da (ICons ?db (INil)))))
                    (union ?s ?cat)
                    (set (dtype ?cat) ?dt)
                )
                :ruleset kernel_lower
                :name \"fold concat pad-add into one dual gather\"
            )",
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let el = |i: usize, lc: &mut FxHashMap<&'a ENodeId, Vec<Expression>>, ec: &mut FxHashMap<&'a ENodeId, Expression>| {
            extract_expr_list(egraph, kind_children[i], lc, ec).unwrap()
        };
        let out_shape = el(0, list_cache, expr_cache);
        let out_stride = el(1, list_cache, expr_cache);
        let a_index_expr = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let a_mask_expr = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let a_mask_stride = el(4, list_cache, expr_cache);
        let a_index_stride = el(5, list_cache, expr_cache);
        let a_data_shape = el(6, list_cache, expr_cache);
        let a_data_stride = el(7, list_cache, expr_cache);
        let b_index_expr = extract_expr(egraph, kind_children[8], expr_cache).unwrap();
        let b_mask_expr = extract_expr(egraph, kind_children[9], expr_cache).unwrap();
        let b_mask_stride = el(10, list_cache, expr_cache);
        let b_index_stride = el(11, list_cache, expr_cache);
        let b_data_shape = el(12, list_cache, expr_cache);
        let b_data_stride = el(13, list_cache, expr_cache);
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape,
                out_stride,
                a_index_expr,
                a_mask_expr,
                a_mask_stride,
                a_index_stride,
                a_data_shape,
                a_data_stride,
                b_index_expr,
                b_mask_expr,
                b_mask_stride,
                b_index_stride,
                b_data_shape,
                b_data_stride,
                dtype: extract_dtype(egraph, kind_children[14]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelGatherIotaPair {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let side = |index_expr: &Expression,
                    mask_expr: &Expression,
                    mask_stride: &[Expression],
                    index_stride: &[Expression]|
         -> (Expression, Expression) {
            let idx_read = flatten_strides(&self.out_shape, index_stride);
            let composed = index_expr.substitute('z', idx_read).simplify();
            let mask_read = flatten_strides(&self.out_shape, mask_stride);
            let mask = mask_expr.substitute('z', mask_read).simplify();
            (composed, mask)
        };
        let (a_idx, a_mask) = side(
            &self.a_index_expr,
            &self.a_mask_expr,
            &self.a_mask_stride,
            &self.a_index_stride,
        );
        let (b_idx, b_mask) = side(
            &self.b_index_expr,
            &self.b_mask_expr,
            &self.b_mask_stride,
            &self.b_index_stride,
        );

        let vars = self
            .out_shape
            .iter()
            .chain(self.out_stride.iter())
            .chain(self.a_data_shape.iter())
            .chain(self.a_data_stride.iter())
            .chain(self.b_data_shape.iter())
            .chain(self.b_data_stride.iter())
            .flat_map(|e| e.dyn_vars())
            .chain(a_idx.dyn_vars())
            .chain(a_mask.dyn_vars())
            .chain(b_idx.dyn_vars())
            .chain(b_mask.dyn_vars())
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_data_idx = flatten_strides(&self.a_data_shape, &self.a_data_stride).to_kernel();
        let b_data_idx = flatten_strides(&self.b_data_shape, &self.b_data_stride).to_kernel();
        // Same fast path as gather_iota: 32-bit index math + 16-byte stores
        // when sizes are static and the output is contiguous.
        let out_size_static = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_usize();
        let a_size_static = self
            .a_data_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_usize();
        let b_size_static = self
            .b_data_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_usize();
        let out_contiguous = out_size_static.is_some_and(|n| {
            luminal::shape::flat_index_linear_coeff(
                &flatten_strides(&self.out_shape, &self.out_stride),
                n,
            ) == Some(1)
        });
        let vec = (16 / self.dtype.bits().div_ceil(8)).min(8);
        let fast = out_size_static.is_some_and(|n| n < i32::MAX as usize)
            && a_size_static.is_some_and(|n| n < i32::MAX as usize)
            && b_size_static.is_some_and(|n| n < i32::MAX as usize)
            && out_contiguous
            && matches!(self.dtype.bits(), 16 | 32);
        let kernel = if fast {
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather_iota_pair({dtype} *C, const {dtype} *a, const {dtype} *b{dyn_dims_param}) {{
        long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * {vec};
        if (base >= {n_elements}) return;
        {dtype} vals[{vec}];
        #pragma unroll
        for (int i = 0; i < {vec}; ++i) {{
            int z = (int)base + i;
            int const_z = z;
            float acc = 0.0f;
            if (z < {n_elements}) {{
                const bool in_a = {a_mask_k};
                const bool in_b = {b_mask_k};
                if (in_a) {{
                    const_z = {a_idx_k};
                    acc += (float)a[{a_data_idx}];
                }}
                if (in_b) {{
                    const_z = z;
                    const_z = {b_idx_k};
                    acc += (float)b[{b_data_idx}];
                }}
            }}
            vals[i] = ({dtype})acc;
        }}
        if (base + {vec} <= {n_elements}) {{
            *reinterpret_cast<uint4*>(C + base) = *reinterpret_cast<const uint4*>(vals);
        }} else {{
            for (int i = 0; base + i < {n_elements}; ++i) C[base + i] = vals[i];
        }}
    }}
}}",
                a_mask_k = a_mask.to_kernel(),
                a_idx_k = a_idx.to_kernel(),
                b_mask_k = b_mask.to_kernel(),
                b_idx_k = b_idx.to_kernel(),
            )
        } else {
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather_iota_pair({dtype} *C, const {dtype} *a, const {dtype} *b{dyn_dims_param}) {{
        long long z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (z >= {n_elements}) return;
        long long const_z = z;
        {dtype}* out = C + {out_idx};
        const bool in_a = {a_mask_k};
        const bool in_b = {b_mask_k};
        float acc = 0.0f;
        if (in_a) {{
            const_z = {a_idx_k};
            acc += (float)a[{a_data_idx}];
        }}
        if (in_b) {{
            const_z = z;
            const_z = {b_idx_k};
            acc += (float)b[{b_data_idx}];
        }}
        *out = ({dtype})acc;
    }}
}}",
                a_mask_k = a_mask.to_kernel(),
                a_idx_k = a_idx.to_kernel(),
                b_mask_k = b_mask.to_kernel(),
                b_idx_k = b_idx.to_kernel(),
            )
        };
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather_iota_pair").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        let threads = if fast {
            out_size.ceil_div(vec as i32)
        } else {
            out_size
        };
        (
            func,
            module,
            kernel,
            (threads.ceil_div(256), 1.into(), 1.into()),
            (threads.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .chain(self.out_stride.iter())
            .chain(self.a_mask_stride.iter())
            .chain(self.a_index_stride.iter())
            .chain(self.a_data_shape.iter())
            .chain(self.a_data_stride.iter())
            .chain(self.b_mask_stride.iter())
            .chain(self.b_index_stride.iter())
            .chain(self.b_data_shape.iter())
            .chain(self.b_data_stride.iter())
            .flat_map(|e| e.dyn_vars())
            .chain(self.a_index_expr.dyn_vars())
            .chain(self.a_mask_expr.dyn_vars())
            .chain(self.b_index_expr.dyn_vars())
            .chain(self.b_mask_expr.dyn_vars())
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Each position reads from exactly one side of the concat (the pad
        // masks partition the output), so total reads ≈ one output's worth.
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GatherIotaPair"
    }
}

// KernelScatter: inverse of gather - out = copy(dest); out[indexes[i]] = src[i]
// Two-phase: memcpy graph node copies dest→output, then scatter kernel runs in same CUDA graph.
#[derive(Debug, Clone)]
pub struct KernelScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl Default for KernelScatter {
    fn default() -> Self {
        Self {
            dest_shape: Vec::new(),
            dest_strides: Vec::new(),
            index_shape: Vec::new(),
            index_strides: Vec::new(),
            src_strides: Vec::new(),
            out_strides: Vec::new(),
            dtype: DType::F32,
        }
    }
}

impl EgglogOp for KernelScatter {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelScatter",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match HLIR Scatter (now in Op format) and rewrite to KernelScatter
        let hlir_scatter = luminal::hlir::Scatter::default().sort();
        let (scatter_args, scatter_kind_term) = hlir_scatter.new_call();
        // HLIR Scatter inputs: [dest, indexes, src] (n_inputs=3)
        let dest = v("?__dest");
        let indexes = v("?__indexes");
        let src = v("?__src");
        let scatter_inputs = ilist(vec![dest.clone(), indexes.clone(), src.clone()]);
        let scatter_op = op_term(scatter_kind_term, scatter_inputs);

        let out_strides = SORTS
            .row_major
            .call(("list".to_string(), scatter_args["dest_shape"].clone()));
        let dt = v("?__dt");
        let kernel_kind_args = [
            ("dest_shape".to_string(), scatter_args["dest_shape"].clone()),
            (
                "dest_strides".to_string(),
                scatter_args["dest_strides"].clone(),
            ),
            (
                "index_shape".to_string(),
                scatter_args["index_shape"].clone(),
            ),
            (
                "index_strides".to_string(),
                scatter_args["index_strides"].clone(),
            ),
            (
                "src_strides".to_string(),
                scatter_args["src_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
            ("dtype".to_string(), dt.clone()),
        ];
        let kernel_kind_term = self.sort().call(kernel_kind_args);
        let kernel_op = op_term(kernel_kind_term, ilist(vec![dest, indexes, src.clone()]));
        vec![
            rule(union(scatter_op, kernel_op))
                .fact(eq(dt, dtype(src)))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                dest_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                dest_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                src_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[5], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[6]),
            })),
            input_enodes, // dest, indexes, src
        )
    }
}

impl KernelOp for KernelScatter {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let all_vars: FxHashSet<char> = self
            .dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&all_vars);
        let dyn_dims_param = if all_vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        // Single-kernel scatter: copy dest→output then scatter src→output[indexes]
        // Launched as 1 block of 1024 threads with __syncthreads() barrier.
        let n_src_elements = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let n_dest_elements = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let copy_dest_idx = flatten_strides(&self.dest_shape, &self.dest_strides).to_kernel();
        let copy_out_idx = flatten_strides(&self.dest_shape, &self.out_strides).to_kernel();
        let scatter_idx_idx = flatten_strides(&self.index_shape, &self.index_strides).to_kernel();
        let scatter_src_idx = flatten_strides(&self.index_shape, &self.src_strides).to_kernel();
        let scatter_kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void scatter(
        {dtype} *out, const {dtype} *dest, const int *indexes, const {dtype} *src{dyn_dims_param}
    ) {{
        int tid = threadIdx.x;
        long long n_dest = {n_dest_elements};
        long long n_src = {n_src_elements};
        // Phase 1: materialize dest into the contiguous output layout.
        // dest may be a strided or broadcast view, so copying dest[i] would read
        // past the physical source buffer for expanded tensors.
        for (long long const_z = tid; const_z < n_dest; const_z += blockDim.x) {{
            out[{copy_out_idx}] = dest[{copy_dest_idx}];
        }}
        __syncthreads();
        // Phase 2: scatter src → output[indexes[i]]
        for (long long const_z = tid; const_z < n_src; const_z += blockDim.x) {{
            int idx = indexes[{scatter_idx_idx}];
            if (idx >= 0 && idx < n_dest) {{
                out[idx] = src[{scatter_src_idx}];
            }}
        }}
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&scatter_kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx =
                compile_module_image_for_current_device(stream.context(), &scatter_kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("scatter").unwrap();
            compile_cache.insert(scatter_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            scatter_kernel,
            (1.into(), 1.into(), 1.into()),    // grid: 1 block
            (1024.into(), 1.into(), 1.into()), // block: 1024 threads
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.dest_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter output_bytes: {other:?}"),
        }
        .into();
        self.output_size() * elem_size
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // params: (out, dest, indexes, src [, dyn_dims])
        // input_ptrs: [dest, indexes, src]
        let mut params = vec![output_ptr, input_ptrs[0], input_ptrs[1], input_ptrs[2]];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn bytes_loaded(&self) -> Expression {
        let data_elem_size: Expression = match self.dtype {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter bytes_loaded: {other:?}"),
        }
        .into();
        let n_src: Expression = self.index_shape.iter().copied().product();
        // dest (copy) + indices + src
        self.output_size() * data_elem_size + n_src * 4 + n_src * data_elem_size
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_data_input(&self) -> Option<usize> {
        Some(0) // output is derived from dest (input 0): copy dest→output then scatter
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Scatter"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for KernelIota {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelIota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn n_inputs(&self) -> usize {
        0
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_iota_kind) = luminal::hlir::Iota::default().sort().new_call();
        let hlir_inputs = v("?__inputs");
        let hlir_op = op_term(hlir_iota_kind, hlir_inputs.clone());
        let kernel_kind = self.sort().call(&args);
        let kernel_op = op_term(kernel_kind, hlir_inputs);
        vec![
            rule(union(hlir_op, kernel_op.clone()))
                .set(dtype(kernel_op), app(&SORTS.int_dt, vec![]))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                expr: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl KernelOp for KernelIota {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let mut vars = self.expr.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        vars.extend(self.range.dyn_vars());
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let range = self.range.to_kernel();
        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void iota_k(int *C{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {range}) return;
        C[const_z] = {};
    }}
}}",
            self.expr.to_kernel(),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("iota_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.range.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn output_bytes(&self) -> Expression {
        // Iota always outputs int32 (4 bytes)
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn kernel_name(&self) -> &'static str {
        "Iota"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelMod {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelMod {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMod",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Mod, Self>()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelMod {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void mod_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = fmodf(A[{a_idx}], B[{b_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("mod_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Both inputs have same dtype
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Mod"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelLessThan {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelLessThan {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelLessThan",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        let hlir = LessThan::default().sort();
        let (mut args, hlir_kind_term) = hlir.new_call();
        // LessThan's dtype is Bool (output type), but the kernel needs the INPUT dtype
        // HLIR LessThan inputs: [inp_a, inp_b]
        let inp_a = v("?__inp_a");
        let inp_b = v("?__inp_b");
        let hlir_inputs = ilist(vec![inp_a.clone(), inp_b.clone()]);
        let hlir_op = op_term(hlir_kind_term, hlir_inputs.clone());
        let dt = v("?__dt");
        args.add("dtype", dt.clone());
        let kernel_kind_term = self.sort().call(&args);
        let kernel_op = op_term(kernel_kind_term, hlir_inputs);
        vec![
            rule(union(hlir_op, kernel_op))
                .fact(eq(dt, dtype(inp_a)))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelLessThan {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void less_than_k(unsigned char *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] < B[{b_idx}] ? 1 : 0;
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("less_than_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        // LessThan outputs Bool (unsigned char, 1 byte per element)
        self.output_size()
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        DType::Bool
    }

    fn kernel_name(&self) -> &'static str {
        "LessThan"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelConstant {
    value: f32,
    dtype: DType,
}

impl EgglogOp for KernelConstant {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelConstant",
            &[("value", F64), ("dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        0
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (mut args, const_kind) = luminal::hlir::Constant::default().sort().new_call();
        let hlir_inputs = v("?__inputs");
        let hlir_op = op_term(const_kind, hlir_inputs.clone());
        args.add("dtype", app(&SORTS.f32_dt, vec![]));
        let kernel_kind = self.sort().call(&args);
        let kernel_op = op_term(kernel_kind, hlir_inputs);
        let mut rules = vec![
            rule(union(hlir_op, kernel_op.clone()))
                .set(dtype(kernel_op), app(&SORTS.f32_dt, vec![]))
                .ruleset("kernel_lower"),
        ];
        // Fold an explicit Cast around a Constant into a dtype-typed
        // KernelConstant. HLIR constants are always F32 (the frontend emits
        // `constant(v).cast(dt)` for non-F32 dtypes), so this is the only
        // way a non-F32 constant reaches the kernel level. The fused op is
        // unioned into the Cast's eclass, whose dtype it matches exactly.
        // F32 included: the frontend emits `constant(v).cast(F32)` identity
        // casts for scalars on f32 tensors; folding gives downstream rules a
        // constant-valued enode (see const_like) in the cast's eclass.
        for dt in ["F16", "Bf16", "F32"] {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?c (Op (Constant ?val) (INil)))
                    (= ?cast (Op (Cast ?size ({dt})) (ICons ?c (INil))))
                 ) (
                    (let ?kc (Op (KernelConstant ?val ({dt})) (INil)))
                    (union ?cast ?kc)
                    (set (dtype ?kc) ({dt}))
                 ) :ruleset kernel_lower :name \"kernel-constant-cast-{dt}\")"
            )));
        }
        rules
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                value: egraph.enodes[kind_children[0]]
                    .0
                    .replace("\"", "")
                    .parse::<f32>()
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[1]),
            })),
            vec![],
        )
    }
}

impl KernelOp for KernelConstant {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let value_str = if self.value.is_nan() {
            "__int_as_float(0x7fc00000)".to_string()
        } else if self.value.is_infinite() {
            if self.value > 0.0 {
                "__int_as_float(0x7f800000)".to_string()
            } else {
                "__int_as_float(0xff800000)".to_string()
            }
        } else {
            format!("{:.10}f", self.value)
        };
        let cuda_ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let kernel = format!(
            "{includes}
extern \"C\" {{
    __global__ void constant_k({cuda_ty} *out) {{
        out[0] = ({cuda_ty})({value_str});
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("constant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (1.into(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        1.into()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Constant"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelCast {
    size: Expression,
    in_dtype: DType,
    out_dtype: DType,
}

impl EgglogOp for KernelCast {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelCast",
            &[("size", EXPRESSION), ("dtype", DTYPE), ("src_dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match HLIR Cast and rewrite to KernelCast
        let hlir_cast = luminal::hlir::Cast::default().sort();
        let (mut cast_args, cast_kind_term) = hlir_cast.new_call();
        let inp = v("?__inp");
        let cast_inputs = ilist(vec![inp.clone()]);
        let cast_op = op_term(cast_kind_term, cast_inputs.clone());

        let out_dty = cast_args.remove("dtype");
        let in_dty = v("?__in_dt");
        cast_args.add("dtype", in_dty.clone());
        cast_args.add("src_dtype", out_dty);
        let kernel_kind_term = self.sort().call(&cast_args);
        let kernel_op = op_term(kernel_kind_term, cast_inputs);
        vec![
            rule(union(cast_op, kernel_op))
                .fact(eq(in_dty, dtype(inp)))
                .ruleset("kernel_lower"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                size: extract_expr(egraph, kind_children[0], expr_cache).unwrap_or_default(),
                in_dtype: extract_dtype(egraph, kind_children[1]),
                out_dtype: extract_dtype(egraph, kind_children[2]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelCast {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let out_dtype = cuda_dtype(self.out_dtype);
        let includes = dtype_includes(&[self.in_dtype, self.out_dtype]);
        let vars = self.size.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let size = self.size.to_kernel();

        let kernel = if self.in_dtype.bits() < 8 {
            // Sub-byte packed types: multiple values packed per byte.
            // Extract the correct bits using bit-level addressing.
            let bits = self.in_dtype.bits();
            let in_cuda_type = cuda_dtype(self.in_dtype);
            let mask = (1u32 << bits) - 1;
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const unsigned char *in_raw{dyn_dims_param}) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= {size}) return;
        long long bit_offset = idx * {bits};
        long long byte_idx = bit_offset >> 3;
        int bit_pos = (int)(bit_offset & 7);
        unsigned short raw = (unsigned short)in_raw[byte_idx];
        if (bit_pos + {bits} > 8) raw |= ((unsigned short)in_raw[byte_idx + 1]) << 8;
        {in_cuda_type} val;
        val.__x = (unsigned char)((raw >> bit_pos) & {mask}u);
        out[idx] = ({out_dtype})val;
    }}
}}"
            )
        } else if matches!(self.in_dtype.bits(), 16 | 32) && matches!(self.out_dtype.bits(), 16 | 32)
        {
            // Flat 1:1 copy — vectorize 4 elements per thread (8/16-byte
            // transactions) instead of scalar 2/4-byte accesses.
            let in_dtype = cuda_dtype(self.in_dtype);
            let in_vec = if self.in_dtype.bits() == 16 { "uint2" } else { "uint4" };
            let out_vec = if self.out_dtype.bits() == 16 { "uint2" } else { "uint4" };
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const {in_dtype} *in{dyn_dims_param}) {{
        long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
        if (base >= {size}) return;
        if (base + 4 <= {size}) {{
            {in_dtype} a[4];
            {out_dtype} b[4];
            *reinterpret_cast<{in_vec}*>(a) = *reinterpret_cast<const {in_vec}*>(in + base);
            #pragma unroll
            for (int i = 0; i < 4; ++i) b[i] = ({out_dtype})a[i];
            *reinterpret_cast<{out_vec}*>(out + base) = *reinterpret_cast<const {out_vec}*>(b);
        }} else {{
            for (long long i = base; i < {size}; ++i) out[i] = ({out_dtype})in[i];
        }}
    }}
}}"
            )
        } else {
            let in_dtype = cuda_dtype(self.in_dtype);
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const {in_dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {size}) return;
        out[const_z] = ({out_dtype})in[const_z];
    }}
}}"
            )
        };
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("cast_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let vectorized =
            self.in_dtype.bits() >= 8 && matches!(self.in_dtype.bits(), 16 | 32) && matches!(self.out_dtype.bits(), 16 | 32);
        let threads = if vectorized {
            self.size.ceil_div(4)
        } else {
            self.size
        };
        (
            func,
            module,
            kernel,
            (threads.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn output_bytes(&self) -> Expression {
        (self.size * self.out_dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.size * self.in_dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.out_dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Cast"
    }
}

/// Thread-local global dim ordering override. When set, `generate_dyn_dims_defines`
/// uses this ordering for buffer indices instead of the kernel's local ordering.
/// This ensures all kernels in a CudaGraphOp use consistent indices into the shared
/// dyn_dims buffer.
thread_local! {
    static GLOBAL_DYN_DIMS: std::cell::RefCell<Option<Vec<char>>> = const { std::cell::RefCell::new(None) };
}

/// Set the global dyn dims ordering for subsequent kernel compilations.
pub fn set_global_dyn_dims(dims: Vec<char>) {
    GLOBAL_DYN_DIMS.with(|g| *g.borrow_mut() = Some(dims));
}

/// Clear the global dyn dims ordering.
pub fn clear_global_dyn_dims() {
    GLOBAL_DYN_DIMS.with(|g| *g.borrow_mut() = None);
}

/// Get the current global dyn dims ordering.
pub fn get_global_dyn_dims() -> Option<Vec<char>> {
    GLOBAL_DYN_DIMS.with(|g| g.borrow().clone())
}

/// Generate #define macros for dynamic dimensions that read from a shared dyn_dims buffer.
/// The buffer layout is alphabetically sorted by dim char for consistency.
/// Returns (defines_string, sorted_dims) where sorted_dims gives the order of dims in the buffer.
///
/// When a global dyn dims ordering is set (via `set_global_dyn_dims`), indices are based
/// on the global ordering to ensure consistency across kernels sharing a dyn_dims buffer.
pub fn generate_dyn_dims_defines(vars: &FxHashSet<char>) -> (String, Vec<char>) {
    if vars.is_empty() {
        return (String::new(), Vec::new());
    }
    // Check for global ordering override
    let global = GLOBAL_DYN_DIMS.with(|g| g.borrow().clone());
    if let Some(mut global_order) = global {
        // Use global ordering for indices - each dim gets its position in the global list
        // Dynamically extend the ordering if a kernel uses a dim not in the pre-scan
        let mut extended = false;
        for dim in vars.iter().sorted() {
            if !global_order.contains(dim) {
                global_order.push(*dim);
                extended = true;
            }
        }
        if extended {
            global_order.sort();
            // Update the thread-local so subsequent kernels see the extended ordering
            set_global_dyn_dims(global_order.clone());
        }
        let defines = vars
            .iter()
            .sorted()
            .map(|dim| {
                let idx = global_order
                    .iter()
                    .position(|d| d == dim)
                    .expect("Dim must be in global ordering after extension");
                format!("#define const_{dim} dyn_dims[{idx}]")
            })
            .collect::<Vec<_>>()
            .join("\n");
        return (defines, global_order);
    }
    // Default: local ordering
    let mut sorted_dims: Vec<char> = vars.iter().copied().collect();
    sorted_dims.sort();
    let defines = sorted_dims
        .iter()
        .enumerate()
        .map(|(idx, dim)| format!("#define const_{dim} dyn_dims[{idx}]"))
        .collect::<Vec<_>>()
        .join("\n");
    (defines, sorted_dims)
}

/// Get the offset for a dynamic dimension in the shared dyn_dims buffer.
/// Returns None if the dim is not in the set.
pub fn get_dyn_dim_offset(dim: char, sorted_dims: &[char]) -> Option<usize> {
    sorted_dims.iter().position(|&d| d == dim)
}

#[derive(Default, Debug, Clone)]
pub struct KernelEmbed {
    batch_shape: Vec<Expression>,  // batch dimensions (e.g., [seq_len])
    token_stride: Vec<Expression>, // stride for token_ids input
    out_stride: Vec<Expression>,   // stride for output
    embed_dim: Expression,         // row length copied per token
    row_stride: Expression,        // table row pitch (== embed_dim for embeddings)
    dtype: DType,                  // embedding table / output dtype
}

const KERNEL_EMBED_LAYOUT_DECLARATIONS: &str =
    "(relation kernel_embed_row_major (EList EList Expression))";

impl EgglogOp for KernelEmbed {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelEmbed",
            &[
                ("batch_shape", ELIST),
                ("token_stride", ELIST),
                ("out_stride", ELIST),
                ("embed_dim", EXPRESSION),
                ("row_stride", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![KERNEL_EMBED_LAYOUT_DECLARATIONS.to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "; Prove row-major storage locally in kernel_specialize rather
                 ; than relying on the earlier expression schedule to retain a
                 ; RowMajor(...) term. The carried expression is the number of
                 ; logical elements covered by the proven suffix.
                 (rule
                    (
                        (= ?shape (ECons ?dim (ENil)))
                        (= ?strides (ECons (MIter) (ENil)))
                    )
                    ((kernel_embed_row_major ?shape ?strides ?dim))
                    :ruleset kernel_specialize
                    :name \"prove rank-one row-major table\"
                 )
                 (rule
                    (
                        (= ?shape (ECons ?dim ?tail_shape))
                        (= ?strides (ECons ?head_stride ?tail_strides))
                        (kernel_embed_row_major ?tail_shape ?tail_strides ?tail_elements)
                        (= ?head_stride (MMul (MIter) ?tail_elements))
                    )
                    ((kernel_embed_row_major
                        ?shape ?strides (MMul ?dim ?tail_elements)))
                    :ruleset kernel_specialize
                    :name \"prove recursive row-major table\"
                 )",
            ),
            // Match Gather with Add(Mul(Cast(token_ids), const), Iota) indices
            // Now uses (Op (OpKind ...) (ICons ...)) format
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    ; KernelEmbed directly addresses the backing allocation as
                    ; `row * row_stride + column`.  Prove that Gather observes
                    ; the same contiguous index and table layouts.  The table
                    ; may have any row-major shape, including the flattened
                    ; source emitted by fancy indexing.
                    (= ?idx_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?idx_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (kernel_embed_row_major ?embed_shape ?embed_stride ?table_elements)
                    (= ?indices (Op (Add ?add_shape ?mul_stride ?iota_stride ?add_out_stride) (ICons ?mul_result (ICons ?iota_result (INil)))))
                    (= ?add_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?mul_stride (ECons (MIter) (ECons (MNum 0) (ENil))))
                    (= ?iota_stride (ECons (MNum 0) (ECons (MIter) (ENil))))
                    (= ?add_out_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_cast_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids_cast (ICons ?mul_const (INil)))))
                    (= ?mul_shape (ECons ?batch (ENil)))
                    (= ?token_cast_stride (ECons ?token_batch_stride (ENil)))
                    (= ?mul_const_stride (ECons (MNum 0) (ENil)))
                    (= ?mul_out_stride (ECons (MIter) (ENil)))
                    ; One row selector per batch item: token_ids has one batch
                    ; dimension, matching RemoveNthFromEnd(idx_shape). The
                    ; explicit list form also prevents flatten_strides from
                    ; receiving mismatched shape and stride ranks.
                    (= ?token_ids_cast (Op (Cast ?cast_size (Int)) (ICons ?token_ids (INil))))
                    (= ?mul_const (Op (Iota ?row_stride (MNum 1)) (INil)))
                    (= ?iota_result (Op (Iota (MIter) ?embed_dim) (INil)))
                    (= ?embed_dt (dtype ?embed_table))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_cast_stride ?out_stride_batch ?embed_dim ?row_stride ?embed_dt) (ICons ?token_ids_cast (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) ?embed_dt)
                )
                :ruleset kernel_specialize
                :name \"kernel embed with cast mul\"
            )"),
            // Match Gather with Add(Iota, Mul(Cast(token_ids), const)) indices (reversed order)
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= ?idx_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?idx_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (kernel_embed_row_major ?embed_shape ?embed_stride ?table_elements)
                    (= ?indices (Op (Add ?add_shape ?iota_stride ?mul_stride ?add_out_stride) (ICons ?iota_result (ICons ?mul_result (INil)))))
                    (= ?add_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?mul_stride (ECons (MIter) (ECons (MNum 0) (ENil))))
                    (= ?iota_stride (ECons (MNum 0) (ECons (MIter) (ENil))))
                    (= ?add_out_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_cast_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids_cast (ICons ?mul_const (INil)))))
                    (= ?mul_shape (ECons ?batch (ENil)))
                    (= ?token_cast_stride (ECons ?token_batch_stride (ENil)))
                    (= ?mul_const_stride (ECons (MNum 0) (ENil)))
                    (= ?mul_out_stride (ECons (MIter) (ENil)))
                    ; One row selector per batch item; keep its rank aligned
                    ; with the derived batch shape.
                    (= ?token_ids_cast (Op (Cast ?cast_size (Int)) (ICons ?token_ids (INil))))
                    (= ?mul_const (Op (Iota ?row_stride (MNum 1)) (INil)))
                    (= ?iota_result (Op (Iota (MIter) ?embed_dim) (INil)))
                    (= ?embed_dt (dtype ?embed_table))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_cast_stride ?out_stride_batch ?embed_dim ?row_stride ?embed_dt) (ICons ?token_ids_cast (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) ?embed_dt)
                )
                :ruleset kernel_specialize
                :name \"kernel embed with cast mul reversed\"
            )"),
            // Match Gather with Add(Mul(token_ids, const), Iota) indices (no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= ?idx_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?idx_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (kernel_embed_row_major ?embed_shape ?embed_stride ?table_elements)
                    (= ?indices (Op (Add ?add_shape ?mul_stride ?iota_stride ?add_out_stride) (ICons ?mul_result (ICons ?iota_result (INil)))))
                    (= ?add_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?mul_stride (ECons (MIter) (ECons (MNum 0) (ENil))))
                    (= ?iota_stride (ECons (MNum 0) (ECons (MIter) (ENil))))
                    (= ?add_out_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids (ICons ?mul_const (INil)))))
                    (= ?mul_shape (ECons ?batch (ENil)))
                    (= ?token_stride (ECons ?token_batch_stride (ENil)))
                    (= ?mul_const_stride (ECons (MNum 0) (ENil)))
                    (= ?mul_out_stride (ECons (MIter) (ENil)))
                    ; One row selector per batch item; keep its rank aligned
                    ; with the derived batch shape.
                    (= (dtype ?token_ids) (Int))
                    (= ?mul_const (Op (Iota ?row_stride (MNum 1)) (INil)))
                    (= ?iota_result (Op (Iota (MIter) ?embed_dim) (INil)))
                    (= ?embed_dt (dtype ?embed_table))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_stride ?out_stride_batch ?embed_dim ?row_stride ?embed_dt) (ICons ?token_ids (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) ?embed_dt)
                )
                :ruleset kernel_specialize
                :name \"kernel embed with mul\"
            )"),
            // Match Gather with Add(Iota, Mul(token_ids, const)) indices (reversed order, no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= ?idx_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?idx_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (kernel_embed_row_major ?embed_shape ?embed_stride ?table_elements)
                    (= ?indices (Op (Add ?add_shape ?iota_stride ?mul_stride ?add_out_stride) (ICons ?iota_result (ICons ?mul_result (INil)))))
                    (= ?add_shape (ECons ?batch (ECons ?embed_dim (ENil))))
                    (= ?mul_stride (ECons (MIter) (ECons (MNum 0) (ENil))))
                    (= ?iota_stride (ECons (MNum 0) (ECons (MIter) (ENil))))
                    (= ?add_out_stride
                        (ECons (MMul (MIter) ?embed_dim) (ECons (MIter) (ENil))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids (ICons ?mul_const (INil)))))
                    (= ?mul_shape (ECons ?batch (ENil)))
                    (= ?token_stride (ECons ?token_batch_stride (ENil)))
                    (= ?mul_const_stride (ECons (MNum 0) (ENil)))
                    (= ?mul_out_stride (ECons (MIter) (ENil)))
                    ; One row selector per batch item; keep its rank aligned
                    ; with the derived batch shape.
                    (= (dtype ?token_ids) (Int))
                    (= ?mul_const (Op (Iota ?row_stride (MNum 1)) (INil)))
                    (= ?iota_result (Op (Iota (MIter) ?embed_dim) (INil)))
                    (= ?embed_dt (dtype ?embed_table))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_stride ?out_stride_batch ?embed_dim ?row_stride ?embed_dt) (ICons ?token_ids (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) ?embed_dt)
                )
                :ruleset kernel_specialize
                :name \"kernel embed with mul reversed\"
            )"),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                batch_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                token_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                embed_dim: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                row_stride: extract_expr(egraph, kind_children[4], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, kind_children[5]),
            })),
            input_enodes, // token_ids, embedding_table
        )
    }
}

impl KernelOp for KernelEmbed {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        let vars = self
            .batch_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.token_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.embed_dim.dyn_vars())
            .chain(self.row_stride.dyn_vars())
            .collect::<FxHashSet<_>>();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let token_offset_expr = flatten_strides(&self.batch_shape, &self.token_stride).to_kernel();
        let out_offset_expr = flatten_strides(&self.batch_shape, &self.out_stride).to_kernel();
        let embed_dim_expr = self.embed_dim.to_kernel();
        let row_stride_expr = self.row_stride.to_kernel();
        let total_threads = batch_size * self.embed_dim;
        let n_elements = total_threads.to_kernel();
        let cuda_ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void embed({cuda_ty} *out, const int *token_ids, const {cuda_ty} *embed_table{dyn_dims_param}) {{
        long long const_z = 0;
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= {n_elements}) return;
        long long embed_dim = {embed_dim_expr};
        long long row_stride = {row_stride_expr};
        long long batch_idx = idx / embed_dim;
        long long embed_idx = idx % embed_dim;
        const_z = batch_idx;
        long long token_offset = {token_offset_expr};
        long long out_offset = {out_offset_expr};
        int token_id = token_ids[token_offset];
        out[out_offset + embed_idx] = embed_table[(long long)token_id * row_stride + embed_idx];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("embed").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        // Return empty constants map - we now use shared dyn_dims buffer
        let constants = FxHashMap::default();
        (
            func,
            module,
            kernel,
            (total_threads.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
            * self.embed_dim
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        // Load: 1 token ID (4 bytes) per batch + 1 embedding row per batch
        batch_size * ((self.embed_dim * self.dtype.bits()).ceil_div(8) + 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store: 1 embedding row per batch element
        self.output_bytes()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn flops(&self) -> Expression {
        // No FLOPs - just memory copy
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Embed"
    }
}
