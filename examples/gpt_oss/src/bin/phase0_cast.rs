//! Phase 0: prove that MXFP4 component casts run correctly on this GPU.
//!
//! gpt-oss-120b stores its MoE experts in MXFP4: 4-bit weights (`F4E2M1`) with
//! a per-32-element power-of-two scale (`F8UE8M0`, e8m0). To dequantize in-graph
//! we only need *conversion* casts (fp4 -> f32, e8m0 -> f32), not Blackwell
//! tensor-core matmuls. CUDA 12.8's headers provide software fallbacks for
//! `__CUDA_ARCH__ < 1000`, so these casts should work on sm_90 (H100).
//!
//! This binary feeds known packed bytes through `cast` on the real CudaRuntime
//! and checks the results against the reference math. Run with:
//!   cargo run -p gpt_oss --bin phase0_cast

use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};

/// 16-entry FP4 E2M1 table (1 sign + 2 exp + 1 mantissa, no NaN/Inf).
const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // sign = 0
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // sign = 1
];

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol || (a.is_sign_negative() == b.is_sign_negative() && a == 0.0 && b == 0.0)
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let cc = ctx.compute_capability().unwrap();
    println!("GPU compute capability: sm_{}{}", cc.0, cc.1);
    let stream = ctx.default_stream();

    // ---- Build a graph that casts packed F4E2M1 -> F32 and F8UE8M0 -> F32 ----
    let mut cx = Graph::default();

    // 16 fp4 values (all nibble codes 0..=15), packed two-per-byte. Per the
    // flux2 convention, the low nibble is the even element and the high nibble
    // the odd element, so element 2i = byte[i] & 0xF, element 2i+1 = byte[i] >> 4.
    let fp4 = cx
        .named_tensor("fp4_packed", 16)
        .as_dtype(DType::F4E2M1)
        .persist();
    let fp4_out = fp4.cast(DType::F32).output();

    // e8m0 scales: 2^(byte - 127). Use 1.0, 2.0, 0.5, and a small subnormal-ish.
    let e8m0 = cx
        .named_tensor("e8m0", 4)
        .as_dtype(DType::F8UE8M0)
        .persist();
    let e8m0_out = e8m0.cast(DType::F32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    // Packed bytes for elements [0,1],[2,3],...,[14,15].
    let packed: Vec<u8> = (0..8).map(|i| ((2 * i + 1) << 4) | (2 * i)).collect();
    // 127 -> 2^0 = 1, 128 -> 2^1 = 2, 126 -> 2^-1 = 0.5, 120 -> 2^-7.
    let e8m0_bytes = vec![127u8, 128, 126, 120];

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(fp4, packed.clone());
    rt.set_data(e8m0, e8m0_bytes.clone());
    rt = cx.search(rt, CompileOptions::default());

    rt.set_data(fp4, packed);
    rt.set_data(e8m0, e8m0_bytes);
    rt.execute(&cx.dyn_map);

    let fp4_res = rt.get_f32(fp4_out);
    let e8m0_res = rt.get_f32(e8m0_out);

    println!("\nfp4 cast results:");
    let mut ok = true;
    for (i, (&got, &want)) in fp4_res.iter().zip(FP4_E2M1_LUT.iter()).enumerate() {
        let pass = approx(got, want, 0.0);
        ok &= pass;
        println!(
            "  code {i:>2}: got {got:>6} want {want:>6} {}",
            if pass { "ok" } else { "MISMATCH" }
        );
    }

    println!("\ne8m0 cast results:");
    let e8m0_want = [1.0f32, 2.0, 0.5, 2f32.powi(-7)];
    for (i, (&got, &want)) in e8m0_res.iter().zip(e8m0_want.iter()).enumerate() {
        let pass = approx(got, want, 1e-9);
        ok &= pass;
        println!(
            "  scale {i}: got {got:>12e} want {want:>12e} {}",
            if pass { "ok" } else { "MISMATCH" }
        );
    }

    println!("\n{}", if ok { "PHASE 0 PASS" } else { "PHASE 0 FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
