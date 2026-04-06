#pragma once

// Minimal args for the fused base+delta GEMV kernel.
// Only used during generation (batch=1, no MoE, no 3D batching).
struct mmvq_fused_delta_args {
    const void * vx_base;    // base weights (IQ2_S / IQ3_S) in VRAM
    const void * vx_delta;   // delta weights (QD4_K / QD1_K) from pipeline slot
    const void * vy;         // Q8_1 quantized input
    const void * vy_wht;    // WHT-preprocessed INT8 input (nullptr if no WHT)
    float      * dst;        // output (direct write, no accumulate)
    int ncols_x;             // weight width (ne00)
    int nrows_x;             // weight height (ne01)
    int nrows_y;             // input height padded (ne10_padded)
    int nrows_dst;           // output height (dst->ne[0])
};
