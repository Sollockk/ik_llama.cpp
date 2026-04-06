#pragma once
#include "common.cuh"

// WHT preprocessing for QD1_K delta correction.
// Transforms Q8_1 input activations via 256-point Walsh-Hadamard Transform,
// then re-quantizes to INT8 for dp4a-compatible delta dot products.
//
// The delta weights were WHT'd at quantization time (signs of H·delta),
// so computing dot(H·delta_signs, H·input) = dot(delta, input) gives the
// correct result while allowing sign-only quantization to retain ~85-95%
// of correction energy.
//
// Output: device buffer of wht_q8_block (per 256-element super-block),
// one per QK_K elements of the input vector.

struct wht_q8_block {
    int8_t  qs[256];   // WHT'd activations quantized to int8
    float   d;         // quantization scale (max_abs / 127)
};

// Preprocess Q8_1 input: dequant → WHT → requant to wht_q8_block.
// n_blocks = number of QK_K super-blocks in the input vector.
// Launched once per tensor, output reused across all output rows.
void ggml_cuda_wht_preprocess_q8(
    const void * q8_input,    // Q8_1 quantized input (device)
    wht_q8_block * wht_out,   // output buffer (device), n_blocks entries
    int n_blocks,              // number of QK_K-element super-blocks
    cudaStream_t stream);
