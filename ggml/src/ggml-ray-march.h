#pragma once

// ggml-ray-march.h — Neural Raymarcher: Three-Tier PIM Kernel
//
// Three tiers of computation based on input activation energy per block:
//   SKIP:        energy ≈ 0 → zero memory reads, zero compute
//   BLURRY ONLY: medium energy → read base weights only (fast, always cached)
//   SHARP:       high energy → read base + delta (additive correction)
//
// Block size is dynamic — adapts to the delta quantization type's blck_size.
// Q4_0: 32-element blocks (works with ne[0]=1408). K-quants: 256-element blocks.

#include <stddef.h>
#include <stdint.h>

typedef int ggml_ray_march_type;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Block energy precomputation (Level 2 SDF)
// ---------------------------------------------------------------------------

// Compute per-block activation energies. block_size is dynamic (e.g. 32 for Q4_0, 256 for K-quants).
// energy[b] = sum(|x[b*block_size .. (b+1)*block_size - 1]|)
void ggml_ray_march_block_energies(
    const float * x,          // input activations [n] (f32)
    float       * energies,   // output [n/block_size]
    int           n,          // total elements
    int           block_size); // elements per block (e.g. 32 or 256)

// ---------------------------------------------------------------------------
// Three-tier gather: partition blocks into skip / blurry / sharp
// ---------------------------------------------------------------------------

struct ggml_ray_march_tiers {
    int * blurry_blocks;    // indices of blurry-only blocks
    int   n_blurry;
    int * sharp_blocks;     // indices of sharp (blurry+delta) blocks
    int   n_sharp;
    int   n_skip;           // count of skipped blocks (for stats)
    int   n_total;
};

void ggml_ray_march_partition_blocks(
    const float * energies,
    int           n_blocks,
    float         skip_threshold,
    float         sharp_threshold,
    struct ggml_ray_march_tiers * tiers,
    const void  * delta_mmap,
    size_t        delta_block_stride);

// ---------------------------------------------------------------------------
// Three-tier vec_dot with dynamic block size
// ---------------------------------------------------------------------------

// Computes: result = full_blurry_baseline + sparse_delta_correction(non-skip blocks)
// block_size determines the granularity of delta block skipping.
void ggml_vec_dot_ray_march_3tier(
    int                    n,
    float                * s,
    const void           * blurry_row,
    ggml_ray_march_type    blurry_type,
    const void           * delta_row,
    ggml_ray_march_type    delta_type,
    const void           * input_row,        // quantized for blurry's vec_dot_type
    const void           * input_row_delta,  // quantized for delta's vec_dot_type (may differ)
    const struct ggml_ray_march_tiers * tiers,
    int                    block_size);      // elements per block (e.g. 32 or 256)

#ifdef __cplusplus
}
#endif
