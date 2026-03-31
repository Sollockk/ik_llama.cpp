#pragma once

// ggml-ray-march.h — Neural Raymarcher: Dense Baseline + Sparse Additive Correction
//
// Architecture:
//   Step 1: Dense TQ1_0 vec_dot over ALL blocks (branchless, SIMD, always cached)
//   Step 2: Precompute block energies → build gather list → MADV_WILLNEED
//   Step 3: Sparse delta vec_dot over gathered blocks only (branchless, additive)
//   Result = Step1 + Step3
//
// No branching in inner loops. No Frankenstein tensors. Mathematical continuity
// guaranteed. Delta pages prefetched asynchronously while baseline computes.

#include <stddef.h>
#include <stdint.h>

// ggml_type as int to avoid include order issues with ggml.h
typedef int ggml_ray_march_type;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Step 2a: Precompute per-block activation energies (Level 2 SDF)
// ---------------------------------------------------------------------------

// energy[b] = sum(|x[b*256 .. (b+1)*256 - 1]|)   (L1 norm per block)
// Cost: ~n additions. Negligible.
void ggml_ray_march_block_energies(
    const float * x,         // input activations [n] (f32)
    float       * energies,  // output [n/256]
    int           n);

// ---------------------------------------------------------------------------
// Step 2b: Build gather list + async page prefetch (Stream Compaction)
// ---------------------------------------------------------------------------

// Scans block energies against threshold. Blocks above threshold are added to
// the gather list. For each gathered block, issues MADV_WILLNEED on the
// containing OS page so the DMA controller prefetches delta data from SSD
// while the CPU runs the dense baseline pass.
//
// Returns number of blocks in gather list.
int ggml_ray_march_build_gather_list(
    const float * energies,      // [n_blocks]
    int           n_blocks,
    float         threshold,
    int         * gather_list,   // output: block indices [capacity >= n_blocks]
    const void  * delta_mmap,    // base of delta mmap (for MADV_WILLNEED, may be NULL)
    size_t        block_stride); // bytes per delta block (for page offset calculation)

// ---------------------------------------------------------------------------
// Step 3: Sparse additive delta vec_dot (branchless gather loop)
// ---------------------------------------------------------------------------

// Computes the CORRECTION: sum of delta dot products for gathered blocks only.
// The caller adds this to the dense baseline result.
//
// delta_row: pointer to the start of this weight row in the delta mmap.
// input_row: pointer to the start of this input row (quantized).
// gather_list: dense array of block indices to process.
// gather_count: number of entries in gather_list.
//
// Returns correction value in *s.
void ggml_vec_dot_ray_march_sparse(
    float                * s,            // output: correction scalar
    const void           * delta_row,    // delta weight row (mmap'd, may page-fault)
    ggml_ray_march_type    delta_type,   // quantization type of delta
    const void           * input_row,    // input data (quantized)
    ggml_ray_march_type    input_type,   // quantization type of input
    const int            * gather_list,  // block indices to process
    int                    gather_count);// number of blocks

// ---------------------------------------------------------------------------
// Combined: full ray march dot product for one weight row
// ---------------------------------------------------------------------------

// Convenience function that runs the full pipeline for one output dimension:
//   1. Dense baseline vec_dot over blurry weights (all blocks)
//   2. Sparse delta vec_dot over gathered blocks (additive correction)
//   3. Returns baseline + correction
//
// The caller must precompute block_energies and gather_list before calling this.
void ggml_vec_dot_ray_march_full(
    int                    n,             // number of elements
    float                * s,             // output: final dot product
    const void           * blurry_row,    // blurry weight row (always resident)
    ggml_ray_march_type    blurry_type,
    const void           * delta_row,     // delta weight row (mmap'd)
    ggml_ray_march_type    delta_type,
    const void           * input_row,     // input (quantized)
    ggml_ray_march_type    input_type,
    const int            * gather_list,   // high-energy block indices
    int                    gather_count);

#ifdef __cplusplus
}
#endif
