// ggml-ray-march.c — Neural Raymarcher: Three-Tier PIM Kernel
//
// Dynamic block size: adapts to the delta quantization type.
// Q4_0 (blck_size=32): 44 blocks for ne[0]=1408 — compatible with all tensors.
// K-quants (blck_size=256): 16 blocks for ne[0]=4096.

#include "ggml-impl.h"
#include "ggml-ray-march.h"

#include <math.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

// ---------------------------------------------------------------------------
// Block energy (Level 2 SDF) — dynamic block size
// ---------------------------------------------------------------------------

void ggml_ray_march_block_energies(
        const float * x,
        float       * energies,
        int           n,
        int           block_size) {
    if (block_size <= 0) block_size = 256;
    const int n_blocks = n / block_size;
    for (int b = 0; b < n_blocks; b++) {
        float energy = 0.0f;
        const float * xb = x + b * block_size;
        for (int j = 0; j < block_size; j++) {
            energy += fabsf(xb[j]);
        }
        energies[b] = energy;
    }
}

// ---------------------------------------------------------------------------
// Three-tier partition + async prefetch
// ---------------------------------------------------------------------------

void ggml_ray_march_partition_blocks(
        const float * energies,
        int           n_blocks,
        float         skip_threshold,
        float         sharp_threshold,
        struct ggml_ray_march_tiers * tiers,
        const void  * delta_mmap,
        size_t        delta_block_stride) {

    tiers->n_blurry = 0;
    tiers->n_sharp  = 0;
    tiers->n_skip   = 0;
    tiers->n_total  = n_blocks;

    uint64_t page_hints[64] = {0};

    for (int b = 0; b < n_blocks; b++) {
        if (energies[b] < skip_threshold) {
            tiers->n_skip++;
        } else if (energies[b] >= sharp_threshold) {
            tiers->sharp_blocks[tiers->n_sharp++] = b;
#ifndef _WIN32
            if (delta_mmap && delta_block_stride > 0) {
                size_t offset = (size_t)b * delta_block_stride;
                size_t page_idx = offset >> 12;
                size_t word = page_idx >> 6;
                uint64_t bit = (uint64_t)1 << (page_idx & 63);
                if (word < 64 && !(page_hints[word] & bit)) {
                    page_hints[word] |= bit;
                    size_t page_start = offset & ~(size_t)0xFFF;
                    posix_madvise((char *)delta_mmap + page_start, 4096, POSIX_MADV_WILLNEED);
                }
            }
#endif
        } else {
            tiers->blurry_blocks[tiers->n_blurry++] = b;
        }
    }
}

// ---------------------------------------------------------------------------
// Three-tier vec_dot — dynamic block size
// ---------------------------------------------------------------------------

void ggml_vec_dot_ray_march_3tier(
        int                    n,
        float                * s,
        const void           * blurry_row,
        ggml_ray_march_type    blurry_type_,
        const void           * delta_row,
        ggml_ray_march_type    delta_type_,
        const void           * input_row,
        const void           * input_row_delta,
        const struct ggml_ray_march_tiers * tiers,
        int                    block_size) {

    enum ggml_type blurry_type = (enum ggml_type)blurry_type_;
    enum ggml_type delta_type  = (enum ggml_type)delta_type_;

    ggml_type_traits_t blurry_traits = ggml_internal_get_type_traits(blurry_type);
    ggml_type_traits_t delta_traits  = ggml_internal_get_type_traits(delta_type);

    // Bytes per block for delta type
    const int    delta_blk_elems = delta_traits.blck_size;
    const size_t delta_blk_bytes = delta_traits.type_size;
    // How many quant blocks per energy block
    const size_t delta_qblocks_per_eblock = (delta_blk_elems > 0) ? block_size / delta_blk_elems : 1;

    // Same for delta input type
    enum ggml_type dinput_vdt = delta_traits.vec_dot_type;
    const int    dinput_blk_elems = ggml_blck_size(dinput_vdt);
    const size_t dinput_blk_bytes = ggml_type_size(dinput_vdt);
    const size_t dinput_qblocks_per_eblock = (dinput_blk_elems > 0) ? block_size / dinput_blk_elems : 1;

    // --- Step 1: FULL blurry baseline (dense, branchless, SIMD) ---
    float baseline = 0.0f;
    if (blurry_traits.vec_dot) {
        blurry_traits.vec_dot(n, &baseline, 0, blurry_row, 0, input_row, 0, 1);
    }

    // --- Step 2: Delta correction for ALL non-skip blocks ---
    float correction = 0.0f;
    if (delta_row && delta_traits.vec_dot) {
        const void * delta_input = input_row_delta ? input_row_delta : input_row;

        // Blurry-tier blocks
        for (int g = 0; g < tiers->n_blurry; g++) {
            int b = tiers->blurry_blocks[g];
            float block_result = 0.0f;
            delta_traits.vec_dot(
                block_size, &block_result, 0,
                (const char *)delta_row   + b * delta_qblocks_per_eblock * delta_blk_bytes, 0,
                (const char *)delta_input + b * dinput_qblocks_per_eblock * dinput_blk_bytes, 0, 1);
            correction += block_result;
        }

        // Sharp-tier blocks
        for (int g = 0; g < tiers->n_sharp; g++) {
            int b = tiers->sharp_blocks[g];
            float block_result = 0.0f;
            delta_traits.vec_dot(
                block_size, &block_result, 0,
                (const char *)delta_row   + b * delta_qblocks_per_eblock * delta_blk_bytes, 0,
                (const char *)delta_input + b * dinput_qblocks_per_eblock * dinput_blk_bytes, 0, 1);
            correction += block_result;
        }
    }

    *s = baseline + correction;
}
