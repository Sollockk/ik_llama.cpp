// ggml-ray-march.c — Neural Raymarcher: Dense Baseline + Sparse Additive Correction
//
// The weight matrix is a volumetric field. Each output dimension is a ray.
// Step 1: Dense scan through TQ1_0 (coarse distance estimate, branchless)
// Step 2: Gather list identifies where the ray hits "high-density" regions
// Step 3: Sparse delta correction at those points (additive, branchless)
//
// Weights never move. Only the correction data is fetched on demand.
// The OS prefetches delta pages asynchronously while the baseline computes.

#include "ggml-impl.h"
#include "ggml-ray-march.h"

#include <math.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>  // posix_madvise, POSIX_MADV_WILLNEED
#endif

#ifndef QK_K
#define QK_K 256
#endif

// ---------------------------------------------------------------------------
// Step 2a: Block energy precomputation (Level 2 SDF)
// ---------------------------------------------------------------------------

void ggml_ray_march_block_energies(
        const float * x,
        float       * energies,
        int           n) {
    const int n_blocks = n / QK_K;

    for (int b = 0; b < n_blocks; b++) {
        float energy = 0.0f;
        const float * xb = x + b * QK_K;

        // L1 norm: sum of absolute values for this block.
        // High L1 = input "hits" this region hard = ray surface intersection.
        // Low L1 = input barely touches this region = empty space, coast.
        for (int j = 0; j < QK_K; j++) {
            energy += fabsf(xb[j]);
        }
        energies[b] = energy;
    }
}

// ---------------------------------------------------------------------------
// Step 2b: Stream compaction + async page prefetch
// ---------------------------------------------------------------------------

int ggml_ray_march_build_gather_list(
        const float * energies,
        int           n_blocks,
        float         threshold,
        int         * gather_list,
        const void  * delta_mmap,
        size_t        block_stride) {
    int count = 0;

    // Track which pages we've already hinted to avoid redundant syscalls.
    // Using a simple bitmask: up to 4096 pages (covers ~16MB of delta data,
    // enough for one expert's weight row in most models).
    uint64_t page_hints[64] = {0};  // 4096 bits = 4096 pages × 4KB = 16MB

    for (int b = 0; b < n_blocks; b++) {
        if (energies[b] >= threshold) {
            gather_list[count++] = b;

#ifndef _WIN32
            // Issue non-blocking prefetch for the delta page containing this block.
            // While the CPU runs the dense TQ1_0 baseline, the DMA controller
            // pulls these pages from SSD in the background.
            if (delta_mmap && block_stride > 0) {
                size_t offset = (size_t)b * block_stride;
                size_t page_idx = offset >> 12;  // 4KB page index

                // Deduplicate: don't hint the same page twice
                size_t word = page_idx >> 6;
                uint64_t bit = (uint64_t)1 << (page_idx & 63);
                if (word < 64 && !(page_hints[word] & bit)) {
                    page_hints[word] |= bit;

                    size_t page_start = offset & ~(size_t)0xFFF;
                    posix_madvise(
                        (char *)delta_mmap + page_start,
                        4096,
                        POSIX_MADV_WILLNEED);
                }
            }
#endif
        }
    }

    return count;
}

// ---------------------------------------------------------------------------
// Step 3: Sparse additive delta vec_dot (branchless gather loop)
// ---------------------------------------------------------------------------

void ggml_vec_dot_ray_march_sparse(
        float              * s,
        const void         * delta_row,
        ggml_ray_march_type  delta_type_,
        const void         * input_row,
        ggml_ray_march_type  input_type_,
        const int          * gather_list,
        int                  gather_count) {

    enum ggml_type delta_type = (enum ggml_type)delta_type_;
    (void)input_type_;  // input type determined by delta's vec_dot_type

    ggml_type_traits_t traits = ggml_internal_get_type_traits(delta_type);

    if (!traits.vec_dot || gather_count == 0) {
        *s = 0.0f;
        return;
    }

    // Block sizes for pointer arithmetic
    const size_t delta_blk_bytes = ggml_type_size(delta_type);
    const int    delta_blk_elems = ggml_blck_size(delta_type);
    const size_t delta_blocks_per_super = QK_K / delta_blk_elems;

    // Input block size (determined by delta's vec_dot_type)
    enum ggml_type input_vdt = traits.vec_dot_type;
    const size_t input_blk_bytes = ggml_type_size(input_vdt);
    const int    input_blk_elems = ggml_blck_size(input_vdt);
    const size_t input_blocks_per_super = QK_K / input_blk_elems;

    float correction = 0.0f;

    // Branchless gather loop: iterate dense array of block indices.
    // Each iteration computes one QK_K-element dot product.
    for (int g = 0; g < gather_count; g++) {
        int b = gather_list[g];

        float block_result = 0.0f;
        traits.vec_dot(
            QK_K, &block_result, 0,
            (const char *)delta_row + b * delta_blocks_per_super * delta_blk_bytes, 0,
            (const char *)input_row + b * input_blocks_per_super * input_blk_bytes, 0,
            1);

        correction += block_result;
    }

    *s = correction;
}

// ---------------------------------------------------------------------------
// Combined: full ray march (dense baseline + sparse correction)
// ---------------------------------------------------------------------------

void ggml_vec_dot_ray_march_full(
        int                    n,
        float                * s,
        const void           * blurry_row,
        ggml_ray_march_type    blurry_type_,
        const void           * delta_row,
        ggml_ray_march_type    delta_type_,
        const void           * input_row,
        ggml_ray_march_type    input_type_,
        const int            * gather_list,
        int                    gather_count) {

    enum ggml_type blurry_type = (enum ggml_type)blurry_type_;

    // Step 1: Dense baseline over ALL blocks (branchless, SIMD-friendly)
    float baseline = 0.0f;
    ggml_type_traits_t blurry_traits = ggml_internal_get_type_traits(blurry_type);
    if (blurry_traits.vec_dot) {
        blurry_traits.vec_dot(n, &baseline, 0, blurry_row, 0, input_row, 0, 1);
    }

    // Step 3: Sparse delta correction over gathered blocks only
    float correction = 0.0f;
    if (delta_row && gather_count > 0) {
        ggml_vec_dot_ray_march_sparse(
            &correction, delta_row, delta_type_,
            input_row, input_type_,
            gather_list, gather_count);
    }

    // Additive reconstruction: baseline + correction
    *s = baseline + correction;
}
