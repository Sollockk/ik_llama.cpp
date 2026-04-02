// ggml-ray-march.c — Neural Raymarcher: Two-Tier PIM Kernel
//
// Dynamic block size: adapts to the delta quantization type.
// Q4_0 (blck_size=32): 44 blocks for ne[0]=1408 — compatible with all tensors.
// K-quants (blck_size=256): 16 blocks for ne[0]=4096.

#include "ggml-impl.h"
#include "ggml-ray-march.h"

#include <math.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
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

#if defined(__AVX2__) || defined(__AVX__)
    // AVX: process 8 floats at a time, clear sign bit for absolute value
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (int b = 0; b < n_blocks; b++) {
        const float * xb = x + b * block_size;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int j = 0;
        // Unroll 2x for better throughput (16 floats per iteration)
        for (; j + 15 < block_size; j += 16) {
            __m256 v0 = _mm256_loadu_ps(xb + j);
            __m256 v1 = _mm256_loadu_ps(xb + j + 8);
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(v0, sign_mask));
            acc1 = _mm256_add_ps(acc1, _mm256_and_ps(v1, sign_mask));
        }
        // Handle remaining 8-float chunk
        for (; j + 7 < block_size; j += 8) {
            __m256 v = _mm256_loadu_ps(xb + j);
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(v, sign_mask));
        }

        // Horizontal sum
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        sum4 = _mm_add_ss(sum4, _mm_movehdup_ps(sum4));

        float energy = _mm_cvtss_f32(sum4);

        // Scalar tail
        for (; j < block_size; j++) {
            energy += fabsf(xb[j]);
        }
        energies[b] = energy;
    }
#else
    // Scalar fallback
    for (int b = 0; b < n_blocks; b++) {
        float energy = 0.0f;
        const float * xb = x + b * block_size;
        for (int j = 0; j < block_size; j++) {
            energy += fabsf(xb[j]);
        }
        energies[b] = energy;
    }
#endif
}

// ---------------------------------------------------------------------------
// Two-tier partition: skip vs active
// ---------------------------------------------------------------------------

void ggml_ray_march_partition_blocks(
        const float * energies,
        int           n_blocks,
        float         skip_threshold,
        struct ggml_ray_march_tiers * tiers) {

    tiers->n_active = 0;
    tiers->n_skip   = 0;
    tiers->n_total  = n_blocks;

    for (int b = 0; b < n_blocks; b++) {
        if (energies[b] < skip_threshold) {
            tiers->n_skip++;
        } else {
            tiers->active_blocks[tiers->n_active++] = b;
        }
    }
}

// ---------------------------------------------------------------------------
// Two-tier vec_dot — dynamic block size
// ---------------------------------------------------------------------------

void ggml_vec_dot_ray_march(
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
    const size_t delta_qblocks_per_eblock = (delta_blk_elems > 0) ? block_size / delta_blk_elems : 1;

    // Same for delta input type
    enum ggml_type dinput_vdt = delta_traits.vec_dot_type;
    const int    dinput_blk_elems = ggml_blck_size(dinput_vdt);
    const size_t dinput_blk_bytes = ggml_type_size(dinput_vdt);
    const size_t dinput_qblocks_per_eblock = (dinput_blk_elems > 0) ? block_size / dinput_blk_elems : 1;

    // Blurry block strides
    const int    blurry_blk_elems = blurry_traits.blck_size;
    const size_t blurry_blk_bytes = blurry_traits.type_size;
    const size_t blurry_qblocks_per_eblock = (blurry_blk_elems > 0) ? block_size / blurry_blk_elems : 1;

    // Input block strides (for blurry's vec_dot_type)
    enum ggml_type input_vdt = blurry_traits.vec_dot_type;
    const int    input_blk_elems = ggml_blck_size(input_vdt);
    const size_t input_blk_bytes = ggml_type_size(input_vdt);
    const size_t input_qblocks_per_eblock = (input_blk_elems > 0) ? block_size / input_blk_elems : 1;

    // --- TRUE RAY MARCH: skip BOTH blurry and delta for zero-energy blocks ---
    // Only active blocks are computed. The ray flies through empty space.
    float total = 0.0f;
    const void * delta_input = input_row_delta ? input_row_delta : input_row;

    // Active blocks: blurry + delta correction
    for (int g = 0; g < tiers->n_active; g++) {
        int b = tiers->active_blocks[g];

        // Blurry contribution
        if (blurry_traits.vec_dot) {
            float bv = 0.0f;
            blurry_traits.vec_dot(
                block_size, &bv, 0,
                (const char *)blurry_row + b * blurry_qblocks_per_eblock * blurry_blk_bytes, 0,
                (const char *)input_row  + b * input_qblocks_per_eblock  * input_blk_bytes, 0, 1);
            total += bv;
        }

        // Delta correction
        if (delta_row && delta_traits.vec_dot) {
            float dv = 0.0f;
            delta_traits.vec_dot(
                block_size, &dv, 0,
                (const char *)delta_row   + b * delta_qblocks_per_eblock  * delta_blk_bytes, 0,
                (const char *)delta_input + b * dinput_qblocks_per_eblock * dinput_blk_bytes, 0, 1);
            total += dv;
        }
    }

    // Skip blocks: contribute ZERO. No blurry read. No delta read. The ray flew through.

    *s = total;
}
