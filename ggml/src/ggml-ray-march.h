#pragma once

// ggml-ray-march.h — Neural Raymarcher: Radiance Cascade PIM Kernel
//
// Four tiers of computation based on input activation energy per block
// (radiance cascade style — spend effort proportional to signal strength):
//
//   SKIP:     energy ≈ 0    → zero memory reads, zero compute
//   COARSE:   weak energy   → base vec_dot ONLY, delta skipped (saves ~63% BW)
//   STANDARD: medium energy → base + delta vec_dot (full PIM correction)
//   CRITICAL: strong energy → base + delta vec_dot (same compute, tracked separately for stats)
//
// The key insight: weak activations contribute little to the output, so the
// difference between Q2_K and Q8_0 for those blocks is negligible. Skipping
// the delta read for weak blocks saves ~22% total bandwidth with <0.1% quality loss.
//
// Block size is dynamic — adapts to the delta quantization type's blck_size.
// Q4_0: 32-element blocks (works with ne[0]=1408). K-quants: 256-element blocks.
//
// Backward-compatible: the old 2-tier API still works via the same partition function
// and vec_dot (coarse_threshold=0 disables the cascade, falling back to 2-tier).
//
// APPLICABILITY NOTE (profiled 2026-04-10 on Qwen3.5-9B Q4_K_M):
//
//   This ray march is effective for PIM DELTA overlays where:
//   - Delta weights (sharp - blurry) have genuinely sparse blocks
//   - MoE expert selection creates binary active/inactive patterns
//   - The energy check catches real near-zero blocks in the delta
//
//   It is NOT effective for general dense SiLU-gated FFN skipping because:
//   - RMSNorm distributes energy uniformly across input blocks
//     (lowest block energy = 0.71x mean at p1 — no near-zero blocks exist)
//   - SiLU gating is smooth, not sparse (gate std ~0.8, no deep-negative tails)
//   - silu(gate)*up intermediate is also uniform (~0.5x mean at p1)
//   - Forcing a 10% block skip on h_norm causes 25% relative gate error
//   - SVD low-rank prediction of gate signs: 53-58% accuracy (near coin-flip)
//
//   Bottom line: dense SiLU+RMSNorm models have no "empty space" to skip through.
//   The cascade tiers work because DELTA weights are sparse, not because
//   activations are sparse. Do not extend this to skip base FFN computation
//   on dense models without re-profiling — the sparsity isn't there.

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
// Two-tier partition (legacy API, still works)
// ---------------------------------------------------------------------------

struct ggml_ray_march_tiers {
    int * active_blocks;    // indices of non-skip blocks (blurry+delta computed)
    int   n_active;
    int   n_skip;           // count of skipped blocks (for stats)
    int   n_total;
};

void ggml_ray_march_partition_blocks(
    const float * energies,
    int           n_blocks,
    float         skip_threshold,
    struct ggml_ray_march_tiers * tiers);

// ---------------------------------------------------------------------------
// Four-tier cascade partition (radiance cascade)
// ---------------------------------------------------------------------------

struct ggml_ray_march_cascade {
    int * coarse_blocks;     // weak energy: base only, delta skipped
    int * standard_blocks;   // medium energy: base + delta
    int * critical_blocks;   // strong energy: base + delta (tracked for stats)
    int   n_skip;            // zero energy: nothing read
    int   n_coarse;
    int   n_standard;
    int   n_critical;
    int   n_total;
};

// Partition blocks into 4 tiers based on energy thresholds.
// skip_threshold:     energy < this → SKIP (no reads)
// coarse_threshold:   energy < this → COARSE (base only, delta skipped)
// critical_threshold: energy ≥ this → CRITICAL (base + delta, tracked separately)
// Blocks between coarse and critical → STANDARD (base + delta)
//
// Set coarse_threshold = skip_threshold to disable the coarse tier (2-tier fallback).
// Set critical_threshold = FLT_MAX to disable the critical tier (3-tier mode).
void ggml_ray_march_partition_cascade(
    const float * energies,
    int           n_blocks,
    float         skip_threshold,
    float         coarse_threshold,
    float         critical_threshold,
    struct ggml_ray_march_cascade * cascade);

// ---------------------------------------------------------------------------
// Two-tier vec_dot with dynamic block size (legacy API)
// ---------------------------------------------------------------------------

// Computes: result = sum of (blurry + delta) for active blocks only.
// Skip blocks contribute zero. block_size determines skip granularity.
void ggml_vec_dot_ray_march(
    int                    n,
    float                * s,
    const void           * blurry_row,
    ggml_ray_march_type    blurry_type,
    const void           * delta_row,
    ggml_ray_march_type    delta_type,
    const void           * input_row,        // quantized for blurry's vec_dot_type
    const void           * input_row_delta,   // quantized for delta's vec_dot_type (may differ)
    const struct ggml_ray_march_tiers * tiers,
    int                    block_size);       // elements per block (e.g. 32 or 256)

// ---------------------------------------------------------------------------
// Four-tier cascade vec_dot (radiance cascade)
// ---------------------------------------------------------------------------

// Computes: result = Σ(coarse: base only) + Σ(standard+critical: base + delta)
// Skip blocks contribute zero. Coarse blocks skip delta reads entirely.
// Saves ~63% of bandwidth for coarse blocks (delta is ~63% of total per-block BW).
void ggml_vec_dot_cascade(
    int                    n,
    float                * s,
    const void           * blurry_row,
    ggml_ray_march_type    blurry_type,
    const void           * delta_row,
    ggml_ray_march_type    delta_type,
    const void           * input_row,         // quantized for blurry's vec_dot_type
    const void           * input_row_delta,    // quantized for delta's vec_dot_type (may differ)
    const struct ggml_ray_march_cascade * cascade,
    int                    block_size);        // elements per block (e.g. 32 or 256)

#ifdef __cplusplus
}
#endif
