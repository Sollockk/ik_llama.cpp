// scatter-condensed.cuh — Scatter condensed expert output to correct dst positions.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Simple scatter: one expert, one token's output
static __global__ void scatter_condensed_to_dst(
        const float * __restrict__ src,          // [n_alive] contiguous output
        float       * __restrict__ dst,          // [n_rows_total] full output row
        const uint16_t * __restrict__ alive_idx, // [n_alive] index mapping
        int n_alive) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_alive) {
        dst[alive_idx[i]] = src[i];
    }
}

static void launch_scatter_condensed(
        const float * src, float * dst,
        const uint16_t * alive_idx_gpu,
        int n_alive, cudaStream_t stream) {
    const int block_size = 256;
    const int n_blocks = (n_alive + block_size - 1) / block_size;
    scatter_condensed_to_dst<<<n_blocks, block_size, 0, stream>>>(
        src, dst, alive_idx_gpu, n_alive);
}

// Batch scatter: multiple tokens from contiguous dst_condensed buffer.
// dst_condensed has shape [n_alive, num_tokens] (contiguous matmul output).
// For each token r, scatter n_alive values to dst_original at the correct
// position determined by row_mapping and alive_idx.
//
// Grid: (num_tokens, 1, 1),  Block: (256, 1, 1)
// Each block handles one token's n_alive values.
// row_mapping layout: pairs of int32 {i1, i2}
static __global__ void scatter_condensed_batch(
        const float * __restrict__ dst_condensed,        // [n_alive * num_tokens] contiguous
        char        * __restrict__ dst_original,         // full dst tensor
        const int32_t * __restrict__ row_mapping,        // [num_tokens * 2]: {i1, i2} pairs
        const uint16_t * __restrict__ alive_idx,         // [n_alive]
        int n_alive,
        int n_alive_stride,  // stride in floats between tokens in dst_condensed (= n_alive)
        size_t nb1,          // dst stride for dim 1 (expert slot)
        size_t nb2) {        // dst stride for dim 2 (token)
    int token_r = blockIdx.x;
    // Read mapping for this token
    int32_t i1 = row_mapping[token_r * 2 + 0];
    int32_t i2 = row_mapping[token_r * 2 + 1];

    float * dst_row = (float *)(dst_original + i1 * nb1 + i2 * nb2);
    const float * src_row = dst_condensed + token_r * n_alive_stride;

    for (int j = threadIdx.x; j < n_alive; j += blockDim.x) {
        dst_row[alive_idx[j]] = src_row[j];
    }
}

static void launch_scatter_condensed_batch(
        const float * dst_condensed, char * dst_original,
        const int32_t * row_mapping_gpu,
        const uint16_t * alive_idx_gpu,
        int n_alive, int num_tokens,
        size_t nb1, size_t nb2,
        cudaStream_t stream) {
    if (num_tokens == 0 || n_alive == 0) return;
    const int block_size = 256;
    scatter_condensed_batch<<<num_tokens, block_size, 0, stream>>>(
        dst_condensed, dst_original, row_mapping_gpu, alive_idx_gpu,
        n_alive, n_alive, nb1, nb2);
}
