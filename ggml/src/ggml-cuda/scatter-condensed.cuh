// scatter-condensed.cuh — Scatter condensed MMVQ output to correct dst positions.
//
// After MMVQ computes n_alive output values (contiguous), this kernel writes
// each value to its correct position in the full output row using an index array.
// For n_alive=89, this is a single CUDA block — trivially fast.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

static __global__ void scatter_condensed_to_dst(
        const float * __restrict__ src,          // [n_alive] contiguous MMVQ output
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
