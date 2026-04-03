// mmvq-sparse.cuh — Sparse MMVQ kernel for speculative delta correction.
//
// Like mul_mat_vec_q but iterates only over non-zero blocks using an index.
// Used for delta correction where only ~20% of blocks are significant.
// Always accumulates (adds to dst, never overwrites).

#pragma once

#include "mmvq-templates.cuh"
#include "mmvq-sparse-args.h"

// Device function: sparse matmul-vec with block index scatter.
// Iterates only over stored (non-zero) blocks, reading input at scattered positions.
template <ggml_type type, int ncols_y, int nwarps>
static __device__ void mul_mat_vec_q_sparse(
    const void * __restrict__ vx_packed,
    const uint16_t * __restrict__ block_idx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const int n_blocks_stored, const int packed_row_blocks) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int rows_per_cuda_block = ncols_y < 4 ? 1 : 2;
#endif

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Iterate only over stored (non-zero) blocks
    for (int sbx = tid / (qi/vdr); sbx < n_blocks_stored; sbx += blocks_per_iter) {
        // Map packed index → original block position for input lookup
        const int orig_bx = block_idx[sbx];
        const int kby = orig_bx * (qk / QK8_1); // y block index at original position

        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                // Weight index: packed layout (row * packed_row_blocks + sbx)
                // Input index: original layout (using orig_bx for scatter)
                tmp[j][i] += vec_dot_q_cuda(
                    vx_packed,
                    &y[j * blocks_per_col_y + kby],
                    (row0 + i) * packed_row_blocks + sbx,
                    kqs);
            }
        }
    }

    // Reduce across warps
    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        // Always accumulate (sparse delta adds to existing blurry result)
        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            dst[j * nrows_dst + row0 + threadIdx.x] += tmp[j][threadIdx.x];
        }
    }
}

// Global kernel wrapper
template <ggml_type type, int ncols_y, int nwarps>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif
static __global__ void mul_mat_vec_q_sparse_kernel(
    const void * __restrict__ vx_packed,
    const uint16_t * __restrict__ block_idx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const int n_blocks_stored, const int packed_row_blocks,
    const uint64_t nb02_packed, const uint64_t nb12, const uint64_t nb2) {

    int i2 = blockIdx.y;
    const char * cx = (const char *)vx_packed + i2 * nb02_packed;
    const char * cy = (const char *)vy + i2 * nb12;
    char * cdst = (char *)dst + i2 * nb2;

    mul_mat_vec_q_sparse<type, ncols_y, nwarps>(
        cx, block_idx, cy, (float *)cdst,
        ncols_x, nrows_x, nrows_y, nrows_dst,
        n_blocks_stored, packed_row_blocks);
}

// Dispatch helper: launch sparse MMVQ for a given type
template <ggml_type type, int nwarps>
static void mul_mat_vec_q_sparse_cuda_T(const mmvq_sparse_args & args, cudaStream_t stream) {
    // 1 row per CUDA block for sparse (simpler, avoids row count issues)
    const dim3 block_nums(args.nrows_x, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    // nb02_packed = bytes of packed data per expert = nrows_x * n_blocks_stored * block_type_size
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const size_t block_type_size = ggml_type_size(type);
    const uint64_t nb02_packed = (uint64_t)args.nrows_x * args.packed_row_blocks * block_type_size;
    const uint64_t nb12 = (uint64_t)args.nrows_y * sizeof(block_q8_1) / QK8_1;
    const uint64_t nb2 = (uint64_t)args.nrows_dst * sizeof(float);

    mul_mat_vec_q_sparse_kernel<type, 1, nwarps><<<block_nums, block_dims, 0, stream>>>(
        args.packed_weights, (const uint16_t *)args.block_index,
        args.vy, args.dst,
        args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
        args.n_blocks_stored, args.packed_row_blocks,
        nb02_packed, nb12, nb2);
}

// Public dispatch by quant type
void ggml_cuda_mmvq_sparse_dispatch(ggml_type type, const mmvq_sparse_args & args, cudaStream_t stream) {
    constexpr int nwarps = 4; // same as standard MMVQ
    switch (type) {
        case GGML_TYPE_Q4_0: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q4_0, nwarps>(args, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q4_1, nwarps>(args, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q8_0, nwarps>(args, stream); break;
        case GGML_TYPE_Q2_K: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q2_K, nwarps>(args, stream); break;
        case GGML_TYPE_Q3_K: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q3_K, nwarps>(args, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q4_K, nwarps>(args, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q5_K, nwarps>(args, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q_sparse_cuda_T<GGML_TYPE_Q6_K, nwarps>(args, stream); break;
        default:
            fprintf(stderr, "[sparse-delta] unsupported type %s\n", ggml_type_name(type));
            break;
    }
}
