//
// Fused base+delta GEMV kernel: computes dot(base + delta, Q8_1) in a single
// kernel launch.  Templated on base_type, delta_type, and use_wht.
//
// When use_wht=true (QD1_K + WHT):
//   - Delta was WHT'd at quantization time
//   - Input is WHT'd + requantized to INT8 by preprocessing kernel
//   - Delta loop uses full-block dp4a against WHT'd input
//   - Retains ~85-95% of correction energy at dp4a speed
//
// When use_wht=false:
//   - Standard per-sub-block vec_dot against Q8_1 input
//

#pragma once

#include "mmvq-templates.cuh"
#include "mmvq-fused-delta-args.h"

// Helper: read energy byte from a delta block (offset 2 for both QD4_K and QD1_K)
template <ggml_type delta_type>
static __device__ __forceinline__ uint8_t read_delta_energy(const void * vx_delta, int block_idx) {
    // Energy is always at byte offset 2 (after ggml_half d)
    const uint8_t * bytes = (const uint8_t *)vx_delta;
    constexpr size_t bsz = (delta_type == GGML_TYPE_QD4_K) ? sizeof(block_qd4_k) : sizeof(block_qd1_k);
    return bytes[block_idx * bsz + 2];
}

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------
template <ggml_type base_type, ggml_type delta_type, bool use_wht, int nwarps>
static __device__ void mul_mat_vec_fused_delta_impl(
    const void * __restrict__ vx_base,
    const void * __restrict__ vx_delta,
    const void * __restrict__ vy,
    const void * __restrict__ vy_wht,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int nrows_y, const int nrows_dst) {

    constexpr int qk      = ggml_cuda_type_traits<base_type>::qk;
    constexpr int qi_base  = ggml_cuda_type_traits<base_type>::qi;
    constexpr int vdr_base = get_vdr_mmvq(base_type);

    constexpr int qi_delta  = ggml_cuda_type_traits<delta_type>::qi;
    constexpr int vdr_delta = get_vdr_mmvq(delta_type);

    static_assert(qk == QK_K, "");
    static_assert(ggml_cuda_type_traits<delta_type>::qk == QK_K, "");

    constexpr vec_dot_q_cuda_t vec_dot_base  = get_vec_dot_q_cuda(base_type);
    constexpr vec_dot_q_cuda_t vec_dot_delta = get_vec_dot_q_cuda(delta_type);

    constexpr int ncols_y = 1;
    constexpr int rows_per_cuda_block = 1;

    const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row = ncols_x / qk;

    const block_q8_1 * y = (const block_q8_1 *) vy;

    float tmp[ncols_y][rows_per_cuda_block] = {{0.0f}};

    // --- Loop 1: Base type dot product (always uses original Q8_1 input) ---
    {
        constexpr int blocks_per_iter = vdr_base * nwarps * WARP_SIZE / qi_base;
        for (int kbx = tid / (qi_base / vdr_base); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr_base * (tid % (qi_base / vdr_base));
            tmp[0][0] += vec_dot_base(vx_base, &y[kby], row0 * blocks_per_row + kbx, kqs);
        }
    }

    // --- Loop 2: Delta dot product ---
    if constexpr (use_wht) {
        // WHT path: process full 256-element blocks using WHT'd INT8 input.
        // Each thread handles a subset of the 64 dp4a operations per block.
        // Thread assignment: 128 threads / block, 64 dp4a ops per block.
        // Each thread does ceil(64 / (nwarps*32)) dp4a ops per block, across blocks.
        //
        // Simpler approach: iterate over blocks, each thread handles a stripe of
        // the 256 elements, then warp-reduce. But the standard MMVQ pattern
        // already handles this via blocks_per_iter striding.
        //
        // For WHT, we use a simpler per-block approach:
        // Each block is processed by ALL threads (tid < 64 does dp4a work).
        for (int kbx = 0; kbx < blocks_per_row; kbx++) {
            const int block_idx = row0 * blocks_per_row + kbx;
            if (read_delta_energy<delta_type>(vx_delta, block_idx) == 0) continue;

            const block_qd1_k * bd = (const block_qd1_k *)vx_delta + block_idx;
            const float d_delta = __half2float(bd->d);

            // WHT'd input: 256 int8 values + float scale, per super-block
            // wht_q8_block is { int8_t qs[256]; float d; } = 260 bytes
            const int8_t * wht_qs = (const int8_t *)vy_wht + kbx * 260;
            const float d_wht = *(const float *)(wht_qs + 256);

            // Each of 128 threads processes 2 dp4a operations (= 8 elements)
            // 128 threads × 8 elements = 1024... but we only have 256 elements.
            // So: 64 dp4a ops total. Each thread does 1 dp4a if tid < 64, else idle.
            // Then warp-reduce within each warp, then cross-warp reduce.

            int my_sumi = 0;
            if (tid < 64) {
                const uint32_t * signs_u32 = (const uint32_t *)bd->qs;
                const int * wht_i32 = (const int *)wht_qs;

                // tid maps to one of 64 groups of 4 elements
                int j = tid / 8;    // which uint32 of signs (0..7)
                int k = tid % 8;    // which nibble within that uint32 (0..7)

                uint32_t s = signs_u32[j];
                int s4 = (s >> (4*k)) & 0xF;

                int expanded = 0;
                expanded |= (( s4       & 1) * 2 - 1) & 0xFF;
                expanded |= ((((s4 >> 1) & 1) * 2 - 1) & 0xFF) << 8;
                expanded |= ((((s4 >> 2) & 1) * 2 - 1) & 0xFF) << 16;
                expanded |= ((((s4 >> 3) & 1) * 2 - 1) & 0xFF) << 24;

                my_sumi = ggml_cuda_dp4a(expanded, wht_i32[tid], 0);
            }

            // Warp-level reduction (all 128 threads participate, idle ones contribute 0)
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_sumi += __shfl_xor_sync(0xffffffff, my_sumi, offset);
            }
            // Cross-warp reduction via shared memory
            __shared__ int warp_sums[4];
            if (threadIdx.x == 0) {
                warp_sums[threadIdx.y] = my_sumi;
            }
            __syncthreads();
            if (tid == 0) {
                int total = 0;
                for (int w = 0; w < nwarps; w++) total += warp_sums[w];
                // Only thread 0 accumulates the final result
                tmp[0][0] += d_delta * d_wht * (float)total;
            }
            __syncthreads();  // ensure warp_sums not reused before all warps read
        }
    } else {
        // Standard path: per-sub-block vec_dot against Q8_1 input
        constexpr int blocks_per_iter = vdr_delta * nwarps * WARP_SIZE / qi_delta;
        for (int kbx = tid / (qi_delta / vdr_delta); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int block_idx = row0 * blocks_per_row + kbx;
            if (read_delta_energy<delta_type>(vx_delta, block_idx) == 0) continue;

            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr_delta * (tid % (qi_delta / vdr_delta));
            tmp[0][0] += vec_dot_delta(vx_delta, &y[kby], block_idx, kqs);
        }
    }

    // --- Epilogue: warp reduction + single write to dst ---
    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][0][0][threadIdx.x] = tmp[0][0];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    for (int l = 0; l < nwarps - 1; ++l) {
        tmp[0][0] += tmp_shared[l][0][0][threadIdx.x];
    }
    tmp[0][0] = warp_reduce_sum(tmp[0][0]);

    if (threadIdx.x == 0 && row0 < nrows_dst) {
        dst[row0] = tmp[0][0];
    }
}

// ---------------------------------------------------------------------------
// Global kernel wrapper
// ---------------------------------------------------------------------------
template <ggml_type base_type, ggml_type delta_type, bool use_wht, int nwarps>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void mul_mat_vec_fused_delta_kernel(
    const void * __restrict__ vx_base,
    const void * __restrict__ vx_delta,
    const void * __restrict__ vy,
    const void * __restrict__ vy_wht,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int nrows_y, const int nrows_dst) {

    mul_mat_vec_fused_delta_impl<base_type, delta_type, use_wht, nwarps>(
        vx_base, vx_delta, vy, vy_wht, dst,
        ncols_x, nrows_x, nrows_y, nrows_dst);
}

// ---------------------------------------------------------------------------
// Host launch function
// ---------------------------------------------------------------------------
template <ggml_type base_type, ggml_type delta_type, bool use_wht = false>
static void mul_mat_vec_fused_delta_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream) {
    GGML_ASSERT(args.ncols_x % QK_K == 0);

    int nwarps = 1;
    int id = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[id].cc < CC_RDNA2) {
        nwarps = 4;
    }

    const int64_t nblocks = args.nrows_x;
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (nwarps) {
        case 1:
            mul_mat_vec_fused_delta_kernel<base_type, delta_type, use_wht, 1>
                <<<block_nums, block_dims, 0, stream>>>(
                args.vx_base, args.vx_delta, args.vy, args.vy_wht, args.dst,
                args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst);
            break;
        case 2:
            mul_mat_vec_fused_delta_kernel<base_type, delta_type, use_wht, 2>
                <<<block_nums, block_dims, 0, stream>>>(
                args.vx_base, args.vx_delta, args.vy, args.vy_wht, args.dst,
                args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst);
            break;
        default:
            mul_mat_vec_fused_delta_kernel<base_type, delta_type, use_wht, 4>
                <<<block_nums, block_dims, 0, stream>>>(
                args.vx_base, args.vx_delta, args.vy, args.vy_wht, args.dst,
                args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst);
            break;
    }
}

// Extern declarations for all template instances
extern void mul_mat_vec_iq2s_qd4k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3s_qd4k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2s_qd1k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3s_qd1k_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2s_qd1k_wht_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3s_qd1k_wht_fused_cuda(const mmvq_fused_delta_args & args, cudaStream_t stream);
