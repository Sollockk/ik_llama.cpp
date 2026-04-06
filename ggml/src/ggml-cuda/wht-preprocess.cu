#include "wht-preprocess.cuh"

// 256-point Walsh-Hadamard Transform in shared memory.
// 128 threads, each handles 2 elements. 8 butterfly stages.
// Matches the pattern from hadamard.cu but operates on shared memory in-place.
static __device__ void wht_256_inplace(float * __restrict__ smem, int tid) {
    constexpr float ksqrt2 = 0.707106781f;

    // Stage 0: pairs of 2
    {
        float a = smem[2*tid+0];
        float b = smem[2*tid+1];
        smem[2*tid+0] = a + b;
        smem[2*tid+1] = a - b;
    }

    float scale = ksqrt2;

    #pragma unroll
    for (int h = 2; h < 256; h <<= 1) {
        __syncthreads();
        int ii = tid / h, jj = tid % h;
        int j = 2 * h * ii + jj;
        float u = smem[j], v = smem[j + h];
        smem[j + 0] = u + v;
        smem[j + h] = u - v;
        scale *= ksqrt2;
    }
    __syncthreads();

    // Apply normalization
    smem[2*tid+0] *= scale;
    smem[2*tid+1] *= scale;
    __syncthreads();
}

// Kernel: dequant Q8_1 → float → WHT → find max → quantize to int8
// One CUDA block per QK_K super-block. 128 threads per block.
static __global__ void wht_preprocess_q8_kernel(
    const void * __restrict__ q8_input,
    wht_q8_block * __restrict__ wht_out,
    int n_blocks) {

    const int block_idx = blockIdx.x;
    if (block_idx >= n_blocks) return;

    const int tid = threadIdx.x;  // 0..127

    __shared__ float smem[256];
    __shared__ float smem_max;

    // Step 1: Dequant Q8_1 to float in shared memory.
    // QK_K=256 elements = 8 Q8_1 blocks (each 32 elements).
    // Each Q8_1 block: { half2 ds; int8_t qs[32]; }
    // ds.x = scale (d), ds.y = sum
    const block_q8_1 * q8 = (const block_q8_1 *)q8_input;
    const int q8_base = block_idx * (QK_K / QK8_1);  // 8 Q8_1 blocks per super-block

    // Each thread dequants 2 elements
    {
        int elem = 2 * tid;  // 0..254
        int q8_sub = elem / QK8_1;          // which Q8_1 block (0..7)
        int q8_off = elem % QK8_1;          // offset within Q8_1 block (0..31)
        const block_q8_1 * b = &q8[q8_base + q8_sub];
        float d = __low2float(b->ds);
        smem[elem + 0] = d * (float)b->qs[q8_off + 0];
        smem[elem + 1] = d * (float)b->qs[q8_off + 1];
    }
    __syncthreads();

    // Step 2: 256-point WHT in-place
    wht_256_inplace(smem, tid);

    // Step 3: Find max absolute value (parallel reduction)
    {
        float local_max = fmaxf(fabsf(smem[2*tid+0]), fabsf(smem[2*tid+1]));
        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }
        // Thread 0 of each warp writes to shared
        __shared__ float warp_max[4];
        if (tid % 32 == 0) warp_max[tid / 32] = local_max;
        __syncthreads();
        if (tid == 0) {
            float m = warp_max[0];
            for (int i = 1; i < 4; i++) m = fmaxf(m, warp_max[i]);
            smem_max = m;
        }
        __syncthreads();
    }

    // Step 4: Quantize to int8
    float inv_scale = (smem_max > 0.0f) ? 127.0f / smem_max : 0.0f;
    wht_q8_block * out = &wht_out[block_idx];

    out->qs[2*tid+0] = (int8_t)rintf(smem[2*tid+0] * inv_scale);
    out->qs[2*tid+1] = (int8_t)rintf(smem[2*tid+1] * inv_scale);

    if (tid == 0) {
        out->d = smem_max / 127.0f;
    }
}

void ggml_cuda_wht_preprocess_q8(
    const void * q8_input,
    wht_q8_block * wht_out,
    int n_blocks,
    cudaStream_t stream) {

    if (n_blocks <= 0) return;
    const dim3 grid(n_blocks, 1, 1);
    const dim3 block(128, 1, 1);
    wht_preprocess_q8_kernel<<<grid, block, 0, stream>>>(q8_input, wht_out, n_blocks);
}
