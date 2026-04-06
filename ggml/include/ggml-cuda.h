#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

// Synchronize all streams on the specified CUDA device.
// This is a full device barrier (cudaDeviceSynchronize) — all pending
// kernels on ALL streams will complete before this call returns.
// Used by the blurry-sharp overlay system to ensure async kernels have
// finished before evicting cached GPU buffers.
GGML_API void ggml_backend_cuda_device_synchronize(int device);

// JIT expert upload on a dedicated CUDA stream (stream 1), separate from
// the compute stream (stream 0).  This allows expert weight uploads to
// overlap with compute on the GPU.
// After all uploads, call ggml_backend_cuda_jit_sync_uploads() to insert
// a CUDA event dependency: compute stream 0 waits for upload stream 1.
// Both calls are non-blocking on the CPU.
GGML_API void ggml_backend_cuda_jit_upload(ggml_backend_t backend, struct ggml_tensor * tensor,
                                            const void * data, size_t offset, size_t size);
GGML_API void ggml_backend_cuda_jit_sync_uploads(ggml_backend_t backend);

// Pin host memory for GPU zero-copy access (PCIe reads).
// Unlike ggml_backend_cuda_register_host_buffer, not gated by env var.
// Returns true on success. Memory must be unpinned before free.
GGML_API bool ggml_backend_cuda_pin_host_memory(void * ptr, size_t size);
GGML_API void ggml_backend_cuda_enable_delta_post_graph(ggml_backend_t backend, bool enable);
GGML_API void ggml_backend_cuda_set_delta_vram_budget(ggml_backend_t backend, int mb);
GGML_API void ggml_backend_cuda_fill_delta_ring(ggml_backend_t backend,
    const struct ggml_tensor * const * tensors, const char * const * host_ptrs,
    const size_t * sizes, int n_tensors);
GGML_API void ggml_backend_cuda_set_delta_ready(ggml_backend_t backend, bool ready);
GGML_API void ggml_backend_cuda_set_offload_batch_size(ggml_backend_t backend, int batch_size);
GGML_API void ggml_backend_cuda_unpin_host_memory(void * ptr);

// Sharp expert ring buffer: enables kernel-level sharp replacement
// via ray_march_sharp_cache, bypassing the eval callback dcpy path.
GGML_API void ggml_backend_cuda_set_sharp_vram_budget(ggml_backend_t backend, int mb);
GGML_API void ggml_backend_cuda_enable_sharp_ring(ggml_backend_t backend, bool enable);

// Enable static delta streaming pipeline for dense models.
// Replaces LRU ring buffer with deterministic prefetch when layer
// execution order is fully predictable (no MoE routing).
// n_slots = pipeline depth (0 = auto, typically 3-4).  Must be called
// before first graph evaluation.
GGML_API void ggml_backend_cuda_set_delta_streaming(ggml_backend_t backend, bool enable, int n_slots);

// Disable CUDA graph capture/replay at init time (clean disable).
// Call when delta correction is active — cooperative delta requires host sync
// which is incompatible with graph capture. Must be called BEFORE first graph eval.
GGML_API void ggml_backend_cuda_disable_graphs(ggml_backend_t backend);

// Allocate device memory and copy host data to it. Returns device pointer, or NULL on failure.
GGML_API void * ggml_backend_cuda_device_malloc_and_copy(const void * host_data, size_t size);
GGML_API void   ggml_backend_cuda_device_free(void * dev_ptr);

#ifdef  __cplusplus
}
#endif
