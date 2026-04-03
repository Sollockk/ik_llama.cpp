// ggml-pim-cache.h — Lazy per-expert delta cache with pointer swap
//
// On first access per expert: pread from delta GGUF → anonymous heap buffer.
// Subsequent accesses: direct heap pointer (no I/O).
// Under memory pressure: kernel swaps anonymous pages to swap partition
// (fast sequential recovery) instead of dropping file-backed mmap pages
// (slow random re-fault from GGUF).

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pim_delta_cache {
    char *       heap_buf;        // anonymous heap buffer (same layout as mmap tensor)
    const char * mmap_ptr;        // original mmap pointer (fallback / source for pread)
    int          fd;              // file descriptor for pread
    int64_t      file_offset;     // byte offset of tensor data in delta file
    size_t       expert_bytes;    // bytes per expert slice
    int          n_experts;       // number of experts (ne02)
    uint64_t     populated[4];    // bitmask: bit set = expert slice has been pread into heap
                                  // supports up to 256 experts
    bool         gpu_pinned;      // true if heap_buf is pinned for GPU zero-copy access
    void *       dev_ptr;         // VRAM copy of delta data (cudaMalloc'd). NULL = use mmap.
};

// Sparse delta data uploaded to GPU VRAM for fast speculative correction.
// Packed format: only significant blocks stored, with uint16 index for scatter.
struct pim_sparse_delta_gpu {
    void *       dev_packed;      // VRAM: packed weight blocks (non-zero only)
    void *       dev_index;       // VRAM: uint16 block indices [n_blocks_stored]
    int          n_blocks_stored; // number of non-zero blocks per expert
    int          n_blocks_total;  // total blocks per expert row (for bounds)
    int          n_experts;       // number of experts
    size_t       packed_expert_bytes; // bytes of packed data per expert
    int          block_bytes;     // bytes per quantization block
    int          block_elems;     // elements per quantization block (qk)
};

// Allocate and initialize a cache. heap_buf is allocated but not populated.
struct pim_delta_cache * pim_delta_cache_create(
    const void * mmap_ptr,    // mmap pointer to tensor data
    int          fd,          // file descriptor for pread
    int64_t      file_offset, // byte offset in file
    size_t       expert_bytes,// bytes per expert slice (row_size * n_rows)
    int          n_experts);  // number of experts

// Free cache and heap buffer.
void pim_delta_cache_free(struct pim_delta_cache * cache);

// Get pointer to expert data. If not yet in heap, pread from file and swap.
// Thread-safe: concurrent calls for different experts are safe.
// Concurrent calls for the SAME expert may redundantly pread (harmless).
const char * pim_delta_cache_get(struct pim_delta_cache * cache, int expert_idx);

// Check if expert data is already cached (no I/O).
bool pim_delta_cache_is_warm(struct pim_delta_cache * cache, int expert_idx);

// Prefetch multiple experts in parallel using background threads.
// Non-blocking: returns immediately after launching background preads.
// n_experts_to_prefetch = number of entries in expert_indices array.
void pim_delta_cache_prefetch(struct pim_delta_cache * cache,
    const int * expert_indices, int n_experts_to_prefetch);

// Get heap buffer pointer and total size (for GPU pinning by external code).
// Returns NULL if no heap buffer (mmap fallback mode).
char * pim_delta_cache_get_heap_buf(struct pim_delta_cache * cache, size_t * out_total_bytes);

#ifdef __cplusplus
}
#endif
