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

#ifdef __cplusplus
}
#endif
