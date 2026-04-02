// ggml-pim-cache.c — Lazy per-expert delta cache implementation

#include "ggml-pim-cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

struct pim_delta_cache * pim_delta_cache_create(
    const void * mmap_ptr,
    int          fd,
    int64_t      file_offset,
    size_t       expert_bytes,
    int          n_experts) {

    struct pim_delta_cache * cache = (struct pim_delta_cache *)calloc(1, sizeof(struct pim_delta_cache));
    if (!cache) return NULL;

    cache->mmap_ptr     = (const char *)mmap_ptr;
    cache->fd           = fd;
    cache->file_offset  = file_offset;
    cache->expert_bytes = expert_bytes;
    cache->n_experts    = n_experts;
    memset(cache->populated, 0, sizeof(cache->populated));

    // Allocate heap buffer for all experts (virtual only, no physical pages yet)
    size_t total = expert_bytes * (size_t)n_experts;
    cache->heap_buf = (char *)malloc(total);
    if (!cache->heap_buf) {
        // Fallback: will use mmap_ptr directly
        fprintf(stderr, "[pim-cache] warning: failed to allocate %zu MiB heap buffer, using mmap fallback\n",
                total / (1024*1024));
    }

    return cache;
}

void pim_delta_cache_free(struct pim_delta_cache * cache) {
    if (!cache) return;
    free(cache->heap_buf);
    free(cache);
}

const char * pim_delta_cache_get(struct pim_delta_cache * cache, int expert_idx) {
    if (!cache) return NULL;
    if (expert_idx < 0 || expert_idx >= cache->n_experts) return NULL;

    // If no heap buffer, fall back to mmap
    if (!cache->heap_buf) {
        return cache->mmap_ptr + (size_t)expert_idx * cache->expert_bytes;
    }

    int word = expert_idx >> 6;
    uint64_t bit = (uint64_t)1 << (expert_idx & 63);

    if (!(cache->populated[word] & bit)) {
        // First access: pread from file into heap buffer
#ifndef _WIN32
        if (cache->fd >= 0) {
            off_t off = (off_t)(cache->file_offset + (size_t)expert_idx * cache->expert_bytes);
            ssize_t n = pread(cache->fd, cache->heap_buf + (size_t)expert_idx * cache->expert_bytes,
                              cache->expert_bytes, off);
            (void)n; // ignore short reads for now
        } else
#endif
        {
            // No fd: copy from mmap
            memcpy(cache->heap_buf + (size_t)expert_idx * cache->expert_bytes,
                   cache->mmap_ptr  + (size_t)expert_idx * cache->expert_bytes,
                   cache->expert_bytes);
        }

        // Mark as populated (benign race: worst case = redundant pread)
        __atomic_or_fetch(&cache->populated[word], bit, __ATOMIC_RELAXED);
    }

    return cache->heap_buf + (size_t)expert_idx * cache->expert_bytes;
}
