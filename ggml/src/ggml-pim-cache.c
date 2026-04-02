// ggml-pim-cache.c — Lazy per-expert delta cache implementation

#include "ggml-pim-cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#include <pthread.h>
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

bool pim_delta_cache_is_warm(struct pim_delta_cache * cache, int expert_idx) {
    if (!cache) return false;
    if (expert_idx < 0 || expert_idx >= cache->n_experts) return false;
    if (!cache->heap_buf) return true; // mmap fallback is always "warm"

    int word = expert_idx >> 6;
    uint64_t bit = (uint64_t)1 << (expert_idx & 63);
    return (__atomic_load_n(&cache->populated[word], __ATOMIC_RELAXED) & bit) != 0;
}

#ifndef _WIN32
struct pim_prefetch_arg {
    struct pim_delta_cache * cache;
    int expert_idx;
};

static void * pim_prefetch_thread(void * arg) {
    struct pim_prefetch_arg * pa = (struct pim_prefetch_arg *)arg;
    // pim_delta_cache_get triggers pread if cold, returns immediately if warm
    pim_delta_cache_get(pa->cache, pa->expert_idx);
    free(pa);
    return NULL;
}
#endif

void pim_delta_cache_prefetch(struct pim_delta_cache * cache,
        const int * expert_indices, int n_experts_to_prefetch) {
    if (!cache || !expert_indices || n_experts_to_prefetch <= 0) return;
    if (!cache->heap_buf) return; // mmap fallback, nothing to prefetch

#ifndef _WIN32
    // Launch detached threads for cold experts (up to 8 concurrent)
    // This overlaps disk I/O with computation on the calling thread.
    pthread_t threads[8];
    int n_launched = 0;

    for (int i = 0; i < n_experts_to_prefetch && n_launched < 8; i++) {
        int idx = expert_indices[i];
        if (pim_delta_cache_is_warm(cache, idx)) continue;

        struct pim_prefetch_arg * arg = (struct pim_prefetch_arg *)malloc(sizeof(struct pim_prefetch_arg));
        if (!arg) continue;
        arg->cache = cache;
        arg->expert_idx = idx;

        pthread_t t;
        if (pthread_create(&t, NULL, pim_prefetch_thread, arg) == 0) {
            threads[n_launched++] = t;
        } else {
            free(arg);
        }
    }

    // Detach all — we don't wait for them. They'll finish in background.
    // The pim_delta_cache_get calls in the compute loop will either find
    // the data already populated (fast) or wait for the ongoing pread.
    for (int i = 0; i < n_launched; i++) {
        pthread_detach(threads[i]);
    }
#else
    // On Windows, fall back to sequential prefetch
    for (int i = 0; i < n_experts_to_prefetch; i++) {
        pim_delta_cache_get(cache, expert_indices[i]);
    }
#endif
}
