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
    cache->gpu_pinned   = false;
    cache->dev_ptr      = NULL;
    cache->n_experts    = n_experts;
    memset(cache->populated, 0, sizeof(cache->populated));

    // Use mmap fallback instead of heap allocation.
    // Allocating heap buffers for all delta tensors (~16GB) causes memory pressure
    // that corrupts GPU operations on systems with limited RAM headroom.
    // The mmap path reads directly from the delta GGUF via kernel page cache.
    cache->heap_buf = NULL;

    return cache;
}

void pim_delta_cache_free(struct pim_delta_cache * cache) {
    if (!cache) return;
    // Note: if gpu_pinned was set, the caller (CUDA-aware code) must call
    // cudaHostUnregister(cache->heap_buf) before calling this function.
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

// Mmap page-warming thread: touch every 4KB page in an expert slice to
// trigger concurrent page faults.  No madvise(WILLNEED) — that triggers
// kernel readahead which inflates cache pressure on memory-constrained
// systems.  Direct page touching faults only the exact pages needed.
#ifndef _WIN32
struct pim_mmap_warm_arg {
    const char * base;
    size_t       expert_bytes;
    const int  * expert_indices;
    int          n_experts;
    int          thread_id;
    int          n_threads;
};

static void * pim_mmap_warm_thread(void * arg) {
    struct pim_mmap_warm_arg * a = (struct pim_mmap_warm_arg *)arg;
    volatile char sink = 0;
    for (int i = a->thread_id; i < a->n_experts; i += a->n_threads) {
        const char * ptr = a->base + (size_t)a->expert_indices[i] * a->expert_bytes;
        // Touch every 4KB page to force the kernel to load it from SSD
        for (size_t off = 0; off < a->expert_bytes; off += 4096) {
            sink = ptr[off];
        }
    }
    (void)sink;
    free(arg);
    return NULL;
}
#endif

void pim_delta_cache_prefetch(struct pim_delta_cache * cache,
        const int * expert_indices, int n_experts_to_prefetch) {
    if (!cache || !expert_indices || n_experts_to_prefetch <= 0) return;

#ifndef _WIN32
    if (!cache->heap_buf) {
        // Mmap mode: warm pages by touching them in parallel threads.
        // This triggers concurrent page faults — the kernel schedules
        // SSD reads in parallel, exploiting NVMe queue depth or SATA NCQ.
        int n_threads = n_experts_to_prefetch;
        if (n_threads > 8) n_threads = 8;
        if (n_threads < 1) n_threads = 1;

        pthread_t threads[8];
        int n_launched = 0;

        for (int t = 0; t < n_threads; t++) {
            struct pim_mmap_warm_arg * arg = (struct pim_mmap_warm_arg *)
                malloc(sizeof(struct pim_mmap_warm_arg));
            if (!arg) continue;
            arg->base           = cache->mmap_ptr;
            arg->expert_bytes   = cache->expert_bytes;
            arg->expert_indices = expert_indices;
            arg->n_experts      = n_experts_to_prefetch;
            arg->thread_id      = t;
            arg->n_threads      = n_threads;

            if (pthread_create(&threads[n_launched], NULL, pim_mmap_warm_thread, arg) == 0) {
                n_launched++;
            } else {
                free(arg);
            }
        }

        // Block until all pages are warm — the subsequent cudaMemcpyAsync
        // reads from these pages and needs them in the page cache.
        for (int i = 0; i < n_launched; i++) {
            pthread_join(threads[i], NULL);
        }
        return;
    }

    // Heap mode: launch detached threads for cold experts (up to 8 concurrent).
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

char * pim_delta_cache_get_heap_buf(struct pim_delta_cache * cache, size_t * out_total_bytes) {
    if (!cache || !cache->heap_buf) return NULL;
    if (out_total_bytes) {
        *out_total_bytes = cache->expert_bytes * (size_t)cache->n_experts;
    }
    return cache->heap_buf;
}

// --- Condensed expert index ---

struct condense_index * condense_index_create(
        const uint16_t * alive_indices, int n_alive, int n_rows_total) {
    if (!alive_indices || n_alive <= 0 || n_rows_total <= 0) return NULL;

    int n_dead = n_rows_total - n_alive;
    struct condense_index * idx = (struct condense_index *)calloc(1, sizeof(*idx));
    if (!idx) return NULL;

    idx->alive_idx = (uint16_t *)malloc(n_alive * sizeof(uint16_t));
    idx->dead_idx  = (uint16_t *)malloc(n_dead  * sizeof(uint16_t));
    if (!idx->alive_idx || !idx->dead_idx) {
        free(idx->alive_idx);
        free(idx->dead_idx);
        free(idx);
        return NULL;
    }

    memcpy(idx->alive_idx, alive_indices, n_alive * sizeof(uint16_t));
    idx->n_alive = n_alive;
    idx->n_dead  = n_dead;
    idx->n_rows_total = n_rows_total;

    // Compute dead indices as complement of alive
    int di = 0, ai = 0;
    for (int r = 0; r < n_rows_total; r++) {
        if (ai < n_alive && alive_indices[ai] == r) {
            ai++;
        } else {
            idx->dead_idx[di++] = (uint16_t)r;
        }
    }

    return idx;
}

void condense_index_free(struct condense_index * idx) {
    if (!idx) return;
    free(idx->alive_idx);
    free(idx->dead_idx);
    free(idx);
}

// --- Condense index map: .cidx file loader ---

struct cidx_entry {
    char                    tensor_name[256];
    int                     expert_id;
    struct condense_index * index;
};

struct cidx_map_impl {
    struct cidx_entry * entries;
    int                 n_entries;
    int                 capacity;
};

struct condense_index_map * condense_index_map_load(const char * cidx_path) {
    if (!cidx_path) return NULL;

    FILE * f = fopen(cidx_path, "rb");
    if (!f) {
        fprintf(stderr, "[cidx] ERROR: cannot open %s\n", cidx_path);
        return NULL;
    }

    // Read header
    char magic[4];
    uint32_t version, n_tensors;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CIDX", 4) != 0) {
        fprintf(stderr, "[cidx] ERROR: bad magic in %s\n", cidx_path);
        fclose(f);
        return NULL;
    }
    if (fread(&version, 4, 1, f) != 1 || fread(&n_tensors, 4, 1, f) != 1) {
        fprintf(stderr, "[cidx] ERROR: truncated header in %s\n", cidx_path);
        fclose(f);
        return NULL;
    }
    if (version != 1) {
        fprintf(stderr, "[cidx] ERROR: unsupported version %u in %s\n", version, cidx_path);
        fclose(f);
        return NULL;
    }

    // Allocate map
    struct cidx_map_impl * impl = (struct cidx_map_impl *)calloc(1, sizeof(*impl));
    impl->capacity = 4096;
    impl->entries = (struct cidx_entry *)calloc(impl->capacity, sizeof(struct cidx_entry));
    impl->n_entries = 0;

    int total_experts = 0;

    for (uint32_t ti = 0; ti < n_tensors; ti++) {
        uint16_t name_len;
        if (fread(&name_len, 2, 1, f) != 1) break;
        char name[256] = {0};
        if (name_len >= sizeof(name)) name_len = sizeof(name) - 1;
        if (fread(name, 1, name_len, f) != name_len) break;
        name[name_len] = '\0';

        uint16_t n_experts, n_alive;
        if (fread(&n_experts, 2, 1, f) != 1) break;
        if (fread(&n_alive, 2, 1, f) != 1) break;

        for (int eid = 0; eid < n_experts; eid++) {
            uint16_t * alive_buf = (uint16_t *)malloc(n_alive * sizeof(uint16_t));
            if (!alive_buf || fread(alive_buf, sizeof(uint16_t), n_alive, f) != (size_t)n_alive) {
                free(alive_buf);
                break;
            }

            // We need n_rows_total to compute dead indices. Infer from max alive index.
            // The caller should know the actual ne01 — for now use a heuristic:
            // n_rows_total is stored nowhere in .cidx, so we'll set it to 0 and
            // let the caller fix it up when attaching to the tensor.
            // Actually, we CAN'T create the condense_index without n_rows_total.
            // Store the raw alive indices and n_alive; create the full index later.

            if (impl->n_entries >= impl->capacity) {
                impl->capacity *= 2;
                impl->entries = (struct cidx_entry *)realloc(impl->entries,
                    impl->capacity * sizeof(struct cidx_entry));
            }

            struct cidx_entry * e = &impl->entries[impl->n_entries++];
            strncpy(e->tensor_name, name, sizeof(e->tensor_name) - 1);
            e->expert_id = eid;

            // Create condense_index with a placeholder n_rows_total.
            // Use max(alive_indices) + 1 as lower bound — will be corrected
            // when the actual tensor shape is known at wiring time.
            int max_row = 0;
            for (int i = 0; i < n_alive; i++) {
                if (alive_buf[i] > max_row) max_row = alive_buf[i];
            }
            e->index = condense_index_create(alive_buf, n_alive, max_row + 1);
            free(alive_buf);
            total_experts++;
        }
    }

    fclose(f);

    fprintf(stderr, "[cidx] Loaded %s: %u tensors, %d expert entries\n",
            cidx_path, n_tensors, total_experts);

    struct condense_index_map * map = (struct condense_index_map *)calloc(1, sizeof(*map));
    map->_impl = impl;
    return map;
}

const struct condense_index * condense_index_map_get(
        const struct condense_index_map * map,
        const char * tensor_name, int expert_id) {
    if (!map || !map->_impl || !tensor_name) return NULL;
    const struct cidx_map_impl * impl = (const struct cidx_map_impl *)map->_impl;

    for (int i = 0; i < impl->n_entries; i++) {
        if (impl->entries[i].expert_id == expert_id &&
            strcmp(impl->entries[i].tensor_name, tensor_name) == 0) {
            return impl->entries[i].index;
        }
    }
    return NULL;
}

void condense_index_map_free(struct condense_index_map * map) {
    if (!map) return;
    struct cidx_map_impl * impl = (struct cidx_map_impl *)map->_impl;
    if (impl) {
        for (int i = 0; i < impl->n_entries; i++) {
            condense_index_free(impl->entries[i].index);
        }
        free(impl->entries);
        free(impl);
    }
    free(map);
}
