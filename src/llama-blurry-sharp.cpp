//
// Blurry→Sharp overlay engine for ik_llama.cpp / GGUF
//
// See llama-blurry-sharp.h for the design overview.
//

#include "llama-blurry-sharp.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-impl.h"

#include <array>

#include "ggml.h"
#include "ggml-pim-cache.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#endif

// madvise for mmap page lifecycle management (Linux / macOS / POSIX)
// Used for MADV_DONTNEED (release consumed pages), MADV_SEQUENTIAL (readahead hint)
#ifndef _WIN32
#  include <sys/mman.h>
#  include <unistd.h>   // sysconf(_SC_PAGESIZE)
#endif

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <numeric>
#include <regex>
#include <set>

// ---------------------------------------------------------------------------
// Forward declarations (file-local)
// ---------------------------------------------------------------------------

static uint64_t bs_gpu_cache_key(const std::string & tensor_name, int expert_id);

// ---------------------------------------------------------------------------
// Helpers (file-local)
// ---------------------------------------------------------------------------

static int64_t bs_time_us() {
    return ggml_time_us();
}

// ---------------------------------------------------------------------------
// Device synchronization before GPU buffer eviction
//
// In JIT mode, CUDA kernels from a previous layer may still be executing
// asynchronously when we try to evict cached device buffers to make room
// for the next layer's sharp data.  Freeing a GPU buffer while a kernel
// is still reading from it causes "CUDA error: an illegal memory access".
//
// This helper synchronizes the GPU device to ensure all pending kernels
// have completed before we free any cached buffers.  It's only called
// when we're about to evict — not on every apply — so the overhead is
// limited to eviction situations (which already involve GPU alloc/free).
//
// Uses ggml_backend_cuda_device_synchronize() (declared in ggml-cuda.h,
// implemented in ggml-cuda.cu) which wraps cudaDeviceSynchronize().
// This avoids including <cuda_runtime.h> in this .cpp translation unit
// while still performing a full device barrier across ALL CUDA streams.
// ---------------------------------------------------------------------------
static void bs_sync_device_before_eviction(
        [[maybe_unused]] const llama_blurry_sharp_context * bsctx) {
#ifdef GGML_USE_CUDA
    int gpu = 0;
    if (bsctx && bsctx->model) {
        gpu = bsctx->model->main_gpu;
    }
    ggml_backend_cuda_device_synchronize(gpu);
#endif
}

// ---------------------------------------------------------------------------
// Parallel pread for expert slices (inspired by flash-moe's io_pool_dispatch)
//
// Persistent thread pool that reads multiple expert slices from the sharp GGUF
// file in parallel.  On NVMe SSDs with warm page cache, parallel reads achieve
// significantly higher throughput than sequential reads.
// ---------------------------------------------------------------------------

#include <pthread.h>

struct bs_pread_task {
    int         fd;
    void      * dst;
    off_t       offset;
    size_t      size;
    ssize_t     result;
};

// Simple persistent I/O thread pool — avoids creating/destroying threads per call.
// 1:1 worker-to-task mapping: worker `id` processes tasks[id].
// For more tasks than MAX_WORKERS, the caller batches via dispatch_batched().
struct bs_io_pool {
    static constexpr int MAX_WORKERS = 16;

    std::mutex              mtx;
    std::condition_variable cv_work;     // workers wait for work
    std::condition_variable cv_done;     // caller waits for completion
    std::vector<std::thread> workers;
    bs_pread_task *         tasks   = nullptr;
    int                     n_tasks = 0;
    std::atomic<int>        n_done  = 0;
    bool                    stop    = false;

    void ensure_workers(int n) {
        n = std::min(n, MAX_WORKERS);
        if ((int)workers.size() >= n) return;
        int old = (int)workers.size();
        workers.resize(n);
        for (int i = old; i < n; ++i) {
            workers[i] = std::thread([this, i]() { worker_loop(i); });
        }
    }

    void worker_loop(int id) {
        for (;;) {
            std::unique_lock<std::mutex> lock(mtx);
            cv_work.wait(lock, [this, id]() { return stop || (tasks && id < n_tasks); });
            if (stop) return;
            if (!tasks || id >= n_tasks) continue;
            auto & t = tasks[id];
            lock.unlock();

            t.result = pread(t.fd, t.dst, t.size, t.offset);

            if (n_done.fetch_add(1) + 1 == n_tasks) {
                cv_done.notify_one();
            }
        }
    }

    // Dispatch up to MAX_WORKERS tasks (1:1 mapping).
    void dispatch(bs_pread_task * t, int n) {
        ensure_workers(std::min(n, MAX_WORKERS));
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks   = t;
            n_tasks = n;
            n_done  = 0;
        }
        cv_work.notify_all();
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_done.wait(lock, [this]() { return n_done.load() >= n_tasks; });
            tasks   = nullptr;
            n_tasks = 0;
        }
    }

    // Dispatch any number of tasks by batching into chunks of MAX_WORKERS.
    void dispatch_batched(bs_pread_task * t, int n) {
        for (int off = 0; off < n; off += MAX_WORKERS) {
            int batch = std::min(n - off, MAX_WORKERS);
            dispatch(t + off, batch);
        }
    }

    ~bs_io_pool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) w.join();
        }
    }
};

static bs_io_pool & bs_get_io_pool() {
    static bs_io_pool pool;
    return pool;
}

// Separate IO pool for the async prefetch worker thread.
// The main pool (bs_get_io_pool) is used by the eval callback on the main
// thread.  dispatch() is not reentrant — two concurrent callers clobber
// each other's task state → deadlock.  This second pool lets the prefetch
// worker do parallel pread without contending with the main thread.
static bs_io_pool & bs_get_prefetch_io_pool() {
    static bs_io_pool pool;
    return pool;
}

// ---------------------------------------------------------------------------
// Split each pread task into N page-aligned sub-reads.
//
// Given an array of pread tasks, produces a new expanded array where each
// original task is split into `n_split` sub-tasks.  Each sub-task reads
// ~1/N of the data at a different file offset, allowing multiple SSD
// channels / NVMe queues to service a single large read in parallel.
//
// Sub-task boundaries are rounded to page alignment (4096 bytes) so the
// kernel can issue full-page DMA transfers.  The last sub-task absorbs
// any remainder from rounding.
// ---------------------------------------------------------------------------
static constexpr size_t BS_PAGE_SIZE = 4096;

static void bs_split_pread_tasks(
        const bs_pread_task * tasks_in,
        int                   n_tasks_in,
        int                   n_split,
        std::vector<bs_pread_task> & tasks_out) {
    tasks_out.clear();
    tasks_out.reserve((size_t)n_tasks_in * n_split);

    for (int i = 0; i < n_tasks_in; ++i) {
        const auto & t = tasks_in[i];
        if (n_split <= 1 || t.size <= BS_PAGE_SIZE) {
            tasks_out.push_back(t);
            continue;
        }

        // Compute page-aligned chunk size (round up to page boundary)
        size_t chunk_base = t.size / (size_t)n_split;
        chunk_base = (chunk_base + BS_PAGE_SIZE - 1) & ~(BS_PAGE_SIZE - 1);

        size_t off = 0;
        for (int s = 0; s < n_split && off < t.size; ++s) {
            size_t chunk = (s == n_split - 1) ? (t.size - off) : std::min(chunk_base, t.size - off);
            bs_pread_task sub;
            sub.fd     = t.fd;
            sub.dst    = (char *)t.dst + off;
            sub.offset = t.offset + (off_t)off;
            sub.size   = chunk;
            sub.result = 0;
            tasks_out.push_back(sub);
            off += chunk;
        }
    }
}

// Read n_tasks expert slices in parallel.  Falls back to sequential if n_tasks <= 1.
// When io_split > 1, each task is split into io_split page-aligned sub-reads to
// saturate multiple SSD channels / NVMe submission queues simultaneously.
static void bs_parallel_pread(bs_pread_task * tasks, int n_tasks, int io_split = 1) {
    if (n_tasks <= 0) return;
    if (io_split > 1) {
        std::vector<bs_pread_task> split_tasks;
        bs_split_pread_tasks(tasks, n_tasks, io_split, split_tasks);
        if ((int)split_tasks.size() <= 1) {
            split_tasks[0].result = pread(split_tasks[0].fd, split_tasks[0].dst,
                                          split_tasks[0].size, split_tasks[0].offset);
            // Propagate result back to original task
            tasks[0].result = split_tasks[0].result;
            return;
        }
        // Use batched dispatch — split may produce more tasks than MAX_WORKERS
        bs_get_io_pool().dispatch_batched(split_tasks.data(), (int)split_tasks.size());
        // Aggregate sub-task results back to original tasks
        size_t si = 0;
        for (int i = 0; i < n_tasks; ++i) {
            ssize_t total = 0;
            size_t orig_size = tasks[i].size;
            while (si < split_tasks.size() && total < (ssize_t)orig_size) {
                if (split_tasks[si].result < 0) { total = -1; break; }
                total += split_tasks[si].result;
                si++;
            }
            tasks[i].result = total;
        }
        return;
    }
    if (n_tasks == 1) {
        tasks[0].result = pread(tasks[0].fd, tasks[0].dst, tasks[0].size, tasks[0].offset);
        return;
    }
    bs_get_io_pool().dispatch(tasks, n_tasks);
}

// Same as bs_parallel_pread but uses the dedicated prefetch IO pool.
// Safe to call from the async prefetch background thread.
static void bs_prefetch_parallel_pread(bs_pread_task * tasks, int n_tasks, int io_split = 1) {
    if (n_tasks <= 0) return;
    if (io_split > 1) {
        std::vector<bs_pread_task> split_tasks;
        bs_split_pread_tasks(tasks, n_tasks, io_split, split_tasks);
        if ((int)split_tasks.size() <= 1) {
            split_tasks[0].result = pread(split_tasks[0].fd, split_tasks[0].dst,
                                          split_tasks[0].size, split_tasks[0].offset);
            tasks[0].result = split_tasks[0].result;
            return;
        }
        // Use batched dispatch — split may produce more tasks than MAX_WORKERS
        bs_get_prefetch_io_pool().dispatch_batched(split_tasks.data(), (int)split_tasks.size());
        size_t si = 0;
        for (int i = 0; i < n_tasks; ++i) {
            ssize_t total = 0;
            size_t orig_size = tasks[i].size;
            while (si < split_tasks.size() && total < (ssize_t)orig_size) {
                if (split_tasks[si].result < 0) { total = -1; break; }
                total += split_tasks[si].result;
                si++;
            }
            tasks[i].result = total;
        }
        return;
    }
    if (n_tasks == 1) {
        tasks[0].result = pread(tasks[0].fd, tasks[0].dst, tasks[0].size, tasks[0].offset);
        return;
    }
    bs_get_prefetch_io_pool().dispatch(tasks, n_tasks);
}

// Check whether a tensor name corresponds to a merged expert tensor.
// Merged expert tensors have names like:
//   blk.N.ffn_gate_exps, blk.N.ffn_up_exps, blk.N.ffn_down_exps
//   blk.N.ffn_norm_exps
// These are 3D tensors where ne[2] = n_expert.
static bool bs_is_gate_tensor(const std::string & name) {
    return name.find("ffn_gate_inp") != std::string::npos;
}

static bool bs_is_expert_tensor(const std::string & name) {
    // Match tensor names ending with "_exps" (the merged expert format)
    // but NOT "_shexp" (shared expert) or "_inp" (gate input)
    if (name.find("ffn_gate_exps") != std::string::npos) return true;
    if (name.find("ffn_up_exps")   != std::string::npos) return true;
    if (name.find("ffn_down_exps") != std::string::npos) return true;
    if (name.find("ffn_norm_exps") != std::string::npos) return true;
    return false;
}

// Extract layer index from a GGUF tensor name.
// Handles patterns like "blk.3.attn_norm.weight" or "layers.12.ffn_up.weight".
int llama_blurry_sharp_extract_layer_idx(const std::string & name) {
    // pattern: blk.<N>.  or  layers.<N>.
    static const std::regex re_blk(R"(blk\.(\d+)\.)");
    static const std::regex re_layers(R"(layers\.(\d+)\.)");
    std::smatch m;
    if (std::regex_search(name, m, re_blk)) {
        return std::stoi(m[1].str());
    }
    if (std::regex_search(name, m, re_layers)) {
        return std::stoi(m[1].str());
    }
    return -1;
}

// Check shape compatibility.
//
// We require a per-dimension match, not just the same total element count.
// Two tensors with the same number of elements but different shapes (e.g.
// [4096, 4096, 1, 1] vs [16384, 1024, 1, 1]) have incompatible memory
// layouts for quantized types — quantization blocks are organized row-by-row,
// so the kernel would read the data with wrong row boundaries, producing
// garbage or NaN.  A total-element-only check would silently accept tensors
// from a completely different model architecture that happen to share a
// tensor name and parameter count.
bool llama_blurry_sharp_shapes_compatible(
        const ggml_tensor           * blurry_tensor,
        const blurry_sharp_tensor_info & sharp_info) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (blurry_tensor->ne[d] != sharp_info.ne[d]) {
            return false;
        }
    }
    return true;
}

// Find a tensor in the base model by name.
// NOTE: This is O(n) in the number of model tensors.  Hot paths should use
// the cached blurry_sharp_tensor_info::base_tensor pointer instead.
static ggml_tensor * bs_find_model_tensor(llama_model * model, const char * name) {
    for (auto & kv : model->tensors_by_name) {
        if (kv.first == name) {
            return kv.second;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Per-layer mmap prefetch for selective sharpening
//
// Instead of prefetching the entire 60+ GiB sharp model (which would evict
// the blurry model from RAM), we prefetch only the specific layer's data
// ranges right before apply_layer processes them.  This gives the kernel a
// head start on reading ~1-2 GiB of data per layer from disk, while the
// MADV_DONTNEED calls in bs_read_to_staging / bs_read_sharp_tensor release
// each tensor's pages immediately after consumption.
//
// Net effect: RAM usage stays bounded to roughly one layer's worth of sharp
// data at a time, regardless of how many layers are sharpened.
// ---------------------------------------------------------------------------
static void bs_prefetch_layer_mmap(
        llama_blurry_sharp_context * bsctx,
        int                          layer_idx) {
#ifndef _WIN32
    auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
    if (lt_it == bsctx->layer_tensor_names.end()) return;

    for (const auto & tensor_name : lt_it->second) {
        // --- RAM cache swap-in prefetch ---
        // If this tensor's data is in the ram_cache (lazy swap or precache),
        // it may have been paged out to swap via MADV_PAGEOUT.  Issue
        // MADV_WILLNEED on the anonymous heap pages to trigger async swap-in.
        // This is the key optimisation for subsequent passes with lazy swap:
        // the kernel starts swapping pages in while we're still processing
        // earlier layers.
        if (bsctx->lazy_swap_enabled || bsctx->ram_cache_populated) {
            auto cache_it = bsctx->ram_cache.find(tensor_name);
            if (cache_it != bsctx->ram_cache.end() && !cache_it->second.empty()) {
                madvise(cache_it->second.data(), cache_it->second.size(), MADV_WILLNEED);
                continue;  // data is in ram_cache, no need for mmap prefetch
            }
        }

        // --- mmap page prefetch (first pass or no ram_cache) ---
        auto info_it = bsctx->sharp_index.find(tensor_name);
        if (info_it == bsctx->sharp_index.end()) continue;
        const auto & si = info_it->second;

        int split = si.split_idx;
        if (split < 0 || split >= (int)bsctx->sharp_mmaps.size()) continue;
        if (!bsctx->sharp_mmaps[split]) continue;

        const uint8_t * base = (const uint8_t *)bsctx->sharp_mmaps[split]->addr();
        size_t file_size = bsctx->sharp_mmaps[split]->size();
        size_t end = si.file_offset + si.nbytes;
        if (end > file_size) end = file_size;
        if (si.file_offset >= end) continue;

        posix_madvise((void *)(base + si.file_offset), end - si.file_offset,
                      POSIX_MADV_WILLNEED);
    }
#else
    GGML_UNUSED(bsctx);
    GGML_UNUSED(layer_idx);
#endif
}

// ---------------------------------------------------------------------------
// Async prefetch thread: pre-reads tensor data for upcoming layers
//
// On the first pass with lazy swap, ram_cache is empty.  Each tensor read
// blocks on disk I/O.  The prefetch thread reads ahead by several layers,
// populating prefetch_cache (separate from ram_cache to avoid main mutex
// contention).  When apply_layer/apply_experts needs a tensor, it checks
// prefetch_cache first and moves the data to ram_cache (O(1) move).
//
// On subsequent passes, the prefetch is handled by MADV_WILLNEED on
// ram_cache pages (see bs_prefetch_layer_mmap above).
// ---------------------------------------------------------------------------

static void bs_prefetch_thread_func(llama_blurry_sharp_context * bsctx) {
    while (!bsctx->prefetch_stop.load(std::memory_order_relaxed)) {
        std::vector<int> layers_to_prefetch;

        {
            std::unique_lock<std::mutex> lock(bsctx->prefetch_mtx);
            bsctx->prefetch_cv.wait(lock, [&] {
                return bsctx->prefetch_stop.load(std::memory_order_relaxed)
                    || !bsctx->prefetch_queue.empty();
            });
            if (bsctx->prefetch_stop.load(std::memory_order_relaxed)) break;
            layers_to_prefetch = std::move(bsctx->prefetch_queue);
            bsctx->prefetch_queue.clear();
        }

        for (int layer_idx : layers_to_prefetch) {
            if (bsctx->prefetch_stop.load(std::memory_order_relaxed)) break;

            auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
            if (lt_it == bsctx->layer_tensor_names.end()) continue;

            // Sort tensors by file offset for sequential I/O
            struct tensor_read_info {
                std::string name;
                int         split_idx;
                size_t      file_offset;
                size_t      nbytes;
            };
            std::vector<tensor_read_info> reads;

            for (const auto & tensor_name : lt_it->second) {
                // Skip if already in ram_cache or prefetch_cache
                {
                    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
                    if (bsctx->prefetch_cache.count(tensor_name)) continue;
                }
                // Check ram_cache (read-only check, safe without main mutex
                // since we only INSERT new keys and never delete during prefetch)
                if (bsctx->ram_cache.count(tensor_name)) continue;

                auto info_it = bsctx->sharp_index.find(tensor_name);
                if (info_it == bsctx->sharp_index.end()) continue;
                const auto & si = info_it->second;

                reads.push_back({tensor_name, si.split_idx, si.file_offset, si.nbytes});
            }

            // Sort by (split_idx, file_offset) for sequential disk I/O
            std::sort(reads.begin(), reads.end(), [](const tensor_read_info & a, const tensor_read_info & b) {
                if (a.split_idx != b.split_idx) return a.split_idx < b.split_idx;
                return a.file_offset < b.file_offset;
            });

            for (const auto & ri : reads) {
                if (bsctx->prefetch_stop.load(std::memory_order_relaxed)) break;

                // Double-check not already cached (may have been populated by main thread)
                {
                    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
                    if (bsctx->prefetch_cache.count(ri.name)) continue;
                }
                if (bsctx->ram_cache.count(ri.name)) continue;

                std::vector<uint8_t> buf(ri.nbytes);
                bool ok = false;

                // Read from mmap if available (sequential thanks to sorting)
                int si = ri.split_idx;
                if (si >= 0 && si < (int)bsctx->sharp_mmaps.size() && bsctx->sharp_mmaps[si]) {
                    const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
                    size_t file_size = bsctx->sharp_mmaps[si]->size();
                    if (ri.file_offset + ri.nbytes <= file_size) {
                        std::memcpy(buf.data(), mmap_base + ri.file_offset, ri.nbytes);
                        ok = true;
                    }
                }

                // Fallback: file read
                if (!ok && si >= 0 && si < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si]) {
                    // File reads need serialization — only the prefetch thread
                    // reads from these file handles during prefetch, so no
                    // additional locking needed (apply path uses mmap when available).
                    bsctx->sharp_files[si]->seek(ri.file_offset, SEEK_SET);
                    bsctx->sharp_files[si]->read_raw(buf.data(), ri.nbytes);
                    ok = true;
                }

                if (ok) {
                    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
                    bsctx->prefetch_cache[ri.name] = std::move(buf);
                    bsctx->prefetch_cache_bytes += (int64_t)ri.nbytes;
                }
            }
        }
    }
}

// Start the prefetch thread if lazy swap is enabled and not already started.
static void bs_ensure_prefetch_thread(llama_blurry_sharp_context * bsctx) {
    if (!bsctx->lazy_swap_enabled) return;
    if (bsctx->prefetch_thread_started) return;

    bsctx->prefetch_stop.store(false, std::memory_order_relaxed);
    bsctx->prefetch_thread = std::thread(bs_prefetch_thread_func, bsctx);
    bsctx->prefetch_thread_started = true;

    LLAMA_LOG_INFO("%s: async prefetch thread started\n", __func__);
}

// Stop the prefetch thread gracefully.
static void bs_stop_prefetch_thread(llama_blurry_sharp_context * bsctx) {
    if (!bsctx->prefetch_thread_started) return;

    bsctx->prefetch_stop.store(true, std::memory_order_relaxed);
    bsctx->prefetch_cv.notify_all();
    if (bsctx->prefetch_thread.joinable()) {
        bsctx->prefetch_thread.join();
    }
    bsctx->prefetch_thread_started = false;

    // Move any remaining prefetch_cache entries into ram_cache
    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
    for (auto & kv : bsctx->prefetch_cache) {
        if (bsctx->ram_cache.count(kv.first) == 0) {
            bsctx->ram_cache_bytes += (int64_t)kv.second.size();
            bsctx->ram_cache[kv.first] = std::move(kv.second);
        }
    }
    bsctx->prefetch_cache.clear();
    bsctx->prefetch_cache_bytes = 0;
}

// Queue layers for async prefetch.  Called from the public prefetch API
// or internally before apply loops.  The thread will process them in order.
static void bs_queue_prefetch_layers(
        llama_blurry_sharp_context * bsctx,
        const int                  * layer_indices,
        int                          n_layers) {
    if (!bsctx->prefetch_thread_started) return;
    if (n_layers <= 0) return;

    {
        std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
        for (int i = 0; i < n_layers; ++i) {
            bsctx->prefetch_queue.push_back(layer_indices[i]);
        }
    }
    bsctx->prefetch_cv.notify_one();
}

// Check prefetch_cache for a tensor and move it to ram_cache if found.
// Called from the lazy swap population path in bs_overlay_single_tensor.
// Returns true if data was moved to ram_cache.
static bool bs_consume_prefetched_tensor(
        llama_blurry_sharp_context * bsctx,
        const std::string          & tensor_name,
        size_t                       expected_nbytes) {
    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
    auto it = bsctx->prefetch_cache.find(tensor_name);
    if (it == bsctx->prefetch_cache.end()) return false;
    if (it->second.size() < expected_nbytes) return false;

    // Move to ram_cache (O(1) vector move)
    bsctx->ram_cache[tensor_name] = std::move(it->second);
    bsctx->ram_cache_bytes += (int64_t)expected_nbytes;
    bsctx->prefetch_cache_bytes -= (int64_t)expected_nbytes;
    bsctx->prefetch_cache.erase(it);

    return true;
}

// ---------------------------------------------------------------------------
// Memory-tier helpers: GPU > RAM > Swap > Disk
//
// The sharp model can be 60+ GiB.  There are two pathologies to avoid:
//
//   1. MMAP BLOAT: Reading via mmap fills the page cache with file-backed
//      sharp pages, evicting the blurry model (and everything else) into
//      swap.  Every subsequent access then hits disk through swap — slow.
//
//   2. MMAP DROP: Calling MADV_DONTNEED on file-backed pages simply drops
//      them — they must be re-faulted from the GGUF file on disk, which
//      can be extremely slow for random access on HDDs or network storage.
//
// Strategy (with RAM cache enabled):
//   - Pre-read sharp tensor data into anonymous heap buffers at init time.
//   - Anonymous pages go to SWAP under pressure (not dropped entirely).
//   - Re-access from swap (SSD) is 10-50x faster than from GGUF on disk.
//   - Still release file-backed mmap pages after copying to avoid bloat.
//   - Use MADV_PAGEOUT to proactively stage anonymous pages to swap,
//     freeing RAM for active data (blurry model, KV cache).
//
// Strategy (without RAM cache — legacy behavior):
//   - Release mmap pages with MADV_DONTNEED after consumption.
//   - Keeps RSS bounded but re-access requires full disk I/O.
// ---------------------------------------------------------------------------

// Release file-backed mmap pages back to the OS.
// For clean (unmodified) file-backed pages this simply drops them from the
// page cache — no swap I/O, no data loss.  Next access re-faults from the
// backing file.  Safe to call on any address; silently ignored if the range
// isn't a valid mapping.
// When bsctx->params.retain_mmap_pages is true, skip the DONTNEED — keep
// pages in the page cache so repeated sharpen/restore cycles hit RAM
// instead of disk.  Pass NULL for bsctx when the context is unavailable
// (always releases in that case).
static void bs_release_mmap_pages(const llama_blurry_sharp_context * bsctx,
                                  const void * addr, size_t len) {
    if (bsctx && bsctx->params.retain_mmap_pages) return;
#ifdef __linux__
    if (!addr || len == 0) return;
    static const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t start = (uintptr_t)addr & ~(uintptr_t)(page_size - 1);
    uintptr_t end   = ((uintptr_t)addr + len + page_size - 1)
                    & ~(uintptr_t)(page_size - 1);
    madvise((void *)start, end - start, MADV_DONTNEED);
#elif defined(__APPLE__)
    if (!addr || len == 0) return;
    madvise((void *)addr, len, MADV_DONTNEED);
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(len);
#endif
}

// Compute the byte size of a tensor with the given type and shape.
static size_t bs_tensor_nbytes(ggml_type type, const int64_t ne[GGML_MAX_DIMS]) {
    // ggml row_size already handles quantized block sizes
    int64_t nrows = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        nrows *= ne[d];
    }
    return ggml_row_size(type, ne[0]) * (size_t)nrows;
}

// Compute the number of elements from a shape array.
static int64_t bs_nelements(const int64_t ne[GGML_MAX_DIMS]) {
    int64_t n = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        n *= ne[d];
    }
    return n;
}

// ---------------------------------------------------------------------------
// Helper: detect CUDA split buffers and obtain a usable (regular) buffer type.
//
// In ik_llama.cpp, GPU-offloaded weight tensors are typically placed in a
// "split buffer" designed for multi-GPU tensor parallelism.  Split buffers
// have fundamentally different semantics from regular CUDA buffers:
//
//   - alloc_buffer()  → allocates NO device memory (just a context object)
//   - get_base()      → returns 0x1000 (a dummy, never-dereference pointer)
//   - clear()         → NO-OP
//   - set_tensor()    → writes to the ORIGINAL device memory via tensor->extra
//
// If the overlay naively uses the split buffer type, tensor->data ends up as
// 0x1000 (invalid) and the CUDA compute kernel reads garbage → all NaN.
//
// This helper detects split buffers by name and returns the regular CUDA
// buffer type for the model's main GPU instead.
// ---------------------------------------------------------------------------
static ggml_backend_buffer_type_t bs_get_device_buft(
        const llama_blurry_sharp_context * bsctx,
        ggml_backend_buffer_t              original_buffer) {
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(original_buffer);

#ifdef GGML_USE_CUDA
    // Check if this is a split buffer by examining its name.
    // Split buffer names contain "_Split" (e.g. "CUDA_Split").
    const char * buf_name = ggml_backend_buffer_name(original_buffer);
    if (buf_name && strstr(buf_name, "Split")) {
        int gpu = 0;
        if (bsctx->model) {
            gpu = bsctx->model->main_gpu;
        }
        ggml_backend_buffer_type_t regular_buft = ggml_backend_cuda_buffer_type(gpu);
        if (regular_buft) {
            buft = regular_buft;
        }
    }
#else
    GGML_UNUSED(bsctx);
#endif

    return buft;
}

// Read tensor data from the appropriate sharp split file (via RAM cache, mmap, or file read).
static bool bs_read_sharp_tensor(
        llama_blurry_sharp_context * bsctx,
        const blurry_sharp_tensor_info & info,
        std::vector<uint8_t> & dest) {
    dest.resize(info.nbytes);

    // Fast path: RAM cache hit — data is in anonymous heap memory (swap-backed).
    // This avoids mmap page faults and file I/O entirely.  If the pages were
    // swapped out by the OS, swap-in from SSD is much faster than re-reading
    // from the GGUF file.
    if (bsctx->ram_cache_populated) {
        auto cache_it = bsctx->ram_cache.find(info.name);
        if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= info.nbytes) {
            std::memcpy(dest.data(), cache_it->second.data(), info.nbytes);
            return true;
        }
    }

    int si = info.split_idx;

    // Bounds-check the split index
    if (si < 0 || si >= (int)bsctx->sharp_files.size() || !bsctx->sharp_files[si]) {
        LLAMA_LOG_ERROR("%s: no file handle for split %d (tensor '%s')\n",
                       __func__, si, info.name.c_str());
        return false;
    }

    // Try mmap first, fall back to file read
    if (si < (int)bsctx->sharp_mmaps.size() && bsctx->sharp_mmaps[si]) {
        const uint8_t * base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
        if (info.file_offset + info.nbytes > bsctx->sharp_mmaps[si]->size()) {
            LLAMA_LOG_ERROR("%s: mmap read out of bounds for tensor '%s' "
                           "(split=%d, offset=%zu, nbytes=%zu, file_size=%zu)\n",
                           __func__, info.name.c_str(), si,
                           info.file_offset, info.nbytes,
                           bsctx->sharp_mmaps[si]->size());
            return false;
        }
        std::memcpy(dest.data(), base + info.file_offset, info.nbytes);
        // Data is now in `dest` — release the mmap pages so they don't
        // accumulate in the page cache and push other data into swap.
        bs_release_mmap_pages(bsctx, base + info.file_offset, info.nbytes);
    } else {
        bsctx->sharp_files[si]->seek(info.file_offset, SEEK_SET);
        bsctx->sharp_files[si]->read_raw(dest.data(), info.nbytes);
    }
    return true;
}

// Read sharp tensor data directly into a staging buffer suitable for GPU upload.
// When RAM cache or mmap + pinned staging are available, this copies directly
// into pinned memory in a single memcpy, skipping the intermediate heap buffer.
// Returns a pointer to the staging data, or nullptr on failure.
static const void * bs_read_to_staging(
        llama_blurry_sharp_context       * bsctx,
        const blurry_sharp_tensor_info   & info) {
    int si = info.split_idx;

    // Fastest path: RAM cache → pinned staging (one memcpy, data from swap-backed heap)
    if ((bsctx->ram_cache_populated || bsctx->lazy_swap_enabled) && bsctx->pinned_staging_buf && info.nbytes <= bsctx->pinned_staging_size) {
        auto cache_it = bsctx->ram_cache.find(info.name);
        if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= info.nbytes) {
            memcpy(bsctx->pinned_staging_ptr, cache_it->second.data(), info.nbytes);
            return bsctx->pinned_staging_ptr;
        }
    }

    // Fast path: RAM cache → direct return (no pinned staging, OR tensor larger than staging)
    // Anonymous heap memory works fine as a source for ggml_backend_tensor_set —
    // it's just slower than pinned memory for DMA transfers.
    if (bsctx->ram_cache_populated || bsctx->lazy_swap_enabled) {
        auto cache_it = bsctx->ram_cache.find(info.name);
        if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= info.nbytes) {
            return cache_it->second.data();
        }
    }

    // Lazy swap: populate the RAM cache on demand for device tensors too.
    // This ensures GPU tensor data is also swap-backed after the first access,
    // so subsequent sharpen cycles read from swap (fast SSD) instead of the
    // GGUF file (slow).  The data is read once from disk into anonymous heap
    // memory, then staged to swap on restore via MADV_PAGEOUT.
    if (bsctx->lazy_swap_enabled) {
        // Fast path: check if the async prefetch thread already has it
        bool ok = bs_consume_prefetched_tensor(bsctx, info.name, info.nbytes);

        if (!ok) {
            std::vector<uint8_t> buf(info.nbytes);

            bool have_mmap = (si >= 0 && si < (int)bsctx->sharp_mmaps.size()
                              && bsctx->sharp_mmaps[si]);
            if (have_mmap) {
                const uint8_t * base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
                if (info.file_offset + info.nbytes <= bsctx->sharp_mmaps[si]->size()) {
                    std::memcpy(buf.data(), base + info.file_offset, info.nbytes);
                    bs_release_mmap_pages(bsctx, base + info.file_offset, info.nbytes);
                    ok = true;
                }
            }
            if (!ok && si >= 0 && si < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si]) {
                bsctx->sharp_files[si]->seek(info.file_offset, SEEK_SET);
                bsctx->sharp_files[si]->read_raw(buf.data(), info.nbytes);
                ok = true;
            }

            if (ok) {
                bsctx->ram_cache[info.name] = std::move(buf);
                bsctx->ram_cache_bytes += (int64_t)info.nbytes;
            }
        }

        if (ok) {

            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: lazy-swap: cached device tensor '%s' (%zu bytes, "
                              "total cache %.2f MiB)\n",
                              __func__, info.name.c_str(), info.nbytes,
                              bsctx->ram_cache_bytes / (1024.0 * 1024.0));
            }

            // Now use the freshly cached data via the normal paths above
            auto cache_it = bsctx->ram_cache.find(info.name);
            if (cache_it != bsctx->ram_cache.end()) {
                if (bsctx->pinned_staging_buf && info.nbytes <= bsctx->pinned_staging_size) {
                    memcpy(bsctx->pinned_staging_ptr, cache_it->second.data(), info.nbytes);
                    return bsctx->pinned_staging_ptr;
                }
                return cache_it->second.data();
            }
        }
    }

    // Fast path: mmap → pinned staging (one memcpy instead of two)
    if (bsctx->pinned_staging_buf && info.nbytes <= bsctx->pinned_staging_size) {
        bool have_mmap = (si >= 0 && si < (int)bsctx->sharp_mmaps.size()
                          && bsctx->sharp_mmaps[si]);
        if (have_mmap) {
            const uint8_t * base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
            if (info.file_offset + info.nbytes <= bsctx->sharp_mmaps[si]->size()) {
                memcpy(bsctx->pinned_staging_ptr, base + info.file_offset, info.nbytes);
                // Data is now in pinned staging — release mmap pages immediately
                // to avoid accumulating the entire sharp model in the page cache.
                bs_release_mmap_pages(bsctx, base + info.file_offset, info.nbytes);
                return bsctx->pinned_staging_ptr;
            }
        }
        // Fallback: file read → read_buf → pinned
        if (bs_read_sharp_tensor(bsctx, info, bsctx->read_buf)) {
            memcpy(bsctx->pinned_staging_ptr, bsctx->read_buf.data(), info.nbytes);
            return bsctx->pinned_staging_ptr;
        }
        return nullptr;
    }

    // No pinned staging: read into heap buffer and use that directly
    if (bs_read_sharp_tensor(bsctx, info, bsctx->read_buf)) {
        return bsctx->read_buf.data();
    }
    return nullptr;
}

// Dequantize a quantized buffer to f32.
// Returns the number of floats written.
static int64_t bs_dequantize_to_f32(
        ggml_type         src_type,
        const void      * src_data,
        size_t            src_nbytes,
        const int64_t     ne[GGML_MAX_DIMS],
        std::vector<float> & dest) {
    int64_t nel = bs_nelements(ne);
    dest.resize((size_t)nel);

    if (src_type == GGML_TYPE_F32) {
        std::memcpy(dest.data(), src_data, nel * sizeof(float));
        return nel;
    }

    if (src_type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)src_data;
        for (int64_t i = 0; i < nel; ++i) {
            dest[(size_t)i] = ggml_fp16_to_fp32(src[i]);
        }
        return nel;
    }

    // Use ggml type traits to_float
    ggml_type_traits_t traits = ggml_internal_get_type_traits(src_type);
    if (!traits.to_float) {
        LLAMA_LOG_ERROR("%s: no to_float for type %s\n",
                        __func__, ggml_type_name(src_type));
        return -1;
    }

    // Dequantize row by row
    int64_t n_rows = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        n_rows *= ne[d];
    }
    int64_t row_elements = ne[0];
    size_t  src_row_size = ggml_row_size(src_type, row_elements);

    const uint8_t * src_ptr = (const uint8_t *)src_data;
    float * dst_ptr = dest.data();

    for (int64_t row = 0; row < n_rows; ++row) {
        traits.to_float(src_ptr, dst_ptr, (int)row_elements);
        src_ptr += src_row_size;
        dst_ptr += row_elements;
    }

    return nel;
}

// Quantize f32 data to a target quantized type.
// Returns the number of bytes written.
static size_t bs_quantize_from_f32(
        ggml_type           dst_type,
        const float       * src_data,
        int64_t             nel,
        const int64_t       ne[GGML_MAX_DIMS],
        std::vector<uint8_t> & dest) {
    if (dst_type == GGML_TYPE_F32) {
        size_t sz = (size_t)nel * sizeof(float);
        dest.resize(sz);
        std::memcpy(dest.data(), src_data, sz);
        return sz;
    }

    if (dst_type == GGML_TYPE_F16) {
        size_t sz = (size_t)nel * sizeof(ggml_fp16_t);
        dest.resize(sz);
        ggml_fp16_t * dst = (ggml_fp16_t *)dest.data();
        for (int64_t i = 0; i < nel; ++i) {
            dst[i] = ggml_fp32_to_fp16(src_data[i]);
        }
        return sz;
    }

    // Use ggml_quantize_chunk for quantized types
    size_t dst_nbytes = bs_tensor_nbytes(dst_type, ne);
    dest.resize(dst_nbytes);

    // ggml_quantize_chunk expects (type, src_f32, dst, start, nrows, n_per_row, imatrix)
    int64_t n_rows = 1;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        n_rows *= ne[d];
    }
    int64_t n_per_row = ne[0];

    ggml_quantize_init(dst_type);
    size_t result = ggml_quantize_chunk(
        dst_type,
        src_data,
        dest.data(),
        0,           // start
        (int)n_rows,
        (int)n_per_row,
        nullptr      // no importance matrix
    );

    return result;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

llama_blurry_sharp_context * llama_blurry_sharp_init(
        llama_model              * model,
        llama_blurry_sharp_params  params) {

    if (!model) {
        LLAMA_LOG_ERROR("%s: model is null\n", __func__);
        return nullptr;
    }
    const bool has_sharp = params.sharp_model_path && params.sharp_model_path[0] != '\0';
    const bool has_delta = params.n_delta_levels > 0 && params.delta_paths;
    if (!has_sharp && !has_delta) {
        LLAMA_LOG_ERROR("%s: neither sharp_model_path nor delta_paths specified\n", __func__);
        return nullptr;
    }

    if (has_delta && !has_sharp) {
        LLAMA_LOG_INFO("%s: initializing delta-only correction (no sharp model)\n", __func__);
    } else {
        LLAMA_LOG_INFO("%s: initializing blurry-sharp overlay system\n", __func__);
        LLAMA_LOG_INFO("%s: sharp model path: %s\n", __func__, params.sharp_model_path);
    }
    if (params.permanent) {
        LLAMA_LOG_INFO("%s: PERMANENT mode enabled — sharp overlay is one-way, blurry data will be discarded\n", __func__);
        LLAMA_LOG_INFO("%s:   same-type GPU tensors: in-place overwrite (zero extra VRAM)\n", __func__);
        LLAMA_LOG_INFO("%s:   different-type GPU tensors: new buffer, no backup\n", __func__);
        LLAMA_LOG_INFO("%s:   CPU tensors: pointer-swap to sharp mmap, release blurry pages\n", __func__);
    }
    if (params.flash_experts) {
        LLAMA_LOG_INFO("%s: FLASH-EXPERT mode enabled — stream Q4_K_M experts from SSD on demand\n", __func__);
        LLAMA_LOG_INFO("%s:   blurry expert data will be released (MADV_DONTNEED)\n", __func__);
        LLAMA_LOG_INFO("%s:   all expert compute at sharp (Q4_K_M) quality\n", __func__);
        LLAMA_LOG_INFO("%s:   no backup/restore overhead — OS page cache handles caching\n", __func__);
    }

    auto * bsctx = new llama_blurry_sharp_context();
    bsctx->model  = model;
    bsctx->params = params;
    bsctx->retain_device_buffers = params.retain_device_buffers;
    bsctx->lazy_swap_enabled     = params.lazy_swap;

    // -----------------------------------------------------------------------
    // Sharp model loading (steps 1-7) — only when --sharp is specified
    // -----------------------------------------------------------------------
    if (has_sharp) {

    // -----------------------------------------------------------------------
    // 1) Open the primary sharp GGUF and read metadata (no tensor data alloc)
    // -----------------------------------------------------------------------
    {
        ggml_context * tensor_ctx = nullptr;
        struct gguf_init_params gguf_params = {
            /* .no_alloc = */ true,
            /* .ctx      = */ &tensor_ctx,
        };
        gguf_context * primary_gguf = gguf_init_from_file(params.sharp_model_path, gguf_params);
        if (!primary_gguf) {
            LLAMA_LOG_ERROR("%s: failed to open sharp GGUF file '%s'\n",
                           __func__, params.sharp_model_path);
            delete bsctx;
            return nullptr;
        }
        bsctx->sharp_ggufs.push_back(primary_gguf);
        bsctx->sharp_tensor_ctxs.push_back(tensor_ctx);
    }

    // -----------------------------------------------------------------------
    // 2) Open primary sharp file for data reading
    // -----------------------------------------------------------------------
    try {
        bsctx->sharp_files.push_back(
            std::make_unique<llama_file>(params.sharp_model_path, "rb"));
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: failed to open sharp file for reading: %s\n",
                       __func__, e.what());
        gguf_free(bsctx->sharp_ggufs[0]);
        if (bsctx->sharp_tensor_ctxs[0]) ggml_free(bsctx->sharp_tensor_ctxs[0]);
        delete bsctx;
        return nullptr;
    }

    // -----------------------------------------------------------------------
    // 2b) Detect split GGUF and open additional split files
    // -----------------------------------------------------------------------
    {
        // Check if the primary GGUF declares a split count > 1
        int n_split_key = gguf_find_key(bsctx->sharp_ggufs[0], "split.count");
        uint16_t n_split = 1;
        if (n_split_key >= 0) {
            n_split = (uint16_t)gguf_get_val_u16(bsctx->sharp_ggufs[0], n_split_key);
        }
        bsctx->n_sharp_splits = (int)n_split;

        if (n_split > 1) {
            // Derive the split prefix from the primary filename
            char split_prefix[PATH_MAX] = {0};
            if (!llama_split_prefix(split_prefix, sizeof(split_prefix),
                                    params.sharp_model_path, 0, n_split)) {
                LLAMA_LOG_ERROR("%s: could not extract split prefix from '%s'\n",
                               __func__, params.sharp_model_path);
                // Fall back to treating it as a single file
                bsctx->n_sharp_splits = 1;
            } else {
                LLAMA_LOG_INFO("%s: sharp model is split across %d files\n",
                              __func__, n_split);

                for (uint16_t idx = 1; idx < n_split; ++idx) {
                    char split_path[PATH_MAX] = {0};
                    llama_split_path(split_path, sizeof(split_path),
                                    split_prefix, idx, n_split);

                    // Open GGUF metadata for this split
                    ggml_context * split_tensor_ctx = nullptr;
                    struct gguf_init_params split_gp = {
                        /* .no_alloc = */ true,
                        /* .ctx      = */ &split_tensor_ctx,
                    };
                    gguf_context * split_gguf = gguf_init_from_file(split_path, split_gp);
                    if (!split_gguf) {
                        LLAMA_LOG_ERROR("%s: failed to open split GGUF '%s'\n",
                                       __func__, split_path);
                        // Continue without this split — tensors in it won't be available
                        bsctx->sharp_ggufs.push_back(nullptr);
                        bsctx->sharp_tensor_ctxs.push_back(nullptr);
                        bsctx->sharp_files.push_back(nullptr);
                        continue;
                    }

                    bsctx->sharp_ggufs.push_back(split_gguf);
                    bsctx->sharp_tensor_ctxs.push_back(split_tensor_ctx);

                    // Open file handle for this split
                    try {
                        bsctx->sharp_files.push_back(
                            std::make_unique<llama_file>(split_path, "rb"));
                    } catch (const std::exception & e) {
                        LLAMA_LOG_WARN("%s: failed to open split file '%s': %s\n",
                                      __func__, split_path, e.what());
                        bsctx->sharp_files.push_back(nullptr);
                    }

                    LLAMA_LOG_INFO("%s: opened split %d/%d: %s\n",
                                  __func__, idx + 1, n_split, split_path);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3) Optionally mmap each sharp file (prefetch = 0 → demand-paged)
    // -----------------------------------------------------------------------
    if (params.use_mmap && llama_mmap::SUPPORTED) {
        size_t total_mmap_size = 0;
        for (size_t si = 0; si < bsctx->sharp_files.size(); ++si) {
            if (!bsctx->sharp_files[si]) {
                bsctx->sharp_mmaps.push_back(nullptr);
                continue;
            }
            try {
                auto mmap = std::make_unique<llama_mmap>(
                    bsctx->sharp_files[si].get(),
                    0,     // prefetch = 0 (don't prefetch everything)
                    false  // numa = false
                );
                total_mmap_size += mmap->size();
                bsctx->sharp_mmaps.push_back(std::move(mmap));
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("%s: mmap failed for sharp split %zu, "
                              "falling back to file reads: %s\n",
                              __func__, si, e.what());
                bsctx->sharp_mmaps.push_back(nullptr);
            }
        }
        LLAMA_LOG_INFO("%s: sharp model mmap'd (%zu bytes across %zu file%s)\n",
                      __func__, total_mmap_size,
                      bsctx->sharp_files.size(),
                      bsctx->sharp_files.size() > 1 ? "s" : "");
    } else {
        // Fill with nullptrs so indexing by split_idx works
        for (size_t si = 0; si < bsctx->sharp_files.size(); ++si) {
            bsctx->sharp_mmaps.push_back(nullptr);
        }
    }

    // -----------------------------------------------------------------------
    // 4) Build the tensor index (across all splits)
    // -----------------------------------------------------------------------
    {
        int total_tensors = 0;

        for (int si = 0; si < (int)bsctx->sharp_ggufs.size(); ++si) {
            gguf_context  * sgguf = bsctx->sharp_ggufs[si];
            ggml_context  * stctx = bsctx->sharp_tensor_ctxs[si];
            if (!sgguf || !stctx) continue;

            int n_tensors = gguf_get_n_tensors(sgguf);
            size_t data_offset = gguf_get_data_offset(sgguf);

            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(sgguf, i);
                ggml_type    type = gguf_get_tensor_type(sgguf, i);
                size_t       tensor_offset = gguf_get_tensor_offset(sgguf, i);

                blurry_sharp_tensor_info info;
                info.name        = name;
                info.tensor_idx  = i;
                info.split_idx   = si;
                info.file_offset = data_offset + tensor_offset;
                info.type        = type;

                // Get shape from the tensor metadata context
                ggml_tensor * meta = ggml_get_tensor(stctx, name);
                if (meta) {
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        info.ne[d] = meta->ne[d];
                    }
                    info.nbytes = ggml_nbytes(meta);
                } else {
                    LLAMA_LOG_WARN("%s: tensor '%s' (split %d) found in GGUF index "
                                  "but not in context, skipping\n",
                                  __func__, name, si);
                    continue;
                }

                info.layer_idx = llama_blurry_sharp_extract_layer_idx(info.name);

                bsctx->sharp_index[info.name] = info;

                if (info.layer_idx >= 0) {
                    bsctx->layer_tensor_names[info.layer_idx].push_back(info.name);
                }

                if (params.verbose) {
                    LLAMA_LOG_INFO("%s:   tensor %-50s  split=%d  type=%-8s  "
                                  "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]  "
                                  "bytes=%zu  layer=%d\n",
                                  __func__, name, si, ggml_type_name(type),
                                  info.ne[0], info.ne[1], info.ne[2], info.ne[3],
                                  info.nbytes, info.layer_idx);
                }

                ++total_tensors;
            }
        }

        LLAMA_LOG_INFO("%s: sharp model contains %d tensors across %d split%s\n",
                      __func__, total_tensors, bsctx->n_sharp_splits,
                      bsctx->n_sharp_splits > 1 ? "s" : "");
    }

    // -----------------------------------------------------------------------
    // 5) Collect sorted layer indices present in sharp model
    // -----------------------------------------------------------------------
    {
        std::set<int> idx_set;
        for (auto & kv : bsctx->layer_tensor_names) {
            idx_set.insert(kv.first);
        }
        bsctx->sharp_layer_indices.assign(idx_set.begin(), idx_set.end());
        LLAMA_LOG_INFO("%s: sharp model has %zu layer groups (layers %d..%d)\n",
                      __func__,
                      bsctx->sharp_layer_indices.size(),
                      bsctx->sharp_layer_indices.empty() ? -1 : bsctx->sharp_layer_indices.front(),
                      bsctx->sharp_layer_indices.empty() ? -1 : bsctx->sharp_layer_indices.back());
    }

    // -----------------------------------------------------------------------
    // 6) Determine eligible layers
    // -----------------------------------------------------------------------
    {
        // Start with all layers present in the sharp model
        for (int idx : bsctx->sharp_layer_indices) {
            bsctx->eligible_layers.insert(idx);
        }

        // Apply allowlist (if any)
        if (params.n_layer_allowlist > 0 && params.layer_allowlist) {
            std::unordered_set<int> allow;
            for (int i = 0; i < params.n_layer_allowlist; ++i) {
                allow.insert(params.layer_allowlist[i]);
            }
            for (auto it = bsctx->eligible_layers.begin(); it != bsctx->eligible_layers.end(); ) {
                if (allow.find(*it) == allow.end()) {
                    it = bsctx->eligible_layers.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Apply denylist
        if (params.n_layer_denylist > 0 && params.layer_denylist) {
            for (int i = 0; i < params.n_layer_denylist; ++i) {
                bsctx->eligible_layers.erase(params.layer_denylist[i]);
            }
        }

        // Also filter out layers that don't exist in the base model
        int n_model_layers = (int)model->layers.size();
        for (auto it = bsctx->eligible_layers.begin(); it != bsctx->eligible_layers.end(); ) {
            if (*it < 0 || *it >= n_model_layers) {
                LLAMA_LOG_WARN("%s: layer %d is in sharp model but not in base model (%d layers), skipping\n",
                              __func__, *it, n_model_layers);
                it = bsctx->eligible_layers.erase(it);
            } else {
                ++it;
            }
        }

        LLAMA_LOG_INFO("%s: %zu layers eligible for sharpening\n",
                      __func__, bsctx->eligible_layers.size());
    }

    // -----------------------------------------------------------------------
    // 7) Validate: for each eligible layer, check that we can find matching
    //    tensors in the base model with compatible shapes.
    // -----------------------------------------------------------------------
    {
        int n_matched = 0;
        int n_skipped = 0;
        int n_skip_no_base = 0;
        int n_skip_shape   = 0;
        std::vector<std::string> skipped_names;

        for (int layer_idx : bsctx->sharp_layer_indices) {
            if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
                continue;
            }
            auto & tensor_names = bsctx->layer_tensor_names[layer_idx];
            for (auto it = tensor_names.begin(); it != tensor_names.end(); ) {
                ggml_tensor * base_t = bs_find_model_tensor(model, it->c_str());
                if (!base_t) {
                    LLAMA_LOG_INFO("%s: skip (no base match): '%s'\n",
                                  __func__, it->c_str());
                    skipped_names.push_back(*it);
                    bsctx->sharp_index.erase(*it);
                    it = tensor_names.erase(it);
                    ++n_skipped;
                    ++n_skip_no_base;
                    continue;
                }

                auto & info = bsctx->sharp_index[*it];

                // Cache the base tensor pointer so apply/restore can skip O(n) lookup
                info.base_tensor = base_t;

                if (!llama_blurry_sharp_shapes_compatible(base_t, info)) {
                    LLAMA_LOG_WARN("%s: skip (shape mismatch): '%s' "
                                  "base=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "](%s) vs "
                                  "sharp=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "](%s)\n",
                                  __func__, it->c_str(),
                                  base_t->ne[0], base_t->ne[1], base_t->ne[2], base_t->ne[3],
                                  ggml_type_name(base_t->type),
                                  info.ne[0], info.ne[1], info.ne[2], info.ne[3],
                                  ggml_type_name(info.type));
                    skipped_names.push_back(*it);
                    bsctx->sharp_index.erase(*it);
                    it = tensor_names.erase(it);
                    ++n_skipped;
                    ++n_skip_shape;
                    continue;
                }

                ++n_matched;
                ++it;
            }

            // If no tensors remain for this layer, remove eligibility
            if (tensor_names.empty()) {
                bsctx->eligible_layers.erase(layer_idx);
            }
        }

        // Count sharp tensors that are NOT in any layer (layer_idx == -1)
        // These are non-layer tensors (embeddings, output head, norms, etc.)
        // that exist in the sharp GGUF but are never overlaid.
        int n_non_layer_sharp = 0;
        for (auto & kv : bsctx->sharp_index) {
            if (kv.second.layer_idx < 0) {
                ++n_non_layer_sharp;
            }
        }

        LLAMA_LOG_INFO("%s: %d tensors matched between blurry and sharp models, %d skipped\n",
                      __func__, n_matched, n_skipped);
        if (n_skipped > 0) {
            LLAMA_LOG_INFO("%s:   skip breakdown: %d no base match, %d shape mismatch\n",
                          __func__, n_skip_no_base, n_skip_shape);
        }
        if (n_non_layer_sharp > 0) {
            LLAMA_LOG_INFO("%s: %d non-layer tensors in sharp model (not overlaid: embeddings, output head, etc.)\n",
                          __func__, n_non_layer_sharp);
        }

        // Check for layers with partial tensor coverage (some tensors matched, some didn't)
        for (int layer_idx : bsctx->sharp_layer_indices) {
            if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
                continue;
            }
            auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
            if (lt_it == bsctx->layer_tensor_names.end()) continue;

            int n_layer_tensors = (int)lt_it->second.size();
            // Count how many skipped tensors belonged to this layer
            int n_layer_skipped = 0;
            for (const auto & sn : skipped_names) {
                int skip_layer = llama_blurry_sharp_extract_layer_idx(sn);
                if (skip_layer == layer_idx) ++n_layer_skipped;
            }
            if (n_layer_skipped > 0) {
                LLAMA_LOG_WARN("%s: layer %d has partial coverage: %d tensors matched, "
                              "%d skipped (may produce inconsistent results)\n",
                              __func__, layer_idx, n_layer_tensors, n_layer_skipped);
            }
        }
    }

    if (bsctx->eligible_layers.empty()) {
        LLAMA_LOG_WARN("%s: no eligible layers found for blurry-sharp overlay\n", __func__);
        // Still return the context; it's valid but will be a no-op
    }

    // NOTE: No mmap safety check needed.  The pointer-swap overlay strategy
    // never writes to the base model's tensor buffers — it swaps the data
    // pointer to an owned sharp-data buffer instead.  So mmap'd (read-only)
    // base models work fine.

    // -----------------------------------------------------------------------
    // Allocate pinned staging buffer for fast CPU→GPU copies
    // -----------------------------------------------------------------------
    // Find the largest sharp tensor to size the staging buffer appropriately.
    // For MoE models, merged expert tensors (ffn_*_exps) can be several GiB.
    // The staging buffer only needs to hold ONE tensor upload at a time, and
    // for MoE combination mode we only upload individual expert slices.
    // Cap the staging buffer at 256 MiB to avoid wasting non-swappable
    // pinned (page-locked) memory on a buffer that's mostly unused.
    {
        size_t max_sharp_tensor = 0;
        size_t max_expert_slice = 0;
        for (auto & kv : bsctx->sharp_index) {
            if (kv.second.nbytes > max_sharp_tensor) {
                max_sharp_tensor = kv.second.nbytes;
            }
            // For expert tensors, compute per-expert slice size
            if (bs_is_expert_tensor(kv.first) && kv.second.ne[2] > 1) {
                size_t slice = kv.second.nbytes / (size_t)kv.second.ne[2];
                if (slice > max_expert_slice) {
                    max_expert_slice = slice;
                }
            }
        }

        // In MoE combination mode, we typically upload individual expert
        // slices, not entire merged tensors.  Size the staging buffer to
        // the largest non-expert tensor OR the largest expert slice,
        // whichever is bigger — but cap at 256 MiB to avoid pinning
        // excessive RAM.  For non-MoE or permanent mode, use the full
        // largest tensor (still capped).
        size_t staging_size = max_sharp_tensor;

        // Find the largest non-expert tensor
        size_t max_non_expert = 0;
        for (auto & kv : bsctx->sharp_index) {
            if (!bs_is_expert_tensor(kv.first) && kv.second.nbytes > max_non_expert) {
                max_non_expert = kv.second.nbytes;
            }
        }
        // For MoE models, staging only needs to hold a non-expert tensor
        // or one expert slice (whichever is larger)
        if (max_expert_slice > 0) {
            staging_size = std::max(max_non_expert, max_expert_slice);
        }

        // Hard cap at 256 MiB — pinned memory is page-locked (non-swappable)
        static const size_t STAGING_CAP = 256ULL * 1024 * 1024;
        if (staging_size > STAGING_CAP) {
            LLAMA_LOG_INFO("%s: capping pinned staging buffer from %.2f MiB to %.2f MiB "
                          "(largest tensor = %.2f MiB, largest expert slice = %.2f MiB)\n",
                          __func__,
                          staging_size / (1024.0 * 1024.0),
                          STAGING_CAP / (1024.0 * 1024.0),
                          max_sharp_tensor / (1024.0 * 1024.0),
                          max_expert_slice / (1024.0 * 1024.0));
            staging_size = STAGING_CAP;
        }

        if (staging_size > 0) {
            ggml_backend_buffer_type_t pinned_buft = llama_default_buffer_type_cpu(true);
            if (pinned_buft && pinned_buft != ggml_backend_cpu_buffer_type()) {
                // We have a pinned buffer type (CUDA/Vulkan/SYCL host memory)
                ggml_backend_buffer_t pbuf = ggml_backend_buft_alloc_buffer(pinned_buft, staging_size);
                if (pbuf) {
                    bsctx->pinned_staging_buf  = pbuf;
                    bsctx->pinned_staging_ptr  = ggml_backend_buffer_get_base(pbuf);
                    bsctx->pinned_staging_size = staging_size;
                    LLAMA_LOG_INFO("%s: allocated %.2f MiB pinned staging buffer for fast CPU->GPU copies\n",
                                  __func__, staging_size / (1024.0 * 1024.0));
                } else {
                    LLAMA_LOG_WARN("%s: failed to allocate pinned staging buffer, falling back to pageable memory\n",
                                  __func__);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Reset mmap advice to NORMAL (kernel default).
    //
    // The primary use case is SELECTIVE sharpening — only a handful of
    // layers are sharpened per token, not the entire model.  The access
    // pattern is therefore random (jumping to whichever layers the router
    // picks), not sequential.  MADV_SEQUENTIAL would enable aggressive
    // readahead that pulls in unneeded data and evicts the blurry model.
    //
    // Instead we rely on:
    //   - Per-layer targeted prefetch (bs_prefetch_layer_mmap) in apply_layer
    //   - MADV_DONTNEED after each tensor read to release consumed pages
    //   - MADV_SEQUENTIAL only in apply_all when we know we'll read everything
    //
    // This keeps RAM usage bounded to roughly one layer at a time.
    // -----------------------------------------------------------------------
#ifndef _WIN32
    for (size_t si = 0; si < bsctx->sharp_mmaps.size(); ++si) {
        if (bsctx->sharp_mmaps[si]) {
            posix_madvise(bsctx->sharp_mmaps[si]->addr(),
                          bsctx->sharp_mmaps[si]->size(),
                          POSIX_MADV_NORMAL);
        }
    }
#endif

    } // end if (has_sharp)

    // -----------------------------------------------------------------------
    // Fractal delta correction chain loading
    // -----------------------------------------------------------------------
    if (params.n_delta_levels > 0 && params.delta_paths) {
        LLAMA_LOG_INFO("%s: loading %d delta correction level(s)...\n",
                       __func__, params.n_delta_levels);

        for (int32_t lvl = 0; lvl < params.n_delta_levels; ++lvl) {
            const char * delta_path = params.delta_paths[lvl];
            LLAMA_LOG_INFO("%s:   level %d: %s\n", __func__, lvl + 1, delta_path);

            llama_blurry_sharp_context::bs_correction_level level;
            level.level = lvl + 1;

            // Open the delta GGUF
            gguf_init_params gp = { .no_alloc = true, .ctx = nullptr };
            gguf_context * gctx = gguf_init_from_file(delta_path, gp);
            if (!gctx) {
                LLAMA_LOG_WARN("%s:   failed to open delta GGUF: %s\n", __func__, delta_path);
                continue;
            }

            // Open file + mmap
            auto file_ptr = std::make_unique<llama_file>(delta_path, "rb");
            if (!file_ptr || file_ptr->size() == 0) {
                LLAMA_LOG_WARN("%s:   failed to open delta file: %s\n", __func__, delta_path);
                gguf_free(gctx);
                continue;
            }
            std::unique_ptr<llama_mmap> mmap_ptr;
            try {
                mmap_ptr = std::make_unique<llama_mmap>(file_ptr.get(), 0, false);
            } catch (...) {
                LLAMA_LOG_WARN("%s:   failed to mmap delta: %s\n", __func__, delta_path);
                gguf_free(gctx);
                continue;
            }

            // Build delta tensor index
            ggml_init_params tip = { .mem_size = 256*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
            ggml_context * tctx = ggml_init(tip);
            gguf_init_params tp2 = { .no_alloc = true, .ctx = &tctx };
            gguf_context * g2 = gguf_init_from_file(delta_path, tp2);

            if (g2) {
                size_t data_off = gguf_get_data_offset(g2);
                int n_tensors = gguf_get_n_tensors(g2);

                for (int i = 0; i < n_tensors; i++) {
                    const char * tname = gguf_get_tensor_name(g2, i);
                    blurry_sharp_tensor_info info;
                    info.name        = tname;
                    info.tensor_idx  = i;
                    info.split_idx   = 0;
                    info.file_offset = data_off + gguf_get_tensor_offset(g2, i);
                    info.type        = gguf_get_tensor_type(g2, i);

                    ggml_tensor * t = ggml_get_tensor(tctx, tname);
                    if (t) {
                        for (int d = 0; d < GGML_MAX_DIMS; d++) info.ne[d] = t->ne[d];
                        info.nbytes = ggml_nbytes(t);
                    }

                    // Find the corresponding base model tensor
                    info.base_tensor = llama_get_model_tensor(model, tname);

                    level.delta_index[tname] = info;
                }

                LLAMA_LOG_INFO("%s:   indexed %d delta tensors for level %d\n",
                               __func__, n_tensors, lvl + 1);
                gguf_free(g2);
            }
            ggml_free(tctx);

            level.ggufs.push_back(gctx);
            level.files.push_back(std::move(file_ptr));
            level.mmaps.push_back(std::move(mmap_ptr));
            level.n_splits = 1;

            bsctx->correction_levels.push_back(std::move(level));
        }

        if (!bsctx->correction_levels.empty()) {
            bsctx->fractal_mode = true;
            bsctx->fractal_max_depth = (int32_t)bsctx->correction_levels.size();
            LLAMA_LOG_INFO("%s: fractal mode enabled with %d correction level(s)\n",
                           __func__, bsctx->fractal_max_depth);
        }
    }

    bsctx->initialized = true;

    LLAMA_LOG_INFO("%s: overlay system initialized successfully\n", __func__);
    if (has_sharp) {
        llama_blurry_sharp_log_index_summary(bsctx);
    }

    // -----------------------------------------------------------------------
    // Pre-flight GPU budget analysis
    //
    // For each eligible layer, estimate how much GPU memory the sharp tensors
    // would need (i.e. device tensors that would require a NEW allocation).
    // Warn if the configured budget can't fit even a single layer.
    // -----------------------------------------------------------------------
    if (bsctx->params.gpu_budget_bytes > 0) {
        int64_t total_device_sharp_bytes = 0;
        int     n_layers_fit  = 0;
        int     n_layers_skip = 0;
        int64_t running_bytes = 0;

        for (int layer_idx : bsctx->sharp_layer_indices) {
            if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
                continue;
            }
            auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
            if (lt_it == bsctx->layer_tensor_names.end()) continue;

            int64_t layer_device_bytes = 0;
            for (const auto & tname : lt_it->second) {
                auto si_it = bsctx->sharp_index.find(tname);
                if (si_it == bsctx->sharp_index.end()) continue;

                ggml_tensor * bt = bs_find_model_tensor(bsctx->model, tname.c_str());
                if (!bt || !bt->buffer) continue;

                if (!ggml_backend_buffer_is_host(bt->buffer)) {
                    // This tensor lives on a device — overlay needs a new allocation
                    layer_device_bytes += (int64_t)si_it->second.nbytes;
                }
            }

            total_device_sharp_bytes += layer_device_bytes;

            if (layer_device_bytes == 0 || (running_bytes + layer_device_bytes) <= bsctx->params.gpu_budget_bytes) {
                running_bytes += layer_device_bytes;
                n_layers_fit++;
            } else {
                n_layers_skip++;
            }
        }

        int n_eligible = n_layers_fit + n_layers_skip;
        LLAMA_LOG_INFO("%s: GPU budget analysis: %.0f MiB budget, %.0f MiB needed for all %d eligible layers\n",
                      __func__,
                      bsctx->params.gpu_budget_bytes / (1024.0 * 1024.0),
                      total_device_sharp_bytes / (1024.0 * 1024.0),
                      n_eligible);

        if (n_layers_skip > 0) {
            LLAMA_LOG_INFO("%s: GPU budget can fully sharpen ~%d/%d layers (%.0f MiB used)\n",
                          __func__, n_layers_fit, n_eligible,
                          running_bytes / (1024.0 * 1024.0));
            LLAMA_LOG_WARN("%s: %d layers will be rolled back if sharpened "
                          "(GPU tensors exceed budget). Increase --bs-gpu-budget-mb "
                          "or reduce -ngl to move more tensors to CPU.\n",
                          __func__, n_layers_skip);
        }

        if (n_layers_fit == 0 && n_eligible > 0) {
            LLAMA_LOG_WARN("%s: *** GPU budget (%.0f MiB) is too small to sharpen even ONE layer! ***\n",
                          __func__, bsctx->params.gpu_budget_bytes / (1024.0 * 1024.0));
            LLAMA_LOG_WARN("%s: All sharpen attempts will be rolled back. Either:\n"
                          "%s:   1. Increase --bs-gpu-budget-mb to at least %.0f MiB\n"
                          "%s:   2. Reduce -ngl to keep more weight tensors on CPU\n"
                          "%s:   3. Use a smaller sharp model (fewer bits per weight)\n",
                          __func__, __func__,
                          (total_device_sharp_bytes / (double)n_eligible) / (1024.0 * 1024.0),
                          __func__, __func__);
        }
    }

    // -----------------------------------------------------------------------
    // Initialize RAM expert cache
    // -----------------------------------------------------------------------
    {
        int64_t budget = bsctx->params.ram_cache_bytes;
        if (budget == 0) {
            // Auto: default 4 GiB
            budget = (int64_t)4 * 1024 * 1024 * 1024;
        }
        if (budget > 0) {
            bsctx->ram_expert_cache.enabled      = true;
            bsctx->ram_expert_cache.budget_bytes  = (size_t)budget;
            LLAMA_LOG_INFO("%s: RAM expert cache enabled: %.1f GiB budget\n",
                           __func__, budget / (1024.0 * 1024.0 * 1024.0));
        } else {
            LLAMA_LOG_INFO("%s: RAM expert cache disabled\n", __func__);
        }
    }

    // -----------------------------------------------------------------------
    // Flash-expert mode: release blurry expert tensor pages.
    //
    // The model loader has already read TQ1_0 data into the expert tensor
    // buffers (heap-allocated by --n-cpu-moe).  In flash mode, this data is
    // never used — the eval callback streams Q4_K_M from SSD instead.
    // Release the physical pages back to the OS so they can be used as
    // page cache for the Q4_K_M reads.
    //
    // MADV_DONTNEED on anonymous (heap) pages zeros them — that's fine here
    // since the data is irrelevant.  The virtual address space is preserved.
    // -----------------------------------------------------------------------
    if (params.flash_experts) {
        LLAMA_LOG_INFO("%s: flash-expert mode: releasing blurry expert tensor pages\n", __func__);
        int64_t released_bytes = 0;
        for (auto & [name, info] : bsctx->sharp_index) {
            if (!bs_is_expert_tensor(name)) continue;
            ggml_tensor * t = bs_find_model_tensor(bsctx->model, name.c_str());
            if (!t || !t->data) continue;

            // Only release plain CPU buffers (heap-allocated by --n-cpu-moe).
            // mmap'd buffers are file-backed and would just drop clean pages.
            bool is_plain_cpu = t->buffer &&
                ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cpu_buffer_type();
            if (is_plain_cpu) {
                size_t nbytes = ggml_nbytes(t);
                madvise(t->data, nbytes, MADV_DONTNEED);
                released_bytes += (int64_t)nbytes;
            }
        }
        LLAMA_LOG_INFO("%s: flash-expert mode: released %.1f GiB of blurry expert pages\n",
                       __func__, released_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    return bsctx;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

void llama_blurry_sharp_free(llama_blurry_sharp_context * bsctx) {
    if (!bsctx) return;

    LLAMA_LOG_INFO("%s: freeing blurry-sharp context\n", __func__);

    // Restore all layers first (return model to blurry state)
    llama_blurry_sharp_restore_all(bsctx);

    // Free pinned staging buffer
    if (bsctx->pinned_staging_buf) {
        ggml_backend_buffer_free(bsctx->pinned_staging_buf);
        bsctx->pinned_staging_buf  = nullptr;
        bsctx->pinned_staging_ptr  = nullptr;
        bsctx->pinned_staging_size = 0;
    }

    // Stop async prefetch thread before freeing any data it may reference
    bs_stop_prefetch_thread(bsctx);

    // Free RAM cache (anonymous heap buffers for swap-backed sharp data)
    if (!bsctx->ram_cache.empty()) {
        LLAMA_LOG_INFO("%s: freeing RAM cache: %d tensors, %.2f MiB\n",
                      __func__, (int)bsctx->ram_cache.size(),
                      bsctx->ram_cache_bytes / (1024.0 * 1024.0));
        bsctx->ram_cache.clear();
        bsctx->ram_cache_bytes = 0;
        bsctx->ram_cache_populated = false;
        bsctx->ram_cache_staged = false;
    }

    // Free any cached device buffers.
    // Sync the GPU first to ensure no async kernels are still reading from
    // these buffers (same race condition as LRU eviction in JIT mode).
    if (!bsctx->device_cache.empty()) {
        bs_sync_device_before_eviction(bsctx);
    }
    for (auto & kv : bsctx->device_cache) {
        if (kv.second.buffer) {
            ggml_backend_buffer_free(kv.second.buffer);
        }
    }
    bsctx->device_cache.clear();
    bsctx->device_cache_bytes = 0;

    if (bsctx->params.verbose) {
        LLAMA_LOG_INFO("%s: device cache stats: %" PRId64 " hits, %" PRId64 " allocs, %" PRId64 " recopies, %" PRId64 " evictions\n",
                      __func__, bsctx->n_cache_hits, bsctx->n_cache_allocs, bsctx->n_cache_recopies, bsctx->n_cache_evictions);
    }

    // Free RAM expert cache
    if (bsctx->ram_expert_cache.enabled) {
        auto & rc = bsctx->ram_expert_cache;
        int64_t total = rc.n_hits + rc.n_misses;
        LLAMA_LOG_INFO("%s: RAM expert cache stats: %" PRId64 " hits, %" PRId64 " misses (%.0f%% hit rate), %.1f MiB used, %d entries\n",
                       __func__, rc.n_hits, rc.n_misses,
                       total > 0 ? 100.0 * rc.n_hits / total : 0.0,
                       rc.used_bytes / (1024.0 * 1024.0),
                       (int)rc.entries.size());
        rc.entries.clear();
        rc.used_bytes = 0;
    }
    bsctx->combo_buffers.clear();

    // Free GPU expert cache
    if (bsctx->gpu_cache.cache_buf) {
        if (bsctx->gpu_cache.enabled) {
            LLAMA_LOG_INFO("%s: GPU expert cache stats: %" PRId64 " hits, %" PRId64 " misses (%.0f%% hit rate)\n",
                          __func__, bsctx->gpu_cache.n_hits, bsctx->gpu_cache.n_misses,
                          (bsctx->gpu_cache.n_hits + bsctx->gpu_cache.n_misses) > 0
                              ? 100.0 * bsctx->gpu_cache.n_hits / (bsctx->gpu_cache.n_hits + bsctx->gpu_cache.n_misses) : 0.0);
        }
        ggml_backend_buffer_free(bsctx->gpu_cache.cache_buf);
        bsctx->gpu_cache.cache_buf = nullptr;
        bsctx->gpu_cache.enabled = false;
    }

    // Free sharp GGUF resources (all splits)
    for (auto & m : bsctx->sharp_mmaps)       { m.reset(); }
    for (auto & f : bsctx->sharp_files)        { f.reset(); }
    for (auto * ctx : bsctx->sharp_tensor_ctxs) { if (ctx) ggml_free(ctx); }
    for (auto * g   : bsctx->sharp_ggufs)       { if (g)   gguf_free(g);   }
    bsctx->sharp_mmaps.clear();
    bsctx->sharp_files.clear();
    bsctx->sharp_tensor_ctxs.clear();
    bsctx->sharp_ggufs.clear();

    delete bsctx;
}

// ---------------------------------------------------------------------------
// Internal: apply a single tensor overlay
// ---------------------------------------------------------------------------

// Overlay a single tensor.  Three strategies depending on where the tensor
// lives and whether the sharp file is mmap'd:
//
//   1. ZERO-COPY  (host tensor + mmap sharp file)
//      Point tensor->data directly at the mmap'd sharp page.
//      Cost: ~nanoseconds.  OS demand-pages data from disk.
//
//   2. BUFFERED   (host tensor + no mmap)
//      Read sharp data into a heap buffer, point tensor at it.
//      Cost: ~milliseconds (file I/O).
//
//   3. DEVICE     (non-host tensor, e.g. CUDA)
//      Cannot pointer-swap — tensor->data is a device pointer.
//      Instead: back up original device data, read sharp data on CPU,
//      dequant → F32 → requant to base type, then ggml_backend_tensor_set
//      into the existing device buffer.  Type/strides are unchanged.
//      Cost: depends on tensor size and quant types.
//
// Returns bytes of sharp data referenced/written, or -1 on error.
static int64_t bs_overlay_single_tensor(
        llama_blurry_sharp_context      * bsctx,
        const blurry_sharp_tensor_info  & sharp_info,
        ggml_tensor                     * base_tensor,
        blurry_sharp_tensor_backup      & backup_out) {

    // 1) Save original tensor metadata
    backup_out.tensor_name       = sharp_info.name;
    backup_out.base_tensor       = base_tensor;
    backup_out.original_data     = base_tensor->data;
    backup_out.original_type     = base_tensor->type;
    backup_out.original_view_src = base_tensor->view_src;
    backup_out.original_extra    = base_tensor->extra;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        backup_out.original_nb[d] = base_tensor->nb[d];
        backup_out.ne[d]          = base_tensor->ne[d];
    }

    // 2) Determine strategy: host (CPU) tensor vs device (GPU) tensor
    bool tensor_is_host = true;
    if (base_tensor->buffer) {
        tensor_is_host = ggml_backend_buffer_is_host(base_tensor->buffer);
    }

    // JIT SAFETY CHECK: when overlays are applied mid-graph-execution (JIT
    // mode), the backend scheduler has already allocated device copy tensors
    // at the ORIGINAL (blurry) type/size.  Changing a host tensor's type
    // would make copy_inputs try to write sharp-sized data into the
    // blurry-sized device copy → assertion failure:
    //   "ggml_backend_tensor_set_async: tensor write out of bounds"
    //
    // Same-type overlays are safe (only data pointer changes, sizes match).
    // Device tensor overlays are safe (we swap the buffer directly, no
    // scheduler copy involved).
    //
    // EXCEPTION: tensors in plain CPU buffers (e.g. from --n-cpu-moe) are
    // computed entirely on the CPU backend — no device copies exist for them.
    // Cross-type overlay is safe because the CPU backend reads directly from
    // tensor->data with the updated type/strides.
    if (tensor_is_host && bsctx->jit_active && base_tensor->type != sharp_info.type) {
        bool is_plain_cpu = base_tensor->buffer &&
            ggml_backend_buffer_get_type(base_tensor->buffer) == ggml_backend_cpu_buffer_type();
        if (!is_plain_cpu) {
            bsctx->n_jit_host_crosstype_skipped++;
            if (bsctx->params.verbose || bsctx->n_jit_host_crosstype_skipped <= 3) {
                LLAMA_LOG_WARN("%s: JIT mode: CUDA host/pinned buffer — cross-type overlay skipped for '%s' "
                              "(%s -> %s, %zu -> %zu bytes). This tensor uses BLURRY weights. "
                              "Fix: use --n-cpu-moe to place on plain CPU (supports cross-type), "
                              "or increase -ngl to keep on GPU.\n",
                              __func__, sharp_info.name.c_str(),
                              ggml_type_name(base_tensor->type),
                              ggml_type_name(sharp_info.type),
                              ggml_nbytes(base_tensor), sharp_info.nbytes);
            }
            return -1;
        }
        // Plain CPU buffer — safe to overlay, proceeding with cross-type swap
    }

    if (tensor_is_host) {
        // ================================================================
        // HOST PATH — pointer-swap (zero-copy or buffered)
        // ================================================================
        backup_out.is_device = false;

        int si = sharp_info.split_idx;
        bool have_mmap = (si >= 0 && si < (int)bsctx->sharp_mmaps.size()
                          && bsctx->sharp_mmaps[si]);

        // Check RAM cache first — anonymous heap buffers that swap properly.
        // Under memory pressure, these pages go to swap (fast SSD) instead
        // of being dropped entirely like file-backed mmap pages.
        bool used_ram_cache = false;
        if (bsctx->ram_cache_populated || bsctx->lazy_swap_enabled) {
            auto cache_it = bsctx->ram_cache.find(sharp_info.name);

            // Lazy swap: if the tensor is NOT yet in the cache, populate it
            // on demand by reading from mmap or file into an anonymous heap
            // buffer.  This spreads the disk I/O across the first inference
            // pass instead of a 30-minute upfront precache.
            if (cache_it == bsctx->ram_cache.end() && bsctx->lazy_swap_enabled) {
                // Fast path: check if the async prefetch thread already has it
                bool ok = bs_consume_prefetched_tensor(bsctx, sharp_info.name, sharp_info.nbytes);

                if (!ok) {
                    // Prefetch miss — read synchronously (original path)
                    std::vector<uint8_t> buf(sharp_info.nbytes);

                    // Read from mmap if available
                    if (have_mmap) {
                        const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
                        if (sharp_info.file_offset + sharp_info.nbytes <= bsctx->sharp_mmaps[si]->size()) {
                            std::memcpy(buf.data(), mmap_base + sharp_info.file_offset, sharp_info.nbytes);
                            // Release the mmap pages — data is now in anonymous heap memory.
                            bs_release_mmap_pages(bsctx, mmap_base + sharp_info.file_offset, sharp_info.nbytes);
                            ok = true;
                        }
                    }

                    // Fallback: file read
                    if (!ok && si >= 0 && si < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si]) {
                        bsctx->sharp_files[si]->seek(sharp_info.file_offset, SEEK_SET);
                        bsctx->sharp_files[si]->read_raw(buf.data(), sharp_info.nbytes);
                        ok = true;
                    }

                    if (ok) {
                        bsctx->ram_cache[sharp_info.name] = std::move(buf);
                        bsctx->ram_cache_bytes += (int64_t)sharp_info.nbytes;
                    }
                }

                if (ok) {
                    cache_it = bsctx->ram_cache.find(sharp_info.name);

                    if (bsctx->params.verbose) {
                        LLAMA_LOG_INFO("%s: lazy-swap: cached tensor '%s' (%zu bytes, "
                                      "total cache %.2f MiB)\n",
                                      __func__, sharp_info.name.c_str(), sharp_info.nbytes,
                                      bsctx->ram_cache_bytes / (1024.0 * 1024.0));
                    }
                }
            }

            if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= sharp_info.nbytes) {
                // Point tensor directly at the RAM cache buffer.
                // This is "zero-copy" from the cache's perspective — no
                // new allocation — but the data lives in anonymous (swappable)
                // memory, not file-backed mmap.
                base_tensor->data = cache_it->second.data();
                backup_out.zero_copy = true;  // no owned sharp_data buffer
                used_ram_cache = true;

                // Release old BLURRY mmap pages — the tensor now points at
                // the RAM cache, so the blurry file-backed pages can be
                // dropped from the page cache.
                if (backup_out.original_data) {
                    size_t old_nbytes = bs_tensor_nbytes(backup_out.original_type, backup_out.ne);
                    bs_release_mmap_pages(bsctx, backup_out.original_data, old_nbytes);
                }
            }
        }

        if (!used_ram_cache && have_mmap) {
            // ---- ZERO-COPY: point directly at the mmap'd sharp page ----
            const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
            if (sharp_info.file_offset + sharp_info.nbytes > bsctx->sharp_mmaps[si]->size()) {
                LLAMA_LOG_ERROR("%s: mmap read out of bounds for tensor '%s' (split %d)\n",
                               __func__, sharp_info.name.c_str(), si);
                return -1;
            }
            // Read-only mmap data — we never modify it.  The OS pages it in
            // from disk on first access and evicts under memory pressure.
            base_tensor->data = const_cast<uint8_t *>(mmap_base + sharp_info.file_offset);
            backup_out.zero_copy = true;

            // Release old BLURRY mmap pages — the tensor now points at the
            // sharp mmap, so the blurry file-backed pages can be dropped
            // from the page cache.  They'll be re-faulted from the blurry
            // model file on restore.  This prevents the blurry + sharp
            // models from both being resident in RAM simultaneously.
            if (backup_out.original_data) {
                size_t old_nbytes = bs_tensor_nbytes(backup_out.original_type, backup_out.ne);
                bs_release_mmap_pages(bsctx, backup_out.original_data, old_nbytes);
            }
        } else if (!used_ram_cache) {
            // ---- BUFFERED: read into an owned heap buffer ----
            if (si < 0 || si >= (int)bsctx->sharp_files.size() || !bsctx->sharp_files[si]) {
                LLAMA_LOG_ERROR("%s: no file handle for split %d (tensor '%s')\n",
                               __func__, si, sharp_info.name.c_str());
                return -1;
            }
            backup_out.sharp_data.resize(sharp_info.nbytes);
            bsctx->sharp_files[si]->seek(sharp_info.file_offset, SEEK_SET);
            bsctx->sharp_files[si]->read_raw(backup_out.sharp_data.data(), sharp_info.nbytes);
            base_tensor->data = backup_out.sharp_data.data();
            backup_out.zero_copy = false;
        }

        // Update tensor type and recompute strides for the sharp quant format
        base_tensor->type = sharp_info.type;
        base_tensor->nb[0] = ggml_type_size(sharp_info.type);
        base_tensor->nb[1] = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
        }

        // Clear view_src so that ggml_backend_tensor_set/get dispatch through
        // tensor->buffer (which may differ from view_src->buffer after swap).
        // The original value is saved in backup_out.original_view_src for restore.
        base_tensor->view_src = nullptr;

        bsctx->metrics.n_direct_copies++;

    } else {
        // ================================================================
        // DEVICE PATH — pointer-swap with a new device buffer
        //
        // The tensor lives on a device (GPU).  We cannot point it at CPU
        // mmap memory.  Instead we:
        //   a) Allocate a NEW device buffer sized for the sharp tensor
        //   b) Read sharp data from file/mmap into a CPU staging buffer
        //   c) Temporarily wire the tensor to the new buffer, then use
        //      ggml_backend_tensor_set to copy CPU → device
        //   d) The original device buffer is NOT modified — it stays
        //      intact in the model's shared allocation.  No backup needed.
        //   e) On restore: swap {data, type, nb[], buffer} back, free
        //      the new device buffer.
        //
        // Cost: one CPU→GPU memcpy per tensor (PCIe bandwidth bound).
        // No dequant, no requant, no backup copy.
        // ================================================================
        backup_out.is_device  = true;
        backup_out.zero_copy  = false;

        // Obtain a usable (non-split) buffer type for device allocation.
        // Must be computed BEFORE we modify base_tensor->buffer.
        ggml_backend_buffer_type_t device_buft = bs_get_device_buft(bsctx, base_tensor->buffer);

        // ---- Check device buffer cache first (retain_device_buffers mode) ----
        // If we previously sharpened this tensor and retained the GPU buffer,
        // we can skip cudaMalloc AND the PCIe copy entirely — just pointer-swap.
        if (bsctx->retain_device_buffers) {
            auto cache_it = bsctx->device_cache.find(sharp_info.name);
            if (cache_it != bsctx->device_cache.end() &&
                cache_it->second.buffer &&
                cache_it->second.nbytes >= sharp_info.nbytes) {

                // Cache hit!  Buffer is already on GPU with sharp data.
                cache_it->second.use_sequence = bsctx->cache_use_counter++;

                backup_out.original_buffer     = base_tensor->buffer;
                backup_out.device_sharp_buffer = cache_it->second.buffer;
                backup_out.device_buf_cached   = true;  // do NOT free on restore

                void * cached_base = ggml_backend_buffer_get_base(cache_it->second.buffer);
                base_tensor->data   = cached_base;
                base_tensor->buffer = cache_it->second.buffer;
                base_tensor->type   = sharp_info.type;
                base_tensor->nb[0]  = ggml_type_size(sharp_info.type);
                base_tensor->nb[1]  = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
                for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                    base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
                }

                // Clear view_src so dispatch goes through tensor->buffer
                // (the cached sharp buffer), not the stale original.
                base_tensor->view_src = nullptr;

                // Clear extra so CUDA compute uses tensor->data (our new
                // regular buffer) instead of the split tensor metadata.
                base_tensor->extra = nullptr;

                if (!cache_it->second.populated) {
                    // Buffer exists but data hasn't been copied yet — re-copy.
                    // Zero the buffer first to ensure padding is clean.
                    ggml_backend_buffer_set_usage(cache_it->second.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_buffer_clear(cache_it->second.buffer, 0);

                    // Use optimized staging path (direct mmap→pinned when possible)
                    const void * staging_src = bs_read_to_staging(bsctx, sharp_info);
                    if (staging_src) {
                        ggml_backend_tensor_set(base_tensor, staging_src, 0, sharp_info.nbytes);
                        cache_it->second.populated = true;
                        bsctx->n_cache_recopies++;
                    }
                } else {
                    bsctx->n_cache_hits++;
                }

                bsctx->metrics.n_direct_copies++;
                return (int64_t)sharp_info.nbytes;
            }
        }

        // ---- No cache hit — check GPU budget and evict if needed ----
        int64_t effective_device_bytes = bsctx->retain_device_buffers
            ? bsctx->device_cache_bytes
            : bsctx->current_device_sharp_bytes;

        if (bsctx->params.gpu_budget_bytes > 0 &&
            (effective_device_bytes + (int64_t)sharp_info.nbytes) > bsctx->params.gpu_budget_bytes) {

            if (bsctx->retain_device_buffers && !bsctx->device_cache.empty()) {
                // ---- LRU eviction: free the least-recently-used cached buffers
                //      until we have room for this new tensor ----
                //
                // IMPORTANT: In JIT mode, CUDA kernels from a restored layer
                // may still be executing asynchronously.  We MUST synchronize
                // the device before freeing any buffers to prevent
                // "CUDA error: illegal memory access" from kernels reading
                // freed memory.
                int64_t needed = (effective_device_bytes + (int64_t)sharp_info.nbytes)
                               - bsctx->params.gpu_budget_bytes;
                int64_t freed = 0;
                bool synced = false;  // only sync once per eviction batch

                while (freed < needed && !bsctx->device_cache.empty()) {
                    // Find the LRU entry (lowest use_sequence)
                    auto lru_it = bsctx->device_cache.end();
                    uint64_t min_seq = UINT64_MAX;
                    for (auto it = bsctx->device_cache.begin(); it != bsctx->device_cache.end(); ++it) {
                        if (it->second.buffer && it->second.use_sequence < min_seq) {
                            // Don't evict buffers that are currently wired to a live tensor.
                            // Two cases:
                            //   1. The layer is fully sharpened (is_sharpened == true) — its
                            //      tensors' data pointers reference cached buffers.
                            //   2. The layer is currently BEING BUILT (building_layer_idx) —
                            //      earlier tensors in this layer already had their data
                            //      pointers swapped to cached buffers, but is_sharpened
                            //      hasn't been set yet.  Evicting these would leave
                            //      base_tensor->data pointing at freed GPU memory.
                            bool in_use = false;
                            if (it->second.layer_idx >= 0) {
                                // Guard against self-eviction during layer construction
                                if (it->second.layer_idx == bsctx->building_layer_idx) {
                                    in_use = true;
                                }
                                auto lb_it = bsctx->layer_backups.find(it->second.layer_idx);
                                if (lb_it != bsctx->layer_backups.end() && lb_it->second.is_sharpened) {
                                    in_use = true;
                                }
                            }
                            if (!in_use) {
                                min_seq = it->second.use_sequence;
                                lru_it = it;
                            }
                        }
                    }

                    if (lru_it == bsctx->device_cache.end()) {
                        break;  // all remaining cached buffers are in use
                    }

                    // Sync device ONCE before the first free to ensure all
                    // async CUDA kernels (from a previous JIT layer) have
                    // completed and are no longer reading from these buffers.
                    if (!synced) {
                        bs_sync_device_before_eviction(bsctx);
                        synced = true;
                    }

                    if (bsctx->params.verbose) {
                        LLAMA_LOG_INFO("%s: cache evict '%s' (%.2f MiB, seq=%" PRIu64 ")\n",
                                      __func__, lru_it->first.c_str(),
                                      lru_it->second.nbytes / (1024.0 * 1024.0),
                                      lru_it->second.use_sequence);
                    }

                    freed += (int64_t)lru_it->second.nbytes;
                    bsctx->device_cache_bytes -= (int64_t)lru_it->second.nbytes;
                    ggml_backend_buffer_free(lru_it->second.buffer);
                    bsctx->device_cache.erase(lru_it);
                    bsctx->n_cache_evictions++;
                }

                // Recompute effective bytes after eviction
                effective_device_bytes = bsctx->device_cache_bytes;
            }

            // After eviction attempt, re-check budget
            if (bsctx->params.gpu_budget_bytes > 0 &&
                (effective_device_bytes + (int64_t)sharp_info.nbytes) > bsctx->params.gpu_budget_bytes) {
                if (bsctx->params.verbose) {
                    LLAMA_LOG_WARN("%s: GPU budget exceeded (used %" PRId64 " + %zu > budget %" PRId64 "), "
                                  "cannot overlay device tensor '%s'\n",
                                  __func__,
                                  effective_device_bytes,
                                  sharp_info.nbytes,
                                  bsctx->params.gpu_budget_bytes,
                                  sharp_info.name.c_str());
                }
                bsctx->n_device_tensors_skipped++;
                return -1;
            }
        }

        // a) Allocate a device buffer on the same backend as the tensor.
        //
        //    IMPORTANT: CUDA quantized matmul kernels read up to
        //    MATRIX_ROW_PADDING (512) elements past the logical end of each
        //    row.  The normal model-loading path sizes every tensor's
        //    allocation via ggml_backend_buft_get_alloc_size() which adds
        //    that padding, and then ggml_backend_buffer_init_tensor() zeros
        //    it.  We must do the same here, otherwise the kernel reads
        //    uninitialized GPU memory — which may contain NaN bit patterns
        //    — and the NaN cascades through every subsequent layer, making
        //    all logits NaN.
        //
        ggml_backend_buffer_type_t buft = device_buft;

        // Temporarily set the tensor type AND nb[] to the sharp type so
        // that ggml_backend_buft_get_alloc_size computes the correct
        // padded size.  ggml_nbytes() for quantized types uses
        // ne[1]*nb[1], so nb[] MUST match the type — otherwise we get
        // a size based on the blurry row stride, which is wrong.
        ggml_type saved_type = base_tensor->type;
        size_t saved_nb[GGML_MAX_DIMS];
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            saved_nb[d] = base_tensor->nb[d];
        }

        base_tensor->type  = sharp_info.type;
        base_tensor->nb[0] = ggml_type_size(sharp_info.type);
        base_tensor->nb[1] = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
        }
        size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, base_tensor);

        // Restore original type and nb[] until we're ready to commit
        base_tensor->type = saved_type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = saved_nb[d];
        }

        // Ensure alloc_size is at least as large as the raw tensor data
        if (alloc_size < sharp_info.nbytes) {
            alloc_size = sharp_info.nbytes;
        }

        ggml_backend_buffer_t sharp_buf = ggml_backend_buft_alloc_buffer(buft, alloc_size);
        bool using_pinned_fallback = false;

        if (!sharp_buf) {
            // VRAM exhausted — try pinned host memory (GPU-accessible via UVA)
            ggml_backend_buffer_type_t pinned_buft = llama_default_buffer_type_cpu(true);
            if (pinned_buft && pinned_buft != ggml_backend_cpu_buffer_type()) {
                sharp_buf = ggml_backend_buft_alloc_buffer(pinned_buft, sharp_info.nbytes);
            }
            if (!sharp_buf) {
                sharp_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), sharp_info.nbytes);
            }
            if (!sharp_buf) {
                LLAMA_LOG_ERROR("%s: failed to allocate device or pinned buffer (%zu bytes) for tensor '%s'\n",
                               __func__, alloc_size, sharp_info.name.c_str());
                bsctx->n_device_tensors_skipped++;
                return -1;
            }
            using_pinned_fallback = true;
            LLAMA_LOG_WARN("%s: VRAM exhausted for tensor '%s' (%zu bytes), "
                          "using pinned host memory (sharp quality, PCIe speed)\n",
                          __func__, sharp_info.name.c_str(), alloc_size);
        }

        // Mark the buffer as holding weight data so the backend scheduler
        // treats it the same as the original model weight buffers.  Without
        // this flag the scheduler's "operations with weights" heuristic is
        // skipped and backend assignment falls through to weaker heuristics,
        // which can mis-assign operations or fail to create graph splits.
        ggml_backend_buffer_set_usage(sharp_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Zero the entire buffer so that padding bytes beyond the
        // tensor data are 0, not uninitialised garbage / NaN.
        if (!using_pinned_fallback) {
            ggml_backend_buffer_clear(sharp_buf, 0);
        } else {
            void * buf_base = ggml_backend_buffer_get_base(sharp_buf);
            memset(buf_base, 0, sharp_info.nbytes);
        }

        // b) Read sharp data into staging buffer for GPU upload.
        //    Uses optimized path: mmap → pinned directly when possible,
        //    skipping the intermediate heap buffer copy.
        const void * staging_src = bs_read_to_staging(bsctx, sharp_info);
        if (!staging_src) {
            ggml_backend_buffer_free(sharp_buf);
            return -1;
        }

        // --- Source data integrity check (verbose mode only) ---
        if (bsctx->params.verbose && sharp_info.nbytes >= 16) {
            const uint8_t * raw = (const uint8_t *)staging_src;
            bool all_zero = true;
            bool all_ff   = true;
            for (size_t b = 0; b < 16; ++b) {
                if (raw[b] != 0x00) all_zero = false;
                if (raw[b] != 0xFF) all_ff   = false;
            }
            for (size_t b = sharp_info.nbytes - 16; b < sharp_info.nbytes; ++b) {
                if (raw[b] != 0x00) all_zero = false;
                if (raw[b] != 0xFF) all_ff   = false;
            }
            if (all_zero) {
                LLAMA_LOG_WARN("%s: *** tensor '%s' sharp data is ALL ZEROS — "
                              "possible file read failure (offset=%zu, nbytes=%zu) ***\n",
                              __func__, sharp_info.name.c_str(),
                              sharp_info.file_offset, sharp_info.nbytes);
            }
            if (all_ff) {
                LLAMA_LOG_WARN("%s: *** tensor '%s' sharp data is ALL 0xFF — "
                              "possible corrupt/unmapped region (offset=%zu, nbytes=%zu) ***\n",
                              __func__, sharp_info.name.c_str(),
                              sharp_info.file_offset, sharp_info.nbytes);
            }
        }

        // c) Save original tensor state, then wire tensor to the new buffer
        //    so that ggml_backend_tensor_set dispatches to the right device.
        backup_out.original_buffer      = base_tensor->buffer;
        backup_out.device_sharp_buffer  = sharp_buf;

        void * sharp_base = ggml_backend_buffer_get_base(sharp_buf);
        base_tensor->data   = sharp_base;
        base_tensor->buffer = sharp_buf;
        base_tensor->type   = sharp_info.type;
        base_tensor->nb[0]  = ggml_type_size(sharp_info.type);
        base_tensor->nb[1]  = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
        }

        // Clear view_src BEFORE tensor_set so dispatch goes through the
        // correct (new) buffer, not the stale original via view_src.
        base_tensor->view_src = nullptr;

        // Clear extra so CUDA compute uses tensor->data (our new regular
        // buffer) instead of following split-tensor metadata that still
        // points at the original device allocation.
        base_tensor->extra = nullptr;

        // Copy sharp data into the buffer
        if (using_pinned_fallback) {
            // Pinned/CPU buffer — direct memcpy (GPU reads via UVA/PCIe)
            void * buf_base = ggml_backend_buffer_get_base(sharp_buf);
            memcpy(buf_base, staging_src, sharp_info.nbytes);
        } else {
            // Device buffer — upload via backend's set_tensor (H2D copy)
            ggml_backend_tensor_set(base_tensor, staging_src, 0, sharp_info.nbytes);
        }

        // --- Post-copy device readback verification (verbose only, device path) ---
        if (bsctx->params.verbose && !using_pinned_fallback) {
            const size_t verify_bytes = std::min<size_t>(64, sharp_info.nbytes);
            std::vector<uint8_t> readback(verify_bytes, 0xAA);
            ggml_backend_tensor_get(base_tensor, readback.data(), 0, verify_bytes);

            bool mismatch = false;
            const uint8_t * src_bytes = (const uint8_t *)staging_src;
            for (size_t b = 0; b < verify_bytes; ++b) {
                if (readback[b] != src_bytes[b]) {
                    mismatch = true;
                    break;
                }
            }

            if (mismatch) {
                LLAMA_LOG_ERROR("%s: *** READBACK MISMATCH for tensor '%s' — "
                               "device data does not match staging buffer! ***\n",
                               __func__, sharp_info.name.c_str());
                LLAMA_LOG_ERROR("%s:   first bytes staging: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                               __func__,
                               src_bytes[0], src_bytes[1], src_bytes[2], src_bytes[3],
                               src_bytes[4], src_bytes[5], src_bytes[6], src_bytes[7]);
                LLAMA_LOG_ERROR("%s:   first bytes device:  %02x %02x %02x %02x %02x %02x %02x %02x\n",
                               __func__,
                               readback[0], readback[1], readback[2], readback[3],
                               readback[4], readback[5], readback[6], readback[7]);
            }

            LLAMA_LOG_INFO("%s: device tensor '%s': alloc=%zu data=%zu type=%s ne=[%" PRId64 ",%" PRId64 "] "
                          "buf=%p data=%p verify=%s\n",
                          __func__, sharp_info.name.c_str(),
                          alloc_size, sharp_info.nbytes,
                          ggml_type_name(sharp_info.type),
                          base_tensor->ne[0], base_tensor->ne[1],
                          (void *)sharp_buf, base_tensor->data,
                          mismatch ? "FAIL" : "ok");
        }

        // Track GPU memory usage (only for actual device buffers)
        if (!using_pinned_fallback) {
            bsctx->current_device_sharp_bytes += (int64_t)alloc_size;
        }

        // If retaining, store this buffer in the cache for future reuse
        if (bsctx->retain_device_buffers) {
            blurry_sharp_device_cache_entry entry;
            entry.buffer       = sharp_buf;
            entry.nbytes       = alloc_size;
            entry.populated    = true;
            entry.use_sequence = bsctx->cache_use_counter++;
            entry.layer_idx    = sharp_info.layer_idx;
            bsctx->device_cache[sharp_info.name] = entry;
            bsctx->device_cache_bytes += (int64_t)alloc_size;
            bsctx->n_cache_allocs++;
            backup_out.device_buf_cached = true;  // do NOT free on restore
        }

        bsctx->metrics.n_direct_copies++;
    }

    return (int64_t)sharp_info.nbytes;
}

// ---------------------------------------------------------------------------
// Internal: permanent (one-way) overlay of a single tensor
//
// Unlike bs_overlay_single_tensor, this does NOT create a backup.
// The blurry data is overwritten / abandoned and cannot be restored.
//
// Memory strategy:
//   - GPU tensor, SAME type+size: upload sharp data directly into the
//     existing device buffer via ggml_backend_tensor_set.  Zero extra VRAM.
//   - GPU tensor, DIFFERENT type: allocate a new device buffer, upload
//     sharp data, pointer-swap.  The old data stays in the model's shared
//     allocation (can't free it piecewise) but we don't allocate a backup.
//   - CPU tensor, mmap: pointer-swap to sharp mmap page.  Release blurry
//     mmap pages via MADV_DONTNEED.
//   - CPU tensor, no mmap: read sharp data into a heap buffer, swap pointer.
//
// Returns bytes of sharp data written, or -1 on error.
// ---------------------------------------------------------------------------

static int64_t bs_overlay_single_tensor_permanent(
        llama_blurry_sharp_context      * bsctx,
        const blurry_sharp_tensor_info  & sharp_info,
        ggml_tensor                     * base_tensor) {

    bool tensor_is_host = true;
    if (base_tensor->buffer) {
        tensor_is_host = ggml_backend_buffer_is_host(base_tensor->buffer);
    }

    if (tensor_is_host) {
        // ================================================================
        // HOST PATH — pointer-swap (permanent, no backup)
        // ================================================================
        int si = sharp_info.split_idx;
        bool have_mmap = (si >= 0 && si < (int)bsctx->sharp_mmaps.size()
                          && bsctx->sharp_mmaps[si]);

        if (have_mmap) {
            // ZERO-COPY: point directly at the mmap'd sharp page.
            const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
            if (sharp_info.file_offset + sharp_info.nbytes > bsctx->sharp_mmaps[si]->size()) {
                LLAMA_LOG_ERROR("%s: mmap read out of bounds for tensor '%s' (split %d)\n",
                               __func__, sharp_info.name.c_str(), si);
                return -1;
            }

            // Release old blurry mmap pages — we're done with them permanently
            if (base_tensor->data) {
                int64_t ne_arr[GGML_MAX_DIMS];
                for (int d = 0; d < GGML_MAX_DIMS; ++d) ne_arr[d] = base_tensor->ne[d];
                size_t old_nbytes = bs_tensor_nbytes(base_tensor->type, ne_arr);
                bs_release_mmap_pages(bsctx, base_tensor->data, old_nbytes);
            }

            base_tensor->data = const_cast<uint8_t *>(mmap_base + sharp_info.file_offset);
        } else {
            // BUFFERED: read into heap memory.  In permanent mode we leak the
            // old data pointer (it's part of the model's mmap or allocation,
            // can't free it piecewise).  The new buffer is allocated and never
            // freed until the model itself is freed.
            if (si < 0 || si >= (int)bsctx->sharp_files.size() || !bsctx->sharp_files[si]) {
                LLAMA_LOG_ERROR("%s: no file handle for split %d (tensor '%s')\n",
                               __func__, si, sharp_info.name.c_str());
                return -1;
            }
            // Allocate persistent buffer (intentionally leaked — lives as long as model)
            uint8_t * buf = new uint8_t[sharp_info.nbytes];
            bsctx->sharp_files[si]->seek(sharp_info.file_offset, SEEK_SET);
            bsctx->sharp_files[si]->read_raw(buf, sharp_info.nbytes);
            base_tensor->data = buf;
        }

        // Update tensor type and recompute strides for the sharp quant format
        base_tensor->type = sharp_info.type;
        base_tensor->nb[0] = ggml_type_size(sharp_info.type);
        base_tensor->nb[1] = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
        }

        // Clear view_src so dispatch goes through tensor->buffer
        base_tensor->view_src = nullptr;

        bsctx->metrics.n_direct_copies++;

    } else {
        // ================================================================
        // DEVICE PATH — permanent overwrite
        //
        // Two sub-strategies:
        //   A) SAME TYPE + SAME SIZE: upload sharp data directly into the
        //      existing device buffer.  No new allocation.  Zero extra VRAM.
        //   B) DIFFERENT TYPE/SIZE: allocate a new device buffer, upload
        //      sharp data, pointer-swap.  Old data stays in the model's
        //      shared allocation (unavoidable) but no backup is created.
        // ================================================================

        bool same_type = (base_tensor->type == sharp_info.type);
        size_t base_nbytes = ggml_nbytes(base_tensor);
        bool same_size = (base_nbytes == sharp_info.nbytes);

        if (same_type && same_size) {
            // ---- Strategy A: IN-PLACE OVERWRITE (zero extra VRAM) ----
            // Read sharp data into staging, then ggml_backend_tensor_set
            // directly into the existing device buffer.
            const void * staging_src = bs_read_to_staging(bsctx, sharp_info);
            if (!staging_src) {
                LLAMA_LOG_ERROR("%s: failed to read sharp data for in-place overwrite of '%s'\n",
                               __func__, sharp_info.name.c_str());
                return -1;
            }

            // For split buffers, we need to handle the tensor_set differently.
            // The standard ggml_backend_tensor_set dispatches through
            // tensor->buffer which handles split buffers correctly via
            // tensor->extra.  So this works for both regular and split buffers.
            ggml_backend_tensor_set(base_tensor, staging_src, 0, sharp_info.nbytes);

            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: permanent in-place overwrite '%s' (%s, %zu bytes, zero extra VRAM)\n",
                              __func__, sharp_info.name.c_str(),
                              ggml_type_name(sharp_info.type), sharp_info.nbytes);
            }

        } else {
            // ---- Strategy B: NEW BUFFER (different type/size) ----
            ggml_backend_buffer_type_t device_buft = bs_get_device_buft(bsctx, base_tensor->buffer);

            // Temporarily set tensor type/strides for correct alloc_size computation
            ggml_type saved_type = base_tensor->type;
            size_t saved_nb[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; ++d) saved_nb[d] = base_tensor->nb[d];

            base_tensor->type  = sharp_info.type;
            base_tensor->nb[0] = ggml_type_size(sharp_info.type);
            base_tensor->nb[1] = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
            }
            size_t alloc_size = ggml_backend_buft_get_alloc_size(device_buft, base_tensor);

            // Restore temporarily (we'll set them again after successful alloc)
            base_tensor->type = saved_type;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) base_tensor->nb[d] = saved_nb[d];

            if (alloc_size < sharp_info.nbytes) alloc_size = sharp_info.nbytes;

            ggml_backend_buffer_t sharp_buf = ggml_backend_buft_alloc_buffer(device_buft, alloc_size);
            bool perm_using_pinned = false;

            if (!sharp_buf) {
                // VRAM exhausted — try pinned host memory
                ggml_backend_buffer_type_t pinned_buft = llama_default_buffer_type_cpu(true);
                if (pinned_buft && pinned_buft != ggml_backend_cpu_buffer_type()) {
                    sharp_buf = ggml_backend_buft_alloc_buffer(pinned_buft, sharp_info.nbytes);
                }
                if (!sharp_buf) {
                    sharp_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), sharp_info.nbytes);
                }
                if (!sharp_buf) {
                    LLAMA_LOG_ERROR("%s: failed to allocate device or pinned buffer (%zu bytes) "
                                   "for permanent overlay of '%s'\n",
                                   __func__, alloc_size, sharp_info.name.c_str());
                    return -1;
                }
                perm_using_pinned = true;
                LLAMA_LOG_WARN("%s: VRAM exhausted for permanent overlay '%s' (%zu bytes), "
                              "using pinned host memory (sharp quality, PCIe speed)\n",
                              __func__, sharp_info.name.c_str(), alloc_size);
            }

            ggml_backend_buffer_set_usage(sharp_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            if (!perm_using_pinned) {
                ggml_backend_buffer_clear(sharp_buf, 0);
            } else {
                void * buf_base = ggml_backend_buffer_get_base(sharp_buf);
                memset(buf_base, 0, sharp_info.nbytes);
            }

            // Read sharp data into staging
            const void * staging_src = bs_read_to_staging(bsctx, sharp_info);
            if (!staging_src) {
                ggml_backend_buffer_free(sharp_buf);
                return -1;
            }

            // Wire tensor to new buffer
            void * sharp_base = ggml_backend_buffer_get_base(sharp_buf);
            base_tensor->data     = sharp_base;
            base_tensor->buffer   = sharp_buf;
            base_tensor->type     = sharp_info.type;
            base_tensor->nb[0]    = ggml_type_size(sharp_info.type);
            base_tensor->nb[1]    = ggml_row_size(sharp_info.type, base_tensor->ne[0]);
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                base_tensor->nb[d] = base_tensor->nb[d - 1] * base_tensor->ne[d - 1];
            }
            base_tensor->view_src = nullptr;
            base_tensor->extra    = nullptr;

            // Copy sharp data into buffer
            if (perm_using_pinned) {
                memcpy(sharp_base, staging_src, sharp_info.nbytes);
            } else {
                ggml_backend_tensor_set(base_tensor, staging_src, 0, sharp_info.nbytes);
            }

            // Track the new buffer in device_cache so it gets freed on context destruction.
            blurry_sharp_device_cache_entry entry;
            entry.buffer       = sharp_buf;
            entry.nbytes       = perm_using_pinned ? sharp_info.nbytes : alloc_size;
            entry.populated    = true;
            entry.use_sequence = bsctx->cache_use_counter++;
            entry.layer_idx    = sharp_info.layer_idx;
            bsctx->device_cache[sharp_info.name] = entry;
            if (!perm_using_pinned) {
                bsctx->device_cache_bytes += (int64_t)alloc_size;
            }

            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: permanent new-buffer overlay '%s' (%s → %s, %zu bytes)\n",
                              __func__, sharp_info.name.c_str(),
                              ggml_type_name(saved_type),
                              ggml_type_name(sharp_info.type), alloc_size);
            }
        }

        bsctx->metrics.n_direct_copies++;
    }

    return (int64_t)sharp_info.nbytes;
}

// ---------------------------------------------------------------------------
// Internal: restore a single tensor from backup
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lazy swap: push a layer's RAM-cached tensor pages to swap via MADV_PAGEOUT.
//
// Called after restore_layer / restore_all when lazy_swap_enabled is true.
// The tensor data is now "cold" (tensor pointers have been restored to blurry),
// so we can push the anonymous heap pages to swap, freeing RAM for the blurry
// model / KV cache.  Next time apply_layer is called, the OS will swap-in
// from SSD (fast) instead of re-reading from the GGUF file (slow).
// ---------------------------------------------------------------------------
static void bs_lazy_swap_pageout_layer(
        llama_blurry_sharp_context * bsctx,
        int                          layer_idx) {
#ifdef __linux__
    #ifndef MADV_PAGEOUT
    #define MADV_PAGEOUT 21
    #endif

    auto layer_it = bsctx->layer_tensor_names.find(layer_idx);
    if (layer_it == bsctx->layer_tensor_names.end()) return;

    int32_t n_paged = 0;
    int64_t paged_bytes = 0;

    for (const auto & tensor_name : layer_it->second) {
        auto cache_it = bsctx->ram_cache.find(tensor_name);
        if (cache_it == bsctx->ram_cache.end() || cache_it->second.empty()) continue;

        int ret = madvise(cache_it->second.data(), cache_it->second.size(), MADV_PAGEOUT);
        if (ret == 0) {
            ++n_paged;
            paged_bytes += (int64_t)cache_it->second.size();
        }
    }

    if (bsctx->params.verbose && n_paged > 0) {
        LLAMA_LOG_INFO("%s: lazy-swap: paged out %d tensors (%.2f MiB) for layer %d\n",
                      __func__, n_paged, paged_bytes / (1024.0 * 1024.0), layer_idx);
    }
#else
    (void)bsctx;
    (void)layer_idx;
#endif
}

static bool bs_restore_single_tensor(
        llama_blurry_sharp_context     * bsctx,
        blurry_sharp_tensor_backup     & backup) {
    // Use cached pointer (O(1)) instead of linear search (O(n))
    ggml_tensor * base_tensor = backup.base_tensor;
    if (!base_tensor) {
        // Fallback to linear search if cache wasn't populated
        base_tensor = bs_find_model_tensor(bsctx->model, backup.tensor_name.c_str());
    }
    if (!base_tensor) {
        LLAMA_LOG_ERROR("%s: cannot find tensor '%s' for restoration\n",
                       __func__, backup.tensor_name.c_str());
        return false;
    }

    // ---- Flash-expert restore: pointer-only, no data copy ----
    // In flash mode, the original tensor data is irrelevant (never loaded).
    // Just restore the pointer, type, and strides so the graph builder sees
    // correct metadata for the next token.
    if (backup.flash_expert) {
        base_tensor->data     = backup.original_data;
        base_tensor->type     = backup.original_type;
        base_tensor->view_src = backup.original_view_src;
        base_tensor->extra    = backup.original_extra;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = backup.original_nb[d];
        }
        return true;
    }

    // ---- In-place expert slice restore (host or device) ----
    // If expert_backup_ids is non-empty, this tensor was sharpened via the
    // in-place expert slice strategy (Strategy A/B in overlay_expert_tensor).
    // Restore by writing back the saved blurry expert slices.
    if (!backup.expert_backup_ids.empty()) {
        size_t n_backed = backup.expert_backup_ids.size();
        // Compute per-expert slice size from the backup dimensions
        size_t total_bytes = bs_tensor_nbytes(backup.original_type, backup.ne);
        int32_t n_expert_total = (int32_t)backup.ne[2];
        size_t expert_slice = (n_expert_total > 0) ? total_bytes / (size_t)n_expert_total : total_bytes;
        size_t data_offset = 0;

        if (backup.is_device) {
            // Write back blurry slices to device
            for (size_t i = 0; i < n_backed; ++i) {
                size_t slice_off = (size_t)backup.expert_backup_ids[i] * expert_slice;
                ggml_backend_tensor_set(base_tensor,
                    backup.expert_backup_data.data() + data_offset,
                    slice_off, expert_slice);
                data_offset += expert_slice;
            }
        } else {
            // Write back blurry slices to host memory
            uint8_t * base_data = (uint8_t *)base_tensor->data;
            for (size_t i = 0; i < n_backed; ++i) {
                size_t slice_off = (size_t)backup.expert_backup_ids[i] * expert_slice;
                std::memcpy(base_data + slice_off,
                            backup.expert_backup_data.data() + data_offset,
                            expert_slice);
                data_offset += expert_slice;
            }
        }

        backup.expert_backup_ids.clear();
        backup.expert_backup_data.clear();
        backup.expert_backup_data.shrink_to_fit();
        return true;
    }

    if (backup.is_device) {
        // DEVICE PATH: pointer-swap restore — put back original {data, type, nb[], buffer, view_src, extra}
        //
        // Release sharp GPU tensor's backing mmap pages (if the data was also
        // left resident in the page cache by a cached-but-repopulated path).
        // This is a no-op if the pages were already released during staging.

        base_tensor->data     = backup.original_data;
        base_tensor->type     = backup.original_type;
        base_tensor->buffer   = backup.original_buffer;
        base_tensor->view_src = backup.original_view_src;
        base_tensor->extra    = backup.original_extra;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = backup.original_nb[d];
        }
        if (backup.device_sharp_buffer) {
            if (backup.device_buf_cached) {
                // Buffer is in the cache — do NOT free it.  It will be reused
                // next time this tensor is sharpened (zero-cost re-sharpen).
                // Just detach it from the backup so it isn't double-freed.
                backup.device_sharp_buffer = nullptr;
            } else {
                // Not cached — free the device buffer and track GPU memory freed
                bsctx->current_device_sharp_bytes -= (int64_t)ggml_backend_buffer_get_size(backup.device_sharp_buffer);
                ggml_backend_buffer_free(backup.device_sharp_buffer);
                backup.device_sharp_buffer = nullptr;
            }
        }
    } else {
        // HOST PATH: pointer-swap restore — put back the saved metadata including view_src and extra.
        //
        // For zero-copy tensors pointing at mmap pages, release them so the
        // sharp model doesn't linger in the page cache.
        //
        // IMPORTANT: Do NOT call bs_release_mmap_pages on RAM cache buffers!
        // On Linux, MADV_DONTNEED on anonymous (heap) pages ZEROS them out,
        // destroying the cached data.  Only release file-backed mmap pages.
        if (backup.zero_copy && base_tensor->data) {
            bool is_heap_ptr = false;
            // Check if data points into any RAM cache buffer (old full-tensor cache)
            if (bsctx->ram_cache_populated || bsctx->lazy_swap_enabled) {
                for (auto & kv : bsctx->ram_cache) {
                    const uint8_t * buf_start = kv.second.data();
                    const uint8_t * buf_end   = buf_start + kv.second.size();
                    const uint8_t * data_ptr  = (const uint8_t *)base_tensor->data;
                    if (data_ptr >= buf_start && data_ptr < buf_end) {
                        is_heap_ptr = true;
                        break;
                    }
                }
            }
            // Check if data points into a combo buffer (per-expert cache path)
            if (!is_heap_ptr) {
                for (auto & kv : bsctx->combo_buffers) {
                    const uint8_t * buf_start = kv.second.data();
                    const uint8_t * buf_end   = buf_start + kv.second.size();
                    const uint8_t * data_ptr  = (const uint8_t *)base_tensor->data;
                    if (data_ptr >= buf_start && data_ptr < buf_end) {
                        is_heap_ptr = true;
                        break;
                    }
                }
            }
            if (!is_heap_ptr) {
                size_t sharp_nbytes = bs_tensor_nbytes(base_tensor->type, backup.ne);
                bs_release_mmap_pages(bsctx, base_tensor->data, sharp_nbytes);
            }
        }

        base_tensor->data     = backup.original_data;
        base_tensor->type     = backup.original_type;
        base_tensor->view_src = backup.original_view_src;
        base_tensor->extra    = backup.original_extra;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            base_tensor->nb[d] = backup.original_nb[d];
        }

        // Release the sharp data buffer (only if we allocated one)
        if (!backup.zero_copy) {
            backup.sharp_data.clear();
            backup.sharp_data.shrink_to_fit();
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_evict_to_budget(
        llama_blurry_sharp_context * bsctx,
        int64_t                      target_bytes) {
    if (!bsctx) return 0;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    int32_t n_evicted = 0;

    while (bsctx->current_backup_bytes > target_bytes && !bsctx->layer_backups.empty()) {
        // Find the layer to evict based on policy
        int evict_layer = -1;

        switch (bsctx->params.eviction_policy) {
            case LLAMA_BS_EVICT_LRU: {
                // Evict the layer with the smallest apply_sequence (oldest use)
                uint64_t min_seq = UINT64_MAX;
                for (auto & kv : bsctx->layer_backups) {
                    if (kv.second.is_sharpened && kv.second.apply_sequence < min_seq) {
                        min_seq = kv.second.apply_sequence;
                        evict_layer = kv.first;
                    }
                }
                break;
            }
            case LLAMA_BS_EVICT_FIFO: {
                // Same as LRU for this implementation (oldest first)
                uint64_t min_seq = UINT64_MAX;
                for (auto & kv : bsctx->layer_backups) {
                    if (kv.second.is_sharpened && kv.second.apply_sequence < min_seq) {
                        min_seq = kv.second.apply_sequence;
                        evict_layer = kv.first;
                    }
                }
                break;
            }
            case LLAMA_BS_EVICT_PRIORITY: {
                // Evict the layer with the most backup bytes (biggest recovery)
                int64_t max_bytes = -1;
                for (auto & kv : bsctx->layer_backups) {
                    if (kv.second.is_sharpened && kv.second.backup_bytes > max_bytes) {
                        max_bytes = kv.second.backup_bytes;
                        evict_layer = kv.first;
                    }
                }
                break;
            }
        }

        if (evict_layer < 0) break;

        // Restore this layer (unlock not needed, we hold the lock already)
        auto it = bsctx->layer_backups.find(evict_layer);
        if (it == bsctx->layer_backups.end()) break;

        int64_t t_start = bs_time_us();

        for (auto & tb : it->second.tensor_backups) {
            bs_restore_single_tensor(bsctx, tb);
        }

        bsctx->current_backup_bytes -= it->second.backup_bytes;
        bsctx->layer_backups.erase(it);

        int64_t t_end = bs_time_us();
        bsctx->metrics.total_time_restore_us += (t_end - t_start);
        bsctx->metrics.n_evictions++;
        bsctx->metrics.n_restore_calls++;

        ++n_evicted;

        if (bsctx->params.verbose) {
            LLAMA_LOG_INFO("%s: evicted layer %d (backup_bytes now = %" PRId64 ")\n",
                          __func__, evict_layer, bsctx->current_backup_bytes);
        }
    }

    return n_evicted;
}

// ---------------------------------------------------------------------------
// Apply layer
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_apply_layer(
        llama_blurry_sharp_context * bsctx,
        int                          layer_idx) {
    if (!bsctx || !bsctx->initialized) return -1;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    // Check eligibility
    if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
        if (bsctx->params.verbose) {
            LLAMA_LOG_INFO("%s: layer %d is not eligible for sharpening\n",
                          __func__, layer_idx);
        }
        return -1;
    }

    // Check if already sharpened
    {
        auto it = bsctx->layer_backups.find(layer_idx);
        if (it != bsctx->layer_backups.end() && it->second.is_sharpened) {
            // Already sharpened – just touch the sequence counter for LRU
            it->second.apply_sequence = bsctx->apply_sequence_counter++;
            it->second.apply_timestamp_us = bs_time_us();
            return 0;
        }
    }

    // Check max_sharp_layers limit
    if (bsctx->params.max_sharp_layers > 0) {
        int32_t n_currently_sharp = 0;
        for (auto & kv : bsctx->layer_backups) {
            if (kv.second.is_sharpened) n_currently_sharp++;
        }
        if (n_currently_sharp >= bsctx->params.max_sharp_layers) {
            // Need to evict at least one layer
            int64_t budget = bsctx->current_backup_bytes; // keep same byte budget
            int32_t evicted = llama_blurry_sharp_evict_to_budget(bsctx,
                budget > 0 ? budget - 1 : 0);
            // Recount
            n_currently_sharp = 0;
            for (auto & kv : bsctx->layer_backups) {
                if (kv.second.is_sharpened) n_currently_sharp++;
            }
            if (n_currently_sharp >= bsctx->params.max_sharp_layers) {
                LLAMA_LOG_WARN("%s: max_sharp_layers=%d reached, cannot sharpen layer %d\n",
                              __func__, bsctx->params.max_sharp_layers, layer_idx);
                return -3;
            }
            (void)evicted;
        }
    }

    // Get tensor names for this layer
    auto layer_it = bsctx->layer_tensor_names.find(layer_idx);
    if (layer_it == bsctx->layer_tensor_names.end() || layer_it->second.empty()) {
        LLAMA_LOG_WARN("%s: no sharp tensors found for layer %d\n", __func__, layer_idx);
        return -2;
    }

    int64_t t_start = bs_time_us();

    // -----------------------------------------------------------------------
    // PERMANENT MODE: one-way overwrite, no backup, no restore capability.
    // Dramatically reduces memory usage for static overlay (apply once).
    // -----------------------------------------------------------------------
    if (bsctx->params.permanent) {
        int n_success = 0;
        int n_fail    = 0;

        bs_prefetch_layer_mmap(bsctx, layer_idx);

        for (const auto & tensor_name : layer_it->second) {
            auto info_it = bsctx->sharp_index.find(tensor_name);
            if (info_it == bsctx->sharp_index.end()) continue;

            const auto & sharp_info = info_it->second;

            ggml_tensor * base_tensor = sharp_info.base_tensor;
            if (!base_tensor) {
                base_tensor = bs_find_model_tensor(bsctx->model, tensor_name.c_str());
            }
            if (!base_tensor) {
                LLAMA_LOG_WARN("%s: base tensor '%s' not found, skipping\n",
                              __func__, tensor_name.c_str());
                ++n_fail;
                continue;
            }

            ggml_type old_type = base_tensor->type;
            int64_t sharp_bytes = bs_overlay_single_tensor_permanent(bsctx, sharp_info, base_tensor);
            if (sharp_bytes < 0) {
                LLAMA_LOG_ERROR("%s: permanent overlay failed for tensor '%s'\n",
                               __func__, tensor_name.c_str());
                ++n_fail;
                continue;
            }

            bsctx->metrics.total_sharp_bytes_read += sharp_bytes;
            ++n_success;

            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: permanent overlay '%s' (%s → %s, %zu bytes)\n",
                              __func__, tensor_name.c_str(),
                              ggml_type_name(old_type),
                              ggml_type_name(sharp_info.type),
                              sharp_info.nbytes);
            }
        }

        int64_t t_end = bs_time_us();
        bsctx->metrics.n_apply_calls++;
        bsctx->metrics.total_time_apply_us += (t_end - t_start);

        // Mark as sharpened (but with no backups — permanent)
        blurry_sharp_layer_backup marker;
        marker.layer_idx          = layer_idx;
        marker.is_sharpened       = (n_success > 0);
        marker.backup_bytes       = 0;  // no backup in permanent mode
        marker.sharp_bytes_read   = 0;
        marker.apply_timestamp_us = t_start;
        marker.apply_sequence     = bsctx->apply_sequence_counter++;
        // tensor_backups is intentionally empty — no restore possible
        bsctx->layer_backups[layer_idx] = std::move(marker);

        if (bsctx->params.verbose) {
            LLAMA_LOG_INFO("%s: layer %d permanently sharpened (%d/%d tensors, %" PRId64 " us)\n",
                          __func__, layer_idx, n_success, n_success + n_fail,
                          t_end - t_start);
        }

        return (n_success > 0) ? 0 : -4;
    }

    // -----------------------------------------------------------------------
    // NORMAL MODE: overlay with backup for later restore
    // -----------------------------------------------------------------------

    blurry_sharp_layer_backup layer_backup;
    layer_backup.layer_idx       = layer_idx;
    layer_backup.is_sharpened    = false; // set to true only on success
    layer_backup.backup_bytes    = 0;
    layer_backup.sharp_bytes_read = 0;
    layer_backup.apply_timestamp_us = t_start;
    layer_backup.apply_sequence  = bsctx->apply_sequence_counter++;

    bool any_failed = false;

    // Mark this layer as "being built" so the LRU eviction code inside
    // bs_overlay_single_tensor does NOT evict cache entries that were
    // just allocated for earlier tensors of THIS layer.  Without this
    // guard, tensor B's overlay could evict tensor A's freshly-cached
    // buffer, leaving tensor A's base_tensor->data pointing at freed
    // GPU memory → illegal memory access when the compute kernel runs.
    bsctx->building_layer_idx = layer_idx;

    // Release combo_buffers from previous layers to prevent unbounded
    // memory growth.  Each layer has unique tensor names so entries
    // accumulate across layers (60 layers × 3 tensors × full tensor
    // size = tens of GB).  Only keep buffers for the current layer.
    {
        auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
        std::unordered_set<std::string> current_layer_tensors;
        if (lt_it != bsctx->layer_tensor_names.end()) {
            current_layer_tensors.insert(lt_it->second.begin(), lt_it->second.end());
        }
        for (auto it = bsctx->combo_buffers.begin(); it != bsctx->combo_buffers.end(); ) {
            if (current_layer_tensors.find(it->first) == current_layer_tensors.end()) {
                it = bsctx->combo_buffers.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Prefetch this layer's sharp data from disk into the page cache.
    // The madvise(WILLNEED) calls are non-blocking — the kernel starts
    // async I/O immediately.  By the time we iterate through the tensors
    // below, many pages will already be resident, avoiding synchronous
    // page-fault stalls.  Each tensor's pages are then released via
    // MADV_DONTNEED after consumption, so RAM never accumulates more
    // than ~one layer's worth of sharp data.
    bs_prefetch_layer_mmap(bsctx, layer_idx);

    // Sort tensor names by (split_idx, file_offset) for sequential disk I/O.
    // This ensures reads within a layer are issued in file order, maximizing
    // throughput on HDDs and improving prefetch locality on SSDs.
    std::vector<std::string> sorted_tensors(layer_it->second.begin(), layer_it->second.end());
    std::sort(sorted_tensors.begin(), sorted_tensors.end(), [&](const std::string & a, const std::string & b) {
        auto ia = bsctx->sharp_index.find(a);
        auto ib = bsctx->sharp_index.find(b);
        if (ia == bsctx->sharp_index.end()) return false;
        if (ib == bsctx->sharp_index.end()) return true;
        if (ia->second.split_idx != ib->second.split_idx) return ia->second.split_idx < ib->second.split_idx;
        return ia->second.file_offset < ib->second.file_offset;
    });

    for (const auto & tensor_name : sorted_tensors) {
        auto info_it = bsctx->sharp_index.find(tensor_name);
        if (info_it == bsctx->sharp_index.end()) continue;

        const auto & sharp_info = info_it->second;

        // Find matching tensor in base model (use cached pointer for O(1) lookup)
        ggml_tensor * base_tensor = sharp_info.base_tensor;
        if (!base_tensor) {
            base_tensor = bs_find_model_tensor(bsctx->model, tensor_name.c_str());
        }
        if (!base_tensor) {
            LLAMA_LOG_WARN("%s: base tensor '%s' not found, skipping\n",
                          __func__, tensor_name.c_str());
            continue;
        }

        // Check memory budget before proceeding.
        // With pointer-swap, the cost is the sharp data buffer we allocate
        // (not the blurry tensor size, which stays in place untouched).
        int64_t sharp_cost = (int64_t)sharp_info.nbytes;
        if (bsctx->params.memory_budget_bytes > 0 &&
            (bsctx->current_backup_bytes + sharp_cost) > bsctx->params.memory_budget_bytes) {
            // Try eviction
            int64_t needed = bsctx->current_backup_bytes + sharp_cost
                           - bsctx->params.memory_budget_bytes;
            int64_t target = bsctx->current_backup_bytes - needed;
            if (target < 0) target = 0;
            llama_blurry_sharp_evict_to_budget(bsctx, target);

            // Re-check
            if ((bsctx->current_backup_bytes + sharp_cost) > bsctx->params.memory_budget_bytes) {
                LLAMA_LOG_WARN("%s: memory budget exceeded, cannot overlay tensor '%s'\n",
                              __func__, tensor_name.c_str());
                any_failed = true;
                continue;
            }
        }

        blurry_sharp_tensor_backup tb;
        ggml_type backup_out_type = base_tensor->type;  // save before swap
        int64_t sharp_bytes = bs_overlay_single_tensor(bsctx, sharp_info, base_tensor, tb);
        if (sharp_bytes < 0) {
            LLAMA_LOG_ERROR("%s: failed to overlay tensor '%s'\n",
                           __func__, tensor_name.c_str());
            any_failed = true;
            continue;
        }

        layer_backup.backup_bytes     += sharp_bytes;
        layer_backup.sharp_bytes_read += sharp_bytes;
        layer_backup.tensor_backups.push_back(std::move(tb));

        bsctx->current_backup_bytes += sharp_bytes;
        bsctx->metrics.total_sharp_bytes_read += sharp_bytes;
        bsctx->metrics.total_backup_bytes_written += sharp_bytes;

        // Capture strategy info AFTER the overlay
        const blurry_sharp_tensor_backup & last_tb = layer_backup.tensor_backups.back();
        const char * strategy_str =
            last_tb.is_device  ? "device-swap" :
            last_tb.zero_copy  ? "zero-copy"   : "buffered";

        if (bsctx->params.verbose) {
            LLAMA_LOG_INFO("%s: overlaid tensor '%s' (%s, %s → %s, %zu sharp bytes)\n",
                          __func__, tensor_name.c_str(),
                          strategy_str,
                          ggml_type_name(backup_out_type),
                          ggml_type_name(sharp_info.type),
                          sharp_info.nbytes);
        }
    }

    if (layer_backup.tensor_backups.empty()) {
        bsctx->building_layer_idx = -1;
        return any_failed ? -4 : -2;
    }

    // Safety: if any tensor failed to overlay, roll back the entire layer.
    // A partially-sharpened layer (some tensors IQ1_S, others Q4_K_S) will
    // produce garbage output because the quant types are incompatible within
    // a single layer's computation.  It's better to leave the layer fully
    // blurry than to have it in a mixed/corrupt state.
    if (any_failed) {
        LLAMA_LOG_WARN("%s: layer %d had %d tensor failures — rolling back %d "
                      "successful overlays to prevent mixed-quant corruption. "
                      "This layer will run with BLURRY (low-quality) weights. "
                      "To fix: increase --n-cpu-moe to cover this layer (CPU path "
                      "uses zero-copy mmap, no extra memory needed), or free VRAM "
                      "with lower -c, -ctk q8_0, or fewer -ngl.\n",
                      __func__, layer_idx,
                      (int)(layer_it->second.size() - layer_backup.tensor_backups.size()),
                      (int)layer_backup.tensor_backups.size());

        for (auto & tb : layer_backup.tensor_backups) {
            // If this tensor's device buffer was added to the cache during
            // overlay, evict it NOW so the GPU memory is actually freed.
            // Otherwise the cache holds onto buffers from a rolled-back
            // layer, leaking GPU memory and causing OOM cascades for
            // subsequent layers.
            if (tb.is_device && tb.device_buf_cached) {
                auto cache_it = bsctx->device_cache.find(tb.tensor_name);
                if (cache_it != bsctx->device_cache.end()) {
                    if (cache_it->second.buffer) {
                        bsctx->device_cache_bytes -= (int64_t)cache_it->second.nbytes;
                        ggml_backend_buffer_free(cache_it->second.buffer);
                    }
                    bsctx->device_cache.erase(cache_it);
                }
                // Null out the pointer so bs_restore_single_tensor does NOT
                // double-free the same buffer we just freed via the cache.
                tb.device_sharp_buffer = nullptr;
                tb.device_buf_cached = false;
            }
            bs_restore_single_tensor(bsctx, tb);
        }
        bsctx->current_backup_bytes -= layer_backup.backup_bytes;
        bsctx->building_layer_idx = -1;
        return -4;
    }

    layer_backup.is_sharpened = true;
    bsctx->layer_backups[layer_idx] = std::move(layer_backup);
    bsctx->building_layer_idx = -1;

    int64_t t_end = bs_time_us();
    bsctx->metrics.n_apply_calls++;
    bsctx->metrics.total_time_apply_us += (t_end - t_start);

    if (bsctx->params.verbose) {
        LLAMA_LOG_INFO("%s: layer %d sharpened (%d tensors, %" PRId64 " backup bytes, %" PRId64 " us)\n",
                      __func__, layer_idx,
                      (int)bsctx->layer_backups[layer_idx].tensor_backups.size(),
                      bsctx->layer_backups[layer_idx].backup_bytes,
                      t_end - t_start);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Restore layer
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_restore_layer(
        llama_blurry_sharp_context * bsctx,
        int                          layer_idx) {
    if (!bsctx) return -1;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    // In permanent mode, restore is impossible — blurry data was discarded.
    if (bsctx->params.permanent) {
        auto it = bsctx->layer_backups.find(layer_idx);
        if (it != bsctx->layer_backups.end() && it->second.is_sharpened) {
            LLAMA_LOG_WARN("%s: permanent mode — cannot restore layer %d, blurry data was discarded\n",
                          __func__, layer_idx);
        }
        return -1;
    }

    auto it = bsctx->layer_backups.find(layer_idx);
    if (it == bsctx->layer_backups.end() || !it->second.is_sharpened) {
        return -1;
    }

    int64_t t_start = bs_time_us();

    for (auto & tb : it->second.tensor_backups) {
        if (!bs_restore_single_tensor(bsctx, tb)) {
            LLAMA_LOG_ERROR("%s: failed to restore tensor '%s' for layer %d\n",
                           __func__, tb.tensor_name.c_str(), layer_idx);
        }
    }

    bsctx->current_backup_bytes -= it->second.backup_bytes;
    bsctx->layer_backups.erase(it);

    // Lazy swap: push this layer's RAM-cached pages to swap now that the
    // tensor pointers have been restored to blurry.  The data is "cold" —
    // freeing RAM for the blurry model / KV cache while keeping the sharp
    // data quickly accessible via swap-in on the next apply_layer call.
    if (bsctx->lazy_swap_enabled) {
        bs_lazy_swap_pageout_layer(bsctx, layer_idx);
    }

    int64_t t_end = bs_time_us();
    bsctx->metrics.n_restore_calls++;
    bsctx->metrics.total_time_restore_us += (t_end - t_start);

    if (bsctx->params.verbose) {
        LLAMA_LOG_INFO("%s: layer %d restored to blurry (%" PRId64 " us)\n",
                      __func__, layer_idx, t_end - t_start);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Restore all
// ---------------------------------------------------------------------------

void llama_blurry_sharp_restore_all(llama_blurry_sharp_context * bsctx) {
    if (!bsctx) return;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    // In permanent mode, restore is a no-op — data is gone, warn the caller.
    if (bsctx->params.permanent) {
        bool any_sharp = false;
        for (auto & kv : bsctx->layer_backups) {
            if (kv.second.is_sharpened) { any_sharp = true; break; }
        }
        if (any_sharp) {
            LLAMA_LOG_WARN("%s: permanent mode — restore_all is a no-op, blurry data was discarded\n",
                          __func__);
        }
        return;
    }

    int64_t t_start = bs_time_us();

    // Collect layer indices first (we modify the map during iteration)
    std::vector<int> layers_to_restore;
    for (auto & kv : bsctx->layer_backups) {
        if (kv.second.is_sharpened) {
            layers_to_restore.push_back(kv.first);
        }
    }

    for (int layer_idx : layers_to_restore) {
        auto it = bsctx->layer_backups.find(layer_idx);
        if (it == bsctx->layer_backups.end()) continue;

        for (auto & tb : it->second.tensor_backups) {
            bs_restore_single_tensor(bsctx, tb);
        }
        bsctx->current_backup_bytes -= it->second.backup_bytes;
        bsctx->layer_backups.erase(it);
        bsctx->metrics.n_restore_calls++;

        // Lazy swap: push this layer's RAM-cached pages to swap now that
        // tensor pointers have been restored to blurry.  Frees RAM for
        // the blurry model / KV cache; next apply_layer will swap-in
        // from SSD instead of re-reading from the GGUF file.
        if (bsctx->lazy_swap_enabled) {
            bs_lazy_swap_pageout_layer(bsctx, layer_idx);
        }
    }

    int64_t t_end = bs_time_us();
    bsctx->metrics.total_time_restore_us += (t_end - t_start);

    if (bsctx->params.verbose) {
        LLAMA_LOG_INFO("%s: restored %zu layers to blurry (%" PRId64 " us)\n",
                      __func__, layers_to_restore.size(), t_end - t_start);
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

std::vector<blurry_sharp_routing_decision> llama_blurry_sharp_route(
        llama_blurry_sharp_context * bsctx,
        const float                * activations,
        int32_t                      n_tokens,
        int32_t                      n_embd) {
    std::vector<blurry_sharp_routing_decision> decisions;
    if (!bsctx || !bsctx->initialized) return decisions;

    for (int layer_idx : bsctx->sharp_layer_indices) {
        if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
            continue;
        }

        blurry_sharp_routing_decision dec;
        dec.layer_idx = layer_idx;
        dec.confidence = 1.0f;
        dec.should_sharpen = false;

        switch (bsctx->params.router_strategy) {
            case LLAMA_BS_ROUTER_ALWAYS:
                dec.should_sharpen = true;
                dec.confidence = 0.0f;
                break;

            case LLAMA_BS_ROUTER_NEVER:
                dec.should_sharpen = false;
                dec.confidence = 1.0f;
                break;

            case LLAMA_BS_ROUTER_NORM: {
                // Compute per-token L2 norms and derive confidence.
                // Higher norm → lower confidence → more likely to sharpen.
                if (!activations || n_tokens <= 0 || n_embd <= 0) {
                    dec.should_sharpen = false;
                    dec.confidence = 1.0f;
                    break;
                }

                dec.per_token_confidences.resize((size_t)n_tokens);
                float max_norm = 0.0f;

                // First pass: compute norms
                std::vector<float> norms((size_t)n_tokens);
                for (int t = 0; t < n_tokens; ++t) {
                    const float * row = activations + (size_t)t * (size_t)n_embd;
                    float sum_sq = 0.0f;
                    for (int e = 0; e < n_embd; ++e) {
                        sum_sq += row[e] * row[e];
                    }
                    norms[t] = std::sqrt(sum_sq);
                    if (norms[t] > max_norm) max_norm = norms[t];
                }

                // Second pass: map to confidences
                float inv_max = (max_norm > 1e-12f) ? (1.0f / max_norm) : 1.0f;
                float avg_confidence = 0.0f;
                int n_low_confidence = 0;

                for (int t = 0; t < n_tokens; ++t) {
                    float score = norms[t] * inv_max;
                    float conf = std::max(0.0f, 1.0f - score);
                    dec.per_token_confidences[t] = conf;
                    avg_confidence += conf;
                    if (conf < bsctx->params.router_min_confidence) {
                        ++n_low_confidence;
                    }
                }
                avg_confidence /= (float)n_tokens;

                dec.confidence = avg_confidence;
                dec.should_sharpen = (n_low_confidence > 0) ||
                                    (avg_confidence < bsctx->params.router_min_confidence);
                break;
            }
        }

        decisions.push_back(std::move(dec));
    }

    return decisions;
}

// ---------------------------------------------------------------------------
// Auto-sharpen
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_auto_sharpen(
        llama_blurry_sharp_context * bsctx,
        const float                * activations,
        int32_t                      n_tokens,
        int32_t                      n_embd) {
    if (!bsctx || !bsctx->initialized) return 0;

    auto decisions = llama_blurry_sharp_route(bsctx, activations, n_tokens, n_embd);

    int32_t n_sharpened = 0;
    for (auto & dec : decisions) {
        if (dec.should_sharpen) {
            int32_t ret = llama_blurry_sharp_apply_layer(bsctx, dec.layer_idx);
            if (ret == 0) {
                ++n_sharpened;
            }
        }
    }

    return n_sharpened;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

bool llama_blurry_sharp_is_layer_sharp(
        const llama_blurry_sharp_context * bsctx,
        int                                layer_idx) {
    if (!bsctx) return false;
    auto it = bsctx->layer_backups.find(layer_idx);
    if (it == bsctx->layer_backups.end()) return false;
    return it->second.is_sharpened;
}

bool llama_blurry_sharp_get_layer_state(
        const llama_blurry_sharp_context * bsctx,
        int                                layer_idx,
        llama_blurry_sharp_layer_state   * out) {
    if (!bsctx || !out) return false;
    if (layer_idx < 0 || layer_idx >= (int)bsctx->model->layers.size()) return false;

    out->layer_idx          = layer_idx;
    out->is_sharpened       = false;
    out->n_tensors_overlaid = 0;
    out->backup_bytes       = 0;
    out->sharp_bytes_read   = 0;
    out->timestamp_us       = 0;

    auto it = bsctx->layer_backups.find(layer_idx);
    if (it != bsctx->layer_backups.end() && it->second.is_sharpened) {
        out->is_sharpened       = true;
        out->n_tensors_overlaid = (int32_t)it->second.tensor_backups.size();
        out->backup_bytes       = it->second.backup_bytes;
        out->sharp_bytes_read   = it->second.sharp_bytes_read;
        out->timestamp_us       = it->second.apply_timestamp_us;
    }

    return true;
}

llama_blurry_sharp_state llama_blurry_sharp_get_state(
        const llama_blurry_sharp_context * bsctx) {
    llama_blurry_sharp_state state = {};
    if (!bsctx) return state;

    state.n_layers_total = (int32_t)bsctx->model->layers.size();
    state.n_layers_sharpened = 0;
    state.total_backup_bytes = bsctx->current_backup_bytes;
    state.total_sharp_bytes_read = bsctx->metrics.total_sharp_bytes_read;
    state.memory_budget_bytes = bsctx->params.memory_budget_bytes;
    state.gpu_budget_bytes = bsctx->params.gpu_budget_bytes;
    state.gpu_device_bytes_used = bsctx->current_device_sharp_bytes;
    state.max_sharp_layers = bsctx->params.max_sharp_layers;
    state.n_device_tensors_skipped = bsctx->n_device_tensors_skipped;

    for (auto & kv : bsctx->layer_backups) {
        if (kv.second.is_sharpened) {
            state.n_layers_sharpened++;
        }
    }

    return state;
}

blurry_sharp_metrics llama_blurry_sharp_get_metrics(
        const llama_blurry_sharp_context * bsctx) {
    if (!bsctx) {
        blurry_sharp_metrics m = {};
        return m;
    }
    return bsctx->metrics;
}

void llama_blurry_sharp_reset_metrics(llama_blurry_sharp_context * bsctx) {
    if (!bsctx) return;
    std::lock_guard<std::mutex> lock(bsctx->mtx);
    bsctx->metrics = {};
}

// ---------------------------------------------------------------------------
// Batch apply all eligible layers
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_apply_all(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;

    int64_t t_start = bs_time_us();

    if (bsctx->params.permanent) {
        LLAMA_LOG_INFO("%s: applying all layers in PERMANENT mode (no backup, reduced memory)\n",
                      __func__);
    }

    // When applying ALL layers we know the access pattern is sequential
    // through the file.  Set MADV_SEQUENTIAL so the kernel does aggressive
    // readahead and reclaims pages behind the cursor.  Each tensor's pages
    // are still explicitly released via MADV_DONTNEED after consumption.
    // After the loop we reset to MADV_NORMAL for subsequent selective use.
#ifndef _WIN32
    for (size_t si = 0; si < bsctx->sharp_mmaps.size(); ++si) {
        if (bsctx->sharp_mmaps[si]) {
            posix_madvise(bsctx->sharp_mmaps[si]->addr(),
                          bsctx->sharp_mmaps[si]->size(),
                          POSIX_MADV_SEQUENTIAL);
        }
    }
#endif

    int32_t n_sharpened = 0;
    int32_t n_eligible  = 0;

    for (int layer_idx : bsctx->sharp_layer_indices) {
        if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
            continue;
        }
        ++n_eligible;
        int32_t ret = llama_blurry_sharp_apply_layer(bsctx, layer_idx);
        if (ret == 0) {
            ++n_sharpened;
        }
    }

    int64_t t_end = bs_time_us();
    double elapsed_ms = (t_end - t_start) / 1000.0;

    // Reset to NORMAL for subsequent selective (random) access patterns.
#ifndef _WIN32
    for (size_t si = 0; si < bsctx->sharp_mmaps.size(); ++si) {
        if (bsctx->sharp_mmaps[si]) {
            posix_madvise(bsctx->sharp_mmaps[si]->addr(),
                          bsctx->sharp_mmaps[si]->size(),
                          POSIX_MADV_NORMAL);
        }
    }
#endif

    LLAMA_LOG_INFO("%s: sharpened %d/%d eligible layers in %.2f ms\n",
                  __func__, n_sharpened, n_eligible, elapsed_ms);

    return n_sharpened;
}

// ---------------------------------------------------------------------------
// Permanently overlay non-expert tensors (attention, norms, etc.)
//
// MoE expert tensors (_exps) are handled by JIT overlay during graph execution.
// All other tensors (attention Q/K/V/O projections, layer norms, embeddings,
// output head) are always used and can be permanently upgraded to sharp quality
// at startup.  This is especially important for GPU tensors — they hold
// attention weights that directly affect KV cache quality.
// ---------------------------------------------------------------------------

void llama_blurry_sharp_set_skip_non_expert_gpu(
        llama_blurry_sharp_context * bsctx, bool skip_gpu) {
    if (!bsctx) return;
    bsctx->skip_non_expert_gpu = skip_gpu;
}

int32_t llama_blurry_sharp_apply_non_expert_permanent(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;

    int64_t t_start = bs_time_us();
    int n_overlaid_gpu = 0;
    int n_overlaid_cpu = 0;
    int n_skipped  = 0;
    int n_failed   = 0;
    int64_t total_bytes_gpu = 0;
    int64_t total_bytes_cpu = 0;

    LLAMA_LOG_INFO("%s: permanently overlaying non-expert tensors with sharp data\n", __func__);

    for (auto & [name, sharp_info] : bsctx->sharp_index) {
        // Skip expert tensors — those are handled by JIT
        if (bs_is_expert_tensor(name)) {
            ++n_skipped;
            continue;
        }
        // Skip gate tensors — handled by apply_gates_permanent
        if (bs_is_gate_tensor(name)) {
            ++n_skipped;
            continue;
        }

        ggml_tensor * base_tensor = sharp_info.base_tensor;
        if (!base_tensor) {
            base_tensor = bs_find_model_tensor(bsctx->model, name.c_str());
        }
        if (!base_tensor) {
            if (bsctx->params.verbose) {
                LLAMA_LOG_WARN("%s: base tensor '%s' not found, skipping\n",
                              __func__, name.c_str());
            }
            ++n_failed;
            continue;
        }

        bool is_gpu = base_tensor->buffer && !ggml_backend_buffer_is_host(base_tensor->buffer);

        // Skip GPU tensors if requested (saves VRAM for constrained setups)
        if (is_gpu && bsctx->skip_non_expert_gpu) {
            ++n_skipped;
            continue;
        }

        ggml_type old_type = base_tensor->type;
        int64_t sharp_bytes = bs_overlay_single_tensor_permanent(bsctx, sharp_info, base_tensor);
        if (sharp_bytes < 0) {
            LLAMA_LOG_WARN("%s: permanent overlay failed for '%s' (%s, %s)\n",
                          __func__, name.c_str(),
                          is_gpu ? "GPU" : "CPU",
                          ggml_type_name(old_type));
            ++n_failed;
            continue;
        }

        if (is_gpu) {
            ++n_overlaid_gpu;
            total_bytes_gpu += sharp_bytes;
        } else {
            ++n_overlaid_cpu;
            total_bytes_cpu += sharp_bytes;
        }

        LLAMA_LOG_INFO("%s: %s '%s' (%s → %s, %.1f MiB)\n",
                      __func__, is_gpu ? "GPU" : "CPU", name.c_str(),
                      ggml_type_name(old_type),
                      ggml_type_name(sharp_info.type),
                      sharp_info.nbytes / (1024.0 * 1024.0));
    }

    int64_t t_end = bs_time_us();
    int n_overlaid = n_overlaid_gpu + n_overlaid_cpu;
    LLAMA_LOG_INFO("%s: done — %d GPU tensors (%.1f MiB), %d CPU tensors (%.1f MiB), "
                  "%d expert tensors skipped, %d failed, %.2f ms\n",
                  __func__,
                  n_overlaid_gpu, total_bytes_gpu / (1024.0 * 1024.0),
                  n_overlaid_cpu, total_bytes_cpu / (1024.0 * 1024.0),
                  n_skipped, n_failed, (t_end - t_start) / 1000.0);

    return n_overlaid;
}

// ---------------------------------------------------------------------------
// Permanently overlay gate/router tensors only
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_apply_gates_permanent(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;

    int n_overlaid = 0;
    int n_failed   = 0;

    LLAMA_LOG_INFO("%s: permanently overlaying gate/router tensors with sharp data\n", __func__);

    for (auto & [name, sharp_info] : bsctx->sharp_index) {
        if (!bs_is_gate_tensor(name)) continue;

        ggml_tensor * base_tensor = sharp_info.base_tensor;
        if (!base_tensor) {
            base_tensor = bs_find_model_tensor(bsctx->model, name.c_str());
        }
        if (!base_tensor) {
            ++n_failed;
            continue;
        }

        bool is_gpu = base_tensor->buffer && !ggml_backend_buffer_is_host(base_tensor->buffer);
        ggml_type old_type = base_tensor->type;
        int64_t sharp_bytes = bs_overlay_single_tensor_permanent(bsctx, sharp_info, base_tensor);
        if (sharp_bytes < 0) {
            LLAMA_LOG_WARN("%s: failed for '%s'\n", __func__, name.c_str());
            ++n_failed;
            continue;
        }

        ++n_overlaid;
        LLAMA_LOG_INFO("%s: %s '%s' (%s -> %s, %.1f MiB)\n",
                      __func__, is_gpu ? "GPU" : "CPU", name.c_str(),
                      ggml_type_name(old_type),
                      ggml_type_name(sharp_info.type),
                      sharp_info.nbytes / (1024.0 * 1024.0));
    }

    LLAMA_LOG_INFO("%s: done — %d gate tensors overlaid, %d failed\n",
                  __func__, n_overlaid, n_failed);
    return n_overlaid;
}

// ---------------------------------------------------------------------------
// Warm live pages into RAM
// ---------------------------------------------------------------------------

void llama_blurry_sharp_warm_live_pages(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return;

#ifndef _WIN32
    int64_t t_start = bs_time_us();
    int     n_warmed = 0;
    int64_t warmed_bytes = 0;

    for (auto & kv : bsctx->layer_backups) {
        if (!kv.second.is_sharpened) continue;

        for (auto & tb : kv.second.tensor_backups) {
            // Only warm CPU zero-copy tensors — their data is either an mmap
            // page (file-backed) or a RAM cache buffer (anonymous/swap-backed).
            // GPU tensors already live in VRAM (no warming needed).
            // Buffered CPU tensors live in heap memory (already in RAM).
            if (tb.is_device || !tb.zero_copy) continue;

            ggml_tensor * t = tb.base_tensor;
            if (!t || !t->data) continue;

            size_t nbytes = ggml_nbytes(t);
            if (nbytes == 0) continue;

            // Tell the kernel: "I need these pages, please pull them into
            // RAM."  For mmap pages this reads from the GGUF file; for
            // RAM-cache pages this swaps them in from the swap partition.
            // Both paths are non-blocking — the kernel starts async I/O
            // and returns immediately.
            posix_madvise(t->data, nbytes, POSIX_MADV_WILLNEED);

            ++n_warmed;
            warmed_bytes += (int64_t)nbytes;
        }
    }

    int64_t t_end = bs_time_us();

    if (n_warmed > 0) {
        LLAMA_LOG_INFO("%s: requested prefetch of %d zero-copy tensors "
                      "(%.2f MiB) into page cache in %.2f ms\n",
                      __func__, n_warmed,
                      warmed_bytes / (1024.0 * 1024.0),
                      (t_end - t_start) / 1000.0);
    }
#else
    GGML_UNUSED(bsctx);
#endif
}

// ---------------------------------------------------------------------------
// Prefetch a single layer's sharp mmap data into the page cache.
// Thin public wrapper around bs_prefetch_layer_mmap().
// ---------------------------------------------------------------------------

void llama_blurry_sharp_prefetch_layer(
        llama_blurry_sharp_context * bsctx,
        int32_t                      layer_idx) {
    if (!bsctx || !bsctx->initialized) return;

    // Start the async prefetch thread on first call (lazy init).
    // This avoids starting it during init when we don't yet know
    // which layers will be needed.
    bs_ensure_prefetch_thread(bsctx);

    // Queue this layer for async prefetch (first-pass ram_cache population).
    // The thread reads tensors from mmap/file into prefetch_cache in the
    // background, sorted by file offset for sequential I/O.
    int li = (int)layer_idx;
    bs_queue_prefetch_layers(bsctx, &li, 1);

    // Issue madvise(WILLNEED) for mmap pages (first pass) and
    // ram_cache pages (subsequent passes — triggers async swap-in).
    bs_prefetch_layer_mmap(bsctx, layer_idx);
}

// ---------------------------------------------------------------------------
// Targeted expert-slice prefetch for a single layer.
//
// Only issues madvise(WILLNEED) on the specific expert slices identified
// by expert_ids, not the full tensor.  For a 160-expert model with 2 active
// experts, this prefetches ~27 MiB per layer instead of ~2.2 GiB.
// ---------------------------------------------------------------------------

void llama_blurry_sharp_prefetch_expert_slices(
        llama_blurry_sharp_context * bsctx,
        int32_t                      layer_idx,
        const int32_t              * expert_ids,
        int32_t                      n_experts) {
    if (!bsctx || !bsctx->initialized || !expert_ids || n_experts <= 0) return;

#ifndef _WIN32
    auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
    if (lt_it == bsctx->layer_tensor_names.end()) return;

    for (const auto & tensor_name : lt_it->second) {
        // Only prefetch expert tensors (skip attention, norms, etc.)
        if (!bs_is_expert_tensor(tensor_name)) continue;

        // Check RAM cache first
        if (bsctx->lazy_swap_enabled || bsctx->ram_cache_populated) {
            auto cache_it = bsctx->ram_cache.find(tensor_name);
            if (cache_it != bsctx->ram_cache.end() && !cache_it->second.empty()) {
                auto info_it = bsctx->sharp_index.find(tensor_name);
                if (info_it != bsctx->sharp_index.end() && info_it->second.ne[2] > 1) {
                    size_t slice_bytes = cache_it->second.size() / (size_t)info_it->second.ne[2];
                    for (int32_t ei = 0; ei < n_experts; ++ei) {
                        size_t off = (size_t)expert_ids[ei] * slice_bytes;
                        if (off + slice_bytes <= cache_it->second.size()) {
                            madvise(cache_it->second.data() + off, slice_bytes, MADV_WILLNEED);
                        }
                    }
                }
                continue;
            }
        }

        // mmap path: prefetch only the requested expert slices
        auto info_it = bsctx->sharp_index.find(tensor_name);
        if (info_it == bsctx->sharp_index.end()) continue;
        const auto & si = info_it->second;

        int split = si.split_idx;
        if (split < 0 || split >= (int)bsctx->sharp_mmaps.size()) continue;
        if (!bsctx->sharp_mmaps[split]) continue;

        const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[split]->addr();
        size_t file_size = bsctx->sharp_mmaps[split]->size();

        if (si.ne[2] > 1) {
            size_t slice_bytes = si.nbytes / (size_t)si.ne[2];
            for (int32_t ei = 0; ei < n_experts; ++ei) {
                int32_t eid = expert_ids[ei];
                if (eid < 0 || eid >= si.ne[2]) continue;
                size_t slice_off = si.file_offset + (size_t)eid * slice_bytes;
                size_t end = slice_off + slice_bytes;
                if (end > file_size) end = file_size;
                if (slice_off < end) {
                    posix_madvise((void *)(mmap_base + slice_off), end - slice_off,
                                  POSIX_MADV_WILLNEED);
                }
            }
        }
    }
#else
    GGML_UNUSED(bsctx);
    GGML_UNUSED(layer_idx);
    GGML_UNUSED(expert_ids);
    GGML_UNUSED(n_experts);
#endif
}

// ---------------------------------------------------------------------------
// Async prefetch: read the NEXT layer's expert slices in the background
// using parallel pread, so data is ready when the callback fires.
//
// Called from the JIT callback after applying layer N.  Fires a background
// thread that reads layer N+1's predicted expert slices (same experts as
// layer N — ~70% overlap) into prefetch_buffers via parallel pread.
// When layer N+1's apply_experts runs, it checks prefetch_buffers first
// and swaps them into combo_buffers if the prediction matches.
// ---------------------------------------------------------------------------

void llama_blurry_sharp_async_prefetch_start(
        llama_blurry_sharp_context * bsctx,
        int32_t                      layer_idx,
        const int32_t              * expert_ids,
        int32_t                      n_experts) {
    if (!bsctx || !expert_ids || n_experts <= 0) return;
    if (!bsctx->params.parallel_expert_io) return;

    auto & ap = bsctx->async_prefetch;

    // Wait for any previous prefetch to finish
    if (ap.active.load()) {
        if (ap.worker.joinable()) ap.worker.join();
        ap.active = false;
    }

    // Save the prediction
    ap.layer_idx = layer_idx;
    ap.expert_ids.assign(expert_ids, expert_ids + n_experts);
    ap.ready  = false;
    ap.active = true;

    // Double-buffered prefetch: alternate between dbufs[0] and dbufs[1].
    // Each map accumulates buffers for one layer's expert tensors.
    // On the next cycle, we harvest buffers from the target map for reuse
    // (move out → move back under new tensor names).  Since all MoE layers
    // have identical tensor shapes, resize() is a no-op after first use.
    //
    // Flip write_idx: the NEW prefetch writes to the OTHER buffer.
    // The previous cycle's completed data (dbufs[old write_idx]) remains
    // available for consumption until this new prefetch completes.
    ap.write_idx = 1 - ap.write_idx;
    int wi = ap.write_idx;

    // Harvest reusable buffers from the target map (this map was last written
    // 2 cycles ago, so its data has already been consumed).
    // Move buffers to a std::array — avoids heap alloc for the reuse container.
    static constexpr size_t MAX_REUSE = 4;  // gate_exps, up_exps, down_exps (+spare)
    std::array<bs_aligned_buffer, MAX_REUSE> reuse_arr;
    int n_reuse = 0;
    for (auto & [name, buf] : ap.dbufs[wi]) {
        if (!buf.empty() && n_reuse < (int)MAX_REUSE) {
            reuse_arr[n_reuse++] = std::move(buf);
        }
    }
    ap.dbufs[wi].clear();  // safe: all buffers moved out (null ptrs)

    // Launch worker thread that does parallel pread into dbufs[wi]
    ap.worker = std::thread([bsctx, layer_idx, n_experts, wi,
                             reuse_bufs = std::move(reuse_arr),
                             n_reuse_bufs = n_reuse]() mutable {
        auto & ap = bsctx->async_prefetch;
        auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
        if (lt_it == bsctx->layer_tensor_names.end()) {
            ap.ready = true;
            return;
        }

        int reuse_idx = 0;
        for (const auto & tname : lt_it->second) {
            if (!bs_is_expert_tensor(tname)) continue;

            auto info_it = bsctx->sharp_index.find(tname);
            if (info_it == bsctx->sharp_index.end()) continue;
            const auto & si = info_it->second;
            if (si.ne[2] <= 1) continue;

            int split = si.split_idx;
            bool have_file = (split >= 0 && split < (int)bsctx->sharp_files.size()
                              && bsctx->sharp_files[split]);
            if (!have_file) continue;
            int fd = bsctx->sharp_files[split]->file_id();
            if (fd < 0) continue;

            size_t sharp_expert_slice = si.nbytes / (size_t)si.ne[2];

            // Reuse buffer from harvested pool or resize in-place
            if (reuse_idx < n_reuse_bufs && reuse_bufs[reuse_idx].size() >= si.nbytes) {
                ap.dbufs[wi][tname] = std::move(reuse_bufs[reuse_idx]);
                ap.dbufs[wi][tname].resize(si.nbytes);  // no-op if same size
                reuse_idx++;
            } else {
                ap.dbufs[wi][tname].resize(si.nbytes);
            }
            auto & buf = ap.dbufs[wi][tname];

            std::vector<bs_pread_task> tasks;
            tasks.reserve(n_experts);

            for (int32_t i = 0; i < n_experts; ++i) {
                int32_t eid = ap.expert_ids[i];
                if (eid < 0 || eid >= si.ne[2]) continue;
                size_t off = (size_t)eid * sharp_expert_slice;

                bs_pread_task t;
                t.fd     = fd;
                t.dst    = buf.data() + off;
                t.offset = (off_t)(si.file_offset + off);
                t.size   = sharp_expert_slice;
                t.result = 0;
                tasks.push_back(t);
            }

            // Use the dedicated prefetch IO pool (not the main one —
            // dispatch() is not reentrant and the main thread uses
            // bs_get_io_pool() concurrently from the eval callback).
            bs_prefetch_parallel_pread(tasks.data(), (int)tasks.size(),
                                       bsctx->params.cache_io_split);
        }

        ap.ready = true;
    });
}

// Check if async prefetch has data for the given layer+tensor.
// Copies matching expert slices from the prefetch buffer into combo_buffers.
// Returns the number of experts consumed (0 = no match).
// consumed_out[i] is set to true for each expert_ids[i] found in prefetch.
// The caller only needs sync I/O for experts where consumed_out[i] is false.
static int bs_async_prefetch_consume(
        llama_blurry_sharp_context * bsctx,
        const std::string          & tensor_name,
        int32_t                      layer_idx,
        const int32_t              * expert_ids,
        int32_t                      n_experts_req,
        bool                       * consumed_out) {
    auto & ap = bsctx->async_prefetch;
    if (!ap.ready.load() || ap.layer_idx != layer_idx) return 0;

    // Wait for worker to finish (should already be done since ready=true)
    if (ap.active.load()) {
        if (ap.worker.joinable()) ap.worker.join();
        ap.active = false;
    }

    // Read from the last-written buffer (dbufs[write_idx]).
    // The flip to the next write_idx happens later in prefetch_start,
    // so at consume time write_idx still points to the completed data.
    auto & read_map = ap.dbufs[ap.write_idx];
    auto pf_it = read_map.find(tensor_name);
    if (pf_it == read_map.end() || pf_it->second.empty()) return 0;

    // Look up tensor info for slice size calculation
    auto info_it = bsctx->sharp_index.find(tensor_name);
    if (info_it == bsctx->sharp_index.end()) return 0;
    const auto & si = info_it->second;
    if (si.ne[2] <= 1) return 0;
    size_t sharp_expert_slice = si.nbytes / (size_t)si.ne[2];

    // Build set of prefetched expert IDs
    std::unordered_set<int32_t> prefetched(ap.expert_ids.begin(), ap.expert_ids.end());

    // Ensure combo buffer is allocated
    auto & combo = bsctx->combo_buffers[tensor_name];
    if (combo.size() < si.nbytes) {
        combo.resize(si.nbytes);
    }

    int n_consumed = 0;
    for (int32_t i = 0; i < n_experts_req; ++i) {
        consumed_out[i] = false;
        int32_t eid = expert_ids[i];
        if (prefetched.find(eid) != prefetched.end()) {
            size_t off = (size_t)eid * sharp_expert_slice;
            if (off + sharp_expert_slice <= si.nbytes &&
                off + sharp_expert_slice <= pf_it->second.size()) {
                std::memcpy(combo.data() + off,
                            pf_it->second.data() + off,
                            sharp_expert_slice);
                consumed_out[i] = true;
                n_consumed++;
            }
        }
    }

    if (n_consumed > 0) {
        ap.n_hits += n_consumed;
    }
    ap.n_misses += (n_experts_req - n_consumed);

    return n_consumed;
}

// ---------------------------------------------------------------------------
// Parallel prefetch: pre-read multiple layers' data concurrently.
//
// Spawns up to n_threads temporary worker threads.  Each worker reads a
// subset of the requested layers from mmap/file into the prefetch_cache
// (the same staging area the single-thread prefetch uses).  When the main
// thread's apply loop needs a tensor, bs_consume_prefetched_tensor moves
// data from prefetch_cache → ram_cache in O(1).
//
// The mmap memcpy and file reads are the bottleneck — they benefit from
// parallelism because:
//   - Multiple mmap fault handlers run concurrently in the kernel
//   - NVMe/SSD controllers can serve multiple outstanding I/O requests
//   - Readahead at different file offsets avoids head-of-line blocking
//
// Thread-safety: workers only write to prefetch_cache (under prefetch_mtx),
// never to ram_cache or the tensor graph.  The main thread applies
// (pointer-swaps) sequentially after prefetch completes.
// ---------------------------------------------------------------------------

void llama_blurry_sharp_prefetch_layers_parallel(
        llama_blurry_sharp_context * bsctx,
        const int32_t              * layer_indices,
        int32_t                      n_layers,
        int32_t                      n_threads) {
    if (!bsctx || !bsctx->initialized || n_layers <= 0) return;
    if (n_threads <= 0) n_threads = 4;
    if (n_threads > n_layers) n_threads = n_layers;

    // Also issue madvise(WILLNEED) for all layers upfront so the kernel
    // starts async readahead in parallel with our worker threads.
    for (int32_t i = 0; i < n_layers; ++i) {
        bs_prefetch_layer_mmap(bsctx, layer_indices[i]);
    }

    // Single-thread fast path
    if (n_threads <= 1) {
        for (int32_t i = 0; i < n_layers; ++i) {
            int li = layer_indices[i];
            bs_queue_prefetch_layers(bsctx, &li, 1);
        }
        return;
    }

    // Worker function: process assigned layer indices
    auto worker = [&](int32_t thread_id) {
        for (int32_t i = thread_id; i < n_layers; i += n_threads) {
            int layer_idx = layer_indices[i];

            auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
            if (lt_it == bsctx->layer_tensor_names.end()) continue;

            // Collect tensor read descriptors, sorted by file offset
            struct read_desc {
                std::string name;
                int         split_idx;
                size_t      file_offset;
                size_t      nbytes;
            };
            std::vector<read_desc> reads;

            for (const auto & tensor_name : lt_it->second) {
                // Skip if already cached
                if (bsctx->ram_cache.count(tensor_name)) continue;
                {
                    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
                    if (bsctx->prefetch_cache.count(tensor_name)) continue;
                }

                auto info_it = bsctx->sharp_index.find(tensor_name);
                if (info_it == bsctx->sharp_index.end()) continue;
                const auto & si = info_it->second;

                reads.push_back({tensor_name, si.split_idx, si.file_offset, si.nbytes});
            }

            std::sort(reads.begin(), reads.end(),
                      [](const read_desc & a, const read_desc & b) {
                          if (a.split_idx != b.split_idx) return a.split_idx < b.split_idx;
                          return a.file_offset < b.file_offset;
                      });

            for (const auto & rd : reads) {
                // Double-check not cached (another thread may have read it)
                if (bsctx->ram_cache.count(rd.name)) continue;
                {
                    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
                    if (bsctx->prefetch_cache.count(rd.name)) continue;
                }

                std::vector<uint8_t> buf(rd.nbytes);
                bool ok = false;

                // Read from mmap
                int si = rd.split_idx;
                if (si >= 0 && si < (int)bsctx->sharp_mmaps.size() && bsctx->sharp_mmaps[si]) {
                    const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
                    size_t file_size = bsctx->sharp_mmaps[si]->size();
                    if (rd.file_offset + rd.nbytes <= file_size) {
                        std::memcpy(buf.data(), mmap_base + rd.file_offset, rd.nbytes);
                        ok = true;
                    }
                }

                if (ok) {
                    std::lock_guard<std::mutex> plock(bsctx->prefetch_mtx);
                    if (bsctx->prefetch_cache.count(rd.name) == 0) {
                        bsctx->prefetch_cache_bytes += (int64_t)rd.nbytes;
                        bsctx->prefetch_cache[rd.name] = std::move(buf);
                    }
                }
            }
        }
    };

    // Spawn workers
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int32_t t = 0; t < n_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto & th : threads) {
        th.join();
    }
}

// ---------------------------------------------------------------------------
// Readahead: aggressively pre-populate the page cache for ALL layers.
//
// WARNING: For large models (100+ GB), calling this is counterproductive —
// the kernel blocks walking page tables and evicts the blurry model from
// the page cache.  Prefer llama_blurry_sharp_prefetch_layers_parallel()
// with specific layer indices instead.
//
// This function is kept for small models or when the caller explicitly
// wants full readahead.  For models > 32 GB it logs a warning and
// limits readahead to the first 32 GB per split to avoid thrashing.
// ---------------------------------------------------------------------------

void llama_blurry_sharp_readahead_all(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return;

    constexpr size_t MAX_READAHEAD = (size_t)32 * 1024 * 1024 * 1024ULL; // 32 GB

#ifndef _WIN32
    for (size_t si = 0; si < bsctx->sharp_mmaps.size(); ++si) {
        if (!bsctx->sharp_mmaps[si]) continue;
        void * addr = const_cast<void *>(bsctx->sharp_mmaps[si]->addr());
        size_t size = bsctx->sharp_mmaps[si]->size();
        if (!addr || size == 0) continue;

        if (size > MAX_READAHEAD) {
            LLAMA_LOG_WARN("%s: split %zu is %.1f GB — capping readahead at 32 GB "
                          "to avoid page cache thrashing.  Use "
                          "prefetch_layers_parallel() for targeted I/O.\n",
                          __func__, si, size / (1024.0 * 1024.0 * 1024.0));
            size = MAX_READAHEAD;
        }

#ifdef __linux__
        madvise(addr, size, MADV_WILLNEED);
#else
        posix_madvise(addr, size, POSIX_MADV_WILLNEED);
#endif
    }

    // Readahead ram_cache pages that may have been paged out to swap
    for (auto & kv : bsctx->ram_cache) {
        if (!kv.second.empty()) {
            madvise(kv.second.data(), kv.second.size(), MADV_WILLNEED);
        }
    }
#else
    GGML_UNUSED(bsctx);
#endif
}

// ---------------------------------------------------------------------------
// RAM cache: pre-read sharp tensor data into anonymous heap buffers
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_precache_ram(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    if (bsctx->ram_cache_populated) {
        LLAMA_LOG_INFO("%s: RAM cache already populated (%d tensors, %.2f MiB)\n",
                      __func__, (int)bsctx->ram_cache.size(),
                      bsctx->ram_cache_bytes / (1024.0 * 1024.0));
        return (int32_t)bsctx->ram_cache.size();
    }

    int64_t t_start = bs_time_us();

    int32_t n_cached = 0;
    int64_t total_bytes = 0;
    int32_t n_failed = 0;

    LLAMA_LOG_INFO("%s: pre-reading all sharp tensor data into RAM cache "
                  "(anonymous heap memory, swap-backed under pressure)...\n", __func__);

    // Iterate all sharp tensors (not just eligible ones — cache everything
    // so that any future apply_layer/apply_experts has data ready).
    for (auto & kv : bsctx->sharp_index) {
        const std::string & tensor_name = kv.first;
        const blurry_sharp_tensor_info & info = kv.second;

        // Skip if already cached
        if (bsctx->ram_cache.find(tensor_name) != bsctx->ram_cache.end()) {
            continue;
        }

        // Allocate anonymous heap buffer and read sharp data into it.
        // This creates anonymous pages that go to swap under pressure
        // (instead of being dropped like file-backed mmap pages).
        std::vector<uint8_t> buf(info.nbytes);

        int si = info.split_idx;
        bool ok = false;

        // Read from mmap if available
        if (si >= 0 && si < (int)bsctx->sharp_mmaps.size() && bsctx->sharp_mmaps[si]) {
            const uint8_t * base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
            if (info.file_offset + info.nbytes <= bsctx->sharp_mmaps[si]->size()) {
                std::memcpy(buf.data(), base + info.file_offset, info.nbytes);
                // Release the mmap pages — data is now in anonymous heap memory.
                // The mmap pages are file-backed and would just waste page cache.
                bs_release_mmap_pages(bsctx, base + info.file_offset, info.nbytes);
                ok = true;
            }
        }

        // Fallback: file read
        if (!ok && si >= 0 && si < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si]) {
            bsctx->sharp_files[si]->seek(info.file_offset, SEEK_SET);
            bsctx->sharp_files[si]->read_raw(buf.data(), info.nbytes);
            ok = true;
        }

        if (!ok) {
            ++n_failed;
            if (bsctx->params.verbose) {
                LLAMA_LOG_WARN("%s: failed to read tensor '%s' (split %d)\n",
                              __func__, tensor_name.c_str(), si);
            }
            continue;
        }

        total_bytes += (int64_t)info.nbytes;
        bsctx->ram_cache[tensor_name] = std::move(buf);
        ++n_cached;

        // Progress reporting for large models
        if (n_cached % 50 == 0) {
            LLAMA_LOG_INFO("%s: cached %d tensors (%.2f MiB)...\n",
                          __func__, n_cached, total_bytes / (1024.0 * 1024.0));
        }
    }

    bsctx->ram_cache_bytes = total_bytes;
    bsctx->ram_cache_populated = true;

    int64_t t_end = bs_time_us();

    LLAMA_LOG_INFO("%s: RAM cache populated: %d tensors, %.2f MiB in %.2f s "
                  "(%.2f MiB/s)\n",
                  __func__, n_cached,
                  total_bytes / (1024.0 * 1024.0),
                  (t_end - t_start) / 1e6,
                  (t_end > t_start) ? (total_bytes / (1024.0 * 1024.0)) / ((t_end - t_start) / 1e6) : 0.0);
    if (n_failed > 0) {
        LLAMA_LOG_WARN("%s: %d tensors failed to cache (will fall back to mmap/file)\n",
                      __func__, n_failed);
    }
    LLAMA_LOG_INFO("%s: memory tier: VRAM > RAM cache (%.2f MiB anonymous/swappable) > Swap > Disk\n",
                  __func__, total_bytes / (1024.0 * 1024.0));

    return n_cached;
}

// ---------------------------------------------------------------------------
// Swap staging: proactively move RAM-cached pages to swap
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_stage_to_swap(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;
    if (!bsctx->ram_cache_populated) {
        LLAMA_LOG_WARN("%s: RAM cache not populated — call precache_ram() first\n", __func__);
        return 0;
    }

#ifdef __linux__
    // MADV_PAGEOUT (value 21) was added in Linux 5.4.  It tells the kernel
    // to move the specified anonymous pages to swap, freeing their physical
    // frames for other use.  The pages remain valid — accessing them later
    // triggers a swap-in (fast from SSD).
    //
    // This is exactly what we want: free RAM for the blurry model / KV cache
    // while keeping sharp data quickly accessible via swap-in.
    #ifndef MADV_PAGEOUT
    #define MADV_PAGEOUT 21
    #endif

    int64_t t_start = bs_time_us();
    int32_t n_staged = 0;
    int64_t staged_bytes = 0;

    LLAMA_LOG_INFO("%s: staging %.2f MiB of RAM-cached sharp data to swap "
                  "(freeing RAM for active use)...\n",
                  __func__, bsctx->ram_cache_bytes / (1024.0 * 1024.0));

    for (auto & kv : bsctx->ram_cache) {
        if (kv.second.empty()) continue;

        int ret = madvise(kv.second.data(), kv.second.size(), MADV_PAGEOUT);
        if (ret == 0) {
            ++n_staged;
            staged_bytes += (int64_t)kv.second.size();
        }
        // MADV_PAGEOUT can fail (e.g. kernel < 5.4, no swap configured) —
        // silently ignore failures.  The data stays in RAM, which is still
        // better than file-backed mmap (RAM > Disk).
    }

    bsctx->ram_cache_staged = true;

    int64_t t_end = bs_time_us();

    if (n_staged > 0) {
        LLAMA_LOG_INFO("%s: staged %d tensors (%.2f MiB) to swap in %.2f ms\n",
                      __func__, n_staged,
                      staged_bytes / (1024.0 * 1024.0),
                      (t_end - t_start) / 1000.0);
        LLAMA_LOG_INFO("%s: RAM is now free for blurry model / KV cache; "
                      "sharp data accessible via fast swap-in\n", __func__);
    } else {
        LLAMA_LOG_WARN("%s: MADV_PAGEOUT not available or no swap configured — "
                      "sharp data stays in RAM (still better than file-backed mmap)\n",
                      __func__);
    }

    return n_staged;
#else
    LLAMA_LOG_INFO("%s: MADV_PAGEOUT not available on this platform — "
                  "sharp data stays in RAM (still better than file-backed mmap)\n",
                  __func__);
    bsctx->ram_cache_staged = true;
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Pre-allocate device buffers
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_preload_device_cache(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    int64_t t_start = bs_time_us();

    // Implicitly enable retain_device_buffers — pre-allocation only makes
    // sense if the buffers survive across restore cycles.
    bsctx->retain_device_buffers = true;

    int32_t n_preloaded = 0;
    int64_t preloaded_bytes = 0;

    for (int layer_idx : bsctx->sharp_layer_indices) {
        if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
            continue;
        }

        auto lt_it = bsctx->layer_tensor_names.find(layer_idx);
        if (lt_it == bsctx->layer_tensor_names.end()) continue;

        for (const auto & tensor_name : lt_it->second) {
            // Skip if already in the cache
            if (bsctx->device_cache.find(tensor_name) != bsctx->device_cache.end()) {
                continue;
            }

            auto info_it = bsctx->sharp_index.find(tensor_name);
            if (info_it == bsctx->sharp_index.end()) continue;
            const auto & sharp_info = info_it->second;

            // Find the base tensor (use cached pointer)
            ggml_tensor * base_t = sharp_info.base_tensor;
            if (!base_t) {
                base_t = bs_find_model_tensor(bsctx->model, tensor_name.c_str());
            }
            if (!base_t || !base_t->buffer) continue;

            // Only pre-allocate for device (non-host) tensors
            if (ggml_backend_buffer_is_host(base_t->buffer)) continue;

            ggml_backend_buffer_type_t device_buft = bs_get_device_buft(bsctx, base_t->buffer);

            // Temporarily set tensor type/strides to sharp type to compute
            // the correct padded allocation size.
            ggml_type saved_type = base_t->type;
            size_t saved_nb[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; ++d) saved_nb[d] = base_t->nb[d];

            base_t->type  = sharp_info.type;
            base_t->nb[0] = ggml_type_size(sharp_info.type);
            base_t->nb[1] = ggml_row_size(sharp_info.type, base_t->ne[0]);
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                base_t->nb[d] = base_t->nb[d - 1] * base_t->ne[d - 1];
            }
            size_t alloc_size = ggml_backend_buft_get_alloc_size(device_buft, base_t);

            // Restore original type/strides
            base_t->type = saved_type;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) base_t->nb[d] = saved_nb[d];

            if (alloc_size < sharp_info.nbytes) alloc_size = sharp_info.nbytes;

            // Respect GPU budget
            if (bsctx->params.gpu_budget_bytes > 0 &&
                (bsctx->device_cache_bytes + (int64_t)alloc_size) > bsctx->params.gpu_budget_bytes) {
                continue;
            }

            // Allocate the device buffer
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(device_buft, alloc_size);
            if (!buf) continue;

            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(buf, 0);

            // Add to cache as unpopulated — apply_layer will fill with data
            blurry_sharp_device_cache_entry entry;
            entry.buffer       = buf;
            entry.nbytes       = alloc_size;
            entry.populated    = false;
            entry.use_sequence = bsctx->cache_use_counter++;
            entry.layer_idx    = sharp_info.layer_idx;
            bsctx->device_cache[tensor_name] = entry;
            bsctx->device_cache_bytes += (int64_t)alloc_size;
            bsctx->n_cache_allocs++;

            preloaded_bytes += (int64_t)alloc_size;
            ++n_preloaded;
        }
    }

    int64_t t_end = bs_time_us();

    LLAMA_LOG_INFO("%s: pre-allocated %d device buffers (%.2f MiB) in %.2f ms\n",
                  __func__, n_preloaded,
                  preloaded_bytes / (1024.0 * 1024.0),
                  (t_end - t_start) / 1000.0);

    return n_preloaded;
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

void llama_blurry_sharp_log_index_summary(const llama_blurry_sharp_context * bsctx) {
    if (!bsctx) return;

    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("=== Blurry→Sharp Overlay Index Summary ===\n");
    LLAMA_LOG_INFO("  Sharp model splits:  %d\n",  bsctx->n_sharp_splits);
    LLAMA_LOG_INFO("  Sharp model tensors: %zu\n", bsctx->sharp_index.size());
    LLAMA_LOG_INFO("  Layer groups:        %zu\n", bsctx->layer_tensor_names.size());
    LLAMA_LOG_INFO("  Eligible layers:     %zu\n", bsctx->eligible_layers.size());

    // Compute total sharp model size
    int64_t total_sharp_bytes = 0;
    for (auto & kv : bsctx->sharp_index) {
        total_sharp_bytes += (int64_t)kv.second.nbytes;
    }
    LLAMA_LOG_INFO("  Total sharp data:    %.2f MiB\n",
                  (double)total_sharp_bytes / (1024.0 * 1024.0));

    // Show per-layer tensor count
    if (bsctx->params.verbose) {
        for (int idx : bsctx->sharp_layer_indices) {
            auto it = bsctx->layer_tensor_names.find(idx);
            if (it == bsctx->layer_tensor_names.end()) continue;
            bool eligible = bsctx->eligible_layers.find(idx) != bsctx->eligible_layers.end();
            LLAMA_LOG_INFO("    layer %3d: %2zu tensors %s\n",
                          idx, it->second.size(),
                          eligible ? "(eligible)" : "(ineligible)");
        }
    }

    LLAMA_LOG_INFO("==========================================\n\n");
}

// ---------------------------------------------------------------------------
// MoE Combination Expert API
// ---------------------------------------------------------------------------

bool llama_blurry_sharp_is_expert_tensor(const char * tensor_name) {
    if (!tensor_name) return false;
    return bs_is_expert_tensor(std::string(tensor_name));
}

int32_t llama_blurry_sharp_n_experts(
        const llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->initialized) return 0;

    // Find the first ffn_gate_exps tensor and inspect ne[2]
    for (const auto & kv : bsctx->sharp_index) {
        if (bs_is_expert_tensor(kv.first) &&
            kv.first.find("ffn_gate_exps") != std::string::npos) {
            return (int32_t)kv.second.ne[2];
        }
    }
    // Fallback: try any expert tensor
    for (const auto & kv : bsctx->sharp_index) {
        if (bs_is_expert_tensor(kv.first) && kv.second.ne[2] > 1) {
            return (int32_t)kv.second.ne[2];
        }
    }
    return 0;
}

int32_t llama_blurry_sharp_n_experts_used(
        const llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->model) return 0;
    return (int32_t)bsctx->model->hparams.n_expert_used;
}

// ---------------------------------------------------------------------------
// Expert-level overlay: create a "combination tensor" that is mostly blurry
// but with selected expert slices replaced by sharp data.
//
// For each expert tensor (ffn_gate_exps, ffn_up_exps, ffn_down_exps):
//   1. Allocate a buffer the size of the full tensor
//   2. Copy all data from the blurry tensor into it
//   3. Read only the requested expert slices from the sharp GGUF
//   4. Overwrite those slices in the combination buffer
//   5. Pointer-swap tensor->data to point at the combination buffer
//
// For non-expert tensors (attention, norms, gate_inp):
//   Falls through to the standard bs_overlay_single_tensor path.
//
// This reduces I/O by n_expert/n_experts_requested for MoE models.
// For example, in a 128-expert model with top-8 routing, this reads
// ~16x less data from the sharp GGUF per layer.
// ---------------------------------------------------------------------------

// Helper: overlay a single expert tensor with selective expert slices.
// Returns bytes read from sharp file, or negative on error.
static int64_t bs_overlay_expert_tensor(
        llama_blurry_sharp_context       * bsctx,
        const blurry_sharp_tensor_info   & sharp_info,
        ggml_tensor                      * base_tensor,
        const int32_t                    * expert_ids,
        int32_t                            n_experts_req,
        blurry_sharp_tensor_backup       & backup_out) {

    int32_t n_expert_total = (int32_t)sharp_info.ne[2];
    if (n_expert_total <= 1) {
        // Not actually a multi-expert tensor, fall back to full overlay
        return bs_overlay_single_tensor(bsctx, sharp_info, base_tensor, backup_out);
    }

    // Validate expert indices
    for (int32_t i = 0; i < n_experts_req; ++i) {
        if (expert_ids[i] < 0 || expert_ids[i] >= n_expert_total) {
            LLAMA_LOG_ERROR("%s: expert_id %d out of range [0, %d) for tensor '%s'\n",
                           __func__, expert_ids[i], n_expert_total,
                           sharp_info.name.c_str());
            return -1;
        }
    }

    // JIT SAFETY CHECK: cross-type overlays on pinned/host buffers are unsafe
    // (scheduler device copies pre-allocated at blurry size).  But plain CPU
    // buffers (from --n-cpu-moe) have no device copies — safe to overlay.
    // Flash-expert mode always uses combo_buffer swap (never in-place), so
    // the original buffer is never written to — safe regardless of buffer type.
    {
        bool tensor_is_host = true;
        if (base_tensor->buffer) {
            tensor_is_host = ggml_backend_buffer_is_host(base_tensor->buffer);
        }
        if (tensor_is_host && bsctx->jit_active && base_tensor->type != sharp_info.type
            && !bsctx->params.flash_experts) {
            bool is_plain_cpu = base_tensor->buffer &&
                ggml_backend_buffer_get_type(base_tensor->buffer) == ggml_backend_cpu_buffer_type();
            if (!is_plain_cpu) {
                bsctx->n_jit_host_crosstype_skipped++;
                if (bsctx->params.verbose || bsctx->n_jit_host_crosstype_skipped <= 3) {
                    LLAMA_LOG_WARN("%s: JIT mode: CUDA host/pinned buffer — cross-type expert overlay skipped for '%s' "
                                  "(%s -> %s, %zu -> %zu bytes, %d experts requested). "
                                  "This tensor uses BLURRY weights. "
                                  "Fix: use --n-cpu-moe to place on plain CPU (supports cross-type), "
                                  "or increase -ngl to keep on GPU.\n",
                                  __func__, sharp_info.name.c_str(),
                                  ggml_type_name(base_tensor->type),
                                  ggml_type_name(sharp_info.type),
                                  ggml_nbytes(base_tensor), sharp_info.nbytes,
                                  n_experts_req);
                }
                return -1;
            }
        }
    }

    // 1) Save original tensor metadata
    backup_out.tensor_name       = sharp_info.name;
    backup_out.base_tensor       = base_tensor;
    backup_out.original_data     = base_tensor->data;
    backup_out.original_type     = base_tensor->type;
    backup_out.original_view_src = base_tensor->view_src;
    backup_out.original_extra    = base_tensor->extra;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        backup_out.original_nb[d] = base_tensor->nb[d];
        backup_out.ne[d]          = base_tensor->ne[d];
    }

    // Compute per-expert slice sizes.
    // Expert tensors are laid out with expert as the outermost (ne[2]) dimension.
    // Each expert slice is contiguous: expert_slice_bytes = total_bytes / n_expert
    size_t sharp_expert_slice = sharp_info.nbytes / (size_t)n_expert_total;

    // For the base (blurry) tensor, compute the slice size from its type/shape
    size_t base_total_bytes = ggml_nbytes(base_tensor);
    size_t base_expert_slice = base_total_bytes / (size_t)n_expert_total;

    // 2) Determine strategy: host (CPU) or device (GPU)
    bool tensor_is_host = true;
    if (base_tensor->buffer) {
        tensor_is_host = ggml_backend_buffer_is_host(base_tensor->buffer);
    }

    if (tensor_is_host) {
        // ================================================================
        // HOST PATH — three sub-strategies:
        //
        //   A) ZERO-COPY (mmap available):
        //      Pointer-swap to the sharp mmap page.  The OS demand-pages
        //      only the expert slices the MoE kernel actually reads.
        //      Non-selected slices are never touched → never faulted in.
        //      Cost: ~nanoseconds.  Heap allocation: ZERO.
        //
        //      This is THE critical optimisation.  Without it, every expert
        //      tensor allocates a full combination buffer (e.g. 800 MiB for
        //      a [4096, 1408, 128] Q4_K tensor) on the heap.  Across 48
        //      layers × 3 expert tensors that's ~115 GiB of heap allocation
        //      per apply cycle — far more than available RAM.
        //
        //   B) SAME-TYPE, NO MMAP:
        //      Read only the selected expert slices from the sharp file
        //      directly into the existing tensor buffer at the correct
        //      expert offsets (in-place overwrite of blurry slices).
        //      Cost: n_experts_req × slice_read.  Heap allocation: ZERO.
        //      The backup saves the overwritten blurry slices so we can
        //      restore later.
        //
        //   C) DIFFERENT-TYPE, NO MMAP:
        //      Allocate a full combination buffer at the sharp type/size,
        //      write selected expert slices, pointer-swap.  This is the
        //      old (expensive) path — only used when mmap is unavailable
        //      AND the types differ.
        // ================================================================
        backup_out.is_device = false;

        // ============================================================
        // FLASH-EXPERT PATH: stream Q4_K_M experts from SSD on demand.
        //
        // No blurry data exists — expert tensors are empty placeholders.
        // Read only the K active expert slices from the sharp GGUF via
        // pread() into a shared scratch buffer.  The OS page cache
        // naturally caches hot experts (~40-50 GB/s for cache hits).
        //
        // Benefits over standard overlay:
        //  - No backup/restore of blurry data (no blurry data loaded)
        //  - All compute at Q4_K_M quality (faster dequant than TQ1_0)
        //  - Freed RAM → OS page cache for expert reads
        //  - Simpler code path: pread → swap → compute
        // ============================================================
        if (bsctx->params.flash_experts) {
            int si_f = sharp_info.split_idx;
            bool have_file = (si_f >= 0 && si_f < (int)bsctx->sharp_files.size()
                              && bsctx->sharp_files[si_f]);
            int file_fd = have_file ? bsctx->sharp_files[si_f]->file_id() : -1;

            if (file_fd < 0) {
                LLAMA_LOG_ERROR("%s: flash-experts: no file descriptor for split %d (tensor '%s')\n",
                               __func__, si_f, sharp_info.name.c_str());
                return -1;
            }

            // Use combo_buffer as scratch (per-layer cleanup keeps only current layer).
            auto & scratch = bsctx->combo_buffers[sharp_info.name];
            if (scratch.size() < sharp_info.nbytes) {
                scratch.resize(sharp_info.nbytes);
            }

            // Read active expert slices via parallel pread
            if (bsctx->params.parallel_expert_io && n_experts_req > 1) {
                std::vector<bs_pread_task> tasks(n_experts_req);
                for (int32_t i = 0; i < n_experts_req; ++i) {
                    int32_t eid = expert_ids[i];
                    tasks[i].fd     = file_fd;
                    tasks[i].dst    = scratch.data() + (size_t)eid * sharp_expert_slice;
                    tasks[i].offset = (off_t)(sharp_info.file_offset + (size_t)eid * sharp_expert_slice);
                    tasks[i].size   = sharp_expert_slice;
                    tasks[i].result = 0;
                }
                bs_parallel_pread(tasks.data(), n_experts_req,
                                  bsctx->params.cache_io_split);
            } else {
                for (int32_t i = 0; i < n_experts_req; ++i) {
                    int32_t eid = expert_ids[i];
                    size_t off = (size_t)eid * sharp_expert_slice;
                    pread(file_fd, scratch.data() + off, sharp_expert_slice,
                          (off_t)(sharp_info.file_offset + off));
                }
            }

            // Pointer-swap to scratch buffer
            base_tensor->data = scratch.data();
            backup_out.zero_copy = true;
            backup_out.flash_expert = true;

            // Update type/strides to Q4_K_M
            if (base_tensor->type != sharp_info.type) {
                base_tensor->type  = sharp_info.type;
                base_tensor->nb[0] = ggml_type_size(sharp_info.type);
                base_tensor->nb[1] = base_tensor->nb[0] * (base_tensor->ne[0] / ggml_blck_size(sharp_info.type));
                base_tensor->nb[2] = base_tensor->nb[1] * base_tensor->ne[1];
                base_tensor->nb[3] = base_tensor->nb[2] * base_tensor->ne[2];
            }
            base_tensor->view_src = nullptr;

            bsctx->metrics.n_direct_copies++;

            int64_t total_bytes = (int64_t)sharp_expert_slice * n_experts_req;
            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: flash-expert overlay '%s': %d/%d experts, "
                              "read %" PRId64 " bytes (%.1f%% of full tensor)\n",
                              __func__, sharp_info.name.c_str(),
                              n_experts_req, n_expert_total,
                              total_bytes, 100.0 * total_bytes / sharp_info.nbytes);
            }

            return total_bytes;
        }

        int si = sharp_info.split_idx;
        bool have_mmap = (si >= 0 && si < (int)bsctx->sharp_mmaps.size()
                          && bsctx->sharp_mmaps[si]);

        // Check RAM cache first — anonymous heap buffers that swap properly.
        // Under memory pressure, these pages go to swap (fast SSD) instead
        // of being dropped entirely like file-backed mmap pages.
        bool used_ram_cache = false;
        if (bsctx->ram_cache_populated) {
            auto cache_it = bsctx->ram_cache.find(sharp_info.name);
            if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= sharp_info.nbytes) {
                // Point tensor directly at the RAM cache buffer.
                // Data lives in anonymous (swappable) memory, not file-backed mmap.
                base_tensor->data = cache_it->second.data();
                backup_out.zero_copy = true;
                used_ram_cache = true;

                // Release old blurry mmap pages
                if (backup_out.original_data) {
                    size_t old_nbytes = bs_tensor_nbytes(backup_out.original_type, backup_out.ne);
                    bs_release_mmap_pages(bsctx, backup_out.original_data, old_nbytes);
                }
            }
        }

        if (!used_ram_cache && have_mmap) {
            const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si]->addr();
            if (sharp_info.file_offset + sharp_info.nbytes > bsctx->sharp_mmaps[si]->size()) {
                LLAMA_LOG_ERROR("%s: mmap read out of bounds for expert tensor '%s' (split %d)\n",
                               __func__, sharp_info.name.c_str(), si);
                return -1;
            }

            auto & rcache = bsctx->ram_expert_cache;
            const uint8_t * sharp_data = mmap_base + sharp_info.file_offset;

            // Check if ALL requested experts are in RAM cache
            int n_cached = 0;
            if (rcache.enabled) {
                for (int32_t i = 0; i < n_experts_req; ++i) {
                    uint64_t key = bs_gpu_cache_key(sharp_info.name, expert_ids[i]);
                    if (rcache.entries.count(key)) n_cached++;
                }
            }

            if (n_cached == n_experts_req && n_cached > 0) {
                // ---- Strategy A-cached: ALL experts in RAM cache ----
                // Use combo buffer populated entirely from anonymous RAM.
                // No mmap page faults — data is in swap-backed memory.
                auto & combo = bsctx->combo_buffers[sharp_info.name];
                if (combo.size() < sharp_info.nbytes) {
                    combo.resize(sharp_info.nbytes);
                }
                // NOTE: No MADV_DONTNEED — combo uses bs_aligned_buffer (heap).
                // MADV_DONTNEED on anonymous pages zeros them on Linux.
                for (int32_t i = 0; i < n_experts_req; ++i) {
                    int32_t eid = expert_ids[i];
                    size_t off = (size_t)eid * sharp_expert_slice;
                    if (off + sharp_expert_slice > sharp_info.nbytes) continue;

                    uint64_t key = bs_gpu_cache_key(sharp_info.name, eid);
                    auto it = rcache.entries.find(key);
                    std::memcpy(combo.data() + off, it->second.data.data(), sharp_expert_slice);
                    it->second.last_access = ++rcache.access_counter;
                    rcache.n_hits++;
                }

                if (backup_out.original_data) {
                    size_t old_nbytes = bs_tensor_nbytes(backup_out.original_type, backup_out.ne);
                    bs_release_mmap_pages(bsctx, backup_out.original_data, old_nbytes);
                }

                base_tensor->data = combo.data();
                backup_out.zero_copy = true;
            } else {
                // ---- Strategy A-mmap/pread: expert slice access ----
                if (backup_out.original_data) {
                    size_t old_nbytes = bs_tensor_nbytes(backup_out.original_type, backup_out.ne);
                    bs_release_mmap_pages(bsctx, backup_out.original_data, old_nbytes);
                }

                // Try async prefetch — partial match OK.  Copies matched
                // expert slices from prefetch buffer into combo_buffers.
                // Only experts NOT consumed need sync I/O below.
                bool prefetch_consumed[256] = {};  // max experts per request
                int n_prefetched = 0;
                if (n_experts_req <= 256) {
                    n_prefetched = bs_async_prefetch_consume(bsctx, sharp_info.name,
                            bsctx->building_layer_idx, expert_ids, n_experts_req,
                            prefetch_consumed);
                }

                // Ensure combo buffer is allocated (consume may have done
                // this already for partial hits, but not for zero hits).
                auto & combo = bsctx->combo_buffers[sharp_info.name];
                if (combo.size() < sharp_info.nbytes) {
                    combo.resize(sharp_info.nbytes);
                }

                if (n_prefetched == n_experts_req) {
                    // Full prefetch hit — no further I/O needed.
                    base_tensor->data = combo.data();
                    backup_out.zero_copy = true;
                } else {
                    // Partial or no prefetch hit — check RAM cache and
                    // do file I/O for remaining experts.
                    std::vector<int> file_miss_indices;
                    file_miss_indices.reserve(n_experts_req);

                    for (int32_t i = 0; i < n_experts_req; ++i) {
                        if (prefetch_consumed[i]) continue;  // already from prefetch

                        int32_t eid = expert_ids[i];
                        size_t off = (size_t)eid * sharp_expert_slice;
                        if (off + sharp_expert_slice > sharp_info.nbytes) continue;

                        // Try RAM expert cache
                        if (rcache.enabled) {
                            uint64_t key = bs_gpu_cache_key(sharp_info.name, eid);
                            auto it = rcache.entries.find(key);
                            if (it != rcache.entries.end()) {
                                std::memcpy(combo.data() + off, it->second.data.data(), sharp_expert_slice);
                                it->second.last_access = ++rcache.access_counter;
                                rcache.n_hits++;
                                continue;
                            }
                        }
                        file_miss_indices.push_back(i);
                    }

                    // Read cache misses — parallel pread or mmap fallback
                    bool have_file = (si >= 0 && si < (int)bsctx->sharp_files.size()
                                      && bsctx->sharp_files[si]);
                    int file_fd = have_file ? bsctx->sharp_files[si]->file_id() : -1;

                    if (!file_miss_indices.empty() && file_fd >= 0 &&
                        bsctx->params.parallel_expert_io && (int)file_miss_indices.size() > 1) {
                        std::vector<bs_pread_task> tasks(file_miss_indices.size());
                        for (size_t m = 0; m < file_miss_indices.size(); ++m) {
                            int32_t eid = expert_ids[file_miss_indices[m]];
                            tasks[m].fd     = file_fd;
                            tasks[m].dst    = combo.data() + (size_t)eid * sharp_expert_slice;
                            tasks[m].offset = (off_t)(sharp_info.file_offset + (size_t)eid * sharp_expert_slice);
                            tasks[m].size   = sharp_expert_slice;
                            tasks[m].result = 0;
                        }
                        bs_parallel_pread(tasks.data(), (int)tasks.size(),
                                          bsctx->params.cache_io_split);
                    } else {
                        for (int idx : file_miss_indices) {
                            int32_t eid = expert_ids[idx];
                            size_t off = (size_t)eid * sharp_expert_slice;
                            if (file_fd >= 0) {
                                pread(file_fd, combo.data() + off, sharp_expert_slice,
                                      (off_t)(sharp_info.file_offset + off));
                            } else {
                                std::memcpy(combo.data() + off,
                                            sharp_data + off, sharp_expert_slice);
                            }
                        }
                    }

                    // Populate RAM expert cache with newly-read slices
                    if (rcache.enabled && rcache.budget_bytes > 0) {
                        for (int idx : file_miss_indices) {
                            int32_t eid = expert_ids[idx];
                            size_t off = (size_t)eid * sharp_expert_slice;
                            if (off + sharp_expert_slice > sharp_info.nbytes) continue;

                            rcache.n_misses++;
                            uint64_t key = bs_gpu_cache_key(sharp_info.name, eid);

                            while (rcache.used_bytes + sharp_expert_slice > rcache.budget_bytes
                                   && !rcache.entries.empty()) {
                                uint64_t lru_key = 0;
                                int64_t lru_access = INT64_MAX;
                                for (auto & [k, e] : rcache.entries) {
                                    if (e.last_access < lru_access) {
                                        lru_access = e.last_access;
                                        lru_key = k;
                                    }
                                }
                                auto evict_it = rcache.entries.find(lru_key);
                                if (evict_it != rcache.entries.end()) {
                                    rcache.used_bytes -= evict_it->second.data.size();
                                    rcache.entries.erase(evict_it);
                                }
                            }
                            if (rcache.used_bytes + sharp_expert_slice <= rcache.budget_bytes) {
                                llama_blurry_sharp_context::ram_expert_cache::entry e;
                                e.data.assign(combo.data() + off, combo.data() + off + sharp_expert_slice);
                                e.last_access = ++rcache.access_counter;
                                rcache.used_bytes += sharp_expert_slice;
                                rcache.entries[key] = std::move(e);
                            }
                        }
                    }

                    base_tensor->data = combo.data();
                    backup_out.zero_copy = true;
                }
            }

        } else if (!used_ram_cache && base_tensor->type == sharp_info.type && base_expert_slice == sharp_expert_slice) {
            // ---- Strategy B: IN-PLACE slice overwrite (same type) ----
            //
            // Read the selected sharp expert slices directly into the
            // existing tensor buffer, overwriting the blurry data at those
            // offsets.  Save the overwritten blurry slices for restore.
            // Heap allocation: only n_experts_req small slices for backup.
            backup_out.zero_copy = false;

            // Save the blurry expert slices we're about to overwrite
            backup_out.expert_backup_ids.clear();
            backup_out.expert_backup_data.clear();
            backup_out.expert_backup_ids.reserve(n_experts_req);

            uint8_t * base_data = (uint8_t *)base_tensor->data;

            // Phase 1: back up all blurry slices and check RAM cache
            std::vector<int> file_miss_indices;  // indices into expert_ids[] needing file read
            bool have_file = (si >= 0 && si < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si]);
            int file_fd = have_file ? bsctx->sharp_files[si]->file_id() : -1;

            for (int32_t i = 0; i < n_experts_req; ++i) {
                int32_t eidx = expert_ids[i];
                size_t slice_offset = (size_t)eidx * base_expert_slice;

                // Back up the blurry slice
                backup_out.expert_backup_ids.push_back(eidx);
                size_t old_size = backup_out.expert_backup_data.size();
                backup_out.expert_backup_data.resize(old_size + base_expert_slice);
                std::memcpy(backup_out.expert_backup_data.data() + old_size,
                            base_data + slice_offset, base_expert_slice);

                // Try RAM cache first
                bool slice_ok = false;
                if (bsctx->ram_cache_populated) {
                    auto cache_it = bsctx->ram_cache.find(sharp_info.name);
                    if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= sharp_info.nbytes) {
                        size_t slice_cache_offset = (size_t)eidx * sharp_expert_slice;
                        std::memcpy(base_data + slice_offset,
                                    cache_it->second.data() + slice_cache_offset,
                                    sharp_expert_slice);
                        slice_ok = true;
                    }
                }

                if (!slice_ok) {
                    file_miss_indices.push_back(i);
                }
            }

            // Phase 2: parallel pread for all cache misses
            if (!file_miss_indices.empty()) {
                if (file_fd < 0) {
                    LLAMA_LOG_ERROR("%s: no data source for split %d (tensor '%s', %d cache misses)\n",
                                   __func__, si, sharp_info.name.c_str(), (int)file_miss_indices.size());
                    // Restore all overwritten slices
                    size_t restore_off = 0;
                    for (int32_t j = 0; j < n_experts_req; ++j) {
                        size_t roff = (size_t)backup_out.expert_backup_ids[j] * base_expert_slice;
                        std::memcpy(base_data + roff,
                                    backup_out.expert_backup_data.data() + restore_off,
                                    base_expert_slice);
                        restore_off += base_expert_slice;
                    }
                    return -1;
                }

                if (bsctx->params.parallel_expert_io && (int)file_miss_indices.size() > 1) {
                    // Parallel pread: read all cache-miss expert slices simultaneously
                    std::vector<bs_pread_task> tasks(file_miss_indices.size());
                    for (size_t m = 0; m < file_miss_indices.size(); ++m) {
                        int idx = file_miss_indices[m];
                        int32_t eidx = expert_ids[idx];
                        tasks[m].fd     = file_fd;
                        tasks[m].dst    = base_data + (size_t)eidx * base_expert_slice;
                        tasks[m].offset = (off_t)(sharp_info.file_offset + (size_t)eidx * sharp_expert_slice);
                        tasks[m].size   = sharp_expert_slice;
                        tasks[m].result = 0;
                    }
                    bs_parallel_pread(tasks.data(), (int)tasks.size(),
                                      bsctx->params.cache_io_split);

                    // Validate results
                    for (size_t m = 0; m < tasks.size(); ++m) {
                        if (tasks[m].result != (ssize_t)sharp_expert_slice) {
                            LLAMA_LOG_ERROR("%s: pread failed for expert %d of tensor '%s': %zd/%zu\n",
                                           __func__, expert_ids[file_miss_indices[m]],
                                           sharp_info.name.c_str(), tasks[m].result, sharp_expert_slice);
                            // Restore all overwritten slices
                            size_t restore_off = 0;
                            for (int32_t j = 0; j < n_experts_req; ++j) {
                                size_t roff = (size_t)backup_out.expert_backup_ids[j] * base_expert_slice;
                                std::memcpy(base_data + roff,
                                            backup_out.expert_backup_data.data() + restore_off,
                                            base_expert_slice);
                                restore_off += base_expert_slice;
                            }
                            return -1;
                        }
                    }
                } else {
                    // Sequential fallback (single miss or parallel disabled)
                    for (int idx : file_miss_indices) {
                        int32_t eidx = expert_ids[idx];
                        size_t slice_offset = (size_t)eidx * base_expert_slice;
                        size_t slice_file_offset = sharp_info.file_offset + (size_t)eidx * sharp_expert_slice;
                        bsctx->sharp_files[si]->seek(slice_file_offset, SEEK_SET);
                        bsctx->sharp_files[si]->read_raw(base_data + slice_offset, sharp_expert_slice);
                    }
                }
            }

            // Type and strides are unchanged (same type) — no update needed.
            // base_tensor->data already points at the (now patched) buffer.

            bsctx->metrics.n_direct_copies++;

            int64_t total_sharp_bytes_read = (int64_t)sharp_expert_slice * n_experts_req;
            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: host in-place expert overlay '%s': %d/%d experts, "
                              "read %" PRId64 " bytes, backup %" PRId64 " bytes\n",
                              __func__, sharp_info.name.c_str(),
                              n_experts_req, n_expert_total,
                              total_sharp_bytes_read,
                              (int64_t)backup_out.expert_backup_data.size());
            }

            return total_sharp_bytes_read;

        } else if (!used_ram_cache) {
            // ---- Strategy C: FULL COMBINATION BUFFER (different types, no mmap) ----
            // This is the expensive fallback — allocate sharp_info.nbytes on heap.
            backup_out.zero_copy = false;
            backup_out.sharp_data.resize(sharp_info.nbytes);
            std::memset(backup_out.sharp_data.data(), 0, sharp_info.nbytes);

            int64_t total_sharp_bytes_read = 0;
            for (int32_t i = 0; i < n_experts_req; ++i) {
                int32_t eidx = expert_ids[i];
                size_t slice_file_offset = sharp_info.file_offset + (size_t)eidx * sharp_expert_slice;
                size_t slice_buf_offset  = (size_t)eidx * sharp_expert_slice;

                bool slice_ok = false;

                // RAM cache: read the expert slice from the cached buffer
                if (bsctx->ram_cache_populated) {
                    auto cache_it = bsctx->ram_cache.find(sharp_info.name);
                    if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= sharp_info.nbytes) {
                        std::memcpy(backup_out.sharp_data.data() + slice_buf_offset,
                                    cache_it->second.data() + slice_buf_offset,
                                    sharp_expert_slice);
                        slice_ok = true;
                    }
                }

                // Fallback: file read
                if (!slice_ok) {
                    if (si >= 0 && si < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si]) {
                        bsctx->sharp_files[si]->seek(slice_file_offset, SEEK_SET);
                        bsctx->sharp_files[si]->read_raw(
                            backup_out.sharp_data.data() + slice_buf_offset,
                            sharp_expert_slice);
                        slice_ok = true;
                    }
                }

                if (!slice_ok) {
                    LLAMA_LOG_ERROR("%s: no data source for split %d (tensor '%s' expert %d)\n",
                                   __func__, si, sharp_info.name.c_str(), eidx);
                    backup_out.sharp_data.clear();
                    return -1;
                }
                total_sharp_bytes_read += (int64_t)sharp_expert_slice;
            }

            base_tensor->data = backup_out.sharp_data.data();
            // Fall through to update type/strides below
        }

        // Update type and byte strides to match the sharp tensor so that
        // the kernel uses the correct dequantization and stride arithmetic
        // when indexing into the selected expert slices.
        if (base_tensor->type != sharp_info.type) {
            base_tensor->type = sharp_info.type;
            base_tensor->nb[0] = ggml_type_size(sharp_info.type);
            base_tensor->nb[1] = base_tensor->nb[0] * (base_tensor->ne[0] / ggml_blck_size(sharp_info.type));
            base_tensor->nb[2] = base_tensor->nb[1] * base_tensor->ne[1];
            base_tensor->nb[3] = base_tensor->nb[2] * base_tensor->ne[2];
        }

        // Clear view_src for correct dispatch.
        base_tensor->view_src = nullptr;

        bsctx->metrics.n_direct_copies++;

        int64_t total_sharp_bytes_read = (int64_t)sharp_expert_slice * n_experts_req;
        if (bsctx->params.verbose) {
            const char * strat = used_ram_cache ? "ram-cache" :
                                 backup_out.zero_copy ? "zero-copy-mmap" : "combination-buf";
            LLAMA_LOG_INFO("%s: host expert overlay '%s' (%s): %d/%d experts, "
                          "read %" PRId64 " bytes (%.1f%% of full tensor %zu bytes)\n",
                          __func__, sharp_info.name.c_str(), strat,
                          n_experts_req, n_expert_total,
                          total_sharp_bytes_read,
                          100.0 * total_sharp_bytes_read / sharp_info.nbytes,
                          sharp_info.nbytes);
        }

        return total_sharp_bytes_read;

    } else {
        // ================================================================
        // DEVICE PATH: combination expert on GPU
        //
        // Three sub-strategies:
        //
        //   A) SAME TYPE + SAME SIZE: write selected expert slices directly
        //      into the existing device buffer at the correct offsets via
        //      ggml_backend_tensor_set with an offset.  ZERO extra VRAM.
        //      Back up overwritten slices for restore.
        //
        //   B) DIFFERENT TYPE: allocate new device buffer at sharp size,
        //      upload only selected expert slices (zeroed elsewhere).
        //      This is the expensive path but unavoidable for cross-type.
        //
        //   C) CACHE HIT: reuse a previously allocated device buffer from
        //      the cache, re-upload only the needed expert slices.
        // ================================================================
        backup_out.is_device = true;
        backup_out.zero_copy = false;

        ggml_backend_buffer_type_t device_buft = bs_get_device_buft(bsctx, base_tensor->buffer);

        bool same_type = (base_tensor->type == sharp_info.type);
        bool same_size = (base_total_bytes == sharp_info.nbytes);

        if (same_type && same_size) {
            // ---- Strategy A: IN-PLACE expert slice overwrite (zero extra VRAM) ----
            //
            // Write selected sharp expert slices directly into the existing
            // device buffer at the correct byte offsets.  Back up the old
            // device data for restore.
            //
            // This eliminates the enormous full-tensor device buffer allocation
            // that was the primary cause of VRAM exhaustion in MoE models.

            // Save blurry expert slices from device for later restore
            backup_out.expert_backup_ids.clear();
            backup_out.expert_backup_data.clear();
            backup_out.expert_backup_ids.reserve(n_experts_req);
            backup_out.original_buffer = base_tensor->buffer;

            int64_t total_sharp_bytes_read = 0;

            for (int32_t i = 0; i < n_experts_req; ++i) {
                int32_t eidx = expert_ids[i];
                size_t slice_offset_device = (size_t)eidx * base_expert_slice;
                size_t slice_file_offset   = sharp_info.file_offset + (size_t)eidx * sharp_expert_slice;

                // Back up the blurry expert slice FROM device
                backup_out.expert_backup_ids.push_back(eidx);
                size_t old_size = backup_out.expert_backup_data.size();
                backup_out.expert_backup_data.resize(old_size + base_expert_slice);
                ggml_backend_tensor_get(base_tensor,
                    backup_out.expert_backup_data.data() + old_size,
                    slice_offset_device, base_expert_slice);

                // Read sharp expert slice into staging (reuse read_buf since
                // staging may be too small for full tensors but is fine for
                // individual expert slices)
                int ssi = sharp_info.split_idx;
                bool have_mmap = (ssi >= 0 && ssi < (int)bsctx->sharp_mmaps.size()
                                  && bsctx->sharp_mmaps[ssi]);

                const void * src_data = nullptr;

                // Fastest path: RAM cache → pinned staging (swap-backed anonymous pages)
                if (!src_data && bsctx->ram_cache_populated) {
                    auto cache_it = bsctx->ram_cache.find(sharp_info.name);
                    if (cache_it != bsctx->ram_cache.end() && cache_it->second.size() >= sharp_info.nbytes) {
                        size_t slice_cache_offset = (size_t)eidx * sharp_expert_slice;
                        if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                            memcpy(bsctx->pinned_staging_ptr,
                                   cache_it->second.data() + slice_cache_offset,
                                   sharp_expert_slice);
                            src_data = bsctx->pinned_staging_ptr;
                        } else {
                            src_data = cache_it->second.data() + slice_cache_offset;
                        }
                    }
                }

                if (!src_data && have_mmap) {
                    const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[ssi]->addr();
                    if (slice_file_offset + sharp_expert_slice <= bsctx->sharp_mmaps[ssi]->size()) {
                        // For pinned staging: copy mmap → pinned if staging is big enough
                        if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                            memcpy(bsctx->pinned_staging_ptr, mmap_base + slice_file_offset, sharp_expert_slice);
                            src_data = bsctx->pinned_staging_ptr;
                        } else {
                            // Use mmap pointer directly (pageable copy — slower but works)
                            src_data = mmap_base + slice_file_offset;
                        }
                        bs_release_mmap_pages(bsctx, mmap_base + slice_file_offset, sharp_expert_slice);
                    } else {
                        LLAMA_LOG_ERROR("%s: mmap OOB for expert %d of device tensor '%s'\n",
                                       __func__, eidx, sharp_info.name.c_str());
                        // Restore already-written slices
                        size_t ro = 0;
                        for (int32_t j = 0; j < i; ++j) {
                            size_t roff = (size_t)backup_out.expert_backup_ids[j] * base_expert_slice;
                            ggml_backend_tensor_set(base_tensor,
                                backup_out.expert_backup_data.data() + ro,
                                roff, base_expert_slice);
                            ro += base_expert_slice;
                        }
                        return -1;
                    }
                }

                if (!src_data && ssi >= 0 && ssi < (int)bsctx->sharp_files.size() && bsctx->sharp_files[ssi]) {
                    bsctx->read_buf.resize(sharp_expert_slice);
                    bsctx->sharp_files[ssi]->seek(slice_file_offset, SEEK_SET);
                    bsctx->sharp_files[ssi]->read_raw(bsctx->read_buf.data(), sharp_expert_slice);
                    if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                        memcpy(bsctx->pinned_staging_ptr, bsctx->read_buf.data(), sharp_expert_slice);
                        src_data = bsctx->pinned_staging_ptr;
                    } else {
                        src_data = bsctx->read_buf.data();
                    }
                }

                if (!src_data) {
                    LLAMA_LOG_ERROR("%s: no data source for split %d (device tensor '%s' expert %d)\n",
                                   __func__, ssi, sharp_info.name.c_str(), eidx);
                    return -1;
                }

                // Write the sharp expert slice INTO the existing device buffer
                ggml_backend_tensor_set(base_tensor, src_data, slice_offset_device, sharp_expert_slice);
                total_sharp_bytes_read += (int64_t)sharp_expert_slice;
            }

            // Type and strides unchanged (same type) — tensor is still valid
            // with the same layout.  Just mark the backup so restore knows
            // to write back the saved expert slices.
            backup_out.device_sharp_buffer = nullptr;  // no new allocation
            backup_out.device_buf_cached   = false;

            bsctx->metrics.n_direct_copies++;

            if (bsctx->params.verbose) {
                LLAMA_LOG_INFO("%s: device in-place expert overlay '%s': %d/%d experts, "
                              "read %" PRId64 " bytes, zero extra VRAM\n",
                              __func__, sharp_info.name.c_str(),
                              n_experts_req, n_expert_total,
                              total_sharp_bytes_read);
            }

            return total_sharp_bytes_read;

        } else {
            // ---- Strategy B: NEW DEVICE BUFFER (different type/size) ----
            //
            // Allocate a device buffer at the sharp tensor size, upload
            // only the selected expert slices, pointer-swap.
            //
            // This allocates sharp_info.nbytes of extra VRAM per expert
            // tensor.  It's the only option when types differ.

            // Check device buffer cache first
            if (bsctx->retain_device_buffers) {
                auto cache_it = bsctx->device_cache.find(sharp_info.name);
                if (cache_it != bsctx->device_cache.end() &&
                    cache_it->second.buffer &&
                    cache_it->second.nbytes >= sharp_info.nbytes) {
                    // Cache hit — reuse the buffer.  Re-upload expert slices
                    // (we can't know which experts were cached from a prior call).
                    cache_it->second.use_sequence = bsctx->cache_use_counter++;

                    backup_out.original_buffer     = base_tensor->buffer;
                    backup_out.device_sharp_buffer = cache_it->second.buffer;
                    backup_out.device_buf_cached   = true;

                    void * cached_base = ggml_backend_buffer_get_base(cache_it->second.buffer);
                    base_tensor->data   = cached_base;
                    base_tensor->buffer = cache_it->second.buffer;
                    base_tensor->type   = sharp_info.type;
                    base_tensor->nb[0]  = ggml_type_size(sharp_info.type);
                    base_tensor->nb[1]  = base_tensor->nb[0] * (base_tensor->ne[0] / ggml_blck_size(sharp_info.type));
                    base_tensor->nb[2]  = base_tensor->nb[1] * base_tensor->ne[1];
                    base_tensor->nb[3]  = base_tensor->nb[2] * base_tensor->ne[2];
                    base_tensor->view_src = nullptr;
                    base_tensor->extra    = nullptr;

                    // Zero the buffer, then upload only selected expert slices
                    ggml_backend_buffer_clear(cache_it->second.buffer, 0);

                    int si_idx = sharp_info.split_idx;
                    int64_t total_sharp_bytes_read = 0;
                    for (int32_t i = 0; i < n_experts_req; ++i) {
                        int32_t eidx = expert_ids[i];
                        size_t slice_file_offset = sharp_info.file_offset + (size_t)eidx * sharp_expert_slice;
                        size_t slice_buf_offset  = (size_t)eidx * sharp_expert_slice;

                        bool have_mm = (si_idx >= 0 && si_idx < (int)bsctx->sharp_mmaps.size()
                                        && bsctx->sharp_mmaps[si_idx]);
                        const void * src_data = nullptr;

                        // RAM cache: fastest path (swap-backed anonymous pages)
                        if (!src_data && bsctx->ram_cache_populated) {
                            auto rc_it = bsctx->ram_cache.find(sharp_info.name);
                            if (rc_it != bsctx->ram_cache.end() && rc_it->second.size() >= sharp_info.nbytes) {
                                size_t slice_cache_offset = (size_t)eidx * sharp_expert_slice;
                                if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                                    memcpy(bsctx->pinned_staging_ptr,
                                           rc_it->second.data() + slice_cache_offset,
                                           sharp_expert_slice);
                                    src_data = bsctx->pinned_staging_ptr;
                                } else {
                                    src_data = rc_it->second.data() + slice_cache_offset;
                                }
                            }
                        }

                        if (!src_data && have_mm) {
                            const uint8_t * mm_base = (const uint8_t *)bsctx->sharp_mmaps[si_idx]->addr();
                            if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                                memcpy(bsctx->pinned_staging_ptr, mm_base + slice_file_offset, sharp_expert_slice);
                                src_data = bsctx->pinned_staging_ptr;
                            } else {
                                src_data = mm_base + slice_file_offset;
                            }
                            bs_release_mmap_pages(bsctx, mm_base + slice_file_offset, sharp_expert_slice);
                        }

                        if (!src_data && si_idx >= 0 && si_idx < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si_idx]) {
                            bsctx->read_buf.resize(sharp_expert_slice);
                            bsctx->sharp_files[si_idx]->seek(slice_file_offset, SEEK_SET);
                            bsctx->sharp_files[si_idx]->read_raw(bsctx->read_buf.data(), sharp_expert_slice);
                            src_data = bsctx->read_buf.data();
                        }
                        if (src_data) {
                            ggml_backend_tensor_set(base_tensor, src_data, slice_buf_offset, sharp_expert_slice);
                            total_sharp_bytes_read += (int64_t)sharp_expert_slice;
                        }
                    }

                    bsctx->n_cache_hits++;
                    bsctx->metrics.n_direct_copies++;
                    return total_sharp_bytes_read;
                }
            }

            // No cache hit — check GPU budget (with LRU eviction)
            ggml_type saved_type = base_tensor->type;
            size_t saved_nb[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; ++d) saved_nb[d] = base_tensor->nb[d];

            // Set type/strides to sharp for correct alloc_size computation
            base_tensor->type  = sharp_info.type;
            base_tensor->nb[0] = ggml_type_size(sharp_info.type);
            base_tensor->nb[1] = base_tensor->nb[0] * (base_tensor->ne[0] / ggml_blck_size(sharp_info.type));
            base_tensor->nb[2] = base_tensor->nb[1] * base_tensor->ne[1];
            base_tensor->nb[3] = base_tensor->nb[2] * base_tensor->ne[2];

            size_t alloc_size = ggml_backend_buft_get_alloc_size(device_buft, base_tensor);
            if (alloc_size < sharp_info.nbytes) alloc_size = sharp_info.nbytes;

            // Restore type/strides temporarily
            base_tensor->type = saved_type;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) base_tensor->nb[d] = saved_nb[d];

            int64_t effective_device_bytes = bsctx->retain_device_buffers
                ? bsctx->device_cache_bytes
                : bsctx->current_device_sharp_bytes;

            if (bsctx->params.gpu_budget_bytes > 0 &&
                (effective_device_bytes + (int64_t)alloc_size) > bsctx->params.gpu_budget_bytes) {

                // Try LRU eviction if retain_device_buffers is on
                if (bsctx->retain_device_buffers && !bsctx->device_cache.empty()) {
                    int64_t needed = (effective_device_bytes + (int64_t)alloc_size)
                                   - bsctx->params.gpu_budget_bytes;
                    int64_t freed = 0;
                    bool synced = false;
                    while (freed < needed && !bsctx->device_cache.empty()) {
                        auto lru_it = bsctx->device_cache.end();
                        uint64_t min_seq = UINT64_MAX;
                        for (auto it = bsctx->device_cache.begin(); it != bsctx->device_cache.end(); ++it) {
                            if (it->second.buffer && it->second.use_sequence < min_seq) {
                                bool in_use = false;
                                if (it->second.layer_idx >= 0) {
                                    // Guard against self-eviction during layer construction
                                    if (it->second.layer_idx == bsctx->building_layer_idx) {
                                        in_use = true;
                                    }
                                    auto lb_it = bsctx->layer_backups.find(it->second.layer_idx);
                                    if (lb_it != bsctx->layer_backups.end() && lb_it->second.is_sharpened) {
                                        in_use = true;
                                    }
                                }
                                if (!in_use) {
                                    min_seq = it->second.use_sequence;
                                    lru_it = it;
                                }
                            }
                        }
                        if (lru_it == bsctx->device_cache.end()) break;
                        // Sync device before first free to prevent async
                        // CUDA kernels from reading freed memory.
                        if (!synced) {
                            bs_sync_device_before_eviction(bsctx);
                            synced = true;
                        }
                        freed += (int64_t)lru_it->second.nbytes;
                        bsctx->device_cache_bytes -= (int64_t)lru_it->second.nbytes;
                        ggml_backend_buffer_free(lru_it->second.buffer);
                        bsctx->device_cache.erase(lru_it);
                        bsctx->n_cache_evictions++;
                    }
                    effective_device_bytes = bsctx->device_cache_bytes;
                }

                // Re-check after eviction
                bool budget_exceeded = bsctx->params.gpu_budget_bytes > 0 &&
                    (effective_device_bytes + (int64_t)alloc_size) > bsctx->params.gpu_budget_bytes;
                if (budget_exceeded && bsctx->params.verbose) {
                    LLAMA_LOG_WARN("%s: GPU budget exceeded for device expert tensor '%s' "
                                  "(need %zu, budget %" PRId64 ", used %" PRId64 "), "
                                  "will try pinned host fallback\n",
                                  __func__, sharp_info.name.c_str(),
                                  alloc_size, bsctx->params.gpu_budget_bytes,
                                  effective_device_bytes);
                }
                if (budget_exceeded) goto pinned_expert_fallback;
            }

            {
                ggml_backend_buffer_t sharp_buf = ggml_backend_buft_alloc_buffer(device_buft, alloc_size);
                if (!sharp_buf) {
                    // OOM on device — try LRU eviction then retry once
                    if (bsctx->retain_device_buffers && !bsctx->device_cache.empty()) {
                        int64_t needed = (int64_t)alloc_size;
                        int64_t freed = 0;
                        bool synced = false;
                        while (freed < needed && !bsctx->device_cache.empty()) {
                            auto lru_it = bsctx->device_cache.end();
                            uint64_t min_seq = UINT64_MAX;
                            for (auto it = bsctx->device_cache.begin(); it != bsctx->device_cache.end(); ++it) {
                                if (it->second.buffer && it->second.use_sequence < min_seq) {
                                    bool in_use = false;
                                    if (it->second.layer_idx >= 0) {
                                        if (it->second.layer_idx == bsctx->building_layer_idx) in_use = true;
                                        auto lb_it = bsctx->layer_backups.find(it->second.layer_idx);
                                        if (lb_it != bsctx->layer_backups.end() && lb_it->second.is_sharpened) in_use = true;
                                    }
                                    if (!in_use) {
                                        min_seq = it->second.use_sequence;
                                        lru_it = it;
                                    }
                                }
                            }
                            if (lru_it == bsctx->device_cache.end()) break;
                            if (!synced) { bs_sync_device_before_eviction(bsctx); synced = true; }
                            freed += (int64_t)lru_it->second.nbytes;
                            bsctx->device_cache_bytes -= (int64_t)lru_it->second.nbytes;
                            ggml_backend_buffer_free(lru_it->second.buffer);
                            bsctx->device_cache.erase(lru_it);
                            bsctx->n_cache_evictions++;
                        }
                        // Retry allocation after eviction
                        sharp_buf = ggml_backend_buft_alloc_buffer(device_buft, alloc_size);
                    }
                    if (!sharp_buf) {
                        LLAMA_LOG_WARN("%s: VRAM exhausted for expert tensor '%s' (%zu bytes), "
                                      "falling back to pinned host memory (sharp quality, PCIe speed)\n",
                                      __func__, sharp_info.name.c_str(), alloc_size);
                        goto pinned_expert_fallback;
                    }
                }

                ggml_backend_buffer_set_usage(sharp_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                ggml_backend_buffer_clear(sharp_buf, 0);

                // Pointer-swap to new device buffer
                backup_out.original_buffer     = base_tensor->buffer;
                backup_out.device_sharp_buffer = sharp_buf;

                void * sharp_base = ggml_backend_buffer_get_base(sharp_buf);
                base_tensor->data     = sharp_base;
                base_tensor->buffer   = sharp_buf;
                base_tensor->type     = sharp_info.type;
                base_tensor->nb[0]    = ggml_type_size(sharp_info.type);
                base_tensor->nb[1]    = base_tensor->nb[0] * (base_tensor->ne[0] / ggml_blck_size(sharp_info.type));
                base_tensor->nb[2]    = base_tensor->nb[1] * base_tensor->ne[1];
                base_tensor->nb[3]    = base_tensor->nb[2] * base_tensor->ne[2];
                base_tensor->view_src = nullptr;
                base_tensor->extra    = nullptr;

                // Upload only the selected expert slices (not the entire tensor)
                int si_idx = sharp_info.split_idx;
                int64_t total_sharp_bytes_read = 0;

                for (int32_t i = 0; i < n_experts_req; ++i) {
                    int32_t eidx = expert_ids[i];
                    size_t slice_file_offset = sharp_info.file_offset + (size_t)eidx * sharp_expert_slice;
                    size_t slice_buf_offset  = (size_t)eidx * sharp_expert_slice;

                    bool have_mmap = (si_idx >= 0 && si_idx < (int)bsctx->sharp_mmaps.size()
                                      && bsctx->sharp_mmaps[si_idx]);
                    const void * src_data = nullptr;

                    // RAM cache: fastest path (swap-backed anonymous pages)
                    if (bsctx->ram_cache_populated) {
                        auto rc_it = bsctx->ram_cache.find(sharp_info.name);
                        if (rc_it != bsctx->ram_cache.end() && rc_it->second.size() >= sharp_info.nbytes) {
                            size_t slice_cache_offset = (size_t)eidx * sharp_expert_slice;
                            if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                                memcpy(bsctx->pinned_staging_ptr,
                                       rc_it->second.data() + slice_cache_offset,
                                       sharp_expert_slice);
                                src_data = bsctx->pinned_staging_ptr;
                            } else {
                                src_data = rc_it->second.data() + slice_cache_offset;
                            }
                        }
                    }

                    if (!src_data && have_mmap) {
                        const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si_idx]->addr();
                        if (slice_file_offset + sharp_expert_slice <= bsctx->sharp_mmaps[si_idx]->size()) {
                            if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                                memcpy(bsctx->pinned_staging_ptr, mmap_base + slice_file_offset, sharp_expert_slice);
                                src_data = bsctx->pinned_staging_ptr;
                            } else {
                                src_data = mmap_base + slice_file_offset;
                            }
                            bs_release_mmap_pages(bsctx, mmap_base + slice_file_offset, sharp_expert_slice);
                        } else {
                            LLAMA_LOG_ERROR("%s: mmap OOB for expert %d of device tensor '%s'\n",
                                           __func__, eidx, sharp_info.name.c_str());
                            ggml_backend_buffer_free(sharp_buf);
                            return -1;
                        }
                    }

                    if (!src_data && si_idx >= 0 && si_idx < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si_idx]) {
                        bsctx->read_buf.resize(sharp_expert_slice);
                        bsctx->sharp_files[si_idx]->seek(slice_file_offset, SEEK_SET);
                        bsctx->sharp_files[si_idx]->read_raw(bsctx->read_buf.data(), sharp_expert_slice);
                        if (bsctx->pinned_staging_buf && sharp_expert_slice <= bsctx->pinned_staging_size) {
                            memcpy(bsctx->pinned_staging_ptr, bsctx->read_buf.data(), sharp_expert_slice);
                            src_data = bsctx->pinned_staging_ptr;
                        } else {
                            src_data = bsctx->read_buf.data();
                        }
                    }

                    if (!src_data) {
                        LLAMA_LOG_ERROR("%s: no data source for split %d (device tensor '%s' expert %d)\n",
                                       __func__, si_idx, sharp_info.name.c_str(), eidx);
                        ggml_backend_buffer_free(sharp_buf);
                        return -1;
                    }

                    if (src_data) {
                        ggml_backend_tensor_set(base_tensor, src_data, slice_buf_offset, sharp_expert_slice);
                        total_sharp_bytes_read += (int64_t)sharp_expert_slice;
                    }
                }

                // Track GPU memory
                bsctx->current_device_sharp_bytes += (int64_t)alloc_size;

                if (bsctx->retain_device_buffers) {
                    blurry_sharp_device_cache_entry entry;
                    entry.buffer       = sharp_buf;
                    entry.nbytes       = alloc_size;
                    entry.populated    = true;
                    entry.use_sequence = bsctx->cache_use_counter++;
                    entry.layer_idx    = sharp_info.layer_idx;
                    bsctx->device_cache[sharp_info.name] = entry;
                    bsctx->device_cache_bytes += (int64_t)alloc_size;
                    bsctx->n_cache_allocs++;
                    backup_out.device_buf_cached = true;
                }

                bsctx->metrics.n_direct_copies++;

                if (bsctx->params.verbose) {
                    LLAMA_LOG_INFO("%s: device new-buffer expert '%s': %d/%d experts, "
                                  "read %" PRId64 " bytes, alloc %zu bytes VRAM\n",
                                  __func__, sharp_info.name.c_str(),
                                  n_experts_req, n_expert_total,
                                  total_sharp_bytes_read, alloc_size);
                }

                return total_sharp_bytes_read;
            }

            // ---- PINNED HOST FALLBACK ----
            // When VRAM is exhausted, allocate sharp data in CUDA pinned host
            // memory.  Pinned memory is in RAM but directly accessible from GPU
            // kernels via unified virtual addressing (UVA / zero-copy).  The GPU
            // reads over PCIe — slower than VRAM but preserves sharp quality
            // instead of falling back to blurry.
            pinned_expert_fallback:
            {
                ggml_backend_buffer_type_t pinned_buft = llama_default_buffer_type_cpu(true);
                ggml_backend_buffer_t pinned_buf = nullptr;

                // Try CUDA pinned memory first (GPU-accessible via UVA)
                if (pinned_buft && pinned_buft != ggml_backend_cpu_buffer_type()) {
                    pinned_buf = ggml_backend_buft_alloc_buffer(pinned_buft, sharp_info.nbytes);
                }

                // Fallback: plain CPU buffer (may not be GPU-accessible, but
                // the backend scheduler will insert a copy if needed)
                if (!pinned_buf) {
                    pinned_buf = ggml_backend_buft_alloc_buffer(
                        ggml_backend_cpu_buffer_type(), sharp_info.nbytes);
                }

                if (!pinned_buf) {
                    LLAMA_LOG_ERROR("%s: failed to allocate pinned host buffer (%zu bytes) "
                                   "for expert tensor '%s' — falling back to BLURRY weights\n",
                                   __func__, sharp_info.nbytes, sharp_info.name.c_str());
                    bsctx->n_device_tensors_skipped++;
                    return -1;
                }

                ggml_backend_buffer_set_usage(pinned_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                void * pinned_base = ggml_backend_buffer_get_base(pinned_buf);
                memset(pinned_base, 0, sharp_info.nbytes);

                backup_out.original_buffer     = base_tensor->buffer;
                backup_out.device_sharp_buffer = pinned_buf;

                base_tensor->data     = pinned_base;
                base_tensor->buffer   = pinned_buf;
                base_tensor->type     = sharp_info.type;
                base_tensor->nb[0]    = ggml_type_size(sharp_info.type);
                base_tensor->nb[1]    = base_tensor->nb[0] * (base_tensor->ne[0] / ggml_blck_size(sharp_info.type));
                base_tensor->nb[2]    = base_tensor->nb[1] * base_tensor->ne[1];
                base_tensor->nb[3]    = base_tensor->nb[2] * base_tensor->ne[2];
                base_tensor->view_src = nullptr;
                base_tensor->extra    = nullptr;

                // Read expert slices directly into pinned memory (memcpy, no GPU upload needed)
                int si_idx = sharp_info.split_idx;
                int64_t total_sharp_bytes_read = 0;

                for (int32_t i = 0; i < n_experts_req; ++i) {
                    int32_t eidx = expert_ids[i];
                    size_t slice_file_offset = sharp_info.file_offset + (size_t)eidx * sharp_expert_slice;
                    size_t slice_buf_offset  = (size_t)eidx * sharp_expert_slice;
                    uint8_t * dst = (uint8_t *)pinned_base + slice_buf_offset;

                    bool have_mmap = (si_idx >= 0 && si_idx < (int)bsctx->sharp_mmaps.size()
                                      && bsctx->sharp_mmaps[si_idx]);
                    bool slice_ok = false;

                    // RAM cache
                    if (bsctx->ram_cache_populated) {
                        auto rc_it = bsctx->ram_cache.find(sharp_info.name);
                        if (rc_it != bsctx->ram_cache.end() && rc_it->second.size() >= sharp_info.nbytes) {
                            size_t slice_cache_offset = (size_t)eidx * sharp_expert_slice;
                            memcpy(dst, rc_it->second.data() + slice_cache_offset, sharp_expert_slice);
                            slice_ok = true;
                        }
                    }

                    // mmap
                    if (!slice_ok && have_mmap) {
                        const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[si_idx]->addr();
                        if (slice_file_offset + sharp_expert_slice <= bsctx->sharp_mmaps[si_idx]->size()) {
                            memcpy(dst, mmap_base + slice_file_offset, sharp_expert_slice);
                            bs_release_mmap_pages(bsctx, mmap_base + slice_file_offset, sharp_expert_slice);
                            slice_ok = true;
                        }
                    }

                    // File read
                    if (!slice_ok && si_idx >= 0 && si_idx < (int)bsctx->sharp_files.size() && bsctx->sharp_files[si_idx]) {
                        bsctx->sharp_files[si_idx]->seek(slice_file_offset, SEEK_SET);
                        bsctx->sharp_files[si_idx]->read_raw(dst, sharp_expert_slice);
                        slice_ok = true;
                    }

                    if (!slice_ok) {
                        LLAMA_LOG_ERROR("%s: no data source for split %d (pinned fallback tensor '%s' expert %d)\n",
                                       __func__, si_idx, sharp_info.name.c_str(), eidx);
                        ggml_backend_buffer_free(pinned_buf);
                        return -1;
                    }

                    total_sharp_bytes_read += (int64_t)sharp_expert_slice;
                }

                // Do NOT add to device cache or track as device bytes — this is host memory
                bsctx->metrics.n_direct_copies++;

                LLAMA_LOG_INFO("%s: pinned host fallback expert '%s': %d/%d experts, "
                              "read %" PRId64 " bytes (VRAM exhausted, using PCIe zero-copy)\n",
                              __func__, sharp_info.name.c_str(),
                              n_experts_req, n_expert_total,
                              total_sharp_bytes_read);

                return total_sharp_bytes_read;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Apply sharp weights for specific experts within a layer.
// Non-expert tensors get full overlay; expert tensors get selective overlay.
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_apply_experts(
        llama_blurry_sharp_context * bsctx,
        int32_t                      layer_idx,
        const int32_t              * expert_ids,
        int32_t                      n_experts_req) {
    if (!bsctx || !bsctx->initialized) return -1;
    if (!expert_ids || n_experts_req <= 0) return -1;
    std::lock_guard<std::mutex> lock(bsctx->mtx);

    // Check eligibility
    if (bsctx->eligible_layers.find(layer_idx) == bsctx->eligible_layers.end()) {
        static int n_elig_log = 0;
        if (n_elig_log++ < 10) {
            fprintf(stderr, "apply_experts: layer %d NOT ELIGIBLE\n", layer_idx);
        }
        return -1;
    }

    // Check if already sharpened (full layer overlay takes precedence)
    {
        auto it = bsctx->layer_backups.find(layer_idx);
        if (it != bsctx->layer_backups.end() && it->second.is_sharpened) {
            it->second.apply_sequence = bsctx->apply_sequence_counter++;
            it->second.apply_timestamp_us = bs_time_us();
            return 0;
        }
    }

    // Check max_sharp_layers limit
    if (bsctx->params.max_sharp_layers > 0) {
        int32_t n_currently_sharp = 0;
        for (auto & kv : bsctx->layer_backups) {
            if (kv.second.is_sharpened) n_currently_sharp++;
        }
        if (n_currently_sharp >= bsctx->params.max_sharp_layers) {
            int64_t budget = bsctx->current_backup_bytes;
            llama_blurry_sharp_evict_to_budget(bsctx, budget > 0 ? budget - 1 : 0);
            n_currently_sharp = 0;
            for (auto & kv : bsctx->layer_backups) {
                if (kv.second.is_sharpened) n_currently_sharp++;
            }
            if (n_currently_sharp >= bsctx->params.max_sharp_layers) {
                LLAMA_LOG_WARN("%s: max_sharp_layers=%d reached\n",
                              __func__, bsctx->params.max_sharp_layers);
                return -3;
            }
        }
    }

    // Get tensor names for this layer
    auto layer_it = bsctx->layer_tensor_names.find(layer_idx);
    if (layer_it == bsctx->layer_tensor_names.end() || layer_it->second.empty()) {
        LLAMA_LOG_WARN("%s: no sharp tensors for layer %d\n", __func__, layer_idx);
        return -2;
    }

    // Check if any tensor in this layer is actually an expert tensor.
    // If not, fall back to standard full-layer overlay.
    bool has_expert_tensors = false;
    for (const auto & tensor_name : layer_it->second) {
        if (bs_is_expert_tensor(tensor_name)) {
            has_expert_tensors = true;
            break;
        }
    }
    if (!has_expert_tensors) {
        if (bsctx->params.verbose) {
            LLAMA_LOG_INFO("%s: layer %d has no expert tensors, falling back to full layer overlay\n",
                          __func__, layer_idx);
        }
        // Unlock and call apply_layer (which takes its own lock)
        // Can't call apply_layer directly since we hold the lock.
        // Instead, inline the same logic but without re-locking.
        // For simplicity, just proceed with full overlay below.
    }

    int64_t t_start = bs_time_us();

    blurry_sharp_layer_backup layer_backup;
    layer_backup.layer_idx          = layer_idx;
    layer_backup.is_sharpened       = false;
    layer_backup.backup_bytes       = 0;
    layer_backup.sharp_bytes_read   = 0;
    layer_backup.apply_timestamp_us = t_start;
    layer_backup.apply_sequence     = bsctx->apply_sequence_counter++;

    bool any_hard_failed = false;   // real errors (I/O, alloc, OOM)
    int  n_jit_skipped = 0;         // JIT cross-type host skips (expected, non-fatal)
    int  n_expert_tensors_overlaid = 0;
    int  n_nonexpert_tensors_overlaid = 0;

    // Mark this layer as "being built" so the LRU eviction code inside
    // bs_overlay_single_tensor / bs_overlay_expert_tensor does NOT evict
    // cache entries that were just allocated for earlier tensors of THIS
    // layer.  Without this guard, overlaying tensor B could evict tensor
    // A's freshly-cached buffer, leaving tensor A's base_tensor->data
    // pointing at freed GPU memory → illegal memory access when the
    // compute kernel runs.
    bsctx->building_layer_idx = layer_idx;

    // Prefetch only the expert slices we need, not the whole layer.
    // For MoE models, each expert tensor (ffn_gate_exps, ffn_up_exps,
    // ffn_down_exps) stores all N experts contiguously (e.g. 160 × 6 MiB).
    // Issuing MADV_WILLNEED on the full tensor reads ~1 GiB per tensor,
    // but we only need 8/160 = 5%.  Targeted prefetch reads ~50 MiB total
    // instead of ~3 GiB — a 60× reduction in I/O.
#ifndef _WIN32
    for (const auto & tensor_name : layer_it->second) {
        // RAM cache swap-in: always prefetch the full cached buffer
        // (it's already in swap, not on disk)
        if (bsctx->lazy_swap_enabled || bsctx->ram_cache_populated) {
            auto cache_it = bsctx->ram_cache.find(tensor_name);
            if (cache_it != bsctx->ram_cache.end() && !cache_it->second.empty()) {
                if (has_expert_tensors && bs_is_expert_tensor(tensor_name)) {
                    // Only prefetch the needed expert slices from ram_cache
                    auto info_it = bsctx->sharp_index.find(tensor_name);
                    if (info_it != bsctx->sharp_index.end() && info_it->second.ne[2] > 1) {
                        size_t slice_bytes = cache_it->second.size() / (size_t)info_it->second.ne[2];
                        for (int32_t ei = 0; ei < n_experts_req; ++ei) {
                            size_t off = (size_t)expert_ids[ei] * slice_bytes;
                            if (off + slice_bytes <= cache_it->second.size()) {
                                madvise(cache_it->second.data() + off, slice_bytes, MADV_WILLNEED);
                            }
                        }
                    } else {
                        madvise(cache_it->second.data(), cache_it->second.size(), MADV_WILLNEED);
                    }
                } else {
                    // Non-expert tensors (attention, norms): prefetch fully
                    madvise(cache_it->second.data(), cache_it->second.size(), MADV_WILLNEED);
                }
                continue;
            }
        }

        // mmap page prefetch: only touch expert slices we need
        auto info_it = bsctx->sharp_index.find(tensor_name);
        if (info_it == bsctx->sharp_index.end()) continue;
        const auto & si = info_it->second;

        int split = si.split_idx;
        if (split < 0 || split >= (int)bsctx->sharp_mmaps.size()) continue;
        if (!bsctx->sharp_mmaps[split]) continue;

        const uint8_t * mmap_base = (const uint8_t *)bsctx->sharp_mmaps[split]->addr();
        size_t file_size = bsctx->sharp_mmaps[split]->size();

        if (has_expert_tensors && bs_is_expert_tensor(tensor_name) && si.ne[2] > 1) {
            // Targeted expert-slice prefetch: only MADV_WILLNEED the slices
            // for the requested experts, leaving the other 152 experts on disk.
            size_t slice_bytes = si.nbytes / (size_t)si.ne[2];
            for (int32_t ei = 0; ei < n_experts_req; ++ei) {
                size_t slice_off = si.file_offset + (size_t)expert_ids[ei] * slice_bytes;
                size_t end = slice_off + slice_bytes;
                if (end > file_size) end = file_size;
                if (slice_off < end) {
                    posix_madvise((void *)(mmap_base + slice_off), end - slice_off,
                                  POSIX_MADV_WILLNEED);
                }
            }
        } else {
            // Non-expert tensors: prefetch fully (they're small)
            size_t end = si.file_offset + si.nbytes;
            if (end > file_size) end = file_size;
            if (si.file_offset < end) {
                posix_madvise((void *)(mmap_base + si.file_offset), end - si.file_offset,
                              POSIX_MADV_WILLNEED);
            }
        }
    }
#endif

    // Sort tensor names by (split_idx, file_offset) for sequential disk I/O.
    std::vector<std::string> sorted_tensors(layer_it->second.begin(), layer_it->second.end());
    std::sort(sorted_tensors.begin(), sorted_tensors.end(), [&](const std::string & a, const std::string & b) {
        auto ia = bsctx->sharp_index.find(a);
        auto ib = bsctx->sharp_index.find(b);
        if (ia == bsctx->sharp_index.end()) return false;
        if (ib == bsctx->sharp_index.end()) return true;
        if (ia->second.split_idx != ib->second.split_idx) return ia->second.split_idx < ib->second.split_idx;
        return ia->second.file_offset < ib->second.file_offset;
    });

    for (const auto & tensor_name : sorted_tensors) {
        auto info_it = bsctx->sharp_index.find(tensor_name);
        if (info_it == bsctx->sharp_index.end()) continue;

        const auto & sharp_info = info_it->second;

        ggml_tensor * base_tensor = sharp_info.base_tensor;
        if (!base_tensor) {
            base_tensor = bs_find_model_tensor(bsctx->model, tensor_name.c_str());
        }
        if (!base_tensor) {
            LLAMA_LOG_WARN("%s: base tensor '%s' not found, skipping\n",
                          __func__, tensor_name.c_str());
            continue;
        }

        // In JIT mode, check if this tensor would be a cross-type host
        // overlay BEFORE doing any work.  Cross-type overlays change the
        // tensor type; the scheduler's pre-allocated device copies have the
        // original type/size.
        //
        // Safe cases:
        //   - Expert tensors on any CPU-computed buffer (plain CPU or CUDA
        //     host/pinned from --n-cpu-moe): no device copies for MoE tensors,
        //     the CPU backend reads directly from tensor->data.
        //   - Any tensor where sharp nbytes == blurry nbytes: the device
        //     copy fits, tensor_copy re-types dst in place.
        //
        // Unsafe cases (skip):
        //   - Non-plain-CPU host buffers (pinned): have device copies at
        //     blurry size, any type change risks overflow.
        //   - Plain CPU non-expert tensors where sharp nbytes > blurry
        //     nbytes: the scheduler may copy them to GPU, and the device
        //     copy is too small for the larger sharp data.
        if (bsctx->jit_active && base_tensor->type != sharp_info.type) {
            bool tih = base_tensor->buffer && ggml_backend_buffer_is_host(base_tensor->buffer);
            if (tih) {
                bool is_plain_cpu = base_tensor->buffer &&
                    ggml_backend_buffer_get_type(base_tensor->buffer) == ggml_backend_cpu_buffer_type();
                if (!is_plain_cpu) {
                    n_jit_skipped++;
                    continue;  // pinned buffer — skip to avoid device copy overflow
                }
                // Plain CPU: expert tensors are safe (no device copies).
                // Non-expert tensors (shared experts, norms) may be copied
                // to GPU by the scheduler.  Only safe if nbytes match.
                if (!bs_is_expert_tensor(tensor_name)) {
                    size_t blurry_nbytes = ggml_nbytes(base_tensor);
                    size_t sharp_nbytes  = sharp_info.nbytes;
                    if (sharp_nbytes != blurry_nbytes) {
                        n_jit_skipped++;
                        continue;  // device copy too small for larger sharp type
                    }
                }
            }
        }

        // Check memory budget
        // For expert tensors, the cost is the full tensor (we need the
        // combination buffer), but the I/O is only the expert slices.
        int64_t cost = (int64_t)sharp_info.nbytes;
        if (bsctx->params.memory_budget_bytes > 0 &&
            (bsctx->current_backup_bytes + cost) > bsctx->params.memory_budget_bytes) {
            int64_t target = bsctx->current_backup_bytes
                           - ((bsctx->current_backup_bytes + cost) - bsctx->params.memory_budget_bytes);
            if (target < 0) target = 0;
            llama_blurry_sharp_evict_to_budget(bsctx, target);
            if ((bsctx->current_backup_bytes + cost) > bsctx->params.memory_budget_bytes) {
                LLAMA_LOG_WARN("%s: memory budget exceeded for tensor '%s'\n",
                              __func__, tensor_name.c_str());
                any_hard_failed = true;
                continue;
            }
        }

        blurry_sharp_tensor_backup tb;
        int64_t sharp_bytes;

        if (has_expert_tensors && bs_is_expert_tensor(tensor_name)) {
            // Expert tensor: use selective expert overlay
            sharp_bytes = bs_overlay_expert_tensor(
                bsctx, sharp_info, base_tensor,
                expert_ids, n_experts_req, tb);
            if (sharp_bytes >= 0) n_expert_tensors_overlaid++;
        } else {
            // Non-expert tensor (attention, norms, gate_inp): full overlay
            sharp_bytes = bs_overlay_single_tensor(bsctx, sharp_info, base_tensor, tb);
            if (sharp_bytes >= 0) n_nonexpert_tensors_overlaid++;
        }

        if (sharp_bytes < 0) {
            LLAMA_LOG_ERROR("%s: failed to overlay tensor '%s'\n",
                           __func__, tensor_name.c_str());
            any_hard_failed = true;
            continue;
        }

        layer_backup.backup_bytes     += sharp_bytes;
        layer_backup.sharp_bytes_read += sharp_bytes;
        layer_backup.tensor_backups.push_back(std::move(tb));

        bsctx->current_backup_bytes += sharp_bytes;
        bsctx->metrics.total_sharp_bytes_read += sharp_bytes;
        bsctx->metrics.total_backup_bytes_written += sharp_bytes;
    }

    if (layer_backup.tensor_backups.empty()) {
        bsctx->building_layer_idx = -1;
        int rc = (any_hard_failed || n_jit_skipped > 0) ? -4 : -2;
        static int n_empty_log = 0;
        if (n_empty_log++ < 10) {
            fprintf(stderr, "apply_experts: layer %d EMPTY BACKUPS rc=%d "
                    "(hard_fail=%d, jit_skip=%d, expert_ok=%d, nonexp_ok=%d)\n",
                    layer_idx, rc, any_hard_failed, n_jit_skipped,
                    n_expert_tensors_overlaid, n_nonexpert_tensors_overlaid);
        }
        return rc;
    }

    // Roll back on hard failure (I/O error, allocation failure, etc.)
    // JIT cross-type skips are NOT hard failures — they're expected when
    // expert tensors are on the CPU with a different quant type.  In that
    // case we keep the successful overlays (e.g. attention/norm tensors on
    // GPU) and run with a partially sharpened layer.
    if (any_hard_failed) {
        LLAMA_LOG_WARN("%s: layer %d had failures — rolling back %d overlays\n",
                      __func__, layer_idx,
                      (int)layer_backup.tensor_backups.size());

        for (auto & tb : layer_backup.tensor_backups) {
            if (tb.is_device && tb.device_buf_cached) {
                auto cache_it = bsctx->device_cache.find(tb.tensor_name);
                if (cache_it != bsctx->device_cache.end()) {
                    if (cache_it->second.buffer) {
                        bsctx->device_cache_bytes -= (int64_t)cache_it->second.nbytes;
                        ggml_backend_buffer_free(cache_it->second.buffer);
                    }
                    bsctx->device_cache.erase(cache_it);
                }
                // Null out the pointer so bs_restore_single_tensor does NOT
                // double-free the same buffer we just freed via the cache.
                tb.device_sharp_buffer = nullptr;
                tb.device_buf_cached = false;
            }
            bs_restore_single_tensor(bsctx, tb);
        }
        bsctx->current_backup_bytes -= layer_backup.backup_bytes;
        bsctx->building_layer_idx = -1;
        return -4;
    }

    layer_backup.is_sharpened = true;
    bsctx->layer_backups[layer_idx] = std::move(layer_backup);
    bsctx->building_layer_idx = -1;

    int64_t t_end = bs_time_us();
    bsctx->metrics.n_apply_calls++;
    bsctx->metrics.total_time_apply_us += (t_end - t_start);

    if (bsctx->params.verbose || n_jit_skipped > 0) {
        LLAMA_LOG_INFO("%s: layer %d combination-expert sharpened "
                      "(%d expert tensors with %d/%d experts + %d non-expert tensors",
                      __func__, layer_idx,
                      n_expert_tensors_overlaid, n_experts_req,
                      llama_blurry_sharp_n_experts(bsctx),
                      n_nonexpert_tensors_overlaid);
        if (n_jit_skipped > 0) {
            LLAMA_LOG_INFO(", %d cross-type host tensors skipped (JIT mode)", n_jit_skipped);
        }
        LLAMA_LOG_INFO(", %" PRId64 " bytes read, %" PRId64 " us)\n",
                      bsctx->layer_backups[layer_idx].sharp_bytes_read,
                      t_end - t_start);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// GPU Expert Cache
//
// Persistent GPU-side cache of sharp (Q3_K_M) expert slices.  Cache hits
// use fast GPU→GPU copy (~10μs per 5MB slice on a 3090).  Cache misses
// read from SSD/page-cache → host → GPU cache + device copy.
//
// Between tokens, expert routing has strong temporal locality (70-95% of
// experts repeat), so most slices are already cached → dramatically reduces
// PCIe and SSD I/O per token.
// ---------------------------------------------------------------------------

static uint64_t bs_gpu_cache_key(const std::string & tensor_name, int expert_id) {
    // Simple hash combining tensor name and expert ID
    uint64_t h = 14695981039346656037ULL; // FNV-1a offset basis
    for (char c : tensor_name) {
        h ^= (uint64_t)(unsigned char)c;
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)expert_id;
    h *= 1099511628211ULL;
    return h;
}

void llama_blurry_sharp_set_gpu_cache_buffer(
        llama_blurry_sharp_context * bsctx,
        ggml_backend_buffer_t        buf,
        ggml_backend_t               backend) {
    if (!bsctx || !buf) return;
    auto & cache = bsctx->gpu_cache;
    if (cache.enabled) return;

    cache.cache_buf   = buf;
    cache.gpu_base    = ggml_backend_buffer_get_base(buf);
    cache.total_bytes = ggml_backend_buffer_get_size(buf);
    cache.used_bytes  = 0;
    cache.gpu_backend = backend;  // may be null; set during JIT if needed
    cache.enabled     = true;

    cache.free_list.clear();
    cache.free_list.push_back({0, cache.total_bytes});

    LLAMA_LOG_INFO("%s: GPU expert cache set from pre-allocated buffer: %.1f MiB\n",
                   __func__,
                   cache.total_bytes / (1024.0 * 1024.0));
}

void llama_blurry_sharp_gpu_cache_init(
        llama_blurry_sharp_context * bsctx,
        ggml_backend_t               gpu_backend) {
    if (!bsctx || !gpu_backend) return;
    auto & cache = bsctx->gpu_cache;
    if (cache.enabled) return;  // already initialized

    int64_t cache_bytes = bsctx->params.gpu_cache_bytes;
    if (cache_bytes <= 0) return;  // disabled

    // Allocate GPU buffer
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(gpu_backend);
    cache.cache_buf = ggml_backend_buft_alloc_buffer(buft, (size_t)cache_bytes);
    if (!cache.cache_buf) {
        LLAMA_LOG_WARN("%s: failed to allocate GPU expert cache (%lld MiB)\n",
                       __func__, (long long)(cache_bytes / (1024 * 1024)));
        return;
    }

    cache.gpu_base    = ggml_backend_buffer_get_base(cache.cache_buf);
    cache.total_bytes = (size_t)cache_bytes;
    cache.used_bytes  = 0;
    cache.gpu_backend = gpu_backend;
    cache.enabled     = true;

    // Initialize free list with one big block
    cache.free_list.clear();
    cache.free_list.push_back({0, cache.total_bytes});

    LLAMA_LOG_INFO("%s: GPU expert cache initialized: %.1f MiB on %s\n",
                   __func__,
                   cache.total_bytes / (1024.0 * 1024.0),
                   ggml_backend_name(gpu_backend));
}

// Allocate space in the cache, evicting LRU entries if needed.
// Returns byte offset into cache buffer, or -1 on failure.
static int64_t bs_gpu_cache_alloc(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        size_t size) {
    // Alignment to 512 bytes for GPU efficiency
    size = (size + 511) & ~(size_t)511;

    // First-fit search
    for (size_t i = 0; i < cache.free_list.size(); ++i) {
        auto & blk = cache.free_list[i];
        if (blk.size >= size) {
            size_t offset = blk.offset;
            if (blk.size == size) {
                cache.free_list.erase(cache.free_list.begin() + i);
            } else {
                blk.offset += size;
                blk.size   -= size;
            }
            cache.used_bytes += size;
            return (int64_t)offset;
        }
    }

    // No space — evict LRU entries until we have enough
    while (cache.used_bytes + size > cache.total_bytes && !cache.entries.empty()) {
        // Find LRU entry
        uint64_t lru_key = 0;
        int64_t  lru_access = INT64_MAX;
        for (auto & [key, entry] : cache.entries) {
            if (entry.last_access < lru_access) {
                lru_access = entry.last_access;
                lru_key    = key;
            }
        }
        auto it = cache.entries.find(lru_key);
        if (it == cache.entries.end()) break;

        // Free the entry
        size_t freed_offset = it->second.offset;
        size_t freed_size   = (it->second.size + 511) & ~(size_t)511;
        cache.used_bytes -= freed_size;
        cache.entries.erase(it);

        // Add to free list and coalesce
        cache.free_list.push_back({freed_offset, freed_size});
        // Simple coalescing: sort by offset and merge adjacent
        std::sort(cache.free_list.begin(), cache.free_list.end(),
            [](const auto & a, const auto & b) { return a.offset < b.offset; });
        for (size_t j = 0; j + 1 < cache.free_list.size(); ) {
            auto & a = cache.free_list[j];
            auto & b = cache.free_list[j + 1];
            if (a.offset + a.size == b.offset) {
                a.size += b.size;
                cache.free_list.erase(cache.free_list.begin() + j + 1);
            } else {
                ++j;
            }
        }
    }

    // Try first-fit again after eviction
    for (size_t i = 0; i < cache.free_list.size(); ++i) {
        auto & blk = cache.free_list[i];
        if (blk.size >= size) {
            size_t offset = blk.offset;
            if (blk.size == size) {
                cache.free_list.erase(cache.free_list.begin() + i);
            } else {
                blk.offset += size;
                blk.size   -= size;
            }
            cache.used_bytes += size;
            return (int64_t)offset;
        }
    }

    return -1;  // still can't fit
}

// Look up an expert slice in the GPU cache.
// On hit: updates LRU, returns cache_offset.  On miss: returns -1.
int64_t bs_gpu_cache_lookup(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        const std::string & tensor_name,
        int expert_id) {
    uint64_t key = bs_gpu_cache_key(tensor_name, expert_id);
    auto it = cache.entries.find(key);
    if (it == cache.entries.end()) return -1;
    it->second.last_access = ++cache.access_counter;
    return (int64_t)it->second.offset;
}

// Store an expert slice in the GPU cache (host_data → GPU cache buffer).
// Returns cache offset on success, -1 on failure.
int64_t bs_gpu_cache_store(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        const std::string & tensor_name,
        int expert_id,
        const void * host_data,
        size_t size) {
    uint64_t key = bs_gpu_cache_key(tensor_name, expert_id);

    // Check if already stored (update access time)
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
        it->second.last_access = ++cache.access_counter;
        return (int64_t)it->second.offset;
    }

    // Allocate space (may evict LRU entries)
    int64_t offset = bs_gpu_cache_alloc(cache, size);
    if (offset < 0) return -1;

    // Upload host data to GPU cache buffer using a temporary tensor descriptor
    {
        ggml_tensor tmp = {};
        tmp.type  = GGML_TYPE_I8;
        tmp.ne[0] = (int64_t)size;
        tmp.ne[1] = tmp.ne[2] = tmp.ne[3] = 1;
        tmp.nb[0] = 1;
        tmp.nb[1] = (int64_t)size;
        tmp.nb[2] = (int64_t)size;
        tmp.nb[3] = (int64_t)size;
        tmp.data   = (uint8_t *)cache.gpu_base + offset;
        tmp.buffer = cache.cache_buf;

        ggml_backend_tensor_set(&tmp, host_data, 0, size);
    }

    // Record entry
    cache.entries[key] = {(size_t)offset, size, ++cache.access_counter};
    return offset;
}

// Copy an expert slice from GPU cache to a device copy tensor.
// Uses GPU→GPU copy (ggml_backend_buffer_copy_tensor) — ~10μs per 5MB.
void bs_gpu_cache_copy_to_dcpy(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        int64_t cache_offset,
        size_t  size,
        ggml_tensor * dcpy,
        size_t  dcpy_offset) {
    // Create temporary tensor descriptors for GPU→GPU copy
    ggml_tensor src = {};
    src.type  = GGML_TYPE_I8;
    src.ne[0] = (int64_t)size;
    src.ne[1] = src.ne[2] = src.ne[3] = 1;
    src.nb[0] = 1;
    src.nb[1] = (int64_t)size;
    src.nb[2] = (int64_t)size;
    src.nb[3] = (int64_t)size;
    src.data   = (uint8_t *)cache.gpu_base + cache_offset;
    src.buffer = cache.cache_buf;

    ggml_tensor dst = {};
    dst.type  = GGML_TYPE_I8;
    dst.ne[0] = (int64_t)size;
    dst.ne[1] = dst.ne[2] = dst.ne[3] = 1;
    dst.nb[0] = 1;
    dst.nb[1] = (int64_t)size;
    dst.nb[2] = (int64_t)size;
    dst.nb[3] = (int64_t)size;
    dst.data   = (uint8_t *)dcpy->data + dcpy_offset;
    dst.buffer = dcpy->buffer;

    // ggml_backend_tensor_copy handles same-backend GPU→GPU via
    // cudaMemcpyDeviceToDevice, and falls back to host staging otherwise.
    ggml_backend_tensor_copy(&src, &dst);
}

// ---------------------------------------------------------------------------
// Expert tensor type inflation/deflation for scheduler device copy sizing
//
// The backend scheduler creates device copies of host tensors at graph-alloc
// time, sized according to the tensor's current type.  For cross-type JIT
// overlays (e.g. TQ1_0 → Q3_K_M), the sharp data is larger than the blurry
// tensor.  By temporarily setting expert tensor types to the sharp type
// BEFORE alloc, the scheduler allocates device copies large enough to hold
// sharp data.  We restore the original types immediately after alloc.
// ---------------------------------------------------------------------------

void llama_blurry_sharp_inflate_expert_types(
        llama_blurry_sharp_context * bsctx,
        int32_t n_tokens,
        const int32_t * priority_layers,
        int32_t         n_priority_layers) {
    if (!bsctx || !bsctx->model) return;

    bsctx->inflate_saved_types.clear();

    const int64_t n_expert_used = bsctx->model->hparams.n_expert_used;

    // Always inflate expert tensor types to sharp (Q4_K_M) so the scheduler
    // allocates device copies large enough for sharp data.  The scheduler
    // reuses device copy buffers across layers, so the extra VRAM cost is
    // just ONE layer of (Q4_K_M - TQ1_0) ≈ 700MB — fits on 24GB GPU.
    //
    // For prompt: only_active_experts copies sharp slices CPU→GPU automatically.
    // For generation: the JIT callback uploads active slices manually.
    bsctx->inflate_shrunk_ne2 = true;

    // Build set of priority layer indices.  When non-empty, only inflate
    // tensors belonging to these layers.  Non-priority layers keep their
    // original TQ1_0/ne[2]=n_expert type — the scheduler creates normal
    // device copies for them, and only_active_experts handles the copy.
    std::unordered_set<int> prio_set;
    if (priority_layers && n_priority_layers > 0) {
        for (int32_t i = 0; i < n_priority_layers; ++i) {
            prio_set.insert(priority_layers[i]);
        }
    }

    for (auto & [name, info] : bsctx->sharp_index) {
        if (!bs_is_expert_tensor(name)) continue;

        // If priority set is active, only inflate tensors in priority layers.
        // Extract layer index from tensor name "blk.{N}.ffn_..."
        if (!prio_set.empty()) {
            int layer_idx = -1;
            if (name.compare(0, 4, "blk.") == 0) {
                layer_idx = atoi(name.c_str() + 4);
            }
            if (layer_idx < 0 || prio_set.find(layer_idx) == prio_set.end()) {
                continue;  // not a priority layer — skip inflation
            }
        }

        ggml_tensor * t = info.base_tensor;
        if (!t) {
            t = bs_find_model_tensor(bsctx->model, name.c_str());
            if (!t) continue;
            info.base_tensor = t;
        }

        // Only inflate if the sharp type is different (and larger)
        if (t->type == info.type) continue;

        // Save original type and strides for deflation
        llama_blurry_sharp_context::inflate_saved_entry entry;
        entry.tensor    = t;
        entry.orig_type = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            entry.orig_nb[d] = t->nb[d];
        }
        bsctx->inflate_saved_types.push_back(entry);

        // Set tensor to the sharp type so scheduler sizes device copies for it.
        // ne[2] stays at the original n_expert — shrinking it to n_expert_used
        // would break fused MoE kernels that use ne[2] as the expert count.
        // The scheduler efficiently shares buffer space between layers.
        t->type  = info.type;
        t->nb[0] = ggml_type_size(info.type);
        t->nb[1] = ggml_row_size(info.type, t->ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
        }
    }
}

void llama_blurry_sharp_deflate_expert_types(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx) return;

    for (auto & saved : bsctx->inflate_saved_types) {
        saved.tensor->type = saved.orig_type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            saved.tensor->nb[d] = saved.orig_nb[d];
        }
    }
    bsctx->inflate_saved_types.clear();
}

// ---------------------------------------------------------------------------
// Ray march PIM: wire delta correction data to model expert tensors
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_wire_delta_tensors(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->fractal_mode || bsctx->correction_levels.empty()) {
        return 0;
    }

    // Use the first correction level's delta index
    auto & level = bsctx->correction_levels[0];
    if (level.delta_index.empty() || level.mmaps.empty()) {
        return 0;
    }

    int32_t n_wired = 0;

    // For each tensor in the delta index, find the corresponding base model tensor
    // and set its ray_march_delta_data pointer to the mmap'd delta data.
    for (auto & [name, dinfo] : level.delta_index) {
        if (!dinfo.base_tensor) {
            // Try to find the base tensor in the model
            if (bsctx->model) {
                ggml_tensor * bt = llama_get_model_tensor(bsctx->model, name.c_str());
                if (bt) {
                    dinfo.base_tensor = bt;
                }
            }
        }

        if (dinfo.base_tensor && dinfo.split_idx < (int)level.mmaps.size() && level.mmaps[dinfo.split_idx]) {
            // Skip if already wired (double-init protection)
            if (dinfo.base_tensor->ray_march_delta_cache != NULL) {
                n_wired++;
                continue;
            }

            // Skip non-weight tensors: norms, scales, biases (1D or scalar).
            // Delta correction only works for MUL_MAT weight tensors (2D+).
            // Wiring delta to norm/scale tensors confuses code that checks
            // ray_march_delta_cache on non-matmul ops → NaN.
            if (dinfo.base_tensor->ne[1] <= 1) {
                continue;
            }

            const uint8_t * mmap_base = (const uint8_t *)level.mmaps[dinfo.split_idx]->addr();
            const void * mmap_ptr = (const void *)(mmap_base + dinfo.file_offset);
            int fd = level.files[dinfo.split_idx]->file_id();
            int n_experts = (int)dinfo.base_tensor->ne[2];
            size_t expert_bytes = ggml_row_size(dinfo.type, dinfo.base_tensor->ne[0]) * dinfo.base_tensor->ne[1];

            struct pim_delta_cache * cache = pim_delta_cache_create(
                mmap_ptr, fd, (int64_t)dinfo.file_offset, expert_bytes, n_experts);

            dinfo.base_tensor->ray_march_delta_cache = (void *)cache;
            dinfo.base_tensor->ray_march_delta_type  = dinfo.type;
            n_wired++;
        }
    }

    if (n_wired > 0) {
        LLAMA_LOG_INFO("%s: wired %d expert tensors with delta correction data for ray march PIM\n",
                       __func__, n_wired);
    }

    // Pin delta mmap for async DMA (swing buffer prefetch).
    // One large cudaHostRegister on the entire mmap — required for true async cudaMemcpyAsync.
#ifdef GGML_USE_CUDA
    if (n_wired > 0 && !level.mmaps.empty() && level.mmaps[0]) {
        const uint8_t * mmap_base = (const uint8_t *)level.mmaps[0]->addr();
        size_t mmap_size = level.mmaps[0]->size();
        if (mmap_base && mmap_size > 0) {
            madvise((void *)mmap_base, mmap_size, MADV_WILLNEED);
            if (ggml_backend_cuda_pin_host_memory((void *)mmap_base, mmap_size)) {
                LLAMA_LOG_INFO("%s: pinned %.1f MiB delta mmap for async DMA prefetch\n",
                               __func__, mmap_size / (1024.0 * 1024.0));
            } else {
                LLAMA_LOG_WARN("%s: failed to pin delta mmap — swing buffer will use sync DMA\n", __func__);
            }
        }
    }
#endif

    // Per-tensor VRAM upload REMOVED — ring buffer handles GPU delta.

    // CPU JIT Pre-Merge: build queue of CPU tensors for background merge.
    // The worker thread is started later (after first gen token populates the queue).
    {
        auto & jm = bsctx->jit_merge;
        for (auto & [name, dinfo] : level.delta_index) {
            if (!dinfo.base_tensor || !dinfo.base_tensor->ray_march_delta_cache) continue;
            if (!dinfo.base_tensor->buffer || !ggml_backend_buffer_is_host(dinfo.base_tensor->buffer)) continue;
            if (dinfo.base_tensor->ne[1] <= 1) continue;

            ggml_tensor * base = dinfo.base_tensor;
            struct pim_delta_cache * cache = (struct pim_delta_cache *)base->ray_march_delta_cache;
            const char * delta_host = pim_delta_cache_get(cache, 0);
            if (!delta_host) continue;

            int64_t ne0 = base->ne[0];
            int64_t n_elements = 1;
            for (int d = 0; d < GGML_MAX_DIMS; d++) if (base->ne[d] > 0) n_elements *= base->ne[d];
            int64_t n_rows = n_elements / ne0;

            jm.queue.push_back({
                base, delta_host, base->type, dinfo.type,
                ne0, n_rows,
                ggml_row_size(base->type, ne0),
                ggml_row_size(dinfo.type, ne0),
                ggml_row_size(GGML_TYPE_Q8_0, ne0),
                ggml_row_size(GGML_TYPE_Q8_0, ne0) * (size_t)n_rows,
                base->data
            });
        }

        if (!jm.queue.empty()) {
            // Allocate rotating buffer slots
            // Find max tensor size for slot allocation
            size_t max_bytes = 0;
            for (auto & e : jm.queue) max_bytes = std::max(max_bytes, e.q8_total_bytes);
            for (int s = 0; s < 4; s++) {
                jm.buf[s].resize(max_bytes);
            }

            // Start worker thread
            jm.queue_built = true;
            jm.worker = std::thread([bsctx]() {
                auto & jm = bsctx->jit_merge;
                while (!jm.stop.load()) {
                    {
                        std::unique_lock<std::mutex> lock(jm.mtx);
                        jm.cv.wait(lock, [&]{ return jm.has_work.load() || jm.stop.load(); });
                    }
                    if (jm.stop.load()) break;

                    // Merge tensors ahead of consumption
                    while (jm.ready_count.load() < 4 && jm.queue_pos < (int)jm.queue.size()) {
                        int slot = jm.next_slot;
                        auto & e = jm.queue[jm.queue_pos];

                        // Row-by-row merge: blurry + delta → Q8_0
                        ggml_type_traits_t bt = ggml_internal_get_type_traits(e.blurry_type);
                        ggml_type_traits_t dt = ggml_internal_get_type_traits(e.delta_type);
                        ggml_type_traits_t q8t = ggml_internal_get_type_traits(GGML_TYPE_Q8_0);

                        if (bt.to_float && dt.to_float && q8t.from_float) {
                            // Row-by-row merge into slot buffer
                            #pragma omp parallel for schedule(static) num_threads(4)
                            for (int64_t r = 0; r < e.n_rows; r++) {
                                std::vector<float> rf(e.ne0), df(e.ne0);
                                bt.to_float((const char *)e.original_data + r * e.blurry_row_bytes,
                                           rf.data(), e.ne0);
                                dt.to_float(e.delta_host + r * e.delta_row_bytes,
                                           df.data(), e.ne0);
                                for (int64_t i = 0; i < e.ne0; i++) rf[i] += df[i];
                                q8t.from_float(rf.data(), jm.buf[slot].data() + r * e.q8_row_bytes, e.ne0);
                            }

                            // Swap oldest slot back to blurry if slot is being reused
                            if (jm.tensor[slot] != nullptr) {
                                auto * old = jm.tensor[slot];
                                // Find the old entry and restore
                                for (auto & oe : jm.queue) {
                                    if (oe.base == old) {
                                        old->data = oe.original_data;
                                        old->type = oe.blurry_type;
                                        old->nb[0] = ggml_type_size(oe.blurry_type);
                                        old->nb[1] = oe.blurry_row_bytes;
                                        for (int d = 2; d < GGML_MAX_DIMS; d++)
                                            old->nb[d] = old->nb[d-1] * old->ne[d-1];
                                        // Re-enable delta cache (was cleared on swap-in)
                                        // Actually, don't clear it — the delta path checks
                                        // ray_march_delta_cache which stays set
                                        break;
                                    }
                                }
                            }

                            // Swap in: tensor now points to Q8_0 merged data
                            e.base->data = jm.buf[slot].data();
                            e.base->type = GGML_TYPE_Q8_0;
                            e.base->nb[0] = ggml_type_size(GGML_TYPE_Q8_0);
                            e.base->nb[1] = e.q8_row_bytes;
                            for (int d = 2; d < GGML_MAX_DIMS; d++)
                                e.base->nb[d] = e.base->nb[d-1] * e.base->ne[d-1];
                            // Clear delta cache so IQK doesn't also do delta vec_dot
                            e.base->ray_march_delta_cache = NULL;

                            jm.tensor[slot] = e.base;
                            jm.next_slot = (slot + 1) % 4;
                            jm.ready_count.fetch_add(1);

                            static int merge_diag = 0;
                            if (merge_diag < 6) { merge_diag++;
                                fprintf(stderr, "[jit-merge] %s → Q8_0 slot %d\n",
                                        e.base->name, slot);
                            }
                        }

                        jm.queue_pos++;
                        if (jm.queue_pos >= (int)jm.queue.size()) {
                            jm.queue_pos = 0; // wrap for next token
                        }
                    }
                    jm.has_work.store(false);
                }
            });

            LLAMA_LOG_INFO("%s: JIT pre-merge worker started: %zu CPU tensors, %d slots (%.1f MiB/slot)\n",
                           __func__, jm.queue.size(), 4, max_bytes / (1024.0 * 1024.0));

            // Kick off initial merge
            jm.has_work.store(true);
            jm.cv.notify_one();
        }
    }

    return n_wired;
}

// ---------------------------------------------------------------------------
// Load sparse delta GGUF to VRAM for fast speculative delta correction
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_load_sparse_delta_gpu(
        llama_blurry_sharp_context * bsctx,
        const char * sparse_path) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(bsctx); GGML_UNUSED(sparse_path);
    LLAMA_LOG_WARN("%s: CUDA not available, sparse GPU delta disabled\n", __func__);
    return 0;
#else
    if (!bsctx || !bsctx->model || !sparse_path) return 0;

    // Open and parse the sparse delta GGUF
    gguf_init_params gparams = { .no_alloc = true, .ctx = nullptr };
    gguf_context * guf = gguf_init_from_file(sparse_path, gparams);
    if (!guf) {
        LLAMA_LOG_ERROR("%s: failed to open sparse delta: %s\n", __func__, sparse_path);
        return 0;
    }

    // Verify it's a sparse delta
    int k_sparse = gguf_find_key(guf, "bs.delta.sparse");
    if (k_sparse < 0 || !gguf_get_val_bool(guf, k_sparse)) {
        LLAMA_LOG_ERROR("%s: %s is not a sparse delta GGUF\n", __func__, sparse_path);
        gguf_free(guf);
        return 0;
    }

    // Read sparse delta file via llama_file (avoids raw mmap headers)
    auto sparse_file = std::make_unique<llama_file>(sparse_path, "rb");
    if (!sparse_file || sparse_file->size() == 0) { gguf_free(guf); return 0; }
    size_t sfile_size = sparse_file->size();

    size_t data_offset = gguf_get_data_offset(guf);
    int n_tensors = gguf_get_n_tensors(guf);

    // Get quant type from metadata
    int k_qtype = gguf_find_key(guf, "bs.delta.quant_type");
    std::string qtype_str = k_qtype >= 0 ? gguf_get_val_str(guf, k_qtype) : "Q4_0";

    int32_t n_loaded = 0;
    size_t total_vram = 0;

    // Process tensor pairs: {name} (packed data) + {name}.idx (block index)
    for (int ti = 0; ti < n_tensors; ti++) {
        const char * tname = gguf_get_tensor_name(guf, ti);
        std::string name(tname);

        // Skip index tensors (processed with their data tensor)
        if (name.size() > 4 && name.substr(name.size() - 4) == ".idx") continue;

        // Find matching index tensor
        std::string idx_name = name + ".idx";
        int idx_ti = -1;
        for (int j = 0; j < n_tensors; j++) {
            if (idx_name == gguf_get_tensor_name(guf, j)) { idx_ti = j; break; }
        }
        if (idx_ti < 0) continue;

        // Find the base model tensor
        ggml_tensor * base = llama_get_model_tensor(bsctx->model, name.c_str());
        if (!base) continue;
        if (!base->buffer || ggml_backend_buffer_is_host(base->buffer)) continue; // CPU tensor

        // Get metadata
        char key_total[256], key_stored[256];
        snprintf(key_total, sizeof(key_total), "bs.delta.%s.n_blocks_total", name.c_str());
        snprintf(key_stored, sizeof(key_stored), "bs.delta.%s.n_blocks_stored", name.c_str());
        int k_total = gguf_find_key(guf, key_total);
        int k_stored = gguf_find_key(guf, key_stored);
        if (k_total < 0 || k_stored < 0) continue;

        int n_blocks_total = gguf_get_val_i32(guf, k_total);
        int n_blocks_stored = gguf_get_val_i32(guf, k_stored);
        if (n_blocks_stored <= 0) continue;

        // Compute sizes
        ggml_type delta_type = base->ray_march_delta_type;
        size_t block_bytes = ggml_type_size(delta_type);
        int block_elems = ggml_blck_size(delta_type);
        int n_experts = (int)(base->ne[2] > 0 ? base->ne[2] : 1);
        size_t packed_expert_bytes = (size_t)n_blocks_stored * block_bytes;
        size_t packed_total = packed_expert_bytes * n_experts;
        size_t idx_bytes = (size_t)n_blocks_stored * sizeof(uint16_t);

        // Get tensor data offsets and read from file
        size_t packed_offset = data_offset + gguf_get_tensor_offset(guf, ti);
        size_t idx_offset = data_offset + gguf_get_tensor_offset(guf, idx_ti);

        std::vector<uint8_t> packed_buf(packed_total);
        sparse_file->seek(packed_offset, SEEK_SET);
        sparse_file->read_raw(packed_buf.data(), packed_total);

        std::vector<uint8_t> idx_buf(idx_bytes);
        sparse_file->seek(idx_offset, SEEK_SET);
        sparse_file->read_raw(idx_buf.data(), idx_bytes);

        const void * packed_src = packed_buf.data();
        const void * idx_src = idx_buf.data();

        // Upload packed data + index to VRAM
        void * dev_packed = ggml_backend_cuda_device_malloc_and_copy(packed_src, packed_total);
        void * dev_index = ggml_backend_cuda_device_malloc_and_copy(idx_src, idx_bytes);

        if (!dev_packed || !dev_index) {
            if (dev_packed) ggml_backend_cuda_device_free(dev_packed);
            if (dev_index) ggml_backend_cuda_device_free(dev_index);
            continue;
        }

        // Create and attach sparse delta struct
        struct pim_sparse_delta_gpu * sdg = (struct pim_sparse_delta_gpu *)calloc(1, sizeof(*sdg));
        sdg->dev_packed = dev_packed;
        sdg->dev_index = dev_index;
        sdg->n_blocks_stored = n_blocks_stored;
        sdg->n_blocks_total = n_blocks_total;
        sdg->n_experts = n_experts;
        sdg->packed_expert_bytes = packed_expert_bytes;
        sdg->block_bytes = (int)block_bytes;
        sdg->block_elems = block_elems;

        base->sparse_delta_gpu = (void *)sdg;
        n_loaded++;
        total_vram += packed_total + idx_bytes;
    }

    sparse_file.reset();
    gguf_free(guf);

    if (n_loaded > 0) {
        LLAMA_LOG_INFO("%s: loaded %d sparse delta tensors to VRAM (%.1f MiB)\n",
                       __func__, n_loaded, total_vram / (1024.0 * 1024.0));
    }
    return n_loaded;
#endif // GGML_USE_CUDA
}

// ---------------------------------------------------------------------------
// Apply delta correction to non-expert tensors at startup (permanent upgrade)
// ---------------------------------------------------------------------------

static bool bs_is_expert_tensor_name(const std::string & name) {
    return name.find("ffn_gate_exps") != std::string::npos ||
           name.find("ffn_up_exps")   != std::string::npos ||
           name.find("ffn_down_exps") != std::string::npos ||
           name.find("ffn_up_gate_exps") != std::string::npos;
}

int32_t llama_blurry_sharp_apply_delta_non_expert(
        llama_blurry_sharp_context * bsctx) {
    LLAMA_LOG_INFO("%s: starting (bsctx=%p, n_levels=%d)\n", __func__,
                   (void*)bsctx, bsctx ? (int)bsctx->correction_levels.size() : 0);
    fflush(stderr);

    if (!bsctx || bsctx->correction_levels.empty()) {
        return 0;
    }

    auto & level = bsctx->correction_levels[0];
    LLAMA_LOG_INFO("%s: level 0 has %zu delta tensors, %zu mmaps\n", __func__,
                   level.delta_index.size(), level.mmaps.size());
    fflush(stderr);
    int32_t n_upgraded = 0;
    int32_t n_skipped = 0;
    int64_t total_bytes = 0;

    int iter = 0;
    for (auto & [name, dinfo] : level.delta_index) {
        // Skip expert tensors — those use the runtime PIM kernel
        if (bs_is_expert_tensor_name(name)) continue;

        ggml_tensor * base = dinfo.base_tensor;
        if (!base) {
            if (bsctx->model) {
                base = llama_get_model_tensor(bsctx->model, name.c_str());
                dinfo.base_tensor = base;
            }
        }
        if (!base || !base->data) { n_skipped++; continue; }

        LLAMA_LOG_INFO("%s: [%d] %s type=%s buf=%p data=%p ne0=%lld\n", __func__, iter++,
                       name.c_str(), ggml_type_name(base->type),
                       (void*)base->buffer, base->data, (long long)base->ne[0]);
        fflush(stderr);

        // Skip GPU-resident tensors — can't access device memory from CPU.
        if (base->buffer) {
            bool is_host = false;
            // ggml_backend_buffer_is_host may crash if backend not initialized
            // Use a safe check: try-catch or just check buffer type name
            const char * buf_name = ggml_backend_buffer_name(base->buffer);
            is_host = (buf_name && (strstr(buf_name, "CPU") || strstr(buf_name, "cpu") ||
                                    strstr(buf_name, "host") || strstr(buf_name, "Host") ||
                                    strstr(buf_name, "mmap") || strstr(buf_name, "Mmap")));
            LLAMA_LOG_INFO("%s:   buffer=%s is_host=%d\n", __func__, buf_name ? buf_name : "null", (int)is_host);
            fflush(stderr);
            if (!is_host) {
                n_skipped++;
                continue;
            }
        }

        // Get delta data from mmap
        if (dinfo.split_idx >= (int)level.mmaps.size() || !level.mmaps[dinfo.split_idx]) {
            n_skipped++; continue;
        }
        const uint8_t * mmap_base_ptr = (const uint8_t *)level.mmaps[dinfo.split_idx]->addr();
        size_t mmap_total_size = level.mmaps[dinfo.split_idx]->size();

        // Bounds check: delta offset + nbytes must be within the mmap
        if (dinfo.file_offset + dinfo.nbytes > mmap_total_size) {
            LLAMA_LOG_INFO("%s:   SKIP %s — delta offset %zu + nbytes %zu > mmap size %zu\n",
                           __func__, name.c_str(), dinfo.file_offset, dinfo.nbytes, mmap_total_size);
            n_skipped++; continue;
        }

        const uint8_t * delta_data = mmap_base_ptr + dinfo.file_offset;

        int64_t n_elements = 1;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (base->ne[d] > 0) n_elements *= base->ne[d];
        }
        if (n_elements == 0) { n_skipped++; continue; }

        // Verify delta nbytes matches expected size for the type
        size_t expected_delta_bytes = ggml_row_size(dinfo.type, dinfo.ne[0]);
        for (int d = 1; d < GGML_MAX_DIMS; d++) {
            if (dinfo.ne[d] > 0) expected_delta_bytes *= dinfo.ne[d];
        }

        ggml_type_traits_t dt = ggml_internal_get_type_traits(dinfo.type);
        if (!dt.to_float) { n_skipped++; continue; }

        LLAMA_LOG_INFO("%s:   delta_offset=%zu delta_nbytes=%zu mmap_size=%zu n_elem=%lld\n",
                       __func__, dinfo.file_offset, dinfo.nbytes, mmap_total_size, (long long)n_elements);
        fflush(stderr);

        LLAMA_LOG_DEBUG("%s: upgrading %s (type=%s, %lld elements, delta_type=%s)\n",
                        __func__, name.c_str(), ggml_type_name(base->type),
                        (long long)n_elements, ggml_type_name(dinfo.type));

        // Skip F32 base tensors — norms/embeddings are identical between
        // blurry and sharp models (both stored as F32). Delta is quantized zeros.
        if (base->type == GGML_TYPE_F32) {
            n_skipped++;
            continue;
        }

        if (false) { // disabled F32 path — kept for reference
            std::vector<float> delta_f32(n_elements);
            dt.to_float(delta_data, delta_f32.data(), n_elements);

            float * base_f32 = (float *)base->data;
            for (int64_t i = 0; i < n_elements; i++) {
                base_f32[i] += delta_f32[i];
            }
            n_upgraded++;
            total_bytes += n_elements * sizeof(float);
        } else {
            // Quantized base: dequant both → add → requant back to base type.
            // For low-precision base types (e.g. TQ1_0) the delta correction
            // is largely lost, but the non-expert tensors stay compatible with
            // the blurry model's compute graph.
            ggml_type_traits_t bt = ggml_internal_get_type_traits(base->type);
            if (!bt.to_float || !bt.from_float) {
                LLAMA_LOG_DEBUG("%s: skipping %s — no round-trip for type %s\n",
                                __func__, name.c_str(), ggml_type_name(base->type));
                n_skipped++;
                continue;
            }

            size_t original_bytes = ggml_nbytes(base);
            size_t requant_bytes  = ggml_row_size(base->type, base->ne[0]);
            int64_t n_rows = n_elements / base->ne[0];
            if (requant_bytes * n_rows != original_bytes) {
                LLAMA_LOG_DEBUG("%s: skipping %s — size mismatch: %zu vs %zu\n",
                                __func__, name.c_str(), requant_bytes * n_rows, original_bytes);
                n_skipped++;
                continue;
            }

            std::vector<float> base_f32(n_elements);
            std::vector<float> delta_f32(n_elements);

            bt.to_float(base->data, base_f32.data(), n_elements);
            dt.to_float(delta_data, delta_f32.data(), n_elements);

            for (int64_t i = 0; i < n_elements; i++) {
                base_f32[i] += delta_f32[i];
            }

            // The base tensor data lives in a read-only mmap region
            // (PROT_READ | MAP_SHARED).  Allocate a writable copy.
            bsctx->delta_owned_bufs.emplace_back(original_bytes);
            auto & owned = bsctx->delta_owned_bufs.back();

            // Requantize back to base type into writable memory
            bt.from_float(base_f32.data(), owned.data(), n_elements);
            base->data = owned.data();

            n_upgraded++;
            total_bytes += original_bytes;
        }
    }

    LLAMA_LOG_INFO("%s: upgraded %d non-expert tensors (%.1f MiB), skipped %d\n",
                   __func__, n_upgraded, total_bytes / (1024.0 * 1024.0), n_skipped);

    return n_upgraded;
}

// ---------------------------------------------------------------------------
// Apply delta correction to GPU expert tensors at startup (permanent upgrade)
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_apply_delta_gpu_experts(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || bsctx->correction_levels.empty()) {
        return 0;
    }

    auto & level = bsctx->correction_levels[0];
    int32_t n_upgraded = 0;
    int32_t n_skipped = 0;
    int64_t total_bytes = 0;

    LLAMA_LOG_INFO("%s: upgrading GPU tensors with delta correction (experts + attention)...\n", __func__);

    for (auto & [name, dinfo] : level.delta_index) {

        ggml_tensor * base = dinfo.base_tensor;
        if (!base) {
            if (bsctx->model) {
                base = llama_get_model_tensor(bsctx->model, name.c_str());
                dinfo.base_tensor = base;
            }
        }
        if (!base || !base->data) { n_skipped++; continue; }

        // Only process GPU-resident tensors (skip CPU ones — PIM kernel handles those)
        bool is_gpu = base->buffer && !ggml_backend_buffer_is_host(base->buffer);
        const char * buf_name = base->buffer ? ggml_backend_buffer_name(base->buffer) : "null";
        if (!is_gpu) {
            if (n_skipped < 3) {
                LLAMA_LOG_INFO("%s: SKIP (CPU) %s — buffer=%s, is_host=%d\n",
                               __func__, name.c_str(), buf_name,
                               base->buffer ? (int)ggml_backend_buffer_is_host(base->buffer) : -1);
            }
            n_skipped++;
            continue;
        }
        // Skip F32 tensors (norms, biases) — identical between blurry and sharp
        if (base->type == GGML_TYPE_F32) {
            n_skipped++;
            continue;
        }

        LLAMA_LOG_INFO("%s: FOUND GPU tensor: %s type=%s — buffer=%s\n",
                       __func__, name.c_str(), ggml_type_name(base->type), buf_name);

        // Get delta data from mmap
        if (dinfo.split_idx >= (int)level.mmaps.size() || !level.mmaps[dinfo.split_idx]) {
            n_skipped++; continue;
        }
        const uint8_t * delta_data = (const uint8_t *)level.mmaps[dinfo.split_idx]->addr()
                                     + dinfo.file_offset;

        if (dinfo.file_offset + dinfo.nbytes > level.mmaps[dinfo.split_idx]->size()) {
            n_skipped++; continue;
        }

        int64_t n_elements = 1;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (base->ne[d] > 0) n_elements *= base->ne[d];
        }
        if (n_elements == 0) { n_skipped++; continue; }

        ggml_type_traits_t bt = ggml_internal_get_type_traits(base->type);
        ggml_type_traits_t dt = ggml_internal_get_type_traits(dinfo.type);
        if (!bt.to_float || !bt.from_float || !dt.to_float) { n_skipped++; continue; }

        size_t original_bytes = ggml_nbytes(base);

        std::vector<uint8_t> gpu_data(original_bytes);
        ggml_backend_tensor_get(base, gpu_data.data(), 0, original_bytes);

        // Dequant both, add, requant — process per-expert-slice for large tensors
        const int64_t n_experts = (dinfo.ne[2] > 0) ? dinfo.ne[2] : 1;
        const int64_t elems_per_expert = n_elements / n_experts;
        const size_t blurry_bytes_per_expert = original_bytes / n_experts;
        const size_t delta_bytes_per_expert = dinfo.nbytes / n_experts;

        #pragma omp parallel for schedule(dynamic) num_threads(8)
        for (int64_t ei = 0; ei < n_experts; ei++) {
            std::vector<float> bf(elems_per_expert);
            std::vector<float> df(elems_per_expert);

            bt.to_float(gpu_data.data()  + ei * blurry_bytes_per_expert, bf.data(), elems_per_expert);
            dt.to_float(delta_data       + ei * delta_bytes_per_expert,  df.data(), elems_per_expert);

            for (int64_t i = 0; i < elems_per_expert; i++) {
                bf[i] += df[i];
            }

            bt.from_float(bf.data(), gpu_data.data() + ei * blurry_bytes_per_expert, elems_per_expert);
        }

        // Upload corrected data to GPU (single transfer, no round-trip)
        ggml_backend_tensor_set(base, gpu_data.data(), 0, original_bytes);

        n_upgraded++;
        total_bytes += original_bytes;

        if (n_upgraded % 10 == 0) {
            LLAMA_LOG_INFO("%s: %d tensors upgraded (%.1f MiB)...\n",
                           __func__, n_upgraded, total_bytes / (1024.0 * 1024.0));
        }
    }

    LLAMA_LOG_INFO("%s: upgraded %d GPU tensors (%.1f MiB), skipped %d\n",
                   __func__, n_upgraded, total_bytes / (1024.0 * 1024.0), n_skipped);

    // Handle delta data on GPU tensors.
    // Check if the sharp overlay actually upgraded each tensor to a DIFFERENT type.
    // If upgraded: clear delta (sharp weights are better, delta is redundant).
    // If NOT upgraded (same type or no sharp): pin delta for GPU zero-copy correction.
    int n_gpu_delta_pinned = 0;
    int n_gpu_delta_cleared = 0;
    for (auto & [name, dinfo] : level.delta_index) {
        if (dinfo.base_tensor && dinfo.base_tensor->buffer &&
            !ggml_backend_buffer_is_host(dinfo.base_tensor->buffer)) {
            // Check if sharp actually upgraded this tensor to a higher-quality type
            bool was_upgraded = false;
            if (!bsctx->sharp_index.empty()) {
                auto sit = bsctx->sharp_index.find(name);
                if (sit != bsctx->sharp_index.end() && sit->second.type != dinfo.base_tensor->type) {
                    // Sharp provided a different (presumably better) type → tensor was upgraded
                    was_upgraded = true;
                }
            }

            if (was_upgraded) {
                // Sharp overlay corrected this GPU tensor — clear delta
                if (dinfo.base_tensor->ray_march_delta_cache) {
                    pim_delta_cache_free((struct pim_delta_cache *)dinfo.base_tensor->ray_march_delta_cache);
                    dinfo.base_tensor->ray_march_delta_cache = NULL;
                    n_gpu_delta_cleared++;
                }
            } else {
                // Tensor NOT upgraded — pin delta for GPU zero-copy correction.
                struct pim_delta_cache * cache =
                    (struct pim_delta_cache *)dinfo.base_tensor->ray_march_delta_cache;
                if (cache && !cache->gpu_pinned) {
                    size_t total_bytes = 0;
                    char * heap = pim_delta_cache_get_heap_buf(cache, &total_bytes);
                    if (heap && total_bytes > 0) {
#ifdef GGML_USE_CUDA
                        // Pre-populate ALL expert data before pinning (pread from disk)
                        for (int e = 0; e < cache->n_experts; e++) {
                            pim_delta_cache_get(cache, e);
                        }
                        if (ggml_backend_cuda_pin_host_memory(heap, total_bytes)) {
                            cache->gpu_pinned = true;
                            n_gpu_delta_pinned++;
                        } else {
                            LLAMA_LOG_WARN("%s: failed to pin delta for %s\n",
                                __func__, name.c_str());
                        }
#endif
                    }
                }
            }
        }
    }
    if (n_gpu_delta_pinned > 0 || n_gpu_delta_cleared > 0) {
        LLAMA_LOG_INFO("%s: GPU delta: %d pinned for zero-copy, %d cleared (sharp-upgraded)\n",
                       __func__, n_gpu_delta_pinned, n_gpu_delta_cleared);
    }

    // ---- Audit: find ALL GPU tensors and report which are NOT upgraded ----
    if (bsctx->model) {
        int n_gpu_total = 0;
        int n_gpu_upgraded = 0;
        int n_gpu_no_delta = 0;
        int n_gpu_f32_skip = 0;

        // Iterate all tensors in the model
        for (int i = 0; ; i++) {
            char tname_buf[256];
            snprintf(tname_buf, sizeof(tname_buf), "%d", i);
            // Use the layer_tensor_names from bsctx if available,
            // otherwise scan the delta index for base tensors
            break; // can't iterate model tensors without an API
        }

        // Alternative: scan delta index for GPU tensors that were NOT upgraded
        for (auto & [name, dinfo] : level.delta_index) {
            ggml_tensor * base = dinfo.base_tensor;
            if (!base) continue;
            if (!base->buffer || ggml_backend_buffer_is_host(base->buffer)) continue;

            n_gpu_total++;
            if (base->type == GGML_TYPE_F32) {
                n_gpu_f32_skip++;
            } else {
                // Check if it was upgraded (delta cache pointer cleared = upgraded)
                if (base->ray_march_delta_cache == NULL) {
                    n_gpu_upgraded++;
                } else {
                    n_gpu_no_delta++;
                    LLAMA_LOG_WARN("%s: GPU tensor NOT upgraded: %s (type=%s)\n",
                                   __func__, name.c_str(), ggml_type_name(base->type));
                }
            }
        }

        // Also check for GPU tensors that have NO delta entry at all
        // (tensors in the model but not in the delta GGUF)
        // We need to iterate model tensors for this — use sharp_index as proxy
        int n_gpu_missing_delta = 0;
        for (auto & [name, sinfo] : bsctx->sharp_index) {
            if (!sinfo.base_tensor) continue;
            ggml_tensor * base = sinfo.base_tensor;
            if (!base->buffer || ggml_backend_buffer_is_host(base->buffer)) continue;
            if (base->type == GGML_TYPE_F32) continue;

            // Check if this tensor has a delta entry
            if (level.delta_index.find(name) == level.delta_index.end()) {
                n_gpu_missing_delta++;
                LLAMA_LOG_WARN("%s: GPU tensor has NO delta data: %s (type=%s)\n",
                               __func__, name.c_str(), ggml_type_name(base->type));
            }
        }

        LLAMA_LOG_INFO("%s: GPU audit: %d upgraded, %d f32 (skipped), %d failed, %d no delta entry\n",
                       __func__, n_gpu_upgraded, n_gpu_f32_skip, n_gpu_no_delta, n_gpu_missing_delta);

        if (n_gpu_no_delta == 0 && n_gpu_missing_delta == 0) {
            LLAMA_LOG_INFO("%s: ✓ All non-F32 GPU tensors are delta-upgraded\n", __func__);
        }
    }

    return n_upgraded;
}

// ---------------------------------------------------------------------------
// Create ggml_tensor objects backed by mmap'd delta data for graph inclusion
// ---------------------------------------------------------------------------

int32_t llama_blurry_sharp_create_graph_delta_tensors(
        llama_blurry_sharp_context * bsctx) {
    if (!bsctx || !bsctx->fractal_mode || bsctx->correction_levels.empty()) {
        return 0;
    }

    auto & level = bsctx->correction_levels[0];
    if (level.delta_index.empty() || level.mmaps.empty()) {
        return 0;
    }

    // Create a CPU buffer wrapping the entire mmap
    if (!level.mmaps[0]) return 0;
    void * mmap_addr = (void *)level.mmaps[0]->addr();
    size_t mmap_size = level.mmaps[0]->size();

    bsctx->delta_tensor_buf = ggml_backend_cpu_buffer_from_ptr(mmap_addr, mmap_size);
    if (!bsctx->delta_tensor_buf) {
        LLAMA_LOG_ERROR("%s: failed to create CPU buffer for delta tensors\n", __func__);
        return 0;
    }

    // Create a ggml context for the delta tensor metadata
    size_t n_tensors = level.delta_index.size();
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    bsctx->delta_tensor_ctx = ggml_init(ctx_params);
    if (!bsctx->delta_tensor_ctx) {
        LLAMA_LOG_ERROR("%s: failed to create ggml context for delta tensors\n", __func__);
        return 0;
    }

    int32_t n_created = 0;

    for (auto & [name, dinfo] : level.delta_index) {
        if (bs_is_expert_tensor_name(name)) continue;
        if (dinfo.split_idx != 0) continue;
        if (dinfo.nbytes == 0) continue;

        int64_t ne[4] = {
            dinfo.ne[0] > 0 ? dinfo.ne[0] : 1,
            dinfo.ne[1] > 0 ? dinfo.ne[1] : 1,
            dinfo.ne[2] > 0 ? dinfo.ne[2] : 1,
            dinfo.ne[3] > 0 ? dinfo.ne[3] : 1,
        };

        ggml_tensor * dt = ggml_new_tensor_4d(bsctx->delta_tensor_ctx, dinfo.type, ne[0], ne[1], ne[2], ne[3]);
        if (!dt) continue;

        dt->data = (void *)((uint8_t *)mmap_addr + dinfo.file_offset);
        dt->buffer = bsctx->delta_tensor_buf;

        std::string delta_name = name + "_delta";
        ggml_set_name(dt, delta_name.c_str());

        bsctx->delta_graph_tensors[name] = dt;
        n_created++;
    }

    LLAMA_LOG_INFO("%s: created %d graph-level delta tensors in CPU buffer (%.1f MiB mmap)\n",
                   __func__, n_created, mmap_size / (1024.0 * 1024.0));

    return n_created;
}

ggml_tensor * llama_blurry_sharp_get_delta_tensor(
        llama_blurry_sharp_context * bsctx,
        const char * name) {
    if (!bsctx || !name) return nullptr;
    auto it = bsctx->delta_graph_tensors.find(name);
    if (it == bsctx->delta_graph_tensors.end()) return nullptr;
    return it->second;
}
