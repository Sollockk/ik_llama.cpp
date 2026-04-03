#pragma once

//
// Blurry→Sharp overlay system for ik_llama.cpp / GGUF
//
// This module enables selective quality upgrading during inference:
// load a low-quality (blurry) GGUF model for speed, then hotswap
// specific layer weights from a high-quality (sharp) GGUF model
// on-the-fly based on routing decisions.
//
// The system works by:
// 1. Opening a sharp GGUF file and indexing its tensor metadata
// 2. Before a layer's forward pass, reading sharp tensor data
// 3. Dequantizing sharp data → f32 → requantizing to blurry type if needed
// 4. Writing the sharp-quality data into the blurry model's tensor buffers
// 5. Backing up displaced blurry data for later restoration
//
// This is the GGUF-native counterpart of the vLLM blurry_sharp overlay
// engine which operates on safetensors.
//
// PUBLIC API (enums, param/state structs, C functions):
//   Declared in include/llama.h inside extern "C".
//   Do NOT redeclare those types or functions here.
//
// INTERNAL API (C++ helpers, data structures with STL members):
//   Declared below.  These are used only by llama-blurry-sharp.cpp
//   and optionally by other internal translation units.
//

#include "llama-impl.h"   // pulls in llama.h (public enums/structs), LLAMA_LOG_*, format(), etc.
#include "llama-mmap.h"   // llama_file, llama_mmap

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;
struct gguf_context;

// ---------------------------------------------------------------------------
// 2MB-aligned buffer for expert data.
//
// Uses posix_memalign to get huge-page-friendly alignment, reducing TLB
// misses when the kernel reads large expert tensors.  Supports move
// semantics for efficient swap between prefetch and combo buffers.
// ---------------------------------------------------------------------------
struct bs_aligned_buffer {
    static constexpr size_t ALIGNMENT = 2 * 1024 * 1024;  // 2MB

    bs_aligned_buffer() = default;
    ~bs_aligned_buffer() { if (ptr_) free(ptr_); }

    // Move only — no copies (owns raw allocation)
    bs_aligned_buffer(bs_aligned_buffer && o) noexcept
        : ptr_(o.ptr_), size_(o.size_), capacity_(o.capacity_) {
        o.ptr_ = nullptr; o.size_ = 0; o.capacity_ = 0;
    }
    bs_aligned_buffer & operator=(bs_aligned_buffer && o) noexcept {
        if (this != &o) {
            if (ptr_) free(ptr_);
            ptr_ = o.ptr_; size_ = o.size_; capacity_ = o.capacity_;
            o.ptr_ = nullptr; o.size_ = 0; o.capacity_ = 0;
        }
        return *this;
    }
    bs_aligned_buffer(const bs_aligned_buffer &) = delete;
    bs_aligned_buffer & operator=(const bs_aligned_buffer &) = delete;

    void resize(size_t n) {
        if (n <= capacity_) { size_ = n; return; }
        void * p = nullptr;
        if (posix_memalign(&p, ALIGNMENT, n) != 0) {
            throw std::bad_alloc();
        }
#ifndef _WIN32
        // Hint the kernel to back this allocation with huge pages (2MB).
        // Reduces TLB misses when reading/writing large expert tensors.
        madvise(p, n, MADV_HUGEPAGE);
#endif
        if (ptr_) free(ptr_);
        ptr_ = static_cast<uint8_t *>(p);
        size_ = capacity_ = n;
    }

    uint8_t * data()             { return ptr_; }
    const uint8_t * data() const { return ptr_; }
    size_t size()    const       { return size_; }
    bool   empty()   const       { return size_ == 0; }
    void   clear()               { size_ = 0; }  // keeps allocation

    // For compatibility with code that does assign(begin, end)
    void assign(const uint8_t * begin, const uint8_t * end) {
        size_t n = end - begin;
        resize(n);
        std::memcpy(ptr_, begin, n);
    }

private:
    uint8_t * ptr_      = nullptr;
    size_t    size_     = 0;
    size_t    capacity_ = 0;
};

// ---------------------------------------------------------------------------
// Per-layer state query result  (internal – used by get_layer_state)
// ---------------------------------------------------------------------------

struct llama_blurry_sharp_layer_state {
    int32_t layer_idx;
    bool    is_sharpened;
    int32_t n_tensors_overlaid;
    int64_t backup_bytes;        // bytes of blurry data backed up
    int64_t sharp_bytes_read;    // bytes read from sharp file for this layer
    int64_t timestamp_us;        // last apply/restore timestamp (microseconds)
};

// ---------------------------------------------------------------------------
// Sharp tensor index entry  (internal – has STL members)
// ---------------------------------------------------------------------------

struct blurry_sharp_tensor_info {
    std::string name;          // tensor name in the sharp GGUF
    int         tensor_idx;    // index in the sharp GGUF file (within its split)
    int         split_idx;     // which split file this tensor lives in (0 = first/only file)
    size_t      file_offset;   // byte offset of data in the sharp file
    ggml_type   type;          // quantization type in the sharp file
    int64_t     ne[GGML_MAX_DIMS]; // shape
    size_t      nbytes;        // total bytes of this tensor in the sharp file
    int         layer_idx;     // extracted layer index (-1 if not a layer tensor)

    // Cached pointer to the base model tensor (populated during init validation).
    // Avoids O(n) linear search through model->tensors_by_name on every apply/restore.
    ggml_tensor * base_tensor = nullptr;
};

// ---------------------------------------------------------------------------
// Per-tensor backup entry  (internal)
// ---------------------------------------------------------------------------

struct blurry_sharp_tensor_backup {
    std::string            tensor_name;

    // Cached pointer to the base model tensor.  Stable for the model's lifetime.
    // Avoids O(n) linear search on restore.
    ggml_tensor          * base_tensor = nullptr;

    // Pointer-swap backup: we save the original tensor metadata and restore it.
    // The original data stays in place (heap, mmap, or device buffer) untouched.
    void                 * original_data;            // original tensor->data pointer
    ggml_type              original_type;            // original tensor->type
    size_t                 original_nb[GGML_MAX_DIMS]; // original tensor->nb strides
    int64_t                ne[GGML_MAX_DIMS];
    ggml_backend_buffer_t  original_buffer = nullptr; // original tensor->buffer (saved for device path)
    ggml_tensor          * original_view_src = nullptr; // original tensor->view_src (cleared during overlay to avoid stale dispatch)
    void                 * original_extra = nullptr;    // original tensor->extra (split tensor metadata, cleared during overlay)

    // --- Three overlay strategies depending on tensor location ---
    //
    //  1. ZERO-COPY (CPU tensor + mmap sharp file):
    //     Point tensor->data directly at the mmap'd sharp page.
    //     No allocation, no memcpy.  OS demand-pages from disk.
    //
    //  2. BUFFERED (CPU tensor + no mmap):
    //     Read sharp data into sharp_data heap buffer, point tensor at it.
    //
    //  3. DEVICE (GPU/non-host tensor):
    //     Allocate a NEW device buffer sized for the sharp data (e.g. Q8_0),
    //     copy sharp data from CPU (mmap/file) → device via tensor_set,
    //     then pointer-swap tensor->{data, type, nb[], buffer} to the new
    //     allocation.  The original device buffer is NOT modified — it stays
    //     intact in the model's shared allocation, so NO backup copy is needed.
    //     On restore: swap back and free the new device buffer.
    //
    bool                   zero_copy      = false;   // true  → data lives in sharp mmap (not owned)
    bool                   is_device      = false;   // true  → GPU pointer-swap with separate device buffer
    std::vector<uint8_t>   sharp_data;               // sharp tensor data (buffered CPU path only)
    ggml_backend_buffer_t  device_sharp_buffer = nullptr; // device buffer holding sharp data (GPU path only, freed on restore)
    bool                   device_buf_cached = false;     // true → device_sharp_buffer came from the pool and should NOT be freed on restore

    // --- In-place expert slice backup (MoE combination mode) ---
    //
    // When the blurry and sharp models have the SAME type for an expert
    // tensor, we overwrite only the selected expert slices in-place
    // (zero extra VRAM on GPU, zero heap allocation on CPU).  To enable
    // restore, we save the original blurry data for just those slices.
    //
    // expert_backup_ids:  which expert indices were overwritten
    // expert_backup_data: concatenated original blurry data for those slices
    //                     (n_experts_req × expert_slice_bytes)
    std::vector<int32_t>   expert_backup_ids;         // expert indices that were overwritten
    std::vector<uint8_t>   expert_backup_data;        // saved blurry data for those expert slices

    // --- Flash-expert mode ---
    // When true, the overlay used a shared scratch buffer and the original
    // tensor data is irrelevant (never loaded).  Restore only needs to
    // swap pointers back — no data copy needed.
    bool                   flash_expert = false;
};

// ---------------------------------------------------------------------------
// Cached device buffer entry  (internal — reused across sharpen/restore cycles)
//
// When retain_device_buffers is enabled, device buffers allocated for sharp
// tensor data are NOT freed on restore.  Instead they are kept in a cache
// keyed by tensor name.  On the next sharpen of the same tensor, the cached
// buffer is reused — only the PCIe copy (CPU→GPU) is needed, the expensive
// cudaMalloc is skipped.  If the sharp data hasn't changed, even the copy
// can be skipped (the data is already on the device from last time).
// ---------------------------------------------------------------------------

struct blurry_sharp_device_cache_entry {
    ggml_backend_buffer_t  buffer  = nullptr;   // the pre-allocated device buffer
    size_t                 nbytes  = 0;         // size of the buffer
    bool                   populated = false;   // true if sharp data has been copied into it
    uint64_t               use_sequence = 0;    // incremented each time this entry is used (for LRU eviction)
    int                    layer_idx = -1;      // which layer this tensor belongs to (for layer-level eviction)
    // The tensor name is the map key, not stored here.
};

// ---------------------------------------------------------------------------
// Prompt heatmap — expert routing statistics collected during prompt processing.
//
// Records how many tokens activated each (layer, expert) pair.
// High activation count = this expert heavily influenced the KV cache.
// Used to prioritize which experts to sharpen in a second pass.
// ---------------------------------------------------------------------------
struct bs_prompt_heatmap {
    struct expert_heat {
        int32_t n_activations = 0;
    };

    // per_layer[layer_idx][expert_idx]
    std::unordered_map<int, std::vector<expert_heat>> per_layer;
    int n_experts_total = 0;

    void init(int n_experts) {
        n_experts_total = n_experts;
        per_layer.clear();
    }

    void record(int layer_idx, int expert_idx) {
        if (n_experts_total <= 0 || expert_idx < 0 || expert_idx >= n_experts_total) return;
        auto & layer = per_layer[layer_idx];
        if ((int)layer.size() != n_experts_total) layer.resize(n_experts_total);
        layer[expert_idx].n_activations++;
    }

    int activations(int layer_idx, int expert_idx) const {
        auto it = per_layer.find(layer_idx);
        if (it == per_layer.end()) return 0;
        if (expert_idx >= (int)it->second.size()) return 0;
        return it->second[expert_idx].n_activations;
    }

    // Returns (layer_idx, expert_idx) pairs sorted hottest first.
    std::vector<std::pair<int,int>> sorted_hottest() const {
        std::vector<std::tuple<int,int,int>> items;
        for (auto & [li, layer] : per_layer) {
            for (int ei = 0; ei < (int)layer.size(); ++ei) {
                if (layer[ei].n_activations > 0)
                    items.push_back({layer[ei].n_activations, li, ei});
            }
        }
        std::sort(items.begin(), items.end(), std::greater<>());
        std::vector<std::pair<int,int>> result;
        result.reserve(items.size());
        for (auto & [cnt, li, ei] : items) result.push_back({li, ei});
        return result;
    }

    void clear() { per_layer.clear(); }
};

// ---------------------------------------------------------------------------
// Per-layer backup state  (internal)
// ---------------------------------------------------------------------------

struct blurry_sharp_layer_backup {
    int                                      layer_idx;
    bool                                     is_sharpened;
    int64_t                                  backup_bytes;
    int64_t                                  sharp_bytes_read;
    int64_t                                  apply_timestamp_us;
    uint64_t                                 apply_sequence;   // for LRU / FIFO eviction
    std::vector<blurry_sharp_tensor_backup>  tensor_backups;
};

// ---------------------------------------------------------------------------
// Router decision  (internal – returned by the C++ route function)
// ---------------------------------------------------------------------------

struct blurry_sharp_routing_decision {
    int32_t              layer_idx;
    bool                 should_sharpen;
    float                confidence;     // 0.0 = definitely sharpen, 1.0 = skip
    std::vector<float>   per_token_confidences;
};

// ---------------------------------------------------------------------------
// Statistics / metrics  (internal)
// ---------------------------------------------------------------------------

struct blurry_sharp_metrics {
    int64_t  n_apply_calls;
    int64_t  n_restore_calls;
    int64_t  n_evictions;
    int64_t  n_dequant_requant;       // cross-type conversions
    int64_t  n_direct_copies;         // same-type fast copies
    int64_t  total_sharp_bytes_read;
    int64_t  total_backup_bytes_written;
    int64_t  total_time_apply_us;
    int64_t  total_time_restore_us;
    int64_t  total_time_dequant_us;
};

// ---------------------------------------------------------------------------
// Main context  (internal – the opaque pointer behind the public C handle)
//
// In llama.h we forward-declare:
//   struct llama_blurry_sharp_context;
// Here we provide the full definition.
// ---------------------------------------------------------------------------

struct llama_blurry_sharp_context {
    // -- references --
    llama_model * model = nullptr;

    // -- configuration  (uses the public struct from llama.h) --
    llama_blurry_sharp_params params = {};

    // -- sharp GGUF file(s) --
    // For single GGUF: one entry in each vector.
    // For split GGUF (e.g. model-00001-of-00003.gguf): one entry per split file.
    // Index 0 is always the primary file passed via sharp_model_path.
    std::vector<gguf_context *>                  sharp_ggufs;        // GGUF metadata per split
    std::vector<ggml_context *>                  sharp_tensor_ctxs;  // tensor metadata contexts per split
    std::vector<std::unique_ptr<llama_file>>      sharp_files;       // file handles per split
    std::vector<std::unique_ptr<llama_mmap>>      sharp_mmaps;       // mmaps per split (may be null entries)
    int                                           n_sharp_splits = 0;

    // -- tensor index --
    // Maps tensor name → info about that tensor in the sharp GGUF.
    std::unordered_map<std::string, blurry_sharp_tensor_info> sharp_index;

    // Maps layer index → list of tensor names belonging to that layer.
    std::unordered_map<int, std::vector<std::string>> layer_tensor_names;

    // Sorted list of layer indices present in the sharp model.
    std::vector<int> sharp_layer_indices;

    // -- eligibility --
    std::unordered_set<int> eligible_layers;

    // -- per-layer backup state --
    std::unordered_map<int, blurry_sharp_layer_backup> layer_backups;

    // -- eviction ordering --
    uint64_t apply_sequence_counter = 0;

    // -- memory accounting --
    int64_t current_backup_bytes = 0;
    int64_t current_device_sharp_bytes = 0;  // GPU memory currently allocated for sharp tensors
    int32_t n_device_tensors_skipped = 0;    // tensors skipped because GPU budget was exceeded

    // -- device buffer cache --
    // When retain_device_buffers is true, device buffers are kept across
    // restore cycles so that re-sharpening the same tensor skips cudaMalloc
    // and (if data unchanged) also skips the PCIe copy.
    //
    // When the GPU budget is exceeded and retain_device_buffers is on,
    // the cache uses LRU eviction: the least-recently-used cached buffers
    // are freed to make room for new ones.  This way "hot" layers stay
    // in VRAM while "cold" ones get paged out.
    bool retain_device_buffers = false;
    std::unordered_map<std::string, blurry_sharp_device_cache_entry> device_cache;
    int64_t device_cache_bytes = 0;   // total bytes held in the cache (always on GPU)
    uint64_t cache_use_counter = 0;   // monotonic counter for LRU ordering

    // -- cache hit/miss stats --
    int64_t n_cache_hits     = 0;     // re-sharpen with cached buffer (no alloc, no copy)
    int64_t n_cache_allocs   = 0;     // first-time alloc that goes into the cache
    int64_t n_cache_recopies = 0;     // cached buffer reused but data re-copied (size matched but !populated)
    int64_t n_cache_evictions = 0;    // cached buffers evicted to make room for new ones

    // -- pinned staging buffer --
    // When a GPU backend is available, the staging buffer for CPU→GPU copies
    // is allocated as CUDA/Vulkan/SYCL pinned (page-locked) host memory.
    // This enables DMA transfers at full PCIe bandwidth (~12-26 GB/s)
    // instead of going through pageable memory (~4-8 GB/s).
    ggml_backend_buffer_t pinned_staging_buf = nullptr;  // pinned host buffer (freed on cleanup)
    void *                pinned_staging_ptr = nullptr;   // base pointer into pinned buffer
    size_t                pinned_staging_size = 0;        // current allocation size

    // -- temporary buffers (reused across calls to avoid re-allocation) --
    std::vector<float>   dequant_buf;
    std::vector<uint8_t> requant_buf;
    std::vector<uint8_t> read_buf;    // fallback when pinned staging is unavailable or too small

    // -- owned buffers for delta-upgraded tensors --
    // When apply_delta_non_expert() modifies mmap'd tensors, we must allocate
    // writable copies (mmap is PROT_READ / MAP_SHARED).  These are kept alive
    // for the model's lifetime and freed when this context is destroyed.
    std::vector<std::vector<uint8_t>> delta_owned_bufs;

    // -- RAM cache for sharp tensor data (memory-tier management) --
    //
    // When enabled, sharp tensor data is pre-read into anonymous heap buffers
    // instead of relying solely on file-backed mmap pages.  This changes the
    // memory pressure behavior:
    //
    //   File-backed mmap pages (zero-copy):
    //     Under pressure → DROPPED (kernel discards them, re-faults from disk)
    //     Re-access cost: full disk I/O from GGUF file (slow)
    //
    //   Anonymous heap buffers (RAM cache):
    //     Under pressure → SWAPPED OUT to swap partition
    //     Re-access cost: swap-in from SSD (fast, typically 10-50x faster)
    //
    // The RAM cache fills available RAM at startup, then lets the OS swap
    // manager push cold pages to swap as needed.  This ensures the memory
    // hierarchy is: VRAM > RAM > Swap > Disk, instead of: VRAM > Disk.
    //
    // Optional: after populating the cache, MADV_PAGEOUT can proactively
    // move pages to swap, freeing RAM for the blurry model / KV cache
    // while keeping sharp data quickly accessible via swap-in.
    //
    std::unordered_map<std::string, std::vector<uint8_t>> ram_cache;  // tensor name → heap buffer
    int64_t ram_cache_bytes     = 0;    // total bytes currently in the RAM cache
    bool    ram_cache_populated = false; // true after precache_ram() completes
    bool    ram_cache_staged    = false; // true after stage_to_swap() completes

    // -- lazy swap staging --
    // When enabled, the RAM cache is populated lazily on a per-layer basis
    // the first time apply_layer() is called for each layer.  After a layer
    // is restored (restore_layer / restore_all), its cached pages are pushed
    // to swap via MADV_PAGEOUT.  Subsequent accesses swap-in from SSD
    // (fast) instead of re-reading from the GGUF file (slow).
    //
    // This eliminates the 30-minute startup cost of --bs-stage-swap by
    // spreading the disk→RAM→swap migration across the first inference pass.
    bool    lazy_swap_enabled   = false; // true when --bs-lazy-swap is active

    // -- async prefetch thread --
    // Background thread that pre-reads tensor data into a staging area
    // ahead of the main apply loop.  On first pass (no ram_cache yet),
    // this reads from mmap/file into prefetch_cache.  The main thread
    // moves data from prefetch_cache → ram_cache when it needs it.
    //
    // Thread-safety: prefetch_cache is protected by prefetch_mtx (NOT
    // the main bsctx->mtx).  The main thread briefly locks prefetch_mtx
    // to move data out, avoiding contention with the main mutex.
    std::thread                prefetch_thread;
    std::mutex                 prefetch_mtx;
    std::condition_variable    prefetch_cv;
    std::atomic<bool>          prefetch_stop{false};
    // Layers queued for prefetch (sorted ascending).
    std::vector<int>           prefetch_queue;
    // Pre-read tensor data: tensor_name → heap buffer.
    std::unordered_map<std::string, std::vector<uint8_t>> prefetch_cache;
    int64_t                    prefetch_cache_bytes = 0;
    bool                       prefetch_thread_started = false;

    // -- metrics --
    blurry_sharp_metrics metrics = {};

    // -- JIT mode tracking --
    // When true, overlays are being applied DURING graph execution (from the
    // eval callback).  In this mode, the backend scheduler has already
    // allocated device copy tensors at the ORIGINAL (blurry) type/size.
    // Cross-type overlays on HOST tensors are unsafe because the scheduler's
    // copy_inputs would try to write sharp-sized data into blurry-sized
    // device copies, triggering an assertion failure:
    //
    //   ggml_backend_tensor_set_async: offset + size <= ggml_nbytes(tensor)
    //
    // Same-type overlays are safe (sizes match).  Device tensor overlays
    // are also safe (no scheduler copy involved — we swap the buffer directly).
    bool jit_active = false;

    // When true, JIT sharpening only applies to host (CPU) tensors.
    // Device (GPU) layers are left at blurry quality.  This preserves
    // CUDA graph capture on GPU splits (eval callback skips device layers).
    bool jit_host_only = false;

    // Count of host cross-type tensors skipped in JIT mode (for diagnostics)
    int32_t n_jit_host_crosstype_skipped = 0;

    // True when inflate shrunk ne[2] for this batch (generation).
    // Set during inflate — tells the JIT callback to upload sharp expert
    // slices to GPU device copies (type was inflated for correct sizing).
    bool inflate_shrunk_ne2 = false;

    // Saved original types for inflate/deflate cycle.
    // Populated by llama_blurry_sharp_inflate_expert_types(), consumed by deflate.
    struct inflate_saved_entry {
        ggml_tensor * tensor;
        ggml_type     orig_type;
        size_t        orig_nb[GGML_MAX_DIMS];
    };
    std::vector<inflate_saved_entry> inflate_saved_types;

    // ---- GPU Expert Cache ----
    // Persistent GPU-side cache of Q3_K_M (sharp) expert slices.
    // Cache hits use fast GPU→GPU copy; misses read from SSD and populate cache.
    // Between tokens, expert routing has temporal locality, so most slices
    // are already cached → dramatically reduces PCIe/SSD I/O.
    struct gpu_expert_cache {
        bool enabled = false;
        ggml_backend_t gpu_backend = nullptr;
        ggml_backend_buffer_t cache_buf = nullptr;
        void * gpu_base = nullptr;          // device pointer (base of cache_buf)
        size_t total_bytes = 0;
        size_t used_bytes = 0;

        struct entry {
            size_t offset;          // byte offset in cache_buf
            size_t size;            // byte size of the cached slice
            int64_t last_access;    // access counter for LRU eviction
        };

        // Key: "tensor_name\0expert_id" → cache entry
        std::unordered_map<uint64_t, entry> entries;
        int64_t access_counter = 0;

        // Simple free-list arena allocator (first-fit).
        struct free_block { size_t offset, size; };
        std::vector<free_block> free_list;

        // Stats
        int64_t n_hits = 0;
        int64_t n_misses = 0;
    } gpu_cache;

    // ---- RAM Expert Cache ----
    // Persistent host-side cache of sharp expert slices in anonymous memory.
    // File-backed mmap pages get evicted by the kernel under memory pressure;
    // anonymous pages go to swap instead (faster to recover from SSD).
    // Cache key: FNV-1a hash of (tensor_name, expert_id).
    struct ram_expert_cache {
        bool enabled = false;
        size_t budget_bytes = 0;        // max cache size
        size_t used_bytes = 0;

        struct entry {
            std::vector<uint8_t> data;
            int64_t last_access;
        };

        std::unordered_map<uint64_t, entry> entries;
        int64_t access_counter = 0;
        int64_t n_hits = 0;
        int64_t n_misses = 0;
    } ram_expert_cache;

    // Reusable combination buffers for expert tensor overlay.
    // One per concurrent expert tensor (gate, up, down).  Allocated lazily,
    // sized for the full tensor (160 experts × slice_size).  Only the
    // active expert slots are populated; untouched pages consume no RAM.
    std::unordered_map<std::string, bs_aligned_buffer> combo_buffers;

    // Flash-expert scratch buffers (shared across layers).
    // When flash_experts is active, expert tensors are NOT loaded from the
    // blurry GGUF.  Instead, Q4_K_M expert slices are streamed from SSD
    // into these buffers on demand.  Keyed by tensor suffix (e.g.,
    // "ffn_gate_exps.weight") so all layers share the same 3 buffers.
    // Each buffer holds n_expert_used expert slots at Q4_K_M size.
    std::unordered_map<std::string, bs_aligned_buffer> flash_scratch;

    // ---- Async prefetch (double-buffered I/O) ----
    // While the MoE kernel computes layer N using combo_buffers, a background
    // thread reads layer N+1's predicted expert slices into prefetch_buffers.
    // When layer N+1's callback fires, we swap prefetch→combo (no I/O).
    //
    // Double-buffered: dbufs[0] and dbufs[1] alternate each cycle.
    // The prefetch thread writes to dbufs[write_idx]; the consumer reads
    // from dbufs[1 - write_idx] (the previous cycle's completed data).
    // Buffer allocations are reused across cycles (never freed during
    // normal operation) to avoid heap corruption from rapid alloc/free
    // of large 2MB-aligned buffers.
    struct {
        std::thread                worker;
        std::atomic<bool>          ready{false};
        std::atomic<bool>          active{false};  // worker is running
        int                        layer_idx = -1;
        std::vector<int32_t>       expert_ids;

        std::unordered_map<std::string, bs_aligned_buffer> dbufs[2];
        int                        write_idx = 0;  // prefetch writes to dbufs[write_idx]

        int64_t                    n_hits = 0;     // prefetch buffer used (I/O saved)
        int64_t                    n_misses = 0;   // prefetch missed, fell back to sync I/O
    } async_prefetch;

    // Heatmap — populated passively during prompt processing.
    // Records per-(layer, expert) activation frequency for two-pass analysis.
    // Set heatmap_collecting = true before a prompt to enable recording.
    bs_prompt_heatmap prompt_heatmap;
    bool heatmap_collecting = false;

    // -- self-eviction guard --
    // During apply_layer / apply_experts, tensors are overlaid one at a time.
    // Each overlay may allocate a new device buffer and insert it into the
    // device_cache.  If a subsequent tensor in the SAME layer triggers LRU
    // eviction, the eviction code must NOT free buffers that were just
    // allocated for earlier tensors of the same layer — because those
    // tensors' base_tensor->data pointers already point at those buffers.
    //
    // The layer_backups[layer].is_sharpened flag is only set AFTER all
    // tensors are successfully overlaid, so the existing "in_use" check
    // (which looks at is_sharpened) does NOT protect the layer currently
    // being built.  This field closes that gap.
    //
    // Set to the layer_idx at the start of apply_layer / apply_experts,
    // reset to -1 when the function returns.  The eviction loop skips
    // any cache entry whose layer_idx matches this value.
    int building_layer_idx = -1;

    // -- fractal delta correction chain --
    // When fractal_mode is true, weights are reconstructed additively:
    //   result = blurry + sum(delta_level[i] for i in 1..depth)
    // Each level is a separate GGUF file containing quantized residuals.
    struct bs_correction_level {
        int32_t level = 0;                    // 1-indexed level number
        std::vector<gguf_context *>                  ggufs;
        std::vector<ggml_context *>                  tensor_ctxs;
        std::vector<std::unique_ptr<llama_file>>     files;
        std::vector<std::unique_ptr<llama_mmap>>     mmaps;
        int                                          n_splits = 0;
        std::unordered_map<std::string, blurry_sharp_tensor_info> delta_index;
        std::unordered_map<int, float> layer_rmse;  // from GGUF metadata
    };
    std::vector<bs_correction_level> correction_levels;  // levels 1..N
    bool    fractal_mode = false;
    int32_t fractal_max_depth = 0;

    // Scratch buffer for f32 accumulation during fractal reconstruction
    std::vector<float> fractal_accum_buf;

    // -- thread safety --
    std::mutex mtx;

    // -- lifecycle --
    bool initialized = false;

    // Skip GPU tensors in apply_non_expert_permanent (saves VRAM)
    bool skip_non_expert_gpu = false;

    // -- graph-level delta tensors --
    ggml_context *          delta_tensor_ctx  = nullptr;
    ggml_backend_buffer_t   delta_tensor_buf  = nullptr;
    std::unordered_map<std::string, ggml_tensor *> delta_graph_tensors;

    // -- CPU JIT pre-merge: background merge of blurry+delta → Q8_0 --
    struct {
        std::vector<uint8_t> buf[4];                 // Q8_0 data per slot (4 rotating slots)
        ggml_tensor *        tensor[4] = {};         // which tensor is in each slot
        int                  next_slot = 0;          // round-robin

        // Queue of CPU tensors to merge (built during first gen token)
        struct merge_entry {
            ggml_tensor *    base;                   // model tensor (to swap data ptr)
            const char *     delta_host;             // delta mmap pointer
            ggml_type        blurry_type;
            ggml_type        delta_type;
            int64_t          ne0, n_rows;
            size_t           blurry_row_bytes;
            size_t           delta_row_bytes;
            size_t           q8_row_bytes;
            size_t           q8_total_bytes;
            void *           original_data;          // original blurry data ptr (for swap-back)
        };
        std::vector<merge_entry> queue;
        int queue_pos = 0;                           // next to merge
        bool queue_built = false;

        // Worker thread
        std::thread          worker;
        std::mutex           mtx;
        std::condition_variable cv;
        std::atomic<bool>    has_work{false};
        std::atomic<int>     ready_count{0};          // how many are merged and ready
        std::atomic<bool>    stop{false};

        // Lookup: tensor ptr → slot index
        std::unordered_map<const ggml_tensor *, int> ready_map;
    } jit_merge;

    // Pre-built list of GPU delta tensors for ring buffer fill
    struct gpu_delta_entry {
        const ggml_tensor * tensor;
        const char * host_ptr;
        size_t bytes;
    };
    std::vector<gpu_delta_entry> gpu_delta_fill_tensors;
};

// ---------------------------------------------------------------------------
// Internal C++ helpers  (not exposed through the public C API)
// ---------------------------------------------------------------------------

// Initialize the GPU expert cache.  Call once with a GPU backend after
// the blurry-sharp context is created and params.gpu_cache_bytes > 0.
void llama_blurry_sharp_gpu_cache_init(
        llama_blurry_sharp_context * bsctx,
        ggml_backend_t               gpu_backend);

// GPU expert cache helpers (used by JIT callback in llama.cpp)
int64_t bs_gpu_cache_lookup(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        const std::string & tensor_name,
        int expert_id);

int64_t bs_gpu_cache_store(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        const std::string & tensor_name,
        int expert_id,
        const void * host_data,
        size_t size);

void bs_gpu_cache_copy_to_dcpy(
        llama_blurry_sharp_context::gpu_expert_cache & cache,
        int64_t cache_offset,
        size_t  size,
        ggml_tensor * dcpy,
        size_t  dcpy_offset);

// Pre-read all sharp tensor data into anonymous heap buffers (RAM cache).
// This populates the RAM tier so that data can be swapped out under pressure
// instead of being dropped entirely (as happens with file-backed mmap pages).
// Returns the number of tensors cached.
int32_t llama_blurry_sharp_precache_ram(
        llama_blurry_sharp_context * bsctx);

// Proactively move RAM-cached pages to swap using MADV_PAGEOUT (Linux 5.4+).
// This frees RAM for the blurry model / KV cache while keeping sharp data
// quickly accessible via swap-in.  No-op on systems without MADV_PAGEOUT.
// Returns the number of tensors whose pages were staged to swap.
int32_t llama_blurry_sharp_stage_to_swap(
        llama_blurry_sharp_context * bsctx);


// Compute routing decisions for all eligible layers.
// `activations` is an optional float[n_tokens * n_embd] hidden-state buffer.
// For ALWAYS/NEVER strategies it may be nullptr.
std::vector<blurry_sharp_routing_decision> llama_blurry_sharp_route(
        llama_blurry_sharp_context * bsctx,
        const float                * activations,
        int32_t                      n_tokens,
        int32_t                      n_embd);

// Get a snapshot of collected metrics.
blurry_sharp_metrics llama_blurry_sharp_get_metrics(
        const llama_blurry_sharp_context * bsctx);

// Reset all metric counters to zero.
void llama_blurry_sharp_reset_metrics(
        llama_blurry_sharp_context * bsctx);

// Get per-layer overlay state.  Returns false if layer_idx is out of range.
bool llama_blurry_sharp_get_layer_state(
        const llama_blurry_sharp_context    * bsctx,
        int                                   layer_idx,
        llama_blurry_sharp_layer_state      * out);

// Apply all eligible layers in one call.
// Benefits over looping apply_layer() externally:
//   - Prefetches mmap pages with madvise(WILLNEED) before starting
//   - Better progress reporting
// Returns the number of layers successfully sharpened.
int32_t llama_blurry_sharp_apply_all(
        llama_blurry_sharp_context * bsctx);

// Prefetch live zero-copy tensor pages into the page cache (RAM).
//
// After sharpening layers, CPU zero-copy tensors point at mmap pages that
// may not yet be resident in RAM (the sharp mmap is created with prefetch=0
// for selective use).  This function calls madvise(WILLNEED) on every live
// zero-copy tensor's data range, telling the kernel to pull those pages
// into the page cache using available RAM.
//
// Call this after apply_layer / apply_all when the sharpened tensors will be
// accessed repeatedly during generation.  Without this, each forward pass
// may stall on page faults reading from disk even when RAM is available.
//
// For GPU tensors this is a no-op (data lives in VRAM, mmap pages were
// already released).  For selective sharpening (top-K layers), only those
// layers' pages are warmed.
void llama_blurry_sharp_warm_live_pages(
        llama_blurry_sharp_context * bsctx);

// Pre-allocate device (GPU) buffers for all eligible tensors.
// This moves cudaMalloc out of the apply hot-path: subsequent apply_layer()
// calls hit the device cache and only need the data copy (PCIe transfer),
// skipping the expensive per-tensor cudaMalloc.
// Implicitly enables retain_device_buffers.
// Returns the number of device buffers pre-allocated.
int32_t llama_blurry_sharp_preload_device_cache(
        llama_blurry_sharp_context * bsctx);

// Extract a layer index from a GGUF tensor name.
// E.g. "blk.3.attn_norm.weight" → 3.  Returns -1 if not matched.
int llama_blurry_sharp_extract_layer_idx(const std::string & tensor_name);

// Check whether a base-model tensor and a sharp tensor have compatible shapes
// (same total number of elements, though possibly different quantization types).
bool llama_blurry_sharp_shapes_compatible(
        const ggml_tensor              * blurry_tensor,
        const blurry_sharp_tensor_info & sharp_info);

// Log a human-readable summary of the sharp-model index to LLAMA_LOG_INFO.
void llama_blurry_sharp_log_index_summary(
        const llama_blurry_sharp_context * bsctx);