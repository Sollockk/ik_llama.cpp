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

#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;
struct gguf_context;

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

    // Count of host cross-type tensors skipped in JIT mode (for diagnostics)
    int32_t n_jit_host_crosstype_skipped = 0;

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

    // -- thread safety --
    std::mutex mtx;

    // -- lifecycle --
    bool initialized = false;
};

// ---------------------------------------------------------------------------
// Internal C++ helpers  (not exposed through the public C API)
// ---------------------------------------------------------------------------

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