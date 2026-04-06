// bs-deltacalc: generate hierarchical delta correction GGUFs for fractal weight reconstruction.
//
// Streaming architecture: writes tensor data to output files as each tensor
// completes, so RAM usage stays bounded and no work is lost on crash.
//
// Usage:
//   bs-deltacalc \
//     --blurry  model_tq10.gguf \
//     --sharp   model_q4km-00001-of-00002.gguf \
//     --levels  3 \
//     --quant   Q2_K \
//     --out-prefix delta \
//     --threads 20

#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include <omp.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef BS_HAVE_LAPACK
extern "C" {
    // LAPACK truncated SVD: compute min(M,N) singular values/vectors
    void sgesvd_(const char * jobu, const char * jobvt,
                 const int * m, const int * n, float * a, const int * lda,
                 float * s, float * u, const int * ldu,
                 float * vt, const int * ldvt,
                 float * work, const int * lwork, int * info);
}
#endif

// ---------------------------------------------------------------------------
// QD4_K block layout (must match ggml-common.h)
// ---------------------------------------------------------------------------
#ifndef QK_K
#define QK_K 256
#endif

#pragma pack(push, 1)
struct block_qd4_k {
    uint16_t d;              // super-block scale (fp16)
    uint8_t  energy;         // log-scale block energy
    uint8_t  flags;          // reserved
    uint8_t  scales[4];      // 8 sub-block scales, 4-bit each
    uint8_t  qs[QK_K/2];    // 256 signed 4-bit quants as (q+8)
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// Tensor metadata
// ---------------------------------------------------------------------------

struct TensorInfo {
    std::string name;
    int         split_idx;
    size_t      file_offset;
    size_t      nbytes;
    ggml_type   type;
    int64_t     ne[GGML_MAX_DIMS];
    int         layer_idx;
};

static int extract_layer_idx(const std::string & name) {
    auto pos = name.find("blk.");
    if (pos == std::string::npos) return -1;
    return atoi(name.c_str() + pos + 4);
}

static bool is_weight_tensor(const std::string & name) {
    return name.find(".weight") != std::string::npos;
}

// ---------------------------------------------------------------------------
// mmap wrapper
// ---------------------------------------------------------------------------

struct MappedFile {
    void *  data = nullptr;
    size_t  size = 0;
    int     fd   = -1;

    bool open(const char * path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) < 0) { ::close(fd); fd = -1; return false; }
        size = (size_t)st.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; ::close(fd); fd = -1; return false; }
        madvise(data, size, MADV_SEQUENTIAL);
        return true;
    }

    void close() {
        if (data && size > 0) { munmap(data, size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
        size = 0;
    }

    const uint8_t * at(size_t offset) const {
        return (const uint8_t *)data + offset;
    }

    ~MappedFile() { close(); }
};

// ---------------------------------------------------------------------------
// Dequantization helper
// ---------------------------------------------------------------------------

static void dequant_to_f32(const void * src, ggml_type type, int64_t n_elements, float * dst) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, n_elements * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)src;
        for (int64_t i = 0; i < n_elements; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
        return;
    }
    ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    if (!tt.to_float) {
        memset(dst, 0, n_elements * sizeof(float));
        return;
    }
    tt.to_float(src, dst, n_elements);
}

// ---------------------------------------------------------------------------
// Quantization type parsing
// ---------------------------------------------------------------------------

static ggml_type parse_quant_type(const char * s) {
    if (!strcmp(s, "Q4_0"))   return GGML_TYPE_Q4_0;
    if (!strcmp(s, "Q5_0"))   return GGML_TYPE_Q5_0;
    if (!strcmp(s, "Q8_0"))   return GGML_TYPE_Q8_0;
    if (!strcmp(s, "Q2_K"))   return GGML_TYPE_Q2_K;
    if (!strcmp(s, "Q3_K_S")) return GGML_TYPE_Q3_K;
    if (!strcmp(s, "Q3_K_M")) return GGML_TYPE_Q3_K;
    if (!strcmp(s, "Q4_K_S")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "Q4_K_M")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "Q5_K_S")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "Q5_K_M")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "Q6_K"))   return GGML_TYPE_Q6_K;
    if (!strcmp(s, "QD4_K"))  return GGML_TYPE_QD4_K;
    if (!strcmp(s, "QD1_K"))  return GGML_TYPE_QD1_K;
    if (!strcmp(s, "F16"))    return GGML_TYPE_F16;
    if (!strcmp(s, "F32"))    return GGML_TYPE_F32;
    fprintf(stderr, "ERROR: unknown quantization type: '%s'\n", s);
    fprintf(stderr, "  Valid types: Q4_0 Q5_0 Q8_0 Q2_K Q3_K_S Q3_K_M Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K QD4_K QD1_K F16 F32\n");
    return GGML_TYPE_Q2_K;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --blurry <path> --sharp <path> --levels <N> [--quant <type>] --out-prefix <prefix> [options]\n\n"
        "  --blurry     Path to the blurry (low-quality) GGUF model\n"
        "  --sharp      Path to the sharp (high-quality) GGUF model (supports splits)\n"
        "  --levels     Number of delta correction levels to generate (1-8)\n"
        "  --quant      Quantization type for deltas (default: Q2_K)\n"
        "  --out-prefix Output prefix: produces <prefix>_1.gguf, <prefix>_2.gguf, ...\n"
        "  --layers     Layer range to process (e.g. 3-50, default: all)\n"
        "  --threads    Number of parallel threads (default: 4)\n"
        "  --sparse <t> Generate sparse delta with threshold t (e.g. 0.01 = keep blocks\n"
        "               with RMS >= 1%% of max). Outputs <prefix>_sparse_1.gguf alongside\n"
        "               the full delta. Sparse deltas store only significant blocks +\n"
        "               a block index, reducing file size and I/O by up to 80%%.\n",
        prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const char * blurry_path  = nullptr;
    const char * sharp_path   = nullptr;
    const char * out_prefix   = nullptr;
    const char * quant_str    = "Q2_K";
    int          n_levels     = 3;
    int          n_threads    = 4;
    int          layer_min    = 0;
    int          layer_max    = INT32_MAX;
    float        sparse_threshold = 0.0f; // 0 = disabled, >0 = sparse mode (e.g. 0.01)
    int          lowrank = 0;          // 0 = disabled, >0 = SVD rank for LoRA-style delta
    bool         analyze_only = false; // --analyze: print distribution stats, no output

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--blurry")     && i+1 < argc) blurry_path = argv[++i];
        else if (!strcmp(argv[i], "--sharp")      && i+1 < argc) sharp_path  = argv[++i];
        else if (!strcmp(argv[i], "--out-prefix") && i+1 < argc) out_prefix  = argv[++i];
        else if (!strcmp(argv[i], "--quant")      && i+1 < argc) quant_str   = argv[++i];
        else if (!strcmp(argv[i], "--levels")     && i+1 < argc) n_levels    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")    && i+1 < argc) n_threads   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparse")     && i+1 < argc) sparse_threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lowrank")    && i+1 < argc) lowrank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--analyze"))   analyze_only = true;
        else if (!strcmp(argv[i], "--layers")     && i+1 < argc) {
            sscanf(argv[++i], "%d-%d", &layer_min, &layer_max);
        }
        else { print_usage(argv[0]); return 1; }
    }
    if (!blurry_path || !sharp_path || (!out_prefix && !analyze_only)) { print_usage(argv[0]); return 1; }
    if (n_levels < 1 || n_levels > 8) { fprintf(stderr, "levels must be 1-8\n"); return 1; }
    if (n_threads < 1) n_threads = 1;

    const ggml_type delta_type = parse_quant_type(quant_str);
    fprintf(stderr, "Delta quantization: %s, levels: %d, threads: %d\n", quant_str, n_levels, n_threads);
    if (sparse_threshold > 0) {
        fprintf(stderr, "Sparse mode: threshold=%.4f (blocks below %.2f%% of max importance will be pruned)\n",
                sparse_threshold, sparse_threshold * 100.0f);
    }

    // ---- mmap models ----
    MappedFile blurry_mmap;
    if (!blurry_mmap.open(blurry_path)) { fprintf(stderr, "failed to mmap blurry\n"); return 1; }
    fprintf(stderr, "mmap'd blurry: %s (%.1f GB)\n", blurry_path, blurry_mmap.size / 1e9);

    gguf_init_params blurry_params = { .no_alloc = true, .ctx = nullptr };
    gguf_context * blurry_gguf = gguf_init_from_file(blurry_path, blurry_params);
    if (!blurry_gguf) { fprintf(stderr, "failed to open blurry GGUF\n"); return 1; }

    // Sharp splits
    std::vector<std::string> sharp_paths;
    sharp_paths.push_back(sharp_path);
    {
        std::string base = sharp_path;
        auto pos = base.find("-00001-of-");
        if (pos != std::string::npos) {
            std::string total_str = base.substr(pos + 10);
            int total = atoi(total_str.c_str());
            for (int s = 2; s <= total; s++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "-%05d-of-%s", s, total_str.c_str());
                sharp_paths.push_back(base.substr(0, pos) + buf);
            }
        }
    }

    std::vector<MappedFile>     sharp_mmaps(sharp_paths.size());
    std::vector<gguf_context *> sharp_ggufs;
    for (size_t si = 0; si < sharp_paths.size(); si++) {
        if (!sharp_mmaps[si].open(sharp_paths[si].c_str())) break;
        gguf_init_params p = { .no_alloc = true, .ctx = nullptr };
        gguf_context * sg = gguf_init_from_file(sharp_paths[si].c_str(), p);
        if (!sg) { sharp_mmaps[si].close(); break; }
        sharp_ggufs.push_back(sg);
        fprintf(stderr, "mmap'd sharp split: %s (%.1f GB)\n", sharp_paths[si].c_str(), sharp_mmaps[si].size / 1e9);
    }
    if (sharp_ggufs.empty()) { fprintf(stderr, "no sharp splits opened\n"); return 1; }

    // ---- build tensor indices ----
    auto build_index = [](const char * path, gguf_context * gguf_ctx, int split_idx) {
        std::unordered_map<std::string, TensorInfo> index;
        ggml_init_params ip = { .mem_size = 256*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * ctx = ggml_init(ip);
        gguf_init_params p2 = { .no_alloc = true, .ctx = &ctx };
        gguf_context * g2 = gguf_init_from_file(path, p2);
        if (g2) {
            size_t data_off = gguf_get_data_offset(gguf_ctx);
            int n = gguf_get_n_tensors(gguf_ctx);
            for (int i = 0; i < n; i++) {
                const char * tname = gguf_get_tensor_name(gguf_ctx, i);
                TensorInfo info;
                info.name        = tname;
                info.split_idx   = split_idx;
                info.file_offset = data_off + gguf_get_tensor_offset(gguf_ctx, i);
                info.type        = gguf_get_tensor_type(gguf_ctx, i);
                info.layer_idx   = extract_layer_idx(info.name);
                ggml_tensor * t  = ggml_get_tensor(ctx, tname);
                if (t) {
                    for (int d = 0; d < GGML_MAX_DIMS; d++) info.ne[d] = t->ne[d];
                    info.nbytes = ggml_nbytes(t);
                } else {
                    memset(info.ne, 0, sizeof(info.ne));
                    info.nbytes = 0;
                }
                index[info.name] = info;
            }
            gguf_free(g2);
        }
        ggml_free(ctx);
        return index;
    };

    auto blurry_index = build_index(blurry_path, blurry_gguf, 0);
    std::unordered_map<std::string, TensorInfo> sharp_index;
    for (int si = 0; si < (int)sharp_ggufs.size(); si++) {
        auto idx = build_index(sharp_paths[si].c_str(), sharp_ggufs[si], si);
        for (auto & [k, v] : idx) sharp_index[k] = v;
    }

    // ---- find shared tensors (sorted for deterministic output order) ----
    std::vector<std::string> shared_tensors;
    for (auto & [name, binfo] : blurry_index) {
        if (!is_weight_tensor(name)) continue;
        if (binfo.layer_idx < layer_min || binfo.layer_idx > layer_max) continue;
        if (sharp_index.count(name)) shared_tensors.push_back(name);
    }
    std::sort(shared_tensors.begin(), shared_tensors.end());
    const int n_tensors = (int)shared_tensors.size();
    fprintf(stderr, "Found %d shared weight tensors\n", n_tensors);

    std::string model_arch = "unknown";
    { int k = gguf_find_key(blurry_gguf, "general.architecture"); if (k >= 0) model_arch = gguf_get_val_str(blurry_gguf, k); }

    ggml_quantize_init(delta_type);

    // ==================================================================
    // Analyze-only mode: compute delta distribution statistics
    // ==================================================================
    if (analyze_only) {
        fprintf(stderr, "\n=== DELTA DISTRIBUTION ANALYSIS ===\n\n");

        // Global accumulators
        int64_t total_blocks = 0;
        int64_t total_zero_energy_blocks = 0;  // energy == 0
        int64_t total_zero_subblocks = 0;      // sub-block scale == 0
        int64_t total_subblocks = 0;
        int64_t quant_histogram[16] = {0};     // histogram of 4-bit quant values (0-15, where 8=zero)
        int64_t energy_histogram[256] = {0};
        int64_t scale_histogram[16] = {0};     // histogram of 4-bit sub-block scales
        int64_t total_bytes_dense = 0;
        int64_t total_bytes_block_sparse = 0;  // skip zero-energy blocks
        int64_t total_bytes_subblock_sparse = 0; // skip zero sub-blocks too
        int64_t total_moe_tensors = 0;
        int64_t total_non_moe_tensors = 0;

        int64_t t_start = ggml_time_us();

        for (int ti = 0; ti < n_tensors; ti++) {
            const auto & tname = shared_tensors[ti];
            const auto & binfo = blurry_index[tname];
            const auto & sinfo = sharp_index[tname];

            int64_t n_elements = 1;
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (sinfo.ne[d] > 0) n_elements *= sinfo.ne[d];
            }

            const int64_t n_per_row = sinfo.ne[0];
            const int64_t n_rows_total = n_elements / (n_per_row > 0 ? n_per_row : 1);
            const int64_t n_experts = (sinfo.ne[2] > 0) ? sinfo.ne[2] : 1;
            const int64_t elems_per_expert = n_elements / n_experts;
            const int64_t rows_per_expert = n_rows_total / n_experts;

            const size_t sharp_bpe  = sinfo.nbytes / n_experts;
            const size_t blurry_bpe = binfo.nbytes / n_experts;

            const int64_t blk_sz = (int64_t)ggml_blck_size(delta_type);
            const ggml_type eff_type = (blk_sz > 0 && n_per_row < blk_sz) ? GGML_TYPE_F32 : delta_type;

            if (eff_type != GGML_TYPE_QD4_K || n_elements == 0 || n_per_row == 0) {
                double elapsed = (ggml_time_us() - t_start) / 1e6;
                fprintf(stderr, "\r  [%d/%d] %.0fs (skip: %s)          ", ti+1, n_tensors, elapsed, tname.c_str());
                continue;
            }

            if (n_experts > 1) total_moe_tensors++; else total_non_moe_tensors++;

            const uint8_t * sharp_base  = sharp_mmaps[sinfo.split_idx].at(sinfo.file_offset);
            const uint8_t * blurry_base = blurry_mmap.at(binfo.file_offset);

            const size_t delta_bpe = ggml_row_size(eff_type, n_per_row) * rows_per_expert;
            const int64_t blocks_per_expert = elems_per_expert / QK_K;

            // Per-thread histograms to avoid contention
            std::vector<std::array<int64_t, 16>> thread_qhist(n_threads);
            std::vector<std::array<int64_t, 256>> thread_ehist(n_threads);
            std::vector<std::array<int64_t, 16>> thread_shist(n_threads);
            std::vector<int64_t> thread_zero_blocks(n_threads, 0);
            std::vector<int64_t> thread_zero_subblocks(n_threads, 0);
            for (auto & h : thread_qhist) h.fill(0);
            for (auto & h : thread_ehist) h.fill(0);
            for (auto & h : thread_shist) h.fill(0);

            #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
            for (int64_t ei = 0; ei < n_experts; ei++) {
                int tid = omp_get_thread_num();
                std::vector<float> sf(elems_per_expert);
                std::vector<float> ap(elems_per_expert);
                std::vector<float> re(elems_per_expert);
                std::vector<uint8_t> dq(delta_bpe);

                dequant_to_f32(sharp_base + ei * sharp_bpe, sinfo.type, elems_per_expert, sf.data());
                dequant_to_f32(blurry_base + ei * blurry_bpe, binfo.type, elems_per_expert, ap.data());

                for (int64_t i = 0; i < elems_per_expert; i++) re[i] = sf[i] - ap[i];

                ggml_quantize_chunk(eff_type, re.data(), dq.data(), 0, rows_per_expert, n_per_row, nullptr);

                // Analyze the quantized QD4_K blocks
                const block_qd4_k * blocks = (const block_qd4_k *)dq.data();
                for (int64_t b = 0; b < blocks_per_expert; b++) {
                    thread_ehist[tid][blocks[b].energy]++;
                    if (blocks[b].energy == 0) { thread_zero_blocks[tid]++; continue; }

                    // Analyze sub-block scales
                    for (int j = 0; j < 8; j++) {
                        uint8_t sc = (blocks[b].scales[j/2] >> (4*(j%2))) & 0xF;
                        thread_shist[tid][sc]++;
                        if (sc == 0) thread_zero_subblocks[tid]++;
                    }

                    // Analyze quant values
                    for (int q = 0; q < QK_K/2; q++) {
                        thread_qhist[tid][blocks[b].qs[q] & 0xF]++;
                        thread_qhist[tid][blocks[b].qs[q] >> 4]++;
                    }
                }
            }

            // Merge thread histograms
            int64_t tensor_blocks = blocks_per_expert * n_experts;
            int64_t tensor_zero_blocks = 0;
            int64_t tensor_zero_subblocks = 0;
            for (int t = 0; t < n_threads; t++) {
                for (int v = 0; v < 16; v++) { quant_histogram[v] += thread_qhist[t][v]; scale_histogram[v] += thread_shist[t][v]; }
                for (int v = 0; v < 256; v++) energy_histogram[v] += thread_ehist[t][v];
                tensor_zero_blocks += thread_zero_blocks[t];
                tensor_zero_subblocks += thread_zero_subblocks[t];
            }
            int64_t active_blocks = tensor_blocks - tensor_zero_blocks;
            int64_t tensor_subblocks = active_blocks * 8;

            total_blocks += tensor_blocks;
            total_zero_energy_blocks += tensor_zero_blocks;
            total_subblocks += tensor_subblocks;
            total_zero_subblocks += tensor_zero_subblocks;

            total_bytes_dense += tensor_blocks * 136;
            total_bytes_block_sparse += active_blocks * 136 + (tensor_blocks + 7) / 8; // bitmap + active blocks
            int64_t active_subblocks = tensor_subblocks - tensor_zero_subblocks;
            // per active block: 8 bytes header (d+energy+flags+scales) + 16 bytes per active sub-block + 1 byte sub-bitmap
            total_bytes_subblock_sparse += active_blocks * (8 + 1) + active_subblocks * 16 + (tensor_blocks + 7) / 8;

            double pct_zero_blk = 100.0 * tensor_zero_blocks / tensor_blocks;
            double pct_zero_sub = tensor_subblocks > 0 ? 100.0 * tensor_zero_subblocks / tensor_subblocks : 0;

            double elapsed = (ggml_time_us() - t_start) / 1e6;
            double eta = (elapsed / (ti + 1)) * (n_tensors - ti - 1);
            fprintf(stderr, "\r  [%d/%d] %.0fs ETA %.0fs  %s: %lld blocks, %.1f%% zero-block, %.1f%% zero-sub (%lld experts)       \n",
                    ti+1, n_tensors, elapsed, eta, tname.c_str(),
                    (long long)tensor_blocks, pct_zero_blk, pct_zero_sub, (long long)n_experts);
        }

        // Print summary
        fprintf(stderr, "\n=== SUMMARY ===\n");
        fprintf(stderr, "Tensors analyzed: %lld MoE + %lld non-MoE\n",
                (long long)total_moe_tensors, (long long)total_non_moe_tensors);
        fprintf(stderr, "Total blocks: %lld\n", (long long)total_blocks);
        fprintf(stderr, "Zero-energy blocks: %lld (%.1f%%)\n",
                (long long)total_zero_energy_blocks, 100.0 * total_zero_energy_blocks / total_blocks);
        fprintf(stderr, "Zero sub-blocks (in active blocks): %lld / %lld (%.1f%%)\n",
                (long long)total_zero_subblocks, (long long)total_subblocks,
                total_subblocks > 0 ? 100.0 * total_zero_subblocks / total_subblocks : 0.0);

        fprintf(stderr, "\nDense QD4_K:         %.2f GB\n", total_bytes_dense / 1e9);
        fprintf(stderr, "Block-sparse:        %.2f GB (%.1fx compression)\n",
                total_bytes_block_sparse / 1e9,
                total_bytes_dense > 0 ? (double)total_bytes_dense / total_bytes_block_sparse : 0);
        fprintf(stderr, "Sub-block-sparse:    %.2f GB (%.1fx compression)\n",
                total_bytes_subblock_sparse / 1e9,
                total_bytes_dense > 0 ? (double)total_bytes_dense / total_bytes_subblock_sparse : 0);

        // Entropy analysis of quant values
        fprintf(stderr, "\nQuant value histogram (0-15, 8=zero):\n");
        int64_t total_quants = 0;
        for (int v = 0; v < 16; v++) total_quants += quant_histogram[v];
        double entropy = 0;
        for (int v = 0; v < 16; v++) {
            double p = total_quants > 0 ? (double)quant_histogram[v] / total_quants : 0;
            if (p > 0) entropy -= p * log2(p);
            fprintf(stderr, "  [%2d] %12lld (%.1f%%)\n", v, (long long)quant_histogram[v], 100.0*p);
        }
        fprintf(stderr, "Entropy: %.2f bits/value (vs 4.0 fixed) → potential %.0f%% reduction\n",
                entropy, 100.0 * (1.0 - entropy / 4.0));

        fprintf(stderr, "\nSub-block scale histogram (0-15):\n");
        int64_t total_scales = 0;
        for (int v = 0; v < 16; v++) total_scales += scale_histogram[v];
        for (int v = 0; v < 16; v++) {
            double p = total_scales > 0 ? (double)scale_histogram[v] / total_scales : 0;
            fprintf(stderr, "  [%2d] %12lld (%.1f%%)\n", v, (long long)scale_histogram[v], 100.0*p);
        }

        fprintf(stderr, "\nEnergy histogram (top 10 + zero):\n");
        fprintf(stderr, "  [  0] %12lld (%.1f%%) ← fully zero blocks\n",
                (long long)energy_histogram[0], 100.0 * energy_histogram[0] / total_blocks);
        // Find top 10 non-zero energy values
        std::vector<std::pair<int64_t, int>> energy_sorted;
        for (int v = 1; v < 256; v++) {
            if (energy_histogram[v] > 0) energy_sorted.push_back({energy_histogram[v], v});
        }
        std::sort(energy_sorted.rbegin(), energy_sorted.rend());
        for (int i = 0; i < (int)energy_sorted.size() && i < 10; i++) {
            fprintf(stderr, "  [%3d] %12lld (%.1f%%)\n",
                    energy_sorted[i].second, (long long)energy_sorted[i].first,
                    100.0 * energy_sorted[i].first / total_blocks);
        }

        // Estimate combined compression with entropy coding
        double ent_bytes_per_block = 8.0 + (entropy / 4.0) * 128.0; // header + compressed quants
        double ent_total = 0;
        int64_t active = total_blocks - total_zero_energy_blocks;
        ent_total += active * ent_bytes_per_block + (total_blocks + 7) / 8; // bitmap
        fprintf(stderr, "\nWith entropy-coded quants: %.2f GB (%.1fx vs dense)\n",
                ent_total / 1e9, total_bytes_dense / ent_total);

        return 0;
    }

    // ==================================================================
    // Phase 1: Write GGUF headers (metadata + tensor descriptors, no data)
    // ==================================================================
    // We need to know tensor shapes upfront to write the header.
    // Then we'll append raw data as tensors are processed.

    struct LevelFile {
        char    path[512];
        FILE *  fp = nullptr;
        size_t  data_offset = 0;
        double  total_rmse = 0.0;
        int     count = 0;
    };
    std::vector<LevelFile> level_files(n_levels);

    for (int level = 0; level < n_levels; level++) {
        gguf_context * hdr = gguf_init_empty();
        gguf_set_val_str(hdr, "general.architecture", model_arch.c_str());
        gguf_set_val_str(hdr, "general.type",         "delta_correction");
        gguf_set_val_i32(hdr, "bs.delta.level",       level + 1);
        gguf_set_val_i32(hdr, "bs.delta.parent_level", level);
        gguf_set_val_str(hdr, "bs.delta.quant_type",  quant_str);

        // Create a ggml context just for tensor descriptors
        const size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 4*1024*1024;
        fprintf(stderr, "  Level %d: allocating context (%zu bytes for %d tensors)...\n",
                level + 1, ctx_size, n_tensors);
        ggml_init_params ip = { .mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            fprintf(stderr, "  ERROR: ggml_init failed for level %d\n", level + 1);
            gguf_free(hdr);
            continue;
        }

        int n_added = 0;
        for (int ti = 0; ti < n_tensors; ti++) {
            auto & sinfo = sharp_index[shared_tensors[ti]];
            int64_t ne0 = sinfo.ne[0] > 0 ? sinfo.ne[0] : 1;
            int64_t ne1 = sinfo.ne[1] > 0 ? sinfo.ne[1] : 1;
            int64_t ne2 = sinfo.ne[2] > 0 ? sinfo.ne[2] : 1;
            int64_t ne3 = sinfo.ne[3] > 0 ? sinfo.ne[3] : 1;

            const int64_t blk_sz = (int64_t)ggml_blck_size(delta_type);
            const ggml_type eff_type = (blk_sz > 0 && ne0 < blk_sz) ? GGML_TYPE_F32 : delta_type;
            ggml_tensor * t = ggml_new_tensor_4d(ctx, eff_type, ne0, ne1, ne2, ne3);
            if (t) {
                ggml_set_name(t, shared_tensors[ti].c_str());
                gguf_add_tensor(hdr, t);
                n_added++;
            } else {
                fprintf(stderr, "  WARNING: failed to create tensor %d/%d: %s [%lld,%lld,%lld,%lld]\n",
                        ti, n_tensors, shared_tensors[ti].c_str(),
                        (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);
            }
        }
        fprintf(stderr, "  Level %d: added %d/%d tensors to header\n", level + 1, n_added, n_tensors);

        snprintf(level_files[level].path, sizeof(level_files[level].path),
                 "%s_%d.gguf", out_prefix, level + 1);

        fprintf(stderr, "  Writing header to %s ...\n", level_files[level].path);
        fflush(stderr);

        // Write header ONLY (metadata + tensor descriptors, no data).
        // We append raw tensor data ourselves in the streaming phase.
        gguf_write_to_file(hdr, level_files[level].path, /*only_meta=*/true);

        // Compute where tensor data starts (= size of the metadata we just wrote).
        // gguf_get_data_offset() doesn't work for gguf_init_empty() contexts,
        // so use gguf_get_meta_size() which computes it from the header structure.
        level_files[level].data_offset = gguf_get_meta_size(hdr);

        // Re-open for appending data
        level_files[level].fp = fopen(level_files[level].path, "r+b");
        if (level_files[level].fp) {
            fseek(level_files[level].fp, (long)level_files[level].data_offset, SEEK_SET);
            fprintf(stderr, "  Opened %s for streaming write (data offset=%zu)\n",
                    level_files[level].path, level_files[level].data_offset);
        } else {
            fprintf(stderr, "  ERROR: failed to open %s for writing\n", level_files[level].path);
        }

        gguf_free(hdr);
        ggml_free(ctx);
    }

    // ==================================================================
    // Phase 2: Process tensors sequentially, write data as we go
    // ==================================================================
    // Each tensor is processed with OpenMP parallelism across experts,
    // then the result is immediately written to all level files.
    // No buffering of all tensors in RAM.

    int64_t t_start = ggml_time_us();
    std::mutex write_mtx;  // protects file writes

    // Sparse mode: accumulate per-tensor, per-expert block indices for Phase 2.5
    struct SparseInfo {
        std::string name;
        std::vector<std::vector<uint16_t>> per_expert_indices; // [n_experts][n_blocks_stored_for_expert]
        int64_t n_blocks_total;              // total blocks per expert
        int64_t n_per_row;                   // ne00
        int64_t rows_per_expert;
        int64_t n_experts;
    };
    std::vector<SparseInfo> sparse_infos;

    for (int ti = 0; ti < n_tensors; ti++) {
        const auto & tname = shared_tensors[ti];
        const auto & binfo = blurry_index[tname];
        const auto & sinfo = sharp_index[tname];

        int64_t n_elements = 1;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (sinfo.ne[d] > 0) n_elements *= sinfo.ne[d];
        }

        const int64_t n_per_row = sinfo.ne[0];
        const int64_t n_rows_total = n_elements / (n_per_row > 0 ? n_per_row : 1);
        const int64_t n_experts = (sinfo.ne[2] > 0) ? sinfo.ne[2] : 1;
        const int64_t elems_per_expert = n_elements / n_experts;
        const int64_t rows_per_expert = n_rows_total / n_experts;

        const size_t sharp_bpe  = sinfo.nbytes / n_experts;
        const size_t blurry_bpe = binfo.nbytes / n_experts;

        // Use F32 for tensors too small for the target quant block size (e.g. scalar layer_output_scale)
        const int64_t blk_size_check = (int64_t)ggml_blck_size(delta_type);
        const ggml_type effective_delta_type = (blk_size_check > 0 && n_per_row < blk_size_check) ? GGML_TYPE_F32 : delta_type;

        const size_t delta_bpe  = ggml_row_size(effective_delta_type, n_per_row) * rows_per_expert;
        const size_t delta_total = delta_bpe * n_experts;

        if (n_elements == 0 || n_per_row == 0) {
            // Write zeros for empty tensors
            for (int level = 0; level < n_levels; level++) {
                if (!level_files[level].fp) continue;
                std::vector<uint8_t> zeros(delta_total, 0);
                fwrite(zeros.data(), 1, delta_total, level_files[level].fp);
                size_t pad = (32 - (delta_total % 32)) % 32;
                if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, level_files[level].fp); }
            }
            double elapsed = (ggml_time_us() - t_start) / 1e6;
            double eta = (ti > 0) ? (elapsed / ti) * (n_tensors - ti) : 0;
            fprintf(stderr, "\r  [%d/%d] %.0fs ETA %.0fs  (skip: %s)          ", ti+1, n_tensors, elapsed, eta, tname.c_str());
            continue;
        }

        const uint8_t * sharp_base  = sharp_mmaps[sinfo.split_idx].at(sinfo.file_offset);
        const uint8_t * blurry_base = blurry_mmap.at(binfo.file_offset);

        // Per-level output buffers for this tensor (written immediately after)
        std::vector<std::vector<uint8_t>> level_deltas(n_levels);
        for (int l = 0; l < n_levels; l++) level_deltas[l].resize(delta_total);

        std::vector<std::vector<double>> level_mse(n_levels, std::vector<double>(n_experts, 0.0));

        // Block importance analysis for sparse mode (level 1 only)
        // Per-expert adaptive threshold: each expert decides which blocks are
        // significant relative to ITS OWN max importance. A block survives if
        // ANY expert considers it significant (union). This preserves niche
        // knowledge in specialized experts that have small but meaningful deltas.
        const int64_t blk_size = (int64_t)ggml_blck_size(delta_type);
        const int64_t n_blocks_per_expert = (blk_size > 0 && elems_per_expert > 0) ? elems_per_expert / blk_size : 0;
        // Per-expert, per-block importance [n_experts][n_blocks]
        std::vector<std::vector<float>> expert_block_importance(n_experts);
        for (int64_t ei = 0; ei < n_experts; ei++) expert_block_importance[ei].resize(n_blocks_per_expert, 0.0f);

        // Parallel across experts
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int64_t ei = 0; ei < n_experts; ei++) {
            std::vector<float> sf(elems_per_expert);
            std::vector<float> ap(elems_per_expert);
            std::vector<float> re(elems_per_expert);
            std::vector<uint8_t> dq(delta_bpe);
            std::vector<float> df(elems_per_expert);

            dequant_to_f32(sharp_base + ei * sharp_bpe, sinfo.type, elems_per_expert, sf.data());
            dequant_to_f32(blurry_base + ei * blurry_bpe, binfo.type, elems_per_expert, ap.data());

            for (int level = 0; level < n_levels; level++) {
                for (int64_t i = 0; i < elems_per_expert; i++) re[i] = sf[i] - ap[i];

                ggml_quantize_chunk(effective_delta_type, re.data(), dq.data(), 0, rows_per_expert, n_per_row, nullptr);

                memcpy(level_deltas[level].data() + ei * delta_bpe, dq.data(), delta_bpe);

                // Block importance analysis for level 1 sparse output (per-expert)
                if (level == 0 && sparse_threshold > 0 && n_blocks_per_expert > 0) {
                    for (int64_t b = 0; b < n_blocks_per_expert; b++) {
                        float sum_sq = 0.0f;
                        for (int64_t j = 0; j < blk_size; j++) {
                            float v = re[b * blk_size + j];
                            sum_sq += v * v;
                        }
                        expert_block_importance[ei][b] = sqrtf(sum_sq / (float)blk_size);
                    }
                }

                dequant_to_f32(dq.data(), delta_type, elems_per_expert, df.data());
                double mse = 0.0;
                for (int64_t i = 0; i < elems_per_expert; i++) {
                    ap[i] += df[i];
                    double e = (double)sf[i] - (double)ap[i];
                    mse += e * e;
                }
                level_mse[level][ei] = mse;
            }
        }

        // Write immediately to each level's file
        for (int level = 0; level < n_levels; level++) {
            if (!level_files[level].fp) continue;

            fwrite(level_deltas[level].data(), 1, delta_total, level_files[level].fp);
            size_t pad = (32 - (delta_total % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, level_files[level].fp); }

            double total_mse = 0.0;
            for (int64_t ei = 0; ei < n_experts; ei++) total_mse += level_mse[level][ei];
            float rmse = (float)sqrt(total_mse / (double)n_elements);
            level_files[level].total_rmse += rmse;
            level_files[level].count++;
        }

        // Build sparse index for this tensor (level 1)
        // Per-expert independent thresholds: each expert keeps only its own
        // significant blocks. No union — MoE experts have independent patterns.
        if (sparse_threshold > 0 && n_blocks_per_expert > 0) {
            SparseInfo si;
            si.name = tname;
            si.n_blocks_total = n_blocks_per_expert;
            si.n_per_row = n_per_row;
            si.rows_per_expert = rows_per_expert;
            si.n_experts = n_experts;
            si.per_expert_indices.resize(n_experts);

            int64_t total_kept = 0, total_possible = 0;
            for (int64_t ei = 0; ei < n_experts; ei++) {
                // Find this expert's max importance
                float emax = 0.0f;
                for (int64_t b = 0; b < n_blocks_per_expert; b++) {
                    if (expert_block_importance[ei][b] > emax)
                        emax = expert_block_importance[ei][b];
                }
                float thresh = emax * sparse_threshold;

                // Keep blocks above threshold for this expert
                for (int64_t b = 0; b < n_blocks_per_expert; b++) {
                    if (expert_block_importance[ei][b] >= thresh) {
                        si.per_expert_indices[ei].push_back((uint16_t)b);
                    }
                }
                total_kept += (int64_t)si.per_expert_indices[ei].size();
                total_possible += n_blocks_per_expert;
            }

            sparse_infos.push_back(std::move(si));

            double pruned_pct = 100.0 * (1.0 - (double)total_kept / (double)total_possible);
            fprintf(stderr, "  [sparse] %s: avg %.0f%% pruned across %lld experts\n",
                    tname.c_str(), pruned_pct, (long long)n_experts);
        }

        double elapsed = (ggml_time_us() - t_start) / 1e6;
        double eta = (elapsed / (ti + 1)) * (n_tensors - ti - 1);
        if (sparse_threshold > 0 && n_blocks_per_expert > 0 && !sparse_infos.empty()) {
            fprintf(stderr, "  [%d/%d] %.0fs ETA %.0fs\n",
                    ti+1, n_tensors, elapsed, eta);
        } else {
            fprintf(stderr, "\r  [%d/%d] %.0fs ETA %.0fs  (%s: %.1fM elems, %lld experts)          ",
                    ti+1, n_tensors, elapsed, eta, tname.c_str(),
                    n_elements / 1e6, (long long)n_experts);
        }
    }

    // ==================================================================
    // Phase 2.5: Generate sparse delta GGUF (if --sparse was given)
    // ==================================================================
    if (sparse_threshold > 0 && !sparse_infos.empty() && level_files[0].fp) {
        fclose(level_files[0].fp);
        level_files[0].fp = nullptr;

        fprintf(stderr, "\n\nPhase 2.5: Generating sparse delta GGUF...\n");

        // Open the full level-1 delta we just wrote for reading
        FILE * full_fp = fopen(level_files[0].path, "rb");
        if (!full_fp) {
            fprintf(stderr, "  ERROR: cannot reopen %s for sparse conversion\n", level_files[0].path);
            goto skip_sparse;
        }

        char sparse_path[512];
        snprintf(sparse_path, sizeof(sparse_path), "%s_sparse_1.gguf", out_prefix);

        // Build GGUF header with per-expert sparse tensors
        gguf_context * shdr = gguf_init_empty();
        gguf_set_val_str(shdr, "general.architecture", model_arch.c_str());
        gguf_set_val_str(shdr, "general.type", "delta_correction_sparse");
        gguf_set_val_i32(shdr, "bs.delta.level", 1);
        gguf_set_val_str(shdr, "bs.delta.quant_type", quant_str);
        gguf_set_val_bool(shdr, "bs.delta.sparse", true);
        gguf_set_val_f32(shdr, "bs.delta.sparse_threshold", sparse_threshold);
        gguf_set_val_bool(shdr, "bs.delta.per_expert_index", true);

        // For per-expert sparse: use the MAX blocks stored across all experts
        // for the packed tensor shape (experts with fewer blocks are zero-padded).
        const size_t sctx_size = sparse_infos.size() * 2 * ggml_tensor_overhead() + 16*1024*1024;
        ggml_init_params sip = { .mem_size = sctx_size, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * sctx = ggml_init(sip);

        for (auto & si : sparse_infos) {
            // Find max blocks stored across experts (for uniform tensor shape)
            int64_t max_stored = 0;
            for (int64_t ei = 0; ei < si.n_experts; ei++) {
                max_stored = std::max(max_stored, (int64_t)si.per_expert_indices[ei].size());
            }
            if (max_stored == 0) max_stored = 1;

            // Packed delta tensor: [max_stored * block_elems, 1, n_experts]
            int64_t packed_elems = max_stored * ggml_blck_size(delta_type);
            ggml_tensor * td = ggml_new_tensor_3d(sctx, delta_type,
                packed_elems, 1, si.n_experts);
            ggml_set_name(td, si.name.c_str());
            gguf_add_tensor(shdr, td);

            // Per-expert index tensor: [max_stored, n_experts] — uint16 block positions
            // Unused slots filled with 0xFFFF (sentinel).
            std::string idx_name = si.name + ".idx";
            ggml_tensor * ti_t = ggml_new_tensor_2d(sctx, GGML_TYPE_I16,
                max_stored, si.n_experts);
            ggml_set_name(ti_t, idx_name.c_str());
            gguf_add_tensor(shdr, ti_t);

            // Metadata
            char key[256];
            snprintf(key, sizeof(key), "bs.delta.%s.n_blocks_total", si.name.c_str());
            gguf_set_val_i32(shdr, key, (int32_t)si.n_blocks_total);
            snprintf(key, sizeof(key), "bs.delta.%s.max_blocks_stored", si.name.c_str());
            gguf_set_val_i32(shdr, key, (int32_t)max_stored);
        }

        gguf_write_to_file(shdr, sparse_path, true);
        size_t sparse_data_offset = gguf_get_meta_size(shdr);
        gguf_free(shdr);
        ggml_free(sctx);

        // Write sparse data (per-expert packed blocks + per-expert indices)
        FILE * sfp = fopen(sparse_path, "r+b");
        if (!sfp) { fprintf(stderr, "  ERROR: cannot open %s for writing\n", sparse_path); fclose(full_fp); goto skip_sparse; }
        fseek(sfp, (long)sparse_data_offset, SEEK_SET);

        size_t full_data_offset = level_files[0].data_offset;
        size_t full_cursor = full_data_offset;
        size_t total_sparse_bytes = 0;

        for (size_t si_idx = 0; si_idx < sparse_infos.size(); si_idx++) {
            auto & si = sparse_infos[si_idx];
            const size_t block_bytes = ggml_type_size(delta_type);
            const size_t full_expert_bytes = ggml_row_size(delta_type, si.n_per_row) * si.rows_per_expert;

            // Find max stored for this tensor
            int64_t max_stored = 0;
            for (int64_t ei = 0; ei < si.n_experts; ei++) {
                max_stored = std::max(max_stored, (int64_t)si.per_expert_indices[ei].size());
            }
            if (max_stored == 0) max_stored = 1;

            const size_t packed_expert_bytes = max_stored * block_bytes;
            std::vector<uint8_t> full_expert(full_expert_bytes);
            std::vector<uint8_t> packed(packed_expert_bytes, 0);

            // Write packed data for each expert (zero-padded to max_stored)
            for (int64_t ei = 0; ei < si.n_experts; ei++) {
                fseek(full_fp, (long)(full_cursor + ei * full_expert_bytes), SEEK_SET);
                fread(full_expert.data(), 1, full_expert_bytes, full_fp);

                memset(packed.data(), 0, packed_expert_bytes);
                auto & idx = si.per_expert_indices[ei];
                for (size_t bi = 0; bi < idx.size(); bi++) {
                    memcpy(packed.data() + bi * block_bytes,
                           full_expert.data() + idx[bi] * block_bytes,
                           block_bytes);
                }
                fwrite(packed.data(), 1, packed_expert_bytes, sfp);
            }

            // Pad packed tensor
            size_t total_packed = packed_expert_bytes * si.n_experts;
            size_t pad = (32 - (total_packed % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, sfp); }

            // Write per-expert index arrays (padded to max_stored with 0xFFFF sentinel)
            for (int64_t ei = 0; ei < si.n_experts; ei++) {
                auto & idx = si.per_expert_indices[ei];
                fwrite(idx.data(), sizeof(uint16_t), idx.size(), sfp);
                // Pad remaining with sentinel
                size_t remaining = max_stored - idx.size();
                if (remaining > 0) {
                    std::vector<uint16_t> sentinel(remaining, 0xFFFF);
                    fwrite(sentinel.data(), sizeof(uint16_t), remaining, sfp);
                }
            }
            size_t idx_total = max_stored * si.n_experts * sizeof(uint16_t);
            pad = (32 - (idx_total % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, sfp); }

            // Advance past this tensor in the full file
            size_t full_tensor_bytes = full_expert_bytes * si.n_experts;
            size_t full_pad = (32 - (full_tensor_bytes % 32)) % 32;
            full_cursor += full_tensor_bytes + full_pad;
            total_sparse_bytes += total_packed + idx_total;

            // Stats
            int64_t total_kept = 0;
            for (int64_t ei = 0; ei < si.n_experts; ei++) total_kept += si.per_expert_indices[ei].size();
            double avg_pruned = 100.0 * (1.0 - (double)total_kept / (si.n_blocks_total * si.n_experts));
            fprintf(stderr, "  [%zu/%zu] %s: avg %.0f%% pruned (%lld experts, max %lld blocks/expert)\n",
                    si_idx + 1, sparse_infos.size(), si.name.c_str(),
                    avg_pruned, (long long)si.n_experts, (long long)max_stored);
        }

        fclose(sfp);
        fclose(full_fp);

        struct stat sst;
        stat(sparse_path, &sst);
        struct stat fst;
        stat(level_files[0].path, &fst);
        fprintf(stderr, "  Sparse delta: %s (%.1f GB, %.0f%% of full %.1f GB)\n",
                sparse_path, sst.st_size / 1e9,
                100.0 * sst.st_size / fst.st_size, fst.st_size / 1e9);
    }
skip_sparse:

    // ==================================================================
    // Phase 2.7: Generate low-rank (LoRA-style) delta GGUF
    // ==================================================================
#ifdef BS_HAVE_LAPACK
    if (lowrank > 0 && level_files[0].fp) {
        // Close level-1 file first (we'll re-read it)
        fclose(level_files[0].fp);
        level_files[0].fp = nullptr;

        fprintf(stderr, "\n\nPhase 2.7: Generating low-rank delta GGUF (rank=%d)...\n", lowrank);

        char lr_path[512];
        snprintf(lr_path, sizeof(lr_path), "%s_lowrank_%d.gguf", out_prefix, lowrank);

        // Build GGUF header
        gguf_context * lhdr = gguf_init_empty();
        gguf_set_val_str(lhdr, "general.architecture", model_arch.c_str());
        gguf_set_val_str(lhdr, "general.type", "delta_correction_lowrank");
        gguf_set_val_i32(lhdr, "bs.delta.rank", lowrank);

        // Create tensor metadata context
        const size_t lctx_size = shared_tensors.size() * 4 * ggml_tensor_overhead() + 16*1024*1024;
        ggml_init_params lip = { .mem_size = lctx_size, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * lctx = ggml_init(lip);

        // Pre-scan: create tensor entries for all 2D weight tensors
        struct LREntry {
            std::string name;
            int64_t ne00, ne01; // original weight dims
            int split_idx;
            size_t sharp_offset, blurry_offset;
            size_t sharp_bpe, blurry_bpe;
            ggml_type sharp_type, blurry_type;
        };
        std::vector<LREntry> lr_entries;

        for (int ti = 0; ti < n_tensors; ti++) {
            const auto & tname = shared_tensors[ti];
            const auto & binfo = blurry_index[tname];
            const auto & sinfo = sharp_index[tname];

            if (sinfo.ne[1] <= 1) continue; // skip 1D tensors

            int64_t ne00 = sinfo.ne[0];
            int64_t ne01 = sinfo.ne[1];
            int R = std::min(lowrank, (int)std::min(ne00, ne01));

            // U: [ne01, R] in FP16
            ggml_tensor * tu = ggml_new_tensor_2d(lctx, GGML_TYPE_F16, R, ne01);
            std::string u_name = tname + ".U";
            ggml_set_name(tu, u_name.c_str());
            gguf_add_tensor(lhdr, tu);

            // V: [ne00, R] in FP16
            ggml_tensor * tv = ggml_new_tensor_2d(lctx, GGML_TYPE_F16, R, ne00);
            std::string v_name = tname + ".V";
            ggml_set_name(tv, v_name.c_str());
            gguf_add_tensor(lhdr, tv);

            int64_t n_experts = (sinfo.ne[2] > 0) ? sinfo.ne[2] : 1;
            lr_entries.push_back({tname, ne00, ne01, sinfo.split_idx,
                                  sinfo.file_offset, binfo.file_offset,
                                  sinfo.nbytes / (size_t)n_experts,
                                  binfo.nbytes / (size_t)n_experts,
                                  sinfo.type, binfo.type});
        }

        gguf_write_to_file(lhdr, lr_path, true);
        size_t lr_data_offset = gguf_get_meta_size(lhdr);
        gguf_free(lhdr);
        ggml_free(lctx);

        FILE * lfp = fopen(lr_path, "r+b");
        if (!lfp) { fprintf(stderr, "ERROR: cannot open %s\n", lr_path); goto skip_lowrank; }
        fseek(lfp, (long)lr_data_offset, SEEK_SET);

        int64_t lr_t_start = ggml_time_us();
        for (size_t ei = 0; ei < lr_entries.size(); ei++) {
            auto & e = lr_entries[ei];
            int M = (int)e.ne01; // rows (output dim)
            int N = (int)e.ne00; // cols (input dim)
            int R = std::min(lowrank, std::min(M, N));
            int64_t elems = (int64_t)M * N;

            // Dequantize sharp and blurry to f32
            std::vector<float> sf(elems), bf(elems), delta(elems);
            dequant_to_f32(sharp_mmaps[e.split_idx].at(e.sharp_offset), e.sharp_type, elems, sf.data());
            dequant_to_f32(blurry_mmap.at(e.blurry_offset), e.blurry_type, elems, bf.data());

            for (int64_t i = 0; i < elems; i++) delta[i] = sf[i] - bf[i];

            // LAPACK SVD: delta (M×N row-major) = U * diag(S) * V^T
            // LAPACK expects column-major, so we transpose: pass N×M and swap U/V
            // Actually: row-major M×N = column-major N×M (transpose)
            // sgesvd on column-major N×M: A = U_col(N×K) * S(K) * Vt_col(K×M)
            // where K = min(M,N)
            int K = std::min(M, N);
            std::vector<float> S(K);
            std::vector<float> U_col(N * K);  // N×K (left singular vectors of transposed)
            std::vector<float> Vt_col(K * M); // K×M (right singular vectors of transposed)

            // Query optimal workspace
            int lwork = -1;
            float work_query;
            int info;
            sgesvd_("S", "S", &N, &M, delta.data(), &N, S.data(),
                    U_col.data(), &N, Vt_col.data(), &K, &work_query, &lwork, &info);
            lwork = (int)work_query;
            std::vector<float> work(lwork);

            // Compute SVD
            sgesvd_("S", "S", &N, &M, delta.data(), &N, S.data(),
                    U_col.data(), &N, Vt_col.data(), &K, work.data(), &lwork, &info);

            if (info != 0) {
                fprintf(stderr, "  WARNING: sgesvd failed for %s (info=%d), writing zeros\n",
                        e.name.c_str(), info);
            }

            // Reconstruct: for row-major M×N:
            // Original = U_orig(M×K) * diag(S) * V_orig(K×N)
            // Since we passed transposed: U_col is V_orig^T, Vt_col is U_orig^T
            // So: U_orig(M×K) = Vt_col^T and V_orig(N×K) = U_col
            // Truncate to rank R and fold S into U:
            // U_final(M×R) = Vt_col^T[:, :R] * diag(S[:R])
            // V_final(N×R) = U_col[:, :R]

            // Compute reconstruction error
            double full_norm = 0, trunc_err = 0;
            for (int i = 0; i < K; i++) full_norm += (double)S[i] * S[i];
            for (int i = R; i < K; i++) trunc_err += (double)S[i] * S[i];
            double retention = full_norm > 0 ? 1.0 - trunc_err / full_norm : 1.0;

            // Write U tensor: [R, ne01] in FP16 (ne01 rows of R elements)
            // U_final[row][r] = Vt_col[r * M + row] * S[r]
            std::vector<ggml_fp16_t> u_fp16(M * R);
            for (int row = 0; row < M; row++) {
                for (int r = 0; r < R; r++) {
                    u_fp16[row * R + r] = ggml_fp32_to_fp16(Vt_col[r * M + row] * S[r]);
                }
            }
            fwrite(u_fp16.data(), sizeof(ggml_fp16_t), M * R, lfp);
            size_t u_bytes = M * R * sizeof(ggml_fp16_t);
            size_t pad = (32 - (u_bytes % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, lfp); }

            // Write V tensor: [R, ne00] in FP16 (ne00 rows of R elements)
            // V_final[col][r] = U_col[col * K + r]  (but K may > R, take first R)
            // Wait, U_col is N×K column-major → U_col[col][k] = U_col[k * ??? ]
            // Actually U_col is stored column-major N×K: element [i,j] = U_col[j*N + i]
            // V_final[col][r] = U_col[r * N + col]  for r < R, col < N
            std::vector<ggml_fp16_t> v_fp16(N * R);
            for (int col = 0; col < N; col++) {
                for (int r = 0; r < R; r++) {
                    v_fp16[col * R + r] = ggml_fp32_to_fp16(U_col[r * N + col]);
                }
            }
            fwrite(v_fp16.data(), sizeof(ggml_fp16_t), N * R, lfp);
            size_t v_bytes = N * R * sizeof(ggml_fp16_t);
            pad = (32 - (v_bytes % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, lfp); }

            double elapsed = (ggml_time_us() - lr_t_start) / 1e6;
            double eta = (ei > 0) ? (elapsed / ei) * (lr_entries.size() - ei) : 0;
            fprintf(stderr, "  [%zu/%zu] %s: [%d,%d] rank=%d retention=%.1f%% S[0]=%.3f S[%d]=%.3f (%.0fs ETA %.0fs)\n",
                    ei + 1, lr_entries.size(), e.name.c_str(), M, N, R,
                    retention * 100.0, S[0], R - 1, S[std::min(R - 1, K - 1)],
                    elapsed, eta);
        }

        fclose(lfp);

        struct stat lrst;
        stat(lr_path, &lrst);
        fprintf(stderr, "  Low-rank delta: %s (%.1f MB, rank=%d)\n",
                lr_path, lrst.st_size / 1e6, lowrank);
    }
skip_lowrank:
#endif // BS_HAVE_LAPACK

    // ==================================================================
    // Phase 3: Close files and report
    // ==================================================================
    fprintf(stderr, "\n\nDone! Output files:\n");
    for (int level = 0; level < n_levels; level++) {
        if (level_files[level].fp) {
            fclose(level_files[level].fp);
            // Get file size
            struct stat st;
            stat(level_files[level].path, &st);
            fprintf(stderr, "  Level %d: %s (%.1f GB, %d tensors, avg RMSE=%.6f)\n",
                    level + 1, level_files[level].path, st.st_size / 1e9,
                    level_files[level].count,
                    level_files[level].count > 0 ? level_files[level].total_rmse / level_files[level].count : 0.0);
        }
    }

    double total_time = (ggml_time_us() - t_start) / 1e6;
    fprintf(stderr, "Total time: %.1fs\n", total_time);

    ggml_quantize_free();
    for (auto * g : sharp_ggufs) gguf_free(g);
    gguf_free(blurry_gguf);

    return 0;
}
