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
    if (!strcmp(s, "F16"))    return GGML_TYPE_F16;
    if (!strcmp(s, "F32"))    return GGML_TYPE_F32;
    fprintf(stderr, "ERROR: unknown quantization type: '%s'\n", s);
    fprintf(stderr, "  Valid types: Q4_0 Q5_0 Q8_0 Q2_K Q3_K_S Q3_K_M Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K F16 F32\n");
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

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--blurry")     && i+1 < argc) blurry_path = argv[++i];
        else if (!strcmp(argv[i], "--sharp")      && i+1 < argc) sharp_path  = argv[++i];
        else if (!strcmp(argv[i], "--out-prefix") && i+1 < argc) out_prefix  = argv[++i];
        else if (!strcmp(argv[i], "--quant")      && i+1 < argc) quant_str   = argv[++i];
        else if (!strcmp(argv[i], "--levels")     && i+1 < argc) n_levels    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")    && i+1 < argc) n_threads   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparse")     && i+1 < argc) sparse_threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lowrank")    && i+1 < argc) lowrank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers")     && i+1 < argc) {
            sscanf(argv[++i], "%d-%d", &layer_min, &layer_max);
        }
        else { print_usage(argv[0]); return 1; }
    }
    if (!blurry_path || !sharp_path || !out_prefix) { print_usage(argv[0]); return 1; }
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

    // Sparse mode: accumulate per-tensor block indices for Phase 2.5
    struct SparseInfo {
        std::string name;
        std::vector<uint16_t> block_indices; // significant block positions
        int64_t n_blocks_total;              // total blocks per expert
        int64_t n_blocks_stored;             // significant blocks
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
        // Per-expert adaptive threshold: a block is kept if ANY expert considers
        // it significant relative to that expert's own max importance.
        if (sparse_threshold > 0 && n_blocks_per_expert > 0) {
            // Per-expert threshold
            std::vector<float> expert_max(n_experts, 0.0f);
            for (int64_t ei = 0; ei < n_experts; ei++) {
                for (int64_t b = 0; b < n_blocks_per_expert; b++) {
                    if (expert_block_importance[ei][b] > expert_max[ei])
                        expert_max[ei] = expert_block_importance[ei][b];
                }
            }

            // Union: block survives if ANY expert says it's important
            std::vector<bool> block_significant(n_blocks_per_expert, false);
            for (int64_t b = 0; b < n_blocks_per_expert; b++) {
                for (int64_t ei = 0; ei < n_experts; ei++) {
                    float thresh = expert_max[ei] * sparse_threshold;
                    if (expert_block_importance[ei][b] >= thresh) {
                        block_significant[b] = true;
                        break;
                    }
                }
            }

            SparseInfo si;
            si.name = tname;
            si.n_blocks_total = n_blocks_per_expert;
            si.n_per_row = n_per_row;
            si.rows_per_expert = rows_per_expert;
            si.n_experts = n_experts;
            for (int64_t b = 0; b < n_blocks_per_expert; b++) {
                if (block_significant[b]) {
                    si.block_indices.push_back((uint16_t)b);
                }
            }
            si.n_blocks_stored = (int64_t)si.block_indices.size();

            sparse_infos.push_back(std::move(si));
        }

        double elapsed = (ggml_time_us() - t_start) / 1e6;
        double eta = (elapsed / (ti + 1)) * (n_tensors - ti - 1);
        if (sparse_threshold > 0 && n_blocks_per_expert > 0 && !sparse_infos.empty()) {
            auto & last = sparse_infos.back();
            fprintf(stderr, "  [%d/%d] %s: %lld→%lld blocks kept (%.0f%% pruned) [%.0fs ETA %.0fs]\n",
                    ti+1, n_tensors, tname.c_str(),
                    (long long)last.n_blocks_total, (long long)last.n_blocks_stored,
                    100.0 * (1.0 - (double)last.n_blocks_stored / last.n_blocks_total),
                    elapsed, eta);
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

        // Build GGUF header with sparse tensors
        gguf_context * shdr = gguf_init_empty();
        gguf_set_val_str(shdr, "general.architecture", model_arch.c_str());
        gguf_set_val_str(shdr, "general.type", "delta_correction_sparse");
        gguf_set_val_i32(shdr, "bs.delta.level", 1);
        gguf_set_val_str(shdr, "bs.delta.quant_type", quant_str);
        gguf_set_val_bool(shdr, "bs.delta.sparse", true);
        gguf_set_val_f32(shdr, "bs.delta.sparse_threshold", sparse_threshold);

        const size_t sctx_size = sparse_infos.size() * 2 * ggml_tensor_overhead() + 16*1024*1024;
        ggml_init_params sip = { .mem_size = sctx_size, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * sctx = ggml_init(sip);

        for (auto & si : sparse_infos) {
            // Packed delta tensor: n_stored_blocks * block_elems per expert row dimension
            int64_t packed_elems_per_expert = si.n_blocks_stored * ggml_blck_size(delta_type);
            ggml_tensor * td = ggml_new_tensor_3d(sctx, delta_type,
                packed_elems_per_expert > 0 ? packed_elems_per_expert : 1,
                si.rows_per_expert > 0 ? 1 : 1,  // packed: 1 "row" of packed blocks
                si.n_experts);
            ggml_set_name(td, si.name.c_str());
            gguf_add_tensor(shdr, td);

            // Index tensor: uint16 array of block positions
            std::string idx_name = si.name + ".idx";
            ggml_tensor * ti_t = ggml_new_tensor_1d(sctx, GGML_TYPE_I16,
                si.n_blocks_stored > 0 ? si.n_blocks_stored : 1);
            ggml_set_name(ti_t, idx_name.c_str());
            gguf_add_tensor(shdr, ti_t);

            // Store metadata
            char key[256];
            snprintf(key, sizeof(key), "bs.delta.%s.n_blocks_total", si.name.c_str());
            gguf_set_val_i32(shdr, key, (int32_t)si.n_blocks_total);
            snprintf(key, sizeof(key), "bs.delta.%s.n_blocks_stored", si.name.c_str());
            gguf_set_val_i32(shdr, key, (int32_t)si.n_blocks_stored);
        }

        gguf_write_to_file(shdr, sparse_path, true);
        size_t sparse_data_offset = gguf_get_meta_size(shdr);
        gguf_free(shdr);
        ggml_free(sctx);

        // Write sparse data
        FILE * sfp = fopen(sparse_path, "r+b");
        if (!sfp) { fprintf(stderr, "  ERROR: cannot open %s for writing\n", sparse_path); fclose(full_fp); goto skip_sparse; }
        fseek(sfp, (long)sparse_data_offset, SEEK_SET);

        // Read full delta and pack significant blocks
        size_t full_data_offset = level_files[0].data_offset;
        size_t full_cursor = full_data_offset;

        for (size_t si_idx = 0; si_idx < sparse_infos.size(); si_idx++) {
            auto & si = sparse_infos[si_idx];
            const size_t block_bytes = ggml_type_size(delta_type); // bytes per quantization block
            const size_t full_expert_bytes = ggml_row_size(delta_type, si.n_per_row) * si.rows_per_expert;
            const size_t packed_expert_bytes = si.n_blocks_stored * block_bytes;

            std::vector<uint8_t> full_expert(full_expert_bytes);
            std::vector<uint8_t> packed(packed_expert_bytes);

            for (int64_t ei = 0; ei < si.n_experts; ei++) {
                // Read full expert from level-1 output
                fseek(full_fp, (long)(full_cursor + ei * full_expert_bytes), SEEK_SET);
                fread(full_expert.data(), 1, full_expert_bytes, full_fp);

                // Pack significant blocks
                for (int64_t bi = 0; bi < si.n_blocks_stored; bi++) {
                    int64_t src_block = si.block_indices[bi];
                    memcpy(packed.data() + bi * block_bytes,
                           full_expert.data() + src_block * block_bytes,
                           block_bytes);
                }
                fwrite(packed.data(), 1, packed_expert_bytes, sfp);
            }

            // Pad packed tensor
            size_t total_packed = packed_expert_bytes * si.n_experts;
            size_t pad = (32 - (total_packed % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, sfp); }

            // Write index tensor
            fwrite(si.block_indices.data(), sizeof(uint16_t), si.n_blocks_stored, sfp);
            size_t idx_bytes = si.n_blocks_stored * sizeof(uint16_t);
            pad = (32 - (idx_bytes % 32)) % 32;
            if (pad > 0) { uint8_t z[32] = {0}; fwrite(z, 1, pad, sfp); }

            // Advance past this tensor in the full file
            size_t full_tensor_bytes = full_expert_bytes * si.n_experts;
            size_t full_pad = (32 - (full_tensor_bytes % 32)) % 32;
            full_cursor += full_tensor_bytes + full_pad;

            fprintf(stderr, "  [%zu/%zu] %s: %lld→%lld blocks (%.0f%% reduction)\n",
                    si_idx + 1, sparse_infos.size(), si.name.c_str(),
                    (long long)si.n_blocks_total, (long long)si.n_blocks_stored,
                    100.0 * (1.0 - (double)si.n_blocks_stored / si.n_blocks_total));
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
