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
    if (!strcmp(s, "Q2_K"))   return GGML_TYPE_Q2_K;
    if (!strcmp(s, "Q3_K_S")) return GGML_TYPE_Q3_K;
    if (!strcmp(s, "Q3_K_M")) return GGML_TYPE_Q3_K;
    if (!strcmp(s, "Q4_K_S")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "Q4_K_M")) return GGML_TYPE_Q4_K;
    if (!strcmp(s, "Q5_K_S")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "Q5_K_M")) return GGML_TYPE_Q5_K;
    if (!strcmp(s, "Q6_K"))   return GGML_TYPE_Q6_K;
    if (!strcmp(s, "Q8_0"))   return GGML_TYPE_Q8_0;
    if (!strcmp(s, "F16"))    return GGML_TYPE_F16;
    if (!strcmp(s, "F32"))    return GGML_TYPE_F32;
    fprintf(stderr, "unknown quantization type: %s\n", s);
    return GGML_TYPE_Q2_K;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --blurry <path> --sharp <path> --levels <N> [--quant <type>] --out-prefix <prefix> [--threads N]\n\n"
        "  --blurry     Path to the blurry (low-quality) GGUF model\n"
        "  --sharp      Path to the sharp (high-quality) GGUF model (supports splits)\n"
        "  --levels     Number of delta correction levels to generate (1-8)\n"
        "  --quant      Quantization type for deltas (default: Q2_K)\n"
        "  --out-prefix Output prefix: produces <prefix>_1.gguf, <prefix>_2.gguf, ...\n"
        "  --layers     Layer range to process (e.g. 3-50, default: all)\n"
        "  --threads    Number of parallel threads (default: 4)\n",
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

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--blurry")     && i+1 < argc) blurry_path = argv[++i];
        else if (!strcmp(argv[i], "--sharp")      && i+1 < argc) sharp_path  = argv[++i];
        else if (!strcmp(argv[i], "--out-prefix") && i+1 < argc) out_prefix  = argv[++i];
        else if (!strcmp(argv[i], "--quant")      && i+1 < argc) quant_str   = argv[++i];
        else if (!strcmp(argv[i], "--levels")     && i+1 < argc) n_levels    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")    && i+1 < argc) n_threads   = atoi(argv[++i]);
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

            ggml_tensor * t = ggml_new_tensor_4d(ctx, delta_type, ne0, ne1, ne2, ne3);
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
        const size_t delta_bpe  = ggml_row_size(delta_type, n_per_row) * rows_per_expert;
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

                ggml_quantize_chunk(delta_type, re.data(), dq.data(), 0, rows_per_expert, n_per_row, nullptr);

                memcpy(level_deltas[level].data() + ei * delta_bpe, dq.data(), delta_bpe);

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

        double elapsed = (ggml_time_us() - t_start) / 1e6;
        double eta = (elapsed / (ti + 1)) * (n_tensors - ti - 1);
        fprintf(stderr, "\r  [%d/%d] %.0fs ETA %.0fs  (%s: %.1fM elems, %lld experts)          ",
                ti+1, n_tensors, elapsed, eta, tname.c_str(),
                n_elements / 1e6, (long long)n_experts);
    }

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
