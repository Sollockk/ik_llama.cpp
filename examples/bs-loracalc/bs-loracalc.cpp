// bs-loracalc: compute low-rank quantization-error corrections for blurry-sharp inference.
//
// For each expert tensor, computes:
//   delta = dequant(Q4_K_M_expert) - dequant(TQ1_0_expert)
//   [loraA, loraB] = rank-R SVD(delta)   where delta ≈ loraA @ loraB^T
//
// At inference, the correction is applied as two small matmuls:
//   output_corrected = output_blurry + (x @ loraB) @ loraA^T
//
// Output GGUF contains per-layer FP16 tensors:
//   blk.{i}.ffn_{gate,up,down}_exps_loraA.weight  [d_out, R, n_experts]
//   blk.{i}.ffn_{gate,up,down}_exps_loraB.weight  [d_in,  R, n_experts]
//
// Usage:
//   bs-loracalc \
//     --blurry  model_tq10.gguf \
//     --sharp   model_q4km-00001-of-00005.gguf \
//     --output  corrections.gguf \
//     --rank    16

#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

// ============================================================
// Self-contained randomized SVD (no external deps)
// ============================================================
//
// Computes rank-R approximation of an (m × n) FP32 matrix A:
//   A ≈ loraA @ loraB^T   where loraA is [m×R], loraB is [n×R]
//
// loraA absorbs the singular values: loraA = U * S
// loraB is the right singular vectors: loraB = V
//
// Algorithm: Halko/Martinsson/Tropp randomized SVD with Jacobi
// eigendecomposition of the small projected matrix.

// Dense (row-major) matmul: C[m×n] = A[m×k] @ B[k×n]
static void f32_matmul(float * C, const float * A, const float * B, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
    }
}

// Modified Gram-Schmidt: orthonormalize columns of A (m×n) in-place, result in Q.
static void f32_mgs(float * Q, const float * A, int m, int n) {
    memcpy(Q, A, (size_t)m * n * sizeof(float));
    for (int j = 0; j < n; j++) {
        float norm = 0.0f;
        for (int i = 0; i < m; i++) norm += Q[i*n+j] * Q[i*n+j];
        norm = sqrtf(norm);
        if (norm > 1e-12f) {
            float inv = 1.0f / norm;
            for (int i = 0; i < m; i++) Q[i*n+j] *= inv;
        }
        for (int k2 = j+1; k2 < n; k2++) {
            float dot = 0.0f;
            for (int i = 0; i < m; i++) dot += Q[i*n+j] * Q[i*n+k2];
            for (int i = 0; i < m; i++) Q[i*n+k2] -= dot * Q[i*n+j];
        }
    }
}

// Classical Jacobi eigendecomposition of symmetric n×n matrix A.
// After return: A is diagonal (eigenvalues on diagonal), V holds eigenvectors (columns).
static void f32_sym_jacobi(float * A, float * V, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V[i*n+j] = (i == j) ? 1.0f : 0.0f;

    for (int sweep = 0; sweep < 200; sweep++) {
        float off = 0.0f;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                off += A[i*n+j] * A[i*n+j];
        if (off < 1e-20f) break;

        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                float apq = A[p*n+q];
                if (fabsf(apq) < 1e-16f) continue;
                float app = A[p*n+p], aqq = A[q*n+q];
                float tau = (aqq - app) / (2.0f * apq);
                float t   = (tau >= 0.0f)
                            ? 1.0f / (tau + sqrtf(1.0f + tau*tau))
                            : 1.0f / (tau - sqrtf(1.0f + tau*tau));
                float c = 1.0f / sqrtf(1.0f + t*t);
                float s = t * c;

                A[p*n+p] = app - t * apq;
                A[q*n+q] = aqq + t * apq;
                A[p*n+q] = A[q*n+p] = 0.0f;
                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    float arp = A[r*n+p], arq = A[r*n+q];
                    A[r*n+p] = A[p*n+r] = c*arp - s*arq;
                    A[r*n+q] = A[q*n+r] = s*arp + c*arq;
                }
                for (int r = 0; r < n; r++) {
                    float vrp = V[r*n+p], vrq = V[r*n+q];
                    V[r*n+p] = c*vrp - s*vrq;
                    V[r*n+q] = s*vrp + c*vrq;
                }
            }
        }
    }
}

// Compute rank-R randomized SVD of A (m × n).
// loraA_out: m × R  (= U * S, absorbs singular values)
// loraB_out: n × R  (= V, right singular vectors)
// Both stored in caller-provided FP32 buffers.
static void randomized_svd(
    const float * A, int m, int n, int rank,
    std::vector<float> & loraA_out,
    std::vector<float> & loraB_out)
{
    const int oversample = 8;
    const int k = rank + oversample;

    // 1. Random Gaussian projection matrix Omega (n × k)
    std::mt19937 rng(0xdeadbeef);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> Omega((size_t)n * k);
    for (auto & v : Omega) v = dist(rng);

    // 2. Y = A @ Omega  (m × k)
    std::vector<float> Y((size_t)m * k);
    f32_matmul(Y.data(), A, Omega.data(), m, n, k);

    // 3. Q = orthonormal basis for range of Y  (m × k)
    std::vector<float> Q((size_t)m * k);
    f32_mgs(Q.data(), Y.data(), m, k);

    // 4. B = Q^T @ A  (k × n)
    std::vector<float> B((size_t)k * n, 0.0f);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int r = 0; r < m; r++) s += Q[r*k+i] * A[r*n+j];
            B[i*n+j] = s;
        }

    // 5. BBt = B @ B^T  (k × k)  — small, eigendecompose directly
    std::vector<float> BBt((size_t)k * k, 0.0f);
    for (int i = 0; i < k; i++)
        for (int j = i; j < k; j++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++) s += B[i*n+p] * B[j*n+p];
            BBt[i*k+j] = BBt[j*k+i] = s;
        }

    // 6. Jacobi eigendecompose BBt → eigenvalues on diagonal, eigenvecs in Vsmall
    std::vector<float> Vsmall((size_t)k * k);
    f32_sym_jacobi(BBt.data(), Vsmall.data(), k);

    // 7. Sort columns by eigenvalue descending
    std::vector<int> idx(k);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return BBt[a*k+a] > BBt[b*k+b]; });

    // 8. Build output matrices (top rank components only)
    loraA_out.assign((size_t)m * rank, 0.0f);
    loraB_out.assign((size_t)n * rank, 0.0f);

    for (int ri = 0; ri < rank; ri++) {
        int col = idx[ri];
        float sv = sqrtf(std::max(0.0f, BBt[col*k+col]));

        // loraA[:, ri] = Q @ Vsmall[:, col] * sv  (left singular vec × singular value)
        for (int r = 0; r < m; r++) {
            float s = 0.0f;
            for (int j = 0; j < k; j++) s += Q[r*k+j] * Vsmall[j*k+col];
            loraA_out[r*rank+ri] = s * sv;
        }

        // loraB[:, ri] = B^T @ Vsmall[:, col] / sv  (right singular vec)
        if (sv > 1e-12f) {
            float inv_sv = 1.0f / sv;
            for (int r = 0; r < n; r++) {
                float s = 0.0f;
                for (int j = 0; j < k; j++) s += B[j*n+r] * Vsmall[j*k+col];
                loraB_out[r*rank+ri] = s * inv_sv;
            }
        }
    }
}

// ============================================================
// GGUF helpers
// ============================================================

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

static bool is_expert_tensor(const std::string & name) {
    return name.find("ffn_gate_exps") != std::string::npos ||
           name.find("ffn_up_exps")   != std::string::npos ||
           name.find("ffn_down_exps") != std::string::npos;
}

static std::string suffix_of(const std::string & name) {
    // "blk.5.ffn_gate_exps.weight" → "ffn_gate_exps.weight"
    auto pos = name.find("blk.");
    if (pos == std::string::npos) return name;
    pos = name.find('.', pos + 4);
    if (pos == std::string::npos) return name;
    return name.substr(pos + 1);
}

static std::vector<uint8_t> read_tensor_data(int fd, size_t offset, size_t nbytes) {
    std::vector<uint8_t> buf(nbytes);
    size_t remaining = nbytes;
    size_t done = 0;
    while (remaining > 0) {
        ssize_t n = pread(fd, buf.data() + done, remaining, (off_t)(offset + done));
        if (n <= 0) {
            fprintf(stderr, "pread failed at offset %zu\n", offset + done);
            break;
        }
        done += (size_t)n;
        remaining -= (size_t)n;
    }
    return buf;
}

static void dequant_to_f32(const void * src, ggml_type type, int64_t n_elements, std::vector<float> & dst) {
    dst.resize(n_elements);
    ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    if (!tt.to_float) {
        fprintf(stderr, "no to_float for type %d\n", (int)type);
        return;
    }
    tt.to_float(src, dst.data(), n_elements);
}

static void f32_to_f16(const float * src, ggml_fp16_t * dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = ggml_fp32_to_fp16(src[i]);
}

// ============================================================
// Main
// ============================================================

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --blurry <tq10.gguf> --sharp <q4km.gguf> --output <out.gguf> [--rank N]\n"
        "\n"
        "  --blurry  Path to blurry model GGUF (TQ1_0 or similar)\n"
        "  --sharp   Path to first split of sharp model GGUF (Q4_K_M)\n"
        "  --output  Output path for corrections GGUF\n"
        "  --rank    LoRA rank (default: 16)\n"
        "  --layers  Layer range to process, e.g. 0-89 (default: all)\n",
        prog);
}

int main(int argc, char ** argv) {
    const char * blurry_path = nullptr;
    const char * sharp_path  = nullptr;
    const char * output_path = nullptr;
    int          rank        = 16;
    int          layer_min   = 0;
    int          layer_max   = INT32_MAX;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--blurry")  && i+1 < argc) blurry_path = argv[++i];
        else if (!strcmp(argv[i], "--sharp")   && i+1 < argc) sharp_path  = argv[++i];
        else if (!strcmp(argv[i], "--output")  && i+1 < argc) output_path = argv[++i];
        else if (!strcmp(argv[i], "--rank")    && i+1 < argc) rank        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers")  && i+1 < argc) {
            sscanf(argv[++i], "%d-%d", &layer_min, &layer_max);
        }
        else { print_usage(argv[0]); return 1; }
    }
    if (!blurry_path || !sharp_path || !output_path) {
        print_usage(argv[0]);
        return 1;
    }

    // ---- open blurry GGUF ----
    gguf_init_params blurry_params = { .no_alloc = true, .ctx = nullptr };
    gguf_context * blurry_gguf = gguf_init_from_file(blurry_path, blurry_params);
    if (!blurry_gguf) { fprintf(stderr, "failed to open blurry: %s\n", blurry_path); return 1; }
    int blurry_fd = open(blurry_path, O_RDONLY);
    if (blurry_fd < 0) { fprintf(stderr, "failed to open blurry fd: %s\n", blurry_path); return 1; }

    // ---- open sharp GGUF (may be split — handle first split only for now) ----
    // Detect splits by probing paths like 00002-of-00005
    std::vector<std::string> sharp_paths;
    sharp_paths.push_back(sharp_path);
    {
        std::string base = sharp_path;
        // Try to find "-00001-of-" pattern and enumerate all splits
        auto pos = base.find("-00001-of-");
        if (pos != std::string::npos) {
            int total = atoi(base.c_str() + pos + 10);
            for (int s = 2; s <= total; s++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%05d", s);
                std::string path = base.substr(0, pos+1) + buf + "-of-" + (base.c_str() + pos + 10);
                // path looks like "...00002-of-00005.gguf"
                // Rebuild properly
                std::string total_str = base.substr(pos + 10);
                snprintf(buf, sizeof(buf), "-%05d-of-%s", s, total_str.c_str());
                path = base.substr(0, pos) + buf;
                sharp_paths.push_back(path);
            }
        }
    }

    std::vector<gguf_context *> sharp_ggufs;
    std::vector<int>            sharp_fds;
    for (auto & sp : sharp_paths) {
        gguf_init_params p = { .no_alloc = true, .ctx = nullptr };
        gguf_context * sg = gguf_init_from_file(sp.c_str(), p);
        if (!sg) { fprintf(stderr, "warning: could not open sharp split: %s\n", sp.c_str()); break; }
        int fd = open(sp.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "warning: could not open sharp fd: %s\n", sp.c_str()); gguf_free(sg); break; }
        sharp_ggufs.push_back(sg);
        sharp_fds.push_back(fd);
        fprintf(stderr, "opened sharp split: %s\n", sp.c_str());
    }
    if (sharp_ggufs.empty()) { fprintf(stderr, "no sharp splits opened\n"); return 1; }

    // ---- build tensor index for blurry ----
    std::unordered_map<std::string, TensorInfo> blurry_index;
    {
        ggml_init_params ip = { .mem_size = 256*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * ctx = ggml_init(ip);
        gguf_init_params p2 = { .no_alloc = true, .ctx = &ctx };
        // re-open with ctx to get tensor metadata
        gguf_context * bg2 = gguf_init_from_file(blurry_path, p2);
        if (bg2) {
            size_t data_off = gguf_get_data_offset(blurry_gguf);
            int n = gguf_get_n_tensors(blurry_gguf);
            for (int i = 0; i < n; i++) {
                const char * tname = gguf_get_tensor_name(blurry_gguf, i);
                TensorInfo info;
                info.name        = tname;
                info.split_idx   = 0;
                info.file_offset = data_off + gguf_get_tensor_offset(blurry_gguf, i);
                info.type        = gguf_get_tensor_type(blurry_gguf, i);
                info.layer_idx   = extract_layer_idx(info.name);
                ggml_tensor * t  = ggml_get_tensor(ctx, tname);
                if (t) {
                    for (int d = 0; d < GGML_MAX_DIMS; d++) info.ne[d] = t->ne[d];
                    info.nbytes = ggml_nbytes(t);
                } else {
                    memset(info.ne, 0, sizeof(info.ne));
                    info.nbytes = 0;
                }
                blurry_index[info.name] = info;
            }
            gguf_free(bg2);
        }
        ggml_free(ctx);
    }

    // ---- build tensor index for all sharp splits ----
    std::unordered_map<std::string, TensorInfo> sharp_index;
    for (int si = 0; si < (int)sharp_ggufs.size(); si++) {
        ggml_init_params ip = { .mem_size = 256*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
        ggml_context * ctx = ggml_init(ip);
        gguf_init_params p2 = { .no_alloc = true, .ctx = &ctx };
        gguf_context * sg2 = gguf_init_from_file(sharp_paths[si].c_str(), p2);
        if (sg2) {
            size_t data_off = gguf_get_data_offset(sharp_ggufs[si]);
            int n = gguf_get_n_tensors(sharp_ggufs[si]);
            for (int i = 0; i < n; i++) {
                const char * tname = gguf_get_tensor_name(sharp_ggufs[si], i);
                TensorInfo info;
                info.name        = tname;
                info.split_idx   = si;
                info.file_offset = data_off + gguf_get_tensor_offset(sharp_ggufs[si], i);
                info.type        = gguf_get_tensor_type(sharp_ggufs[si], i);
                info.layer_idx   = extract_layer_idx(info.name);
                ggml_tensor * t  = ggml_get_tensor(ctx, tname);
                if (t) {
                    for (int d = 0; d < GGML_MAX_DIMS; d++) info.ne[d] = t->ne[d];
                    info.nbytes = ggml_nbytes(t);
                } else {
                    memset(info.ne, 0, sizeof(info.ne));
                    info.nbytes = 0;
                }
                sharp_index[info.name] = info;
            }
            gguf_free(sg2);
        }
        ggml_free(ctx);
    }

    // ---- find all expert tensor names in sharp ----
    std::vector<std::string> expert_tensor_names;
    for (auto & [name, info] : sharp_index) {
        if (is_expert_tensor(name) &&
            info.layer_idx >= layer_min && info.layer_idx <= layer_max) {
            expert_tensor_names.push_back(name);
        }
    }
    std::sort(expert_tensor_names.begin(), expert_tensor_names.end());

    fprintf(stderr, "Found %zu expert tensors to process (rank=%d)\n",
            expert_tensor_names.size(), rank);

    // ---- prepare output GGUF ----
    gguf_context * out_gguf = gguf_init_empty();
    gguf_set_val_str(out_gguf, "bs.loracalc.blurry_path", blurry_path);
    gguf_set_val_str(out_gguf, "bs.loracalc.sharp_path",  sharp_path);
    gguf_set_val_i32(out_gguf, "bs.loracalc.rank",        rank);

    // We need a ggml_context to hold the output tensors.
    // Allocate enough for 2 tensors per expert tensor × n_experts per tensor × 2 (loraA/B)
    // Each tensor: FP16, max size d_out * R * 160 * 4 bytes
    // Generous: 1GB for tensor metadata + headers
    ggml_init_params out_ip = {
        .mem_size   = (size_t)512 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context * out_ctx = ggml_init(out_ip);

    // Accumulate all output tensor data in a flat buffer (written sequentially to file)
    // We'll write the GGUF in two passes: metadata first, then raw tensor data.
    // Use a temporary file approach: collect tensors with their data, then write.

    struct OutTensor {
        std::string           name;
        int64_t               ne[4];
        std::vector<ggml_fp16_t> data;
    };
    std::vector<OutTensor> out_tensors;

    // ---- main processing loop ----
    int processed = 0;
    int total = (int)expert_tensor_names.size();

    for (auto & sname : expert_tensor_names) {
        auto & sinfo = sharp_index[sname];

        // Find corresponding blurry tensor
        auto bit = blurry_index.find(sname);
        if (bit == blurry_index.end()) {
            fprintf(stderr, "  [skip] no blurry tensor: %s\n", sname.c_str());
            continue;
        }
        auto & binfo = bit->second;

        // ne[2] = n_experts, ne[1] = d_in, ne[0] = d_out (GGML layout)
        int64_t n_experts = sinfo.ne[2];
        int64_t d_out     = sinfo.ne[0];
        int64_t d_in      = sinfo.ne[1];
        int64_t n_elem_expert = d_out * d_in;

        if (n_experts <= 0 || d_in <= 0 || d_out <= 0) {
            fprintf(stderr, "  [skip] bad shape: %s  ne=[%ld,%ld,%ld]\n",
                    sname.c_str(), (long)d_out, (long)d_in, (long)n_experts);
            continue;
        }

        fprintf(stderr, "[%d/%d] %s  [%ld × %ld × %ld experts]\n",
                ++processed, total, sname.c_str(), (long)d_out, (long)d_in, (long)n_experts);

        // Read full blurry tensor (all experts)
        auto blurry_raw = read_tensor_data(blurry_fd, binfo.file_offset, binfo.nbytes);
        // Read full sharp tensor (all experts)
        auto sharp_raw  = read_tensor_data(sharp_fds[sinfo.split_idx],
                                           sinfo.file_offset, sinfo.nbytes);

        // Per-expert slices
        size_t b_slice = binfo.nbytes / (size_t)n_experts;
        size_t s_slice = sinfo.nbytes / (size_t)n_experts;

        // Output stacked loraA [d_out, rank, n_experts] and loraB [d_in, rank, n_experts] (FP16)
        OutTensor ot_A, ot_B;
        {
            std::string suffix = suffix_of(sname);
            // Remove ".weight" suffix if present, then add _loraA/_loraB
            std::string base_suffix = suffix;
            if (base_suffix.size() > 7 &&
                base_suffix.substr(base_suffix.size()-7) == ".weight") {
                base_suffix = base_suffix.substr(0, base_suffix.size()-7);
            }
            ot_A.name = "blk." + std::to_string(sinfo.layer_idx) + "." + base_suffix + "_loraA.weight";
            ot_B.name = "blk." + std::to_string(sinfo.layer_idx) + "." + base_suffix + "_loraB.weight";
        }
        ot_A.ne[0] = d_out;   ot_A.ne[1] = rank; ot_A.ne[2] = n_experts; ot_A.ne[3] = 1;
        ot_B.ne[0] = d_in;    ot_B.ne[1] = rank; ot_B.ne[2] = n_experts; ot_B.ne[3] = 1;
        ot_A.data.resize((size_t)d_out * rank * n_experts);
        ot_B.data.resize((size_t)d_in  * rank * n_experts);

        #pragma omp parallel for schedule(dynamic)
        for (int64_t ei = 0; ei < n_experts; ei++) {
            // Thread-local buffers
            std::vector<float> blurry_f32, sharp_f32, delta_f32;
            std::vector<float> loraA_f32, loraB_f32;

            // Dequantize blurry expert slice
            dequant_to_f32(blurry_raw.data() + ei * b_slice, binfo.type, n_elem_expert, blurry_f32);
            // Dequantize sharp expert slice
            dequant_to_f32(sharp_raw.data()  + ei * s_slice, sinfo.type, n_elem_expert, sharp_f32);

            // delta = sharp - blurry
            delta_f32.resize(n_elem_expert);
            for (int64_t j = 0; j < n_elem_expert; j++) {
                delta_f32[j] = sharp_f32[j] - blurry_f32[j];
            }

            // Randomized SVD of delta [d_out × d_in]
            randomized_svd(delta_f32.data(), (int)d_out, (int)d_in, rank, loraA_f32, loraB_f32);

            // Store as FP16 into stacked output tensors (non-overlapping writes, safe)
            f32_to_f16(loraA_f32.data(), ot_A.data.data() + (size_t)ei * d_out * rank, (int64_t)d_out * rank);
            f32_to_f16(loraB_f32.data(), ot_B.data.data() + (size_t)ei * d_in  * rank, (int64_t)d_in  * rank);
        }

        out_tensors.push_back(std::move(ot_A));
        out_tensors.push_back(std::move(ot_B));
    }

    // ---- write output GGUF ----
    fprintf(stderr, "Writing %zu correction tensors to %s ...\n", out_tensors.size(), output_path);

    // Add tensors to gguf context
    for (auto & ot : out_tensors) {
        ggml_tensor * t = ggml_new_tensor_4d(out_ctx, GGML_TYPE_F16,
                                              ot.ne[0], ot.ne[1], ot.ne[2], ot.ne[3]);
        ggml_set_name(t, ot.name.c_str());
        t->data = (void *)ot.data.data();  // point at our buffer for write
        gguf_add_tensor(out_gguf, t);
    }

    // gguf_write_to_file copies tensor data from t->data pointers
    gguf_write_to_file(out_gguf, output_path, /*only_meta=*/false);

    // ---- cleanup ----
    gguf_free(out_gguf);
    ggml_free(out_ctx);
    for (auto * sg : sharp_ggufs) gguf_free(sg);
    for (int fd : sharp_fds)      close(fd);
    gguf_free(blurry_gguf);
    close(blurry_fd);

    fprintf(stderr, "Done. Wrote %zu tensors (rank=%d).\n", out_tensors.size(), rank);
    return 0;
}
