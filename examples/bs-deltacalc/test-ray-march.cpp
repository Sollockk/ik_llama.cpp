// test-ray-march: standalone test for the PIM kernel.
//
// Loads a blurry tensor + delta tensor + sharp tensor, computes:
//   1. Reference: full sharp vec_dot (ground truth)
//   2. Baseline: full blurry vec_dot (no correction)
//   3. Ray march 2-tier: skip + active (legacy)
//   4. Cascade 4-tier: skip + coarse + standard + critical (radiance cascade)
//
// Compares results and reports error metrics for both paths.
//
// Usage:
//   test-ray-march \
//     --blurry  model_tq10.gguf \
//     --sharp   model_q4km-00001-of-00002.gguf \
//     --delta   delta_1.gguf \
//     --tensor  blk.5.ffn_down_exps.weight \
//     --expert  0

#include "ggml.h"
#include "ggml-ray-march.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

struct MappedFile {
    void * data = nullptr;
    size_t size = 0;
    int    fd   = -1;
    bool open(const char * path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        size = st.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    ~MappedFile() {
        if (data) munmap(data, size);
        if (fd >= 0) ::close(fd);
    }
    const uint8_t * at(size_t off) const { return (const uint8_t *)data + off; }
};

struct TensorLoc {
    size_t    offset;
    size_t    nbytes;
    ggml_type type;
    int64_t   ne[GGML_MAX_DIMS];
};

static void dequant_to_f32(const void * src, ggml_type type, int64_t n, float * dst) {
    if (type == GGML_TYPE_F32) { memcpy(dst, src, n * sizeof(float)); return; }
    ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    if (tt.to_float) tt.to_float(src, dst, n);
    else memset(dst, 0, n * sizeof(float));
}

static bool find_tensor(const char * path, const char * name, TensorLoc & out) {
    gguf_init_params p = { .no_alloc = true, .ctx = nullptr };
    gguf_context * g = gguf_init_from_file(path, p);
    if (!g) return false;

    ggml_init_params ip = { .mem_size = 64*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(ip);
    gguf_init_params p2 = { .no_alloc = true, .ctx = &ctx };
    gguf_context * g2 = gguf_init_from_file(path, p2);

    bool found = false;
    if (g2) {
        size_t data_off = gguf_get_data_offset(g);
        int n_tensors = gguf_get_n_tensors(g);
        for (int i = 0; i < n_tensors; i++) {
            if (strcmp(gguf_get_tensor_name(g, i), name) == 0) {
                out.offset = data_off + gguf_get_tensor_offset(g, i);
                out.type   = gguf_get_tensor_type(g, i);
                ggml_tensor * t = ggml_get_tensor(ctx, name);
                if (t) {
                    for (int d = 0; d < GGML_MAX_DIMS; d++) out.ne[d] = t->ne[d];
                    out.nbytes = ggml_nbytes(t);
                }
                found = true;
                break;
            }
        }
        gguf_free(g2);
    }
    ggml_free(ctx);
    gguf_free(g);
    return found;
}

int main(int argc, char ** argv) {
    const char * blurry_path = nullptr;
    const char * sharp_path  = nullptr;
    const char * delta_path  = nullptr;
    const char * tensor_name = "blk.5.ffn_down_exps.weight";
    int          expert_id   = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--blurry") && i+1 < argc) blurry_path = argv[++i];
        else if (!strcmp(argv[i], "--sharp") && i+1 < argc) sharp_path = argv[++i];
        else if (!strcmp(argv[i], "--delta") && i+1 < argc) delta_path = argv[++i];
        else if (!strcmp(argv[i], "--tensor") && i+1 < argc) tensor_name = argv[++i];
        else if (!strcmp(argv[i], "--expert") && i+1 < argc) expert_id = atoi(argv[++i]);
    }
    if (!blurry_path || !sharp_path || !delta_path) {
        fprintf(stderr, "Usage: %s --blurry <path> --sharp <path> --delta <path> [--tensor <name>] [--expert <id>]\n", argv[0]);
        return 1;
    }

    // Find tensors in all three GGUFs
    TensorLoc blurry_loc, sharp_loc, delta_loc;
    if (!find_tensor(blurry_path, tensor_name, blurry_loc)) {
        fprintf(stderr, "tensor %s not found in blurry\n", tensor_name); return 1;
    }
    if (!find_tensor(sharp_path, tensor_name, sharp_loc)) {
        fprintf(stderr, "tensor %s not found in sharp\n", tensor_name); return 1;
    }
    if (!find_tensor(delta_path, tensor_name, delta_loc)) {
        fprintf(stderr, "tensor %s not found in delta\n", tensor_name); return 1;
    }

    fprintf(stderr, "Tensor: %s\n", tensor_name);
    fprintf(stderr, "  blurry: type=%s ne=[%lld,%lld,%lld,%lld] %zu bytes\n",
            ggml_type_name(blurry_loc.type),
            (long long)blurry_loc.ne[0], (long long)blurry_loc.ne[1],
            (long long)blurry_loc.ne[2], (long long)blurry_loc.ne[3], blurry_loc.nbytes);
    fprintf(stderr, "  sharp:  type=%s ne=[%lld,%lld,%lld,%lld] %zu bytes\n",
            ggml_type_name(sharp_loc.type),
            (long long)sharp_loc.ne[0], (long long)sharp_loc.ne[1],
            (long long)sharp_loc.ne[2], (long long)sharp_loc.ne[3], sharp_loc.nbytes);
    fprintf(stderr, "  delta:  type=%s ne=[%lld,%lld,%lld,%lld] %zu bytes\n",
            ggml_type_name(delta_loc.type),
            (long long)delta_loc.ne[0], (long long)delta_loc.ne[1],
            (long long)delta_loc.ne[2], (long long)delta_loc.ne[3], delta_loc.nbytes);

    // Check QK_K alignment
    if (blurry_loc.ne[0] % 256 != 0) {
        fprintf(stderr, "\n  WARNING: ne[0]=%lld is not a multiple of QK_K=256.\n"
                "  vec_dot requires QK_K alignment. Try a tensor with ne[0] divisible by 256.\n"
                "  Suggestions: ffn_gate_inp, attn_norm, or output tensors.\n\n",
                (long long)blurry_loc.ne[0]);
        // Fall back to f32 dequant comparison instead of vec_dot
    }

    // mmap all files
    MappedFile blurry_mmap, sharp_mmap, delta_mmap;
    if (!blurry_mmap.open(blurry_path)) { fprintf(stderr, "mmap blurry failed\n"); return 1; }
    if (!sharp_mmap.open(sharp_path))   { fprintf(stderr, "mmap sharp failed\n"); return 1; }
    if (!delta_mmap.open(delta_path))   { fprintf(stderr, "mmap delta failed\n"); return 1; }

    // Compute expert slice sizes
    const int64_t n_experts = (blurry_loc.ne[2] > 0) ? blurry_loc.ne[2] : 1;
    const int64_t n_cols = blurry_loc.ne[0];  // input dimension
    const int64_t n_rows = blurry_loc.ne[1];  // output dimension
    const int64_t elems_per_expert = n_cols * n_rows;

    fprintf(stderr, "  expert %d of %lld, %lld rows × %lld cols = %lld elements\n",
            expert_id, (long long)n_experts, (long long)n_rows, (long long)n_cols,
            (long long)elems_per_expert);

    const size_t blurry_expert_bytes = blurry_loc.nbytes / n_experts;
    const size_t sharp_expert_bytes  = sharp_loc.nbytes / n_experts;
    const size_t delta_expert_bytes  = delta_loc.nbytes / n_experts;

    const uint8_t * blurry_data = blurry_mmap.at(blurry_loc.offset) + expert_id * blurry_expert_bytes;
    const uint8_t * sharp_data  = sharp_mmap.at(sharp_loc.offset)   + expert_id * sharp_expert_bytes;
    const uint8_t * delta_data  = delta_mmap.at(delta_loc.offset)   + expert_id * delta_expert_bytes;

    // Create a synthetic input vector (random f32) with realistic sparsity.
    // Post-SwiGLU activations are ~60% near-zero with some blocks entirely dead.
    // We zero out entire blocks to test skip/coarse tiers properly.
    std::vector<float> input_f32(n_cols);
    srand(42);
    {
        const int delta_blk = ggml_blck_size(delta_loc.type);
        const int blk = (delta_blk > 0) ? delta_blk : 32;
        for (int64_t i = 0; i < n_cols; i++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            // Block-level sparsity: ~30% of blocks are completely zero
            int block_id = (int)(i / blk);
            if ((block_id * 7 + 3) % 10 < 3) {
                v = 0.0f;  // dead block
            } else if (fabsf(v) < 0.6f) {
                v *= 0.01f;  // weak activation within active blocks
            }
            input_f32[i] = v;
        }
    }

    // Quantize input for vec_dot.
    // Some IQK types (q8_2_x4, q8_k16, etc.) are packed multi-block formats
    // that don't work with standalone per-block vec_dot calls. Fall back to
    // Q8_0 which is universally compatible with all quant types' vec_dot.
    ggml_type blurry_vdt = ggml_internal_get_type_traits(blurry_loc.type).vec_dot_type;
    ggml_type sharp_vdt  = ggml_internal_get_type_traits(sharp_loc.type).vec_dot_type;
    ggml_type delta_vdt  = ggml_internal_get_type_traits(delta_loc.type).vec_dot_type;

    fprintf(stderr, "  blurry vec_dot_type=%s, sharp vec_dot_type=%s, delta vec_dot_type=%s\n",
            ggml_type_name(blurry_vdt), ggml_type_name(sharp_vdt), ggml_type_name(delta_vdt));

    // Check if a vec_dot_type is a packed IQK format that won't work standalone.
    // These packed formats (q8_*_x4, q8_k*, etc.) require the fused IQK kernel
    // and can't be used with per-block vec_dot calls in the test harness.
    auto is_safe_vdt = [](ggml_type t) -> bool {
        return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q8_1 || t == GGML_TYPE_Q8_K
            || t == GGML_TYPE_F32  || t == GGML_TYPE_F16;
    };

    // For blurry input: use native vec_dot_type if safe, else Q8_0
    ggml_type blurry_input_type = is_safe_vdt(blurry_vdt) ? blurry_vdt : GGML_TYPE_Q8_0;
    if (blurry_input_type != blurry_vdt) {
        fprintf(stderr, "  NOTE: blurry vec_dot_type %s is packed IQK format, using Q8_0 fallback\n",
                ggml_type_name(blurry_vdt));
    }

    // For delta input: use native if safe, else try Q8_K (for K-quants), else Q8_0
    ggml_type delta_input_type = is_safe_vdt(delta_vdt) ? delta_vdt : GGML_TYPE_Q8_0;
    if (delta_input_type != delta_vdt) {
        fprintf(stderr, "  NOTE: delta vec_dot_type %s is packed IQK format, using Q8_0 fallback\n",
                ggml_type_name(delta_vdt));
    }

    // Quantize input to blurry's input type
    size_t input_q_size = ggml_row_size(blurry_input_type, n_cols);
    std::vector<uint8_t> input_q(input_q_size);
    {
        ggml_type_traits_t tt = ggml_internal_get_type_traits(blurry_input_type);
        if (tt.from_float) {
            tt.from_float(input_f32.data(), input_q.data(), n_cols);
        }
    }

    // Quantize input for delta's input type
    size_t dinput_q_size = ggml_row_size(delta_input_type, n_cols);
    std::vector<uint8_t> dinput_q(dinput_q_size);
    {
        ggml_type_traits_t tt = ggml_internal_get_type_traits(delta_input_type);
        if (tt.from_float) {
            tt.from_float(input_f32.data(), dinput_q.data(), n_cols);
        }
    }

    // Row strides
    const size_t blurry_row_bytes = blurry_expert_bytes / n_rows;
    const size_t sharp_row_bytes  = sharp_expert_bytes / n_rows;
    const size_t delta_row_bytes  = delta_expert_bytes / n_rows;

    fprintf(stderr, "  blurry_row=%zu sharp_row=%zu delta_row=%zu bytes\n",
            blurry_row_bytes, sharp_row_bytes, delta_row_bytes);

    // Compute block energies using delta's block size
    const int delta_blk_size = ggml_blck_size(delta_loc.type);
    const int n_blocks = (delta_blk_size > 0 && n_cols % delta_blk_size == 0)
        ? (int)(n_cols / delta_blk_size) : 0;
    fprintf(stderr, "  delta block_size=%d, n_blocks=%d\n", delta_blk_size, n_blocks);
    std::vector<float> energies(std::max(n_blocks, 1));
    if (n_blocks > 0) {
        ggml_ray_march_block_energies(input_f32.data(), energies.data(), (int)n_cols, delta_blk_size);
    }

    // Compute thresholds
    float e_sum = 0, e_sq = 0;
    for (int b = 0; b < n_blocks; b++) { e_sum += energies[b]; e_sq += energies[b]*energies[b]; }
    float e_mean = e_sum / n_blocks;
    float e_var = (e_sq / n_blocks) - (e_mean * e_mean);
    float e_std = e_var > 0 ? sqrtf(e_var) : 0;
    float skip_thresh = e_mean * 0.01f;
    float sharp_thresh = e_mean + 0.5f * e_std;

    // Partition
    std::vector<int> active_blocks(n_blocks);
    struct ggml_ray_march_tiers tiers;
    tiers.active_blocks = active_blocks.data();
    ggml_ray_march_partition_blocks(energies.data(), n_blocks, skip_thresh, &tiers);

    fprintf(stderr, "\nBlock partition: %d skip (%.0f%%), %d active (%.0f%%) of %d\n",
            tiers.n_skip, 100.0f*tiers.n_skip/n_blocks,
            tiers.n_active, 100.0f*tiers.n_active/n_blocks,
            n_blocks);

    // ---- Test using f32 dequant (works for ANY tensor, any alignment) ----
    // This validates the delta correction is mathematically correct,
    // independent of the vec_dot kernel.

    const int test_rows = std::min((int)n_rows, 10);
    const bool qk_aligned = (n_cols % 256 == 0);

    // Dequant one expert's worth of data for all three quality levels
    std::vector<float> blurry_f32_all(elems_per_expert);
    std::vector<float> sharp_f32_all(elems_per_expert);
    std::vector<float> delta_f32_all(elems_per_expert);

    dequant_to_f32(blurry_data, blurry_loc.type, elems_per_expert, blurry_f32_all.data());
    dequant_to_f32(sharp_data,  sharp_loc.type,  elems_per_expert, sharp_f32_all.data());
    dequant_to_f32(delta_data,  delta_loc.type,  elems_per_expert, delta_f32_all.data());

    fprintf(stderr, "\n--- F32 dequant validation (any alignment) ---\n");

    double total_err_blurry = 0, total_err_corrected = 0;

    for (int r = 0; r < test_rows; r++) {
        const float * w_sharp  = sharp_f32_all.data()  + r * n_cols;
        const float * w_blurry = blurry_f32_all.data() + r * n_cols;
        const float * w_delta  = delta_f32_all.data()  + r * n_cols;

        // Reference: dot(sharp, input)
        float ref = 0;
        for (int64_t j = 0; j < n_cols; j++) ref += w_sharp[j] * input_f32[j];

        // Baseline: dot(blurry, input)
        float baseline = 0;
        for (int64_t j = 0; j < n_cols; j++) baseline += w_blurry[j] * input_f32[j];

        // Corrected: dot(blurry + delta, input) = baseline + dot(delta, input)
        float delta_dot = 0;
        for (int64_t j = 0; j < n_cols; j++) delta_dot += w_delta[j] * input_f32[j];
        float corrected = baseline + delta_dot;

        // Three-tier: skip low-energy blocks from delta
        float corrected_3tier = baseline;
        if (n_blocks > 0) {
            for (int b = 0; b < n_blocks; b++) {
                if (energies[b] >= skip_thresh) {
                    float block_dot = 0;
                    int start = b * delta_blk_size;
                    int end = std::min(start + delta_blk_size, (int)n_cols);
                    for (int j = start; j < end; j++) {
                        block_dot += w_delta[j] * input_f32[j];
                    }
                    corrected_3tier += block_dot;
                }
            }
        } else {
            corrected_3tier = corrected; // no blocks, use full delta
        }

        float err_base = fabsf(ref - baseline);
        float err_corr = fabsf(ref - corrected);
        float err_3t   = fabsf(ref - corrected_3tier);
        total_err_blurry += err_base;
        total_err_corrected += err_3t;

        fprintf(stderr, "  row %2d: ref=%8.3f | blurry=%8.3f (err=%.4f) | +delta=%8.3f (err=%.4f) | 3tier=%8.3f (err=%.4f) %s\n",
                r, ref, baseline, err_base, corrected, err_corr, corrected_3tier, err_3t,
                err_3t < err_base ? "IMPROVED" : "same");
    }

    fprintf(stderr, "\nF32 Summary over %d rows:\n", test_rows);
    fprintf(stderr, "  Avg blurry error:    %.6f\n", total_err_blurry / test_rows);
    fprintf(stderr, "  Avg 3-tier error:    %.6f\n", total_err_corrected / test_rows);
    fprintf(stderr, "  Error reduction:     %.1f%%\n",
            100.0 * (1.0 - total_err_corrected / (total_err_blurry + 1e-10)));

    // ---- F32 cascade (4-tier) validation ----
    fprintf(stderr, "\n--- F32 cascade (4-tier radiance) validation ---\n");

    float coarse_thresh = e_mean * 0.10f;
    float critical_thresh = e_mean + 0.5f * e_std;

    // Count tiers for reporting
    int n_skip_c = 0, n_coarse_c = 0, n_std_c = 0, n_crit_c = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (energies[b] < skip_thresh) n_skip_c++;
        else if (energies[b] < coarse_thresh) n_coarse_c++;
        else if (energies[b] >= critical_thresh) n_crit_c++;
        else n_std_c++;
    }
    fprintf(stderr, "  Cascade partition: %d skip (%.0f%%), %d coarse (%.0f%%), %d standard (%.0f%%), %d critical (%.0f%%)\n",
            n_skip_c, 100.0f*n_skip_c/n_blocks,
            n_coarse_c, 100.0f*n_coarse_c/n_blocks,
            n_std_c, 100.0f*n_std_c/n_blocks,
            n_crit_c, 100.0f*n_crit_c/n_blocks);

    // Bandwidth savings estimate
    // Coarse blocks: base only (no delta read) → saves delta fraction of BW
    // Total active blocks = coarse + standard + critical
    int n_active_total = n_coarse_c + n_std_c + n_crit_c;
    if (n_active_total > 0) {
        float delta_frac = (float)delta_row_bytes / (blurry_row_bytes + delta_row_bytes);
        float bw_saved_pct = 100.0f * n_coarse_c * delta_frac / n_blocks;
        fprintf(stderr, "  Delta fraction of total BW: %.1f%%\n", delta_frac * 100.0f);
        fprintf(stderr, "  Estimated BW savings from coarse tier: %.1f%%\n", bw_saved_pct);
    }

    double total_err_cascade = 0;
    for (int r = 0; r < test_rows; r++) {
        const float * w_sharp  = sharp_f32_all.data()  + r * n_cols;
        const float * w_blurry = blurry_f32_all.data() + r * n_cols;
        const float * w_delta  = delta_f32_all.data()  + r * n_cols;

        float ref = 0;
        for (int64_t j = 0; j < n_cols; j++) ref += w_sharp[j] * input_f32[j];

        float baseline = 0;
        for (int64_t j = 0; j < n_cols; j++) baseline += w_blurry[j] * input_f32[j];

        // Cascade: base for all non-skip blocks, delta only for standard+critical
        float cascade_result = 0;
        if (n_blocks > 0) {
            for (int b = 0; b < n_blocks; b++) {
                float e = energies[b];
                int start = b * delta_blk_size;
                int end = std::min(start + delta_blk_size, (int)n_cols);

                if (e < skip_thresh) {
                    // SKIP: no contribution
                    continue;
                }

                // Base contribution (all non-skip tiers)
                float base_dot = 0;
                for (int j = start; j < end; j++) base_dot += w_blurry[j] * input_f32[j];
                cascade_result += base_dot;

                if (e >= coarse_thresh) {
                    // STANDARD or CRITICAL: add delta correction
                    float delta_dot = 0;
                    for (int j = start; j < end; j++) delta_dot += w_delta[j] * input_f32[j];
                    cascade_result += delta_dot;
                }
                // COARSE: base only, delta skipped
            }
        }

        float err_cascade = fabsf(ref - cascade_result);
        float err_base    = fabsf(ref - baseline);
        total_err_cascade += err_cascade;

        fprintf(stderr, "  row %2d: ref=%8.3f | blurry=%8.3f (err=%.4f) | cascade=%8.3f (err=%.4f) %s\n",
                r, ref, baseline, err_base, cascade_result, err_cascade,
                err_cascade < err_base ? "IMPROVED" : "same");
    }

    fprintf(stderr, "\nCascade F32 Summary:\n");
    fprintf(stderr, "  Avg blurry error:    %.6f\n", total_err_blurry / test_rows);
    fprintf(stderr, "  Avg 2-tier error:    %.6f\n", total_err_corrected / test_rows);
    fprintf(stderr, "  Avg cascade error:   %.6f\n", total_err_cascade / test_rows);
    fprintf(stderr, "  Cascade vs blurry:   %.1f%% error reduction\n",
            100.0 * (1.0 - total_err_cascade / (total_err_blurry + 1e-10)));
    fprintf(stderr, "  Cascade vs 2-tier:   %.1f%% MORE error (quality cost of skipping delta on weak blocks)\n",
            100.0 * (total_err_cascade - total_err_corrected) / (total_err_corrected + 1e-10));

    // ---- Test vec_dot kernel (if block-aligned) ----
    //
    // The ray march functions (ggml_vec_dot_ray_march, ggml_vec_dot_cascade)
    // look up vec_dot via ggml_internal_get_type_traits(weight_type).vec_dot.
    // That vec_dot expects input quantized as traits.vec_dot_type.
    //
    // Problem: IQK types like q8_2_x4 are packed multi-block formats that don't
    // work with per-block calls. We detect this and override the delta type
    // to a compatible one for the test (Q4_0 → force Q8_0 input).
    //
    // In the real inference path (ggml.c), the IQK fused kernel handles this
    // internally. Here we test the standalone ray march kernel path.

    const bool block_aligned = (n_blocks > 0) && (n_cols % delta_blk_size == 0);
    if (block_aligned) {
        fprintf(stderr, "\n--- Vec_dot kernel validation (block_size=%d, %d blocks) ---\n",
                delta_blk_size, n_blocks);

        // Use the actual weight types' vec_dot functions. If the delta's
        // vec_dot_type is a packed IQK format, the vec_dot will produce NaN.
        // Detect that and skip the vec_dot kernel test with a warning.
        ggml_type_traits_t blurry_traits = ggml_internal_get_type_traits(blurry_loc.type);
        ggml_type_traits_t delta_traits  = ggml_internal_get_type_traits(delta_loc.type);

        // Sanity check: does the delta vec_dot produce valid results?
        bool delta_vecdot_ok = false;
        {
            float test_val = 0;
            if (delta_traits.vec_dot) {
                delta_traits.vec_dot((int)n_cols, &test_val, 0,
                    delta_data, 0, dinput_q.data(), 0, 1);
                delta_vecdot_ok = std::isfinite(test_val);
            }
        }

        if (!delta_vecdot_ok) {
            fprintf(stderr, "  WARNING: delta vec_dot(%s, %s) produces NaN.\n",
                    ggml_type_name(delta_loc.type), ggml_type_name(delta_input_type));
            fprintf(stderr, "  The delta's native vec_dot_type (%s) is a packed IQK format\n",
                    ggml_type_name(delta_vdt));
            fprintf(stderr, "  that requires the fused IQK kernel, not standalone vec_dot.\n");
            fprintf(stderr, "  Vec_dot kernel tests will be SKIPPED.\n");
            fprintf(stderr, "  (F32 dequant validation above is still valid.)\n");
        }

        // Also check blurry
        bool blurry_vecdot_ok = false;
        {
            float test_val = 0;
            if (blurry_traits.vec_dot) {
                blurry_traits.vec_dot((int)n_cols, &test_val, 0,
                    blurry_data, 0, input_q.data(), 0, 1);
                blurry_vecdot_ok = std::isfinite(test_val);
            }
        }

        if (blurry_vecdot_ok && delta_vecdot_ok) {
            // First: sanity check — does sum of per-block delta vec_dot equal full-row?
            fprintf(stderr, "  Sanity check: full-row vs per-block delta vec_dot\n");
            int dqb_per_eblk = delta_blk_size / delta_traits.blck_size;
            size_t delta_eblk_bytes = (size_t)dqb_per_eblk * delta_traits.type_size;
            int dinp_qb_per_eblk = delta_blk_size / ggml_blck_size(delta_input_type);
            size_t dinp_eblk_bytes = (size_t)dinp_qb_per_eblk * ggml_type_size(delta_input_type);

            for (int r = 0; r < 3; r++) {
                float full_delta = 0;
                delta_traits.vec_dot((int)n_cols, &full_delta, 0,
                    delta_data + r * delta_row_bytes, 0, dinput_q.data(), 0, 1);

                float sum_blocks = 0;
                for (int b = 0; b < n_blocks; b++) {
                    float bv = 0;
                    delta_traits.vec_dot(delta_blk_size, &bv, 0,
                        delta_data + r * delta_row_bytes + b * delta_eblk_bytes, 0,
                        dinput_q.data() + b * dinp_eblk_bytes, 0, 1);
                    sum_blocks += bv;
                }
                fprintf(stderr, "    row %d: full=%.6f sum_blocks=%.6f diff=%.6f\n",
                        r, full_delta, sum_blocks, fabsf(full_delta - sum_blocks));
            }

            double vd_err_base = 0, vd_err_rm = 0;
            for (int r = 0; r < test_rows; r++) {
                const float * w_sharp = sharp_f32_all.data() + r * n_cols;
                float ref = 0;
                for (int64_t j = 0; j < n_cols; j++) ref += w_sharp[j] * input_f32[j];

                float baseline = 0;
                blurry_traits.vec_dot((int)n_cols, &baseline, 0,
                    blurry_data + r * blurry_row_bytes, 0, input_q.data(), 0, 1);

                float full_corr = 0;
                delta_traits.vec_dot((int)n_cols, &full_corr, 0,
                    delta_data + r * delta_row_bytes, 0, dinput_q.data(), 0, 1);

                float raymarch = 0;
                ggml_vec_dot_ray_march(
                    (int)n_cols, &raymarch,
                    blurry_data + r * blurry_row_bytes, (int)blurry_loc.type,
                    delta_data  + r * delta_row_bytes,  (int)delta_loc.type,
                    input_q.data(), dinput_q.data(),
                    &tiers, delta_blk_size);

                vd_err_base += fabsf(ref - baseline);
                vd_err_rm   += fabsf(ref - raymarch);

                fprintf(stderr, "  row %2d: ref=%8.3f | blurry=%8.3f | +full_delta=%8.3f | 2tier=%8.3f %s\n",
                        r, ref, baseline, baseline + full_corr, raymarch,
                        fabsf(ref - raymarch) < fabsf(ref - baseline) ? "IMPROVED" : "");
            }
            fprintf(stderr, "\nVec_dot 2-tier Summary:\n");
            fprintf(stderr, "  Avg blurry error: %.6f\n", vd_err_base / test_rows);
            fprintf(stderr, "  Avg 2-tier error: %.6f\n", vd_err_rm / test_rows);

            // ---- Cascade vec_dot kernel test ----
            fprintf(stderr, "\n--- Cascade vec_dot kernel validation ---\n");

            std::vector<int> coarse_buf(n_blocks), standard_buf(n_blocks), critical_buf(n_blocks);
            struct ggml_ray_march_cascade cascade;
            cascade.coarse_blocks   = coarse_buf.data();
            cascade.standard_blocks = standard_buf.data();
            cascade.critical_blocks = critical_buf.data();
            ggml_ray_march_partition_cascade(
                energies.data(), n_blocks,
                skip_thresh, coarse_thresh, critical_thresh,
                &cascade);

            fprintf(stderr, "  Cascade: %d skip, %d coarse, %d standard, %d critical\n",
                    cascade.n_skip, cascade.n_coarse, cascade.n_standard, cascade.n_critical);

            double vd_err_cascade = 0;
            for (int r = 0; r < test_rows; r++) {
                const float * w_sharp = sharp_f32_all.data() + r * n_cols;
                float ref = 0;
                for (int64_t j = 0; j < n_cols; j++) ref += w_sharp[j] * input_f32[j];

                float baseline = 0;
                blurry_traits.vec_dot((int)n_cols, &baseline, 0,
                    blurry_data + r * blurry_row_bytes, 0, input_q.data(), 0, 1);

                float cascade_result = 0;
                ggml_vec_dot_cascade(
                    (int)n_cols, &cascade_result,
                    blurry_data + r * blurry_row_bytes, (int)blurry_loc.type,
                    delta_data  + r * delta_row_bytes,  (int)delta_loc.type,
                    input_q.data(), dinput_q.data(),
                    &cascade, delta_blk_size);

                float raymarch = 0;
                ggml_vec_dot_ray_march(
                    (int)n_cols, &raymarch,
                    blurry_data + r * blurry_row_bytes, (int)blurry_loc.type,
                    delta_data  + r * delta_row_bytes,  (int)delta_loc.type,
                    input_q.data(), dinput_q.data(),
                    &tiers, delta_blk_size);

                vd_err_cascade += fabsf(ref - cascade_result);

                fprintf(stderr, "  row %2d: ref=%8.3f | blurry=%8.3f | 2tier=%8.3f | cascade=%8.3f %s\n",
                        r, ref, baseline, raymarch, cascade_result,
                        fabsf(ref - cascade_result) < fabsf(ref - baseline) ? "IMPROVED" : "");
            }

            fprintf(stderr, "\nVec_dot Cascade Summary:\n");
            fprintf(stderr, "  Avg blurry error:  %.6f\n", vd_err_base / test_rows);
            fprintf(stderr, "  Avg 2-tier error:  %.6f\n", vd_err_rm / test_rows);
            fprintf(stderr, "  Avg cascade error: %.6f\n", vd_err_cascade / test_rows);
            fprintf(stderr, "  Cascade quality cost vs 2-tier: %.2f%% more error\n",
                    100.0 * (vd_err_cascade - vd_err_rm) / (vd_err_rm + 1e-10));
            fprintf(stderr, "  Cascade quality gain vs blurry: %.1f%% less error\n",
                    100.0 * (1.0 - vd_err_cascade / (vd_err_base + 1e-10)));
        } // if blurry_vecdot_ok && delta_vecdot_ok
    } else {
        fprintf(stderr, "\n--- Vec_dot kernel test SKIPPED (ne[0]=%lld not aligned to block_size=%d) ---\n",
                (long long)n_cols, delta_blk_size);
    }

    return 0;
}
