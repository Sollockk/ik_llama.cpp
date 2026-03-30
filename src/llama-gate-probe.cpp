// llama-gate-probe.cpp — Per-layer importance prediction using low-rank probes
//
// Each probe is a [d_model, rank] FP16 matrix precomputed by bs-loracalc.
// Given a hidden state h, importance = ||probe^T @ h||² (sum of squared projections
// onto the principal correction directions).
//
// Large score = the blurry→sharp correction is large for this input = layer matters.
// Small score = blurry is fine for this input = safe to skip.

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_gate_probe_layer {
    int32_t              layer_idx;
    int64_t              d_model;   // ne[0] of probe tensor
    int32_t              rank;      // ne[1] of probe tensor
    std::vector<float>   data_f32;  // [d_model * rank] in GGML layout (ne[0]-contiguous)
};

struct llama_gate_probe_context {
    std::unordered_map<int32_t, llama_gate_probe_layer> probes;
};

// ---- Public API ----

struct llama_gate_probe_context * llama_gate_probe_load(const char * path_gguf) {
    if (!path_gguf) return nullptr;

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = {
        /* .no_alloc = */ false,  // allocate tensor data
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * gctx = gguf_init_from_file(path_gguf, params);
    if (!gctx) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_gguf);
        return nullptr;
    }

    auto * gpctx = new llama_gate_probe_context();

    // Find all tensors named "bs.gate_probe-{il}"
    const char prefix[] = "bs.gate_probe-";
    const int prefix_len = (int)strlen(prefix);

    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (strncmp(t->name, prefix, prefix_len) != 0) continue;

        int layer_idx = atoi(t->name + prefix_len);
        int64_t d_model = t->ne[0];
        int32_t rank    = (int32_t)t->ne[1];

        if (d_model <= 0 || rank <= 0) {
            fprintf(stderr, "%s: skipping probe '%s' with bad shape [%lld, %d]\n",
                    __func__, t->name, (long long)d_model, rank);
            continue;
        }

        llama_gate_probe_layer probe;
        probe.layer_idx = layer_idx;
        probe.d_model   = d_model;
        probe.rank      = rank;
        probe.data_f32.resize((size_t)d_model * rank);

        // Convert from FP16 to FP32 for fast dot products at runtime
        if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
            for (size_t i = 0; i < probe.data_f32.size(); i++) {
                probe.data_f32[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else if (t->type == GGML_TYPE_F32) {
            memcpy(probe.data_f32.data(), t->data, probe.data_f32.size() * sizeof(float));
        } else {
            fprintf(stderr, "%s: unsupported type %d for probe '%s'\n",
                    __func__, (int)t->type, t->name);
            continue;
        }

        gpctx->probes[layer_idx] = std::move(probe);
    }

    gguf_free(gctx);
    ggml_free(ctx);

    fprintf(stderr, "%s: loaded %zu gate probes from '%s'\n",
            __func__, gpctx->probes.size(), path_gguf);

    if (gpctx->probes.empty()) {
        delete gpctx;
        return nullptr;
    }
    return gpctx;
}

void llama_gate_probe_free(struct llama_gate_probe_context * gpctx) {
    delete gpctx;
}

float llama_gate_probe_score(
        const struct llama_gate_probe_context * gpctx,
        int32_t                                 layer_idx,
        const float                           * hidden_state,
        int32_t                                 n_embd) {
    if (!gpctx || !hidden_state) return -1.0f;

    auto it = gpctx->probes.find(layer_idx);
    if (it == gpctx->probes.end()) return -1.0f;

    const auto & probe = it->second;
    if (n_embd != probe.d_model) return -1.0f;

    // Compute ||probe^T @ h||²
    // probe is stored in GGML layout: data[rank_idx * d_model + embd_idx]
    // Each "column" (rank component) is a contiguous vector of length d_model.
    // We compute dot(column_r, h) for each r, then sum squares.
    float score = 0.0f;
    const float * p = probe.data_f32.data();
    for (int32_t r = 0; r < probe.rank; r++) {
        float dot = 0.0f;
        const float * col = p + (size_t)r * probe.d_model;
        for (int64_t i = 0; i < probe.d_model; i++) {
            dot += col[i] * hidden_state[i];
        }
        score += dot * dot;
    }
    return score;
}

int32_t llama_gate_probe_rank_layers(
        const struct llama_gate_probe_context * gpctx,
        const float                           * hidden_state,
        int32_t                                 n_embd,
        int32_t                               * out_layers,
        float                                 * out_scores,
        int32_t                                 max_layers) {
    if (!gpctx || !hidden_state || !out_layers || !out_scores || max_layers <= 0) return 0;

    struct entry { int32_t layer; float score; };
    std::vector<entry> entries;
    entries.reserve(gpctx->probes.size());

    for (const auto & [layer_idx, probe] : gpctx->probes) {
        float s = llama_gate_probe_score(gpctx, layer_idx, hidden_state, n_embd);
        if (s >= 0.0f) {
            entries.push_back({layer_idx, s});
        }
    }

    // Sort ascending by score (least important first) — same convention as entropy-based importance
    std::sort(entries.begin(), entries.end(),
              [](const entry & a, const entry & b) { return a.score < b.score; });

    int32_t n = std::min((int32_t)entries.size(), max_layers);
    for (int32_t i = 0; i < n; i++) {
        out_layers[i] = entries[i].layer;
        out_scores[i] = entries[i].score;
    }
    return n;
}
