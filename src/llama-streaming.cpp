// llama-streaming.cpp — Streaming micro-graph execution engine (v2)
//
// v2: Multi-signal replanning (probe + entropy + expert diversity),
//     cross-token EMA memory, HALT/CHECKPOINT, diagnostic logging.

#include "llama-streaming.h"
#include "llama.h"
#include "llama-impl.h"
#include <random>  // needed before llama-context.h (for llama-sampling.h's std::mt19937)
#include "llama-context.h"
#include "llama-model.h"
#include "llama-cparams.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Execution plan: composite scoring
// ---------------------------------------------------------------------------

float llama_execution_plan::compute_composite(const llama_layer_signal & signal) const {
    // Normalize entropy: divide by max observed entropy across all layers
    float max_entropy = 0.0f;
    for (auto & [lid, sig] : layer_signals) {
        if (sig.avg_entropy > max_entropy) max_entropy = sig.avg_entropy;
    }
    float norm_entropy = (max_entropy > 0.0f) ? signal.avg_entropy / max_entropy : 0.0f;

    // Expert diversity: fraction of unique experts out of total slots
    float expert_div = (signal.n_total_experts > 0)
        ? (float)signal.n_unique_experts / (float)signal.n_total_experts
        : 0.0f;

    // Composite: probe score amplified by entropy and diversity
    float composite = signal.probe_score
        * (1.0f + entropy_weight * norm_entropy)
        * (1.0f + diversity_weight * expert_div);

    return composite;
}

// ---------------------------------------------------------------------------
// Execution plan: multi-signal replanning
// ---------------------------------------------------------------------------

void llama_execution_plan::replan_multi(
        int32_t layer_id,
        const llama_layer_signal & signal,
        int32_t n_layers) {

    // Store full signal
    layer_signals[layer_id] = signal;
    layer_scores[layer_id] = signal.probe_score;
    layer_exec_count[layer_id]++;

    // Compute composite importance
    float composite = compute_composite(signal);

    // Update cross-token EMA
    if (ema_scores.count(layer_id)) {
        ema_scores[layer_id] = ema_alpha * composite + (1.0f - ema_alpha) * ema_scores[layer_id];
    } else {
        ema_scores[layer_id] = composite;
    }

    // Use EMA score for threshold decisions (smoothed across tokens)
    float decision_score = ema_scores[layer_id];

    // Need enough data for statistics
    if (ema_scores.size() < 3) return;

    // Compute statistics over EMA scores
    float sum = 0.0f, sum_sq = 0.0f;
    for (auto & [lid, sc] : ema_scores) {
        sum    += sc;
        sum_sq += sc * sc;
    }
    const float n = (float)ema_scores.size();
    const float mean = sum / n;
    const float var  = (sum_sq / n) - (mean * mean);
    const float stddev = var > 0.0f ? sqrtf(var) : 0.0f;

    const float sharpen_threshold = mean + replan_sharpen_sigma * stddev;

    // If this layer's smoothed score exceeds threshold and hasn't been
    // revisited too many times, schedule a SHARPEN pass.
    if (decision_score > sharpen_threshold &&
        layer_exec_count[layer_id] <= max_loop_iterations) {

        // Check we haven't already scheduled SHARPEN for this layer
        bool already_scheduled = false;
        for (size_t i = pc; i < ops.size(); ++i) {
            if (ops[i].type == STREAM_SHARPEN && ops[i].layer_id == layer_id) {
                already_scheduled = true;
                break;
            }
        }
        if (!already_scheduled) {
            // Protect first/last N layers
            if (layer_id >= min_protect_layers &&
                layer_id < n_layers - min_protect_layers) {
                append({STREAM_SHARPEN, layer_id, STREAM_Q_SHARP});
            }
        }
    }

    // Convergence check via score norm
    float curr_norm = 0.0f;
    for (auto & [lid, sc] : ema_scores) {
        curr_norm += sc * sc;
    }
    curr_norm = sqrtf(curr_norm);

    if (prev_score_norm > 0.0f && curr_norm > 0.0f) {
        float delta = fabsf(curr_norm - prev_score_norm) / prev_score_norm;
        if (delta < convergence_epsilon) {
            clear_pending_sharpens();
        }
    }
    prev_score_norm = curr_norm;
}

// Single-signal convenience wrapper
void llama_execution_plan::replan(int32_t layer_id, float probe_score, int32_t n_layers) {
    llama_layer_signal signal;
    signal.probe_score = probe_score;

    // Pull entropy and routing data from stored signals if available
    if (layer_signals.count(layer_id)) {
        signal.avg_entropy      = layer_signals[layer_id].avg_entropy;
        signal.n_unique_experts = layer_signals[layer_id].n_unique_experts;
        signal.n_total_experts  = layer_signals[layer_id].n_total_experts;
    }

    replan_multi(layer_id, signal, n_layers);
}

// ---------------------------------------------------------------------------
// CHECKPOINT and HALT support
// ---------------------------------------------------------------------------

void llama_execution_plan::save_checkpoint() {
    checkpoint_scores = layer_scores;
    checkpoint_norm = 0.0f;
    for (auto & [lid, sc] : checkpoint_scores) {
        checkpoint_norm += sc * sc;
    }
    checkpoint_norm = sqrtf(checkpoint_norm);
}

bool llama_execution_plan::check_loop_improvement() const {
    if (checkpoint_norm <= 0.0f) return true;  // no checkpoint, always continue

    float curr_norm = 0.0f;
    for (auto & [lid, sc] : layer_scores) {
        curr_norm += sc * sc;
    }
    curr_norm = sqrtf(curr_norm);

    // Improvement = reduction in score norm (lower = better, means less correction needed)
    float improvement = (checkpoint_norm - curr_norm) / checkpoint_norm;
    return improvement >= min_improvement_threshold;
}

void llama_execution_plan::clear_pending_sharpens() {
    auto it = std::remove_if(ops.begin() + (int64_t)pc, ops.end(),
        [](const llama_stream_op & op) {
            return op.type == STREAM_SHARPEN;
        });
    ops.erase(it, ops.end());
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string llama_execution_plan::summary() const {
    int n_skip = 0, n_blurry = 0, n_sharp = 0, n_sharpen = 0, n_attn = 0;
    for (const auto & op : ops) {
        switch (op.type) {
            case STREAM_SKIP:           n_skip++;    break;
            case STREAM_EXEC_ATTN_ONLY: n_attn++;    break;
            case STREAM_SHARPEN:        n_sharpen++; break;
            case STREAM_EXEC:
                if (op.quality == STREAM_Q_SHARP) n_sharp++;
                else n_blurry++;
                break;
            default: break;
        }
    }

    float ema_min = 1e9f, ema_max = -1e9f;
    for (auto & [lid, sc] : ema_scores) {
        if (sc < ema_min) ema_min = sc;
        if (sc > ema_max) ema_max = sc;
    }

    char buf[256];
    if (ema_scores.empty()) {
        snprintf(buf, sizeof(buf), "%zu ops: %d skip, %d blurry, %d sharp, %d sharpen-pending",
                 ops.size(), n_skip, n_blurry, n_sharp, n_sharpen);
    } else {
        snprintf(buf, sizeof(buf), "%zu ops: %d skip, %d blurry, %d sharp, %d sharpen | EMA [%.2f, %.2f]",
                 ops.size(), n_skip, n_blurry, n_sharp, n_sharpen, ema_min, ema_max);
    }
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Plan factories
// ---------------------------------------------------------------------------

llama_execution_plan llama_plan_from_tiers(
        int n_layers,
        const std::unordered_set<int32_t> & skip_set,
        const std::unordered_set<int32_t> & priority_set) {
    llama_execution_plan plan;
    plan.ops.reserve(n_layers);
    for (int il = 0; il < n_layers; ++il) {
        if (skip_set.count(il)) {
            plan.ops.push_back({STREAM_SKIP, il, STREAM_Q_BLURRY});
        } else if (priority_set.count(il)) {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_SHARP});
        } else {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY});
        }
    }
    return plan;
}

llama_execution_plan llama_plan_sequential(int n_layers) {
    llama_execution_plan plan;
    plan.ops.reserve(n_layers);
    for (int il = 0; il < n_layers; ++il) {
        plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY});
    }
    return plan;
}

llama_execution_plan llama_plan_from_probe_history(
        int n_layers,
        const std::unordered_map<int32_t, float> & ema_scores,
        const std::unordered_set<int32_t> & base_skip_set,
        float skip_sigma,
        float sharp_sigma) {

    llama_execution_plan plan;
    plan.ops.reserve(n_layers);
    plan.ema_scores = ema_scores;  // carry forward

    // Compute statistics over EMA scores
    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;
    for (auto & [lid, sc] : ema_scores) {
        sum    += sc;
        sum_sq += sc * sc;
        count++;
    }

    float mean = 0.0f, stddev = 0.0f;
    if (count > 0) {
        mean = sum / (float)count;
        float var = (sum_sq / (float)count) - (mean * mean);
        stddev = var > 0.0f ? sqrtf(var) : 0.0f;
    }

    const float skip_threshold  = mean + skip_sigma * stddev;
    const float sharp_threshold = mean + sharp_sigma * stddev;

    for (int il = 0; il < n_layers; ++il) {
        // Always skip layers in the base skip set
        if (base_skip_set.count(il)) {
            plan.ops.push_back({STREAM_SKIP, il, STREAM_Q_BLURRY});
            continue;
        }

        // Protect first/last N layers
        if (il < plan.min_protect_layers || il >= n_layers - plan.min_protect_layers) {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY});
            continue;
        }

        float score = ema_scores.count(il) ? ema_scores.at(il) : mean;

        if (score < skip_threshold) {
            plan.ops.push_back({STREAM_SKIP, il, STREAM_Q_BLURRY});
        } else if (score > sharp_threshold) {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_SHARP});
        } else {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY});
        }
    }
    return plan;
}

llama_execution_plan llama_plan_from_entropy(
        int n_layers,
        const std::unordered_map<int32_t, std::pair<float,int32_t>> & layer_entropy,
        float entropy_skip_threshold,
        float entropy_sharp_threshold) {

    llama_execution_plan plan;
    plan.ops.reserve(n_layers);

    for (int il = 0; il < n_layers; ++il) {
        float avg_entropy = 0.0f;
        if (layer_entropy.count(il)) {
            auto & [sum, cnt] = layer_entropy.at(il);
            if (cnt > 0) avg_entropy = sum / (float)cnt;
        }

        // Protect first/last N layers
        if (il < plan.min_protect_layers || il >= n_layers - plan.min_protect_layers) {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY});
            continue;
        }

        if (avg_entropy < entropy_skip_threshold) {
            plan.ops.push_back({STREAM_SKIP, il, STREAM_Q_BLURRY});
        } else if (avg_entropy > entropy_sharp_threshold) {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_SHARP});
        } else {
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY});
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Fractal: probe-to-depth mapping
// ---------------------------------------------------------------------------

static int32_t probe_score_to_depth(
        float score, float mean, float stddev, int32_t max_depth) {
    if (max_depth <= 0) return 0;
    if (stddev < 1e-8f) return max_depth / 2;  // degenerate case

    if (score < mean - 0.5f * stddev) return 0;          // blurry is fine
    if (score > mean + 1.5f * stddev) return max_depth;   // full zoom

    // Linear interpolation between depth 1 and max_depth-1
    float t = (score - (mean - 0.5f * stddev)) / (2.0f * stddev);
    t = std::max(0.0f, std::min(1.0f, t));
    return 1 + (int32_t)(t * (float)(max_depth - 1));
}

llama_execution_plan llama_plan_from_probe_depth(
        int n_layers,
        int max_depth,
        const std::unordered_map<int32_t, float> & ema_scores,
        const std::unordered_set<int32_t> & base_skip_set) {

    llama_execution_plan plan;
    plan.ops.reserve(n_layers);
    plan.ema_scores = ema_scores;

    // Compute statistics
    float sum = 0.0f, sum_sq = 0.0f;
    int count = 0;
    for (auto & [lid, sc] : ema_scores) {
        sum += sc; sum_sq += sc * sc; count++;
    }
    float mean = count > 0 ? sum / (float)count : 0.0f;
    float var  = count > 0 ? (sum_sq / (float)count) - (mean * mean) : 0.0f;
    float stddev = var > 0.0f ? sqrtf(var) : 0.0f;

    for (int il = 0; il < n_layers; ++il) {
        if (base_skip_set.count(il)) {
            plan.ops.push_back({STREAM_SKIP, il, STREAM_Q_BLURRY, -1});
            continue;
        }

        if (il < plan.min_protect_layers || il >= n_layers - plan.min_protect_layers) {
            // Protected layers: always execute at depth 1 minimum
            plan.ops.push_back({STREAM_EXEC, il, STREAM_Q_BLURRY,
                                std::min(1, max_depth)});
            continue;
        }

        float score = ema_scores.count(il) ? ema_scores.at(il) : mean;
        int32_t depth = probe_score_to_depth(score, mean, stddev, max_depth);

        llama_stream_quality q = STREAM_Q_BLURRY;
        if (depth > 0) q = STREAM_Q_SHARP;  // any correction = "sharp" in legacy terms

        llama_stream_op_type type = (depth == 0) ? STREAM_SKIP : STREAM_EXEC;
        plan.ops.push_back({type, il, q, depth});
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Hidden state ping-pong buffers
// ---------------------------------------------------------------------------

bool llama_hidden_state_buffers_init(
        llama_hidden_state_buffers & hs,
        ggml_backend_t               backend,
        int64_t                      n_embd,
        int32_t                      max_tokens) {
    const size_t tensor_meta_size = 2 * ggml_tensor_overhead() + 256;
    struct ggml_init_params params = {
        /*.mem_size   =*/ tensor_meta_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    hs.ctx = ggml_init(params);
    if (!hs.ctx) return false;

    hs.tensor[0] = ggml_new_tensor_2d(hs.ctx, GGML_TYPE_F32, n_embd, max_tokens);
    hs.tensor[1] = ggml_new_tensor_2d(hs.ctx, GGML_TYPE_F32, n_embd, max_tokens);
    ggml_set_name(hs.tensor[0], "hs_ping");
    ggml_set_name(hs.tensor[1], "hs_pong");

    auto * buft = ggml_backend_get_default_buffer_type(backend);
    const size_t buf_size = ggml_nbytes(hs.tensor[0]);

    for (int i = 0; i < 2; ++i) {
        hs.buf[i] = ggml_backend_buft_alloc_buffer(buft, buf_size);
        if (!hs.buf[i]) {
            llama_hidden_state_buffers_free(hs);
            return false;
        }
        ggml_backend_tensor_alloc(hs.buf[i], hs.tensor[i],
            (char *)ggml_backend_buffer_get_base(hs.buf[i]));
    }

    hs.active = 0;
    return true;
}

void llama_hidden_state_buffers_free(llama_hidden_state_buffers & hs) {
    for (int i = 0; i < 2; ++i) {
        if (hs.buf[i]) {
            ggml_backend_buffer_free(hs.buf[i]);
            hs.buf[i] = nullptr;
        }
        hs.tensor[i] = nullptr;
    }
    if (hs.ctx) {
        ggml_free(hs.ctx);
        hs.ctx = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Micro-graph lifecycle
// ---------------------------------------------------------------------------

void llama_micro_graph_free(llama_micro_graph & mg) {
    if (mg.ctx) {
        ggml_free(mg.ctx);
        mg.ctx = nullptr;
    }
    mg.graph = nullptr;
    mg.output = nullptr;
}

// ---------------------------------------------------------------------------
// Streaming decode: the core interpreter loop
// ---------------------------------------------------------------------------

int llama_streaming_decode(
        llama_context      & lctx,
        llama_batch          batch,
        llama_execution_plan * plan) {

    const auto & hparams = lctx.model.hparams;
    const int n_transformer_layers = (int)hparams.n_layer - hparams.nextn_predict_layers;
    const int n_tokens = batch.n_tokens;

    if (n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    llama_execution_plan default_plan;
    if (!plan) {
        default_plan = llama_plan_sequential(n_transformer_layers);
        plan = &default_plan;
    }

    const int max_iterations = plan->max_loop_iterations + 1;

    LLAMA_LOG_INFO("%s: %s\n", __func__, plan->summary().c_str());

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const int64_t t_iter_start = ggml_time_us();

        // ---------------------------------------------------------------
        // Configure skip_layers and jit_priority_layers from the plan
        // ---------------------------------------------------------------
        lctx.skip_layers.clear();
        lctx.jit_priority_layers.clear();
        lctx.skip_layers_epoch++;

        if (iteration == 1) {
            for (const auto & op : plan->ops) {
                if (op.type == STREAM_SKIP || op.type == STREAM_EXEC_ATTN_ONLY) {
                    lctx.skip_layers.insert(op.layer_id);
                }
                if (op.quality == STREAM_Q_SHARP) {
                    lctx.jit_priority_layers.insert(op.layer_id);
                }
            }
            plan->pc = plan->ops.size();

            // Save checkpoint before first pass for loop-back comparison
            plan->save_checkpoint();
        } else {
            // Subsequent passes: only execute SHARPEN ops
            std::unordered_set<int32_t> sharpen_layers;
            for (size_t i = plan->pc; i < plan->ops.size(); ++i) {
                if (plan->ops[i].type == STREAM_SHARPEN) {
                    sharpen_layers.insert(plan->ops[i].layer_id);
                }
            }

            if (sharpen_layers.empty()) {
                LLAMA_LOG_INFO("%s: no sharpen ops, stopping at iter %d\n",
                               __func__, iteration);
                break;
            }

            // Check if previous loop-back actually helped
            if (iteration > 2 && !plan->check_loop_improvement()) {
                LLAMA_LOG_INFO("%s: loop-back improvement below threshold (%.1f%%), stopping\n",
                               __func__, plan->min_improvement_threshold * 100.0f);
                break;
            }

            LLAMA_LOG_INFO("%s: iter %d, sharpening %zu layers\n",
                           __func__, iteration, sharpen_layers.size());

            for (int il = 0; il < n_transformer_layers; ++il) {
                if (!sharpen_layers.count(il)) {
                    lctx.skip_layers.insert(il);
                }
            }
            lctx.jit_priority_layers = sharpen_layers;
            plan->pc = plan->ops.size();

            // Save checkpoint before this pass
            plan->save_checkpoint();

            // Clear KV for re-execution (generation only)
            if (n_tokens <= 4 && batch.pos) {
                for (int i = 0; i < n_tokens; ++i) {
                    llama_kv_cache_seq_rm(
                        &lctx,
                        batch.seq_id ? batch.seq_id[i][0] : 0,
                        batch.pos[i], batch.pos[i] + 1);
                }
            }
        }

        // Enable entropy + routing recording for multi-signal replanning
        const bool had_recording = lctx.router_recording;
        const bool had_entropy   = lctx.router_record_entropy;
        lctx.router_recording      = true;
        lctx.router_record_entropy = true;

        if (lctx.gate_probe_ctx) {
            llama_gate_probe_clear_scores(&lctx);
        }

        // ---------------------------------------------------------------
        // Decode
        // ---------------------------------------------------------------
        const int ret = llama_decode(&lctx, batch);
        if (ret != 0) {
            LLAMA_LOG_ERROR("%s: llama_decode failed (%d) on iter %d\n",
                            __func__, ret, iteration);
            lctx.router_recording      = had_recording;
            lctx.router_record_entropy = had_entropy;
            return ret;
        }

        const double t_iter_ms = (ggml_time_us() - t_iter_start) / 1000.0;

        // ---------------------------------------------------------------
        // Collect all signals and replan
        // ---------------------------------------------------------------
        bool has_new_sharpen_ops = false;

        if (iteration <= plan->max_loop_iterations) {
            // Build per-layer signals from all available data
            for (auto & [layer_id, score_pair] : lctx.gate_probe_layer_scores) {
                llama_layer_signal signal;
                signal.probe_score = score_pair.second > 0
                    ? score_pair.first / (float)score_pair.second : 0.0f;

                // Entropy data
                if (lctx.router_layer_entropy.count(layer_id)) {
                    auto & [esum, ecnt] = lctx.router_layer_entropy[layer_id];
                    signal.avg_entropy = ecnt > 0 ? esum / (float)ecnt : 0.0f;
                }

                // Expert diversity data
                if (lctx.router_expert_sets.count(layer_id)) {
                    signal.n_unique_experts = (int32_t)lctx.router_expert_sets[layer_id].size();
                    signal.n_total_experts = (int32_t)hparams.n_expert_used * n_tokens;
                }

                plan->replan_multi(layer_id, signal, n_transformer_layers);
            }

            // HALT check: if max token entropy is very low, model is confident
            if (iteration == 1 && !lctx.router_token_entropy.empty()) {
                float max_entropy = *std::max_element(
                    lctx.router_token_entropy.begin(),
                    lctx.router_token_entropy.end());
                if (max_entropy < plan->halt_entropy_threshold) {
                    LLAMA_LOG_INFO("%s: HALT — token entropy %.2f < threshold %.2f, skipping refinement\n",
                                   __func__, max_entropy, plan->halt_entropy_threshold);
                    plan->clear_pending_sharpens();
                }
            }

            // Check for new SHARPEN ops
            for (size_t i = plan->pc; i < plan->ops.size(); ++i) {
                if (plan->ops[i].type == STREAM_SHARPEN) {
                    has_new_sharpen_ops = true;
                    break;
                }
            }
        }

        // Restore recording state
        lctx.router_recording      = had_recording;
        lctx.router_record_entropy = had_entropy;

        // Log iteration summary
        {
            int n_sharp_pending = 0;
            for (size_t i = plan->pc; i < plan->ops.size(); ++i) {
                if (plan->ops[i].type == STREAM_SHARPEN) n_sharp_pending++;
            }
            LLAMA_LOG_INFO("%s: iter %d done in %.1fms, %d skip, %d jit, %d sharpen-pending\n",
                           __func__, iteration, t_iter_ms,
                           (int)lctx.skip_layers.size(),
                           (int)lctx.jit_priority_layers.size(),
                           n_sharp_pending);
        }

        if (!has_new_sharpen_ops) {
            LLAMA_LOG_INFO("%s: completed in %d iteration(s)\n", __func__, iteration);
            break;
        }
    }

    return 0;
}
