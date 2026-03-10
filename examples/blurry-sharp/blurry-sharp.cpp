//
// blurry-sharp example
//
// Demonstrates the Blurry->Sharp overlay system: load a low-quality (blurry)
// GGUF model for speed, then selectively hotswap layer weights from a
// high-quality (sharp) GGUF model during inference.
//
// Now uses the standard gpt_params infrastructure, so ALL regular flags work:
//   -ngl, --flash-attn, --jinja, -c, -t, -b, --mlock, --numa, etc.
//
// Blurry-sharp specific flags:
//   -ms, --sharp FILE          Path to sharp (high-quality) GGUF model
//   --bs-router STRATEGY       Router: always|never|norm (default: always)
//   --bs-confidence FLOAT      Min confidence for norm router (default: 0.8)
//   --bs-max-sharp-layers N    Max simultaneously sharpened layers (0=unlimited)
//   --bs-memory-budget-mb N    Memory budget in MiB for backups (0=unlimited)
//   --bs-no-restore            Keep sharp weights between layers
//   --bs-verbose               Verbose overlay logging
//   --bs-no-mmap               Disable mmap for sharp file
//   --bs-precache-ram           Pre-read sharp data into anonymous heap (swap-backed)
//   --bs-stage-swap             After precache, move pages to swap (free RAM)
//   --bs-lazy-swap              Lazy per-layer swap staging (instant startup, no precache)
//   --bs-retain-mmap            Keep mmap pages cached in RAM (fast repeat sharpening)
//   --bs-compare               Run blurry-only then blurry+sharp and compare
//   --bs-dynamic               Per-token entropy-based dynamic sharpening
//   --bs-entropy-threshold F   Logit entropy above this triggers sharpening (default: 3.0)
//   --bs-dynamic-top-k N       Number of layers to sharpen when uncertain (default: 8)
//   --bs-probe-interval N      Every N tokens, probe sharp model (default: 0=disabled)
//   --bs-probe-hold N          After probe disagreement, stay sharp for N tokens (default: 16)
//   --bs-speculative           Speculative verification: draft blurry, verify sharp
//   --bs-spec-draft N          Draft tokens per speculative batch (default: 8)
//   --bs-combined              Combined adaptive mode (entropy+probe+speculative)
//   --bs-combined-probe-stride N  Within draft, probe every N tokens (default: 4)
//   --bs-moe-combination       MoE combination expert mode (router hooks + exact experts)
//   --bs-moe-top-k N           Override active expert count (0=use model default)
//   --bs-allow-layers L1,L2,.. Only sharpen these layers
//   --bs-deny-layers  L1,L2,.. Never sharpen these layers
//
// Usage:
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf -p "Hello world" -n 64
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf -ngl 99 -fa on --jinja -n 128
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-compare -n 64
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-dynamic --bs-entropy-threshold 2.5
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-combined --bs-spec-draft 12
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-moe-combination --bs-spec-draft 12
//
// Memory-tier management (VRAM > RAM > Swap > Disk):
//   # Pre-read sharp data into RAM, let OS swap-manage it (uses RAM then swap):
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-precache-ram -n 128
//
//   # Pre-read into RAM, then proactively stage to swap (frees RAM for blurry/KV):
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-stage-swap -n 128
//
//   # Lazy per-layer swap staging (instant startup, no 30-min precache):
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-lazy-swap -n 128
//
//   # Combined adaptive mode with full memory-tier support:
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-combined --bs-stage-swap -n 128
//   ./blurry-sharp -m model-q4.gguf --sharp model-q8.gguf --bs-combined --bs-lazy-swap -n 128
//

#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>


// ---------------------------------------------------------------------------
// Generation result
// ---------------------------------------------------------------------------

struct generation_result {
    std::string              text;
    std::vector<llama_token> tokens;
    double                   elapsed_s     = 0.0;
    double                   tokens_per_s  = 0.0;
    int                      n_sharp_steps = 0;
    int                      n_blurry_steps = 0;
    int                      n_probes = 0;
    int                      n_probe_disagreements = 0;
    int                      n_draft_total = 0;
    int                      n_accepted_total = 0;
    int64_t                  sharp_bytes_read = 0;       // total sharp bytes read (MoE tracking)
    int64_t                  full_sharp_bytes  = 0;       // what full-layer sharpening would read
};

// ---------------------------------------------------------------------------
// Compute Shannon entropy of logits (in nats). Higher = more uncertain.
// ---------------------------------------------------------------------------
static float compute_logit_entropy(const float * logits, int n_vocab) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += exp((double)(logits[i] - max_logit));
    }
    double log_sum_exp = (double)max_logit + log(sum_exp);

    double entropy = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        double log_p = (double)logits[i] - log_sum_exp;
        double p = exp(log_p);
        if (p > 1e-30) {
            entropy -= p * log_p;
        }
    }
    return (float)entropy;
}

// ---------------------------------------------------------------------------
// Dynamic generation: per-token entropy-based sharpening
// ---------------------------------------------------------------------------
static generation_result run_dynamic_generation(
        llama_model   * model,
        llama_context * ctx,
        llama_blurry_sharp_context * bsctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        float entropy_threshold,
        int top_k_layers,
        bool verbose) {

    generation_result result;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(model);

    llama_blurry_sharp_state st = llama_blurry_sharp_get_state(bsctx);
    int n_layers = st.n_layers_total;

    // Build priority layers: first half + last half (research shows these matter most)
    std::vector<int> priority_layers;
    {
        int half = std::max(1, top_k_layers / 2);
        for (int i = 0; i < half && i < n_layers; ++i) {
            priority_layers.push_back(i);
        }
        for (int i = std::max(0, n_layers - half); i < n_layers; ++i) {
            bool dup = false;
            for (int p : priority_layers) { if (p == i) { dup = true; break; } }
            if (!dup) priority_layers.push_back(i);
        }
    }

    if (verbose) {
        fprintf(stderr, "  Dynamic mode: entropy_threshold=%.2f, top_k=%d, priority_layers=[",
                entropy_threshold, top_k_layers);
        for (size_t i = 0; i < priority_layers.size(); ++i) {
            fprintf(stderr, "%s%d", i > 0 ? "," : "", priority_layers[i]);
        }
        fprintf(stderr, "]\n");
    }

    if ((int)prompt_tokens.size() + n_predict > n_ctx) {
        n_predict = n_ctx - (int)prompt_tokens.size();
        if (n_predict <= 0) {
            fprintf(stderr, "dynamic: prompt longer than context\n");
            return result;
        }
    }

    // Prompt eval
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (int)i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "dynamic: llama_decode() failed on prompt\n");
        llama_batch_free(batch);
        return result;
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;
    bool currently_sharp = false;

    const auto t_start = ggml_time_us();

    while (n_cur < (int)prompt_tokens.size() + n_predict) {
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        float entropy = compute_logit_entropy(logits, n_vocab);
        bool want_sharp = (entropy > entropy_threshold);

        if (want_sharp && !currently_sharp) {
            constexpr int PREFETCH_LOOKAHEAD = 4;
            const int n_prio = (int)priority_layers.size();
            for (int i = 0; i < std::min(PREFETCH_LOOKAHEAD, n_prio); ++i) {
                llama_blurry_sharp_prefetch_layer(bsctx, priority_layers[i]);
            }
            int n_sharpened = 0;
            for (int i = 0; i < n_prio; ++i) {
                if (i + PREFETCH_LOOKAHEAD < n_prio) {
                    llama_blurry_sharp_prefetch_layer(bsctx, priority_layers[i + PREFETCH_LOOKAHEAD]);
                }
                if (llama_blurry_sharp_apply_layer(bsctx, priority_layers[i]) == 0) {
                    ++n_sharpened;
                }
            }
            // Pull CPU zero-copy tensor pages into RAM so the decode
            // doesn't stall on page faults when RAM is available.
            llama_blurry_sharp_warm_live_pages(bsctx);
            currently_sharp = true;
            if (verbose) {
                fprintf(stderr, "  [token %3d] entropy=%.2f > %.2f -> SHARPEN %d layers\n",
                        n_decode, entropy, entropy_threshold, n_sharpened);
            }
        } else if (!want_sharp && currently_sharp) {
            llama_blurry_sharp_restore_all(bsctx);
            currently_sharp = false;
            if (verbose) {
                fprintf(stderr, "  [token %3d] entropy=%.2f < %.2f -> BLURRY (restored)\n",
                        n_decode, entropy, entropy_threshold);
            }
        } else if (verbose && n_decode % 16 == 0) {
            fprintf(stderr, "  [token %3d] entropy=%.2f  %s\n",
                    n_decode, entropy, currently_sharp ? "(sharp)" : "(blurry)");
        }

        // Greedy sample
        llama_token best_id = 0;
        float       best_logit = logits[0];
        for (llama_token id = 1; id < n_vocab; ++id) {
            if (logits[id] > best_logit) {
                best_logit = logits[id];
                best_id    = id;
            }
        }

        if (llama_token_is_eog(model, best_id)) {
            break;
        }

        result.tokens.push_back(best_id);
        result.text += common_token_to_piece(ctx, best_id);

        if (currently_sharp) {
            result.n_sharp_steps++;
        } else {
            result.n_blurry_steps++;
        }

        common_batch_clear(batch);
        common_batch_add(batch, best_id, n_cur, { 0 }, true);
        n_decode++;
        n_cur++;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "dynamic: llama_decode() failed at token %d\n", n_decode);
            break;
        }
    }

    if (currently_sharp) {
        llama_blurry_sharp_restore_all(bsctx);
    }

    const auto t_end = ggml_time_us();
    result.elapsed_s    = (t_end - t_start) / 1e6;
    result.tokens_per_s = (result.elapsed_s > 0.0) ? n_decode / result.elapsed_s : 0.0;

    llama_batch_free(batch);
    return result;
}

// ---------------------------------------------------------------------------
// Probe-based generation: periodically ask the sharp model if blurry is right
//
// Every `probe_interval` tokens, temporarily sharpen all priority layers,
// re-decode the current token, and compare top-1 predictions.
// If sharp disagrees -> stay sharp for `probe_hold` tokens.
// If sharp agrees   -> restore to blurry for speed.
//
// This lets the sharp model say "hey, you're wrong" without relying on
// the blurry model's self-assessment (which fails when it's confidently wrong).
// ---------------------------------------------------------------------------
static generation_result run_probe_generation(
        llama_model   * model,
        llama_context * ctx,
        llama_blurry_sharp_context * bsctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        int probe_interval,
        int probe_hold,
        int top_k_layers,
        bool verbose) {

    generation_result result;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(model);

    llama_blurry_sharp_state st = llama_blurry_sharp_get_state(bsctx);
    int n_layers = st.n_layers_total;

    // Build priority layers: first half + last half
    std::vector<int> priority_layers;
    {
        int half = std::max(1, top_k_layers / 2);
        for (int i = 0; i < half && i < n_layers; ++i) {
            priority_layers.push_back(i);
        }
        for (int i = std::max(0, n_layers - half); i < n_layers; ++i) {
            bool dup = false;
            for (int p : priority_layers) { if (p == i) { dup = true; break; } }
            if (!dup) priority_layers.push_back(i);
        }
    }

    if (verbose) {
        fprintf(stderr, "  Probe mode: interval=%d, hold=%d, top_k=%d, priority_layers=[",
                probe_interval, probe_hold, top_k_layers);
        for (size_t i = 0; i < priority_layers.size(); ++i) {
            fprintf(stderr, "%s%d", i > 0 ? "," : "", priority_layers[i]);
        }
        fprintf(stderr, "]\n");
    }

    if ((int)prompt_tokens.size() + n_predict > n_ctx) {
        n_predict = n_ctx - (int)prompt_tokens.size();
        if (n_predict <= 0) {
            fprintf(stderr, "probe: prompt longer than context\n");
            return result;
        }
    }

    // Prompt eval (blurry)
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (int)i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "probe: llama_decode() failed on prompt\n");
        llama_batch_free(batch);
        return result;
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;
    bool currently_sharp = false;
    int  sharp_hold_remaining = 0;  // countdown: stay sharp for this many more tokens
    int  n_probes = 0;
    int  n_disagreements = 0;

    const auto t_start = ggml_time_us();

    while (n_cur < (int)prompt_tokens.size() + n_predict) {
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        // Greedy sample from current logits (blurry or sharp, whatever is active)
        llama_token blurry_best = 0;
        float       blurry_best_logit = logits[0];
        for (llama_token id = 1; id < n_vocab; ++id) {
            if (logits[id] > blurry_best_logit) {
                blurry_best_logit = logits[id];
                blurry_best = id;
            }
        }

        // Should we probe on this step?
        bool should_probe = (probe_interval > 0 &&
                             !currently_sharp &&
                             n_decode > 0 &&
                             (n_decode % probe_interval) == 0);

        llama_token chosen_id = blurry_best;

        if (should_probe) {
            n_probes++;

            // Sharpen priority layers with lookahead prefetch
            int n_sharpened = 0;
            {
                constexpr int PREFETCH_LOOKAHEAD = 4;
                const int n_prio = (int)priority_layers.size();
                for (int i = 0; i < std::min(PREFETCH_LOOKAHEAD, n_prio); ++i) {
                    llama_blurry_sharp_prefetch_layer(bsctx, priority_layers[i]);
                }
                for (int i = 0; i < n_prio; ++i) {
                    if (i + PREFETCH_LOOKAHEAD < n_prio) {
                        llama_blurry_sharp_prefetch_layer(bsctx, priority_layers[i + PREFETCH_LOOKAHEAD]);
                    }
                    if (llama_blurry_sharp_apply_layer(bsctx, priority_layers[i]) == 0) {
                        ++n_sharpened;
                    }
                }
            }
            // Pull CPU zero-copy tensor pages into RAM so the decode
            // doesn't stall on page faults when RAM is available.
            llama_blurry_sharp_warm_live_pages(bsctx);

            // Re-decode the same token position to get sharp logits.
            // We need to re-evaluate with the now-sharp weights.
            // The KV cache has the blurry-computed KV for previous positions,
            // but the current position's logits come from the current weights.
            // We re-decode the last accepted token to get sharp's opinion.
            if (n_sharpened > 0 && result.tokens.size() > 0) {
                llama_token last_token = result.tokens.back();

                // Re-decode last token with sharp weights
                common_batch_clear(batch);
                common_batch_add(batch, last_token, n_cur - 1, { 0 }, true);

                if (llama_decode(ctx, batch) == 0) {
                    float * sharp_logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

                    // Sharp model's top-1
                    llama_token sharp_best = 0;
                    float sharp_best_logit = sharp_logits[0];
                    for (llama_token id = 1; id < n_vocab; ++id) {
                        if (sharp_logits[id] > sharp_best_logit) {
                            sharp_best_logit = sharp_logits[id];
                            sharp_best = id;
                        }
                    }

                    if (sharp_best != blurry_best) {
                        // Sharp disagrees! Use sharp's token and stay sharp.
                        n_disagreements++;
                        chosen_id = sharp_best;
                        currently_sharp = true;
                        sharp_hold_remaining = probe_hold;

                        if (verbose) {
                            fprintf(stderr, "  [token %3d] PROBE: sharp disagrees! "
                                    "blurry=\"%s\"(%d) vs sharp=\"%s\"(%d) -> using sharp, hold %d\n",
                                    n_decode,
                                    common_token_to_piece(ctx, blurry_best).c_str(), blurry_best,
                                    common_token_to_piece(ctx, sharp_best).c_str(), sharp_best,
                                    probe_hold);
                        }
                    } else {
                        // Sharp agrees -> restore blurry for speed
                        llama_blurry_sharp_restore_all(bsctx);
                        currently_sharp = false;

                        if (verbose) {
                            fprintf(stderr, "  [token %3d] PROBE: sharp agrees (\"%s\") -> stay blurry\n",
                                    n_decode,
                                    common_token_to_piece(ctx, blurry_best).c_str());
                        }
                    }
                } else {
                    // Re-decode failed, just restore and continue
                    llama_blurry_sharp_restore_all(bsctx);
                }
            } else {
                llama_blurry_sharp_restore_all(bsctx);
            }
        }

        // If we're in a sharp hold period, count down
        if (currently_sharp && sharp_hold_remaining > 0) {
            sharp_hold_remaining--;
            if (sharp_hold_remaining == 0) {
                // Hold expired, restore to blurry
                llama_blurry_sharp_restore_all(bsctx);
                currently_sharp = false;
                if (verbose) {
                    fprintf(stderr, "  [token %3d] hold expired -> BLURRY (restored)\n", n_decode);
                }
            }
        }

        if (llama_token_is_eog(model, chosen_id)) {
            break;
        }

        result.tokens.push_back(chosen_id);
        result.text += common_token_to_piece(ctx, chosen_id);

        if (currently_sharp) {
            result.n_sharp_steps++;
        } else {
            result.n_blurry_steps++;
        }

        // Decode next token
        common_batch_clear(batch);
        common_batch_add(batch, chosen_id, n_cur, { 0 }, true);
        n_decode++;
        n_cur++;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "probe: llama_decode() failed at token %d\n", n_decode);
            break;
        }

        if (verbose && n_decode % 32 == 0 && !should_probe) {
            fprintf(stderr, "  [token %3d] %s (probes=%d, disagreements=%d)\n",
                    n_decode, currently_sharp ? "(sharp)" : "(blurry)",
                    n_probes, n_disagreements);
        }
    }

    if (currently_sharp) {
        llama_blurry_sharp_restore_all(bsctx);
    }

    const auto t_end = ggml_time_us();
    result.elapsed_s    = (t_end - t_start) / 1e6;
    result.tokens_per_s = (result.elapsed_s > 0.0) ? n_decode / result.elapsed_s : 0.0;

    if (verbose) {
        fprintf(stderr, "\n  Probe stats: %d probes, %d disagreements (%.1f%% disagree rate)\n",
                n_probes, n_disagreements,
                n_probes > 0 ? 100.0 * n_disagreements / n_probes : 0.0);
    }

    llama_batch_free(batch);
    return result;
}

// ---------------------------------------------------------------------------
// Speculative verification: draft with blurry, verify with sharp
//
// Inspired by speculative decoding.  The blurry model is the "draft" model
// and the sharp overlay is the "target" model.
//
// Algorithm:
//   1. Generate `draft_n` tokens greedily with blurry weights (fast)
//   2. Sharpen all priority layers
//   3. Feed all draft_n tokens through the sharp model in one batch
//   4. Compare sharp vs blurry top-1 at each position
//   5. Accept the longest prefix where they agree
//   6. At the first disagreement, take sharp's token instead
//   7. Restore blurry weights, clear KV beyond accepted prefix, continue
//
// This guarantees output quality >= sharp model for greedy decoding,
// while running blurry (fast) for most of the compute.
// ---------------------------------------------------------------------------
static generation_result run_speculative_generation(
        llama_model   * model,
        llama_context * ctx,
        llama_blurry_sharp_context * bsctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        int draft_n,
        int top_k_layers,
        bool verbose) {

    generation_result result;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(model);

    llama_blurry_sharp_state st = llama_blurry_sharp_get_state(bsctx);
    int n_layers = st.n_layers_total;

    // Build priority layers (sharpen all for verification quality)
    std::vector<int> all_layers;
    for (int i = 0; i < n_layers; ++i) {
        all_layers.push_back(i);
    }

    if (verbose) {
        fprintf(stderr, "  Speculative mode: draft_n=%d, verifying with all %d layers\n",
                draft_n, n_layers);
    }

    if ((int)prompt_tokens.size() + n_predict > n_ctx) {
        n_predict = n_ctx - (int)prompt_tokens.size();
        if (n_predict <= 0) {
            fprintf(stderr, "speculative: prompt longer than context\n");
            return result;
        }
    }

    // Prompt eval (blurry)
    llama_batch batch = llama_batch_init(std::max(512, draft_n + 1), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (int)i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "speculative: llama_decode() failed on prompt\n");
        llama_batch_free(batch);
        return result;
    }

    int n_cur    = batch.n_tokens;  // next position to fill
    int n_decode = 0;
    int n_draft_total   = 0;
    int n_accepted_total = 0;

    const auto t_start = ggml_time_us();

    while (n_cur < (int)prompt_tokens.size() + n_predict) {
        // ---- Phase 1: Draft `draft_n` tokens with blurry model (greedy) ----
        std::vector<llama_token> draft_tokens;
        int draft_limit = std::min(draft_n,
                                   (int)prompt_tokens.size() + n_predict - n_cur);

        for (int d = 0; d < draft_limit; ++d) {
            float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            llama_token best_id = 0;
            float best_logit = logits[0];
            for (llama_token id = 1; id < n_vocab; ++id) {
                if (logits[id] > best_logit) {
                    best_logit = logits[id];
                    best_id = id;
                }
            }

            if (llama_token_is_eog(model, best_id)) {
                break;
            }

            draft_tokens.push_back(best_id);

            // Continue drafting (decode this token to get logits for next)
            if (d + 1 < draft_limit) {
                common_batch_clear(batch);
                common_batch_add(batch, best_id, n_cur + d, { 0 }, true);
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "speculative: draft decode failed at draft token %d\n", d);
                    break;
                }
            }
        }

        if (draft_tokens.empty()) {
            break;  // EOS or decode failure
        }

        n_draft_total += (int)draft_tokens.size();

        // ---- Phase 2: Verify with sharp model ----
        // Sharpen all layers with lookahead prefetch
        {
            constexpr int PREFETCH_LOOKAHEAD = 4;
            const int n_all = (int)all_layers.size();

            for (int i = 0; i < std::min(PREFETCH_LOOKAHEAD, n_all); ++i) {
                llama_blurry_sharp_prefetch_layer(bsctx, all_layers[i]);
            }

            int n_sharpened = 0;
            for (int i = 0; i < n_all; ++i) {
                if (i + PREFETCH_LOOKAHEAD < n_all) {
                    llama_blurry_sharp_prefetch_layer(bsctx, all_layers[i + PREFETCH_LOOKAHEAD]);
                }
                if (llama_blurry_sharp_apply_layer(bsctx, all_layers[i]) == 0) {
                    ++n_sharpened;
                }
            }
        }
        // Pull CPU zero-copy tensor pages into RAM so the verification
        // decode doesn't stall on page faults when RAM is available.
        llama_blurry_sharp_warm_live_pages(bsctx);

        // We need to remove the draft tokens from the KV cache so we can
        // re-evaluate them with sharp weights.  The KV cache currently has
        // entries for positions [0, n_cur + draft_tokens.size()).
        // Remove positions [n_cur, n_cur + draft_tokens.size()) so the
        // sharp verification pass re-computes them.
        llama_kv_cache_seq_rm(ctx, 0, n_cur, n_cur + (int)draft_tokens.size());

        // Build a batch with all draft tokens for parallel evaluation
        common_batch_clear(batch);
        for (int d = 0; d < (int)draft_tokens.size(); ++d) {
            common_batch_add(batch, draft_tokens[d], n_cur + d, { 0 }, true);
        }

        int n_accepted = 0;

        if (llama_decode(ctx, batch) == 0) {
            // Compare sharp's top-1 at each position against draft tokens
            // Position d in the batch gives logits that predict the token AFTER draft_tokens[d].
            // But we want to verify: does sharp agree that draft_tokens[d] is correct?
            //
            // Actually, for verification we need to check: given the context up to position
            // n_cur+d-1, does sharp's top-1 == draft_tokens[d]?
            //
            // The logits at batch position d predict the token at position n_cur+d+1.
            // So logits at position d verify draft_tokens[d+1] (if it exists).
            //
            // For the first draft token (d=0), we need the logits from BEFORE this batch
            // (the prompt eval or previous iteration's last logits). But we've already
            // moved past that. So instead, we just check if the sharp model's continuation
            // agrees with blurry's:
            //
            // A simpler approach: accept all draft tokens where sharp's top-1 continuation
            // matches the next draft token.

            for (int d = 0; d < (int)draft_tokens.size(); ++d) {
                float * sharp_logits = llama_get_logits_ith(ctx, d);

                llama_token sharp_best = 0;
                float sharp_best_logit = sharp_logits[0];
                for (llama_token id = 1; id < n_vocab; ++id) {
                    if (sharp_logits[id] > sharp_best_logit) {
                        sharp_best_logit = sharp_logits[id];
                        sharp_best = id;
                    }
                }

                if (d + 1 < (int)draft_tokens.size()) {
                    // Check: does sharp predict draft_tokens[d+1] as next?
                    if (sharp_best == draft_tokens[d + 1]) {
                        // Accept this token
                        n_accepted++;
                        result.tokens.push_back(draft_tokens[d]);
                        result.text += common_token_to_piece(ctx, draft_tokens[d]);
                    } else {
                        // Accept this token (it's already in context) but take
                        // sharp's prediction for the next position
                        n_accepted++;
                        result.tokens.push_back(draft_tokens[d]);
                        result.text += common_token_to_piece(ctx, draft_tokens[d]);

                        // Add sharp's token as the corrected next token
                        if (!llama_token_is_eog(model, sharp_best)) {
                            n_accepted++;
                            result.tokens.push_back(sharp_best);
                            result.text += common_token_to_piece(ctx, sharp_best);
                        }

                        if (verbose) {
                            fprintf(stderr, "  [draft %3d+%d] REJECT: blurry=\"%s\" vs sharp=\"%s\" "
                                    "-> accepted %d/%d draft tokens\n",
                                    n_decode, d + 1,
                                    common_token_to_piece(ctx, draft_tokens[d + 1]).c_str(),
                                    common_token_to_piece(ctx, sharp_best).c_str(),
                                    n_accepted, (int)draft_tokens.size());
                        }
                        break;
                    }
                } else {
                    // Last draft token — just accept it and take sharp's next prediction
                    n_accepted++;
                    result.tokens.push_back(draft_tokens[d]);
                    result.text += common_token_to_piece(ctx, draft_tokens[d]);

                    // Bonus token from sharp
                    if (!llama_token_is_eog(model, sharp_best)) {
                        n_accepted++;
                        result.tokens.push_back(sharp_best);
                        result.text += common_token_to_piece(ctx, sharp_best);
                    }

                    if (verbose && n_accepted == (int)draft_tokens.size() + 1) {
                        fprintf(stderr, "  [draft %3d] all %d draft tokens accepted + 1 bonus\n",
                                n_decode, (int)draft_tokens.size());
                    }
                }
            }
        } else {
            fprintf(stderr, "speculative: verification decode failed\n");
            // Fall back: accept all draft tokens (can't verify)
            for (auto tok : draft_tokens) {
                if (llama_token_is_eog(model, tok)) break;
                result.tokens.push_back(tok);
                result.text += common_token_to_piece(ctx, tok);
                n_accepted++;
            }
        }

        n_accepted_total += n_accepted;

        // ---- Phase 3: Restore blurry, fix KV cache, prepare for next draft ----
        llama_blurry_sharp_restore_all(bsctx);

        // Remove any extra KV entries beyond what we accepted
        int accepted_end = n_cur + n_accepted;
        if (accepted_end < n_cur + (int)draft_tokens.size()) {
            llama_kv_cache_seq_rm(ctx, 0, accepted_end, n_cur + (int)draft_tokens.size());
        }

        n_decode += n_accepted;
        n_cur     = (int)prompt_tokens.size() + (int)result.tokens.size();

        // Track sharp vs blurry steps
        result.n_blurry_steps += (int)draft_tokens.size();  // drafted with blurry
        result.n_sharp_steps  += n_accepted;                // verified with sharp

        // Check for EOS in accepted tokens
        bool hit_eos = false;
        for (int i = (int)result.tokens.size() - n_accepted; i < (int)result.tokens.size(); ++i) {
            if (i >= 0 && llama_token_is_eog(model, result.tokens[i])) {
                hit_eos = true;
                break;
            }
        }
        if (hit_eos) break;

        // Set up for next draft iteration: decode the last accepted token
        // to prime logits for the next draft sequence
        if (!result.tokens.empty()) {
            common_batch_clear(batch);
            common_batch_add(batch, result.tokens.back(), n_cur - 1, { 0 }, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "speculative: post-verify decode failed\n");
                break;
            }
        }
    }

    const auto t_end = ggml_time_us();
    result.elapsed_s    = (t_end - t_start) / 1e6;
    result.tokens_per_s = (result.elapsed_s > 0.0) ? n_decode / result.elapsed_s : 0.0;

    if (verbose) {
        fprintf(stderr, "\n  Speculative stats: %d drafted, %d accepted (%.1f%% acceptance)\n",
                n_draft_total, n_accepted_total,
                n_draft_total > 0 ? 100.0 * n_accepted_total / n_draft_total : 0.0);
    }

    llama_batch_free(batch);
    return result;
}

// ---------------------------------------------------------------------------
// MoE Combination Expert generation: speculative draft+verify with expert-
// selective sharpening.
//
// For Mixture-of-Experts models, this mode dramatically reduces the I/O cost
// of the sharp verification pass.  Instead of sharpening ALL expert weights
// in every layer (most of which are inactive for any given token), it creates
// "combination tensors" where only the router-selected experts are sharp.
//
// Algorithm:
//   1. DRAFT phase: Generate up to `draft_n` tokens greedily with blurry
//      (fast).  Router recording is enabled so we capture exactly which
//      experts the MoE gate selects on each token/layer.
//      Optionally use entropy-based early stopping.
//
//   2. VERIFY phase: For each layer, query the recorded router selections
//      to get the union of expert IDs that were active during drafting,
//      then call llama_blurry_sharp_apply_experts() with those exact IDs.
//      This creates combination tensors that are mostly blurry but with the
//      activated expert slices replaced by sharp data.  Then run the full
//      draft batch through the combination model in one pass and accept
//      the longest prefix where the combination model agrees with blurry.
//
//   3. RESTORE: Put blurry weights back, fix KV cache, continue.
//
// For a 128-expert top-8 model, this reads ~16x less data per layer than
// full sharpening, since only 8/128 experts need sharp data.
//
// Falls back to standard speculative verification for dense (non-MoE) models.
// ---------------------------------------------------------------------------
static generation_result run_moe_combination_generation(
        llama_model   * model,
        llama_context * ctx,
        llama_blurry_sharp_context * bsctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        int draft_n,
        float entropy_threshold,
        int top_k_layers,
        int moe_top_k_override,
        bool verbose) {

    generation_result result;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(model);

    llama_blurry_sharp_state st = llama_blurry_sharp_get_state(bsctx);
    int n_layers = st.n_layers_total;

    // Query MoE parameters
    int32_t n_expert       = llama_blurry_sharp_n_experts(bsctx);
    int32_t n_expert_used  = llama_blurry_sharp_n_experts_used(bsctx);
    if (moe_top_k_override > 0) {
        n_expert_used = moe_top_k_override;
    }

    bool is_moe = (n_expert > 1 && n_expert_used > 0);

    if (!is_moe) {
        fprintf(stderr, "  MoE combination mode: model is not MoE (n_expert=%d, n_expert_used=%d)\n",
                n_expert, n_expert_used);
        fprintf(stderr, "  Falling back to standard speculative verification.\n\n");
        // Fall back to standard speculative generation
        return run_speculative_generation(model, ctx, bsctx, prompt_tokens,
                                          n_predict, draft_n, top_k_layers, verbose);
    }

    fprintf(stderr, "  MoE combination expert mode:\n");
    fprintf(stderr, "    n_expert:       %d\n", n_expert);
    fprintf(stderr, "    n_expert_used:  %d (per token)\n", n_expert_used);
    fprintf(stderr, "    draft_n:        %d\n", draft_n);
    fprintf(stderr, "    entropy_thresh: %.2f\n", entropy_threshold);
    fprintf(stderr, "    router hooks:   ENABLED (per-token expert tracking)\n");
    fprintf(stderr, "    expert budget:  first 3 draft tokens (~%d experts vs %d total)\n",
            std::min(3 * (int)n_expert_used, (int)n_expert), n_expert);
    fprintf(stderr, "\n");

    if ((int)prompt_tokens.size() + n_predict > n_ctx) {
        n_predict = n_ctx - (int)prompt_tokens.size();
        if (n_predict <= 0) {
            fprintf(stderr, "moe_combination: prompt longer than context\n");
            return result;
        }
    }

    // Prompt eval (blurry)
    llama_batch batch = llama_batch_init(std::max(512, draft_n + 1), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (int)i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "moe_combination: llama_decode() failed on prompt\n");
        llama_batch_free(batch);
        return result;
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    // Stats
    int n_draft_total    = 0;
    int n_accepted_total = 0;
    int n_verify_batches = 0;
    int n_entropy_stops  = 0;
    int64_t total_sharp_bytes   = 0;   // actual bytes read from sharp
    int64_t total_full_bytes    = 0;   // what full sharpening would need
    int     n_unique_experts_total = 0;
    int     n_expert_slots_total   = 0; // n_layers * n_expert for comparison

    // Per-layer work descriptor for batch expert query + selective prefetch.
    // Defined outside the loop to reuse the vector allocation across iterations.
    struct layer_work {
        int     layer_idx;
        int32_t n_experts;               // >0 = MoE, 0 = dense/unreached
        std::vector<int32_t> expert_ids; // populated only when n_experts > 0
    };
    std::vector<layer_work> work_items;
    work_items.reserve(n_layers);

    // Previous iteration's layer indices — used for early prefetch.
    // MoE routing is highly stable across adjacent tokens: the same
    // layers tend to be active.  By prefetching last iteration's layers
    // at the START of drafting, the kernel has the full draft duration
    // (~100ms+) to populate pages before Phase 2 needs them.
    std::vector<int32_t> prev_layer_ids;

    const auto t_start = ggml_time_us();

    while (n_cur < (int)prompt_tokens.size() + n_predict) {
        // ================================================================
        // PHASE 0: EARLY TARGETED READAHEAD
        // Prefetch layers that were needed in the PREVIOUS iteration.
        // This is speculative but cheap: only issues madvise(WILLNEED)
        // for ~60 layers (~6 GB) instead of the full 200 GB model.
        // Pages are read asynchronously while the draft phase runs.
        // ================================================================
        if (!prev_layer_ids.empty()) {
            llama_blurry_sharp_prefetch_layers_parallel(
                bsctx, prev_layer_ids.data(),
                (int32_t)prev_layer_ids.size(), 4);
        }

        // ================================================================
        // PHASE 1: DRAFT with blurry model, monitoring entropy.
        // Router recording captures which experts the MoE gate selects
        // on each token so we can sharpen exactly those in Phase 2.
        // ================================================================
        llama_router_start_recording(ctx);

        std::vector<llama_token> draft_tokens;
        bool early_stop_entropy = false;
        int draft_limit = std::min(draft_n,
                                   (int)prompt_tokens.size() + n_predict - n_cur);

        for (int d = 0; d < draft_limit; ++d) {
            float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            // Check entropy for early stop
            if (entropy_threshold > 0) {
                float entropy = compute_logit_entropy(logits, n_vocab);
                if (entropy > entropy_threshold) {
                    early_stop_entropy = true;
                    n_entropy_stops++;
                    if (verbose) {
                        fprintf(stderr, "  [draft %3d+%d] entropy=%.2f > %.2f -> EARLY STOP\n",
                                n_decode, d, entropy, entropy_threshold);
                    }
                    // Still take this token before stopping
                    llama_token best_id = 0;
                    float best_logit = logits[0];
                    for (llama_token id = 1; id < n_vocab; ++id) {
                        if (logits[id] > best_logit) {
                            best_logit = logits[id];
                            best_id = id;
                        }
                    }
                    if (!llama_token_is_eog(model, best_id)) {
                        draft_tokens.push_back(best_id);
                    }
                    break;
                }
            }

            // Greedy sample
            llama_token best_id = 0;
            float best_logit = logits[0];
            for (llama_token id = 1; id < n_vocab; ++id) {
                if (logits[id] > best_logit) {
                    best_logit = logits[id];
                    best_id = id;
                }
            }

            if (llama_token_is_eog(model, best_id)) {
                break;
            }

            draft_tokens.push_back(best_id);

            // Continue drafting
            if (d + 1 < draft_limit && !early_stop_entropy) {
                common_batch_clear(batch);
                common_batch_add(batch, best_id, n_cur + d, { 0 }, true);
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "moe_combination: draft decode failed at token %d\n", d);
                    break;
                }
            }
        }

        if (draft_tokens.empty()) {
            break;  // EOS or failure
        }

        n_draft_total += (int)draft_tokens.size();

        // Stop recording — we now have the exact expert selections from
        // all draft decode calls.
        llama_router_stop_recording(ctx);

        int32_t n_recorded_layers = llama_router_n_recorded_layers(ctx);

        // ================================================================
        // PHASE 2: VERIFY with combination expert sharpening
        //
        // Query the router recordings to get the exact union of expert IDs
        // that were active in each MoE layer during the draft phase.
        // Only those experts need sharp data — everything else stays blurry.
        // ================================================================
        n_verify_batches++;

        // ---- Phase 2a: Per-token expert query ----
        // Instead of using the full expert union across ALL draft tokens
        // (which saturates to n_expert/n_expert = 1.0 selectivity with
        // enough tokens), use only the first few tokens' experts.
        //
        // Rationale: non-expert tensors (attention, norms, gate_inp) are
        // sharpened for ALL positions regardless.  Expert tensors only
        // matter for the positions whose experts we loaded.  For other
        // positions, both draft and verify use blurry experts → they
        // agree automatically.  Disagreements driven by sharp attention
        // weights are still caught at ALL positions.
        //
        // Budget: first min(draft_len, 3) tokens → ~3*top_k unique
        // experts per layer instead of the full union.
        const int expert_token_budget = std::min((int)draft_tokens.size(), 3);
        work_items.clear();

        int n_layers_skipped = 0;
        int n_full_union_experts = 0;  // for comparison logging
        for (int il = 0; il < n_layers; ++il) {
            // Check if this layer has any per-token data
            int32_t n_tokens_recorded = llama_router_n_tokens_for_layer(ctx, il);
            if (n_tokens_recorded <= 0) {
                n_layers_skipped++;
                continue;
            }

            // Get experts for only the first expert_token_budget tokens
            int32_t n_exp = llama_router_get_token_range_experts(
                ctx, il, 0, expert_token_budget, nullptr, 0);
            if (n_exp <= 0) {
                n_layers_skipped++;
                continue;
            }

            layer_work w;
            w.layer_idx  = il;
            w.n_experts  = n_exp;
            w.expert_ids.resize(n_exp);
            llama_router_get_token_range_experts(
                ctx, il, 0, expert_token_budget,
                w.expert_ids.data(), n_exp);
            work_items.push_back(std::move(w));

            // Track full union size for logging
            n_full_union_experts += llama_router_get_layer_experts(ctx, il, nullptr, 0);
        }

        // ---- Phase 2b: Parallel prefetch + sequential apply ----
        // Use multiple threads to read tensor data into the staging
        // cache concurrently.  The mmap memcpy / page fault is the
        // bottleneck — running N threads overlaps kernel I/O across
        // different file regions.  After prefetch completes, the
        // sequential apply loop finds data already cached (fast
        // pointer swaps, no I/O stalls).
        const int n_work = (int)work_items.size();

        {
            std::vector<int32_t> layer_ids(n_work);
            for (int i = 0; i < n_work; ++i) {
                layer_ids[i] = work_items[i].layer_idx;
            }
            llama_blurry_sharp_prefetch_layers_parallel(
                bsctx, layer_ids.data(), n_work, 4);
        }

        int n_sharpened = 0;
        for (int i = 0; i < n_work; ++i) {
            const auto & w = work_items[i];

            int32_t ret = llama_blurry_sharp_apply_experts(
                bsctx, w.layer_idx, w.expert_ids.data(), w.n_experts);
            if (ret == 0) {
                ++n_sharpened;
            }
            n_unique_experts_total += w.n_experts;
            n_expert_slots_total += n_expert;
        }

        llama_blurry_sharp_warm_live_pages(bsctx);

        // Save this iteration's layer set for early prefetch next iteration
        prev_layer_ids.resize(n_work);
        for (int i = 0; i < n_work; ++i) {
            prev_layer_ids[i] = work_items[i].layer_idx;
        }

        if (verbose) {
            fprintf(stderr, "  [verify %3d] sharpened %d/%d layers (%d skipped), "
                    "experts: %d (from %d/%d draft tokens) vs %d full-union\n",
                    n_decode, n_sharpened, n_layers, n_layers_skipped,
                    n_unique_experts_total, expert_token_budget,
                    (int)draft_tokens.size(), n_full_union_experts);
        }

        // Clear recorded data before the next draft iteration
        llama_router_clear(ctx);

        // Remove draft tokens from KV cache for re-evaluation
        llama_kv_cache_seq_rm(ctx, 0, n_cur, n_cur + (int)draft_tokens.size());

        // Build verification batch
        common_batch_clear(batch);
        for (int d = 0; d < (int)draft_tokens.size(); ++d) {
            common_batch_add(batch, draft_tokens[d], n_cur + d, { 0 }, true);
        }

        int n_accepted = 0;

        if (llama_decode(ctx, batch) == 0) {
            // Compare combination-expert model vs blurry at each position
            for (int d = 0; d < (int)draft_tokens.size(); ++d) {
                float * sharp_logits = llama_get_logits_ith(ctx, d);

                llama_token sharp_best = 0;
                float sharp_best_logit = sharp_logits[0];
                for (llama_token id = 1; id < n_vocab; ++id) {
                    if (sharp_logits[id] > sharp_best_logit) {
                        sharp_best_logit = sharp_logits[id];
                        sharp_best = id;
                    }
                }

                if (d + 1 < (int)draft_tokens.size()) {
                    if (sharp_best == draft_tokens[d + 1]) {
                        n_accepted++;
                        result.tokens.push_back(draft_tokens[d]);
                        result.text += common_token_to_piece(ctx, draft_tokens[d]);
                    } else {
                        n_accepted++;
                        result.tokens.push_back(draft_tokens[d]);
                        result.text += common_token_to_piece(ctx, draft_tokens[d]);

                        if (!llama_token_is_eog(model, sharp_best)) {
                            n_accepted++;
                            result.tokens.push_back(sharp_best);
                            result.text += common_token_to_piece(ctx, sharp_best);
                        }

                        if (verbose) {
                            fprintf(stderr, "  [verify %3d+%d] REJECT: blurry=\"%s\" vs combo=\"%s\" "
                                    "-> accepted %d/%d\n",
                                    n_decode, d + 1,
                                    common_token_to_piece(ctx, draft_tokens[d + 1]).c_str(),
                                    common_token_to_piece(ctx, sharp_best).c_str(),
                                    n_accepted, (int)draft_tokens.size());
                        }
                        break;
                    }
                } else {
                    n_accepted++;
                    result.tokens.push_back(draft_tokens[d]);
                    result.text += common_token_to_piece(ctx, draft_tokens[d]);

                    if (!llama_token_is_eog(model, sharp_best)) {
                        n_accepted++;
                        result.tokens.push_back(sharp_best);
                        result.text += common_token_to_piece(ctx, sharp_best);
                    }

                    if (verbose && n_accepted == (int)draft_tokens.size() + 1) {
                        fprintf(stderr, "  [verify %3d] all %d draft tokens accepted + 1 bonus\n",
                                n_decode, (int)draft_tokens.size());
                    }
                }
            }
        } else {
            fprintf(stderr, "moe_combination: verification decode failed\n");
            for (auto tok : draft_tokens) {
                if (llama_token_is_eog(model, tok)) break;
                result.tokens.push_back(tok);
                result.text += common_token_to_piece(ctx, tok);
                n_accepted++;
            }
        }

        n_accepted_total += n_accepted;

        // ================================================================
        // PHASE 3: RESTORE blurry, fix KV, prepare for next iteration
        // ================================================================
        llama_blurry_sharp_restore_all(bsctx);

        int accepted_end = n_cur + n_accepted;
        if (accepted_end < n_cur + (int)draft_tokens.size()) {
            llama_kv_cache_seq_rm(ctx, 0, accepted_end, n_cur + (int)draft_tokens.size());
        }

        n_decode += n_accepted;
        n_cur     = (int)prompt_tokens.size() + (int)result.tokens.size();

        result.n_blurry_steps += (int)draft_tokens.size();
        result.n_sharp_steps  += n_accepted;

        // Check EOS
        bool hit_eos = false;
        for (int i = (int)result.tokens.size() - n_accepted; i < (int)result.tokens.size(); ++i) {
            if (i >= 0 && llama_token_is_eog(model, result.tokens[i])) {
                hit_eos = true;
                break;
            }
        }
        if (hit_eos) break;

        // Prime logits for next draft
        if (!result.tokens.empty()) {
            common_batch_clear(batch);
            common_batch_add(batch, result.tokens.back(), n_cur - 1, { 0 }, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "moe_combination: post-verify decode failed\n");
                break;
            }
        }
    }

    const auto t_end = ggml_time_us();
    result.elapsed_s    = (t_end - t_start) / 1e6;
    result.tokens_per_s = (result.elapsed_s > 0.0) ? n_decode / result.elapsed_s : 0.0;
    result.n_draft_total    = n_draft_total;
    result.n_accepted_total = n_accepted_total;

    if (verbose || true) {
        fprintf(stderr, "\n  MoE Combination Expert stats (router hooks):\n");
        fprintf(stderr, "    Verify batches:     %d\n", n_verify_batches);
        fprintf(stderr, "    Drafted:            %d tokens\n", n_draft_total);
        fprintf(stderr, "    Accepted:           %d tokens (%.1f%%)\n",
                n_accepted_total,
                n_draft_total > 0 ? 100.0 * n_accepted_total / n_draft_total : 0.0);
        fprintf(stderr, "    Entropy stops:      %d\n", n_entropy_stops);
        fprintf(stderr, "    Expert selection:   router hooks (exact)\n");
        if (n_expert_slots_total > 0) {
            fprintf(stderr, "    Expert efficiency:  %d/%d expert-layer slots sharpened (%.1f%%)\n",
                    n_unique_experts_total, n_expert_slots_total,
                    100.0 * n_unique_experts_total / n_expert_slots_total);
            fprintf(stderr, "    I/O reduction:      ~%.1fx vs full-layer sharpening\n",
                    (double)n_expert_slots_total / std::max(1, n_unique_experts_total));
        }
        if (n_verify_batches > 0) {
            fprintf(stderr, "    Avg draft len:      %.1f tokens\n",
                    (double)n_draft_total / n_verify_batches);
            fprintf(stderr, "    Avg accepted:       %.1f tokens/batch\n",
                    (double)n_accepted_total / n_verify_batches);
        }
    }

    llama_batch_free(batch);
    return result;
}

// ---------------------------------------------------------------------------
// Combined adaptive generation: entropy + probing + speculative verification
//
// This is the unified pipeline that merges all three strategies:
//
//   1. DRAFT phase: Generate up to `draft_n` tokens with blurry (fast).
//      During drafting, monitor two early-stop signals:
//        a) Entropy: if any drafted token's entropy exceeds the threshold,
//           stop drafting early — blurry knows it's uncertain.
//        b) Probe: every `probe_stride` tokens within the draft, do a
//           lightweight sharp probe (sharpen top-K layers only, re-decode
//           one token). If sharp disagrees, stop drafting — blurry is
//           confidently wrong.
//
//   2. VERIFY phase: Sharpen ALL layers. Feed the entire draft batch through
//      the sharp model in one pass. Accept the longest prefix where sharp
//      agrees with blurry. At the first disagreement, take sharp's token.
//
//   3. RESTORE: Put blurry weights back, fix KV cache, continue.
//
// The adaptive draft length is the key: when blurry is on track (low entropy
// + probe agrees), we draft the full `draft_n` tokens → amortized sharp cost
// over many tokens → fast.  When blurry is struggling, we stop early → fewer
// wasted draft tokens → better quality with less overhead.
//
// With good settings, ~20-30% of tokens need full sharp compute, giving
// sharp-level quality at a fraction of the cost.
// ---------------------------------------------------------------------------
static generation_result run_combined_generation(
        llama_model   * model,
        llama_context * ctx,
        llama_blurry_sharp_context * bsctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        int draft_n,
        int probe_stride,
        float entropy_threshold,
        int top_k_layers,
        bool verbose) {

    generation_result result;

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(model);

    llama_blurry_sharp_state st = llama_blurry_sharp_get_state(bsctx);
    int n_layers = st.n_layers_total;

    // Probe layers: top-K (first half + last half) for lightweight probes
    std::vector<int> probe_layers;
    {
        int half = std::max(1, top_k_layers / 2);
        for (int i = 0; i < half && i < n_layers; ++i) {
            probe_layers.push_back(i);
        }
        for (int i = std::max(0, n_layers - half); i < n_layers; ++i) {
            bool dup = false;
            for (int p : probe_layers) { if (p == i) { dup = true; break; } }
            if (!dup) probe_layers.push_back(i);
        }
    }

    // Verify layers: ALL layers for full quality verification
    std::vector<int> all_layers;
    for (int i = 0; i < n_layers; ++i) {
        all_layers.push_back(i);
    }

    if (verbose) {
        fprintf(stderr, "  Combined mode: draft_n=%d, probe_stride=%d, entropy_thresh=%.2f\n",
                draft_n, probe_stride, entropy_threshold);
        fprintf(stderr, "  Probe layers (%d): [", (int)probe_layers.size());
        for (size_t i = 0; i < probe_layers.size(); ++i) {
            fprintf(stderr, "%s%d", i > 0 ? "," : "", probe_layers[i]);
        }
        fprintf(stderr, "], Verify layers: all %d\n", n_layers);
    }

    if ((int)prompt_tokens.size() + n_predict > n_ctx) {
        n_predict = n_ctx - (int)prompt_tokens.size();
        if (n_predict <= 0) {
            fprintf(stderr, "combined: prompt longer than context\n");
            return result;
        }
    }

    // Prompt eval (blurry)
    llama_batch batch = llama_batch_init(std::max(512, draft_n + 1), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (int)i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "combined: llama_decode() failed on prompt\n");
        llama_batch_free(batch);
        return result;
    }

    int n_cur = batch.n_tokens;
    int n_decode = 0;

    // Stats
    int n_draft_total    = 0;
    int n_accepted_total = 0;
    int n_verify_batches = 0;
    int n_entropy_stops  = 0;
    int n_probe_stops    = 0;
    int n_probes_run     = 0;
    int n_probe_agrees   = 0;
    int n_full_drafts    = 0;  // drafts that ran to completion without early stop

    const auto t_start = ggml_time_us();

    while (n_cur < (int)prompt_tokens.size() + n_predict) {
        // ================================================================
        // PHASE 1: DRAFT with blurry, monitoring entropy + probes
        // ================================================================
        std::vector<llama_token> draft_tokens;
        std::vector<float>      draft_entropies;
        int draft_limit = std::min(draft_n,
                                   (int)prompt_tokens.size() + n_predict - n_cur);
        bool early_stop_entropy = false;
        bool early_stop_probe   = false;

        for (int d = 0; d < draft_limit; ++d) {
            float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            // Compute entropy
            float entropy = compute_logit_entropy(logits, n_vocab);
            draft_entropies.push_back(entropy);

            // Greedy sample
            llama_token best_id = 0;
            float best_logit = logits[0];
            for (llama_token id = 1; id < n_vocab; ++id) {
                if (logits[id] > best_logit) {
                    best_logit = logits[id];
                    best_id = id;
                }
            }

            if (llama_token_is_eog(model, best_id)) {
                break;
            }

            draft_tokens.push_back(best_id);

            // --- Early stop check: entropy ---
            if (entropy_threshold > 0 && entropy > entropy_threshold) {
                early_stop_entropy = true;
                n_entropy_stops++;
                if (verbose) {
                    fprintf(stderr, "  [draft %3d+%d] entropy=%.2f > %.2f -> EARLY STOP (entropy)\n",
                            n_decode, d, entropy, entropy_threshold);
                }
                break;
            }

            // --- Early stop check: probe ---
            // Only probe if we have at least 1 drafted token and it's a probe stride
            if (probe_stride > 0 && d > 0 && (d % probe_stride) == 0) {
                n_probes_run++;

                // Sharpen probe layers (lightweight — top-K only) with lookahead prefetch
                int n_sharpened = 0;
                {
                    constexpr int PREFETCH_LOOKAHEAD = 4;
                    const int n_probe = (int)probe_layers.size();
                    for (int pi = 0; pi < std::min(PREFETCH_LOOKAHEAD, n_probe); ++pi) {
                        llama_blurry_sharp_prefetch_layer(bsctx, probe_layers[pi]);
                    }
                    for (int pi = 0; pi < n_probe; ++pi) {
                        if (pi + PREFETCH_LOOKAHEAD < n_probe) {
                            llama_blurry_sharp_prefetch_layer(bsctx, probe_layers[pi + PREFETCH_LOOKAHEAD]);
                        }
                        if (llama_blurry_sharp_apply_layer(bsctx, probe_layers[pi]) == 0) {
                            ++n_sharpened;
                        }
                    }
                }
                // Pull CPU zero-copy tensor pages into RAM so the decode
                // doesn't stall on page faults when RAM is available.
                llama_blurry_sharp_warm_live_pages(bsctx);

                if (n_sharpened > 0) {
                    // Re-decode the last token with probe-sharp weights
                    llama_token probe_token = draft_tokens.back();
                    common_batch_clear(batch);
                    common_batch_add(batch, probe_token, n_cur + d - 1, { 0 }, true);

                    if (llama_decode(ctx, batch) == 0) {
                        float * probe_logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

                        llama_token probe_best = 0;
                        float probe_best_logit = probe_logits[0];
                        for (llama_token id = 1; id < n_vocab; ++id) {
                            if (probe_logits[id] > probe_best_logit) {
                                probe_best_logit = probe_logits[id];
                                probe_best = id;
                            }
                        }

                        // Does the probe's next-token prediction match what blurry
                        // would draft next?  We don't know blurry's next token yet,
                        // but we can compare: does probe agree with the token we
                        // just drafted (best_id)?  Actually, the probe re-evaluated
                        // the previous token to predict THIS position.  So we compare
                        // probe_best vs best_id (the token blurry just picked).
                        if (probe_best != best_id) {
                            early_stop_probe = true;
                            n_probe_stops++;
                            if (verbose) {
                                fprintf(stderr, "  [draft %3d+%d] PROBE disagrees: "
                                        "blurry=\"%s\" vs probe=\"%s\" -> EARLY STOP\n",
                                        n_decode, d,
                                        common_token_to_piece(ctx, best_id).c_str(),
                                        common_token_to_piece(ctx, probe_best).c_str());
                            }
                        } else {
                            n_probe_agrees++;
                            if (verbose) {
                                fprintf(stderr, "  [draft %3d+%d] PROBE agrees (\"%s\") -> continue drafting\n",
                                        n_decode, d,
                                        common_token_to_piece(ctx, best_id).c_str());
                            }
                        }
                    }

                    // Restore probe layers
                    llama_blurry_sharp_restore_all(bsctx);

                    // Need to re-decode the draft token with blurry weights to
                    // restore correct KV state for continued drafting
                    common_batch_clear(batch);
                    common_batch_add(batch, draft_tokens.back(), n_cur + d, { 0 }, true);
                    if (llama_decode(ctx, batch) != 0) {
                        fprintf(stderr, "combined: post-probe re-decode failed\n");
                        break;
                    }

                    if (early_stop_probe) break;
                } else {
                    llama_blurry_sharp_restore_all(bsctx);
                }

                // Continue to next draft token
                continue;
            }

            // Decode this draft token to get logits for next
            if (d + 1 < draft_limit && !early_stop_entropy && !early_stop_probe) {
                common_batch_clear(batch);
                common_batch_add(batch, best_id, n_cur + d, { 0 }, true);
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "combined: draft decode failed at token %d\n", d);
                    break;
                }
            }
        }

        if (draft_tokens.empty()) {
            break;  // EOS or failure
        }

        if (!early_stop_entropy && !early_stop_probe) {
            n_full_drafts++;
        }

        n_draft_total += (int)draft_tokens.size();

        // ================================================================
        // PHASE 2: VERIFY with sharp model (full quality)
        // ================================================================
        n_verify_batches++;

        // Sharpen ALL layers for verification with lookahead prefetch
        {
            constexpr int PREFETCH_LOOKAHEAD = 4;
            const int n_all = (int)all_layers.size();

            // Seed the prefetch pipeline
            for (int i = 0; i < std::min(PREFETCH_LOOKAHEAD, n_all); ++i) {
                llama_blurry_sharp_prefetch_layer(bsctx, all_layers[i]);
            }

            int n_sharpened = 0;
            for (int i = 0; i < n_all; ++i) {
                if (i + PREFETCH_LOOKAHEAD < n_all) {
                    llama_blurry_sharp_prefetch_layer(bsctx, all_layers[i + PREFETCH_LOOKAHEAD]);
                }
                if (llama_blurry_sharp_apply_layer(bsctx, all_layers[i]) == 0) {
                    ++n_sharpened;
                }
            }
        }
        // Pull CPU zero-copy tensor pages into RAM so the verification
        // decode doesn't stall on page faults when RAM is available.
        llama_blurry_sharp_warm_live_pages(bsctx);

        // Remove draft tokens from KV cache so sharp re-evaluates them
        llama_kv_cache_seq_rm(ctx, 0, n_cur, n_cur + (int)draft_tokens.size());

        // Build verification batch
        common_batch_clear(batch);
        for (int d = 0; d < (int)draft_tokens.size(); ++d) {
            common_batch_add(batch, draft_tokens[d], n_cur + d, { 0 }, true);
        }

        int n_accepted = 0;

        if (llama_decode(ctx, batch) == 0) {
            // Compare sharp vs blurry at each position
            for (int d = 0; d < (int)draft_tokens.size(); ++d) {
                float * sharp_logits = llama_get_logits_ith(ctx, d);

                llama_token sharp_best = 0;
                float sharp_best_logit = sharp_logits[0];
                for (llama_token id = 1; id < n_vocab; ++id) {
                    if (sharp_logits[id] > sharp_best_logit) {
                        sharp_best_logit = sharp_logits[id];
                        sharp_best = id;
                    }
                }

                if (d + 1 < (int)draft_tokens.size()) {
                    // Check: does sharp predict draft_tokens[d+1] as next?
                    if (sharp_best == draft_tokens[d + 1]) {
                        n_accepted++;
                        result.tokens.push_back(draft_tokens[d]);
                        result.text += common_token_to_piece(ctx, draft_tokens[d]);
                    } else {
                        // Accept this token, take sharp's correction for next
                        n_accepted++;
                        result.tokens.push_back(draft_tokens[d]);
                        result.text += common_token_to_piece(ctx, draft_tokens[d]);

                        if (!llama_token_is_eog(model, sharp_best)) {
                            n_accepted++;
                            result.tokens.push_back(sharp_best);
                            result.text += common_token_to_piece(ctx, sharp_best);
                        }

                        if (verbose) {
                            fprintf(stderr, "  [verify %3d+%d] REJECT: blurry=\"%s\" vs sharp=\"%s\" "
                                    "-> accepted %d/%d\n",
                                    n_decode, d + 1,
                                    common_token_to_piece(ctx, draft_tokens[d + 1]).c_str(),
                                    common_token_to_piece(ctx, sharp_best).c_str(),
                                    n_accepted, (int)draft_tokens.size());
                        }
                        break;
                    }
                } else {
                    // Last draft token: accept + bonus from sharp
                    n_accepted++;
                    result.tokens.push_back(draft_tokens[d]);
                    result.text += common_token_to_piece(ctx, draft_tokens[d]);

                    if (!llama_token_is_eog(model, sharp_best)) {
                        n_accepted++;
                        result.tokens.push_back(sharp_best);
                        result.text += common_token_to_piece(ctx, sharp_best);
                    }

                    if (verbose && n_accepted == (int)draft_tokens.size() + 1) {
                        fprintf(stderr, "  [verify %3d] all %d draft tokens accepted + 1 bonus\n",
                                n_decode, (int)draft_tokens.size());
                    }
                }
            }
        } else {
            fprintf(stderr, "combined: verification decode failed\n");
            for (auto tok : draft_tokens) {
                if (llama_token_is_eog(model, tok)) break;
                result.tokens.push_back(tok);
                result.text += common_token_to_piece(ctx, tok);
                n_accepted++;
            }
        }

        n_accepted_total += n_accepted;

        // ================================================================
        // PHASE 3: RESTORE blurry, fix KV, prepare for next iteration
        // ================================================================
        llama_blurry_sharp_restore_all(bsctx);

        // Remove any excess KV entries
        int accepted_end = n_cur + n_accepted;
        if (accepted_end < n_cur + (int)draft_tokens.size()) {
            llama_kv_cache_seq_rm(ctx, 0, accepted_end, n_cur + (int)draft_tokens.size());
        }

        n_decode += n_accepted;
        n_cur     = (int)prompt_tokens.size() + (int)result.tokens.size();

        result.n_blurry_steps += (int)draft_tokens.size();
        result.n_sharp_steps  += n_accepted;

        // Check EOS
        bool hit_eos = false;
        for (int i = (int)result.tokens.size() - n_accepted; i < (int)result.tokens.size(); ++i) {
            if (i >= 0 && llama_token_is_eog(model, result.tokens[i])) {
                hit_eos = true;
                break;
            }
        }
        if (hit_eos) break;

        // Prime logits for next draft
        if (!result.tokens.empty()) {
            common_batch_clear(batch);
            common_batch_add(batch, result.tokens.back(), n_cur - 1, { 0 }, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "combined: post-verify decode failed\n");
                break;
            }
        }
    }

    const auto t_end = ggml_time_us();
    result.elapsed_s    = (t_end - t_start) / 1e6;
    result.tokens_per_s = (result.elapsed_s > 0.0) ? n_decode / result.elapsed_s : 0.0;
    result.n_draft_total    = n_draft_total;
    result.n_accepted_total = n_accepted_total;
    result.n_probes         = n_probes_run;
    result.n_probe_disagreements = n_probe_stops;

    if (verbose) {
        fprintf(stderr, "\n  Combined stats:\n");
        fprintf(stderr, "    Verify batches:   %d\n", n_verify_batches);
        fprintf(stderr, "    Drafted:          %d tokens\n", n_draft_total);
        fprintf(stderr, "    Accepted:         %d tokens (%.1f%%)\n",
                n_accepted_total,
                n_draft_total > 0 ? 100.0 * n_accepted_total / n_draft_total : 0.0);
        fprintf(stderr, "    Full drafts:      %d (no early stop)\n", n_full_drafts);
        fprintf(stderr, "    Entropy stops:    %d\n", n_entropy_stops);
        fprintf(stderr, "    Probe stops:      %d\n", n_probe_stops);
        fprintf(stderr, "    Probes run:       %d (%d agreed, %d disagreed)\n",
                n_probes_run, n_probe_agrees, n_probe_stops);
        if (n_verify_batches > 0) {
            fprintf(stderr, "    Avg draft len:    %.1f tokens\n",
                    (double)n_draft_total / n_verify_batches);
            fprintf(stderr, "    Avg accepted:     %.1f tokens/batch\n",
                    (double)n_accepted_total / n_verify_batches);
        }
    }

    llama_batch_free(batch);
    return result;
}

// ---------------------------------------------------------------------------
// Standard generation loop (greedy, for comparison purposes)
// ---------------------------------------------------------------------------
static generation_result run_generation(
        llama_model   * model,
        llama_context * ctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        const char * label) {

    generation_result result;

    const int n_ctx = llama_n_ctx(ctx);

    if ((int)prompt_tokens.size() + n_predict > n_ctx) {
        fprintf(stderr, "%s: warning: prompt + n_predict exceeds context size\n", label);
        n_predict = n_ctx - (int)prompt_tokens.size();
        if (n_predict <= 0) {
            fprintf(stderr, "%s: error: prompt is longer than context\n", label);
            return result;
        }
    }

    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (int)i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed on prompt\n", label);
        llama_batch_free(batch);
        return result;
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_start = ggml_time_us();

    while (n_cur < (int)prompt_tokens.size() + n_predict) {
        auto   n_vocab = llama_n_vocab(model);
        auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        llama_token best_id = 0;
        float       best_logit = logits[0];
        for (llama_token id = 1; id < n_vocab; ++id) {
            if (logits[id] > best_logit) {
                best_logit = logits[id];
                best_id    = id;
            }
        }

        if (llama_token_is_eog(model, best_id)) {
            break;
        }

        result.tokens.push_back(best_id);
        result.text += common_token_to_piece(ctx, best_id);

        common_batch_clear(batch);
        common_batch_add(batch, best_id, n_cur, { 0 }, true);

        n_decode++;
        n_cur++;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s: llama_decode() failed at token %d\n", label, n_decode);
            break;
        }
    }

    const auto t_end = ggml_time_us();
    result.elapsed_s    = (t_end - t_start) / 1e6;
    result.tokens_per_s = (result.elapsed_s > 0.0) ? n_decode / result.elapsed_s : 0.0;

    llama_batch_free(batch);
    return result;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    gpt_params params;

    // Parse all arguments through the standard gpt_params infrastructure.
    // This gives us every flag: -ngl, -fa, --jinja, -c, -t, -b, --mlock,
    // --numa, --mmap, --lora, --rope-*, --cache-type-k, etc.
    // Plus the blurry-sharp specific flags: -ms/--sharp, --bs-*, etc.
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // If no prompt was given, use a sensible default
    if (params.prompt.empty()) {
        params.prompt = "Hello my name is";
    }

    // If n_predict wasn't set, default to 64 for this example
    if (params.n_predict < 0) {
        params.n_predict = 64;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "=== Blurry->Sharp Inference Example ===\n");
    fprintf(stderr, "  Blurry model: %s\n", params.model.c_str());
    fprintf(stderr, "  Sharp  model: %s\n",
            params.sharp_model.empty() ? "(none)" : params.sharp_model.c_str());
    fprintf(stderr, "  Prompt:       \"%s\"\n", params.prompt.c_str());
    fprintf(stderr, "  n_predict:    %d\n", params.n_predict);
    fprintf(stderr, "  n_gpu_layers: %d\n", params.n_gpu_layers);
    fprintf(stderr, "  flash_attn:   %s\n", params.flash_attn ? "on" : "off");
    fprintf(stderr, "  n_ctx:        %d\n", params.n_ctx);
    fprintf(stderr, "  n_threads:    %d\n", params.n_threads);
    if (params.bs_gpu_budget_mb > 0) {
        fprintf(stderr, "  gpu_budget:   %" PRId64 " MiB\n", params.bs_gpu_budget_mb);
    }
    if (params.bs_precache_ram || params.bs_stage_swap || params.bs_lazy_swap) {
        fprintf(stderr, "  memory_tier:  VRAM > RAM > Swap > Disk");
        if (params.bs_stage_swap) {
            fprintf(stderr, " (stage-to-swap enabled)");
        } else if (params.bs_lazy_swap) {
            fprintf(stderr, " (lazy per-layer swap staging)");
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    // -----------------------------------------------------------------------
    // Init backend + NUMA (standard path)
    // -----------------------------------------------------------------------
    llama_backend_init();
    llama_numa_init(params.numa);

    // -----------------------------------------------------------------------
    // Load blurry model + create context using standard gpt_params flow
    // This respects ALL flags: -ngl, -fa, --mlock, --numa, --rope-*, etc.
    // -----------------------------------------------------------------------
    fprintf(stderr, "Loading blurry model (using standard gpt_params)...\n");

    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model   * model = llama_init.model;
    llama_context * ctx   = llama_init.context;

    if (!model) {
        fprintf(stderr, "Error: failed to load blurry model from '%s'\n", params.model.c_str());
        llama_backend_free();
        return 1;
    }
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Apply LoRA adapters if any were specified
    if (!llama_init.lora_adapters.empty()) {
        llama_lora_adapters_apply(ctx, llama_init.lora_adapters);
    }

    // Print system info (includes all relevant flags)
    fprintf(stderr, "\n%s\n\n", gpt_params_get_system_info(params).c_str());

    // -----------------------------------------------------------------------
    // Tokenize prompt
    // -----------------------------------------------------------------------
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, true);
    fprintf(stderr, "Prompt tokens: %zu\n", prompt_tokens.size());

    fprintf(stderr, "Prompt: ");
    for (auto id : prompt_tokens) {
        fprintf(stderr, "%s", common_token_to_piece(ctx, id).c_str());
    }
    fprintf(stderr, "\n\n");

    // -----------------------------------------------------------------------
    // Run 1: Blurry-only (baseline) — only if comparing or no sharp model
    // -----------------------------------------------------------------------
    generation_result result_blurry;

    if (params.bs_compare || params.sharp_model.empty()) {
        fprintf(stderr, "--- [1/2] Blurry-only generation ---\n");

        result_blurry = run_generation(model, ctx, prompt_tokens, params.n_predict, "blurry");

        fprintf(stderr, "Generated: %s\n", result_blurry.text.c_str());
        fprintf(stderr, "Tokens: %zu, Time: %.2f s, Speed: %.2f t/s\n\n",
                result_blurry.tokens.size(),
                result_blurry.elapsed_s,
                result_blurry.tokens_per_s);

        llama_kv_cache_clear(ctx);
    }

    // -----------------------------------------------------------------------
    // Initialize Blurry->Sharp overlay system
    // -----------------------------------------------------------------------
    llama_blurry_sharp_context * bsctx = nullptr;

    if (!params.sharp_model.empty()) {
        fprintf(stderr, "Initializing blurry-sharp overlay system...\n");

        llama_blurry_sharp_params bs_params = llama_blurry_sharp_default_params();
        bs_params.sharp_model_path      = params.sharp_model.c_str();
        bs_params.router_strategy       = (llama_blurry_sharp_router_strategy)params.bs_router_strategy;
        bs_params.router_min_confidence = params.bs_router_confidence;
        bs_params.max_sharp_layers      = params.bs_max_sharp_layers;
        bs_params.memory_budget_bytes   = params.bs_memory_budget_mb * 1024 * 1024;
        bs_params.gpu_budget_bytes      = params.bs_gpu_budget_mb * 1024 * 1024;
        bs_params.restore_after_forward = params.bs_restore_after_fwd;
        bs_params.verbose               = params.bs_verbose;
        bs_params.use_mmap              = params.bs_use_mmap;
        bs_params.lazy_swap             = params.bs_lazy_swap;
        bs_params.retain_mmap_pages     = params.bs_retain_mmap;

        // Auto-enable retain_device_buffers for modes that do repeated
        // sharpen/restore cycles — this eliminates cudaMalloc/Free churn
        // and skips PCIe re-copy, turning ~10s sharpen into ~0s.
        bs_params.retain_device_buffers = params.bs_retain_buffers
            || params.bs_combined
            || params.bs_speculative
            || params.bs_moe_combination
            || (params.bs_probe_interval > 0);

        if (!params.bs_layer_allowlist.empty()) {
            bs_params.layer_allowlist   = params.bs_layer_allowlist.data();
            bs_params.n_layer_allowlist = (int32_t)params.bs_layer_allowlist.size();
        }
        if (!params.bs_layer_denylist.empty()) {
            bs_params.layer_denylist    = params.bs_layer_denylist.data();
            bs_params.n_layer_denylist  = (int32_t)params.bs_layer_denylist.size();
        }

        bsctx = llama_blurry_sharp_init(model, bs_params);

        if (!bsctx) {
            fprintf(stderr, "Warning: failed to initialize blurry-sharp system, "
                           "proceeding with blurry-only inference\n\n");
        } else {
            llama_blurry_sharp_state state = llama_blurry_sharp_get_state(bsctx);
            fprintf(stderr, "Overlay state: %d total layers, %d currently sharpened\n",
                    state.n_layers_total, state.n_layers_sharpened);
            if (params.bs_gpu_budget_mb > 0) {
                fprintf(stderr, "GPU budget: %" PRId64 " MiB for sharp device buffers\n",
                        params.bs_gpu_budget_mb);
            }
            if (bs_params.retain_device_buffers) {
                fprintf(stderr, "Device buffer caching: ENABLED (fast re-sharpen)\n");
            }

            // -----------------------------------------------------------------
            // Memory-tier initialization: VRAM > RAM > Swap > Disk
            //
            // Step 1: Pre-read sharp tensor data into anonymous heap buffers.
            //         Anonymous pages go to SWAP under memory pressure (not
            //         dropped like file-backed mmap pages).  Re-access from
            //         swap (SSD) is 10-50x faster than from GGUF file on disk.
            //
            // Step 2: Proactively move cached pages to swap via MADV_PAGEOUT.
            //         This frees RAM for the blurry model / KV cache while
            //         keeping sharp data quickly accessible via swap-in.
            //
            // Step 3: Pre-allocate GPU buffers so apply_layer only needs the
            //         PCIe data copy, skipping expensive per-tensor cudaMalloc.
            // -----------------------------------------------------------------

            // --bs-stage-swap implies --bs-precache-ram
            // --bs-lazy-swap skips upfront precache entirely (lazy per-layer)
            bool do_precache = (params.bs_precache_ram || params.bs_stage_swap) && !params.bs_lazy_swap;

            if (params.bs_lazy_swap) {
                fprintf(stderr, "\n--- Memory-tier initialization (lazy swap) ---\n");
                fprintf(stderr, "  Strategy: VRAM > RAM (lazy per-layer cache) > Swap > Disk\n");
                fprintf(stderr, "  Sharp data will be cached into RAM on first layer access,\n");
                fprintf(stderr, "  then staged to swap on restore. No upfront precache needed.\n");
                fprintf(stderr, "--- End memory-tier initialization ---\n\n");
            }

            if (do_precache) {
                fprintf(stderr, "\n--- Memory-tier initialization ---\n");
                fprintf(stderr, "  Strategy: VRAM > RAM (anonymous/swappable) > Swap > Disk\n\n");

                const auto t_precache_start = ggml_time_us();
                int32_t n_precached = llama_blurry_sharp_precache_ram(bsctx);
                const auto t_precache_end = ggml_time_us();

                if (n_precached > 0) {
                    fprintf(stderr, "  Pre-cached %d sharp tensors into RAM in %.2f s\n",
                            n_precached, (t_precache_end - t_precache_start) / 1e6);
                }

                if (params.bs_stage_swap) {
                    const auto t_swap_start = ggml_time_us();
                    int32_t n_staged = llama_blurry_sharp_stage_to_swap(bsctx);
                    const auto t_swap_end = ggml_time_us();

                    if (n_staged > 0) {
                        fprintf(stderr, "  Staged %d tensors to swap in %.2f ms "
                                "(RAM freed for blurry model / KV cache)\n",
                                n_staged, (t_swap_end - t_swap_start) / 1000.0);
                    }
                }

                fprintf(stderr, "--- End memory-tier initialization ---\n\n");
            }

            // Pre-allocate GPU buffers now so that apply_layer/apply_all only
            // needs the PCIe data copy, skipping expensive per-tensor cudaMalloc.
            {
                const auto t_preload_start = ggml_time_us();
                int32_t n_preloaded = llama_blurry_sharp_preload_device_cache(bsctx);
                const auto t_preload_end = ggml_time_us();
                if (n_preloaded > 0) {
                    fprintf(stderr, "Pre-allocated %d GPU buffers in %.2f ms (cudaMalloc moved out of hot path)\n",
                            n_preloaded, (t_preload_end - t_preload_start) / 1000.0);
                }
            }

            fprintf(stderr, "\n");
        }
    }

    // -----------------------------------------------------------------------
    // Run 2: Blurry + Sharp generation
    // -----------------------------------------------------------------------
    generation_result result_sharp;

    if (bsctx) {
        fprintf(stderr, "--- [2/2] Blurry+Sharp generation ---\n");

        // Apply sharp overlays (batch apply with mmap prefetch)
        {
            llama_blurry_sharp_state st0 = llama_blurry_sharp_get_state(bsctx);
            int n_total = st0.n_layers_total;

            fprintf(stderr, "Applying sharp overlays to %d layers...\n", n_total);

            const auto t_overlay_start = ggml_time_us();

            // Use batch apply: prefetches mmap pages with madvise(WILLNEED),
            // uses cached base-tensor pointers (O(1) lookup instead of O(n)),
            // and direct mmap→pinned staging copies (skips intermediate heap buffer).
            int32_t n_sharpened = llama_blurry_sharp_apply_all(bsctx);

            const auto t_overlay_end = ggml_time_us();
            fprintf(stderr, "Sharpened %d layers in %.2f ms\n",
                    n_sharpened,
                    (t_overlay_end - t_overlay_start) / 1000.0);

            // Pull CPU zero-copy tensor pages into the page cache (RAM).
            // Without this, the sharp mmap was created with prefetch=0 and
            // every forward pass would stall on page faults reading from disk
            // even when RAM is available.
            llama_blurry_sharp_warm_live_pages(bsctx);
        }

        // Show which layers are sharp
        {
            llama_blurry_sharp_state state = llama_blurry_sharp_get_state(bsctx);
            fprintf(stderr, "Overlay state: %d/%d layers sharpened, "
                           "%" PRId64 " bytes backup\n",
                    state.n_layers_sharpened,
                    state.n_layers_total,
                    state.total_backup_bytes);
            if (state.gpu_device_bytes_used > 0 || state.n_device_tensors_skipped > 0) {
                fprintf(stderr, "GPU overlay: %" PRId64 " MiB allocated",
                        state.gpu_device_bytes_used / (1024 * 1024));
                if (state.gpu_budget_bytes > 0) {
                    fprintf(stderr, " / %" PRId64 " MiB budget",
                            state.gpu_budget_bytes / (1024 * 1024));
                }
                if (state.n_device_tensors_skipped > 0) {
                    fprintf(stderr, ", %d device tensors skipped (GPU budget/OOM)",
                            state.n_device_tensors_skipped);
                }
                fprintf(stderr, "\n");
            }
            if (state.n_layers_sharpened < state.n_layers_total && state.n_device_tensors_skipped > 0) {
                fprintf(stderr, "\n");
                fprintf(stderr, "  WARNING: %d/%d layers could not be fully sharpened and were\n",
                        state.n_layers_total - state.n_layers_sharpened, state.n_layers_total);
                fprintf(stderr, "  rolled back to prevent mixed-quant corruption.\n");
                fprintf(stderr, "  To sharpen more layers, try one of:\n");
                if (state.gpu_budget_bytes > 0) {
                    fprintf(stderr, "    1. Remove --bs-gpu-budget-mb (let it use all free VRAM)\n");
                    fprintf(stderr, "    2. Reduce -ngl to keep blurry weights on CPU, freeing VRAM for sharp\n");
                } else {
                    fprintf(stderr, "    1. Reduce -ngl to keep blurry weights on CPU, freeing VRAM for sharp\n");
                    fprintf(stderr, "    2. Use a smaller sharp model (fewer bits per weight)\n");
                }
            }
        }

        // ---------------------------------------------------------------
        // POST-SHARPEN DIAGNOSTICS
        // Do a single decode and inspect logits to verify the overlay
        // produces valid output before running full generation.
        // ---------------------------------------------------------------
        {
            fprintf(stderr, "\n--- Post-sharpen diagnostics ---\n");

            llama_batch diag_batch = llama_batch_init(512, 0, 1);
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                common_batch_add(diag_batch, prompt_tokens[i], (int)i, { 0 }, false);
            }
            diag_batch.logits[diag_batch.n_tokens - 1] = true;

            int diag_rc = llama_decode(ctx, diag_batch);
            fprintf(stderr, "  Diagnostic decode returned: %d\n", diag_rc);

            if (diag_rc == 0) {
                float * logits = llama_get_logits_ith(ctx, diag_batch.n_tokens - 1);
                int n_vocab_diag = llama_n_vocab(model);

                float lmin = logits[0], lmax = logits[0];
                double lsum = 0.0;
                int n_nan = 0, n_inf = 0;
                for (int i = 0; i < n_vocab_diag; ++i) {
                    if (std::isnan(logits[i])) { n_nan++; continue; }
                    if (std::isinf(logits[i])) { n_inf++; continue; }
                    if (logits[i] < lmin) lmin = logits[i];
                    if (logits[i] > lmax) lmax = logits[i];
                    lsum += logits[i];
                }
                double lmean = lsum / std::max(1, n_vocab_diag - n_nan - n_inf);

                fprintf(stderr, "  Logit stats: min=%.4f  max=%.4f  mean=%.4f  "
                                "nan=%d  inf=%d  vocab=%d\n",
                        lmin, lmax, lmean, n_nan, n_inf, n_vocab_diag);
                fprintf(stderr, "  Max-min spread: %.6f", lmax - lmin);
                if (lmax - lmin < 1e-6f) {
                    fprintf(stderr, "  *** FLAT LOGITS — all values identical! ***");
                }
                fprintf(stderr, "\n");

                // Greedy argmax
                int best = 0;
                for (int i = 1; i < n_vocab_diag; ++i) {
                    if (logits[i] > logits[best]) best = i;
                }
                fprintf(stderr, "  Greedy token: id=%d \"%s\"  logit=%.4f\n",
                        best,
                        common_token_to_piece(ctx, best).c_str(),
                        logits[best]);

                // First 10 logits
                fprintf(stderr, "  First 10 logits: ");
                for (int i = 0; i < std::min(10, n_vocab_diag); ++i) {
                    fprintf(stderr, "[%d]=%.3f ", i, logits[i]);
                }
                fprintf(stderr, "\n");

                // Logits around the blurry model's top token (if available)
                if (!result_blurry.tokens.empty()) {
                    int blurry_top = result_blurry.tokens[0];
                    fprintf(stderr, "  Logit at blurry-top token %d (\"%s\"): %.4f\n",
                            blurry_top,
                            common_token_to_piece(ctx, blurry_top).c_str(),
                            logits[blurry_top]);
                }
            }

            llama_batch_free(diag_batch);
            llama_kv_cache_clear(ctx);

            fprintf(stderr, "--- End diagnostics ---\n\n");
        }

        fprintf(stderr, "\n");

        result_sharp = run_generation(model, ctx, prompt_tokens, params.n_predict, "blurry+sharp");

        fprintf(stderr, "Generated: %s\n", result_sharp.text.c_str());
        fprintf(stderr, "Tokens: %zu, Time: %.2f s, Speed: %.2f t/s\n\n",
                result_sharp.tokens.size(),
                result_sharp.elapsed_s,
                result_sharp.tokens_per_s);

        fprintf(stderr, "Restoring blurry weights...\n");
        llama_blurry_sharp_restore_all(bsctx);

        {
            llama_blurry_sharp_state state = llama_blurry_sharp_get_state(bsctx);
            fprintf(stderr, "Post-restore: %d layers still sharpened, "
                           "%" PRId64 " bytes backup\n\n",
                    state.n_layers_sharpened,
                    state.total_backup_bytes);
        }

        llama_kv_cache_clear(ctx);
    } else if (!params.bs_compare && params.sharp_model.empty()) {
        // No sharp model provided, just run blurry
        fprintf(stderr, "--- Blurry-only generation (no sharp model provided) ---\n");
        result_blurry = run_generation(model, ctx, prompt_tokens, params.n_predict, "blurry");
        fprintf(stderr, "Generated: %s\n", result_blurry.text.c_str());
        fprintf(stderr, "Tokens: %zu, Time: %.2f s, Speed: %.2f t/s\n\n",
                result_blurry.tokens.size(),
                result_blurry.elapsed_s,
                result_blurry.tokens_per_s);
    }

    // -----------------------------------------------------------------------
    // Comparison report
    // -----------------------------------------------------------------------
    if (params.bs_compare && bsctx &&
        !result_blurry.tokens.empty() && !result_sharp.tokens.empty()) {

        fprintf(stderr, "\n");
        fprintf(stderr, "=== Comparison Report ===\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "  Blurry-only:\n");
        fprintf(stderr, "    Output:  %s\n", result_blurry.text.c_str());
        fprintf(stderr, "    Tokens:  %zu\n", result_blurry.tokens.size());
        fprintf(stderr, "    Time:    %.2f s\n", result_blurry.elapsed_s);
        fprintf(stderr, "    Speed:   %.2f t/s\n", result_blurry.tokens_per_s);
        fprintf(stderr, "\n");
        fprintf(stderr, "  Blurry+Sharp:\n");
        fprintf(stderr, "    Output:  %s\n", result_sharp.text.c_str());
        fprintf(stderr, "    Tokens:  %zu\n", result_sharp.tokens.size());
        fprintf(stderr, "    Time:    %.2f s\n", result_sharp.elapsed_s);
        fprintf(stderr, "    Speed:   %.2f t/s\n", result_sharp.tokens_per_s);
        fprintf(stderr, "\n");

        size_t min_len = std::min(result_blurry.tokens.size(), result_sharp.tokens.size());
        size_t n_match = 0;
        size_t first_diff = min_len;
        for (size_t i = 0; i < min_len; ++i) {
            if (result_blurry.tokens[i] == result_sharp.tokens[i]) {
                n_match++;
            } else if (first_diff == min_len) {
                first_diff = i;
            }
        }

        fprintf(stderr, "  Token agreement: %zu/%zu (%.1f%%)\n",
                n_match, min_len,
                min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        if (first_diff < min_len) {
            fprintf(stderr, "  First divergence at token %zu\n", first_diff);
            fprintf(stderr, "    Blurry: \"%s\" (id=%d)\n",
                    common_token_to_piece(ctx, result_blurry.tokens[first_diff]).c_str(),
                    result_blurry.tokens[first_diff]);
            fprintf(stderr, "    Sharp:  \"%s\" (id=%d)\n",
                    common_token_to_piece(ctx, result_sharp.tokens[first_diff]).c_str(),
                    result_sharp.tokens[first_diff]);
        } else if (result_blurry.tokens == result_sharp.tokens) {
            fprintf(stderr, "  Outputs are identical (sharp overlay did not change generation)\n");
            fprintf(stderr, "  This is expected when blurry and sharp have the same weights,\n");
            fprintf(stderr, "  or when the quality difference doesn't affect greedy decoding.\n");
        }

        if (result_blurry.tokens_per_s > 0 && result_sharp.tokens_per_s > 0) {
            double overhead = 1.0 - (result_sharp.tokens_per_s / result_blurry.tokens_per_s);
            fprintf(stderr, "\n  Overlay overhead: %.1f%% speed reduction\n",
                    overhead * 100.0);
            fprintf(stderr, "  (Note: overlay apply/restore cost is amortized; per-layer\n");
            fprintf(stderr, "   restore-after-forward adds overhead per decode step.)\n");
        }
        fprintf(stderr, "\n=========================\n\n");
    }

    // -----------------------------------------------------------------------
    // Dynamic sharpening mode
    // -----------------------------------------------------------------------
    generation_result result_dynamic;

    if (bsctx && params.bs_dynamic) {
        fprintf(stderr, "--- Dynamic entropy-based sharpening ---\n");
        fprintf(stderr, "  Entropy threshold: %.2f\n", params.bs_entropy_threshold);
        fprintf(stderr, "  Top-K layers:      %d\n", params.bs_dynamic_top_k);
        fprintf(stderr, "\n");

        llama_kv_cache_clear(ctx);

        result_dynamic = run_dynamic_generation(
            model, ctx, bsctx,
            prompt_tokens,
            params.n_predict,
            params.bs_entropy_threshold,
            params.bs_dynamic_top_k,
            params.bs_verbose
        );

        fprintf(stderr, "\nDynamic generation results:\n");
        fprintf(stderr, "  Generated: %s\n", result_dynamic.text.c_str());
        fprintf(stderr, "  Tokens:    %zu\n", result_dynamic.tokens.size());
        fprintf(stderr, "  Time:      %.2f s\n", result_dynamic.elapsed_s);
        fprintf(stderr, "  Speed:     %.2f t/s\n", result_dynamic.tokens_per_s);
        fprintf(stderr, "  Sharp steps:  %d / %zu  (%.1f%%)\n",
                result_dynamic.n_sharp_steps,
                result_dynamic.tokens.size(),
                result_dynamic.tokens.empty() ? 0.0 :
                    100.0 * result_dynamic.n_sharp_steps / result_dynamic.tokens.size());
        fprintf(stderr, "  Blurry steps: %d / %zu  (%.1f%%)\n",
                result_dynamic.n_blurry_steps,
                result_dynamic.tokens.size(),
                result_dynamic.tokens.empty() ? 0.0 :
                    100.0 * result_dynamic.n_blurry_steps / result_dynamic.tokens.size());
        fprintf(stderr, "\n");

        if (!result_blurry.tokens.empty()) {
            size_t min_len = std::min(result_blurry.tokens.size(), result_dynamic.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_blurry.tokens[i] == result_dynamic.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs Blurry-only: %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        if (!result_sharp.tokens.empty()) {
            size_t min_len = std::min(result_sharp.tokens.size(), result_dynamic.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_sharp.tokens[i] == result_dynamic.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs All-sharp:   %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // Probe-based sharpening mode
    // -----------------------------------------------------------------------
    generation_result result_probe;

    if (bsctx && params.bs_probe_interval > 0) {
        fprintf(stderr, "--- Probe-based sharpening ---\n");
        fprintf(stderr, "  Probe interval: %d tokens\n", params.bs_probe_interval);
        fprintf(stderr, "  Probe hold:     %d tokens\n", params.bs_probe_hold);
        fprintf(stderr, "  Top-K layers:   %d\n", params.bs_dynamic_top_k);
        fprintf(stderr, "\n");

        llama_kv_cache_clear(ctx);

        result_probe = run_probe_generation(
            model, ctx, bsctx,
            prompt_tokens,
            params.n_predict,
            params.bs_probe_interval,
            params.bs_probe_hold,
            params.bs_dynamic_top_k,
            params.bs_verbose
        );

        fprintf(stderr, "\nProbe generation results:\n");
        fprintf(stderr, "  Generated: %s\n", result_probe.text.c_str());
        fprintf(stderr, "  Tokens:    %zu\n", result_probe.tokens.size());
        fprintf(stderr, "  Time:      %.2f s\n", result_probe.elapsed_s);
        fprintf(stderr, "  Speed:     %.2f t/s\n", result_probe.tokens_per_s);
        fprintf(stderr, "  Sharp steps:  %d / %zu  (%.1f%%)\n",
                result_probe.n_sharp_steps,
                result_probe.tokens.size(),
                result_probe.tokens.empty() ? 0.0 :
                    100.0 * result_probe.n_sharp_steps / result_probe.tokens.size());
        fprintf(stderr, "  Blurry steps: %d / %zu  (%.1f%%)\n",
                result_probe.n_blurry_steps,
                result_probe.tokens.size(),
                result_probe.tokens.empty() ? 0.0 :
                    100.0 * result_probe.n_blurry_steps / result_probe.tokens.size());
        fprintf(stderr, "\n");

        if (!result_blurry.tokens.empty()) {
            size_t min_len = std::min(result_blurry.tokens.size(), result_probe.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_blurry.tokens[i] == result_probe.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs Blurry-only: %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        if (!result_sharp.tokens.empty()) {
            size_t min_len = std::min(result_sharp.tokens.size(), result_probe.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_sharp.tokens[i] == result_probe.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs All-sharp:   %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // Speculative verification mode
    // -----------------------------------------------------------------------
    generation_result result_spec;

    if (bsctx && params.bs_speculative) {
        fprintf(stderr, "--- Speculative verification ---\n");
        fprintf(stderr, "  Draft tokens: %d\n", params.bs_spec_draft);
        fprintf(stderr, "\n");

        llama_kv_cache_clear(ctx);

        result_spec = run_speculative_generation(
            model, ctx, bsctx,
            prompt_tokens,
            params.n_predict,
            params.bs_spec_draft,
            params.bs_dynamic_top_k,
            params.bs_verbose
        );

        fprintf(stderr, "\nSpeculative generation results:\n");
        fprintf(stderr, "  Generated: %s\n", result_spec.text.c_str());
        fprintf(stderr, "  Tokens:    %zu\n", result_spec.tokens.size());
        fprintf(stderr, "  Time:      %.2f s\n", result_spec.elapsed_s);
        fprintf(stderr, "  Speed:     %.2f t/s\n", result_spec.tokens_per_s);
        fprintf(stderr, "  Draft total:    %d\n", result_spec.n_blurry_steps);
        fprintf(stderr, "  Accepted total: %d\n", result_spec.n_sharp_steps);
        if (result_spec.n_blurry_steps > 0) {
            fprintf(stderr, "  Acceptance rate: %.1f%%\n",
                    100.0 * result_spec.n_sharp_steps / result_spec.n_blurry_steps);
        }
        fprintf(stderr, "\n");

        if (!result_blurry.tokens.empty()) {
            size_t min_len = std::min(result_blurry.tokens.size(), result_spec.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_blurry.tokens[i] == result_spec.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs Blurry-only: %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        if (!result_sharp.tokens.empty()) {
            size_t min_len = std::min(result_sharp.tokens.size(), result_spec.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_sharp.tokens[i] == result_spec.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs All-sharp:   %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // MoE Combination Expert mode
    // -----------------------------------------------------------------------
    generation_result result_moe_combo;

    if (bsctx && params.bs_moe_combination) {
        fprintf(stderr, "--- MoE Combination Expert mode ---\n");
        fprintf(stderr, "  Draft tokens:   %d\n", params.bs_spec_draft);
        fprintf(stderr, "  Entropy thresh: %.2f\n", params.bs_entropy_threshold);
        fprintf(stderr, "  Top-K override: %d (0=model default)\n", params.bs_moe_top_k_override);
        fprintf(stderr, "\n");

        llama_kv_cache_clear(ctx);

        result_moe_combo = run_moe_combination_generation(
            model, ctx, bsctx,
            prompt_tokens,
            params.n_predict,
            params.bs_spec_draft,
            params.bs_entropy_threshold,
            params.bs_dynamic_top_k,
            params.bs_moe_top_k_override,
            params.bs_verbose
        );

        fprintf(stderr, "\nMoE Combination Expert results:\n");
        fprintf(stderr, "  Generated: %s\n", result_moe_combo.text.c_str());
        fprintf(stderr, "  Tokens:    %zu\n", result_moe_combo.tokens.size());
        fprintf(stderr, "  Time:      %.2f s\n", result_moe_combo.elapsed_s);
        fprintf(stderr, "  Speed:     %.2f t/s\n", result_moe_combo.tokens_per_s);
        fprintf(stderr, "  Draft total:     %d\n", result_moe_combo.n_draft_total);
        fprintf(stderr, "  Accepted total:  %d\n", result_moe_combo.n_accepted_total);
        if (result_moe_combo.n_draft_total > 0) {
            fprintf(stderr, "  Acceptance rate: %.1f%%\n",
                    100.0 * result_moe_combo.n_accepted_total / result_moe_combo.n_draft_total);
        }
        fprintf(stderr, "\n");

        if (!result_blurry.tokens.empty()) {
            size_t min_len = std::min(result_blurry.tokens.size(), result_moe_combo.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_blurry.tokens[i] == result_moe_combo.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs Blurry-only: %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        if (!result_sharp.tokens.empty()) {
            size_t min_len = std::min(result_sharp.tokens.size(), result_moe_combo.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_sharp.tokens[i] == result_moe_combo.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs All-sharp:   %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // Combined adaptive mode
    // -----------------------------------------------------------------------
    generation_result result_combined;

    if (bsctx && params.bs_combined) {
        fprintf(stderr, "--- Combined adaptive mode (entropy + probe + speculative) ---\n");
        fprintf(stderr, "  Draft tokens:   %d\n", params.bs_spec_draft);
        fprintf(stderr, "  Probe stride:   %d\n", params.bs_combined_probe_stride);
        fprintf(stderr, "  Entropy thresh: %.2f\n", params.bs_entropy_threshold);
        fprintf(stderr, "  Top-K layers:   %d (for probes)\n", params.bs_dynamic_top_k);
        fprintf(stderr, "\n");

        llama_kv_cache_clear(ctx);

        result_combined = run_combined_generation(
            model, ctx, bsctx,
            prompt_tokens,
            params.n_predict,
            params.bs_spec_draft,
            params.bs_combined_probe_stride,
            params.bs_entropy_threshold,
            params.bs_dynamic_top_k,
            params.bs_verbose
        );

        fprintf(stderr, "\nCombined generation results:\n");
        fprintf(stderr, "  Generated: %s\n", result_combined.text.c_str());
        fprintf(stderr, "  Tokens:    %zu\n", result_combined.tokens.size());
        fprintf(stderr, "  Time:      %.2f s\n", result_combined.elapsed_s);
        fprintf(stderr, "  Speed:     %.2f t/s\n", result_combined.tokens_per_s);
        fprintf(stderr, "  Draft total:     %d\n", result_combined.n_draft_total);
        fprintf(stderr, "  Accepted total:  %d\n", result_combined.n_accepted_total);
        if (result_combined.n_draft_total > 0) {
            fprintf(stderr, "  Acceptance rate: %.1f%%\n",
                    100.0 * result_combined.n_accepted_total / result_combined.n_draft_total);
        }
        if (result_combined.n_probes > 0) {
            fprintf(stderr, "  Probes: %d (%d disagreed = %.1f%%)\n",
                    result_combined.n_probes,
                    result_combined.n_probe_disagreements,
                    100.0 * result_combined.n_probe_disagreements / result_combined.n_probes);
        }
        fprintf(stderr, "\n");

        if (!result_blurry.tokens.empty()) {
            size_t min_len = std::min(result_blurry.tokens.size(), result_combined.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_blurry.tokens[i] == result_combined.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs Blurry-only: %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        if (!result_sharp.tokens.empty()) {
            size_t min_len = std::min(result_sharp.tokens.size(), result_combined.tokens.size());
            size_t n_match = 0;
            for (size_t i = 0; i < min_len; ++i) {
                if (result_sharp.tokens[i] == result_combined.tokens[i]) n_match++;
            }
            fprintf(stderr, "  vs All-sharp:   %zu/%zu tokens agree (%.1f%%)\n",
                    n_match, min_len,
                    min_len > 0 ? 100.0 * n_match / min_len : 0.0);
        }
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // Per-layer sharpening demo (verbose only)
    // -----------------------------------------------------------------------
    if (bsctx && params.bs_verbose) {
        fprintf(stderr, "\n--- Per-layer sharpening demo ---\n");

        llama_blurry_sharp_state state = llama_blurry_sharp_get_state(bsctx);

        for (int il = 0; il < state.n_layers_total && il < 4; ++il) {
            int32_t ret = llama_blurry_sharp_apply_layer(bsctx, il);
            fprintf(stderr, "  Layer %d: apply returned %d, is_sharp=%s\n",
                    il, ret,
                    llama_blurry_sharp_is_layer_sharp(bsctx, il) ? "true" : "false");
        }

        state = llama_blurry_sharp_get_state(bsctx);
        fprintf(stderr, "  State: %d layers sharpened, %" PRId64 " bytes backup\n",
                state.n_layers_sharpened, state.total_backup_bytes);

        llama_blurry_sharp_restore_all(bsctx);
        state = llama_blurry_sharp_get_state(bsctx);
        fprintf(stderr, "  After restore_all: %d layers sharpened\n\n",
                state.n_layers_sharpened);
    }

    // -----------------------------------------------------------------------
    // Eviction demo (verbose only)
    // -----------------------------------------------------------------------
    if (bsctx && params.bs_verbose) {
        fprintf(stderr, "--- Eviction demo ---\n");

        llama_blurry_sharp_state state = llama_blurry_sharp_get_state(bsctx);
        int n_to_sharpen = std::min(6, state.n_layers_total);
        for (int il = 0; il < n_to_sharpen; ++il) {
            llama_blurry_sharp_apply_layer(bsctx, il);
        }

        state = llama_blurry_sharp_get_state(bsctx);
        fprintf(stderr, "  Before eviction: %d layers sharpened, %" PRId64 " bytes\n",
                state.n_layers_sharpened, state.total_backup_bytes);

        int64_t target = state.total_backup_bytes / 2;
        int32_t n_evicted = llama_blurry_sharp_evict_to_budget(bsctx, target);

        state = llama_blurry_sharp_get_state(bsctx);
        fprintf(stderr, "  After eviction (target=%" PRId64 "): evicted %d layers, "
                       "%d remain, %" PRId64 " bytes\n\n",
                target, n_evicted,
                state.n_layers_sharpened, state.total_backup_bytes);

        llama_blurry_sharp_restore_all(bsctx);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    if (bsctx) {
        llama_blurry_sharp_free(bsctx);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    fprintf(stderr, "Done.\n");
    return 0;
}