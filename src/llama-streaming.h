#pragma once

// llama-streaming.h — Streaming micro-graph execution engine
//
// Replaces the monolithic graph-build-then-execute model with an interpreter
// loop that builds and executes tiny micro-graphs one layer at a time.
// An execution plan drives dynamic decisions: skip, sharpen, loop, halt.
// Layers exist only for the moment they compute, then are discarded.
//
// v2: Multi-signal replanning (probe + entropy + expert diversity),
//     cross-token EMA memory, HALT/CHECKPOINT ops, diagnostic logging.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct llama_context;
struct llama_batch;
struct llama_blurry_sharp_context;
struct llama_gate_probe_context;

// ---------------------------------------------------------------------------
// Per-layer signal: all data collected about a layer during one decode pass
// ---------------------------------------------------------------------------

struct llama_layer_signal {
    float   probe_score      = 0.0f;  // gate probe importance (>= 0)
    float   avg_entropy      = 0.0f;  // MoE gate entropy, avg across tokens (>= 0)
    int32_t n_unique_experts = 0;     // distinct experts selected across all tokens
    int32_t n_total_experts  = 0;     // total expert slots (n_expert_used * n_tokens)
};

// ---------------------------------------------------------------------------
// Execution plan: a program that the streaming engine interprets
// ---------------------------------------------------------------------------

enum llama_stream_op_type {
    STREAM_EXEC,            // Execute layer at given quality
    STREAM_SKIP,            // Skip layer entirely (residual passthrough)
    STREAM_EXEC_ATTN_ONLY,  // Attention + residual, skip FFN (shared expert only)
    STREAM_SHARPEN,         // Re-execute layer at sharp quality (appended by replan)
    STREAM_CHECKPOINT,      // Snapshot score state for loop-back comparison
    STREAM_HALT,            // Early exit — use current hidden state for logits
};

enum llama_stream_quality {
    STREAM_Q_BLURRY,          // Base TQ1_0 weights (always resident)
    STREAM_Q_SHARED_EXPERT,   // Skip MoE routing, shared expert FFN only
    STREAM_Q_SHARP,           // JIT overlay with Q4_K_M sharp weights
};

struct llama_stream_op {
    llama_stream_op_type type;
    int32_t              layer_id;
    llama_stream_quality quality;
    int32_t              fractal_depth = -1; // -1 = use quality enum, 0 = blurry, N = apply N delta levels
};

struct llama_execution_plan {
    std::vector<llama_stream_op> ops;
    size_t                       pc = 0;  // program counter

    // --- Threshold configuration ---
    float   replan_sharpen_sigma  = 1.5f;  // sharpen layers scoring > mean + sigma*stddev
    float   replan_skip_sigma     = -0.5f; // skip layers scoring < mean + sigma*stddev
    float   convergence_epsilon   = 0.05f; // stop looping when score norm delta < epsilon
    int32_t max_loop_iterations   = 2;     // max times a layer can be revisited
    int32_t min_protect_layers    = 3;     // never skip/sharpen first/last N layers

    // --- Multi-signal weights ---
    float   entropy_weight        = 0.3f;  // weight of entropy in composite score
    float   diversity_weight      = 0.2f;  // weight of expert diversity in composite score

    // --- HALT configuration ---
    float   halt_entropy_threshold = 1.0f; // skip all refinement if max token entropy < this
    float   min_improvement_threshold = 0.02f; // stop looping if improvement < 2%

    // --- EMA (cross-token memory) ---
    float   ema_alpha             = 0.3f;  // decay rate (higher = more responsive to current token)
    std::unordered_map<int32_t, float> ema_scores; // layer -> smoothed composite importance

    // --- Per-layer state (current token) ---
    std::unordered_map<int32_t, float>              layer_scores;   // probe-only scores
    std::unordered_map<int32_t, llama_layer_signal> layer_signals;  // full multi-signal data
    std::unordered_map<int32_t, int32_t>            layer_exec_count;

    // --- Convergence tracking ---
    float prev_score_norm = 0.0f;

    // --- Checkpoint for loop-back comparison ---
    std::unordered_map<int32_t, float> checkpoint_scores; // saved at CHECKPOINT
    float checkpoint_norm = 0.0f;

    // --- Plan navigation ---
    bool has_next() const { return pc < ops.size(); }
    llama_stream_op peek() const { return ops[pc]; }
    llama_stream_op pop() { return ops[pc++]; }

    void insert_after_current(const llama_stream_op & op) {
        ops.insert(ops.begin() + (int64_t)pc + 1, op);
    }

    void append(const llama_stream_op & op) {
        ops.push_back(op);
    }

    // --- Replanning ---

    // Single-signal replan (probe score only). Convenience wrapper.
    void replan(int32_t layer_id, float probe_score, int32_t n_layers);

    // Multi-signal replan: uses probe score + entropy + expert diversity.
    // Computes composite importance, updates EMA, and decides whether to
    // schedule SHARPEN ops for high-importance layers.
    void replan_multi(int32_t layer_id, const llama_layer_signal & signal, int32_t n_layers);

    // Compute composite importance from a layer signal.
    float compute_composite(const llama_layer_signal & signal) const;

    // Check if a loop-back pass improved scores enough to justify continuing.
    // Returns true if improvement >= min_improvement_threshold.
    bool check_loop_improvement() const;

    // Save current scores as a checkpoint for later comparison.
    void save_checkpoint();

    // Remove all pending SHARPEN ops (used by HALT and convergence).
    void clear_pending_sharpens();

    // --- Diagnostics ---

    // Compact summary string: "52 layers: 18 skip, 24 blurry, 7 sharp | EMA [0.12, 4.87]"
    std::string summary() const;
};

// ---------------------------------------------------------------------------
// Plan factories
// ---------------------------------------------------------------------------

// Build plan from current tier state (skip set + priority set).
llama_execution_plan llama_plan_from_tiers(
    int n_layers,
    const std::unordered_set<int32_t> & skip_set,
    const std::unordered_set<int32_t> & priority_set);

// All layers EXEC at BLURRY quality.
llama_execution_plan llama_plan_sequential(int n_layers);

// Build plan from cross-token EMA scores (smarter initial decisions).
// Layers below mean - skip_sigma*stddev → SKIP,
// Layers above mean + sharp_sigma*stddev → SHARP, rest → BLURRY.
// base_skip_set provides a floor of always-skipped layers.
llama_execution_plan llama_plan_from_probe_history(
    int n_layers,
    const std::unordered_map<int32_t, float> & ema_scores,
    const std::unordered_set<int32_t> & base_skip_set,
    float skip_sigma,
    float sharp_sigma);

// Build plan from per-layer entropy data (for prompt processing).
// Low-entropy layers → SKIP, high-entropy → SHARP.
llama_execution_plan llama_plan_from_entropy(
    int n_layers,
    const std::unordered_map<int32_t, std::pair<float,int32_t>> & layer_entropy,
    float entropy_skip_threshold,
    float entropy_sharp_threshold);

// Build plan with fractal depth per layer (probe score → reconstruction depth).
// Each layer gets fractal_depth proportional to its EMA importance.
llama_execution_plan llama_plan_from_probe_depth(
    int n_layers,
    int max_depth,
    const std::unordered_map<int32_t, float> & ema_scores,
    const std::unordered_set<int32_t> & base_skip_set);

// ---------------------------------------------------------------------------
// Hidden state ping-pong buffers
// ---------------------------------------------------------------------------

struct llama_hidden_state_buffers {
    ggml_backend_buffer_t buf[2] = {nullptr, nullptr};
    ggml_tensor *         tensor[2] = {nullptr, nullptr};
    ggml_context *        ctx = nullptr;
    int                   active = 0;

    ggml_tensor * input()  const { return tensor[active]; }
    ggml_tensor * output() const { return tensor[1 - active]; }
    void swap() { active = 1 - active; }
};

bool llama_hidden_state_buffers_init(
    llama_hidden_state_buffers & hs,
    ggml_backend_t               backend,
    int64_t                      n_embd,
    int32_t                      max_tokens);

void llama_hidden_state_buffers_free(llama_hidden_state_buffers & hs);

// ---------------------------------------------------------------------------
// Micro-graph: a tiny compute graph for a single transformer layer
// ---------------------------------------------------------------------------

struct llama_micro_graph {
    ggml_context * ctx   = nullptr;
    ggml_cgraph  * graph = nullptr;
    ggml_tensor  * output = nullptr;
};

void llama_micro_graph_free(llama_micro_graph & mg);

// ---------------------------------------------------------------------------
// Streaming context: state for the streaming decode engine
// ---------------------------------------------------------------------------

struct llama_streaming_context {
    bool enabled = false;

    llama_hidden_state_buffers  hs;
    llama_execution_plan        plan;

    // Metrics
    int64_t total_layers_executed  = 0;
    int64_t total_layers_skipped   = 0;
    int64_t total_layers_sharpened = 0;
    int64_t total_loop_iterations  = 0;
    int64_t total_tokens_streamed  = 0;
};

// ---------------------------------------------------------------------------
// The core: streaming decode
// ---------------------------------------------------------------------------

// Execute a decode pass using the streaming engine.
// If plan is nullptr, a default sequential plan is used.
// Returns 0 on success, negative on error.
int llama_streaming_decode(
    llama_context      & lctx,
    llama_batch          batch,
    llama_execution_plan * plan);
