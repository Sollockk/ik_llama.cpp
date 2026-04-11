#pragma once

// Expert demand bitmask: fast bit-parallel operations for MoE scheduling.
//
// Encodes expert sets as fixed-width bitmasks (up to 256 experts).
// All set operations (union, intersect, diff, count, iterate) are
// branch-free and operate on 3-4 uint64 words — fits in registers.
//
// Primary uses:
//   1. Demand matrix:  demand[layer] = union of experts needed across B tokens
//   2. Ring state:     which experts are currently in VRAM
//   3. Belady eviction: for each cached expert, find its next future use
//   4. Expert similarity (Hamming): approximate routing pattern overlap

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Configuration ---
// Max experts supported. 256 = 4 × uint64. GLM-4 has 160.
#define EXPERT_MASK_WORDS 4
#define EXPERT_MASK_MAX   (EXPERT_MASK_WORDS * 64)  // 256

typedef struct {
    uint64_t bits[EXPERT_MASK_WORDS];
} expert_mask_t;

// --- Basic operations ---

static inline void expert_mask_clear(expert_mask_t * m) {
    memset(m->bits, 0, sizeof(m->bits));
}

static inline void expert_mask_set(expert_mask_t * m, int expert_id) {
    m->bits[expert_id >> 6] |= (1ULL << (expert_id & 63));
}

static inline void expert_mask_unset(expert_mask_t * m, int expert_id) {
    m->bits[expert_id >> 6] &= ~(1ULL << (expert_id & 63));
}

static inline int expert_mask_test(const expert_mask_t * m, int expert_id) {
    return (m->bits[expert_id >> 6] >> (expert_id & 63)) & 1;
}

static inline int expert_mask_empty(const expert_mask_t * m) {
    return (m->bits[0] | m->bits[1] | m->bits[2] | m->bits[3]) == 0;
}

// --- Set operations ---

static inline expert_mask_t expert_mask_union(const expert_mask_t * a, const expert_mask_t * b) {
    expert_mask_t r;
    for (int i = 0; i < EXPERT_MASK_WORDS; i++) r.bits[i] = a->bits[i] | b->bits[i];
    return r;
}

static inline expert_mask_t expert_mask_intersect(const expert_mask_t * a, const expert_mask_t * b) {
    expert_mask_t r;
    for (int i = 0; i < EXPERT_MASK_WORDS; i++) r.bits[i] = a->bits[i] & b->bits[i];
    return r;
}

// Elements in a but not in b
static inline expert_mask_t expert_mask_diff(const expert_mask_t * a, const expert_mask_t * b) {
    expert_mask_t r;
    for (int i = 0; i < EXPERT_MASK_WORDS; i++) r.bits[i] = a->bits[i] & ~b->bits[i];
    return r;
}

static inline expert_mask_t expert_mask_xor(const expert_mask_t * a, const expert_mask_t * b) {
    expert_mask_t r;
    for (int i = 0; i < EXPERT_MASK_WORDS; i++) r.bits[i] = a->bits[i] ^ b->bits[i];
    return r;
}

// --- Counting ---

static inline int _emask_popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    // Fallback: Hamming weight
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

static inline int expert_mask_count(const expert_mask_t * m) {
    int c = 0;
    for (int i = 0; i < EXPERT_MASK_WORDS; i++) c += _emask_popcnt64(m->bits[i]);
    return c;
}

// Hamming distance: number of experts in one set but not the other
static inline int expert_mask_hamming(const expert_mask_t * a, const expert_mask_t * b) {
    expert_mask_t x = expert_mask_xor(a, b);
    return expert_mask_count(&x);
}

// Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 0.0-1.0. 0 if both empty.
static inline float expert_mask_jaccard(const expert_mask_t * a, const expert_mask_t * b) {
    expert_mask_t isect = expert_mask_intersect(a, b);
    expert_mask_t unn   = expert_mask_union(a, b);
    int u = expert_mask_count(&unn);
    if (u == 0) return 0.0f;
    return (float)expert_mask_count(&isect) / (float)u;
}

// --- Iteration (Kernighan's bit trick) ---

// Call fn(expert_id, user_data) for each set bit. Branch-free inner loop.
typedef void (*expert_mask_visitor_fn)(int expert_id, void * user_data);

static inline void expert_mask_foreach(const expert_mask_t * m, expert_mask_visitor_fn fn, void * user_data) {
    for (int w = 0; w < EXPERT_MASK_WORDS; w++) {
        uint64_t bits = m->bits[w];
        while (bits) {
#if defined(__GNUC__) || defined(__clang__)
            int bit = __builtin_ctzll(bits);
#else
            int bit = 0;
            uint64_t tmp = bits;
            if (!(tmp & 0xFFFFFFFF)) { bit += 32; tmp >>= 32; }
            if (!(tmp & 0xFFFF))     { bit += 16; tmp >>= 16; }
            if (!(tmp & 0xFF))       { bit += 8;  tmp >>= 8;  }
            if (!(tmp & 0xF))        { bit += 4;  tmp >>= 4;  }
            if (!(tmp & 0x3))        { bit += 2;  tmp >>= 2;  }
            if (!(tmp & 0x1))        { bit += 1; }
#endif
            fn(w * 64 + bit, user_data);
            bits &= bits - 1;  // clear lowest set bit
        }
    }
}

// --- Belady eviction support ---

// Given demand[n_layers] bitmasks (future expert needs per layer),
// find the NEXT layer at which expert_id is needed.
// Returns n_layers if expert is never needed again (= best eviction candidate).
static inline int expert_mask_next_use(const expert_mask_t * demand, int n_layers,
                                        int expert_id, int from_layer) {
    for (int l = from_layer; l < n_layers; l++) {
        if (expert_mask_test(&demand[l], expert_id)) return l;
    }
    return n_layers;  // never needed again
}

// Find the expert in `candidates` mask whose next use in demand[] is furthest away.
// This is Belady's optimal eviction choice.
// Returns the expert_id to evict, or -1 if candidates is empty.
static inline int expert_mask_belady_victim(const expert_mask_t * candidates,
                                             const expert_mask_t * demand, int n_layers,
                                             int current_layer) {
    int best_expert = -1;
    int best_next_use = -1;

    for (int w = 0; w < EXPERT_MASK_WORDS; w++) {
        uint64_t bits = candidates->bits[w];
        while (bits) {
#if defined(__GNUC__) || defined(__clang__)
            int bit = __builtin_ctzll(bits);
#else
            int bit = 0;
            { uint64_t t = bits; while (!(t & 1)) { bit++; t >>= 1; } }
#endif
            int eid = w * 64 + bit;
            int nu = expert_mask_next_use(demand, n_layers, eid, current_layer);
            if (nu > best_next_use) {
                best_next_use = nu;
                best_expert = eid;
            }
            bits &= bits - 1;
        }
    }
    return best_expert;
}

// --- Demand matrix builder ---

// Build the per-layer demand union from a demand matrix [n_layers][n_tokens].
// Input:  per_token_demand[layer * n_tokens + token] contains expert mask per token per layer
// Output: demand_union[layer] = union across all tokens
static inline void expert_mask_build_demand_union(
        const expert_mask_t * per_token_demand,  // [n_layers * n_tokens]
        int n_layers, int n_tokens,
        expert_mask_t * demand_union) {           // [n_layers] output
    for (int l = 0; l < n_layers; l++) {
        expert_mask_clear(&demand_union[l]);
        for (int t = 0; t < n_tokens; t++) {
            demand_union[l] = expert_mask_union(&demand_union[l],
                                                 &per_token_demand[l * n_tokens + t]);
        }
    }
}

// --- Expert routing signature ---
// Compact representation of an expert's co-occurrence pattern for similarity.
// Built by ORing the per-layer demand masks where this expert appears.
// Two experts with similar signatures are "substitutes" — loading one
// reduces the urgency of loading the other.

static inline expert_mask_t expert_mask_build_signature(
        const expert_mask_t * demand_union,  // [n_layers]
        int n_layers, int expert_id) {
    expert_mask_t sig;
    expert_mask_clear(&sig);
    for (int l = 0; l < n_layers; l++) {
        if (expert_mask_test(&demand_union[l], expert_id)) {
            sig = expert_mask_union(&sig, &demand_union[l]);
        }
    }
    // The signature is the union of all layers where this expert is needed,
    // showing which OTHER experts co-occur with it.
    return sig;
}

#ifdef __cplusplus
}
#endif
