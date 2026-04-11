#!/usr/bin/env python3
"""Train a routing predictor from collected (hidden_state, expert_ids) data.

Usage:
    1. Collect training data by running the server with --routing-collector:
       ./llama-server -m model.gguf --ring-experts 4096 --routing-collector routing_data.bin

    2. Train the predictor:
       python scripts/train_routing_predictor.py \
           --data routing_data.bin \
           --output routing_predictor.bin \
           --lookahead 3

    3. Use the predictor:
       ./llama-server -m model.gguf --ring-experts 4096 --routing-predictor routing_predictor.bin
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def read_training_data(path: str):
    """Read the binary training data file produced by llama_routing_collector.

    File format:
        Header: [int32 n_embd][int32 n_expert][int32 n_expert_used]
        Records: repeated [int32 layer_idx][int32 n_tokens][int32 n_expert_used]
                          [int32[n_expert_used * n_tokens] expert_ids]

    Note: hidden states are collected separately via the ffn_inp callback.
    For the first version, we train on expert co-occurrence patterns
    (which experts at layer K predict which experts at layer K+offset).
    """
    with open(path, 'rb') as f:
        header = struct.unpack('3i', f.read(12))
        n_embd, n_expert, n_expert_used = header
        print(f"Training data: n_embd={n_embd}, n_expert={n_expert}, n_expert_used={n_expert_used}")

        records = []  # list of (layer_idx, expert_ids_per_token)
        while True:
            buf = f.read(12)
            if len(buf) < 12:
                break
            layer_idx, n_tokens, n_eu = struct.unpack('3i', buf)

            ids_buf = f.read(n_eu * n_tokens * 4)
            if len(ids_buf) < n_eu * n_tokens * 4:
                break
            ids = np.frombuffer(ids_buf, dtype=np.int32).reshape(n_tokens, n_eu)
            records.append((layer_idx, ids))

    print(f"Read {len(records)} records")
    return n_embd, n_expert, n_expert_used, records


def build_cooccurrence_predictor(records, n_expert, n_expert_used, n_lookahead):
    """Build a simple co-occurrence based predictor.

    For each (source_layer, target_layer_offset), count how often
    expert E_src at layer K co-occurs with expert E_tgt at layer K+offset.
    The predictor weight matrix is this co-occurrence matrix (normalized).

    This is a baseline that doesn't use hidden states at all — just
    expert-to-expert transition probabilities. A learned linear predictor
    on hidden states would be better but requires the hidden state data.
    """
    # Group records by layer
    layer_records = {}
    for layer_idx, ids in records:
        if layer_idx not in layer_records:
            layer_records[layer_idx] = []
        layer_records[layer_idx].append(ids)

    # Sort layers
    sorted_layers = sorted(layer_records.keys())
    print(f"Layers in data: {sorted_layers[:5]}...{sorted_layers[-5:]}")

    # Build co-occurrence matrices for each lookahead offset
    weights = []
    biases = []

    for offset in range(1, n_lookahead + 1):
        # Co-occurrence: [n_expert, n_expert] — how often expert i at layer K
        # appears with expert j at layer K+offset
        cooccur = np.zeros((n_expert, n_expert), dtype=np.float64)
        freq = np.zeros(n_expert, dtype=np.float64)  # marginal frequency at target

        n_pairs = 0
        for src_layer in sorted_layers:
            tgt_layer = src_layer + offset
            if tgt_layer not in layer_records:
                continue

            src_batches = layer_records[src_layer]
            tgt_batches = layer_records[tgt_layer]

            # Pair up records (they should correspond to the same decode calls)
            for src_ids, tgt_ids in zip(src_batches, tgt_batches):
                # Use last token's experts (generation token)
                src_experts = set(src_ids[-1])
                tgt_experts = set(tgt_ids[-1])

                for se in src_experts:
                    for te in tgt_experts:
                        if 0 <= se < n_expert and 0 <= te < n_expert:
                            cooccur[se, te] += 1.0
                for te in tgt_experts:
                    if 0 <= te < n_expert:
                        freq[te] += 1.0
                n_pairs += 1

        if n_pairs > 0:
            # Normalize to get conditional probability: P(tgt_expert | src_expert_set)
            row_sums = cooccur.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cooccur /= row_sums

            # The weight matrix maps a one-hot expert indicator to target expert logits
            # For actual use with hidden states, this would be [n_embd, n_expert]
            # For co-occurrence baseline, we use [n_expert, n_expert]
            weights.append(cooccur.astype(np.float32))

            # Bias = log marginal frequency (prior)
            freq_norm = freq / max(freq.sum(), 1.0)
            freq_norm = np.clip(freq_norm, 1e-8, 1.0)
            biases.append(np.log(freq_norm).astype(np.float32))
        else:
            weights.append(np.zeros((n_expert, n_expert), dtype=np.float32))
            biases.append(np.zeros(n_expert, dtype=np.float32))

        print(f"  Offset {offset}: {n_pairs} pairs, "
              f"cooccur density: {(cooccur > 0).mean():.3f}")

    return weights, biases


def save_predictor(path, n_lookahead, n_expert, n_embd, n_expert_used, weights, biases):
    """Save predictor in the binary format expected by llama_routing_predictor_load.

    Format: [int32 n_lookahead][int32 n_expert][int32 n_embd][int32 n_expert_used]
            for each offset: [float[n_embd * n_expert] weights][float[n_expert] bias]
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', n_lookahead, n_expert, n_embd, n_expert_used))
        for i in range(n_lookahead):
            f.write(weights[i].tobytes())
            f.write(biases[i].tobytes())
    print(f"Saved predictor to {path}")
    total_mb = (16 + n_lookahead * (n_embd * n_expert + n_expert) * 4) / 1024 / 1024
    print(f"  Size: {total_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Train routing predictor for expert prefetching")
    parser.add_argument("--data", required=True, help="Path to collected routing data (.bin)")
    parser.add_argument("--output", "-o", required=True, help="Output predictor file (.bin)")
    parser.add_argument("--lookahead", type=int, default=3, help="Number of layers to predict ahead")
    args = parser.parse_args()

    n_embd, n_expert, n_expert_used, records = read_training_data(args.data)

    if len(records) < 10:
        print("ERROR: not enough training data. Run inference with --routing-collector first.")
        return

    print(f"\nBuilding co-occurrence predictor (lookahead={args.lookahead})...")
    weights, biases = build_cooccurrence_predictor(
        records, n_expert, n_expert_used, args.lookahead)

    # For the co-occurrence baseline, n_embd = n_expert
    # (input is one-hot expert indicator, not hidden state)
    # A proper hidden-state predictor would use the actual n_embd
    save_predictor(args.output, args.lookahead, n_expert, n_expert, n_expert_used,
                   weights, biases)

    print("\nNote: this is a co-occurrence baseline predictor.")
    print("For better accuracy, train a hidden-state predictor using PyTorch")
    print("with the collected data.")


if __name__ == "__main__":
    main()
