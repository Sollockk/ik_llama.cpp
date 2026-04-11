#!/usr/bin/env python3
"""Train a hidden-state routing predictor from collected data.

This trains a linear projection from hidden states to expert logits,
which is far more accurate than the co-occurrence baseline.

Usage:
    python scripts/train_routing_predictor_hidden.py \
        --data routing_data.bin \
        --output routing_predictor.bin \
        --lookahead 3 \
        --epochs 20

Data format (written by --routing-collector):
    Header: [int32 n_embd][int32 n_expert][int32 n_expert_used]
    Records: [int32 layer_idx][int32 n_expert_used]
             [float[n_embd] hidden_state]
             [int32[n_expert_used] expert_ids]
"""

import argparse
import struct
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def read_training_data(path):
    records = []  # list of (layer_idx, hidden_state, expert_ids)

    with open(path, 'rb') as f:
        # Header
        hdr = f.read(12)
        if len(hdr) < 12:
            raise ValueError("File too small for header")
        n_embd, n_expert, n_expert_used = struct.unpack('3i', hdr)
        print(f"Data: n_embd={n_embd}, n_expert={n_expert}, n_expert_used={n_expert_used}")

        record_bytes = 8 + n_embd * 4 + n_expert_used * 4  # hdr + hidden + ids
        while True:
            buf = f.read(8)
            if len(buf) < 8:
                break
            layer_idx, neu = struct.unpack('2i', buf)

            hs_buf = f.read(n_embd * 4)
            if len(hs_buf) < n_embd * 4:
                break
            hidden = np.frombuffer(hs_buf, dtype=np.float32).copy()

            ids_buf = f.read(neu * 4)
            if len(ids_buf) < neu * 4:
                break
            expert_ids = np.frombuffer(ids_buf, dtype=np.int32).copy()

            records.append((layer_idx, hidden, expert_ids))

    print(f"Read {len(records)} records")
    return n_embd, n_expert, n_expert_used, records


def build_training_pairs(records, n_expert, n_lookahead):
    """Build (hidden_state_at_layer_K, expert_ids_at_layer_K+offset) pairs.

    Groups records by decode step (consecutive layer indices form one step).
    """
    # Group consecutive records into decode steps
    steps = []
    current_step = []
    prev_layer = -1

    for layer_idx, hidden, expert_ids in records:
        if layer_idx <= prev_layer and current_step:
            steps.append(current_step)
            current_step = []
        current_step.append((layer_idx, hidden, expert_ids))
        prev_layer = layer_idx

    if current_step:
        steps.append(current_step)

    print(f"Grouped into {len(steps)} decode steps")

    # Build pairs for each lookahead offset
    pairs = {offset: ([], []) for offset in range(1, n_lookahead + 1)}

    for step in steps:
        layer_map = {layer: (hs, ids) for layer, hs, ids in step}
        sorted_layers = sorted(layer_map.keys())

        for i, src_layer in enumerate(sorted_layers):
            src_hs, _ = layer_map[src_layer]
            for offset in range(1, n_lookahead + 1):
                tgt_layer = src_layer + offset
                if tgt_layer in layer_map:
                    _, tgt_ids = layer_map[tgt_layer]
                    # Convert expert IDs to multi-hot target
                    target = np.zeros(n_expert, dtype=np.float32)
                    for eid in tgt_ids:
                        if 0 <= eid < n_expert:
                            target[eid] = 1.0
                    pairs[offset][0].append(src_hs)
                    pairs[offset][1].append(target)

    for offset in range(1, n_lookahead + 1):
        print(f"  Offset {offset}: {len(pairs[offset][0])} pairs")

    return pairs


def train_linear(X, Y, n_expert, n_epochs=20, lr=0.01, batch_size=4096):
    """Train a linear projection from hidden states to expert logits."""
    if not HAS_TORCH:
        # Fallback: least squares
        print("  (no PyTorch, using least-squares fallback)")
        # X: [N, n_embd], Y: [N, n_expert]
        # Solve: W @ X.T ≈ Y.T  =>  W = Y.T @ X @ (X.T @ X)^-1
        XtX = X.T @ X + 0.01 * np.eye(X.shape[1])  # ridge regularization
        XtY = X.T @ Y
        W = np.linalg.solve(XtX, XtY)  # [n_embd, n_expert]
        bias = Y.mean(axis=0)
        return W.astype(np.float32), bias.astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Training on {device}")

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    n_embd = X.shape[1]
    model = nn.Linear(n_embd, n_expert).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_samples = X_t.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_t[idx])
            loss = loss_fn(logits, Y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)

            # Compute accuracy: top-k prediction vs actual
            with torch.no_grad():
                logits_all = model(X_t[:min(10000, n_samples)])
                pred_topk = torch.topk(logits_all, k=min(8, n_expert), dim=1).indices
                actual = Y_t[:min(10000, n_samples)]
                hits = 0
                total = 0
                for j in range(pred_topk.shape[0]):
                    for k in range(pred_topk.shape[1]):
                        if actual[j, pred_topk[j, k]] > 0.5:
                            hits += 1
                        total += 1
                acc = hits / max(total, 1)

            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, top-k acc={acc:.3f}")

    W = model.weight.data.cpu().numpy().T  # [n_embd, n_expert]
    b = model.bias.data.cpu().numpy()       # [n_expert]
    return W.astype(np.float32), b.astype(np.float32)


def save_predictor(path, n_lookahead, n_expert, n_embd, n_expert_used, weights, biases):
    """Save in v2 format with mode flag."""
    MAGIC_V2 = 0x52504432
    MODE_HIDDEN = 1
    with open(path, 'wb') as f:
        f.write(struct.pack('6i', MAGIC_V2, n_lookahead, n_expert, n_embd, n_expert_used, MODE_HIDDEN))
        for i in range(n_lookahead):
            f.write(weights[i].tobytes())
            f.write(biases[i].tobytes())
    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"Saved predictor to {path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Train hidden-state routing predictor")
    parser.add_argument("--data", required=True, help="Path to collected routing data (.bin)")
    parser.add_argument("--output", "-o", required=True, help="Output predictor file (.bin)")
    parser.add_argument("--lookahead", type=int, default=3, help="Layers to predict ahead")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    n_embd, n_expert, n_expert_used, records = read_training_data(args.data)

    if len(records) < 100:
        print("ERROR: not enough data. Recollect with the updated --routing-collector.")
        return

    print(f"\nBuilding training pairs (lookahead={args.lookahead})...")
    pairs = build_training_pairs(records, n_expert, args.lookahead)

    all_weights = []
    all_biases = []

    for offset in range(1, args.lookahead + 1):
        X_list, Y_list = pairs[offset]
        if len(X_list) < 10:
            print(f"  Offset {offset}: too few pairs, using zeros")
            all_weights.append(np.zeros((n_embd, n_expert), dtype=np.float32))
            all_biases.append(np.zeros(n_expert, dtype=np.float32))
            continue

        X = np.stack(X_list)
        Y = np.stack(Y_list)

        # Normalize hidden states
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X = (X - X_mean) / X_std

        print(f"\n  Training offset {offset} ({len(X)} samples, {n_embd} -> {n_expert})...")
        W, b = train_linear(X, Y, n_expert, n_epochs=args.epochs, lr=args.lr)

        # Fold normalization into weights: W_adjusted = W / X_std, b_adjusted = b - (X_mean/X_std) @ W
        W_adj = W / X_std[:, None]
        b_adj = b - (X_mean / X_std) @ W

        all_weights.append(W_adj)
        all_biases.append(b_adj)

    save_predictor(args.output, args.lookahead, n_expert, n_embd, n_expert_used,
                   all_weights, all_biases)

    print("\nDone! This is a hidden-state predictor (mode=hidden).")
    print("Use with: --routing-predictor", args.output)


if __name__ == "__main__":
    main()
