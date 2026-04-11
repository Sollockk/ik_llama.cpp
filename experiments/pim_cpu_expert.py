#!/usr/bin/env python3
"""
PIM-style expert inference: weights stay in RAM, only activations cross the bus.

Compares:
  1. Full GPU matmul (simulated: load weights to "VRAM" + matmul)
  2. CPU-side full matmul (weights in RAM, send activation, get result)
  3. CPU-side top-K matmul (only compute rows that matter for this token)

Measures wall-clock time, PCIe bytes transferred, and output quality.
"""

import numpy as np
import time
import argparse
from typing import Optional


def simulate_pcie_transfer_time(bytes_transferred: int, bandwidth_gbps: float = 14.0) -> float:
    """Simulate PCIe transfer time in seconds."""
    return bytes_transferred / (bandwidth_gbps * 1e9)


def cpu_matmul_full(hidden: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Full matmul on CPU. hidden: [B, D_in], weights: [D_out, D_in]."""
    return hidden @ weights.T


def cpu_matmul_topk(hidden: np.ndarray, weights: np.ndarray,
                    k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-K matmul: compute all dot products, return only top-K per token.

    Returns (output, indices) where output has zeros except top-K dims.
    """
    # Full matmul (we need all dots to find top-K)
    y_full = hidden @ weights.T  # [B, D_out]
    B, D_out = y_full.shape

    # Select top-K per token
    output = np.zeros_like(y_full)
    indices = np.zeros((B, k), dtype=np.int32)
    for b in range(B):
        idx = np.argpartition(np.abs(y_full[b]), -k)[-k:]
        indices[b] = idx
        output[b, idx] = y_full[b, idx]

    return output, indices


def cpu_matmul_topk_approx(hidden: np.ndarray, weights: np.ndarray,
                           row_norms: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Approximate top-K: use precomputed row norms × hidden norm as proxy.

    Instead of computing ALL dot products then selecting top-K,
    estimate which rows will have large dot products using:
      |dot(h, w_i)| ≈ ||h|| * ||w_i|| * cos(angle)
    Since we don't know cos(angle), use ||w_i|| as a static proxy
    combined with |h @ w_i_normalized| for a cheap directional check.

    This version uses a low-rank projection for cheap selection.
    """
    B, D_in = hidden.shape
    D_out = weights.shape[0]

    # Cheap selection: compute dot products with a random subset to estimate importance
    # Use 1/10th of rows as "pilots"
    n_pilots = max(k, D_out // 10)
    pilot_idx = np.argsort(row_norms)[-n_pilots:]  # top norms as pilots
    pilot_dots = np.abs(hidden @ weights[pilot_idx].T)  # [B, n_pilots]

    output = np.zeros((B, D_out))
    indices = np.zeros((B, k), dtype=np.int32)

    for b in range(B):
        # Select top-K from pilots
        top_pilot = np.argpartition(pilot_dots[b], -k)[-k:]
        actual_idx = pilot_idx[top_pilot]
        indices[b] = actual_idx
        output[b, actual_idx] = hidden[b] @ weights[actual_idx].T

    return output, indices


def benchmark_expert_inference(weights: np.ndarray, hidden: np.ndarray,
                               n_warmup: int = 3, n_iters: int = 10):
    """Benchmark different expert inference strategies."""
    D_out, D_in = weights.shape
    B = hidden.shape[0]
    bytes_per_val = 2  # assume fp16 for transfer calculations

    print(f"Expert weight matrix: {D_out} × {D_in} ({D_out * D_in * bytes_per_val / 1024 / 1024:.1f} MB at fp16)")
    print(f"Hidden states: {B} tokens × {D_in} dims ({B * D_in * bytes_per_val / 1024:.1f} KB at fp16)")
    print(f"Output: {B} tokens × {D_out} dims ({B * D_out * bytes_per_val / 1024:.1f} KB at fp16)")
    print()

    # Ground truth
    y_true = hidden @ weights.T

    # === Strategy 1: Traditional (load full weights over PCIe, GPU matmul) ===
    weight_bytes = D_out * D_in * bytes_per_val
    pcie_time = simulate_pcie_transfer_time(weight_bytes)

    # Warmup
    for _ in range(n_warmup):
        _ = hidden @ weights.T

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y1 = hidden @ weights.T
    t_matmul = (time.perf_counter() - t0) / n_iters

    print(f"Strategy 1: Load weights to GPU + matmul")
    print(f"  PCIe transfer: {weight_bytes / 1024 / 1024:.1f} MB → {pcie_time * 1000:.2f} ms")
    print(f"  CPU matmul time: {t_matmul * 1000:.2f} ms (simulating GPU)")
    print(f"  Total: {(pcie_time + t_matmul) * 1000:.2f} ms")
    print()

    # === Strategy 2: PIM full (send hidden to CPU, CPU matmul, send result back) ===
    hidden_bytes = B * D_in * bytes_per_val
    result_bytes = B * D_out * bytes_per_val
    pim_pcie_bytes = hidden_bytes + result_bytes
    pim_pcie_time = simulate_pcie_transfer_time(pim_pcie_bytes)

    # Warmup
    for _ in range(n_warmup):
        _ = cpu_matmul_full(hidden, weights)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y2 = cpu_matmul_full(hidden, weights)
    t_cpu_full = (time.perf_counter() - t0) / n_iters

    err2 = np.sqrt(np.mean((y_true - y2) ** 2)) / (np.sqrt(np.mean(y_true ** 2)) + 1e-30)

    print(f"Strategy 2: PIM full (weights in RAM, send activations)")
    print(f"  PCIe transfer: {pim_pcie_bytes / 1024:.1f} KB → {pim_pcie_time * 1000:.4f} ms")
    print(f"  CPU matmul time: {t_cpu_full * 1000:.2f} ms")
    print(f"  Total: {(pim_pcie_time + t_cpu_full) * 1000:.2f} ms")
    print(f"  PCIe reduction: {weight_bytes / pim_pcie_bytes:.0f}x")
    print(f"  RMSE vs full: {err2:.2e}")
    print()

    # === Strategy 3: PIM top-K (only compute + send top-K output dims) ===
    row_norms = np.linalg.norm(weights, axis=1)

    for k_frac in [0.05, 0.10, 0.20, 0.50]:
        k = max(1, int(k_frac * D_out))

        # PCIe: send hidden, receive (k values + k indices) per token
        topk_result_bytes = B * k * (bytes_per_val + 4)  # fp16 val + int32 idx
        topk_pcie_bytes = hidden_bytes + topk_result_bytes
        topk_pcie_time = simulate_pcie_transfer_time(topk_pcie_bytes)

        # Oracle top-K (needs full matmul on CPU to select)
        for _ in range(n_warmup):
            _ = cpu_matmul_topk(hidden, weights, k)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            y3, idx3 = cpu_matmul_topk(hidden, weights, k)
        t_cpu_topk = (time.perf_counter() - t0) / n_iters

        err3 = np.sqrt(np.mean((y_true - y3) ** 2)) / (np.sqrt(np.mean(y_true ** 2)) + 1e-30)
        cos3 = np.mean([np.dot(y_true[i], y3[i]) / (np.linalg.norm(y_true[i]) * np.linalg.norm(y3[i]) + 1e-30) for i in range(B)])

        # Approximate top-K (cheap selection, only compute K rows)
        for _ in range(n_warmup):
            _ = cpu_matmul_topk_approx(hidden, weights, row_norms, k)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            y3a, idx3a = cpu_matmul_topk_approx(hidden, weights, row_norms, k)
        t_cpu_topk_approx = (time.perf_counter() - t0) / n_iters

        err3a = np.sqrt(np.mean((y_true - y3a) ** 2)) / (np.sqrt(np.mean(y_true ** 2)) + 1e-30)
        cos3a = np.mean([np.dot(y_true[i], y3a[i]) / (np.linalg.norm(y_true[i]) * np.linalg.norm(y3a[i]) + 1e-30) for i in range(B)])

        print(f"Strategy 3: PIM top-{k_frac:.0%} ({k} rows)")
        print(f"  Oracle (full matmul + select):")
        print(f"    CPU time: {t_cpu_topk * 1000:.2f} ms, RMSE: {err3:.2%}, cos_sim: {cos3:.4f}")
        print(f"  Approx (pilot selection, {k} row matmul):")
        print(f"    CPU time: {t_cpu_topk_approx * 1000:.2f} ms, RMSE: {err3a:.2%}, cos_sim: {cos3a:.4f}")
        print(f"  PCIe: {topk_pcie_bytes / 1024:.1f} KB → {topk_pcie_time * 1000:.4f} ms "
              f"({weight_bytes / topk_pcie_bytes:.0f}x reduction)")
        print()

    # === Summary ===
    print("=" * 70)
    print("SUMMARY: PCIe bytes transferred per expert call")
    print(f"  Traditional (load weights):     {weight_bytes / 1024 / 1024:.1f} MB")
    print(f"  PIM full (send activations):    {pim_pcie_bytes / 1024:.1f} KB  "
          f"({weight_bytes / pim_pcie_bytes:.0f}x reduction)")
    k10 = max(1, int(0.10 * D_out))
    topk10_bytes = hidden_bytes + B * k10 * (bytes_per_val + 4)
    print(f"  PIM top-10% (sparse result):    {topk10_bytes / 1024:.1f} KB  "
          f"({weight_bytes / topk10_bytes:.0f}x reduction)")
    print()
    print("SUMMARY: Wall time per expert (CPU matmul, simulated PCIe)")
    print(f"  Traditional: {(pcie_time + t_matmul) * 1000:.2f} ms "
          f"(PCIe {pcie_time*1000:.2f} + matmul {t_matmul*1000:.2f})")
    print(f"  PIM full:    {(pim_pcie_time + t_cpu_full) * 1000:.2f} ms "
          f"(PCIe {pim_pcie_time*1000:.4f} + matmul {t_cpu_full*1000:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIM-style expert inference benchmark")
    parser.add_argument("--gguf", type=str, help="Path to GGUF file")
    parser.add_argument("--tensor", type=str, default="blk.15.ffn_gate.weight",
                        help="Tensor name to benchmark")
    parser.add_argument("--tokens", type=int, default=1,
                        help="Number of tokens (batch size)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    if args.gguf:
        import gguf
        from gguf.quants import dequantize
        print(f"Loading {args.tensor} from {args.gguf}...")
        reader = gguf.GGUFReader(args.gguf)
        for t in reader.tensors:
            if t.name == args.tensor:
                data = dequantize(t.data, t.tensor_type)
                weights = data.reshape(t.shape[0], t.shape[1]).astype(np.float64)
                break
        else:
            print(f"Tensor '{args.tensor}' not found")
            exit(1)
    else:
        print("Using synthetic weights (4096 × 12288)")
        rng = np.random.default_rng(42)
        weights = rng.standard_normal((4096, 12288)) * 0.01

    rng = np.random.default_rng(123)
    hidden = rng.standard_normal((args.tokens, weights.shape[1])) * 0.02

    print()
    benchmark_expert_inference(weights, hidden, n_iters=args.iters)
