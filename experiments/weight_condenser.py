#!/usr/bin/env python3
"""
Weight Condenser — 2048-style gravity simulation on weight tensors.

Shakes a weight matrix along random direction vectors. Dimensions that are
conceptually aligned (correlated along the shake direction) merge like 2048
tiles, doubling their weight. Unaligned dimensions just slide without merging.

After many shakes, dense clumps form (important blocks) and empty space opens
up (skippable regions). The move history is a lossless decompression key —
replay in reverse to reconstruct the original weights exactly.

Usage:
    python weight_condenser.py                  # run on synthetic Gaussian fog
    python weight_condenser.py --gguf path.gguf # run on a real model tensor
"""

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class Move:
    """One merge event: source_idx was added into target_idx along direction_id."""
    direction_id: int
    source_idx: int
    target_idx: int


@dataclass
class ShakeStep:
    """One full shake: a direction vector + all merges that happened."""
    direction: np.ndarray  # unit vector defining the shake direction
    merges: List[Move] = field(default_factory=list)


class WeightCondenser:
    """2048-style weight condenser.

    Treats rows of a weight matrix as points in space.
    Each shake:
      1. Project all rows onto a direction vector → scalar per row
      2. Sort rows by their projection
      3. Adjacent rows whose projections are close enough → merge (sum)
      4. Source row becomes zero (empty space)
      5. Record the move
    """

    def __init__(self, weights: np.ndarray, merge_threshold: float = 0.1):
        """
        Args:
            weights: [N, D] matrix — N rows of D-dimensional vectors
            merge_threshold: relative cosine similarity threshold for merging.
                Two adjacent rows merge if their cosine similarity along the
                shake direction exceeds this.
        """
        self.original_shape = weights.shape
        self.grid = weights.astype(np.float64).copy()
        self.merge_threshold = merge_threshold
        self.history: List[ShakeStep] = []
        self.n_rows, self.n_dims = self.grid.shape

    def _pick_direction(self, rng: np.random.Generator) -> np.ndarray:
        """Random unit vector in weight space."""
        d = rng.standard_normal(self.n_dims)
        d /= np.linalg.norm(d) + 1e-12
        return d

    def _pick_direction_from_weights(self, rng: np.random.Generator) -> np.ndarray:
        """Pick direction from the dominant variance axis of current non-zero rows."""
        alive = np.where(np.linalg.norm(self.grid, axis=1) > 1e-10)[0]
        if len(alive) < 2:
            return self._pick_direction(rng)
        # Sample a subset for speed
        sample = rng.choice(alive, size=min(len(alive), 256), replace=False)
        subset = self.grid[sample]
        # Direction = difference between two random rows (captures variance)
        i, j = rng.choice(len(subset), size=2, replace=False)
        d = subset[i] - subset[j]
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return self._pick_direction(rng)
        return d / norm

    def shake(self, direction: np.ndarray) -> ShakeStep:
        """Perform one 2048-style slide+merge along a direction (vectorized)."""
        step = ShakeStep(direction=direction.copy())

        projections = self.grid @ direction
        row_norms = np.linalg.norm(self.grid, axis=1)
        sorted_indices = np.argsort(projections)

        alive_mask = row_norms[sorted_indices] > 1e-10
        alive_sorted = sorted_indices[alive_mask]

        if len(alive_sorted) < 2:
            self.history.append(step)
            return step

        # Batch cosine similarity for adjacent alive pairs
        rows_a = self.grid[alive_sorted[:-1]]
        rows_b = self.grid[alive_sorted[1:]]
        norms_a = row_norms[alive_sorted[:-1]]
        norms_b = row_norms[alive_sorted[1:]]
        dots = np.sum(rows_a * rows_b, axis=1)
        cos_sims = dots / (norms_a * norms_b + 1e-30)

        # Greedy non-overlapping merge selection
        candidates = cos_sims > self.merge_threshold
        i = 0
        while i < len(candidates):
            if candidates[i]:
                idx_a = alive_sorted[i]
                idx_b = alive_sorted[i + 1]
                self.grid[idx_a] += self.grid[idx_b]
                self.grid[idx_b] = 0.0
                row_norms[idx_a] = np.linalg.norm(self.grid[idx_a])
                row_norms[idx_b] = 0.0
                step.merges.append(Move(len(self.history), int(idx_b), int(idx_a)))
                i += 2
            else:
                i += 1

        self.history.append(step)
        return step

    def reconstruct(self) -> np.ndarray:
        """Replay move history in reverse to recover original weights exactly."""
        grid = self.grid.copy()

        for step in reversed(self.history):
            for move in reversed(step.merges):
                # Undo: subtract source's contribution from target,
                # but we need the original source value.
                # Since target = old_target + old_source, and source = 0,
                # we need to store the original source value.
                # ... this requires storing values, not just indices.
                pass

        # NOTE: For exact reconstruction we need to store the source vector
        # at merge time. See ReversibleWeightCondenser below.
        raise NotImplementedError("Use ReversibleWeightCondenser for exact reconstruction")


class ReversibleWeightCondenser(WeightCondenser):
    """Extended condenser that stores merge values for exact reconstruction.

    Two modes:
      conserve_energy=False (original): merged = a + b, stores full source vector
      conserve_energy=True:  merged = (a+b) * scale so ||merged||^2 = ||a||^2 + ||b||^2
                             stores source vector + scale factor for exact reversal
    """

    @dataclass
    class MergeWithValue:
        direction_id: int
        source_idx: int
        target_idx: int
        source_value: np.ndarray  # the source row BEFORE merge
        scale_factor: float = 1.0  # energy-conserving scale applied to merged row

    def __init__(self, weights: np.ndarray, merge_threshold: float = 0.1,
                 conserve_energy: bool = False):
        super().__init__(weights, merge_threshold)
        self.conserve_energy = conserve_energy
        self.value_history: List[List['ReversibleWeightCondenser.MergeWithValue']] = []

    def shake(self, direction: np.ndarray) -> ShakeStep:
        """Vectorized 2048-style slide+merge with reversibility.

        Uses numpy batch ops for the cosine similarity computation.
        Merges are still sequential (a merge changes subsequent pairs),
        but the initial candidate detection is vectorized.
        """
        step = ShakeStep(direction=direction.copy())
        value_merges = []

        # Project rows onto direction and sort
        projections = self.grid @ direction  # [N]
        row_norms = np.linalg.norm(self.grid, axis=1)
        sorted_indices = np.argsort(projections)

        # Filter out dead rows from sorted order
        alive_mask = row_norms[sorted_indices] > 1e-10
        alive_sorted = sorted_indices[alive_mask]

        if len(alive_sorted) < 2:
            self.history.append(step)
            self.value_history.append(value_merges)
            return step

        # Vectorized: compute cosine similarity for ALL adjacent alive pairs at once
        rows_a = self.grid[alive_sorted[:-1]]  # [M-1, D]
        rows_b = self.grid[alive_sorted[1:]]   # [M-1, D]
        norms_a = row_norms[alive_sorted[:-1]]
        norms_b = row_norms[alive_sorted[1:]]

        # Batch dot products
        dots = np.sum(rows_a * rows_b, axis=1)  # [M-1]
        cos_sims = dots / (norms_a * norms_b + 1e-30)

        # Find merge candidates (greedy non-overlapping: if i merges, skip i+1)
        candidates = cos_sims > self.merge_threshold
        merge_mask = np.zeros(len(alive_sorted), dtype=bool)

        i = 0
        while i < len(candidates):
            if candidates[i]:
                merge_mask[i] = True  # i is target, i+1 is source
                i += 2  # skip next pair
            else:
                i += 1

        # Execute merges
        for pair_idx in np.where(merge_mask)[0]:
            idx_a = alive_sorted[pair_idx]      # target
            idx_b = alive_sorted[pair_idx + 1]  # source

            norm_a = row_norms[idx_a]
            norm_b = row_norms[idx_b]

            # Store source BEFORE merge
            source_val = self.grid[idx_b].copy()

            # Merge
            self.grid[idx_a] += self.grid[idx_b]
            self.grid[idx_b] = 0.0

            # Energy-conserving rescale
            scale = 1.0
            if self.conserve_energy:
                merged_norm_sq = np.dot(self.grid[idx_a], self.grid[idx_a])
                target_norm_sq = norm_a ** 2 + norm_b ** 2
                if merged_norm_sq > 1e-20:
                    scale = np.sqrt(target_norm_sq / merged_norm_sq)
                    self.grid[idx_a] *= scale

            row_norms[idx_a] = np.linalg.norm(self.grid[idx_a])
            row_norms[idx_b] = 0.0

            move = Move(
                direction_id=len(self.history),
                source_idx=int(idx_b),
                target_idx=int(idx_a),
            )
            step.merges.append(move)
            value_merges.append(self.MergeWithValue(
                direction_id=move.direction_id,
                source_idx=move.source_idx,
                target_idx=move.target_idx,
                source_value=source_val,
                scale_factor=scale,
            ))

        self.history.append(step)
        self.value_history.append(value_merges)
        return step

    def reconstruct(self) -> np.ndarray:
        """Exact reconstruction by replaying merges in reverse."""
        grid = self.grid.copy()

        for merges in reversed(self.value_history):
            for m in reversed(merges):
                # Undo scale
                if m.scale_factor != 1.0:
                    grid[m.target_idx] /= m.scale_factor
                # Undo merge: target had source added to it, source was zeroed
                grid[m.target_idx] -= m.source_value
                grid[m.source_idx] = m.source_value

        return grid

    def storage_cost(self) -> dict:
        """Report how much the move history costs to store."""
        n_merges = sum(len(m) for m in self.value_history)
        # Each merge stores: 2 ints + 1 float (scale) + 1 vector of n_dims floats
        bytes_per_merge = 8 + 8 + self.n_dims * 8  # int64 + float64 scale + float64 vec
        total_bytes = n_merges * bytes_per_merge
        original_bytes = self.n_rows * self.n_dims * 8
        return {
            "n_merges": n_merges,
            "n_shakes": len(self.history),
            "merge_history_bytes": total_bytes,
            "original_weight_bytes": original_bytes,
            "overhead_ratio": total_bytes / original_bytes if original_bytes > 0 else 0,
        }


def measure_density(grid: np.ndarray, eps: float = 1e-10) -> dict:
    """Measure how clumpy the grid has become."""
    row_norms = np.linalg.norm(grid, axis=1)
    alive = np.sum(row_norms > eps)
    dead = len(row_norms) - alive

    alive_norms = row_norms[row_norms > eps]
    return {
        "alive_rows": int(alive),
        "dead_rows": int(dead),
        "dead_fraction": dead / len(row_norms),
        "alive_norm_mean": float(alive_norms.mean()) if len(alive_norms) > 0 else 0,
        "alive_norm_std": float(alive_norms.std()) if len(alive_norms) > 0 else 0,
        "alive_norm_max": float(alive_norms.max()) if len(alive_norms) > 0 else 0,
        "total_energy": float(np.sum(row_norms ** 2)),
    }


def run_synthetic(n_rows=512, n_dims=256, n_shakes=100, merge_threshold=0.1,
                   seed=42, conserve_energy=False):
    """Run condenser on synthetic Gaussian fog and report results."""
    mode = "ENERGY-CONSERVING" if conserve_energy else "ORIGINAL (additive)"
    print(f"=== Synthetic Gaussian Fog: {n_rows} rows x {n_dims} dims [{mode}] ===")
    print(f"    merge_threshold={merge_threshold}, n_shakes={n_shakes}")
    print()

    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((n_rows, n_dims))

    condenser = ReversibleWeightCondenser(weights, merge_threshold=merge_threshold,
                                          conserve_energy=conserve_energy)

    # Initial state
    d0 = measure_density(condenser.grid)
    print(f"Before: {d0['alive_rows']} alive, {d0['dead_rows']} dead "
          f"({d0['dead_fraction']:.1%} empty), energy={d0['total_energy']:.1f}")

    t0 = time.time()
    total_merges = 0
    for i in range(n_shakes):
        direction = condenser._pick_direction_from_weights(rng)
        step = condenser.shake(direction)
        total_merges += len(step.merges)

        if (i + 1) % (n_shakes // 5) == 0 or i == 0:
            d = measure_density(condenser.grid)
            print(f"  Shake {i+1:4d}: {d['alive_rows']:4d} alive, "
                  f"{d['dead_rows']:4d} dead ({d['dead_fraction']:.1%} empty), "
                  f"merges this shake: {len(step.merges)}")

    elapsed = time.time() - t0

    # Final state
    d1 = measure_density(condenser.grid)
    print()
    print(f"After {n_shakes} shakes ({elapsed:.2f}s):")
    print(f"  Alive: {d1['alive_rows']} (was {d0['alive_rows']})")
    print(f"  Dead:  {d1['dead_rows']} ({d1['dead_fraction']:.1%} empty)")
    print(f"  Total merges: {total_merges}")
    print(f"  Alive norm: mean={d1['alive_norm_mean']:.3f}, "
          f"std={d1['alive_norm_std']:.3f}, max={d1['alive_norm_max']:.3f}")
    print(f"  Energy preserved: {d1['total_energy']:.1f} "
          f"(was {d0['total_energy']:.1f}, "
          f"ratio={d1['total_energy']/d0['total_energy']:.6f})")

    # Storage cost
    cost = condenser.storage_cost()
    print()
    print(f"Storage cost:")
    print(f"  {cost['n_merges']} merges, {cost['merge_history_bytes']/1024:.1f} KB")
    print(f"  Original weights: {cost['original_weight_bytes']/1024:.1f} KB")
    print(f"  Overhead: {cost['overhead_ratio']:.2%}")

    # Verify exact reconstruction
    print()
    print("Verifying exact reconstruction...")
    reconstructed = condenser.reconstruct()
    max_err = np.max(np.abs(reconstructed - weights))
    print(f"  Max reconstruction error: {max_err:.2e}")
    assert max_err < 1e-10, f"Reconstruction failed! Max error: {max_err}"
    print("  PASS — lossless reconstruction confirmed")

    # Show the condensed structure
    print()
    print("Condensed row norm distribution:")
    norms = np.linalg.norm(condenser.grid, axis=1)
    norms_sorted = np.sort(norms)[::-1]
    top10 = norms_sorted[:10]
    print(f"  Top 10 row norms: {', '.join(f'{n:.2f}' for n in top10)}")
    print(f"  Rows with norm > 1.0: {np.sum(norms > 1.0)}")
    print(f"  Rows with norm > 5.0: {np.sum(norms > 5.0)}")
    print(f"  Rows with norm > 10.0: {np.sum(norms > 10.0)}")

    return condenser


def run_gguf(gguf_path: str, tensor_name: Optional[str] = None,
             n_shakes: int = 100, merge_threshold: float = 0.1):
    """Run condenser on a real GGUF model tensor."""
    try:
        import gguf
    except ImportError:
        print("pip install gguf  — needed for GGUF reading")
        return

    print(f"Loading {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)

    # List available tensors if none specified
    if tensor_name is None:
        print("\nAvailable tensors (showing first 20):")
        for i, t in enumerate(reader.tensors):
            if i >= 20:
                print(f"  ... and {len(reader.tensors) - 20} more")
                break
            print(f"  {t.name}: {t.shape} ({t.tensor_type.name})")
        print("\nRe-run with --tensor <name> to condense a specific tensor")
        return

    # Find the tensor
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break

    if tensor is None:
        print(f"Tensor '{tensor_name}' not found")
        return

    print(f"Tensor: {tensor.name}, shape={tensor.shape}, type={tensor.tensor_type.name}")

    # Dequantize to float
    from gguf.quants import dequantize
    print("Dequantizing...")
    data = dequantize(tensor.data, tensor.tensor_type)
    # data comes out as [rows, cols] float32
    if len(tensor.shape) == 2:
        weights = data.reshape(tensor.shape[0], tensor.shape[1]).astype(np.float64)
    elif len(tensor.shape) == 1:
        print("1D tensor — reshaping to [N, 1] (not very useful)")
        weights = data.reshape(-1, 1).astype(np.float64)
    else:
        print(f"Unsupported tensor rank: {len(tensor.shape)}")
        return

    print(f"Weight matrix: {weights.shape[0]} rows x {weights.shape[1]} dims")

    condenser = ReversibleWeightCondenser(weights, merge_threshold=merge_threshold,
                                          conserve_energy=True)
    rng = np.random.default_rng(42)

    d0 = measure_density(condenser.grid)
    print(f"\nBefore: {d0['alive_rows']} alive, energy={d0['total_energy']:.1f}")

    t0 = time.time()
    for i in range(n_shakes):
        direction = condenser._pick_direction_from_weights(rng)
        step = condenser.shake(direction)
        if (i + 1) % (n_shakes // 5) == 0:
            d = measure_density(condenser.grid)
            print(f"  Shake {i+1}: {d['alive_rows']} alive, "
                  f"{d['dead_fraction']:.1%} empty")

    elapsed = time.time() - t0
    d1 = measure_density(condenser.grid)

    print(f"\nAfter {n_shakes} shakes ({elapsed:.2f}s):")
    print(f"  Alive: {d1['alive_rows']} / {weights.shape[0]} "
          f"({d1['dead_fraction']:.1%} empty)")
    print(f"  Energy: {d1['total_energy']:.1f} (was {d0['total_energy']:.1f})")

    # Verify
    reconstructed = condenser.reconstruct()
    max_err = np.max(np.abs(reconstructed - weights))
    print(f"  Reconstruction error: {max_err:.2e}")

    return condenser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weight Condenser — 2048-style gravity on weight tensors")
    parser.add_argument("--gguf", type=str, help="Path to GGUF file")
    parser.add_argument("--tensor", type=str, help="Tensor name to condense (list available if omitted)")
    parser.add_argument("--rows", type=int, default=512, help="Rows for synthetic test")
    parser.add_argument("--dims", type=int, default=256, help="Dims for synthetic test")
    parser.add_argument("--shakes", type=int, default=100, help="Number of shake iterations")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Cosine similarity threshold for merging (0=merge everything, 1=merge only identical)")
    args = parser.parse_args()

    if args.gguf:
        run_gguf(args.gguf, args.tensor, args.shakes, args.threshold)
    else:
        # Run both modes for comparison
        run_synthetic(args.rows, args.dims, args.shakes, args.threshold,
                      conserve_energy=False)
        print("\n" + "=" * 70 + "\n")
        run_synthetic(args.rows, args.dims, args.shakes, args.threshold,
                      conserve_energy=True)
