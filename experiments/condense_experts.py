#!/usr/bin/env python3
"""
Offline condensation: reads a GGUF, runs 2048-shake on each MoE expert tensor,
writes a .cidx index file mapping expert → important row indices.

The .cidx file tells the inference engine which rows to keep in the VRAM ring
buffer. At inference time, the GPU computes these rows instantly while the CPU
fills in the rest from RAM.

Usage:
    python condense_experts.py model.gguf --output model.cidx
    python condense_experts.py model.gguf --output model.cidx --threshold 0.03 --shakes 200
    python condense_experts.py model.gguf --list  # just list MoE tensors
"""

import argparse
import struct
import sys
import time
import numpy as np

CIDX_MAGIC = b"CIDX"
CIDX_VERSION = 1


def find_moe_expert_tensors(reader):
    """Find tensors that look like MoE expert weights (3D: [ne0, ne1, n_experts])."""
    moe_tensors = []
    for t in reader.tensors:
        # MoE expert tensors have 3 dimensions: [input_dim, output_dim, n_experts]
        # Common names: blk.N.ffn_gate_exps.weight, blk.N.ffn_up_exps.weight, blk.N.ffn_down_exps.weight
        if len(t.shape) == 3 and t.shape[2] > 1 and "_exps" in t.name:
            moe_tensors.append(t)
    return moe_tensors


def condense_expert(weights_2d, threshold, n_shakes, rng):
    """Fast 2048-shake: only track which rows survive, no merge history.

    Fully vectorized — no Python loops in the hot path except the shake
    iteration and greedy non-overlapping selection (which uses numpy ops).

    Args:
        weights_2d: [n_rows, n_cols] float32 weight matrix (NOT float64)
        threshold: cosine similarity threshold for merging
        n_shakes: number of shake iterations
        rng: numpy random generator

    Returns:
        alive_indices: sorted array of row indices that survived condensation
    """
    grid = weights_2d.copy()
    n_rows, n_dims = grid.shape
    row_norms = np.linalg.norm(grid, axis=1)

    prev_n_alive = n_rows
    stall_count = 0

    for shake_i in range(n_shakes):
        # Pick direction from weight variance
        alive = np.where(row_norms > 1e-10)[0]
        if len(alive) < 2:
            break

        # Early stopping: if no merges for 10 consecutive shakes, converged
        if len(alive) == prev_n_alive:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
            prev_n_alive = len(alive)
        sample = rng.choice(alive, size=min(len(alive), 64), replace=False)
        i, j = rng.choice(len(sample), size=2, replace=False)
        d = grid[sample[i]] - grid[sample[j]]
        dn = np.linalg.norm(d)
        if dn < 1e-12:
            d = rng.standard_normal(n_dims).astype(grid.dtype)
            dn = np.linalg.norm(d)
        d /= dn

        # Project, sort, filter dead
        projections = grid @ d
        sorted_idx = np.argsort(projections)
        alive_mask = row_norms[sorted_idx] > 1e-10
        alive_sorted = sorted_idx[alive_mask]

        if len(alive_sorted) < 2:
            break

        # Vectorized cosine similarity for adjacent pairs
        norms_a = row_norms[alive_sorted[:-1]]
        norms_b = row_norms[alive_sorted[1:]]
        dots = np.einsum('ij,ij->i', grid[alive_sorted[:-1]], grid[alive_sorted[1:]])
        cos_sims = dots / (norms_a * norms_b + 1e-30)

        # Greedy non-overlapping merge selection (vectorized)
        candidates = cos_sims > threshold
        if not np.any(candidates):
            continue

        # Remove overlapping pairs: if i and i+1 both want to merge, keep i, skip i+1
        # Shift candidates right by 1, AND with NOT to block consecutive
        blocked = np.zeros(len(candidates), dtype=bool)
        for ci in range(len(candidates)):
            if candidates[ci] and not blocked[ci]:
                if ci + 1 < len(candidates):
                    blocked[ci + 1] = True
            else:
                candidates[ci] = False

        merge_pairs = np.where(candidates)[0]
        if len(merge_pairs) == 0:
            continue

        targets = alive_sorted[merge_pairs]
        sources = alive_sorted[merge_pairs + 1]

        # Batch merge: target += source, then energy-conserving rescale
        target_norm_sq = row_norms[targets] ** 2 + row_norms[sources] ** 2
        grid[targets] += grid[sources]
        grid[sources] = 0.0
        row_norms[sources] = 0.0

        # Energy-conserving rescale per merged row
        merged_norm_sq = np.sum(grid[targets] ** 2, axis=1)
        scale = np.sqrt(target_norm_sq / (merged_norm_sq + 1e-30))
        grid[targets] *= scale[:, None]
        row_norms[targets] = np.sqrt(target_norm_sq)

    alive = np.where(row_norms > 1e-10)[0].astype(np.uint16)
    return np.sort(alive)


def write_cidx(output_path, tensor_entries):
    """Write a .cidx binary index file.

    tensor_entries: list of (tensor_name, n_experts, n_alive, alive_indices_per_expert)
        where alive_indices_per_expert is a list of n_experts arrays of uint16
    """
    with open(output_path, "wb") as f:
        # Header
        f.write(CIDX_MAGIC)
        f.write(struct.pack("<II", CIDX_VERSION, len(tensor_entries)))

        for name, n_experts, n_alive, per_expert_indices in tensor_entries:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<HH", n_experts, n_alive))

            for eid in range(n_experts):
                indices = per_expert_indices[eid]
                assert len(indices) == n_alive, \
                    f"Expert {eid} has {len(indices)} alive rows, expected {n_alive}"
                f.write(indices.astype(np.uint16).tobytes())

    file_size = output_path if isinstance(output_path, str) else output_path.name
    print(f"Wrote {file_size}")


def read_cidx(path):
    """Read a .cidx file and return its contents for verification."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == CIDX_MAGIC, f"Bad magic: {magic}"
        version, n_tensors = struct.unpack("<II", f.read(8))
        assert version == CIDX_VERSION

        entries = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            n_experts, n_alive = struct.unpack("<HH", f.read(4))

            per_expert = []
            for _ in range(n_experts):
                indices = np.frombuffer(f.read(n_alive * 2), dtype=np.uint16).copy()
                per_expert.append(indices)

            entries.append((name, n_experts, n_alive, per_expert))

    return entries


def _condense_one(eid_weights_seed_thresh_shakes):
    """Worker function for multiprocessing (must be at module level)."""
    eid, w, seed, thresh, shakes = eid_weights_seed_thresh_shakes
    r = np.random.default_rng(seed + eid)
    return eid, condense_expert(w, thresh, shakes, r)


def main():
    parser = argparse.ArgumentParser(
        description="Offline MoE expert condensation → .cidx index file")
    parser.add_argument("gguf", type=str, help="Path to GGUF model file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output .cidx file path (default: <model>.cidx)")
    parser.add_argument("--threshold", "-t", type=float, default=0.03,
                        help="Cosine similarity threshold for merging (default: 0.03)")
    parser.add_argument("--shakes", "-s", type=int, default=200,
                        help="Number of shake iterations (default: 200)")
    parser.add_argument("--list", action="store_true",
                        help="Just list MoE expert tensors, don't condense")
    parser.add_argument("--verify", type=str, default=None,
                        help="Read and print a .cidx file")
    parser.add_argument("--validate", type=str, default=None,
                        help="Validate a .cidx file: check completeness, compare f32 vs f64 on sample experts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.validate:
        try:
            import gguf
            from gguf.quants import dequantize
        except ImportError:
            print("pip install gguf"); sys.exit(1)

        print(f"Validating {args.validate} against {args.gguf}...")
        entries = read_cidx(args.validate)
        reader = gguf.GGUFReader(args.gguf)

        # Build tensor lookup
        tensor_map = {}
        for t in reader.tensors:
            tensor_map[t.name] = t

        all_ok = True
        for name, n_experts, n_alive, per_expert in entries:
            t = tensor_map.get(name)
            if not t:
                print(f"  FAIL: tensor {name} not found in GGUF")
                all_ok = False
                continue

            ne1 = int(t.shape[1])  # rows per expert
            ne2 = int(t.shape[2])  # n_experts

            if ne2 != n_experts:
                print(f"  FAIL: {name} n_experts mismatch: cidx={n_experts} gguf={ne2}")
                all_ok = False
                continue

            # Check 1: completeness — alive + dead = all rows, no gaps/overlaps
            errors_completeness = 0
            for eid in range(n_experts):
                alive_set = set(per_expert[eid].tolist())
                if len(alive_set) != n_alive:
                    print(f"  FAIL: {name} expert {eid}: duplicate alive indices")
                    errors_completeness += 1
                    continue
                if max(alive_set) >= ne1:
                    print(f"  FAIL: {name} expert {eid}: alive index {max(alive_set)} >= n_rows {ne1}")
                    errors_completeness += 1
                    continue
                dead_set = set(range(ne1)) - alive_set
                if len(alive_set) + len(dead_set) != ne1:
                    print(f"  FAIL: {name} expert {eid}: alive+dead != n_rows")
                    errors_completeness += 1

            # Check 2: f32 vs f64 agreement on 3 sample experts
            data = dequantize(t.data, t.tensor_type)
            full = data.reshape(ne2, ne1, int(t.shape[0]))

            n_sample = min(3, n_experts)
            errors_precision = 0
            for eid in range(n_sample):
                w64 = full[eid].astype(np.float64)
                w32 = full[eid].astype(np.float32)

                rng64 = np.random.default_rng(args.seed + eid)
                rng32 = np.random.default_rng(args.seed + eid)

                alive64 = condense_expert(w64, args.threshold, 50, rng64)
                alive32 = condense_expert(w32, args.threshold, 50, rng32)

                # Check: do they produce the same alive set?
                match = np.array_equal(alive64, alive32)
                # Also check: does the cidx match f32?
                cidx_alive = per_expert[eid]
                cidx_match = np.array_equal(np.sort(cidx_alive[:len(alive32)]), np.sort(alive32))

                if not match:
                    # How different?
                    set64 = set(alive64.tolist())
                    set32 = set(alive32.tolist())
                    overlap = set64 & set32
                    print(f"  WARN: {name} expert {eid}: f64 gives {len(alive64)} alive, "
                          f"f32 gives {len(alive32)} alive, overlap={len(overlap)}")
                    errors_precision += 1

                if not cidx_match and len(alive32) == len(cidx_alive):
                    print(f"  WARN: {name} expert {eid}: cidx doesn't match fresh f32 run")

            status = "OK" if errors_completeness == 0 else "FAIL"
            prec = f", f32/f64 differ on {errors_precision}/{n_sample}" if errors_precision > 0 else ""
            print(f"  {status}: {name}: {n_experts} experts × {n_alive}/{ne1} alive{prec}")
            if errors_completeness > 0:
                all_ok = False

        if all_ok:
            print("\nVALIDATION PASSED: all rows accounted for, alive + dead = complete partition")
        else:
            print("\nVALIDATION FAILED: see errors above")
        return

    if args.verify:
        entries = read_cidx(args.verify)
        for name, n_experts, n_alive, per_expert in entries:
            print(f"{name}: {n_experts} experts, {n_alive} alive rows each")
            for eid in range(min(3, n_experts)):
                print(f"  expert {eid}: indices = {per_expert[eid][:10]}...")
        return

    try:
        import gguf
        from gguf.quants import dequantize
    except ImportError:
        print("pip install gguf")
        sys.exit(1)

    print(f"Loading {args.gguf}...")
    reader = gguf.GGUFReader(args.gguf)

    moe_tensors = find_moe_expert_tensors(reader)
    if not moe_tensors:
        print("No MoE expert tensors found (looking for 3D tensors with '_exps' in name)")
        sys.exit(1)

    print(f"Found {len(moe_tensors)} MoE expert tensors:")
    for t in moe_tensors:
        n_experts = t.shape[2]
        n_rows = t.shape[1] if len(t.shape) > 1 else t.shape[0]
        print(f"  {t.name}: shape={list(t.shape)}, type={t.tensor_type.name}, "
              f"{n_experts} experts x {n_rows} rows")

    if args.list:
        return

    output_path = args.output or args.gguf.rsplit(".", 1)[0] + ".cidx"
    print(f"\nCondensation: threshold={args.threshold}, shakes={args.shakes}")
    print(f"Output: {output_path}\n")

    tensor_entries = []
    rng = np.random.default_rng(args.seed)

    for ti, t in enumerate(moe_tensors):
        print(f"[{ti+1}/{len(moe_tensors)}] {t.name}...")
        t0 = time.time()

        # Dequantize full tensor
        data = dequantize(t.data, t.tensor_type)
        # Shape: [ne0, ne1, n_experts] in GGUF layout
        n_experts = t.shape[2]
        ne1 = t.shape[1]  # output rows per expert
        ne0 = t.shape[0]  # input cols per expert

        # Reshape to [n_experts, ne1, ne0]
        full = data.reshape(n_experts, ne1, ne0).astype(np.float32)

        # Run condensation per expert in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        n_workers = min(n_experts, os.cpu_count() or 4)
        per_expert_indices = [None] * n_experts
        alive_counts = []

        tasks = [(eid, full[eid], args.seed, args.threshold, args.shakes)
                 for eid in range(n_experts)]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_condense_one, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                eid, alive = future.result()
                per_expert_indices[eid] = alive
                alive_counts.append(len(alive))
                done += 1
                if done % 16 == 0 or done == n_experts:
                    print(f"    {done}/{n_experts} experts done", end="\r")

        # Use max alive count (pad shorter ones with highest-norm rows)
        max_alive = max(alive_counts)
        min_alive = min(alive_counts)
        mean_alive = np.mean(alive_counts)

        # Pad experts with fewer alive rows to max_alive
        for eid in range(n_experts):
            if len(per_expert_indices[eid]) < max_alive:
                # Add highest-norm rows that aren't already alive
                existing = set(per_expert_indices[eid])
                row_norms = np.linalg.norm(full[eid], axis=1)
                candidates = [(row_norms[j], j) for j in range(ne1) if j not in existing]
                candidates.sort(reverse=True)
                extra = [j for _, j in candidates[:max_alive - len(existing)]]
                per_expert_indices[eid] = np.sort(
                    np.concatenate([per_expert_indices[eid],
                                    np.array(extra, dtype=np.uint16)]))

        elapsed = time.time() - t0
        print(f"  {n_experts} experts: alive={min_alive}-{max_alive} "
              f"(mean={mean_alive:.0f}), padded to {max_alive}, "
              f"{elapsed:.1f}s")

        tensor_entries.append((t.name, n_experts, max_alive, per_expert_indices))

    write_cidx(output_path, tensor_entries)

    # Summary
    print(f"\nDone. Summary:")
    total_alive = 0
    total_rows = 0
    for name, n_experts, n_alive, _ in tensor_entries:
        # Find the tensor to get ne1
        for t in moe_tensors:
            if t.name == name:
                ne1 = t.shape[1]
                break
        total_alive += n_experts * n_alive
        total_rows += n_experts * ne1
        print(f"  {name}: {n_alive}/{ne1} rows alive per expert "
              f"({n_alive/ne1:.1%}), {n_experts} experts")
    print(f"  Total: {total_alive}/{total_rows} rows alive ({total_alive/total_rows:.1%})")
    print(f"  VRAM per expert: {total_alive/total_rows:.1%} of full "
          f"→ {1/(total_alive/total_rows):.0f}x more experts in ring buffer")


if __name__ == "__main__":
    main()
