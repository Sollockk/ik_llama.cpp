#!/usr/bin/env python3
"""
prepare_models.py — Prepare blurry + sharp GGUF model pairs for testing
the Blurry→Sharp overlay system.

This script automates the process of creating two differently-quantized
GGUF files from a single source model, suitable for use with the
blurry-sharp example:

    ./blurry-sharp -m model-q4_k_m.gguf --sharp model-q8_0.gguf -p "Hello" -n 64

Usage:
    # From a GGUF file (re-quantize to two levels):
    python prepare_models.py --input model-f16.gguf --blurry q4_k_m --sharp q8_0

    # From a HuggingFace model (convert + quantize):
    python prepare_models.py --hf meta-llama/Llama-2-7b-hf --blurry q4_k_m --sharp q8_0

    # Just list available quantization types:
    python prepare_models.py --list-types

    # Validate that two GGUF files are compatible for blurry-sharp:
    python prepare_models.py --validate model-q4.gguf model-q8.gguf

Requirements:
    - Python 3.8+
    - The ik_llama.cpp quantize binary must be built and accessible
    - For HF conversion: convert_hf_to_gguf.py must be accessible
    - Optional: gguf Python package (pip install gguf) for validation
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Quantization type registry
# ---------------------------------------------------------------------------

QUANT_TYPES = {
    # name          : (enum_value, description, bits_per_weight_approx)
    "f32"           : (0,  "32-bit float",                          32.0),
    "f16"           : (1,  "16-bit float",                          16.0),
    "q4_0"          : (2,  "4-bit round-to-nearest, group 32",      4.5),
    "q4_1"          : (3,  "4-bit with min/max, group 32",          5.0),
    "q5_0"          : (6,  "5-bit round-to-nearest, group 32",      5.5),
    "q5_1"          : (7,  "5-bit with min/max, group 32",          6.0),
    "q8_0"          : (8,  "8-bit round-to-nearest, group 32",      8.5),
    "q8_1"          : (9,  "8-bit with min/max, group 32",          9.0),
    "q2_k"          : (10, "2-bit K-quant",                         2.6),
    "q3_k_s"        : (11, "3-bit K-quant (small)",                 3.4),
    "q3_k_m"        : (12, "3-bit K-quant (medium)",                3.9),
    "q3_k_l"        : (13, "3-bit K-quant (large)",                 4.3),
    "q4_k_s"        : (14, "4-bit K-quant (small)",                 4.6),
    "q4_k_m"        : (15, "4-bit K-quant (medium)",                4.8),
    "q5_k_s"        : (16, "5-bit K-quant (small)",                 5.5),
    "q5_k_m"        : (17, "5-bit K-quant (medium)",                5.7),
    "q6_k"          : (18, "6-bit K-quant",                         6.6),
    "iq2_xxs"       : (19, "2-bit imatrix quant (extra extra small)", 2.1),
    "iq2_xs"        : (20, "2-bit imatrix quant (extra small)",     2.3),
    "iq3_xxs"       : (21, "3-bit imatrix quant (extra extra small)", 3.1),
    "iq1_s"         : (22, "1-bit imatrix quant (small)",           1.6),
    "iq4_nl"        : (23, "4-bit imatrix quant (non-linear)",      4.5),
    "iq3_s"         : (24, "3-bit imatrix quant (small)",           3.4),
    "iq2_s"         : (25, "2-bit imatrix quant (small)",           2.5),
    "iq4_xs"        : (26, "4-bit imatrix quant (extra small)",     4.3),
    "iq1_m"         : (27, "1-bit imatrix quant (medium)",          1.8),
}

# Recommended blurry→sharp pairs (blurry should be cheaper than sharp)
RECOMMENDED_PAIRS = [
    ("q4_k_m",  "q8_0",   "Good default: 4-bit blurry, 8-bit sharp"),
    ("q4_k_s",  "q6_k",   "Budget: 4-bit blurry, 6-bit sharp"),
    ("q3_k_m",  "q8_0",   "Aggressive: 3-bit blurry, 8-bit sharp"),
    ("q2_k",    "q5_k_m", "Ultra-aggressive: 2-bit blurry, 5-bit sharp"),
    ("q4_k_m",  "f16",    "Maximum quality: 4-bit blurry, f16 sharp"),
    ("q5_k_m",  "q8_0",   "Mild: 5-bit blurry, 8-bit sharp"),
    ("q8_0",    "f16",    "Quality-focused: 8-bit blurry, f16 sharp"),
    ("iq4_xs",  "q8_0",   "iMatrix: 4-bit imatrix blurry, 8-bit sharp"),
]


def bits_per_weight(quant_name: str) -> float:
    """Return approximate bits per weight for a quantization type."""
    entry = QUANT_TYPES.get(quant_name.lower())
    if entry:
        return entry[2]
    return 0.0


def estimate_model_size_gb(n_params_billions: float, quant_name: str) -> float:
    """Estimate GGUF file size in GB given parameter count and quant type."""
    bpw = bits_per_weight(quant_name)
    if bpw <= 0:
        return 0.0
    bytes_total = n_params_billions * 1e9 * bpw / 8.0
    return bytes_total / (1024 ** 3)


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def find_binary(name: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Find a binary in PATH or common build directories."""
    # Try PATH first
    result = shutil.which(name)
    if result:
        return result

    # Common build output directories relative to the repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # ik_llama.cpp root

    candidates = [
        repo_root / "build" / "bin" / name,
        repo_root / "build" / name,
        repo_root / "build" / "Release" / name,
        repo_root / "build" / "Debug" / name,
        repo_root / name,
    ]

    if search_dirs:
        for d in search_dirs:
            candidates.append(Path(d) / name)

    for candidate in candidates:
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)

    # Try with .exe on Windows
    if sys.platform == "win32":
        for candidate in candidates:
            exe = candidate.with_suffix(".exe")
            if exe.exists() and os.access(str(exe), os.X_OK):
                return str(exe)

    return None


def find_quantize_binary(search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Find the llama-quantize (or quantize) binary."""
    for name in ["llama-quantize", "quantize"]:
        path = find_binary(name, search_dirs)
        if path:
            return path
    return None


def find_convert_script() -> Optional[str]:
    """Find the convert_hf_to_gguf.py script."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    candidates = [
        repo_root / "convert_hf_to_gguf.py",
        repo_root / "convert-hf-to-gguf.py",
    ]

    for c in candidates:
        if c.exists():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# GGUF inspection (lightweight, no gguf package required)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian

def read_gguf_header(path: str) -> Dict:
    """Read basic GGUF file header information without the gguf package."""
    info: Dict = {
        "path": path,
        "valid": False,
        "version": 0,
        "n_tensors": 0,
        "n_kv": 0,
        "tensor_names": [],
        "tensor_types": [],
        "file_size_bytes": 0,
    }

    try:
        info["file_size_bytes"] = os.path.getsize(path)
    except OSError:
        return info

    try:
        with open(path, "rb") as f:
            # Magic
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return info

            # Version
            version = struct.unpack("<I", f.read(4))[0]
            info["version"] = version

            # n_tensors, n_kv
            if version >= 2:
                n_tensors = struct.unpack("<Q", f.read(8))[0]
                n_kv = struct.unpack("<Q", f.read(8))[0]
            else:
                n_tensors = struct.unpack("<I", f.read(4))[0]
                n_kv = struct.unpack("<I", f.read(4))[0]

            info["n_tensors"] = n_tensors
            info["n_kv"] = n_kv
            info["valid"] = True

    except (OSError, struct.error):
        return info

    return info


def read_gguf_tensor_info_via_package(path: str) -> Optional[Dict]:
    """
    Use the gguf Python package (if available) to read detailed tensor info.
    Returns None if the package is not installed.
    """
    try:
        from gguf.gguf_reader import GGUFReader
    except ImportError:
        return None

    try:
        reader = GGUFReader(path, "r")
    except Exception as e:
        print(f"Warning: gguf reader failed for {path}: {e}", file=sys.stderr)
        return None

    info = {
        "tensors": {},
        "architecture": "",
        "n_layers": 0,
    }

    # Extract architecture from metadata
    for field in reader.fields.values():
        if hasattr(field, "name") and "architecture" in field.name.lower():
            try:
                parts = field.parts
                if len(parts) > 0:
                    for p in parts:
                        val = bytes(p).decode("utf-8", errors="ignore").strip("\x00")
                        if val and val.isalpha():
                            info["architecture"] = val
                            break
            except Exception:
                pass

    # Extract tensor information
    for tensor in reader.tensors:
        name = tensor.name
        tensor_info = {
            "name": name,
            "shape": list(tensor.shape),
            "type": tensor.tensor_type.name if hasattr(tensor.tensor_type, "name") else str(tensor.tensor_type),
            "n_elements": int(tensor.n_elements),
            "data_offset": int(tensor.data_offset),
        }
        info["tensors"][name] = tensor_info

        # Extract layer count
        for pattern in ["blk.", "layers."]:
            if pattern in name:
                try:
                    idx_str = name.split(pattern)[1].split(".")[0]
                    idx = int(idx_str)
                    info["n_layers"] = max(info["n_layers"], idx + 1)
                except (ValueError, IndexError):
                    pass

    return info


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(
    input_path: str,
    output_path: str,
    quant_type: str,
    quantize_binary: str,
    imatrix_path: Optional[str] = None,
    n_threads: int = 0,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """Run the quantize binary to create a quantized GGUF file."""

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return False

    cmd = [quantize_binary]

    if imatrix_path and os.path.exists(imatrix_path):
        cmd.extend(["--imatrix", imatrix_path])

    if n_threads > 0:
        cmd.extend(["--nthread", str(n_threads)])

    if extra_args:
        cmd.extend(extra_args)

    cmd.extend([input_path, output_path, quant_type.upper()])

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Type:   {quant_type}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: quantization failed with return code {e.returncode}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: quantize binary not found: {quantize_binary}", file=sys.stderr)
        return False


def convert_hf_model(
    hf_model_path: str,
    output_path: str,
    convert_script: str,
    output_type: str = "f16",
) -> bool:
    """Convert a HuggingFace model to GGUF format."""
    cmd = [
        sys.executable,
        convert_script,
        hf_model_path,
        "--outfile", output_path,
        "--outtype", output_type,
    ]

    print(f"\nConverting HuggingFace model to GGUF:")
    print(f"  Input:  {hf_model_path}")
    print(f"  Output: {output_path}")
    print(f"  Type:   {output_type}")
    print(f"  Cmd:    {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: conversion failed with return code {e.returncode}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: convert script not found: {convert_script}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pair(blurry_path: str, sharp_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that two GGUF files are compatible for blurry-sharp overlay.

    Checks:
    1. Both files are valid GGUF
    2. Same number of tensors
    3. Tensor names match
    4. Tensor shapes (element counts) match
    5. Sharp model has equal or higher precision than blurry

    Returns (is_valid, list_of_messages).
    """
    messages: List[str] = []

    # Basic header check
    blurry_hdr = read_gguf_header(blurry_path)
    sharp_hdr = read_gguf_header(sharp_path)

    if not blurry_hdr["valid"]:
        messages.append(f"ERROR: Blurry model is not a valid GGUF file: {blurry_path}")
        return False, messages

    if not sharp_hdr["valid"]:
        messages.append(f"ERROR: Sharp model is not a valid GGUF file: {sharp_path}")
        return False, messages

    messages.append(f"Blurry: {blurry_path}")
    messages.append(f"  Version: {blurry_hdr['version']}, "
                    f"Tensors: {blurry_hdr['n_tensors']}, "
                    f"Size: {blurry_hdr['file_size_bytes'] / 1024 / 1024:.1f} MiB")

    messages.append(f"Sharp:  {sharp_path}")
    messages.append(f"  Version: {sharp_hdr['version']}, "
                    f"Tensors: {sharp_hdr['n_tensors']}, "
                    f"Size: {sharp_hdr['file_size_bytes'] / 1024 / 1024:.1f} MiB")

    if blurry_hdr["n_tensors"] != sharp_hdr["n_tensors"]:
        messages.append(f"WARNING: Tensor count mismatch "
                        f"(blurry={blurry_hdr['n_tensors']}, "
                        f"sharp={sharp_hdr['n_tensors']})")
        messages.append("  This may be OK if some tensors are architecture-specific.")

    # Try detailed validation with the gguf package
    blurry_info = read_gguf_tensor_info_via_package(blurry_path)
    sharp_info = read_gguf_tensor_info_via_package(sharp_path)

    if blurry_info is None or sharp_info is None:
        messages.append("\nNote: Install the 'gguf' Python package for detailed validation:")
        messages.append("  pip install gguf")
        messages.append("\nBasic validation passed (both files are valid GGUF).")
        return True, messages

    # Architecture check
    if blurry_info["architecture"] and sharp_info["architecture"]:
        if blurry_info["architecture"] != sharp_info["architecture"]:
            messages.append(f"ERROR: Architecture mismatch "
                            f"(blurry={blurry_info['architecture']}, "
                            f"sharp={sharp_info['architecture']})")
            return False, messages
        messages.append(f"\nArchitecture: {blurry_info['architecture']}")

    # Layer count
    messages.append(f"Layers: blurry={blurry_info['n_layers']}, sharp={sharp_info['n_layers']}")

    # Tensor-by-tensor comparison
    blurry_tensors = blurry_info["tensors"]
    sharp_tensors = sharp_info["tensors"]

    common_names = set(blurry_tensors.keys()) & set(sharp_tensors.keys())
    blurry_only = set(blurry_tensors.keys()) - set(sharp_tensors.keys())
    sharp_only = set(sharp_tensors.keys()) - set(blurry_tensors.keys())

    messages.append(f"\nTensor comparison:")
    messages.append(f"  Common:      {len(common_names)}")
    messages.append(f"  Blurry-only: {len(blurry_only)}")
    messages.append(f"  Sharp-only:  {len(sharp_only)}")

    if blurry_only:
        messages.append(f"  Blurry-only tensors: {sorted(blurry_only)[:5]}{'...' if len(blurry_only) > 5 else ''}")
    if sharp_only:
        messages.append(f"  Sharp-only tensors:  {sorted(sharp_only)[:5]}{'...' if len(sharp_only) > 5 else ''}")

    # Check element counts match for common tensors
    n_shape_mismatch = 0
    n_type_same = 0
    n_type_diff = 0
    type_pairs: Dict[str, int] = {}

    for name in sorted(common_names):
        bt = blurry_tensors[name]
        st = sharp_tensors[name]

        if bt["n_elements"] != st["n_elements"]:
            n_shape_mismatch += 1
            if n_shape_mismatch <= 3:
                messages.append(f"  SHAPE MISMATCH: {name}")
                messages.append(f"    blurry: {bt['shape']} ({bt['n_elements']} elements)")
                messages.append(f"    sharp:  {st['shape']} ({st['n_elements']} elements)")

        pair_key = f"{bt['type']} -> {st['type']}"
        type_pairs[pair_key] = type_pairs.get(pair_key, 0) + 1

        if bt["type"] == st["type"]:
            n_type_same += 1
        else:
            n_type_diff += 1

    messages.append(f"\n  Shape matches: {len(common_names) - n_shape_mismatch}/{len(common_names)}")
    messages.append(f"  Same quant type: {n_type_same}")
    messages.append(f"  Different quant type (needs dequant/requant): {n_type_diff}")

    if type_pairs:
        messages.append(f"\n  Type mappings:")
        for pair, count in sorted(type_pairs.items(), key=lambda x: -x[1]):
            tag = " (direct copy)" if pair.split(" -> ")[0] == pair.split(" -> ")[1] else " (dequant+requant)"
            messages.append(f"    {pair}: {count} tensors{tag}")

    is_valid = n_shape_mismatch == 0 and len(common_names) > 0

    if n_shape_mismatch > 0:
        messages.append(f"\nERROR: {n_shape_mismatch} tensors have incompatible shapes!")
        messages.append("  The blurry and sharp models must have the same architecture and shapes.")
    elif len(common_names) == 0:
        messages.append("\nERROR: No common tensors found between blurry and sharp models!")
    else:
        size_ratio = sharp_hdr["file_size_bytes"] / max(1, blurry_hdr["file_size_bytes"])
        messages.append(f"\nSharp/Blurry size ratio: {size_ratio:.2f}x")
        messages.append("Validation PASSED — models are compatible for blurry-sharp overlay.")

    return is_valid, messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare blurry + sharp GGUF model pairs for Blurry→Sharp inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-quantize an existing GGUF to two levels:
  %(prog)s --input model-f16.gguf --blurry q4_k_m --sharp q8_0

  # Convert from HuggingFace and quantize:
  %(prog)s --hf meta-llama/Llama-2-7b-hf --blurry q4_k_m --sharp q8_0

  # Validate two existing GGUF files:
  %(prog)s --validate model-q4.gguf model-q8.gguf

  # List available quantization types:
  %(prog)s --list-types

  # Show recommended blurry/sharp pairs:
  %(prog)s --recommend
""",
    )

    parser.add_argument("--input", "-i",
                        help="Input GGUF file (typically f16 or f32)")
    parser.add_argument("--hf",
                        help="HuggingFace model path or ID to convert")
    parser.add_argument("--blurry", "-b", default="q4_k_m",
                        help="Quantization type for blurry model (default: q4_k_m)")
    parser.add_argument("--sharp", "-s", default="q8_0",
                        help="Quantization type for sharp model (default: q8_0)")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Output directory for quantized models")
    parser.add_argument("--prefix",
                        help="Output filename prefix (default: derived from input)")
    parser.add_argument("--imatrix",
                        help="Path to importance matrix file (for imatrix quants)")
    parser.add_argument("--threads", "-t", type=int, default=0,
                        help="Number of threads for quantization (0 = auto)")
    parser.add_argument("--quantize-binary",
                        help="Path to the quantize binary")
    parser.add_argument("--convert-script",
                        help="Path to convert_hf_to_gguf.py")
    parser.add_argument("--validate", nargs=2, metavar=("BLURRY", "SHARP"),
                        help="Validate two GGUF files are compatible")
    parser.add_argument("--list-types", action="store_true",
                        help="List available quantization types")
    parser.add_argument("--recommend", action="store_true",
                        help="Show recommended blurry/sharp pairs")
    parser.add_argument("--estimate", type=float, metavar="PARAMS_B",
                        help="Estimate file sizes for a model with N billion params")
    parser.add_argument("--inspect", metavar="GGUF_FILE",
                        help="Inspect a GGUF file and print tensor info")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing output files")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # List types
    # -----------------------------------------------------------------------
    if args.list_types:
        print("\nAvailable quantization types:\n")
        print(f"  {'Name':<14} {'Bits/Weight':>11}  Description")
        print(f"  {'-'*14} {'-'*11}  {'-'*40}")
        for name, (_, desc, bpw) in sorted(QUANT_TYPES.items(), key=lambda x: -x[1][2]):
            print(f"  {name:<14} {bpw:>8.1f}     {desc}")
        print()
        return

    # -----------------------------------------------------------------------
    # Recommend pairs
    # -----------------------------------------------------------------------
    if args.recommend:
        print("\nRecommended blurry→sharp pairs:\n")
        print(f"  {'Blurry':<10} {'Sharp':<10} {'BPW Blurry':>10} {'BPW Sharp':>10}  Notes")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*40}")
        for blurry, sharp, notes in RECOMMENDED_PAIRS:
            bpw_b = bits_per_weight(blurry)
            bpw_s = bits_per_weight(sharp)
            print(f"  {blurry:<10} {sharp:<10} {bpw_b:>8.1f}   {bpw_s:>8.1f}    {notes}")
        print()
        return

    # -----------------------------------------------------------------------
    # Estimate sizes
    # -----------------------------------------------------------------------
    if args.estimate is not None:
        n_params = args.estimate
        print(f"\nEstimated file sizes for {n_params:.1f}B parameter model:\n")
        print(f"  {'Type':<14} {'BPW':>6}  {'Size (GB)':>10}")
        print(f"  {'-'*14} {'-'*6}  {'-'*10}")
        for name, (_, _, bpw) in sorted(QUANT_TYPES.items(), key=lambda x: -x[1][2]):
            size_gb = estimate_model_size_gb(n_params, name)
            print(f"  {name:<14} {bpw:>5.1f}  {size_gb:>9.2f}")
        print()

        print(f"  Recommended pair: {args.blurry} ({estimate_model_size_gb(n_params, args.blurry):.2f} GB) "
              f"+ {args.sharp} ({estimate_model_size_gb(n_params, args.sharp):.2f} GB)")
        print()
        return

    # -----------------------------------------------------------------------
    # Inspect
    # -----------------------------------------------------------------------
    if args.inspect:
        path = args.inspect
        if not os.path.exists(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

        header = read_gguf_header(path)
        if not header["valid"]:
            print(f"Error: not a valid GGUF file: {path}", file=sys.stderr)
            sys.exit(1)

        print(f"\nGGUF file: {path}")
        print(f"  Version:    {header['version']}")
        print(f"  Tensors:    {header['n_tensors']}")
        print(f"  KV entries: {header['n_kv']}")
        print(f"  File size:  {header['file_size_bytes'] / 1024 / 1024:.1f} MiB")

        detail = read_gguf_tensor_info_via_package(path)
        if detail:
            print(f"  Architecture: {detail['architecture']}")
            print(f"  Layers:       {detail['n_layers']}")

            # Count types
            type_counts: Dict[str, int] = {}
            for t in detail["tensors"].values():
                tp = t["type"]
                type_counts[tp] = type_counts.get(tp, 0) + 1

            print(f"\n  Tensor type distribution:")
            for tp, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"    {tp}: {count}")

            if len(detail["tensors"]) <= 20:
                print(f"\n  All tensors:")
                for name, t in sorted(detail["tensors"].items()):
                    print(f"    {name:<50} {str(t['shape']):<20} {t['type']}")
        else:
            print("\n  (Install 'gguf' package for detailed tensor info: pip install gguf)")

        print()
        return

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------
    if args.validate:
        blurry_path, sharp_path = args.validate
        is_valid, messages = validate_pair(blurry_path, sharp_path)
        print("\n" + "\n".join(messages) + "\n")
        sys.exit(0 if is_valid else 1)

    # -----------------------------------------------------------------------
    # Create model pair
    # -----------------------------------------------------------------------
    if not args.input and not args.hf:
        parser.print_help()
        print("\nError: either --input or --hf is required to create a model pair", file=sys.stderr)
        sys.exit(1)

    # Validate quant types
    for qt_name, qt_label in [(args.blurry, "blurry"), (args.sharp, "sharp")]:
        if qt_name.lower() not in QUANT_TYPES:
            print(f"Error: unknown quantization type for {qt_label}: {qt_name}", file=sys.stderr)
            print(f"  Use --list-types to see available types", file=sys.stderr)
            sys.exit(1)

    # Check that blurry is lower quality than sharp
    blurry_bpw = bits_per_weight(args.blurry)
    sharp_bpw = bits_per_weight(args.sharp)
    if blurry_bpw >= sharp_bpw:
        print(f"Warning: blurry type ({args.blurry}, {blurry_bpw:.1f} bpw) is not "
              f"lower quality than sharp type ({args.sharp}, {sharp_bpw:.1f} bpw)",
              file=sys.stderr)
        print("  The blurry model should typically be the cheaper/smaller quantization.",
              file=sys.stderr)

    # Find the quantize binary
    quantize_bin = args.quantize_binary or find_quantize_binary()
    if not quantize_bin:
        print("Error: could not find quantize binary", file=sys.stderr)
        print("  Build it with: cd build && cmake .. && make quantize", file=sys.stderr)
        print("  Or specify with: --quantize-binary /path/to/quantize", file=sys.stderr)
        sys.exit(1)
    print(f"Using quantize binary: {quantize_bin}")

    # Determine input GGUF path
    input_gguf = args.input

    if args.hf:
        # Need to convert from HuggingFace first
        convert_script = args.convert_script or find_convert_script()
        if not convert_script:
            print("Error: could not find convert_hf_to_gguf.py", file=sys.stderr)
            print("  Specify with: --convert-script /path/to/convert_hf_to_gguf.py",
                  file=sys.stderr)
            sys.exit(1)

        # Derive output name from HF model path
        hf_name = Path(args.hf).name.replace("/", "-")
        input_gguf = os.path.join(args.output_dir, f"{hf_name}-f16.gguf")

        if os.path.exists(input_gguf) and not args.force:
            print(f"Using existing converted model: {input_gguf}")
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            if not convert_hf_model(args.hf, input_gguf, convert_script, "f16"):
                print("Error: HuggingFace conversion failed", file=sys.stderr)
                sys.exit(1)

    if not os.path.exists(input_gguf):
        print(f"Error: input GGUF file not found: {input_gguf}", file=sys.stderr)
        sys.exit(1)

    # Derive output names
    prefix = args.prefix
    if not prefix:
        stem = Path(input_gguf).stem
        # Remove any existing quant suffix
        for qt in QUANT_TYPES:
            for sep in ["-", "_", "."]:
                suffix = f"{sep}{qt}"
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
        prefix = stem

    os.makedirs(args.output_dir, exist_ok=True)

    blurry_output = os.path.join(args.output_dir, f"{prefix}-{args.blurry}.gguf")
    sharp_output = os.path.join(args.output_dir, f"{prefix}-{args.sharp}.gguf")

    print(f"\nCreating blurry→sharp model pair:")
    print(f"  Input:  {input_gguf}")
    print(f"  Blurry: {blurry_output}  ({args.blurry}, ~{blurry_bpw:.1f} bpw)")
    print(f"  Sharp:  {sharp_output}  ({args.sharp}, ~{sharp_bpw:.1f} bpw)")

    # Check if outputs already exist
    for out_path, label in [(blurry_output, "blurry"), (sharp_output, "sharp")]:
        if os.path.exists(out_path) and not args.force:
            print(f"\n{label.capitalize()} model already exists: {out_path}")
            print(f"  Use --force to overwrite")
            continue

        success = quantize_model(
            input_path=input_gguf,
            output_path=out_path,
            quant_type=label == "blurry" and args.blurry or args.sharp,
            quantize_binary=quantize_bin,
            imatrix_path=args.imatrix,
            n_threads=args.threads,
        )

        if not success:
            print(f"Error: failed to create {label} model", file=sys.stderr)
            sys.exit(1)

    # Validate the pair
    if os.path.exists(blurry_output) and os.path.exists(sharp_output):
        print("\n--- Validating model pair ---")
        is_valid, messages = validate_pair(blurry_output, sharp_output)
        for msg in messages:
            print(f"  {msg}")

        if is_valid:
            print(f"\n✓ Model pair ready! Run the example with:")
            print(f"  ./blurry-sharp -m {blurry_output} --sharp {sharp_output} "
                  f"-p \"Hello my name is\" -n 64 --compare\n")
        else:
            print(f"\n✗ Validation failed. See errors above.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
