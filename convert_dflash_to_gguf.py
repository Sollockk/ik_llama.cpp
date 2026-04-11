#!/usr/bin/env python3
"""Convert a HuggingFace DFlash draft model to GGUF format.

Usage:
    python convert_dflash_to_gguf.py \
        --dflash-model <path-to-dflash-hf-model> \
        --target-gguf <path-to-target.gguf> \
        --output <output.gguf> \
        [--outtype f16|f32]

The target GGUF is needed to copy the embedding and LM head weights,
which are shared between the target and DFlash draft model.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors


def load_model_weights(model_path: Path) -> dict[str, torch.Tensor]:
    """Load model weights from safetensors or pytorch files."""
    st_files = list(model_path.glob("*.safetensors"))
    if st_files:
        weights = {}
        for f in sorted(st_files):
            weights.update(load_safetensors(str(f)))
        return weights

    pt_files = list(model_path.glob("*.bin"))
    if pt_files:
        weights = {}
        for f in sorted(pt_files):
            weights.update(torch.load(str(f), map_location="cpu"))
        return weights

    raise FileNotFoundError(f"No safetensors or .bin files found in {model_path}")


def copy_tokenizer_from_gguf(gguf_path: str, writer) -> None:
    """Copy tokenizer metadata from a GGUF file into a GGUFWriter."""
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(gguf_path)

    def get_string(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        return str(bytes(f.parts[-1]), encoding='utf-8')

    def get_uint32(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        return int(f.parts[-1][0])

    def get_string_array(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        result = []
        for idx in f.data:
            result.append(str(bytes(f.parts[idx]), encoding='utf-8'))
        return result

    def get_int_array(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        return [int(f.parts[idx][0]) for idx in f.data]

    model = get_string("tokenizer.ggml.model")
    if model:
        writer.add_tokenizer_model(model)
        print(f"  tokenizer model: {model}")

    pre = get_string("tokenizer.ggml.pre")
    if pre:
        writer.add_string("tokenizer.ggml.pre", pre)

    tokens = get_string_array("tokenizer.ggml.tokens")
    if tokens:
        writer.add_token_list(tokens)
        print(f"  tokens: {len(tokens)}")

    token_types = get_int_array("tokenizer.ggml.token_type")
    if token_types:
        writer.add_token_types(token_types)

    merges = get_string_array("tokenizer.ggml.merges")
    if merges:
        writer.add_token_merges(merges)
        print(f"  merges: {len(merges)}")

    eos = get_uint32("tokenizer.ggml.eos_token_id")
    if eos is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos)

    pad = get_uint32("tokenizer.ggml.padding_token_id")
    if pad is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", pad)

    chat_template = get_string("tokenizer.chat_template")
    if chat_template:
        writer.add_chat_template(chat_template)

    print("  tokenizer copied from target GGUF")


def load_tensors_from_gguf(gguf_path: str, tensor_names: list[str]) -> dict[str, np.ndarray]:
    """Load and dequantize specific tensors from a GGUF file."""
    from gguf.gguf_reader import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType

    reader = GGUFReader(gguf_path)
    result = {}
    for tensor in reader.tensors:
        if tensor.name in tensor_names:
            data = tensor.data
            if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
                arr = data.reshape(tensor.shape[::-1])
            else:
                arr = dequantize(data, tensor.tensor_type).reshape(tensor.shape[::-1])
            result[tensor.name] = arr
            print(f"  Loaded {tensor.name} ({tensor.tensor_type.name}) -> {arr.shape}")
    missing = set(tensor_names) - set(result.keys())
    if missing:
        raise ValueError(f"Tensors not found in GGUF: {missing}")
    return result


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Replicate the DFlash target layer ID selection algorithm."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def write_gguf(args):
    # We use gguf-py if available, otherwise write raw GGUF
    try:
        sys.path.insert(0, str(Path(args.dflash_model).parent.parent / "gguf-py"))
        import gguf
    except ImportError:
        # Try the ik_llama.cpp gguf-py
        sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))
        import gguf

    dflash_path = Path(args.dflash_model)

    # Load configs
    with open(dflash_path / "config.json") as f:
        dflash_config = json.load(f)

    # DFlash model parameters
    n_embd = dflash_config["hidden_size"]
    n_head = dflash_config["num_attention_heads"]
    n_head_kv = dflash_config.get("num_key_value_heads", n_head)
    n_layer = dflash_config["num_hidden_layers"]
    n_ff = dflash_config["intermediate_size"]
    n_vocab = dflash_config["vocab_size"]
    n_ctx = dflash_config.get("max_position_embeddings", 32768)
    rms_eps = dflash_config.get("rms_norm_eps", 1e-6)
    rope_theta = dflash_config.get("rope_theta", 10000.0)

    # DFlash-specific
    dflash_cfg = dflash_config.get("dflash_config", {})
    block_size = dflash_config.get("block_size", 16)
    mask_token_id = dflash_cfg.get("mask_token_id", 0)
    num_target_layers = dflash_config.get("num_target_layers", 32)
    target_layer_ids = dflash_cfg.get("target_layer_ids",
                                       build_target_layer_ids(num_target_layers, n_layer))

    n_embd_head = n_embd // n_head
    n_rot = n_embd_head

    print(f"DFlash model: {n_layer} layers, {n_embd} embd, {n_head} heads, {n_head_kv} kv_heads")
    print(f"  block_size={block_size}, mask_token_id={mask_token_id}")
    print(f"  target_layer_ids={target_layer_ids} (from {num_target_layers} target layers)")
    print(f"  n_ff={n_ff}, n_vocab={n_vocab}")

    # Load weights
    print("Loading DFlash weights...")
    dflash_weights = load_model_weights(dflash_path)
    print(f"Loading embedding + lm_head from target GGUF: {args.target_gguf}")
    target_tensors = load_tensors_from_gguf(args.target_gguf, ["token_embd.weight", "output.weight"])

    # Output type
    ftype = 1  # f16
    dtype = np.float16
    if args.outtype == "f32":
        ftype = 0
        dtype = np.float32

    # Create GGUF writer
    writer = gguf.GGUFWriter(args.output, "dflash")

    # Write metadata
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_feed_forward_length(n_ff)
    writer.add_rope_dimension_count(n_rot)
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(rms_eps)
    writer.add_file_type(ftype)
    writer.add_vocab_size(n_vocab)

    # DFlash-specific metadata
    writer.add_uint32(f"dflash.dflash.block_size", block_size)
    writer.add_uint32(f"dflash.dflash.mask_token_id", mask_token_id)
    writer.add_uint32(f"dflash.dflash.num_target_layers", num_target_layers)
    writer.add_array(f"dflash.dflash.target_layer_ids",
                     [int(x) for x in target_layer_ids])

    # Copy tokenizer from target GGUF
    print("Copying tokenizer from target GGUF...")
    copy_tokenizer_from_gguf(args.target_gguf, writer)

    # Helper to convert and add tensor
    def add_tensor(name: str, tensor: torch.Tensor):
        data = tensor.to(torch.float32).numpy().astype(dtype)
        writer.add_tensor(name, data)

    # --- Global tensors from target GGUF (shared embedding + lm_head) ---
    def add_numpy_tensor(name: str, arr: np.ndarray):
        writer.add_tensor(name, arr.astype(dtype))

    add_numpy_tensor("token_embd.weight", target_tensors["token_embd.weight"])
    add_numpy_tensor("output.weight", target_tensors["output.weight"])

    # --- DFlash-specific global tensors ---
    # Final norm
    add_tensor("output_norm.weight", dflash_weights["norm.weight"])

    # FC projection: [n_embd * n_target_layers, n_embd]
    add_tensor("dflash_fc.weight", dflash_weights["fc.weight"])

    # Hidden norm
    add_tensor("dflash_hidden_norm.weight", dflash_weights["hidden_norm.weight"])

    # --- Per-layer tensors ---
    for il in range(n_layer):
        prefix = f"layers.{il}"

        # Attention
        add_tensor(f"blk.{il}.attn_q.weight", dflash_weights[f"{prefix}.self_attn.q_proj.weight"])
        add_tensor(f"blk.{il}.attn_k.weight", dflash_weights[f"{prefix}.self_attn.k_proj.weight"])
        add_tensor(f"blk.{il}.attn_v.weight", dflash_weights[f"{prefix}.self_attn.v_proj.weight"])
        add_tensor(f"blk.{il}.attn_output.weight", dflash_weights[f"{prefix}.self_attn.o_proj.weight"])
        add_tensor(f"blk.{il}.attn_q_norm.weight", dflash_weights[f"{prefix}.self_attn.q_norm.weight"])
        add_tensor(f"blk.{il}.attn_k_norm.weight", dflash_weights[f"{prefix}.self_attn.k_norm.weight"])

        # Norms
        add_tensor(f"blk.{il}.attn_norm.weight", dflash_weights[f"{prefix}.input_layernorm.weight"])
        add_tensor(f"blk.{il}.ffn_norm.weight", dflash_weights[f"{prefix}.post_attention_layernorm.weight"])

        # MLP (SwiGLU)
        add_tensor(f"blk.{il}.ffn_gate.weight", dflash_weights[f"{prefix}.mlp.gate_proj.weight"])
        add_tensor(f"blk.{il}.ffn_up.weight", dflash_weights[f"{prefix}.mlp.up_proj.weight"])
        add_tensor(f"blk.{il}.ffn_down.weight", dflash_weights[f"{prefix}.mlp.down_proj.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done! Written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Convert DFlash HuggingFace model to GGUF")
    parser.add_argument("--dflash-model", required=True, help="Path to DFlash HuggingFace model directory")
    parser.add_argument("--target-gguf", required=True, help="Path to target model GGUF (for shared embedding/lm_head)")
    parser.add_argument("--output", "-o", required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", default="f16", choices=["f16", "f32"], help="Output tensor type")
    args = parser.parse_args()
    write_gguf(args)


if __name__ == "__main__":
    main()
