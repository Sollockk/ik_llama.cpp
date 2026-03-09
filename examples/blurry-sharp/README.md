# Blurry→Sharp Inference Example

This example demonstrates the **Blurry→Sharp overlay system** for ik_llama.cpp: load a low-quality (blurry) GGUF model for speed, then selectively hotswap layer weights from a high-quality (sharp) GGUF model during inference. The result is inference that starts fast (using cheap quantized weights) but produces output quality closer to the sharp model — without needing to load the entire sharp model into memory.

## Quick Start

```bash
# Build (from the repo root)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j blurry-sharp

# Run with both blurry and sharp models
./bin/blurry-sharp \
    -m models/llama-7b-q4_k_m.gguf \
    --sharp models/llama-7b-q8_0.gguf \
    -p "The quick brown fox" \
    -n 64 \
    --compare
```

## How It Works

### The Core Idea

Large language models are typically quantized to reduce memory and compute requirements (e.g., Q4_K_M uses ~4 bits per weight). This sacrifices quality for speed. The Blurry→Sharp system lets you get the best of both worlds:

1. **Load the blurry model** — a heavily quantized GGUF file (e.g., Q4_K_M). This is your fast, cheap base.
2. **Open the sharp model** — a higher-quality GGUF file of the **same architecture** (e.g., Q8_0 or F16). This is only opened for reading; its weights are not fully loaded into GPU memory.
3. **Selectively overlay** — based on routing decisions, read specific tensor data from the sharp GGUF file and write it into the blurry model's weight buffers, replacing low-quality weights with high-quality ones for targeted layers.
4. **Run inference** — the computation graph automatically uses whatever weights are resident. Sharpened layers produce higher-quality activations.
5. **Restore** — optionally restore the original blurry weights after each layer (or keep sharp weights persistent for repeated use).

### Cross-Quantization Support

The overlay engine handles the case where the blurry and sharp models use **different quantization types**:

- **Same type** (e.g., both Q4_K_M): Direct memcpy — fastest path.
- **Different types** (e.g., Q4_K_M blurry, Q8_0 sharp): The engine dequantizes the sharp tensor to F32, then requantizes to the blurry model's type. This preserves the information from the sharp model as much as the target quantization allows.

The key requirement is that both models must have the **same architecture and tensor shapes** (same number of elements per tensor). They can differ in quantization type.

## Command-Line Options

| Option | Description | Default |
|---|---|---|
| `-m, --model FILE` | Path to blurry (low-quality) GGUF model | (required) |
| `-ms, --sharp FILE` | Path to sharp (high-quality) GGUF model | (none) |
| `-p, --prompt TEXT` | Prompt text | `"Hello my name is"` |
| `-n, --n-predict N` | Number of tokens to generate | 64 |
| `-ngl, --n-gpu-layers N` | Number of GPU layers | 0 |
| `--router STRATEGY` | Router strategy: `always`, `never`, `norm` | `always` |
| `--confidence FLOAT` | Min confidence for norm router [0,1] | 0.8 |
| `--max-sharp-layers N` | Max simultaneously sharpened layers (0=unlimited) | 0 |
| `--memory-budget-mb N` | Memory budget for backups in MiB (0=unlimited) | 0 |
| `--no-restore` | Keep sharp weights between layers (don't restore after forward) | (restore enabled) |
| `--verbose-overlay` | Verbose overlay system logging | off |
| `--compare` | Run blurry-only first, then blurry+sharp, and compare results | off |
| `--allow-layers L1,L2,...` | Only sharpen these layers | (all eligible) |
| `--deny-layers L1,L2,...` | Never sharpen these layers | (none) |
| `-t, --threads N` | Number of threads | 4 |
| `-c, --ctx-size N` | Context size | 2048 |

## Router Strategies

The router decides which layers should receive sharp weights:

- **`always`** — Sharpen every eligible layer. Simplest strategy, gives maximum quality uplift at the cost of more I/O and memory.
- **`never`** — Never sharpen (useful as a baseline for A/B testing).
- **`norm`** — Activation L2-norm heuristic. Computes per-token activation norms; tokens with high norms (indicating uncertainty or difficult content) trigger sharpening. The `--confidence` threshold controls sensitivity.

## Architecture

### Files

| File | Purpose |
|---|---|
| `src/llama-blurry-sharp.h` | Header: data structures, configuration, internal C++ API |
| `src/llama-blurry-sharp.cpp` | Implementation: GGUF parsing, tensor index, overlay engine, router, eviction, MoE combination expert |
| `include/llama.h` | Public C API additions (`llama_blurry_sharp_*` functions) |
| `src/llama.cpp` | C API bridge (default params, include wiring) |
| `examples/blurry-sharp/` | This example program |

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Initialization                            │
│                                                              │
│  1. Open sharp GGUF file (mmap or file I/O)                 │
│  2. Parse tensor metadata (names, types, shapes, offsets)    │
│  3. Build index: tensor_name → {offset, type, shape}         │
│  4. Match tensors to base (blurry) model by name             │
│  5. Validate shapes are compatible (same element count)      │
│  6. Determine eligible layers (allow/deny lists)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Per-Layer Overlay                          │
│                                                              │
│  For each tensor in the target layer:                        │
│                                                              │
│  1. Backup: read blurry tensor data from GPU/CPU buffer      │
│  2. Read: load sharp tensor data from mmap/file              │
│  3. Convert (if types differ):                               │
│     sharp_data → dequantize → f32 → requantize → blurry_type│
│  4. Write: ggml_backend_tensor_set() into blurry buffer      │
│                                                              │
│  The next forward pass automatically uses the new weights.   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Restoration                               │
│                                                              │
│  1. Read backup data                                         │
│  2. Write back to tensor buffer via ggml_backend_tensor_set()│
│  3. Free backup memory                                       │
└─────────────────────────────────────────────────────────────┘
```

### Memory Management

The overlay system tracks memory usage through backup buffers. When a layer is sharpened, the original blurry data must be backed up in CPU RAM so it can be restored later. This creates memory pressure proportional to the number of simultaneously sharpened layers.

Controls:
- `--max-sharp-layers N` — Hard cap on the number of concurrently sharpened layers. When the limit is reached, the oldest layer is evicted (restored to blurry) before a new one can be sharpened.
- `--memory-budget-mb N` — Soft cap on total backup memory. When exceeded, layers are evicted according to the eviction policy (LRU, FIFO, or priority-based).

### MoE Combination Expert Mode

For **Mixture-of-Experts** (MoE) models (e.g., Mixtral, Qwen3MoE, DBRX, DeepSeek-V2), the standard blurry→sharp overlay replaces **all** expert weights in a layer — even though only a small fraction of experts are active per token (e.g., 8 out of 128). The MoE Combination Expert mode fixes this inefficiency:

```bash
./bin/blurry-sharp \
    -m models/qwen3moe-q4_k_m.gguf \
    --sharp models/qwen3moe-q8_0.gguf \
    --bs-moe-combination \
    --bs-spec-draft 12 \
    -p "Explain quantum computing" -n 128
```

**How it works:**

1. **Draft** with the blurry (fast) model for `N` tokens
2. **Sharpen selectively**: instead of replacing the full 3D expert tensor (`[n_embd, n_ff, n_expert]`), create a **combination tensor** — a copy of the blurry tensor with only the router-selected expert slices overwritten by sharp data
3. **Verify** the draft batch with the combination model, accepting the longest agreeing prefix
4. **Restore** to blurry and continue

**Why it's faster:**

| Model | Experts | Active | Full-layer I/O | Combination I/O | Speedup |
|-------|---------|--------|----------------|-----------------|---------|
| Mixtral 8x7B | 8 | 2 | 100% | ~25% | ~4x |
| Qwen3MoE-A3B | 128 | 8 | 100% | ~12.5% | ~8x |
| DeepSeek-V2 | 160 | 6 | 100% | ~7.5% | ~13x |

The I/O reduction comes from reading only `n_expert_used / n_expert` of each expert tensor's data from the sharp GGUF file. Non-expert tensors (attention weights, norms, router gate) are always sharpened in full since they are shared and comparatively small.

**Flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--bs-moe-combination` | Enable MoE combination expert mode | off |
| `--bs-moe-top-k N` | Override number of experts to sharpen per layer (0 = use model's `n_expert_used`) | 0 |
| `--bs-spec-draft N` | Draft tokens per speculative batch | 8 |
| `--bs-entropy-threshold F` | Entropy early-stop during drafting | 3.0 |

For dense (non-MoE) models, `--bs-moe-combination` automatically falls back to standard speculative verification.

### Eviction Policies

When the overlay system needs to free memory:
- **LRU** — Evict the least-recently-used (oldest applied) layer first.
- **FIFO** — Evict in insertion order (same as LRU in this implementation).
- **Priority** — Evict the layer with the largest backup (recovers the most memory).

## Example Output

```
=== Blurry→Sharp Inference Example ===
  Blurry model: models/llama-7b-q4_k_m.gguf
  Sharp  model: models/llama-7b-q8_0.gguf
  Prompt:       "The quick brown fox"
  n_predict:    64

--- [1/2] Blurry-only generation ---
Generated: jumps over the lazy dog. This is a well-known pangram...
Tokens: 64, Time: 2.15 s, Speed: 29.77 t/s

Initializing blurry-sharp overlay system...
  sharp model mmap'd (7234567890 bytes)
  sharp model contains 291 tensors
  32 layers eligible for sharpening

--- [2/2] Blurry+Sharp generation ---
Sharpened 32 layers in 1250.00 ms
Generated: jumps over the lazy dog. This classic English pangram...
Tokens: 64, Time: 2.18 s, Speed: 29.36 t/s

=== Comparison Report ===
  Token agreement: 58/64 (90.6%)
  First divergence at token 12
  Overlay overhead: 1.4% speed reduction
```

### MoE Combination Expert Output

```
=== Blurry→Sharp Inference Example ===
  Blurry model: models/qwen3moe-q4_k_m.gguf
  Sharp  model: models/qwen3moe-q8_0.gguf

--- MoE Combination Expert mode ---
  Draft tokens:   12
  Entropy thresh: 3.00
  Top-K override: 0 (0=model default)

  MoE combination expert mode:
    n_expert:      128
    n_expert_used: 8 (per token)
    I/O reduction:  ~16.0x (only sharpen active experts)

  MoE Combination Expert stats:
    Verify batches:     11
    Drafted:            128 tokens
    Accepted:           119 tokens (93.0%)
    Expert efficiency:  1408/14080 expert-layer slots sharpened (10.0%)
    I/O reduction:      ~10.0x vs full-layer sharpening
    Avg draft len:      11.6 tokens
    Avg accepted:       10.8 tokens/batch

MoE Combination Expert results:
  Generated: ...
  Tokens:    128
  Time:      4.52 s
  Speed:     28.32 t/s
```

## Relationship to vLLM Blurry→Sharp

This is the GGUF-native counterpart of the Blurry→Sharp overlay system implemented in `vllm/vllm/blurry_sharp/`. The vLLM version operates on safetensors files and PyTorch tensors with in-place GPU parameter replacement. This version operates on GGUF files using ggml's backend tensor APIs.

Key differences:
- **Format**: GGUF (quantized blocks) vs safetensors (dense float tensors)
- **Cross-quant handling**: This version includes dequant→requant pipeline for mixed quantization types
- **Integration**: Hooks into ggml's `ggml_backend_tensor_set/get` for backend-agnostic buffer access
- **Memory mapping**: Uses ggml's `llama_mmap` for efficient sharp model access

The conceptual architecture (router → overlay engine → backup/restore → eviction) is shared between both implementations.

## Tips

- **Start with `--router always`** to verify the system works, then experiment with `--router norm` for selective sharpening.
- **Use `--compare`** to see the quality difference between blurry-only and blurry+sharp generation.
- **Use `--verbose-overlay`** to see per-tensor overlay details and understand what the system is doing.
- **For production**, the `--no-restore` flag keeps sharp weights persistent across decode steps, avoiding repeated I/O. Use this when you want to sharpen once at the start and keep those weights for the entire generation.
- **Same-quant overlays** (e.g., Q4_K_S → Q4_K_M) are fastest because no dequant/requant is needed.
- **The sharp model doesn't need to be fully loaded** — it's opened read-only via mmap. Only the tensor data that is actually overlaid gets read into memory.