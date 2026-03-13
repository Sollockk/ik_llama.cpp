## Multiple Quantization level Mixing using High Quality Overlays and Layer Skips

An experimental fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) that adds a **blurry-sharp overlay system** for running massive MoE models (e.g. GLM-4 253B, 160 experts) on a Nvidia 3090 24gb vram + 64gb of ram.

Disclaimer: Written hacky and fast with Claude Opus. Needs some reworking before any attempts at a PR. The description was also written by Claude, it may not be 100% accurate.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

The core idea: load a heavily-quantized "blurry" model (TQ1_0, ~1-bit) that fits in memory, then **JIT-overlay** lightly-quantized "sharp" weights (Q4_K_M) onto specific expert slices during inference. On top of this, **layer-skip** eliminates 50-75% of the model's layers entirely — the layers that are kept get sharp overlays so they run at high precision, while skipped layers cost zero compute and have their weights offloaded from VRAM. The result: you run a fraction of the layers, each at much higher quality than the full blurry model could achieve, with the memory footprint of the small model.

## Blurry and Sharp

The naming comes from an image analogy. A neural network's weights encode its "knowledge" — quantizing those weights is like reducing the resolution of an image.

- **Blurry** = the extremely-quantized model (e.g. TQ1_0, ~1-bit per weight).

- **Sharp** = the desired "full-size" of the model (e.g. Q4_K_M, FP16, etc).

The blurry-sharp system combines both: run the blurry model for speed, but swap in sharp weights for the specific expert slices that are actually used on each token. The theory is that you get the memory footprint of the blurry model with the output quality of the sharp model, on the layers and experts that matter for each particular token.

## Why Layer-Skip?

A 253B-parameter MoE model has 93 transformer layers. Even with TQ1_0 quantization, evaluating all 93 layers is slow — especially when MoE expert computation happens on CPU. Layer-skip addresses this by simply not evaluating a fraction of the layers.

This works because transformer layers have significant redundancy, particularly in the middle of the network. Research on layer pruning shows that the first few and last few layers contribute most to output quality (they handle input embedding and output prediction), while many middle layers perform incremental refinement that can be approximated or skipped.

The system uses two skip patterns:

- **Prompt processing** (building the KV cache): Very aggressive, keeps only every 4th middle layer (~75% skip). The KV cache just needs to be "good enough" since the sharp overlay during generation does the real quality work.
- **Token generation** (producing output): Moderate, keeps every 2nd middle layer (~50% skip). Combined with JIT sharp overlay on the non-skipped layers, this recovers most of the quality lost from skipping.

The layers that ARE evaluated get sharp expert overlays via JIT, so they run at high quality. The layers that are skipped cost nothing — zero compute, and their weights are offloaded to CPU to free VRAM.

### Compensating for Skipped Layers

Layer-skip on its own degrades output quality — you're literally throwing away half the model's computation. The idea is that **sharp overlays on the remaining layers help to compensate** for this loss.

Consider: a full blurry model (93 layers, all TQ1_0 ~1-bit) processes every layer but with severely degraded weight precision. Each layer contributes a "noisy" version of what it should compute. Errors accumulate across layers.

With layer-skip + sharp overlay: 49 layers are evaluated, but each one runs with Q4_K_M weights (~4.8-bit) — roughly 3x the precision per layer. The theory is that **fewer layers at high precision can outperform more layers at low precision**, because:

1. **Quality per layer is much higher**: A Q4_K_M expert is far more accurate than a TQ1_0 expert. The MoE routing decision (which experts to activate) is also more accurate with sharp weights, leading to better expert selection cascading through subsequent layers.
2. **Errors don't accumulate**: With blurry-only, each layer adds quantization noise that compounds through the network. With sharp overlays, the layers that run produce clean activations, so there's less error to propagate.
3. **The first and last layers matter most**: These are always kept (never skipped) and always sharpened. The first layers establish the representation, the last layers produce the output distribution. Getting these right matters more than any middle layer.
4. **Middle layers are redundant**: Research shows that middle transformer layers often learn similar features. Skipping every other one and sharpening the rest is a better allocation of resources than running all of them at low quality.
5. **Only the routed experts are sharpened, not the whole layer**: MoE models activate a small subset of experts per token (e.g. 8 out of 160). The JIT system intercepts the router's decision and overlays sharp weights for only those 8 experts. This keeps the I/O cost per layer small (~150 MiB instead of ~2.3 GiB), making it practical to sharpen every non-skipped layer on every token. Without this expert selectivity, sharpening would be too slow — you'd be reading gigabytes of data per layer from disk. With it, the sharp overlay adds only a modest cost per layer, making the skip-fewer-but-sharpen-each strategy viable.

## What is JIT (Just-In-Time) Overlay?

JIT means the sharp (high-quality) weights are not loaded upfront or all at once. Instead, they are swapped in **during the forward pass**, one layer at a time, exactly when that layer is about to compute. This is analogous to JIT compilation in programming languages: rather than paying the full cost upfront, you pay only for what you actually use, at the moment you use it.

In practice, the inference engine registers an **eval callback** that fires between layers during `llama_decode`. When the callback sees that layer N is about to run its MoE (Mixture of Experts) computation, it:

1. **Restores** the previous layer's weights back to blurry (cheap pointer swap)
2. **Overlays** the current layer's sharp expert data onto the blurry tensor (pointer swap to mmap or GPU buffer copy)
3. Lets the layer compute with the sharp weights
4. Moves on to the next layer

This means at any given moment, only **one layer** holds sharp weights in memory. The rest stay blurry. For a 93-layer model, this reduces the sharp memory footprint from "entire model" to "one layer at a time."

The overlay itself is a data pointer swap, not a copy. For CPU tensors, the tensor's `data` pointer is redirected to the sharp file's mmap region. The OS kernel demand-pages only the bytes the MoE kernel actually reads. For GPU tensors, a new device buffer is allocated at the sharp size and data is copied via the PCIe bus.

### Which Layers and Experts Get Sharpened?

The system doesn't decide upfront which layers to sharpen — the **model itself decides at runtime** through its MoE routing mechanism.

Each MoE layer contains a lightweight router network (a small linear layer) that takes the current hidden state and produces a probability distribution over all experts (e.g. 160). The top-K experts (e.g. 8) with the highest router scores are selected for that token.

The JIT system piggybacks on this existing routing by intercepting the `ffn_moe_topk` tensor that the router produces. This tensor contains the indices of the selected experts. The JIT callback reads these indices and overlays sharp weights for **only those specific experts** in that layer. The other 152 experts in the layer are never read from the sharp file.

This means the sharp I/O is driven by the model's own attention to the input. For a given token:
- The router in layer 4 might select experts [3, 17, 42, 55, 89, 102, 140, 158]
- The router in layer 6 might select a completely different set [1, 12, 33, 67, 78, 95, 111, 149]
- Each layer only reads its own selected expert slices from the sharp file

For a 160-expert model with top-8 routing, this means reading **5%** of each expert tensor's data instead of 100%. Each expert slice is ~6 MiB, so 8 experts ≈ 50 MiB per tensor instead of ~1 GiB. With 3 expert tensors per layer (gate, up, down), the total sharp I/O per layer is ~150 MiB instead of ~2.3 GiB — a 15x reduction.

As for which **layers** get sharpened: during token generation, every non-skipped layer gets JIT sharp overlay. The layer-skip pattern determines which layers are evaluated (and therefore sharpened) vs which are skipped entirely. There is no separate "layer importance" ranking — if a layer runs, it gets sharpened.

## Three-Tier Inference Pipeline

### Tier 1: Layer-Skip (Turbo)

Skip a large fraction of transformer layers entirely during the forward pass. Skipped layers are never evaluated, giving a direct compute reduction.

- **Prompt processing**: Aggressive skip (~75% of middle layers). Only builds the KV cache, so quality requirements are low.
- **Token generation**: Moderate skip (~50% of middle layers). First and last N layers are always kept since they contribute most to output quality.
- Skipped layers are automatically offloaded to CPU at model load time, freeing VRAM for sharp data.

```
--bs-layer-skip 3    # keep first/last 3 layers, skip every other middle layer
```

### Tier 2: Blurry Model (TQ1_0)

The base model at extreme quantization (~1-bit per weight). Runs entirely from VRAM for the non-skipped layers. This is what actually executes during inference.

### Tier 3: Sharp Overlay (JIT)

During token generation, the JIT system intercepts each layer's MoE routing decision and overlays Q4_K_M weights onto the selected expert slices before the layer computes.

**How it works:**

1. The MoE router selects which experts to activate (e.g. 8 of 160)
2. The JIT eval callback intercepts the `ffn_moe_topk` tensor for that layer
3. The previous layer's sharp weights are restored to blurry
4. The current layer's sharp expert slices are overlaid:
   - **GPU tensors**: A new device buffer is allocated at the sharp size, sharp data is copied CPU->GPU
   - **CPU tensors** (from `--n-cpu-moe`): Zero-copy mmap pointer swap to the sharp file. The OS demand-pages only the needed expert slices (~150 MiB vs ~2.3 GiB for the full layer)
5. The layer computes with sharp weights
6. After decode, all sharp overlays are restored to blurry

**Expert-selective I/O**: For a 160-expert model with top-8 routing, only 5% of each expert tensor is read (~150 MiB per layer instead of ~2.3 GiB). Prefetch (`MADV_WILLNEED`) targets only the needed expert slices, not the full tensor.

### How Layer-Skip and JIT Sharp Work Together

Layer-skip and JIT sharp overlay are enabled simultaneously during a single `llama_decode` call — they are not separate passes. Here is what happens for each token during generation:

1. **Layer-skip is set** before the decode call. The forward pass skips layers marked in the skip list — their computations never run, their tensors are never accessed.

2. **JIT is started** before the same decode call. An eval callback is registered that will fire during the forward pass.

3. **A single `llama_decode` executes.** As the forward pass walks through the layers:
   - **Skipped layer** (e.g. layer 5): The layer is entirely bypassed. No compute, no JIT callback, no sharp overlay. Its weights sit idle on CPU.
   - **Non-skipped layer** (e.g. layer 6): The layer runs normally. When it reaches the MoE routing step, the JIT callback fires. The callback reads the router's expert selections, swaps in sharp weights for those experts via mmap pointer swap (CPU) or buffer copy (GPU), and the layer computes at sharp precision. Then the callback restores blurry weights before the next layer.

4. **After decode**, JIT is stopped and any remaining sharp overlays are restored to blurry.

The key point: there is **no separate turbo/scout pass followed by a re-decode**. Layer-skip and JIT sharp coexist in a single forward pass. Skipped layers are free, non-skipped layers are sharpened. This gives you the speed benefit of evaluating fewer layers AND the quality benefit of sharp precision on the layers that do run, in one decode call.

## Memory Tiering

The system uses a four-tier memory hierarchy:

| Tier | Storage | Purpose |
|------|---------|---------|
| VRAM | GPU | Active model weights (non-skipped layers), KV cache |
| RAM | CPU | MoE expert tensors (via `--n-cpu-moe`), sharp data cache |
| Swap | SSD | Sharp data that has been accessed once, staged via `--bs-lazy-swap` |
| Disk | SSD | Sharp model GGUF files, demand-paged via mmap |

Skipped layers are offloaded to CPU at load time (via `tensor_buft_overrides`), freeing VRAM for the layers that are actually evaluated.

## MoE Top-K Override

Reduce the number of experts activated per token to cut MoE compute:

```
--bs-moe-top-k-override 4    # use 4 experts instead of model default (8)
```

Halves MoE compute. The sharp overlay on important layers compensates for the quality reduction.

## Example Usage

```bash
./build/bin/llama-server \
  -m GLM-4.7-UD-TQ1_0.gguf \
  --sharp Q4_K_M/GLM-4.7-Q4_K_M-00001-of-00005.gguf \
  --bs-moe-combination \
  -np 1 -ngl 99 -fa on -c 4000 -t 16 \
  --n-cpu-moe 57 \
  --bs-layer-skip 3 \
  --bs-gpu-budget-mb 2048 \
  --bs-lazy-swap \
  --bs-moe-top-k-override 4
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--sharp <path>` | Path to the sharp (lightly-quantized) model GGUF |
| `--bs-moe-combination` | Enable MoE blurry-sharp combination mode |
| `--bs-layer-skip N` | Keep first/last N layers, skip every other middle layer |
| `--n-cpu-moe N` | Put first N layers' MoE experts on CPU (frees VRAM) |
| `--bs-gpu-budget-mb N` | VRAM budget for sharp overlay buffers |
| `--bs-lazy-swap` | Cache sharp data to RAM on first access, stage to swap on restore |
| `--bs-moe-top-k-override N` | Override experts-per-token (0 = model default) |
| `-ctk q8_0 -ctv q8_0` | KV cache quantization (reduces VRAM usage) |

## How It Differs From Standard Inference

| Standard | VLLMWarpstone |
|----------|---------------|
| One model, one quantization | Two models: blurry (speed) + sharp (quality) |
| All layers evaluated | Layer-skip: 50-75% of middle layers skipped |
| All experts loaded | JIT: only routed expert slices loaded per token |
| Static memory layout | Dynamic: VRAM -> RAM -> Swap -> Disk tiering |
| Fixed expert count | Adjustable experts-per-token at runtime |

## License

MIT
