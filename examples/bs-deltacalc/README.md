# bs-deltacalc

Generate delta correction GGUFs for the PIM (Processing-in-Memory) system. Computes `sharp - blurry` weight differences and quantizes them for additive correction at inference time.

## Basic Usage

```bash
bs-deltacalc \
  --blurry  model_Q2_K.gguf \
  --sharp   model_Q8_0.gguf \
  --levels  1 \
  --quant   Q4_0 \
  --out-prefix delta \
  --threads 20
```

Produces `delta_1.gguf` containing the quantized difference `Q8_0 - Q2_K` for every shared tensor.

## How It Works

For each tensor shared between the blurry and sharp models:

1. Dequantize both to f32
2. Compute residual: `residual = sharp_f32 - blurry_f32`
3. Quantize residual to the delta type (e.g. Q4_0)
4. Write to output GGUF

At inference, the PIM kernel adds the delta correction in the f32 accumulator:
```
result = vec_dot(blurry_weights, input) + vec_dot(delta_weights, input)
```

This gives near-sharp quality at blurry speed, without needing the sharp model at runtime.

## Multi-Level Fractal Correction

With `--levels N`, each level corrects the accumulated error from previous levels:

- Level 1: `delta_1 = quantize(sharp - blurry)`
- Level 2: `delta_2 = quantize(sharp - (blurry + dequant(delta_1)))`
- Level N: corrects the quantization error of levels 1..N-1

Each level reduces the remaining error. Diminishing returns after 2-3 levels for most quant types.

## Sparse Mode (`--sparse`)

```bash
bs-deltacalc \
  --blurry  model_Q2_K.gguf \
  --sharp   model_Q8_0.gguf \
  --levels  1 \
  --quant   Q4_0 \
  --sparse  0.01 \
  --out-prefix delta \
  --threads 20
```

Produces both:
- `delta_1.gguf` — full delta (all blocks)
- `delta_sparse_1.gguf` — "imploded" delta (significant blocks only)

### What `--sparse` does

The delta between a blurry and sharp model isn't uniformly important. Many quantization blocks have near-zero corrections — the blurry model already captured that information well enough. Only a fraction of blocks carry meaningful corrections.

`--sparse <threshold>` analyzes the RMS magnitude of each quantization block across all experts and removes blocks below the threshold:

1. For each block position, compute `RMS = sqrt(mean(residual^2))` across all elements in the block
2. Take the MAX RMS across all experts for that block position
3. If `max_RMS < threshold * global_max_RMS`, the block is pruned

The sparse GGUF stores:
- **Packed data**: only the significant blocks, contiguously (no gaps)
- **Block index**: uint16 array mapping packed position to original block position
- **Metadata**: `bs.delta.sparse = true`, per-tensor block counts

### Threshold guide

| Threshold | Typical blocks kept | Quality impact | Use case |
|-----------|-------------------|----------------|----------|
| 0.001 | ~80-90% | Negligible | Maximum quality |
| 0.01 | ~20-40% | Very small | Recommended default |
| 0.05 | ~5-15% | Noticeable on benchmarks | Disk-constrained |
| 0.1 | ~2-5% | Significant | Extreme compression |

Run perplexity tests to find the right threshold for your model. The threshold is relative to the maximum block importance, so 0.01 means "keep blocks with at least 1% of the strongest correction."

### Why sparse helps

Without sparse:
- 30GB delta file on SSD
- Runtime reads ~3MB per expert slice, then skips 80% of blocks in the kernel
- 80% of I/O is wasted reading blocks that are always zero

With sparse (0.01 threshold, 20% kept):
- 6GB delta file on SSD
- Runtime reads ~600KB per expert slice, every byte is a meaningful correction
- Zero wasted I/O

The sparse format works with the PIM lazy cache: on first access per expert, `pread()` reads only the packed data from SSD into a heap buffer, then scatters to full-size layout for the IQK kernel.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--blurry <path>` | Blurry (low-quality) GGUF model | required |
| `--sharp <path>` | Sharp (high-quality) GGUF model (supports splits) | required |
| `--out-prefix <prefix>` | Output prefix for GGUF files | required |
| `--levels <N>` | Number of fractal correction levels (1-8) | 3 |
| `--quant <type>` | Delta quantization type | Q2_K |
| `--threads <N>` | Parallel threads for expert processing | 4 |
| `--layers <min-max>` | Layer range to process (e.g. `3-50`) | all |
| `--sparse <threshold>` | Generate sparse delta (0.0 = disabled) | 0.0 |

Supported quant types: Q4_0, Q5_0, Q8_0, Q2_K, Q3_K_S, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, F16, F32
