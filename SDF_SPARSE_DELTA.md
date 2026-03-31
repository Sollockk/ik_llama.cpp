# SDF-Guided Sparse Delta Correction for MoE LLM Inference

## Overview

This is a custom inference architecture for Mixture-of-Experts (MoE) language models that eliminates the memory bandwidth bottleneck using **delta correction** — a technique where a compact residual is stored alongside a low-quality base model and applied to upgrade weight quality without needing the full sharp model at inference time.

The system has two modes:
1. **GPU startup upgrade**: Delta correction is applied to GPU-resident tensors at startup (dequant → add → requant → upload). Permanent upgrade, zero per-token overhead.
2. **CPU runtime PIM kernel**: For CPU-resident expert tensors, a sparse delta correction runs inside `mul_mat_id` during inference, guided by input activation energy (the SDF).

**Result on GLM-4.5-Air (47B MoE, 128 experts, RTX 3090 24GB + 64GB RAM):**

| Config | Token speed | Sharp model needed | Disk usage |
|---|---|---|---|
| Original JIT (sharp from SSD) | 705ms/tok | Yes (200GB) | 250GB |
| Delta + no-jit-gen | 626ms/tok | Yes (200GB) | 280GB |
| **Delta + GPU upgrade (no sharp)** | **143ms/tok** | **No** | **80GB** |

**4.9x faster. 68% less disk. No sharp model. Near-sharp quality.**

---

## How It Works

### The Delta GGUF

The delta file stores `quantize(sharp_f32 - blurry_f32)` for every shared tensor. It captures the error between a low-quality model (Q3_K_M) and a high-quality model (Q4_K_S). Generated offline by `bs-deltacalc`.

### At Startup

1. Model loads normally (Q3_K_M, ~50GB). `-ngl 99` puts attention on GPU, `--n-cpu-moe 38` puts expert tensors on CPU.
2. Delta GGUF is mmap'd and indexed.
3. `wire_delta_tensors()` sets `tensor->ray_march_delta_data` pointers on all expert tensors.
4. `apply_delta_gpu_experts()` permanently upgrades GPU tensors:
   - Downloads tensor from GPU to CPU temp buffer
   - Dequants blurry weights to f32
   - Dequants delta to f32, adds element-wise
   - Requants corrected weights back to base type
   - Uploads back to GPU
   - Clears `ray_march_delta_data` (no longer needed at runtime)
5. GPU audit confirms all upgradeable tensors are corrected.

### At Runtime (CPU Expert Layers)

For CPU-resident expert tensors (`mul_mat_id` on CPU backend):

```
Step 1: DENSE BASELINE — full vec_dot over ALL blocks with blurry weights
        (always resident in RAM, branchless, SIMD-optimized)

Step 2: ENERGY SCAN — compute per-block activation energy from f32 input
        energy[b] = Σ|x[b*block_size .. (b+1)*block_size - 1]|

Step 3: PARTITION — three tiers based on energy:
        SKIP:   energy ≈ 0  → zero delta reads, zero compute
        BLURRY: medium      → delta correction applied
        SHARP:  high energy → delta correction + MADV_WILLNEED prefetch

Step 4: SPARSE DELTA — additive vec_dot over non-skip blocks
        correction = Σ vec_dot(delta_block, input_block)

Result = baseline + correction
```

When blurry and delta have different `vec_dot_type`, the input is re-quantized for the delta type (one-time cost per input column, cached across all output rows).

---

## Generating Deltas

```bash
./bs-deltacalc \
  --blurry model_q3km-00001-of-00002.gguf \
  --sharp  model_q4ks-00001-of-00002.gguf \
  --levels 1 --quant Q2_K \
  --out-prefix delta \
  --threads 20
```

- Streaming output (crash-safe)
- OpenMP parallelism across experts within each tensor
- mmap for zero-copy input reads
- Supports Q4_0, Q2_K, Q3_K, Q5_K, Q8_0, F16, F32 delta types

**Delta type selection:**
- Delta quant should be precise enough to capture the blurry→sharp gap
- Q2_K (~2.6 bits): good for large gaps (TQ1_0 → Q4_K_M)
- Q4_0 (4.5 bits): good for smaller gaps (Q3_K_M → Q4_K_S)
- `vec_dot_type` matching: if blurry and delta have the same `vec_dot_type`, the three-tier kernel runs without re-quantization overhead. If they differ, input re-quantization adds ~0.1ms per expert.

---

## Running Inference

### Recommended config (no sharp model needed):

```bash
./llama-server \
  -m model_q3km.gguf \
  -np 1 -ngl 99 -fa on -c 16384 -t 20 \
  --n-cpu-moe 38 --n-gpu-delta-layers 10 \
  --delta delta_1.gguf
```

**Flags:**
- `--delta <path>`: Delta correction GGUF (repeatable for multi-level)
- `--n-gpu-delta-layers N`: Enable GPU expert delta upgrade (implies `--gpu-delta-priority`)
- `--n-cpu-moe M`: First M expert layers on CPU (PIM kernel at runtime)
- `-ngl 99`: All attention on GPU (required — CPU-only attention NaNs on this fork)

**What happens:**
- GPU gets attention + norms + ~10 layers of experts → all delta-upgraded at startup
- CPU gets 38 layers of experts → PIM delta kernel at runtime
- No sharp model loaded, no JIT, no SSD reads during inference

### With sharp model (for maximum quality):

```bash
./llama-server \
  -m model_tq10.gguf \
  --sharp model_q4km.gguf \
  --bs-moe-combination -np 1 -ngl 99 -fa on \
  --n-cpu-moe 38 --bs-lazy-swap \
  --bs-moe-top-k 8 --bs-sharp-experts-cpu -1 --bs-sharp-experts-gpu 2 \
  --bs-no-jit-gen --bs-cache-io-split 1 \
  --delta delta_1.gguf
```

Sharp model upgrades non-expert GPU tensors at startup (permanent overlay). Delta provides PIM correction for CPU expert tensors. `--bs-no-jit-gen` disables SSD reads during generation.

---

## Components

### New Files
| File | Purpose |
|------|---------|
| `ggml/src/ggml-ray-march.h` | Three-tier PIM kernel declarations |
| `ggml/src/ggml-ray-march.c` | Block energy, partition, sparse vec_dot |
| `examples/bs-deltacalc/bs-deltacalc.cpp` | Delta GGUF generator |
| `examples/bs-deltacalc/test-ray-march.cpp` | Standalone kernel validator |
| `src/llama-streaming.h` | Execution plan + streaming engine |
| `src/llama-streaming.cpp` | Streaming decode + plan factories |

### Modified Files
| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `ray_march_delta_data` + `ray_march_delta_type` on `ggml_tensor` |
| `ggml/src/ggml.c` | Three-tier + fullrow + input re-quantization in `mul_mat_id` |
| `src/llama-blurry-sharp.h` | `bs_correction_level`, fractal mode fields, `delta_owned_bufs` |
| `src/llama-blurry-sharp.cpp` | Delta loading, `wire_delta_tensors()`, `apply_delta_gpu_experts()` with audit |
| `src/llama-build-context.h/cpp` | `build_glm4_moe_layer()` per-layer builder |
| `src/llama-context.h` | Ray march state, `jit_fractal_depths`, streaming fields |
| `include/llama.h` | Public API for delta, streaming, GPU expert upgrade |
| `common/common.h` | `delta_paths`, `gpu_delta_priority`, `n_gpu_delta_layers` |
| `common/common.cpp` | `--delta`, `--gpu-delta-priority`, `--n-gpu-delta-layers` CLI |
| `examples/server/server-context.h/cpp` | Delta wiring, GPU upgrade, delta-only mode |

---

## Key Lessons

1. **`ggml_tensor` struct changes require full CUDA rebuild.** Adding fields changes struct layout for all backends.

2. **`--sharp` auto-forces all experts to CPU** via `tensor_buft_overrides`. Use `--gpu-delta-priority` to skip this.

3. **CPU-only attention NaNs** on this fork for GLM4 MoE. Always use `-ngl 99` to keep attention on GPU.

4. **IQ types (iq1_s, iq4_nl) don't round-trip with `from_float`.** Delta upgrade works for standard K-quants (Q3_K, Q4_K, Q5_K) and non-K quants (Q4_0, Q8_0).

5. **vec_dot_type mismatch**: When blurry and delta have different `vec_dot_type`, the kernel re-quantizes the input. Small overhead but enables universal compatibility.

6. **Block size alignment**: K-quants have block_size=256 (QK_K). Q4_0/Q8_0 have block_size=32. Delta type must have a block_size that divides `ne[0]` for the three-tier kernel to activate. Use Q4_0 for models with ne[0]=1408.

7. **Delta quality**: The delta's quantization precision limits how much of the gap it captures. Q4_0 captures ~94% for Q3→Q4 corrections. Q2_K captures ~72%. The delta quant should be at least as precise as the gap.

8. **GPU expert upgrade** downloads, corrects, and uploads each tensor individually. Large tensors (~500MB) take several seconds. Total startup overhead: 1-5 minutes depending on how many GPU experts.

---

## Toward True Ray Marching

The current CPU kernel is SDF-guided sparse delta correction — the blurry baseline is always fully evaluated, only the delta reads are sparse. True ray marching would skip blurry blocks where activation energy ≈ 0. We proved per-block vec_dot sums match full-row exactly (`diff=0.000000`). The implementation exists but the bandwidth savings from skipping blurry data (~1 bit/weight) are minimal compared to skipping delta data.

---

## Reproducing on Clean llama.cpp Fork

Minimal implementation order:

1. **`ggml_tensor` extension** — add `ray_march_delta_data` + `ray_march_delta_type`. Full rebuild required.
2. **`bs-deltacalc` tool** — standalone, generates delta GGUFs.
3. **`ggml-ray-march.{h,c}`** — standalone three-tier kernel.
4. **Delta loading** — open delta GGUF, mmap, index, match to model tensors.
5. **GPU delta upgrade** — download/dequant/add/requant/upload for GPU tensors at startup.
6. **`mul_mat_id` integration** — ~50 lines: check delta data, compute energies, partition, call kernel.
7. **CLI flags** — `--delta`, `--n-gpu-delta-layers`, `--n-cpu-moe`.
