# SDF-Guided Sparse Delta Correction for MoE LLM Inference

## Overview

This is a custom inference architecture for Mixture-of-Experts (MoE) language models that provides **near-Q8_0 quality from a Q2_K base model** by combining three techniques:

1. **GPU sharp overlay**: Non-expert tensors (attention, norms) are permanently upgraded to sharp quality on GPU at startup using the `--sharp` model. No runtime overhead.
2. **CPU PIM delta kernel**: Expert tensors on CPU get additive delta correction during inference — `vec_dot(blurry) + vec_dot(delta)` produces f32 results at near-sharp quality without requantization.
3. **True ray marching**: The PIM kernel skips quantization blocks where input activation energy ≈ 0, reducing memory bandwidth for both blurry and delta reads.

**Result on GLM-4.5-Air (47B MoE, 128 experts, RTX 3090 24GB + 64GB RAM):**

| Config | Generation | Prompt | Quality | Disk |
|---|---|---|---|---|
| Baseline Q2_K (no correction) | ~7 tok/s | ~33 tok/s | Q2_K | 25GB |
| Q3_K_M (no correction) | 7.13 tok/s | 32.7 tok/s | Q3_K_M | 50GB |
| **Q2_K + Q8_0 sharp + PIM delta** | **7.34 tok/s** | **50.8 tok/s** | **≈Q8_0** | **175GB** |
| Q3_K_M + Q4_K_S sharp JIT | 1.42 tok/s | 5.5 tok/s | Q4_K_S | 250GB |

**Key findings:**
- **Quality upgrade at zero generation speed cost** — Q2_K base runs at native speed, PIM delta adds near-Q8_0 quality for free
- **27% faster prompt processing** (32.7 → 50.8 tok/s) from true ray march block skipping
- **5x faster than JIT** — JIT paid a massive speed penalty for quality; delta provides it without SSD reads
- **Sharp overlay and delta are independent** — any sharp quant works for GPU attention, any delta quant works for CPU experts

---

## Architecture

### GPU Path (non-expert tensors)
```
At startup:
  --sharp Q8_0.gguf → permanently overlays attention/norms on GPU
  q2_K → q8_0 (51 MiB per attention weight)
  599 GPU tensors upgraded, 180 expert tensors skipped

During inference:
  GPU attention runs at Q8_0 quality, zero overhead
```

### CPU Path (expert tensors)
```
During inference (per expert matmul via mul_mat_id):
  Step 1: Dense blurry vec_dot (Q2_K, always in RAM)
  Step 2: Block energy scan (L1 norm per block)
  Step 3: Three-tier partition:
          SKIP:   energy ≈ 0 → zero reads (blurry + delta both skipped)
          ACTIVE: energy > 0 → blurry vec_dot + delta vec_dot
  Step 4: result = Σ(blurry_block + delta_block) for non-skip blocks
          → f32 result at near-Q8_0 quality (never requantized!)
```

### Why CPU PIM Produces Real Quality Improvement

The key insight: **the correction lives in the f32 dot product result, not in the weights.**

```
GPU delta upgrade (BROKEN):
  dequant(Q2_K) + dequant(delta) → requant(Q2_K) → Q2_K quality
  The requantization destroys the correction.

CPU PIM kernel (WORKS):
  vec_dot(Q2_K, input) + vec_dot(delta, input) → f32 result ≈ Q8_0 quality
  The correction is preserved in the floating-point accumulator.
```

This is why GPU expert tensors can't be delta-upgraded via requantization — the blurry quant type can't represent the correction. A CUDA kernel that combines blurry + delta on-the-fly (like the CPU PIM kernel) would solve this but requires delta data in VRAM.

### Sharp and Delta Are Independent

- `--sharp` controls GPU non-expert tensor quality (attention, norms, embeddings)
- `--delta` controls CPU expert tensor quality (PIM kernel at runtime)
- They use different source files and don't interact

You can use Q4_K_S as sharp (smaller VRAM, good attention quality) while keeping a Q2_K→Q8_0 delta (maximum expert correction). Or Q8_0 sharp for maximum attention quality if VRAM allows.

---

## Generating Deltas

```bash
./bs-deltacalc \
  --blurry model_q2k.gguf \
  --sharp  model_q8_0-00001-of-00003.gguf \
  --levels 1 --quant Q4_0 \
  --out-prefix delta \
  --threads 20
```

**Delta type selection:**
- The delta captures `sharp_f32 - blurry_f32`, quantized to the chosen type
- Q4_0 (~4.5 bits): captures ~94% of Q2_K→Q8_0 gap. 32-element blocks (works with all tensor dimensions).
- Q2_K (~2.6 bits): captures ~72%. 256-element blocks (fails on ne[0]=1408 tensors).
- The delta type's `vec_dot_type` should match the blurry's for maximum kernel efficiency. If mismatched, the kernel re-quantizes input (adds ~0.1ms overhead per expert).

---

## Running Inference

### Recommended config:

```bash
./llama-server \
  -m model_q2k.gguf \
  --sharp model_q8_0.gguf \
  -np 1 -ngl 99 -fa on -c 16384 -t 20 \
  --n-cpu-moe 45 --gpu-delta-priority \
  --delta delta_1.gguf
```

**What happens:**
- `-ngl 99`: all attention on GPU (required — CPU attention NaNs on this fork)
- `--n-cpu-moe 45`: 45 expert layers on CPU (PIM delta at runtime)
- `--gpu-delta-priority`: skips auto `--cpu-moe` from `--sharp`, enables selective sharp overlay
- `--sharp`: permanently upgrades GPU non-expert tensors (attention, norms) to Q8_0
- `--delta`: wires delta data to CPU expert tensors for PIM kernel

### VRAM budget (24GB RTX 3090):
```
Attention/norms at Q8_0:  ~6.6 GB  (from --sharp overlay)
KV cache (16K context):   ~1.5 GB
Compute buffers:          ~0.8 GB
1 GPU expert layer:       ~2.3 GB  (Q8_0 from overlay)
Total:                    ~11.2 GB
```

### Saving VRAM with smaller sharp:
Using Q4_K_S as `--sharp` instead of Q8_0:
- Attention at Q4_K quality (~3.3 GB instead of 6.6 GB)
- Frees ~3.3 GB for larger context or more GPU expert layers
- CPU expert quality unchanged (PIM delta is independent)

---

## Components

### New Files
| File | Purpose |
|------|---------|
| `ggml/src/ggml-ray-march.h` | True ray march kernel declarations |
| `ggml/src/ggml-ray-march.c` | Block energy, partition, sparse vec_dot (skips both blurry + delta) |
| `examples/bs-deltacalc/bs-deltacalc.cpp` | Delta GGUF generator (streaming, OpenMP parallel) |
| `examples/bs-deltacalc/test-ray-march.cpp` | Standalone kernel validator |
| `src/llama-streaming.h/cpp` | Execution plan + streaming engine |

### Modified Files
| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `ray_march_delta_data` + `ray_march_delta_type` on `ggml_tensor` |
| `ggml/src/ggml.c` | Three-tier + fullrow + input re-quantization in `mul_mat_id` |
| `src/llama-blurry-sharp.h/cpp` | Delta loading, `wire_delta_tensors()`, `apply_delta_gpu_experts()` with audit |
| `include/llama.h` | Public API for delta, streaming, GPU expert upgrade |
| `common/common.h/cpp` | `--delta`, `--gpu-delta-priority`, `--n-gpu-delta-layers` CLI |
| `examples/server/server-context.cpp` | Delta wiring, selective sharp overlay, GPU audit |

---

## Key Lessons

1. **CPU PIM = real quality improvement.** The correction lives in f32 dot product results, never requantized.

2. **GPU delta upgrade via requantization = useless.** Requanting `dequant(blurry) + dequant(delta)` back to blurry type destroys the correction. Need a CUDA kernel for real GPU delta correction.

3. **`--sharp` auto-forces all experts to CPU.** Use `--gpu-delta-priority` to skip this behavior.

4. **CPU-only attention NaNs** on this fork for GLM4 MoE. Always use `-ngl 99`.

5. **Block size alignment matters.** K-quants (block_size=256) fail on ne[0]=1408 tensors. Use Q4_0 (block_size=32) for universal coverage.

6. **vec_dot_type mismatch handled via re-quantization.** The kernel detects mismatches and re-quantizes input automatically.

7. **True ray marching skips BOTH blurry and delta blocks** where activation energy ≈ 0. Per-block vec_dot sums are exact (verified: diff=0.000000). Main benefit during prompt eval (many tokens, sparse activations).

8. **Sharp and delta are independent.** Any sharp quant for GPU attention, any delta for CPU experts. Mix and match based on VRAM budget.

---

## Reproducing on Clean llama.cpp Fork

Minimal implementation order:

1. **`ggml_tensor` extension** — add `ray_march_delta_data` + `ray_march_delta_type`. Full rebuild required (CUDA).
2. **`bs-deltacalc` tool** — standalone delta GGUF generator.
3. **`ggml-ray-march.{h,c}`** — true ray march kernel (skip blurry + delta for zero-energy blocks).
4. **Delta loading** — open delta GGUF, mmap, index, match to model tensors, set pointers.
5. **`mul_mat_id` integration** — ~50 lines: check delta data, compute energies, partition, call kernel.
6. **CLI flags** — `--delta`, `--gpu-delta-priority`, `--n-cpu-moe`.
7. **Selective sharp overlay** — `apply_non_expert_permanent` for GPU attention quality.

The CUDA kernel for GPU expert delta correction (combining blurry + delta on-the-fly during matmul without requantization) is the main missing piece for maximum performance.
