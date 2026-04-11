# Weight Condensation: 2048-Shuffle Method

## The Problem

Transformer weight matrices are Gaussian fog. Every hidden dimension is roughly
equally important at every layer. SVD confirms this: singular values decay at
only 2.8x over 100 dimensions, hidden states need rank 986/1024 for 99% energy.
There is no natural block structure to exploit for skipping or compression.

## The Idea: Impose Structure via Gravity Simulation

If the fog is isotropic, you can rearrange it into any structure without loss —
no orientation is privileged. The 2048-shuffle method treats weight matrix rows
as points in high-dimensional space and repeatedly "shakes" them like the game
2048:

1. **Pick a direction** (random unit vector in weight space)
2. **Project** all rows onto that direction, sort by projection value
3. **Slide + merge**: adjacent rows with cosine similarity > threshold are
   summed. The source row becomes zero (empty space). An energy-conserving
   rescale keeps `||merged||^2 = ||a||^2 + ||b||^2`.
4. **Record the move**: store the source vector and scale factor
5. **Repeat** from many directions

After many shakes, aligned rows collapse together (dense clumps) and empty
space opens up. The merge history is a lossless decompression key — replay
in reverse to reconstruct the original weights exactly.

### Why It Works

The key insight is that even in Gaussian fog, rows are not perfectly
uncorrelated. There is a covariance structure — some row pairs point in
roughly the same direction. The 2048-shake finds and exploits this:

- Rows that are conceptually aligned (cosine sim > threshold) collapse
  together, doubling their weight
- Rows that aren't aligned just slide without merging
- Because the fog is isotropic, every shake direction is equally valid —
  you're not fighting the structure, you're condensing it like shaking
  sand in a box

### Energy Conservation

Naive addition inflates energy: `||a+b||^2 = ||a||^2 + ||b||^2 + 2(a.b)`.
Since we only merge when cosine similarity > 0, the cross term is always
positive. The fix: after merging, rescale by `sqrt((||a||^2 + ||b||^2) / ||a+b||^2)`.
This preserves total matrix energy exactly (measured ratio: 1.000000).

## Measured Results

Tested on Qwen3.5-9B Q4_K_M, `blk.15.ffn_gate.weight` (4096 x 12288), 200
shakes, energy-conserving mode:

```
Threshold   Alive rows   Empty     Merges
0.005            1       100.0%     4095
0.010            2       100.0%     4094
0.020            5        99.9%     4091
0.030           89        97.8%     4007     <-- sweet spot
0.050         2606        36.4%     1490
0.100         3946         3.7%      150
0.200         4093         0.1%        3

Reconstruction: lossless (max error ~1e-15 at all thresholds)
Energy: perfectly conserved (ratio 1.000006 at all thresholds)
```

At threshold 0.03, the matrix condenses from 4096 rows to 89 alive rows.
The steep cliff between 0.02 (5 alive) and 0.05 (2606 alive) reveals where
the real correlation structure lives — most row pairs have cosine similarity
in the 0.02-0.05 range.

## What Condensation Is and Isn't

**It IS:** A lossless reformat. The merge history encodes the exact same
information as the original matrix, just partitioned into dense clumps +
undo instructions.

**It ISN'T:** Compression. `n_alive + n_merges = n_rows` always. Each merge
stores one source vector, so total storage equals original. You can't skip
the undo instructions without losing information.

**It ISN'T:** A valid approximation. Using only the alive rows for inference
gives 147% RMSE at the 89-row condensation. The alive rows contain sums of
many original rows — they produce completely different outputs.

## The Discovery: Per-Token Row Sparsity

The key finding came from analyzing the dot products between hidden states
and source vectors. For any given token, most source vectors contribute
near-zero to the output:

```
Merges needed for 90% recovery energy:  5.1% (205 / 4007)
Merges needed for 95% recovery energy: 10.2% (406 / 4007)
Merges needed for 99% recovery energy: 31.0% (1240 / 4007)
```

This means `|dot(h, W_row)|` is a near-perfect per-token row importance
selector. Condensation-informed top-K selection matches the oracle:

```
Fraction   Oracle (best possible)   Condensation-informed   Random
5%         RMSE 84.5%               RMSE 83.6%              97.5%
10%        RMSE 74.4%               RMSE 73.6%              94.9%
20%        RMSE 58.7%               RMSE 58.0%              89.5%
50%        RMSE 26.4%               RMSE 26.1%              70.8%
```

## Connection to Delta Correction

The condensation produces a structure identical to delta correction:

```
Traditional delta:    low-qual base + noisy residual  = approximate
Condensation delta:   condensed core + exact undo ops = lossless
```

The merge history IS a delta, and it's lossless by construction because it's
a reversible transformation of one matrix, not a difference between two
approximations. The source vectors were selected because they're similar to
their merge targets, so they're structured (not random) — but measured
experiments show they don't compress much better than the original weights
(cosine similarity at merge is only 0.03-0.09, so the residuals are nearly
full-size).

## Connection to PIM (Processing In Memory)

This is where condensation intersects with practical inference.

### What Is a Hidden State?

A hidden state is the intermediate representation of a token as it flows
through the transformer. At every layer, each token is represented as a
single vector of D floating-point numbers (D = "hidden dimension", e.g.,
7168 for GLM-4 253B).

This vector encodes everything the model currently "knows" about that
token in context — its meaning, its relationships to other tokens, and
what the model predicts should come next. It starts as a token embedding
(lookup table), then gets refined by each layer: attention mixes
information from other tokens, then the FFN/MoE layer transforms the
representation through nonlinear projections.

The key operation at each MoE expert is a matrix-vector multiply:

```
output = W_expert @ hidden_state
         [4096 x 12288]  [12288 x 1]  =  [4096 x 1]
         ~100 MB          ~48 KB           ~16 KB
```

The weight matrix is enormous (millions of parameters per expert). The
hidden state is tiny (one vector per token). The output is also tiny.

### Why Move the Hidden State Instead of the Weights?

Traditional approach (ring buffer):
```
RAM ──[100 MB expert weights]──► PCIe (14 GB/s) ──► VRAM ──► GPU matmul
                                 7 ms transfer
```

PIM approach:
```
VRAM ──[48 KB hidden state]──► PCIe ──► CPU RAM ──► CPU vec_dot ──► [16 KB result] ──► VRAM
        0.003 ms                         ~2 ms                       0.001 ms
```

The asymmetry is extreme: the weights are 3000x larger than the
activations they operate on. Moving the small thing (hidden state) to
where the big thing (weights) already lives is obviously cheaper than
moving the big thing to where the small thing lives.

This is the PIM (Processing In Memory) principle: instead of moving data
to computation, move computation to data. The "memory" here is system RAM
where expert weights reside. The "processing" is the CPU doing vec_dot.

The GPU is not idle during this — it continues processing other layers
(attention, norms) or ring-hit experts that are already in VRAM. The CPU
handles only the ring-miss experts, which are the ones that would have
stalled on PCIe.

### The Numbers

For GLM-4 253B (hidden_dim=7168, expert output_dim=4096, top-8 routing):

```
                     Bytes moved    Time
Traditional miss:    ~100 MB        ~7 ms (PCIe-bound)
PIM miss:            ~64 KB         ~2 ms (CPU compute-bound)
                     -------        ------
Reduction:           1500x          3.5x
```

The PCIe bus goes from saturated to effectively unused for expert loading.

### How Condensation Informed This

The condensation experiments proved that per-token, only ~10% of weight rows
contribute 95% of the output energy. This means the CPU PIM path can be
made even cheaper: compute only the rows that matter for this specific token.

### Current PIM Implementation

The PIM full expert path (`--pim-experts`) is implemented in
`ggml/src/ggml-cuda.cu` as a ring cache miss handler:

- **Ring hit**: GPU does the matmul (already in VRAM, fast)
- **Ring miss**: Instead of uploading expert to VRAM (7ms PCIe), send
  hidden state to CPU, CPU does vec_dot from RAM (~2ms), send result back

This eliminates the PCIe bottleneck: 3000x reduction in bytes transferred.
The total FLOPs are the same (CPU does what GPU would have done), but the
data stays where it already lives.

### Future: Adaptive Row Selection on CPU

The condensation measurements suggest a further optimization: on the CPU
PIM path, don't compute all 4096 output rows. Use a cheap proxy
(e.g., low-rank projection of the hidden state) to identify which ~10% of
rows will dominate the output, compute only those, and accept a small
quality loss. This would reduce the CPU PIM cost from ~2ms to ~0.2ms per
expert.

This is effectively **sub-expert routing**: the model already routes at the
expert level (8/160 for GLM-4). Adaptive row selection adds a second
routing level within each expert — route at the row level to skip
unimportant output dimensions.

## Usage Guide

### Step 1: Generate the condensation index (offline, one-time)

The condensation analysis runs ahead of time on your model. It reads each
MoE expert tensor, runs the 2048-shake to identify which rows are most
important, and writes a small `.cidx` index file.

```bash
cd ik_llama.cpp/experiments

# List MoE expert tensors in your model (no computation, instant)
python condense_experts.py /path/to/model.gguf --list

# Run condensation (takes a few minutes depending on model size)
python condense_experts.py /path/to/model.gguf --output /path/to/model.cidx

# Optional: tune the threshold (lower = fewer alive rows = more compression)
python condense_experts.py /path/to/model.gguf -o model.cidx --threshold 0.03 --shakes 200

# Verify the output
python condense_experts.py /path/to/model.gguf --verify /path/to/model.cidx
```

The `.cidx` file is small (~100KB for a 160-expert model) and only needs to
be generated once per model. It contains the indices of the "important" rows
per expert — which rows to keep in the VRAM ring buffer for fast GPU access.

The threshold controls how aggressive the condensation is:
- `0.03` (default): ~89 alive rows out of 4096 → 46x ring capacity increase
- `0.05`: ~2600 alive rows → 1.6x ring capacity increase
- `0.02`: ~5 alive rows → 800x capacity but CPU does almost all the work

### Step 2: Run inference with condensed experts

```bash
./build/bin/llama-server \
    -m model-blurry.gguf \
    --sharp model-sharp.gguf \
    --condense-experts 89 \
    --condense-index /path/to/model.cidx \
    -ngl 99
```

`--condense-experts 89` enables the two-tier mode and auto-enables
`--ring-experts` and `--pim-experts`. The number should match the alive
row count from your condensation (printed by `condense_experts.py`).

`--condense-index` points to the `.cidx` file so the engine knows which
rows belong to GPU vs CPU.

### What happens at runtime

For each MoE layer, for each active expert:

1. **Ring lookup**: Check if this expert's 89 important rows are in the
   VRAM ring buffer (~2MB per expert instead of ~100MB)

2. **Ring hit** (common — 46x more experts fit):
   - GPU: MMVQ on 89 rows → scatter to correct output positions (~0.02ms)
   - CPU: vec_dot on 4007 remaining rows from RAM in parallel (~2ms)
   - Result: complete, bit-exact output

3. **Ring miss** (rare):
   - Upload 89 rows to ring (~0.15ms instead of 7ms for full expert)
   - Then handle as ring hit above

The GPU never waits for the CPU. It gets its 89-row partial result
instantly and moves on to the next expert or layer. The CPU fills in the
rest asynchronously.

### Standalone PIM (without condensation)

If you don't want to run the offline step, `--pim-experts` works on its
own without a `.cidx` file. It handles ring misses by doing the full
matmul on CPU from RAM:

```bash
./build/bin/llama-server \
    -m model-blurry.gguf \
    --sharp model-sharp.gguf \
    --ring-experts 4096 \
    --pim-experts \
    -ngl 99
```

This is simpler (no offline step) but doesn't get the 46x ring capacity
boost. Ring misses are 3.5x faster than PCIe uploads but the hit rate
stays at 40-60%.

## Files

- `weight_condenser.py` — The 2048-shuffle implementation (synthetic + GGUF)
- `condense_experts.py` — Offline condensation → .cidx index generator
- `pim_cpu_expert.py` — PIM-style inference benchmark (PCIe vs CPU vec_dot)
- `ggml/src/ggml-cuda.cu` — PIM full expert + condensed expert dispatch
- `ggml/src/ggml-cuda/scatter-condensed.cuh` — GPU scatter kernel (89 values → correct positions)
- `ggml/src/ggml-pim-cache.h/.c` — condense_index struct + .cidx loader
