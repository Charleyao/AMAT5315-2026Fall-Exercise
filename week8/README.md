# Week 8 — Part 1: From Tokens to Vectors (Embeddings)

Status: **Part 1 complete — VERIFY (1) PASS** (Part 1 output preserved).

This part tests whether pretrained static word embeddings encode three word
relations as consistent vector offsets, and compares the result with a
row-shuffled control table.

## Model and data

| Item | Value |
|---|---|
| Model | `minishlab/potion-base-8M` |
| Revision | `bf8b056` (`bf8b056651a2c21b8d2565580b8569da283cab23`) |
| Embedding table | `29,528 x 256` `float32` (~7.56M parameters) |
| Tokenizer | WordPiece (`tokenizer.json`) |
| Local files | `data/model.safetensors`, `data/tokenizer.json` (ignored by Git) |

The embeddings are read directly from safetensors with
`safetensors.numpy.load_file`; PyTorch is **not** used. Downloads are pinned by
size and SHA-256 in `src/download_model.py`.

## Method

1. Tokenize every word and require exactly one non-special WordPiece token.
2. L2-normalize each word vector.
3. Compute the ordered offset `unit(B) - unit(A)`.
4. For pair `i`, compute the cosine between `offset_i` and the mean of the
   other eight offsets (leave-one-out).
5. Repeat with an embedding table whose rows were permuted by
   `numpy.random.default_rng(1730)`.

All 27 pairs (54 word occurrences) map to valid single token IDs in the
embedding table.

## Results

| Relation | Pretrained mean cosine | Shuffled control mean | Expected |
|---|---:|---:|---:|
| Country → capital | **0.4659** | **0.0295** | ≈ 0.47 |
| Male → female | **0.5485** | **−0.0148** | ≈ 0.55 |
| Adjective → comparative | **0.4330** | **0.0088** | ≈ 0.43 |

Acceptance checks:

- All pretrained group means are ≥ 0.30. ✅
- All shuffled-control means are within [−0.15, 0.15]. ✅
- All 27 pairs use valid token IDs. ✅
- Expected means are reproduced within ±0.05. ✅

The pretrained offsets are consistently aligned across each relation, while
the shuffled control is near zero (as expected if the geometry, not the token
IDs, carries the relation). The weakest individual pairs are
`husband-wife` (+0.066), `old-older` (+0.173), and `peru-lima` (+0.304).

## Files

| Path | Purpose |
|---|---|
| `src/download_model.py` | Download + checksum verification for the two model files |
| `src/embeddings.py` | Safetensors loading, token lookup, offsets, leave-one-out cosine |
| `src/relations.py` | The exact 27 ordered pairs used in the analysis |
| `src/plotting.py` | Three-panel Figure 5 reproduction |
| `src/part1.py` | End-to-end pipeline, acceptance checks, reports |
| `tests/` | Unit tests plus pinned-model integration tests |
| `evidence/figure5_word_relations.png` | Three-panel cosine-fan figure |
| `evidence/part1_verify.txt` | Full printed verification report |
| `artifacts/part1_results.json` | Per-pair cosines, means, token IDs, verification |
| `artifacts/part1_settings.json` | Model revision, checksums, seed, packages, settings |

`data/`, `artifacts/`, `.venv/`, and caches are excluded from Git.

## Reproduce

```sh
cd week8
uv sync
uv run python -m src.download_model   # verifies/skips existing files
uv run python -m src.part1            # writes artifacts + figure
uv run python -m pytest -q            # 88 tests total (Parts 1-6)
```

The figure maps each cosine `c` to a unit arrow at angle `arccos(c)`:
pretrained arrows above the horizontal axis, shuffled-control arrows below,
with a black dashed arrow at the group mean.

# Week 8 — Part 2: Count Models: Markov Chains and n-grams

Status: **Part 2 complete — VERIFY (1), (2), (3), and (4) PASS**.
Part 1 remains verified and unchanged.

## Data and tokenizer

| Item | Value |
|---|---|
| Dataset | `roneneldan/TinyStories` @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| File | `TinyStoriesV2-GPT4-valid.txt` |
| Stories | 27,630 total: **24,867 train / 2,763 validation** (first 90% / last 10%, no shuffle) |
| Tokens | 9,667,035 train / 1,078,996 validation |
| Characters | 19,853,989 train / 2,213,915 validation |
| Tokenizer | `karpathy/tinyllamas` @ `0bd21da`, `stories260K/tok512.model` |
| Vocabulary | 512, EOS id 2, UNK id 0 |
| Average chars/token | **2.053586** overall (train 2.053783, validation 2.051829) |
| Round-trip | exact 27,556/27,630; after whitespace normalization **27,630/27,630** |

EOS (`</s>`, id 2) is appended after every story. There are zero UNK tokens.
The tokenizer collapses repeated whitespace on decode, which explains the 74
non-exact but whitespace-normalized round-trips.

## Method

- Count-based next-token models for n = 1..6 with add-k smoothing:
  `P(b|context) = (count(context,b)+k) / (count(context)+k*512)`.
- Tested k ∈ {0.0001, 0.001, 0.01, 0.1, 1}.
- k chosen on the **last 10% of the training token stream**, never on the
  held-out validation split. This selects k=0.01 for n=2..5 and k=0.001 for
  n=6 (n=1 is insensitive to k at four decimals).
- Counts use sparse int64 hashing of base-512 contexts, so no `512^6` table is
  allocated.

## Held-out metrics

| Model | k | Nats/token | Perplexity | Bits/char | Unseen |
|---|---:|---:|---:|---:|---:|
| Uniform | — | 6.2383 | 512.000 | 4.3863 | 0.00% |
| Unigram (n=1) | 0.0001 | 4.8394 | 126.394 | 3.4027 | 0.00% |
| Bigram (n=2) | 0.01 | 3.0382 | 20.868 | 2.1362 | 0.03% |
| Trigram (n=3) | 0.01 | 2.0943 | 8.120 | 1.4726 | 0.63% |
| 4-gram (n=4) | 0.01 | 1.8627 | 6.441 | **1.3097** | 4.15% |
| 5-gram (n=5) | 0.01 | 2.0825 | 8.025 | 1.4643 | 12.22% |
| 6-gram (n=6) | 0.001 | 2.5504 | 12.812 | 1.7932 | 23.50% |

## Verification results

| Task | Result | Key numbers |
|---|---|---|
| VERIFY (1) | **PASS** | Sum over all sequences of lengths 1, 2, 3: all `1.00000000000000000`; max error `0.0` (limit `1e-12`); max conditional row-sum error `6.66e-16` |
| VERIFY (2) | **PASS** | Max nats difference `+0.0004` (limit 0.03); max perplexity relative difference `0.16%` (limit 3%); split and tokenizer checks true |
| VERIFY (3) | **PASS** | 250,000 sampled tokens, 96,776 tested transitions, pooled chi-squared `1196.86`, dof `1209`, `p = 0.5925` (acceptance `p > 0.001`) |
| VERIFY (4) | **PASS** | Minimum bpc at n=4; bpc within `0.0043` of references; unseen fractions within `0.05` percentage points |

VERIFY (2) comparison (actual vs reference): Uniform 6.2383/6.238, 512.0/512.0,
4.3863/4.386; Unigram 4.8394/4.839, 126.394/126.4, 3.4027/3.403; Bigram
3.0382/3.038, 20.868/20.9, 2.1362/2.136; Trigram 2.0943/2.094, 8.120/8.12,
1.4726/1.473.

VERIFY (4) compared with the learning-sheet values:

| n | Bits/char | Reference | Unseen | Reference |
|---:|---:|---:|---:|---:|
| 3 | 1.4726 | 1.473 | 0.633% | 0.6% |
| 4 | **1.3097** | 1.310 | 4.151% | 4.2% |
| 5 | 1.4643 | 1.46 | 12.224% | 12.2% |
| 6 | 1.7932 | 1.79 | 23.499% | 23.5% |

## Part 2 files

| Path | Purpose |
|---|---|
| `src/download_part2_data.py` | Download + checksum verification for TinyStories and tokenizer |
| `src/part2_data.py` | Story split, EOS insertion, tokenizer statistics |
| `src/ngram.py` | Sparse n-gram counts, add-k evaluation, sampling, normalization check |
| `src/chi_square.py` | Pooled Pearson chi-squared with rare-category grouping |
| `src/part2_plotting.py` | Figure 7 and VERIFY (3) diagnostic figures |
| `src/part2.py` | End-to-end Part 2 pipeline and report |
| `tests/test_ngram.py` | Unit tests for n-gram math, sampling, chi-squared |
| `tests/test_part2_model.py` | Pinned-data integration tests for all four VERIFY tasks |
| `evidence/figure7_curse_of_dimensionality.png` | Loss and unseen-fraction trends versus n |
| `evidence/figure_verify3_bigram_chi2.png` | Observed vs expected counts and per-context chi-squared |
| `evidence/part2_verify.txt` | Full printed VERIFY (1)-(4) report |
| `artifacts/part2_results.json` | Dataset/tokenizer stats, metrics, all verification results |
| `artifacts/part2_settings.json` | Revisions, checksums, smoothing grid, seeds, package versions |
| `artifacts/part2_chi2_table.npz` | Observed/expected chi-squared table for the 20 contexts |

## Reproduce Part 2

```sh
cd week8
uv sync
uv run python -m src.download_part2_data
uv run python -m src.part2
uv run python -m pytest -q
```

# Week 8 — Part 3: The MLP: A Memory in the Weights

Status: **Part 3 complete — VERIFY (1), (2), and (3) PASS**.
Part 1 and Part 2 results are unchanged.

## Memory stores

Five stores are queried with stored keys. The reported interference-to-signal
ratio is the amplitude ratio

`R = ||interference|| / ||signal|| = sqrt(energy ratio)`,

so theoretical curves grow like `sqrt(M)` and capacity is the `M` where `R`
crosses 1. Capacities use log-log interpolation of the median `R` from three
independent draws.

Stores:

1. Outer product: `W = sum_i v_i k_i^T`
2. Least squares: minimum-norm pseudo-inverse store `W = V^T (K^T)^+`
3. ReLU similarity: `F(q) = sum_i v_i ReLU(k_i.q)`
4. ReLU² similarity: same with `ReLU(k_i.q)²`
5. ReLU³ similarity: same with `ReLU(k_i.q)³`

The least-squares `R` is `sqrt((1-h)/h)`, where `h` is the leverage of the
queried key; for `M <= d` the store interpolates exactly, so `R = 0`.

## VERIFY (1): d=64 capacities

| Store | Measured capacity | Reference | Relative error |
|---|---:|---:|---:|
| Outer product | **64.67** | 67.4 | 4.06% |
| Least squares | **128.00** | 129 | 0.78% |
| ReLU (p=1) | **128.39** | ≈128 | 0.30% |
| ReLU² (p=2) | **2736.32** | 2871 | 4.69% |
| ReLU³ (p=3) | **39019.76** | 38124 | 2.35% |

- Outer-product capacity in [50, 85] ✅
- Least-squares capacity in [100, 170] ✅
- Least-squares `R < 1e-10` for all `M <= 64` (measured maximum 0) ✅
- ReLU² and ReLU³ capacities within 10% of the references ✅
- Local log-log growth slopes: 0.542, 0.585, 0.499, 0.501, 0.496
  (expected ≈0.5 for square-root growth) ✅

## VERIFY (2): capacity scaling with dimension

| Store | Fitted slope | Expected |
|---|---:|---:|
| Outer product | **1.004** | 0.96 |
| Least squares | **1.000** | 0.99 |
| ReLU (p=1) | **0.982** | 0.95 |
| ReLU² (p=2) | **1.913** | 1.96 |
| ReLU³ (p=3) | **2.810** | 2.79 |

All slope criteria pass: linear/ReLU slopes are within 0.1 of 1, ReLU² is
within 0.15 of 2, and ReLU³ is above 2.5.

## VERIFY (3): previous-token MLP

The Part 2 tokenizer and story split are reused. The model sees only the
previous token and is trained with JAX on CPU.

| Item | Value |
|---|---|
| Vocabulary | 512 |
| Embedding dimension | 64 |
| Hidden layer | 512, tanh |
| Output | 512 logits, softmax cross-entropy |
| Optimizer | AdamW |
| Steps / batch | 4000 / 16384 |
| Peak learning rate | 0.006 |
| Schedule | linear warmup (200 steps) then cosine decay |
| Weight decay | 0.01 |
| Training tokens | 9,667,035 |
| Validation tokens | 1,078,996 |
| **Final validation loss** | **3.041007 nats/token** |
| Part 2 bigram loss | 3.038 nats/token |
| Absolute difference | **0.003007** (limit 0.01) ✅ |
| Reference MLP result | ≈3.041 (difference 0.000007) ✅ |

## Part 3 figures and artifacts

| Path | Purpose |
|---|---|
| `src/memory.py` | Five memory stores, interference/signal measurement, capacity interpolation |
| `src/mlp.py` | JAX MLP, AdamW schedule, evaluation |
| `src/part3_plotting.py` | Figure 9, Figure 10, and MLP training plots |
| `src/part3.py` | End-to-end Part 3 pipeline and report |
| `tests/test_memory.py` | Memory-store and capacity unit tests |
| `tests/test_mlp.py` | MLP shape, loss, and training smoke tests |
| `tests/test_part3_artifacts.py` | Saved-output acceptance checks |
| `evidence/figure9_memory_capacity_d64.png` | Figure 9 reproduction |
| `evidence/figure10_capacity_vs_dimension.png` | Figure 10 reproduction |
| `evidence/figure_verify1_capacity_bars.png` | Measured vs reference d=64 capacities |
| `evidence/figure_mlp_training.png` | MLP training and held-out loss |
| `evidence/part3_verify.txt` | Full VERIFY (1)–(3) report |
| `artifacts/part3_results.json` | Memory capacities, slopes, MLP loss, checks |
| `artifacts/part3_memory_curves.npz` | Full measured/theoretical R curves |
| `artifacts/part3_memory_median_R.csv` | Median R for every d, M, and store |
| `artifacts/part3_mlp_results.json` | MLP config, final loss, training history |
| `artifacts/part3_mlp_checkpoint.npz` | Trained MLP parameters |
| `artifacts/part3_settings.json` | Revisions, architecture, seeds, packages |

## Reproduce Part 3

```sh
cd week8
uv sync
uv run python -m src.part3
uv run python -m pytest -q
```

The full pipeline takes about 10 minutes on CPU, dominated by the 4000 MLP
training steps. The full test suite currently contains **88 passing tests**
(16 Part 1, 14 Part 2, 19 Part 3, 9 Part 4, 20 Part 5, 10 Part 6).
# Week 8 — Part 4: Attention: A Memory Built from the Text

Status: **Part 4 complete — VERIFY (1) PASS**. Parts 1–3
results are unchanged.

## Picture memory

- 32×32 binary character: block letter **A** (264 ink pixels).
- Encoding: ink = **+1**, background = **−1**; flattened to 1024 dimensions.
- The learning-sheet PDF/glyph is not available in the repository, so the
  block letter is a documented substitution for the Figure 13 glyph.
- Keys: `M` random unit vectors in `d = 64`.
- Target memory: the character is value 0; the other `M−1` values are random
  ±1 pictures.
- Query: the target's own key. Draw seed = `1730 + M`, so every `M` has a
  reproducible independent draw.

Four mechanisms are compared:

| Method | Weight for memory i |
|---|---|
| Outer-product | `k_i · q` |
| ReLU | `max(0, k_i · q)` |
| ReLU² | `max(0, k_i · q)²` |
| Softmax | `softmax(8 * (k_i · q))`, computed stably |

`R = ||interference|| / ||signal||`, with
`signal = target_weight * target_value` and
`interference = recalled - signal`. Each recalled picture is normalized by its
target weight for visualization.

## All 16 interference-to-signal ratios

| M | Outer-product | ReLU | ReLU² | Softmax β=8 |
|---:|---:|---:|---:|---:|
| 64 | 0.9766 | 0.6229 | 0.1325 | 0.0063 |
| 256 | 2.2451 | 1.6727 | 0.3895 | 0.0183 |
| 1024 | 4.1837 | 3.0111 | 0.6846 | 0.0329 |
| 4096 | 8.0405 | 5.7568 | 1.2240 | 0.0612 |

Softmax reconstruction quality (cosine between normalized recall and target):

| M | R | Cosine |
|---:|---:|---:|
| 64 | 0.0063 | 0.999980 |
| 256 | 0.0183 | 0.999833 |
| 1024 | 0.0329 | 0.999458 |
| 4096 | 0.0612 | 0.998128 |

## VERIFY (1): PASS

- M=1024 outer-product: R=**4.1837**, theory 4.0, error **+4.59%** (limit 15%) ✅
- M=1024 ReLU: R=**3.0111**, theory 2.83, error **+6.40%** (limit 15%) ✅
- M=4096 ReLU²: R=**1.2240 > 1** ✅
- Softmax R < 0.1 at every M (maximum 0.0612) ✅
- Softmax clearly reconstructs the character (cosine ≥ 0.998) ✅
- Non-softmax R increases monotonically with M; general trends match Figure 13 ✅

M=4096 comparison with the references:

| Method | Measured R | Reference | Relative error |
|---|---:|---:|---:|
| Outer-product | 8.0405 | 7.94 | +1.27% |
| ReLU | 5.7568 | 5.82 | −1.09% |
| ReLU² | 1.2240 | 1.23 | −0.49% |
| Softmax β=8 | 0.0612 | 0.057 | +7.37% |

## Part 4 figure and artifacts

| Path | Purpose |
|---|---|
| `src/part4_memory.py` | Character bitmap, memory construction, four recall mechanisms |
| `src/part4_plotting.py` | Figure 13 4×4 dot-matrix visualization |
| `src/part4.py` | End-to-end Part 4 pipeline and report |
| `tests/test_part4_memory.py` | Character, softmax, memory, and ratio tests |
| `tests/test_part4_artifacts.py` | Saved-output acceptance checks |
| `evidence/figure13_attention_recall.png` | Figure 13 reproduction |
| `evidence/part4_verify.txt` | Full VERIFY (1) report |
| `artifacts/part4_results.json` | All 16 R values, cosines, checks, settings |
| `artifacts/part4_recall_data.npz` | Character, R/cosine tables, normalized images |
| `artifacts/part4_settings.json` | Picture, seed, mechanisms, acceptance settings |

## Reproduce Part 4

```sh
cd week8
uv sync
uv run python -m src.part4
uv run python -m pytest -q
```

The figure uses dot size for normalized pixel magnitude; green titles/frames
mark `R < 1`. Weak stores become unrecognizable as M grows, while softmax
remains clear.
# Week 8 — Part 5: The Transformer: Training and Generating

Status: **Part 5 complete — VERIFY (1)–(4) PASS**.
Parts 1–4 outputs are preserved.

## Architecture and training

Pre-norm decoder-only Transformer, implemented in JAX and trained on CPU:

| Item | Value |
|---|---|
| Parameter count | **328,832** |
| Layers | 2 |
| Model width | 96 |
| Attention heads | 4 |
| Head dimension | 24 |
| Context length | 64 |
| MLP hidden width | 384, GELU |
| Token / positional embeddings | learned |
| Attention | causal multi-head self-attention |
| Linear maps | all include biases |
| Normalization | LayerNorm (pre-norm + final) |
| Optimizer | AdamW |
| Steps / batch | 6000 / 32 sequences |
| Example length | 65 tokens (64 input + next-token targets) |
| Peak LR / warmup | 0.003 / 300 steps |
| LR schedule | cosine decay to 0.1 × peak |
| Device | CPU |
| Training tokens | 9,667,035 |
| Validation tokens | 1,078,996 |
| **Final validation loss** | **1.716034 nats/token** |
| **Final validation bits/char** | **1.206566** |

The training curve and held-out bpc curve are in
`evidence/figure_transformer_training.png`.

## VERIFY (1): PASS

Comparison of held-out bits/character:

| Model | Bits/char |
|---|---:|
| Uniform | 4.386331 |
| Unigram | 3.402710 |
| Bigram | 2.136232 |
| Trigram | 1.472582 |
| **Transformer** | **1.206566** |

- Transformer bpc < 1.310 ✅
- Reference Transformer value ≈ 1.207; measured **1.206566** ✅
- Figure: `evidence/figure_verify1_bits_per_char.png`

Temperature-1 samples (same seed 1848, increasing quality; 60 tokens each)
are in `artifacts/part5_samples_temperature1.json` and displayed in
`evidence/figure_verify1_model_samples.png`. The trigram sample is seeded with
`[EOS, 403]` (`▁Once`), because `[EOS, EOS]` is an unseen trigram context.

## VERIFY (2): PASS

Loss by context length is in `evidence/figure17_context_length.png`.

| Quantity | Value |
|---|---:|
| Mean loss after 32 tokens | **1.646196** nats/token |
| Acceptance | < 2.09 |
| Mean loss over final 8 positions | **1.654484** nats/token |
| Reference final 8 | ≈ 1.65 |
| Bigram reference | 3.038 |
| Trigram reference | 2.094 |

## VERIFY (3): PASS

A broad validation context (entropy 3.7374 nats) was selected from the
held-out stream. For each sampling setting, 100,000 next tokens were sampled
and compared with the exact target distribution using a pooled chi-squared
test; rare categories were pooled at expected count < 5.

| Sampling setting | χ² | dof | p-value |
|---|---:|---:|---:|
| Temperature T=0.8 | 106.816 | 113 | **0.6460** |
| Top-k (k=20, T=0.8) | 12.160 | 19 | **0.8787** |
| Top-p (p=0.9, T=0.8) | 23.939 | 27 | **0.6337** |
| Min-p (p=0.1, T=0.7) | 7.992 | 13 | **0.8441** |

Every p-value > 0.001 ✅. At T=0.0001 the distribution is one-hot at the
argmax and 1000/1000 sampled tokens were the most likely token ✅.
Examples for all four sampling settings are in
`artifacts/part5_sampling_method_examples.json` and
`evidence/figure_sampling_method_examples.png`.

## VERIFY (4): PASS

128 independent sequences of 128 generated tokens were scored under the
original T=1 model distribution.

| Generation temperature | Mean surprise | Mean entropy | Difference | z-score |
|---|---:|---:|---:|---:|
| T=1.0 | 1.736733 | 1.737105 | −0.000372 | **−0.035** |
| T=0.7, scored at T=1 | 1.216935 | 1.658102 | −0.441167 | **−69.417** |

- T=1: −4 < z < 4 ✅
- T=0.7: |z| > 40 ✅
- Figure: `evidence/figure_verify4_surprise_entropy.png`

## Part 5 files

| Path | Purpose |
|---|---|
| `src/transformer.py` | Transformer, loss, validation-by-position, training, checkpoint loading |
| `src/sampling.py` | Temperature, top-k, top-p, min-p, chi-squared, n-gram sampling |
| `src/part5_generation.py` | Autoregressive generation and surprise/entropy scoring |
| `src/part5_plotting.py` | BPC, Figure 17, training, samples, VERIFY (4) figures |
| `src/part5.py` | End-to-end Part 5 pipeline and report |
| `tests/test_transformer.py` | Architecture, masking, normalization, checkpoint tests |
| `tests/test_sampling.py` | Sampling/truncation/chi-squared/generation tests |
| `tests/test_part5_artifacts.py` | Saved-output acceptance checks |
| `evidence/figure_verify1_bits_per_char.png` | Model comparison |
| `evidence/figure_verify1_model_samples.png` | Five temperature-1 samples |
| `evidence/figure17_context_length.png` | Figure 17 reproduction |
| `evidence/figure_transformer_training.png` | Training and validation curves |
| `evidence/figure_sampling_method_examples.png` | Four sampling-setting examples |
| `evidence/figure_verify4_surprise_entropy.png` | Surprise versus entropy |
| `evidence/part5_verify.txt` | Full VERIFY (1)–(4) report |
| `artifacts/part5_results.json` | Combined results and acceptance flags |
| `artifacts/part5_transformer_results.json` | Losses, per-position loss, history |
| `artifacts/part5_transformer_checkpoint.npz` | Trained Transformer parameters |
| `artifacts/part5_samples_temperature1.json/.txt` | Five T=1 story samples |
| `artifacts/part5_sampling_method_examples.json` | Four sampling-method stories |
| `artifacts/part5_verify4_sequences.npz` | Surprise/entropy arrays |
| `artifacts/part5_settings.json` | Architecture, training, and verification settings |

## Reproduce Part 5

```sh
cd week8
uv sync
uv run python -m src.part5
uv run python -m pytest -q
```

Training takes about 10 minutes on CPU. To regenerate verification outputs
from the saved checkpoint without retraining, set
`PART5_REUSE_CHECKPOINT=1` before running `src.part5`.
# Week 8 — Part 6: Scaling Laws

Status: **Part 6 complete — VERIFY (1), (2), and (3) PASS**. Parts 1–5 outputs
are preserved.

## Dataset

| Item | Value |
|---|---|
| Source | `epoch-research/analyzing-chinchilla` |
| URL | `https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv` |
| Local file | `data/chinchilla-runs.csv` (gitignored) |
| Columns used | `Model Size` (N), `Training FLOP` (C), `loss` (L) |
| Data tokens | `D = C / (6N)` |
| Runs | 245 total; 5 highest-loss runs removed; **240 used** |
| N range | 5.733e7 – 1.618e10 |
| D range | 8.187e8 – 3.178e11 tokens |
| Loss range after trimming | 2.0774 – 3.4059 |

## Fit

Model:

```text
L = E + A/N^alpha + B/D^beta
```

Objective: mean Huber loss on `log(L)` with `delta = 0.001`.
Optimizer: L-BFGS-B from a 72-point deterministic grid; keep the best fit.
Bootstrap: 400 nonparametric resamples of the 240 runs, 3 starts each,
2.5th–97.5th percentile intervals.

| Parameter | Fit | Std. error | 95% CI |
|---|---:|---:|---:|
| E | **1.817190** | 0.025919 | [1.765715, 1.869960] |
| A | **477.701973** | 117.986974 | [280.884713, 706.036702] |
| B | **2142.839554** | 1106.195598 | [1037.299685, 5252.803663] |
| alpha | **0.347296** | 0.014867 | [0.315729, 0.370421] |
| beta | **0.367160** | 0.019422 | [0.332166, 0.410506] |

Huber objective: `4.242809e-06`. Reference:
E=1.817, A=478, B=2143, alpha=0.347, beta=0.367.

## VERIFY (1): PASS

- alpha in [0.32, 0.38] ✅
- beta in [0.32, 0.42] ✅
- E in [1.75, 1.90] ✅
- Figure 19: `evidence/figure19_scaling_fit.png`

## VERIFY (2): PASS

Residual definition: `log(L_pred) - log(L_observed)`.

| Fit | Mean residual | RMS residual |
|---|---:|---:|
| Our fit | **−0.000657** | 0.007550 |
| Published parameters | **+0.016742** | 0.019250 |

- Our mean residual is within ±0.005 ✅
- Published mean residual is > 0.01 ✅
- Figure 20: `evidence/figure20_residuals.png`

## VERIFY (3): PASS

At `C = 5.76e23` FLOP:

| Quantity | Our fit |
|---|---:|
| N* | **73.19B** |
| D* | **1.312T** |
| D*/N* | **17.9206** |
| Bootstrap 95% interval for D*/N* | **[7.9632, 34.6101]** |
| Chinchilla ratio | 20 |
| Relative difference from Chinchilla | **−10.40%** |

- D*/N* is between 10 and 30 ✅
- Reference ratio ≈ 17.9; measured 17.9206 ✅
- Published law at the same compute gives N*=32.19B, D*=2.982T, D*/N*=92.647.
- Figure 21: `evidence/figure21_compute_optimal.png`

## Part 6 files

| Path | Purpose |
|---|---|
| `src/download_part6_data.py` | Download and checksum-verify the CSV |
| `src/scaling_laws.py` | Data prep, Huber fit, L-BFGS multi-start, bootstrap, compute optima |
| `src/part6_plotting.py` | Figures 19, 20, 21 |
| `src/part6.py` | End-to-end Part 6 pipeline and report |
| `tests/test_scaling_laws.py` | Data, fit, residual, ratio, bootstrap tests |
| `tests/test_part6_artifacts.py` | Saved-output acceptance checks |
| `evidence/figure19_scaling_fit.png` | Figure 19 reproduction |
| `evidence/figure20_residuals.png` | Figure 20 reproduction |
| `evidence/figure21_compute_optimal.png` | Figure 21 reproduction |
| `evidence/part6_verify.txt` | Full VERIFY (1)–(3) report |
| `artifacts/part6_results.json` | Fit, bootstrap summary, residual and ratio checks |
| `artifacts/part6_settings.json` | Data URL, objective, bounds, starts, bootstrap settings |
| `artifacts/part6_bootstrap.npz` | 400 bootstrap parameter draws and ratio draws |
| `artifacts/part6_compute_optimal.csv` | N* and D* tables for both laws |

## Reproduce Part 6

```sh
cd week8
uv sync
uv run python -m src.download_part6_data
uv run python -m src.part6
uv run python -m pytest -q
```

---

# Week 8 — Complete Summary (Parts 1–6)

| Part | Topic | Verification | Headline result |
|---|---|---|---|
| 1 | Pretrained embeddings | VERIFY (1) PASS | Relation mean cosines 0.466 / 0.549 / 0.433; shuffled control ≈ 0 |
| 2 | Count models / n-grams | VERIFY (1)–(4) PASS | Bigram 3.038, trigram 2.094 nats/token; 4-gram minimum 1.310 bpc |
| 3 | MLP memory in weights | VERIFY (1)–(3) PASS | Memory capacity slopes 1.00 / 1.00 / 0.98 / 1.91 / 2.81; MLP 3.041007 nats/token |
| 4 | Attention recall | VERIFY (1) PASS | M=4096 R: outer 8.04, ReLU 5.76, ReLU² 1.224, softmax 0.061 |
| 5 | Transformer training/generation | VERIFY (1)–(4) PASS | 328,832 parameters; **1.206566 bits/char**; all sampling p > 0.001 |
| 6 | Scaling laws | VERIFY (1)–(3) PASS | α=0.3473, β=0.3672, E=1.8172; D*/N*=17.92 |

## Key explanations

- **Tokenizer (Part 2).** A 512-token SentencePiece model
  (`karpathy/tinyllamas`, revision `0bd21da`) tokenizes TinyStories at about
  **2.0536 characters/token**, with EOS appended after every story and no UNK
  tokens. All 27,630 stories are used: 24,867 train / 2,763 validation.
- **Memory recall (Parts 3–4).** A memory is a sum of value vectors weighted by
  key similarity. Interference-to-signal ratio
  `R = ||interference|| / ||signal||` crosses 1 at capacity. Linear and ReLU
  stores scale linearly with dimension; ReLU² and ReLU³ scale roughly as d² and
  d^2.8. In the picture task, ordinary stores become noisy as M grows, while
  softmax attention keeps R below 0.07 and reconstructs the character.
- **Model comparison (Part 5).** The count models improve from uniform 4.386 to
  trigram 1.473 bits/char. The 2-layer Transformer reaches **1.2066 bits/char**,
  and its loss falls below both bigram and trigram references after enough
  context. Top-k/top-p/min-p sampling match their exact distributions.
- **Scaling law (Part 6).** Loss follows
  `L = E + A/N^α + B/D^β` with `E≈1.817`, `α≈0.347`, `β≈0.367`.
  Compute-optimal allocation at 5.76e23 FLOP is about 73B parameters and
  1.31T tokens (`D*/N*≈17.9`), close to Chinchilla's ratio of 20.

## All dataset/tokenizer URLs

- TinyStories:
  `https://huggingface.co/datasets/roneneldan/TinyStories` (revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`)
- Part 2 tokenizer:
  `https://huggingface.co/karpathy/tinyllamas` (revision `0bd21da`,
  `stories260K/tok512.model`)
- Part 1 embeddings:
  `https://huggingface.co/minishlab/potion-base-8M` (revision `bf8b056`)
- Chinchilla CSV:
  `https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv`

## Evidence figure index

```text
Part 1: evidence/figure5_word_relations.png
Part 2: evidence/figure7_curse_of_dimensionality.png
        evidence/figure_verify3_bigram_chi2.png
Part 3: evidence/figure9_memory_capacity_d64.png
        evidence/figure10_capacity_vs_dimension.png
        evidence/figure_mlp_training.png
Part 4: evidence/figure13_attention_recall.png
Part 5: evidence/figure_verify1_bits_per_char.png
        evidence/figure_verify1_model_samples.png
        evidence/figure17_context_length.png
        evidence/figure_transformer_training.png
        evidence/figure_sampling_method_examples.png
        evidence/figure_verify4_surprise_entropy.png
Part 6: evidence/figure19_scaling_fit.png
        evidence/figure20_residuals.png
        evidence/figure21_compute_optimal.png
```

## Full reproduction sequence

```sh
cd week8
uv sync

# Part 1
uv run python -m src.download_model
uv run python -m src.part1

# Part 2
uv run python -m src.download_part2_data
uv run python -m src.part2

# Part 3 (about 10 minutes: MLP training)
uv run python -m src.part3

# Part 4
uv run python -m src.part4

# Part 5 (about 10 minutes: Transformer training)
uv run python -m src.part5

# Part 6
uv run python -m src.download_part6_data
uv run python -m src.part6

# All tests
uv run python -m pytest -q
```

Datasets, checkpoints, and artifacts are excluded from Git via `.gitignore`.
The full automated suite contains **88 passing tests**.
