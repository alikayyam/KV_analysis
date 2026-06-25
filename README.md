# Condition C: A Geometric Theory of K=V Weight Tying in Transformers

**Paper:** [arXiv link — coming soon]  
**Authors:** Ali Kayyam, BrainChip Holdings Ltd.

---

## Overview

This repository contains the code and results for the paper *Condition C: A Geometric Theory of K=V Weight Tying in Transformers*.

We provide the first theoretical account of when and how tying the key and value projections in transformer attention (K=V) can be applied to a pretrained model. Our main result (Theorem 1) identifies the exact geometric condition under which a pretrained untied block can be converted to a K=V block without loss (Condition C). We introduce ρ ∈ [0,1] to measure how closely any pretrained model satisfies this condition, and show empirically that retrofit quality scales with ρ as the theory predicts.

---

## Repository Structure

```
├── compute_rho.py              # Experiment 1: compute ρ and diagnostics per layer and head
├── experiment2.py              # Experiment 2: FFN fine-tuning under K=V constraint
├── plot_rho_per_layer.py       # Per-layer ρ line plots (Figure 1, top panels)
├── plot_depth_decomposition.py # Depth decomposition: ‖W_V−W_K‖_F and stable rank of W_K
├── plot_experiment1.py         # Additional Experiment 1 plots
├── plot_experiment2.py         # Experiment 2 plots
├── print_table1.py             # Print Table 1 statistics (mean ρ, min ρ, ‖M‖_F, predictor)
├── pyproject.toml              # Project dependencies
└── results/
    ├── rho_results.npz               # ρ and diagnostics for 8 pretrained models
    ├── exp2_opt-125m_results.npz     # Experiment 2 results, OPT-125M
    ├── exp2_opt-350m_results.npz     # Experiment 2 results, OPT-350M
    ├── rho_per_layer_*.pdf/png       # Per-layer ρ figures
    └── depth_decomp_*.pdf/png        # Depth decomposition figures
```

---

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Or install manually:

```bash
pip install torch transformers datasets numpy matplotlib
```

---

## Reproducing the Results

**Experiment 1** — measure ρ and weight-space diagnostics across pretrained models:

```bash
python compute_rho.py
```

Results are saved to `results/rho_results.npz`. The file contains per-layer, per-head arrays for each model:

| Key | Contents |
|---|---|
| `{model}` | ρ ∈ [0,1] — Condition C coverage |
| `{model}_M_norm` | ‖M‖_F where M = W_O(W_V − W_K) |
| `{model}_diff_norm` | ‖W_V − W_K‖_F — V–K divergence |
| `{model}_stable_rank_K` | Stable rank of W_K = ‖W_K‖_F² / ‖W_K‖_op² |

**Experiment 2** — FFN fine-tuning under K=V constraint:

```bash
python experiment2.py --model facebook/opt-125m
python experiment2.py --model facebook/opt-350m
```

**Generate paper figures:**

```bash
python plot_rho_per_layer.py        # per-layer ρ (Figure 1 top panels)
python plot_depth_decomposition.py  # depth decomposition (Figure 1 middle/bottom panels)
```

**Print Table 1 statistics:**

```bash
python print_table1.py
```

---

## Results Summary

**Experiment 1** — Condition C diagnostics across 8 pretrained models:

| Model | Mean ρ | Min ρ | Mean ‖M‖_F | Mean √(1−ρ²)‖M‖_F |
|---|---|---|---|---|
| OPT-125M    | 0.904 | 0.748 |   7.44 |  3.20 |
| OPT-350M    | 0.677 | 0.480 |   8.90 |  6.53 |
| OPT-1.3B    | 0.674 | 0.308 |   7.44 |  5.65 |
| GPT-2 Small | 0.655 | 0.374 | 137.51 | 98.32 |
| GPT-2 Medium| 0.629 | 0.245 | 118.48 | 92.22 |
| GPT-2 Large | 0.723 | 0.468 |  33.02 | 22.60 |
| BERT-base   | 0.683 | 0.532 |  10.77 |  7.87 |
| BERT-large  | 0.659 | 0.432 |  10.59 |  7.94 |

The column √(1−ρ²)‖M‖_F is the per-head predictor of conversion error from the approximate bound (Proposition 2 in the paper). Note that GPT-2 ‖M‖_F values are not directly comparable to OPT/BERT due to weight-scale differences across architectures.

**Experiment 2** — retrofit quality under K=V constraint (OPT models):

| Model | PPL (untied, zero-shot) | PPL (tied, no tuning) | PPL (tied, FFN FT) |
|---|---|---|---|
| OPT-125M | 59.2 | — | 69.8 |
| OPT-350M | 47.7 | — | 2,031 |

Note: the untied baseline is zero-shot (no fine-tuning); a fine-tuned untied baseline would be the fair reference. ρ and model size are perfectly correlated across these two points; a controlled sweep is needed to isolate ρ as the causal predictor.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{kayyam2026conditionc,
  title   = {Condition C: A Geometric Theory of K=V Weight Tying in Transformers},
  author  = {Kayyam, Ali},
  journal = {arXiv preprint},
  year    = {2026}
}
```

See also the companion empirical paper:

```bibtex
@article{kayyam2026kvtying,
  title   = {Do Transformers Need Three Projections?},
  author  = {Kayyam, Ali},
  year    = {2026}
}
```
