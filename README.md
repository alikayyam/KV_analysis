# Condition C: A Geometric Theory of K=V Weight Tying in Transformers

**Paper:** [arXiv link — coming soon]  
**Authors:** Ali Kayyam, BrainChip Holdings Ltd.

---

## Overview

This repository contains the code and results for the paper *Condition C: A Geometric Theory of K=V Weight Tying in Transformers*.

We provide the first theoretical account of when and why tying the key and value projections in transformer attention (K=V) can succeed. Our main result (Theorem 1) identifies the exact geometric condition under which a pretrained untied block can be converted to a K=V block without loss. We introduce ρ ∈ [0,1] to measure how closely a pretrained model satisfies this condition, and show empirically that conversion quality scales with ρ as the theory predicts.

---

## Repository Structure

```
├── compute_rho.py            # Experiment 1: compute ρ per layer and head
├── experiment2.py            # Experiment 2: FFN fine-tuning under K=V constraint
├── plot_experiment1.py       # Plotting for Experiment 1
├── plot_experiment2.py       # Plotting for Experiment 2
├── plot_rho_per_layer.py     # Per-layer ρ line plots (paper figures)
├── pyproject.toml            # Project dependencies
└── results/
    ├── rho_results.npz               # ρ measurements (8 pretrained models)
    ├── exp2_opt-125m_results.npz     # Experiment 2 results, OPT-125M
    ├── exp2_opt-350m_results.npz     # Experiment 2 results, OPT-350M
    └── rho_per_layer_*.pdf/png       # Paper figures
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

**Experiment 1** — measure ρ across pretrained models:

```bash
python compute_rho.py
```

Results are saved to `results/rho_results.npz`.

**Experiment 2** — FFN fine-tuning under K=V constraint:

```bash
python experiment2.py --model facebook/opt-125m
python experiment2.py --model facebook/opt-350m
```

**Generate paper figures:**

```bash
python plot_rho_per_layer.py
```

---

## Results Summary

| Model | Mean ρ | PPL (untied) | PPL (tied, FFN FT) |
|---|---|---|---|
| OPT-125M | 0.904 | 59.2 | 69.8 |
| OPT-350M | 0.677 | 47.7 | 2,031 |

Higher ρ → smaller perplexity gap after conversion, consistent with Theorem 1.

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
