# Claims Assessment — Revised Narrative (2026-06-23)

## Paper's Core Message

The barrier to K=V sharing is not expressivity — it is the solution found during pretraining.
ρ measures how far a pretrained solution is from being directly convertible to K=V.
Training from scratch sidesteps this by finding K=V solutions directly.
Theorem 1 is a **conversion theorem**, not a capacity theorem.

---

## Claim 1 — Theorem 1: exact conversion is possible when ρ=1

**Status: Theoretically sound, correctly scoped**

If Condition C holds (ρ=1), the pretrained untied model can be converted to a K=V model
exactly using FFN₂(z) = FFN₁((I+P)z) + Pz with width d_ffn + 2d_k.
The proof is constructive and complete.

The claim is intentionally conditional: it tells you the price of conversion (ρ=1 required),
not whether K=V is expressivity-limited in general. Low ρ means the pretrained solution
is not directly convertible — it does not mean K=V solutions of comparable quality
don't exist.

---

## Claim 2 — ρ predicts conversion difficulty

**Status: Supported by Exp 1 + Exp 2**

| Model    | Mean ρ | Retrofit PPL gap |
|----------|--------|-----------------|
| OPT-125M | 0.904  | 1.18×           |
| OPT-350M | 0.677  | 43×             |

Higher ρ → smaller gap after retrofitting. Consistent with Theorem 1: the closer the
pretrained solution is to Condition C, the cheaper the conversion.

What is NOT claimed: that high-ρ layers should be selectively tied for compression.
The KV cache saving from tying only high-ρ layers is modest (≤20% for most models),
making it impractical as a compression method.

---

## Claim 3 — Pretrained models find low-ρ solutions by default

**Status: Supported by Exp 1**

- OPT-125M: mean ρ=0.904 (special case — small model, possibly training-regime specific)
- OPT-350M, 1.3B: mean ρ≈0.67–0.68 (plateau, not monotonically decreasing with scale)
- GPT-2, BERT: 0.63–0.72 regardless of size

ρ is not a simple function of scale. It appears set by the pretraining regime.
Key finding: all models have ρ < 1, meaning none of their pretrained solutions
are directly convertible under Theorem 1.

---

## Claim 4 — K=V solutions of comparable quality exist (from scratch)

**Status: Supported by Kayyam et al. (cited)**

Kayyam et al. showed empirically that models trained from scratch with K=V achieve
comparable perplexity to standard QKV models. This establishes that low ρ in pretrained
models reflects a choice of solution, not a fundamental need for three projections.

---

## What Was Dropped and Why

| Item | Reason |
|------|--------|
| Theorem 2 (revised convergence bound) | Mathematically shaky — PL condition + non-convexity of transformers made bounds non-credible |
| Proposition (snap initialization gap) | Required global strong convexity — false for neural networks |
| Corollary (FT vs. from-scratch) | Applied fine-tuning convergence rate to from-scratch training unjustifiably |
| Experiment 3 (full fine-tuning K=V) | Designed for dropped Theorem 2; deprecated |
| Selective layer retrofitting direction | KV cache savings too modest to be a practical contribution |

---

## Recommendations for the Paper

1. **Frame Theorem 1 as a conversion theorem explicitly** — characterizes when conversion
   is possible, not whether K=V is practically compressive.

2. **Exp 1 as the empirical anchor** — ρ across models/layers shows pretrained solutions
   are far from directly convertible, consistently across architectures.

3. **Exp 2 as validation of Theorem 1** — conversion quality scales with ρ as the theorem predicts.

4. **Kayyam et al. as the punchline** — K=V solutions exist and are competitive;
   the network doesn't need three projections, pretraining just defaults to finding them.

5. **Do not oversell compression** — the paper's contribution is theoretical understanding
   and diagnostic framework, not a compression algorithm.
