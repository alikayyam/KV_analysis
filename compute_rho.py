"""
Experiment 1 — compute Condition-C coverage ρ per layer and head.

For each pretrained model, and for each (layer, head), extracts W_K, W_V, W_O
and computes

    ρ = ‖P_{col(B^T)} M^T‖_F² / ‖M‖_F²

where B = W_O W_K and M = W_O(W_V − W_K).

Saves: results/rho_results.npz   (one key per model, e.g. "OPT-125M")

Supported families
------------------
  opt     — OPT-125M / OPT-350M / OPT-1.3B
  gpt2    — GPT2-S / GPT2-M / GPT2-L   (Conv1D weights, stored transposed)
  pythia  — Pythia-160M / Pythia-410M / Pythia-1B  (interleaved QKV)
  bert    — BERT-base / BERT-large      (encoder-only; separate K, V)

Usage
-----
    python compute_rho.py                            # all models
    python compute_rho.py --models OPT-125M GPT2-S  # subset
    python compute_rho.py --out my_rho.npz           # custom output
"""

import argparse
import os
from typing import NamedTuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

# ── model registry ────────────────────────────────────────────────────────────

class ModelSpec(NamedTuple):
    hf_id:  str    # HuggingFace model identifier
    family: str    # weight-extraction family
    task:   str    # "causal_lm" | "masked_lm"

MODELS: dict[str, ModelSpec] = {
    # OPT — paper's original models
    "OPT-125M":    ModelSpec("facebook/opt-125m",              "opt",    "causal_lm"),
    "OPT-350M":    ModelSpec("facebook/opt-350m",              "opt",    "causal_lm"),
    "OPT-1.3B":    ModelSpec("facebook/opt-1.3b",              "opt",    "causal_lm"),
    # GPT-2 — Conv1D weights (stored as (in, out), i.e. transposed vs nn.Linear)
    "GPT2-S":      ModelSpec("gpt2",                           "gpt2",   "causal_lm"),
    "GPT2-M":      ModelSpec("gpt2-medium",                    "gpt2",   "causal_lm"),
    "GPT2-L":      ModelSpec("gpt2-large",                     "gpt2",   "causal_lm"),
    # BERT — encoder-only, bidirectional; separate K and V projections
    "BERT-base":   ModelSpec("google-bert/bert-base-uncased",  "bert",   "masked_lm"),
    "BERT-large":  ModelSpec("google-bert/bert-large-uncased", "bert",   "masked_lm"),
}


# ── per-family helpers ────────────────────────────────────────────────────────

def _model_shape(model, spec: ModelSpec) -> tuple:
    """Return (layers_iterable, num_layers, num_heads, d_k, d)."""
    f = spec.family
    if f == "opt":
        layers = model.model.decoder.layers
        a = layers[0].self_attn
        return layers, len(layers), a.num_heads, a.head_dim, a.embed_dim
    if f == "gpt2":
        layers = model.transformer.h
        d = model.config.n_embd
        nh = model.config.n_head
        return layers, len(layers), nh, d // nh, d
    if f == "pythia":
        layers = model.gpt_neox.layers
        a = layers[0].attention
        return layers, len(layers), a.num_attention_heads, a.head_size, a.hidden_size
    if f == "bert":
        layers = model.bert.encoder.layer
        d = model.config.hidden_size
        nh = model.config.num_attention_heads
        return layers, len(layers), nh, d // nh, d
    raise ValueError(f"Unknown family: {f!r}")


def _extract_head(layer, family: str, h: int, d_k: int, d: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (W_K, W_V, W_O) as float64 numpy arrays with shapes
    (d_k, d), (d_k, d), (d, d_k) for head index h.
    """
    s, e = h * d_k, (h + 1) * d_k

    if family == "opt":
        a = layer.self_attn
        WK = a.k_proj.weight.detach().float().numpy()     # (h·d_k, d)
        WV = a.v_proj.weight.detach().float().numpy()
        WO = a.out_proj.weight.detach().float().numpy()   # (d, h·d_k)
        return WK[s:e].astype(np.float64), WV[s:e].astype(np.float64), WO[:, s:e].astype(np.float64)

    if family == "gpt2":
        # Conv1D stores weight as (in_features, out_features), i.e. transposed
        # vs nn.Linear.  Operation: y = x @ w + b.
        # c_attn.weight: (d, 3*d)  →  columns [0:d]=Q, [d:2d]=K, [2d:3d]=V
        # For head h: W_K^h = c_attn.weight[:, d+s:d+e].T   shape (d_k, d)
        # c_proj.weight: (d, d)
        # For head h: W_O^h = c_proj.weight[s:e, :].T        shape (d, d_k)
        ca = layer.attn.c_attn.weight.detach().float().numpy()   # (d, 3d)
        cp = layer.attn.c_proj.weight.detach().float().numpy()   # (d, d)
        WK = ca[:, d + s : d + e].T       # (d_k, d)
        WV = ca[:, 2*d + s : 2*d + e].T   # (d_k, d)
        WO = cp[s:e, :].T                  # (d, d_k)
        return WK.astype(np.float64), WV.astype(np.float64), WO.astype(np.float64)

    if family == "pythia":
        # query_key_value.weight: (3*d, d), but the heads are interleaved:
        # head h occupies rows [h*3*d_k : (h+1)*3*d_k] = [Q_h | K_h | V_h]
        a = layer.attention
        qkv = a.query_key_value.weight.detach().float().numpy()  # (3d, d)
        WO  = a.dense.weight.detach().float().numpy()            # (d, d)
        off = h * 3 * d_k
        WK = qkv[off + d_k   : off + 2*d_k, :]   # (d_k, d)
        WV = qkv[off + 2*d_k : off + 3*d_k, :]   # (d_k, d)
        return WK.astype(np.float64), WV.astype(np.float64), WO[:, s:e].astype(np.float64)

    if family == "bert":
        # Standard nn.Linear: weight shape (out, in).  Per-head rows = W_K^h.
        a = layer.attention
        WK = a.self.key.weight.detach().float().numpy()        # (d, d)
        WV = a.self.value.weight.detach().float().numpy()
        WO = a.output.dense.weight.detach().float().numpy()    # (d, d)
        return WK[s:e].astype(np.float64), WV[s:e].astype(np.float64), WO[:, s:e].astype(np.float64)

    raise ValueError(f"Unknown family: {family!r}")


# ── diagnostics ───────────────────────────────────────────────────────────────

def compute_head_diagnostics(
    WK: np.ndarray,
    WV: np.ndarray,
    WO: np.ndarray,
    eps: float = 1e-6,
) -> tuple[float, float, float, float]:
    """
    Compute four per-head diagnostics.

    Parameters
    ----------
    WK : (d_k, d)   key projection
    WV : (d_k, d)   value projection
    WO : (d,  d_k)  output projection (column slice for this head)
    eps : relative SVD truncation threshold

    Returns
    -------
    (rho, diff_norm, stable_rank_K, M_norm)
      rho           : Condition-C coverage in [0, 1]
      diff_norm     : ‖W_V - W_K‖_F  (divergence of V from K)
      stable_rank_K : ‖W_K‖_F² / ‖W_K‖_op²  (1 = rank-1, d_k = full rank)
      M_norm        : ‖M‖_F where M = W_O(W_V - W_K)
    """
    diff = WV - WK  # (d_k, d)

    # ── diff_norm ──────────────────────────────────────────────────────────────
    diff_norm = float(np.linalg.norm(diff, "fro"))

    # ── stable rank of W_K ────────────────────────────────────────────────────
    s_wk = np.linalg.svd(WK, compute_uv=False)
    wk_fro_sq = float(np.sum(s_wk ** 2))
    wk_op_sq  = float(s_wk[0] ** 2) if s_wk[0] > 0 else 1.0
    stable_rank_K = wk_fro_sq / wk_op_sq if wk_op_sq > 0 else 0.0

    # ── ‖M‖_F (avoids forming the d×d matrix M) ───────────────────────────────
    G = WO.T @ WO      # (d_k, d_k)
    H = diff @ diff.T  # (d_k, d_k)
    norm_M_sq = float(np.trace(G @ H))
    M_norm = float(np.sqrt(max(norm_M_sq, 0.0)))

    # ── ρ ─────────────────────────────────────────────────────────────────────
    if norm_M_sq < 1e-20:
        return 1.0, diff_norm, stable_rank_K, M_norm  # M ≈ 0: Condition C trivial

    U_O, s_O, _ = np.linalg.svd(WO.T, full_matrices=False)
    if s_O[0] < 1e-20:
        return 0.0, diff_norm, stable_rank_K, M_norm
    U_O_r = U_O[:, s_O > eps * s_O[0]]        # (d_k, r_O)

    G_factor = WK.T @ U_O_r                   # (d, r_O)
    U_B, s_B, _ = np.linalg.svd(G_factor, full_matrices=False)
    if s_B[0] < 1e-20:
        return 0.0, diff_norm, stable_rank_K, M_norm
    U_B_r = U_B[:, s_B > eps * s_B[0]]        # (d, r_B)

    projected = WO @ (diff @ U_B_r)            # (d, r_B)
    rho = float(np.linalg.norm(projected, "fro") ** 2) / norm_M_sq
    return rho, diff_norm, stable_rank_K, M_norm


def compute_rho_head(WK, WV, WO, eps=1e-6) -> float:
    """Backward-compatible wrapper — returns only ρ."""
    return compute_head_diagnostics(WK, WV, WO, eps)[0]


# ── model loading and processing ──────────────────────────────────────────────

def _load_model(spec: ModelSpec):
    kwargs = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
    if spec.task == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    return AutoModelForMaskedLM.from_pretrained(spec.hf_id, **kwargs)


def process_model(key: str, spec: ModelSpec, eps: float = 1e-6) -> np.ndarray:
    print(f"\nLoading {key}  ({spec.hf_id}) …")
    model = _load_model(spec)
    model.eval()

    layers, num_layers, num_heads, d_k, d = _model_shape(model, spec)
    print(f"  {num_layers} layers · {num_heads} heads · d = {d} · d_k = {d_k}  [{spec.family}]")

    rho           = np.zeros((num_layers, num_heads), dtype=np.float64)
    diff_norm     = np.zeros((num_layers, num_heads), dtype=np.float64)
    stable_rank_K = np.zeros((num_layers, num_heads), dtype=np.float64)
    M_norm        = np.zeros((num_layers, num_heads), dtype=np.float64)

    for l, layer in enumerate(layers):
        for h in range(num_heads):
            WK, WV, WO = _extract_head(layer, spec.family, h, d_k, d)
            rho[l, h], diff_norm[l, h], stable_rank_K[l, h], M_norm[l, h] = \
                compute_head_diagnostics(WK, WV, WO, eps=eps)

        if (l + 1) % 6 == 0 or (l + 1) == num_layers:
            print(f"  layer {l+1:3d}/{num_layers}  running mean ρ = {rho[:l+1].mean():.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m, med, mn = rho.mean(), np.median(rho), rho.min()
    pct = (rho >= 0.95).mean() * 100
    print(f"  ── {key}: mean={m:.4f}  median={med:.4f}  min={mn:.4f}  pct≥0.95={pct:.1f}%")
    return rho, diff_norm, stable_rank_K, M_norm


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS),
                        default=list(MODELS),
                        metavar="MODEL",
                        help=f"models to run (default: all). choices: {list(MODELS)}")
    parser.add_argument("--out", default="results/rho_results.npz",
                        help="output .npz path (default: results/rho_results.npz)")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="relative SVD truncation threshold (default: 1e-6)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results: dict[str, np.ndarray] = {}
    if os.path.exists(args.out):
        existing = np.load(args.out)
        results = {k: existing[k] for k in existing.files}
        print(f"Loaded existing results for: {list(results.keys())}")

    for key in args.models:
        rho, diff_norm, stable_rank_K, M_norm = process_model(key, MODELS[key], eps=args.eps)
        results[key]                          = rho
        results[f"{key}_diff_norm"]           = diff_norm
        results[f"{key}_stable_rank_K"]       = stable_rank_K
        results[f"{key}_M_norm"]              = M_norm

    np.savez(args.out, **results)
    print(f"\nSaved → {args.out}  (keys: {list(results.keys())})")


if __name__ == "__main__":
    main()
