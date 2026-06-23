"""
Experiment 2 visualisation — Figure 3, Figure 4, Table 2.

Reads:  results/exp2_results.npz   (from experiment2.py)
        results/rho_results.npz    (from compute_rho.py, key "OPT-125M")
Writes: results/figure3_gamma.{pdf,png}
        results/figure4_scatter.{pdf,png}

Usage
-----
    python plot_experiment2.py
    python plot_experiment2.py --rho-key OPT-350M  # if you ran a different scale
"""

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
})


# ── load ──────────────────────────────────────────────────────────────────────

def load(exp2_path: str, rho_path: str, rho_key: str) -> dict:
    e2  = np.load(exp2_path)
    rho = np.load(rho_path)

    if rho_key not in rho:
        raise SystemExit(
            f"Key {rho_key!r} not found in {rho_path}. "
            f"Available: {list(rho.files)}"
        )

    gamma       = e2["gamma"]                  # (num_layers,)
    rho_matrix  = rho[rho_key]                 # (num_layers, num_heads)
    mean_rho    = rho_matrix.mean(axis=1)      # (num_layers,)

    return dict(
        gamma       = gamma,
        rho_matrix  = rho_matrix,
        mean_rho    = mean_rho,
        ppl_untied  = float(e2["ppl_untied"]),
        ppl_tied_nt = float(e2["ppl_tied_nt"]),
        ppl_tied_ft = float(e2["ppl_tied_ft"]),
    )


# ── Figure 3 — per-layer γ with mean ρ overlay ───────────────────────────────

def plot_gamma_bars(data: dict, out_stem: str) -> None:
    gamma    = data["gamma"]
    mean_rho = data["mean_rho"]
    num_layers = len(gamma)
    layers = np.arange(num_layers)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    bar_color = np.where(gamma >= 0, "#4477AA", "#DD4444")
    ax.bar(layers, gamma, color=bar_color, alpha=0.8, label=r"$\gamma^{(\ell)}$")
    ax.plot(layers, mean_rho, color="tomato", linestyle="--", linewidth=1.6,
            marker="o", markersize=4, label=r"mean $\rho^{(\ell)}$")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title(
        r"Figure 3 — Per-layer correction ratio $\gamma^{(\ell)}$"
        " and mean $\\rho^{(\\ell)}$  [OPT-125M]"
    )
    ax.legend(fontsize=9)

    _save(fig, out_stem, "gamma")


# ── Figure 4 — scatter γ vs ρ ─────────────────────────────────────────────────

def plot_scatter(data: dict, out_stem: str) -> None:
    gamma      = data["gamma"]        # (L,)
    rho_matrix = data["rho_matrix"]   # (L, H)
    num_layers, num_heads = rho_matrix.shape

    # One point per (layer, head): x = ρ^(ℓ,h), y = γ^(ℓ)
    rho_flat   = rho_matrix.flatten()                            # (L*H,)
    gamma_flat = np.repeat(gamma, num_heads)                    # (L*H,)
    layer_idx  = np.repeat(np.arange(num_layers), num_heads)   # (L*H,)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    sc = ax.scatter(
        rho_flat, gamma_flat,
        c=layer_idx, cmap="viridis", alpha=0.65, s=18, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="Layer index")

    # diagonal reference
    lo, hi = 0.0, 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label=r"$\gamma = \rho$")

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.15, hi)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"$\rho^{(\ell,h)}$")
    ax.set_ylabel(r"$\gamma^{(\ell)}$")
    ax.set_title(
        r"Figure 4 — $\gamma^{(\ell)}$ vs $\rho^{(\ell,h)}$ per (layer, head)  [OPT-125M]"
    )
    ax.legend(fontsize=9)

    _save(fig, out_stem, "scatter")


# ── Table 2 ───────────────────────────────────────────────────────────────────

def print_table(data: dict, model_name: str = "OPT-125M") -> None:
    cols   = ["Model", "Untied (PPL)", "Tied, no tuning (PPL)", "Tied, FFN fine-tuned (PPL)"]
    widths = [12, 14, 24, 28]
    header = "".join(c.rjust(w) for c, w in zip(cols, widths))
    rule   = "─" * len(header)

    print(f"\n{rule}")
    print("Table 2 — Task perplexity comparison")
    print(rule)
    print(header)
    print(rule)
    row = [
        model_name,
        f"{data['ppl_untied']:.2f}",
        f"{data['ppl_tied_nt']:.2f}",
        f"{data['ppl_tied_ft']:.2f}",
    ]
    print("".join(v.rjust(w) for v, w in zip(row, widths)))
    print(rule)


# ── I/O ───────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str, tag: str) -> None:
    for ext in ("pdf", "png"):
        path = f"{stem}_{tag}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exp2",     default="results/exp2_results.npz")
    parser.add_argument("--rho",      default="results/rho_results.npz")
    parser.add_argument("--rho-key",  default="OPT-125M",
                        help="key in rho_results.npz (default: OPT-125M)")
    parser.add_argument("--out-stem", default="results/figure",
                        help="prefix for output files (default: results/figure)")
    args = parser.parse_args()

    data = load(args.exp2, args.rho, args.rho_key)
    print(f"Loaded: γ shape={data['gamma'].shape}, "
          f"ρ shape={data['rho_matrix'].shape}")

    plot_gamma_bars(data, args.out_stem)
    plot_scatter   (data, args.out_stem)
    print_table    (data)


if __name__ == "__main__":
    main()
