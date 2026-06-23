"""
Experiment 1 — visualisation.

Reads:  results/rho_results.npz
Writes: results/figure1_heatmap.{pdf,png}   (Figure 1)
        results/figure2_violin.{pdf,png}    (Figure 2)
Prints: Table 1 to stdout

Usage
-----
    python plot_experiment1.py
    python plot_experiment1.py --models OPT-125M OPT-350M
    python plot_experiment1.py --models GPT2-S GPT2-M GPT2-L
    python plot_experiment1.py --inp my_rho.npz --out-stem results/fig
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


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def present(rho: dict, models: list[str] | None) -> list[str]:
    keys = models if models else sorted(rho.keys())
    missing = [k for k in keys if k not in rho]
    if missing:
        raise SystemExit(f"Keys not found in results: {missing}. Available: {sorted(rho.keys())}")
    return keys


# ── Figure 1 — heatmaps ───────────────────────────────────────────────────────

def plot_heatmaps(rho: dict, out_stem: str, models: list[str] | None = None) -> None:
    scales = present(rho, models)
    fig, axes = plt.subplots(1, len(scales),
                              figsize=(4.8 * len(scales), 4.2),
                              constrained_layout=True)
    if len(scales) == 1:
        axes = [axes]

    for ax, key in zip(axes, scales):
        im = ax.imshow(
            rho[key], aspect="auto", vmin=0, vmax=1,
            cmap="Blues", origin="lower", interpolation="nearest",
        )
        ax.set_title(key)
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer index")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\rho$")

    fig.suptitle(
        r"Figure 1 — Condition-C coverage $\rho^{(\ell,h)}$ per layer and head",
        fontsize=12,
    )
    _save(fig, out_stem, "heatmap")


# ── Figure 2 — violin plots ───────────────────────────────────────────────────

def plot_violins(rho: dict, out_stem: str, models: list[str] | None = None) -> None:
    scales = present(rho, models)
    data   = [rho[s].flatten() for s in scales]

    fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
    parts   = ax.violinplot(data, positions=range(len(data)),
                             showmedians=True, showextrema=True)

    body_color = "#4477AA"
    edge_color = "#224488"
    for pc in parts["bodies"]:
        pc.set_facecolor(body_color)
        pc.set_edgecolor(edge_color)
        pc.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(edge_color)
            parts[key].set_linewidth(1.4)

    ax.axhline(0.95, color="tomato", linestyle="--", linewidth=1.3,
               label=r"$\rho = 0.95$")
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales, rotation=15, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"$\rho$")
    ax.set_title(r"Figure 2 — $\rho$ distribution across all heads and layers")
    ax.legend(fontsize=9)

    _save(fig, out_stem, "violin")


# ── Table 1 ───────────────────────────────────────────────────────────────────

def print_table(rho: dict, models: list[str] | None = None) -> None:
    scales = present(rho, models)
    cols   = ["Model", "Mean ρ", "Median ρ", "Min ρ", "% ≥ 0.95"]
    widths = [14, 10, 11, 9, 10]
    header = "".join(c.rjust(w) for c, w in zip(cols, widths))
    rule   = "─" * len(header)

    print(f"\n{rule}")
    print("Table 1 — Condition-C coverage statistics")
    print(rule)
    print(header)
    print(rule)
    for key in scales:
        r = rho[key]
        row = [
            key,
            f"{r.mean():.4f}",
            f"{np.median(r):.4f}",
            f"{r.min():.4f}",
            f"{(r >= 0.95).mean()*100:.1f}%",
        ]
        print("".join(v.rjust(w) for v, w in zip(row, widths)))
    print(rule)


# ── I/O helper ────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str, tag: str) -> None:
    for ext in ("pdf", "png"):
        path = f"{stem}_{tag}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp",      default="results/rho_results.npz",
                        help="input .npz from compute_rho.py")
    parser.add_argument("--out-stem", default="results/figure",
                        help="prefix for output files (default: results/figure)")
    parser.add_argument("--models", nargs="+", default=None,
                        metavar="MODEL",
                        help="subset of models to plot (default: all in file)")
    args = parser.parse_args()

    rho = load(args.inp)
    print(f"Loaded results for: {sorted(rho.keys())}")

    plot_heatmaps(rho, args.out_stem, args.models)
    plot_violins (rho, args.out_stem, args.models)
    print_table  (rho, args.models)


if __name__ == "__main__":
    main()
