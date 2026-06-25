"""
Figure: depth-dependent decomposition of ρ decline.

For each model family, plots three quantities per layer:
  · mean ρ (Condition-C coverage)
  · mean ‖W_V − W_K‖_F  (V–K divergence)
  · mean stable rank of W_K  (row-space contraction)

This tests whether the late-layer ρ drop is driven by V diverging from K,
by the key projection losing rank, or both.

Usage
-----
    python plot_depth_decomposition.py
    python plot_depth_decomposition.py --npz results/rho_results.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

GROUPS = {
    "OPT":   [("OPT-125M", "125M"), ("OPT-350M", "350M"), ("OPT-1.3B", "1.3B")],
    "GPT-2": [("GPT2-S", "GPT2-S"), ("GPT2-M", "GPT2-M"), ("GPT2-L", "GPT2-L")],
    "BERT":  [("BERT-base", "BERT-base"), ("BERT-large", "BERT-large")],
}

COLORS = ["#2166ac", "#d6604d", "#4dac26"]


def plot_family(ax_rho, ax_div, ax_rank, d, models, colors):
    for (key, label), color in zip(models, colors):
        rho     = d[key]                              # (layers, heads)
        diff    = d.get(f"{key}_diff_norm")           # (layers, heads) or None
        sr_k    = d.get(f"{key}_stable_rank_K")      # (layers, heads) or None

        layers = np.arange(rho.shape[0])

        # ρ
        mean_rho = rho.mean(axis=1)
        ax_rho.plot(layers, mean_rho, color=color, lw=2, marker="o",
                    markersize=3, label=label)
        ax_rho.fill_between(layers, mean_rho - rho.std(axis=1),
                            mean_rho + rho.std(axis=1), color=color, alpha=0.12)

        if diff is not None:
            mean_div = diff.mean(axis=1)
            ax_div.plot(layers, mean_div, color=color, lw=2, marker="o",
                        markersize=3, label=label)
            ax_div.fill_between(layers, mean_div - diff.std(axis=1),
                                mean_div + diff.std(axis=1), color=color, alpha=0.12)

        if sr_k is not None:
            mean_sr = sr_k.mean(axis=1)
            ax_rank.plot(layers, mean_sr, color=color, lw=2, marker="o",
                         markersize=3, label=label)
            ax_rank.fill_between(layers, mean_sr - sr_k.std(axis=1),
                                 mean_sr + sr_k.std(axis=1), color=color, alpha=0.12)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", default="results/rho_results.npz")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)

    for family, models in GROUPS.items():
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 7.5), sharex=True)
        ax_rho, ax_div, ax_rank = axes

        plot_family(ax_rho, ax_div, ax_rank, d, models, COLORS)

        ax_rho.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5,
                       label=r"$\rho=1$")
        ax_rho.set_ylabel(r"Mean $\rho^{(\ell)}$")
        ax_rho.set_ylim(0, 1.08)
        ax_rho.legend(framealpha=0.9, fontsize=8)
        ax_rho.set_title(f"{family} — depth decomposition of $\\rho$ decline")
        ax_rho.grid(axis="y", lw=0.4, alpha=0.5)

        ax_div.set_ylabel(r"Mean $\|W_V^{(\ell)}-W_K^{(\ell)}\|_F$")
        ax_div.legend(framealpha=0.9, fontsize=8)
        ax_div.grid(axis="y", lw=0.4, alpha=0.5)

        ax_rank.set_ylabel(r"Mean stable rank of $W_K^{(\ell)}$")
        ax_rank.set_xlabel("Layer index")
        ax_rank.legend(framealpha=0.9, fontsize=8)
        ax_rank.grid(axis="y", lw=0.4, alpha=0.5)

        fig.tight_layout()
        tag = family.lower().replace("-", "")
        out_pdf = f"results/depth_decomp_{tag}.pdf"
        out_png = f"results/depth_decomp_{tag}.png"
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved {out_png}")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
