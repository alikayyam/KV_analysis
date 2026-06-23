"""
Generate per-layer rho figures — one panel per model family.
Replaces the old gamma+rho bar charts with clean line plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

d = np.load("results/rho_results.npz", allow_pickle=True)

GROUPS = {
    "OPT":   [("OPT-125M", "125M"), ("OPT-350M", "350M"), ("OPT-1.3B", "1.3B")],
    "GPT-2": [("GPT-2 Small", "GPT2-S"), ("GPT-2 Medium", "GPT2-M"), ("GPT-2 Large", "GPT2-L")],
    "BERT":  [("BERT-base", "BERT-base"), ("BERT-large", "BERT-large")],
}

COLORS = ["#2166ac", "#d6604d", "#4dac26"]

for family, models in GROUPS.items():
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for (label, key), color in zip(models, COLORS):
        rho = d[key]                          # (layers, heads)
        mean = rho.mean(axis=1)               # per-layer mean
        std  = rho.std(axis=1)                # per-layer std across heads
        layers = np.arange(len(mean))

        ax.plot(layers, mean, color=color, linewidth=2, marker="o",
                markersize=4, label=label)
        ax.fill_between(layers, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="Condition C ($\\rho=1$)", alpha=0.6)

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$\rho^{(\ell)}$ (mean $\pm$ std across heads)")
    ax.set_title(f"{family} — Condition-C coverage per layer")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper right" if family != "OPT" else "lower left",
              framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_pdf = f"results/rho_per_layer_{family.lower().replace('-','')}.pdf"
    out_png = f"results/rho_per_layer_{family.lower().replace('-','')}.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")
    plt.close(fig)

print("Done.")
