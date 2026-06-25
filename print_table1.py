"""
Print Table 1 statistics: mean ρ, min ρ, mean ‖M‖_F, mean √(1-ρ²)‖M‖_F.
Run after compute_rho.py has been executed.
"""

import numpy as np

NPZ = "results/rho_results.npz"

ROWS = [
    ("OPT-125M",   "OPT-125M"),
    ("OPT-350M",   "OPT-350M"),
    ("OPT-1.3B",   "OPT-1.3B"),
    ("GPT2-S",     "GPT-2 Small"),
    ("GPT2-M",     "GPT-2 Medium"),
    ("GPT2-L",     "GPT-2 Large"),
    ("BERT-base",  "BERT-base"),
    ("BERT-large", "BERT-large"),
]

d = np.load(NPZ, allow_pickle=True)

header = f"{'Model':<14} {'Mean ρ':>8} {'Min ρ':>8} {'Mean ‖M‖_F':>12} {'Mean √(1-ρ²)‖M‖_F':>20}"
print(header)
print("-" * len(header))

for key, label in ROWS:
    if key not in d:
        print(f"{label:<14}  (not computed)")
        continue

    rho    = d[key]                              # (layers, heads)
    M_norm = d.get(f"{key}_M_norm", None)        # (layers, heads) or None

    mean_rho = float(rho.mean())
    min_rho  = float(rho.min())

    if M_norm is not None:
        mean_M   = float(M_norm.mean())
        product  = np.sqrt(np.clip(1.0 - rho**2, 0, None)) * M_norm
        mean_prd = float(product.mean())
    else:
        mean_M   = float("nan")
        mean_prd = float("nan")

    print(f"{label:<14} {mean_rho:>8.4f} {min_rho:>8.4f} {mean_M:>12.4f} {mean_prd:>20.4f}")
