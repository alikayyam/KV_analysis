"""
Experiment 3 — Full fine-tuning under K=V constraint.

Protocol:
  1. Load pretrained OPT model (untied baseline).
  2. Build tied model: share W_V = W_K as the same Parameter object.
     Hard K=V constraint maintained throughout training via gradient
     sharing — no post-step projection needed.  No FFN expansion.
  3. Fine-tune ALL parameters (W_Q, W_KV, W_O, FFN) on WikiText-103.
  4. Checkpoint test PPL every --eval-interval steps.
  5. Measure layer-wise γ on 16 held-out validation sequences.

Tests Theorem 2 (revised):
  Part (i)  — full fine-tuning recovers near-untied PPL regardless of ρ.
  Part (ii) — convergence speed scales with (1−ρ²): compare 125M vs 350M
              PPL curves; ratio of steps-to-threshold should ≈ (1−ρ²_350)
              / (1−ρ²_125) = 0.542 / 0.183 ≈ 3.0.

Outputs
-------
  results/exp3/<tag>_tied_full_ft.pt       fine-tuned model state-dict
  results/exp3/<tag>_ppl_curve.npz         steps, ppls arrays
  results/exp3_<tag>_results.npz           gamma, ppl_curve, perplexities

Usage
-----
  python experiment3.py                            # OPT-125M, 20k steps
  python experiment3.py --model facebook/opt-350m
  python experiment3.py --steps 200               # quick smoke-test
  python experiment3.py --force                   # ignore cached checkpoint
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "facebook/opt-125m"


# ── build tied model via parameter sharing ────────────────────────────────────

def build_tied_model(model_id: str) -> nn.Module:
    """
    Load pretrained weights, then make v_proj.weight and k_proj.weight
    the same Parameter object in every attention layer.

    After this:
      · attn.v_proj.weight IS attn.k_proj.weight (same id)
      · Gradients from both key and value paths accumulate in one tensor
      · model.parameters() deduplicates → k_proj.weight appears once
      · The K=V constraint is maintained automatically throughout training

    No FFN expansion — original width only.  All parameters trainable.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    for layer in model.model.decoder.layers:
        attn = layer.self_attn

        with torch.no_grad():
            attn.v_proj.weight.copy_(attn.k_proj.weight)
            if attn.v_proj.bias is not None and attn.k_proj.bias is not None:
                attn.v_proj.bias.copy_(attn.k_proj.bias)

        # nn.Module.__setattr__ calls register_parameter, replacing
        # v_proj._parameters['weight'] with the k_proj Parameter object.
        attn.v_proj.weight = attn.k_proj.weight
        if attn.v_proj.bias is not None and attn.k_proj.bias is not None:
            attn.v_proj.bias = attn.k_proj.bias

    return model


# ── layer output extraction ───────────────────────────────────────────────────

@torch.no_grad()
def get_layer_outputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> dict[int, np.ndarray]:
    model.eval()
    model.to(device)
    input_ids = input_ids.to(device)

    outs: dict[int, np.ndarray] = {}
    hooks = []

    def _make_hook(l: int):
        def hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            outs[l] = hs.detach().cpu().float().numpy()
        return hook

    for l, layer in enumerate(model.model.decoder.layers):
        hooks.append(layer.register_forward_hook(_make_hook(l)))

    model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return outs


# ── γ ─────────────────────────────────────────────────────────────────────────

def compute_gamma(
    outs_untied:  dict[int, np.ndarray],
    outs_tied_nt: dict[int, np.ndarray],
    outs_tied_ft: dict[int, np.ndarray],
) -> np.ndarray:
    """
    γ^(ℓ) = 1 − ‖T2_ft^(ℓ) − T1^(ℓ)‖_F / ‖T2_nt^(ℓ) − T1^(ℓ)‖_F
    γ = 1 → full recovery; γ = 0 → no improvement over no-tuning baseline.
    """
    num_layers = len(outs_untied)
    gamma = np.zeros(num_layers)
    for l in range(num_layers):
        denom = np.linalg.norm((outs_tied_nt[l] - outs_untied[l]).ravel())
        numer = np.linalg.norm((outs_tied_ft[l] - outs_untied[l]).ravel())
        gamma[l] = 1.0 if denom < 1e-12 else float(1.0 - numer / denom)
    return gamma


# ── dataset ───────────────────────────────────────────────────────────────────

def prepare_dataloaders(
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    def tokenize(batch):
        return tokenizer(batch["text"], return_attention_mask=False, truncation=False)

    def chunk(batch):
        ids   = sum(batch["input_ids"], [])
        total = (len(ids) // seq_len) * seq_len
        return {"input_ids": [ids[i : i + seq_len] for i in range(0, total, seq_len)]}

    def collate(batch):
        return {"input_ids": torch.stack([b["input_ids"] for b in batch])}

    cols    = [c for c in ds["train"].column_names if c != "input_ids"]
    tok_ds  = ds.map(tokenize, batched=True, remove_columns=cols)
    chunked = tok_ds.map(chunk, batched=True)
    chunked.set_format("torch")

    train_loader = DataLoader(chunked["train"],      batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(chunked["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(chunked["test"],       batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader


# ── perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    model.to(device)
    total_nll, total_tok = 0.0, 0

    for batch in tqdm(loader, desc="PPL", leave=False):
        ids = batch["input_ids"].to(device)
        loss = model(input_ids=ids, labels=ids).loss
        n = ids.numel()
        total_nll += loss.item() * n
        total_tok += n

    return math.exp(total_nll / total_tok)


# ── full fine-tuning with PPL checkpoints ─────────────────────────────────────

def finetune_full(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_steps: int,
    lr: float,
    eval_interval: int,
    device: torch.device,
) -> tuple[nn.Module, list[tuple[int, float]]]:
    """
    Fine-tune all parameters with K=V constraint maintained by parameter
    sharing.  Returns (model, ppl_curve) where ppl_curve = [(step, ppl), ...].
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    ppl_curve: list[tuple[int, float]] = []
    data_iter = iter(train_loader)
    pbar = tqdm(range(1, num_steps + 1), desc="fine-tuning (full)")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        loss = model(input_ids=input_ids, labels=input_ids).loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % eval_interval == 0:
            ppl = compute_perplexity(model, test_loader, device)
            ppl_curve.append((step, ppl))
            print(f"\n  step {step:5d}: test PPL = {ppl:.3f}")
            model.train()
            model.to(device)

    model.eval()
    return model, ppl_curve


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",         default=MODEL_ID)
    parser.add_argument("--steps",         type=int,   default=20_000,
                        help="fine-tuning steps (default 20000)")
    parser.add_argument("--lr",            type=float, default=1e-5,
                        help="learning rate (default 1e-5; lower than Exp 2 "
                             "because all attention weights are live)")
    parser.add_argument("--seq-len",       type=int,   default=512)
    parser.add_argument("--batch",         type=int,   default=4)
    parser.add_argument("--eval-interval", type=int,   default=2000,
                        help="checkpoint test PPL every N steps (default 2000)")
    parser.add_argument("--eval-seqs",     type=int,   default=16,
                        help="validation sequences for γ (default 16)")
    parser.add_argument("--device",        default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out",           default=None,
                        help="results .npz path")
    parser.add_argument("--ckpt",          default=None,
                        help="model checkpoint .pt path")
    parser.add_argument("--force",         action="store_true",
                        help="ignore cached checkpoint and rerun")
    args = parser.parse_args()

    tag = args.model.split("/")[-1]
    if args.out  is None: args.out  = f"results/exp3_{tag}_results.npz"
    if args.ckpt is None: args.ckpt = f"results/exp3/{tag}_tied_full_ft.pt"

    curve_path = args.ckpt.replace(".pt", "_curve.npz")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  |  model: {args.model}  |  tag: {tag}")

    os.makedirs(os.path.dirname(args.out)  or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    print("\nLoading tokeniser and WikiText-103 …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        tokenizer, seq_len=args.seq_len, batch_size=args.batch
    )

    eval_ids_list = []
    for batch in val_loader:
        eval_ids_list.append(batch["input_ids"])
        if sum(t.shape[0] for t in eval_ids_list) >= args.eval_seqs:
            break
    eval_ids = torch.cat(eval_ids_list, dim=0)[: args.eval_seqs]
    print(f"Eval batch shape: {tuple(eval_ids.shape)}")

    # ── Step 1: untied baseline ────────────────────────────────────────────────
    print("\n── Step 1: Untied model ──────────────────────────────")
    untied = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    outs_untied = get_layer_outputs(untied, eval_ids, device)
    ppl_untied  = compute_perplexity(untied, test_loader, device)
    print(f"  PPL (untied): {ppl_untied:.3f}")
    del untied
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 2: tied, no tuning ────────────────────────────────────────────────
    print("\n── Step 2: Tied model (no tuning) ───────────────────")
    tied = build_tied_model(args.model)
    outs_tied_nt = get_layer_outputs(tied, eval_ids, device)
    ppl_tied_nt  = compute_perplexity(tied, test_loader, device)
    print(f"  PPL (tied, no tuning): {ppl_tied_nt:.3f}")

    # sanity-check: v_proj.weight IS k_proj.weight after build
    layer0_attn = tied.model.decoder.layers[0].self_attn
    assert layer0_attn.v_proj.weight is layer0_attn.k_proj.weight, \
        "Parameter sharing not established — check build_tied_model()"
    print("  [ok] parameter sharing verified (v_proj.weight is k_proj.weight)")

    # ── Step 3: full fine-tuning ───────────────────────────────────────────────
    print("\n── Step 3: Full fine-tuning ─────────────────────────")
    ppl_curve: list[tuple[int, float]] = []

    if os.path.exists(args.ckpt) and not args.force:
        print(f"  Loading checkpoint from {args.ckpt}")
        tied.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
        if os.path.exists(curve_path):
            data = np.load(curve_path)
            ppl_curve = list(zip(data["steps"].tolist(), data["ppls"].tolist()))
            print(f"  PPL curve loaded ({len(ppl_curve)} checkpoints)")
    else:
        tied, ppl_curve = finetune_full(
            tied, train_loader, test_loader,
            num_steps=args.steps,
            lr=args.lr,
            eval_interval=args.eval_interval,
            device=device,
        )
        tied.cpu()
        torch.save(tied.state_dict(), args.ckpt)
        np.savez(
            curve_path,
            steps=np.array([s for s, _ in ppl_curve]),
            ppls =np.array([p for _, p in ppl_curve]),
        )
        print(f"  Checkpoint saved → {args.ckpt}")
        print(f"  PPL curve saved  → {curve_path}")

    ppl_tied_ft = (
        ppl_curve[-1][1] if ppl_curve
        else compute_perplexity(tied, test_loader, device)
    )
    print(f"  PPL (tied, full fine-tuned): {ppl_tied_ft:.3f}")

    # ── Step 4: γ per layer ───────────────────────────────────────────────────
    print("\n── Step 4: γ per layer ───────────────────────────────")
    outs_tied_ft = get_layer_outputs(tied, eval_ids, device)
    gamma = compute_gamma(outs_untied, outs_tied_nt, outs_tied_ft)
    for l, g in enumerate(gamma):
        print(f"  layer {l:2d}: γ = {g:+.4f}")
    print(f"  Mean γ = {gamma.mean():.4f}")

    # ── Step 5: convergence curve ─────────────────────────────────────────────
    print("\n── PPL convergence curve ─────────────────────────────")
    print(f"  {'step':>6}  {'PPL':>10}  {'gap × untied':>14}")
    for step, ppl in ppl_curve:
        print(f"  {step:6d}  {ppl:10.3f}  {ppl / ppl_untied:14.2f}×")

    # ── save ──────────────────────────────────────────────────────────────────
    np.savez(
        args.out,
        gamma           = gamma,
        ppl_curve_steps = np.array([s for s, _ in ppl_curve]),
        ppl_curve_ppls  = np.array([p for _, p in ppl_curve]),
        ppl_untied      = np.array([ppl_untied]),
        ppl_tied_nt     = np.array([ppl_tied_nt]),
        ppl_tied_ft     = np.array([ppl_tied_ft]),
    )
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
