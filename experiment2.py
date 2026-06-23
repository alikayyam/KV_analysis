"""
Experiment 2 — Can the FFN learn the correction?

Protocol (per the paper):
  1. Load pretrained OPT-125M (untied baseline).
  2. Build tied model: W_V = W_K for all heads, freeze all attention weights,
     expand each FFN by 2d = 1536 hidden neurons initialised to zero.
  3. Fine-tune only the FFN weights on WikiText-2 (train split).
  4. Measure layer-wise γ on a held-out validation batch.
  5. Compute perplexity for untied / tied-no-tuning / tied-fine-tuned.

Outputs
-------
  results/exp2/tied_ft.pt       fine-tuned model state-dict
  results/exp2_results.npz      gamma, perplexities

Usage
-----
  python experiment2.py                   # auto device, 2000 steps
  python experiment2.py --steps 200       # quick smoke-test
  python experiment2.py --device cpu
  python experiment2.py --force           # ignore cached checkpoint
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


# ── build tied model ──────────────────────────────────────────────────────────

def _expand_ffn(layer: nn.Module, extra: int) -> None:
    """
    Replace layer.fc1 / layer.fc2 with wider versions (extra new neurons = 0).
    Preserves original weights exactly; new weights are zero-initialised.
    """
    old1, old2 = layer.fc1, layer.fc2
    d     = old1.in_features
    h_old = old1.out_features
    h_new = h_old + extra
    has_b1 = old1.bias is not None
    has_b2 = old2.bias is not None

    new1 = nn.Linear(d,     h_new, bias=has_b1)
    new2 = nn.Linear(h_new, d,     bias=has_b2)

    with torch.no_grad():
        new1.weight[:h_old] = old1.weight
        new1.weight[h_old:] = 0.0
        if has_b1:
            new1.bias[:h_old] = old1.bias
            new1.bias[h_old:] = 0.0

        new2.weight[:, :h_old] = old2.weight
        new2.weight[:, h_old:] = 0.0
        if has_b2:
            new2.bias.copy_(old2.bias)

    layer.fc1 = new1
    layer.fc2 = new2


def build_tied_model(model_id: str) -> nn.Module:
    """
    Load pretrained weights, then:
      · copy W_V ← W_K for every head (and bias)
      · freeze all attention parameters
      · expand each FFN by 2d neurons (zero-init)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    d = model.config.hidden_size

    for layer in model.model.decoder.layers:
        attn = layer.self_attn

        with torch.no_grad():
            attn.v_proj.weight.copy_(attn.k_proj.weight)
            if attn.v_proj.bias is not None:
                attn.v_proj.bias.copy_(attn.k_proj.bias)

        for param in attn.parameters():
            param.requires_grad = False

        _expand_ffn(layer, extra=2 * d)

    return model


# ── layer output extraction ───────────────────────────────────────────────────

@torch.no_grad()
def get_layer_outputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """
    Forward pass with hooks; returns post-block hidden states per layer.
    Shape per entry: (batch, seq, d) float32 numpy.
    """
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

    Returns array shape (num_layers,).  γ = 1 means full recovery.
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

    cols   = [c for c in ds["train"].column_names if c != "input_ids"]
    tok_ds = ds.map(tokenize, batched=True, remove_columns=cols)
    chunked = tok_ds.map(chunk, batched=True)
    chunked.set_format("torch")

    train_loader = DataLoader(chunked["train"],      batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(chunked["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(chunked["test"],       batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader


# ── fine-tuning ───────────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    num_steps: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    """Fine-tune only fc1 / fc2 in every decoder layer."""
    model.train()
    model.to(device)

    ffn_params = []
    for layer in model.model.decoder.layers:
        ffn_params += list(layer.fc1.parameters())
        ffn_params += list(layer.fc2.parameters())

    optimizer = torch.optim.AdamW(ffn_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    data_iter = iter(train_loader)
    pbar = tqdm(range(1, num_steps + 1), desc="fine-tuning FFN")

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
        nn.utils.clip_grad_norm_(ffn_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    return model


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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",      default=MODEL_ID)
    parser.add_argument("--steps",      type=int,   default=2000,
                        help="fine-tuning steps (default 2000)")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--seq-len",    type=int,   default=512)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--eval-seqs",  type=int,   default=16,
                        help="val sequences for γ computation (default 16)")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out",        default=None,
                        help="output .npz path (default: results/exp2_<tag>_results.npz)")
    parser.add_argument("--ckpt",       default=None,
                        help="checkpoint path (default: results/exp2/<tag>_tied_ft.pt)")
    parser.add_argument("--force",      action="store_true",
                        help="ignore cached checkpoint and rerun fine-tuning")
    args = parser.parse_args()

    # derive a short tag from the model ID, e.g. "facebook/opt-350m" → "opt-350m"
    tag = args.model.split("/")[-1]
    if args.out  is None: args.out  = f"results/exp2_{tag}_results.npz"
    if args.ckpt is None: args.ckpt = f"results/exp2/{tag}_tied_ft.pt"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  |  model: {args.model}  |  tag: {tag}")

    os.makedirs(os.path.dirname(args.out)  or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    # ── data ──
    print("\nLoading tokeniser and WikiText-2 …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        tokenizer, seq_len=args.seq_len, batch_size=args.batch
    )

    # fixed eval batch (validation split) for γ
    eval_ids_list = []
    for batch in val_loader:
        eval_ids_list.append(batch["input_ids"])
        if sum(t.shape[0] for t in eval_ids_list) >= args.eval_seqs:
            break
    eval_ids = torch.cat(eval_ids_list, dim=0)[: args.eval_seqs]
    print(f"Eval batch shape: {tuple(eval_ids.shape)}")

    # ── Step 1: untied baseline ──
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

    # ── Step 2: tied, no tuning ──
    print("\n── Step 2: Tied model (no tuning) ───────────────────")
    tied = build_tied_model(args.model)
    outs_tied_nt = get_layer_outputs(tied, eval_ids, device)
    ppl_tied_nt  = compute_perplexity(tied, test_loader, device)
    print(f"  PPL (tied, no tuning): {ppl_tied_nt:.3f}")

    # ── Step 3: fine-tune FFN ──
    print("\n── Step 3: Fine-tune FFN ─────────────────────────────")
    if os.path.exists(args.ckpt) and not args.force:
        print(f"  Loading checkpoint from {args.ckpt}")
        tied.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    else:
        tied = finetune(tied, train_loader, args.steps, args.lr, device)
        tied.cpu()
        torch.save(tied.state_dict(), args.ckpt)
        print(f"  Checkpoint saved → {args.ckpt}")

    outs_tied_ft = get_layer_outputs(tied, eval_ids, device)
    ppl_tied_ft  = compute_perplexity(tied, test_loader, device)
    print(f"  PPL (tied, fine-tuned): {ppl_tied_ft:.3f}")
    del tied
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 4: compute γ ──
    print("\n── Step 4: γ per layer ───────────────────────────────")
    gamma = compute_gamma(outs_untied, outs_tied_nt, outs_tied_ft)
    for l, g in enumerate(gamma):
        print(f"  layer {l:2d}: γ = {g:+.4f}")

    np.savez(
        args.out,
        gamma       = gamma,
        ppl_untied  = np.array([ppl_untied]),
        ppl_tied_nt = np.array([ppl_tied_nt]),
        ppl_tied_ft = np.array([ppl_tied_ft]),
    )
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
