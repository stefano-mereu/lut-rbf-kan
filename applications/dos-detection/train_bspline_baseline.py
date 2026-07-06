"""
B-spline KAN baseline on CICIDS2017 DoS Hulk.

Loads the preprocessed dataset (dos_data/dataset.pt), optionally subsamples
for speed, trains PyKAN [78,32,16,1] grid=5 k=3, and reports precision/recall/F1
using sklearn (same convention as Kuznetsov's analyze.py).

Mini-batch training to avoid the full-batch OOM on large sample counts.

Target (Kuznetsov arXiv:2502.01835): Acc 0.990, P 0.984, R 0.996, F1 0.990
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from kan import KAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def subsample(x, y, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    y_flat = y.numpy().ravel()
    idx0 = np.where(y_flat == 0)[0]
    idx1 = np.where(y_flat == 1)[0]
    n = min(n_per_class, len(idx0), len(idx1))
    sel = np.concatenate([
        rng.choice(idx0, n, replace=False),
        rng.choice(idx1, n, replace=False),
    ])
    rng.shuffle(sel)
    return x[sel], y[sel]


@torch.no_grad()
def evaluate(model, x, y, batch=8192):
    model.eval()
    logits = []
    for i in range(0, len(x), batch):
        logits.append(model(x[i:i+batch]))
    logits = torch.cat(logits)
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    preds = (probs > 0.5).astype(int)
    yt = y.cpu().numpy().ravel().astype(int)
    return dict(
        acc=accuracy_score(yt, preds),
        precision=precision_score(yt, preds),
        recall=recall_score(yt, preds),
        f1=f1_score(yt, preds),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dos_data/dataset.pt")
    ap.add_argument("--n_per_class", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="dos_data/bspline_model")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    d = torch.load(args.data)
    x_tr, y_tr = d["train_input"], d["train_label"]
    x_te, y_te = d["test_input"], d["test_label"]

    if args.n_per_class > 0:
        x_tr, y_tr = subsample(x_tr, y_tr, args.n_per_class)
        print(f"Subsampled train to {len(x_tr)} ({args.n_per_class}/class)")

    x_tr, y_tr = x_tr.to(args.device), y_tr.to(args.device)
    x_te, y_te = x_te.to(args.device), y_te.to(args.device)

    input_dim = x_tr.shape[1]
    print(f"Input dim {input_dim}  Train {tuple(x_tr.shape)}  Test {tuple(x_te.shape)}")

    model = KAN(width=[input_dim, 32, 16, 1], grid=args.grid, k=args.k,
                seed=42, device=args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss()

    n = len(x_tr)
    print(f"\nTraining ({args.epochs} epochs, batch {args.batch})...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            opt.zero_grad()
            loss = crit(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0:
            m = evaluate(model, x_te, y_te)
            print(f"Epoch {epoch+1:3d}  loss={total/n:.4f}  "
                  f"acc={m['acc']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}")

    m = evaluate(model, x_te, y_te)
    print("\n=== Final test metrics ===")
    print(f"Accuracy  {m['acc']:.4f}")
    print(f"Precision {m['precision']:.4f}")
    print(f"Recall    {m['recall']:.4f}")
    print(f"F1        {m['f1']:.4f}")
    print("Kuznetsov: Acc 0.990 P 0.984 R 0.996 F1 0.990")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "width": [input_dim, 32, 16, 1], "grid": args.grid, "k": args.k,
        "final_metrics": m,
    }, out / "bspline_kan.pt")
    print(f"\nSaved to {out}/bspline_kan.pt")


if __name__ == "__main__":
    main()
