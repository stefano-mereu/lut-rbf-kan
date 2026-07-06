"""
Train multi-layer RBF-KAN on CICIDS2017 DoS Hulk.

Mirrors train_bspline_baseline.py (same data, metrics, mini-batch, GPU) but for
the RBFKANMultiLayer model. Key extra steps:
  - init_first_norm() fixes layer-0 minmax from real training features
  - freeze() locks norm stats + h before saving (ready for LUT compilation)

Goal: reach F1 ~0.99 to confirm the multi-layer RBF approach works on a real
task, matching the B-spline baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from rbf_multilayer import RBFKANMultiLayer


def subsample(x, y, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    yf = y.numpy().ravel()
    i0 = np.where(yf == 0)[0]
    i1 = np.where(yf == 1)[0]
    n = min(n_per_class, len(i0), len(i1))
    sel = np.concatenate([rng.choice(i0, n, False), rng.choice(i1, n, False)])
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
        precision=precision_score(yt, preds, zero_division=0),
        recall=recall_score(yt, preds, zero_division=0),
        f1=f1_score(yt, preds, zero_division=0),
    ), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dos_data/dataset.pt")
    ap.add_argument("--n_per_class", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--out", default="dos_data/rbf_model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    d = torch.load(args.data)
    x_tr, y_tr = d["train_input"], d["train_label"]
    x_te, y_te = d["test_input"], d["test_label"]

    if args.n_per_class > 0:
        x_tr, y_tr = subsample(x_tr, y_tr, args.n_per_class)
        print(f"Subsampled train to {len(x_tr)}")

    x_tr, y_tr = x_tr.to(args.device), y_tr.to(args.device)
    x_te, y_te = x_te.to(args.device), y_te.to(args.device)

    width = [x_tr.shape[1], 32, 16, 1]
    print(f"Architecture {width}  G={args.G}  Train {tuple(x_tr.shape)}")

    model = RBFKANMultiLayer(width, G=args.G).to(args.device)

    # Fix first-layer minmax from real features (use a large sample)
    model.init_first_norm(x_tr[:20000])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss()

    n = len(x_tr)
    print(f"\nTraining ({args.epochs} epochs, batch {args.batch})...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=args.device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            opt.zero_grad()
            loss = crit(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0:
            m, _ = evaluate(model, x_te, y_te)
            h_vals = [f"{blk.rbf.get_h():.3f}" for blk in model.blocks]
            print(f"Epoch {epoch+1:3d}  loss={total/n:.4f}  "
                  f"acc={m['acc']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}  h={h_vals}")

    m, _ = evaluate(model, x_te, y_te)
    print("\n=== Final test metrics ===")
    print(f"Accuracy  {m['acc']:.4f}")
    print(f"Precision {m['precision']:.4f}")
    print(f"Recall    {m['recall']:.4f}")
    print(f"F1        {m['f1']:.4f}")
    print("B-spline baseline: F1 0.9985  |  Kuznetsov: F1 0.990")

    # Freeze norm stats + h for LUT compilation, then save
    model.freeze()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "width": width, "G": args.G,
        "final_metrics": m,
    }, out / "rbf_kan.pt")
    print(f"\nSaved to {out}/rbf_kan.pt")


if __name__ == "__main__":
    main()
