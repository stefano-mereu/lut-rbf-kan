"""
ToN_IoT (Train_Test_Network.csv) preprocessing — mirrors the CICIDS pipeline
(preprocessing.py) exactly: balance, inf->NaN->median, IQR clip (3x),
StandardScaler, stratified 80/20 split with random_state=42.

Differences vs CICIDS, dictated by the dataset:
  - label column is 'type' (multiclass); binary task = 'normal' vs 'dos'
    (20k attack samples -> balanced 20k/20k, mirroring benign-vs-DoS-Hulk)
  - numeric features only (categoricals like proto/service/conn_state dropped
    in this first round, consistent with the CICIDS numeric-only pipeline)
  - 'ts' DROPPED: attacks are time-clustered in the UNSW testbed, so the
    timestamp is a label-leakage shortcut. IPs are excluded automatically
    (non-numeric) — essential, since ToN_IoT labels were assigned by tagging
    attacker IPs (UNSW documentation).
  - 'label' (binary ground truth) dropped from features, obviously.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_COL = "type"
BENIGN = "normal"
DROP_COLS = ["ts", "label"]  # leakage / ground-truth columns


def prepare_ton_data(filepath, attack_type="dos", max_samples_per_class=20000):
    print("Loading data...")
    df = pd.read_csv(filepath, low_memory=False)

    print("\nType distribution:")
    print(df[LABEL_COL].value_counts())

    max_samples = min(
        max_samples_per_class,
        df[df[LABEL_COL] == BENIGN].shape[0],
        df[df[LABEL_COL] == attack_type].shape[0],
    )
    print(f"\nUsing {max_samples} samples per class ({BENIGN} vs {attack_type})")

    benign = df[df[LABEL_COL] == BENIGN].sample(n=max_samples, random_state=42)
    attack = df[df[LABEL_COL] == attack_type].sample(n=max_samples, random_state=42)
    df = pd.concat([benign, attack])

    df["attack"] = (df[LABEL_COL] != BENIGN).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols
                    if c != "attack" and c not in DROP_COLS]
    print(f"Features ({len(numeric_cols)}): {numeric_cols}")

    df = df.replace([np.inf, -np.inf], np.nan)
    kept = []
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:  # clip only when IQR is non-degenerate: on sparse ToN
            # columns q1=q3=0 and clipping would collapse them to constants
            df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)
        if df[col].std() > 1e-9:  # drop residual constant columns
            kept.append(col)
    dropped = [c for c in numeric_cols if c not in kept]
    if dropped:
        print(f"Dropped constant columns ({len(dropped)}): {dropped}")
    numeric_cols = kept

    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols]).astype(np.float32)
    # Winsorize extreme outliers in scaled space (sparse heavy-tailed columns
    # survive the degenerate-IQR skip but can reach ~200 sigma; uniform LUT
    # knots over such a range would leave no resolution where the data lives)
    X = np.clip(X, -10.0, 10.0)
    y = df["attack"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dataset = {
        "train_input": torch.from_numpy(X_train),
        "train_label": torch.from_numpy(y_train).reshape(-1, 1),
        "test_input": torch.from_numpy(X_test),
        "test_label": torch.from_numpy(y_test).reshape(-1, 1),
    }

    domain_stats = {
        "global_min": float(X.min()),
        "global_max": float(X.max()),
        "p01": float(np.percentile(X, 1)),
        "p99": float(np.percentile(X, 99)),
    }

    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")
    print(f"Post-scaler global range: [{domain_stats['global_min']:.2f}, "
          f"{domain_stats['global_max']:.2f}]")
    print(f"Post-scaler [p01, p99]:   [{domain_stats['p01']:.2f}, "
          f"{domain_stats['p99']:.2f}]")

    return dataset, scaler, numeric_cols, domain_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_ton/Train_Test_Network.csv")
    ap.add_argument("--attack", default="dos")
    ap.add_argument("--out", default="ton_data")
    ap.add_argument("--max_per_class", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset, scaler, features, domain_stats = prepare_ton_data(
        args.csv, attack_type=args.attack,
        max_samples_per_class=args.max_per_class,
    )

    torch.save(dataset, out / "dataset.pt")
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out / "features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(out / "domain_stats.pkl", "wb") as f:
        pickle.dump(domain_stats, f)

    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
