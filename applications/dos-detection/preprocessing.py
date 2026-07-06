"""
CICIDS2017 DoS Hulk preprocessing — exact replication of Kuznetsov (2025)
prepare_dos_data, with dataset caching and RBF-domain statistics.

Reference: KuznetsovKarazin/kan-dos-detection, src/train.py

Pipeline (identical to Kuznetsov):
  1. Balance BENIGN vs DoS Hulk (max_samples_per_class, random_state=42)
  2. Binary target: attack = (Label != BENIGN)
  3. Replace inf/-inf with NaN, fill NaN with column median
  4. Clip outliers with IQR rule [Q1 - 3*IQR, Q3 + 3*IQR]
  5. StandardScaler (zero mean, unit variance)
  6. Stratified 80/20 split, random_state=42

Additions for our RBF work:
  - Cache processed tensors to disk (.pt)
  - Save post-StandardScaler per-feature range (for RBF center calibration)
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

LABEL_COL = " Label"  # note leading space in CICIDS2017 CSVs


def prepare_dos_data(filepath, attack_type="DoS Hulk", max_samples_per_class=231073):
    print("Loading data...")
    df = pd.read_csv(filepath)

    print("\nLabel distribution:")
    print(df[LABEL_COL].value_counts())

    # Balance classes
    max_samples = min(
        max_samples_per_class,
        df[df[LABEL_COL] == "BENIGN"].shape[0],
        df[df[LABEL_COL] == attack_type].shape[0],
    )
    print(f"\nUsing {max_samples} samples per class")

    benign = df[df[LABEL_COL] == "BENIGN"].sample(n=max_samples, random_state=42)
    attack = df[df[LABEL_COL] == attack_type].sample(n=max_samples, random_state=42)
    df = pd.concat([benign, attack])

    df["attack"] = (df[LABEL_COL] != "BENIGN").astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "attack"]

    # Clean: inf -> NaN -> median; clip IQR outliers
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols]).astype(np.float32)
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

    # Post-StandardScaler range (for RBF center calibration)
    domain_stats = {
        "x_min": X.min(axis=0),
        "x_max": X.max(axis=0),
        "x_mean": X.mean(axis=0),
        "x_std": X.std(axis=0),
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
    ap.add_argument("--csv", default="kan-dos-detection/data/Wednesday-workingHours.pcap_ISCX.csv")
    ap.add_argument("--out", default="dos_data")
    ap.add_argument("--max_per_class", type=int, default=231073)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset, scaler, features, domain_stats = prepare_dos_data(
        args.csv, attack_type="DoS Hulk", max_samples_per_class=args.max_per_class
    )

    torch.save(dataset, out / "dataset.pt")
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out / "features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(out / "domain_stats.pkl", "wb") as f:
        pickle.dump(domain_stats, f)

    print(f"\nSaved to {out}/")
    print(f"  dataset.pt, scaler.pkl, features.pkl ({len(features)} features), domain_stats.pkl")


if __name__ == "__main__":
    main()
