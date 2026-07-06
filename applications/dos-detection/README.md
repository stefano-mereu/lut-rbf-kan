# DoS Detection: RBF-KAN + Hermite LUT vs B-spline

Application of the RBF-KAN + Hermite LUT extension to network intrusion
detection (CICIDS2017 DoS Hulk), extending Kuznetsov (2025) from B-spline to
Gaussian RBF edges with cubic Hermite LUT interpolation.

## Result summary

Multi-layer KAN [78, 32, 16, 1], balanced BENIGN vs DoS Hulk.

| Model             | float F1 |
|-------------------|----------|
| B-spline (PyKAN)  | 0.9985   |
| RBF-KAN (ours)    | 0.9982   |

LUT compilation, memory-constrained regime (K=8):

| interp  | L | mem/edge | F1     | MAE_logit |
|---------|---|----------|--------|-----------|
| linear  | 2 | 16 B     | 0.9002 | 5.686     |
| hermite | 2 | 32 B     | 0.9969 | 0.275     |
| linear  | 4 | 32 B     | 0.9975 | 0.283     |
| hermite | 4 | 64 B     | 0.9979 | 0.006     |

At L=2 linear collapses (F1 0.90) while Hermite holds (F1 0.997): Hermite
enables a memory budget where linear is unusable. On pre-threshold logits
Hermite reconstructs 40-50x more accurately at equal L.

## Pipeline

Requires the CICIDS2017 `Wednesday-workingHours.pcap_ISCX.csv` in `data/`
(download from the Canadian Institute for Cybersecurity, or Kaggle mirror).

\`\`\`bash
# 1. Preprocess (replicates Kuznetsov: balance, clean, IQR clip, StandardScaler, 80/20)
python preprocessing.py --csv data/Wednesday-workingHours.pcap_ISCX.csv --out dos_data

# 2. Train B-spline baseline (PyKAN)
python train_bspline_baseline.py --data dos_data/dataset.pt --device cuda

# 3. Train multi-layer RBF-KAN
python train_rbf.py --data dos_data/dataset.pt --device cuda

# 4. Compile to LUT and evaluate (linear vs Hermite sweep)
python lut_eval_rbf.py --K 8 --L_list 2 4 8 16
\`\`\`

## Notes

- The RBF layer has centers on [0,1]; a frozen min-max normalization between
  layers keeps each layer's input in that domain (see \`rbf_multilayer.py\`).
- The non-monotonic Hermite MAE at large L is the int8 derivative quantization
  floor, not an algorithmic effect (\`diag_hermite_quant.py\` verifies this).
