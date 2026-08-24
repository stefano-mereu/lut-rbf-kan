# DoS Detection: RBF-KAN + Hermite LUT vs B-spline

Application of the RBF-KAN + Hermite LUT extension to network intrusion
detection, extending Kuznetsov (2025) from B-spline to Gaussian RBF edges
with cubic Hermite LUT interpolation. Two datasets: CICIDS2017 (DoS Hulk)
and ToN_IoT (dos).

## Preprocessing (leak-free)

All statistics (median imputation, IQR bounds, scaler, constant-column
checks) are computed on the TRAIN split only and applied to test
(`preprocessing.py`, `preprocessing_ton.py`). ToN_IoT specifics: IQR clip
skipped when IQR is degenerate (sparse columns), +-10 sigma winsorization
in scaled space, `ts` dropped (time-clustered attacks = leakage), IPs
excluded (labels were assigned by tagging attacker IPs).

## Memory accounting (measured)

The `mem/edge` columns are MEASURED artifact bytes, not a formula. The
artifact stores: value table (K*L uint8), derivative table (K*L int8,
hermite only — stripped for linear), and per-segment quantization metadata
in float16 (scale + y_min = 4 bytes/segment; hermite adds dscale =
2 bytes/segment). `dy_min` is not stored (always zero under symmetric
derivative quantization). Closed form: linear = K(L+4), hermite = K(2L+6).

## Calibration and runtime

B-spline LUT domains are calibrated on a frozen 20k TRAIN subset (no test
statistics anywhere in the pipeline). B-spline Hermite derivatives use
central finite differences with eps=1e-3, verified stable across
eps in [1e-2, 1e-3] (relative agreement 4e-4; fp32 cancellation onset at
1e-4). Runtime (NumPy backend, indicative of relative interpolation cost,
not deployment C timing): linear 5.6 us/sample, Hermite 9.2 us/sample
(1.65x) on ToN, K=8 L=4 -- the memory-at-target advantage is bought with
~1.65x per-sample interpolation cost.

## Float models

| Dataset   | B-spline (PyKAN) | RBF-KAN (ours) |
|-----------|------------------|----------------|
| CICIDS2017| 0.9985           | 0.9976         |
| ToN_IoT   | 0.9934           | 0.9956         |

## CICIDS2017, K=8 (measured bytes)

| base     | interp  | L | mem/edge | F1     | MAE_logit |
|----------|---------|---|----------|--------|-----------|
| B-spline | linear  | 2 | 48 B     | 0.7839 | 7.943     |
| RBF      | linear  | 2 | 48 B     | 0.9898 | 2.318     |
| B-spline | hermite | 2 | 80 B     | 0.7925 | 6.235     |
| RBF      | hermite | 2 | 80 B     | 0.9978 | 0.155     |
| B-spline | linear  | 4 | 64 B     | 0.8296 | 3.591     |
| RBF      | linear  | 4 | 64 B     | 0.9978 | 0.186     |
| B-spline | hermite | 4 | 112 B    | 0.9855 | 0.905     |
| RBF      | hermite | 4 | 112 B    | 0.9977 | 0.007     |
| B-spline | linear  | 8 | 96 B     | 0.9139 | 1.203     |
| RBF      | linear  | 8 | 96 B     | 0.9978 | 0.038     |
| B-spline | hermite | 8 | 176 B    | 0.9987 | 0.026     |
| RBF      | hermite | 8 | 176 B    | 0.9976 | 0.005     |

## ToN_IoT, K=8 (measured bytes)

| base     | interp  | L | mem/edge | F1     | MAE_logit |
|----------|---------|---|----------|--------|-----------|
| B-spline | linear  | 2 | 48 B     | 0.0000 | 6.744     |
| RBF      | linear  | 2 | 48 B     | 0.9240 | 3.842     |
| B-spline | hermite | 2 | 80 B     | 0.0286 | 6.378     |
| RBF      | hermite | 2 | 80 B     | 0.9656 | 1.579     |
| B-spline | linear  | 4 | 64 B     | 0.9870 | 2.047     |
| RBF      | linear  | 4 | 64 B     | 0.9965 | 0.220     |
| B-spline | hermite | 4 | 112 B    | 0.9927 | 0.334     |
| RBF      | hermite | 4 | 112 B    | 0.9956 | 0.008     |
| B-spline | linear  | 8 | 96 B     | 0.9929 | 0.598     |
| RBF      | linear  | 8 | 96 B     | 0.9956 | 0.059     |
| B-spline | hermite | 8 | 176 B    | 0.9934 | 0.005     |
| RBF      | hermite | 8 | 176 B    | 0.9955 | 0.004     |

## Memory at fixed accuracy (K in {4,8} sweep, RBF base)

The Hermite advantage grows with target severity. At loose targets
(MAE_logit <= 0.05 on ToN) linear and Hermite are near parity (80 vs 88 B).
At MAE <= 0.02: Hermite 88 B vs linear 160 B (1.8x, both datasets). At
MAE <= 0.01: Hermite reaches it at 112 B on both datasets; linear needs
160 B on CICIDS and does not reach it at all on ToN_IoT within the sweep.
On CICIDS at MAE <= 0.05, Hermite K=4 L=4 = 56 B vs linear 80 B.

Design note: Hermite benefits from sample spacing dx = segw/(L-1) not
exceeding the learned kernel width h (layer-wise bisect during
development); the leak-free CICIDS retrain (h ~ 0.19 > dx = 0.125)
confirms it — hermite L=2 holds at F1 0.998.

## Pipeline

```bash
python preprocessing.py --csv data/Wednesday-workingHours.pcap_ISCX.csv --out dos_data
python preprocessing_ton.py --csv data_ton/Train_Test_Network.csv --out ton_data
python train_bspline_baseline.py --data dos_data/dataset.pt --device cuda
python train_rbf.py --data dos_data/dataset.pt --device cuda
python lut_eval_rbf.py --K 8 --L_list 2 4 8 16
python lut_eval_bspline.py --K 8 --L_list 2 4 8 16
```

## Notes

- RBF centers live on [0,1]; frozen min-max normalization between layers
  keeps every layer in that domain (`rbf_multilayer.py`).
- Non-monotonic Hermite MAE at large L is the int8 derivative quantization
  floor (`diag_hermite_quant.py`).
- Analytic Cox-de Boor derivatives for B-spline Hermite and the
  Lobachevsky3 interpolator (fixed, benchmarked, dominated by Hermite)
  are documented in the root README.
