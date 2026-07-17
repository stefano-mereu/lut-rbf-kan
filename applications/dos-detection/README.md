# DoS Detection: RBF-KAN + Hermite LUT vs B-spline

Application of the RBF-KAN + Hermite LUT extension to network intrusion
detection (CICIDS2017 DoS Hulk), extending Kuznetsov (2025) from B-spline to
Gaussian RBF edges with cubic Hermite LUT interpolation.

## Float models

Multi-layer KAN [78, 32, 16, 1], balanced BENIGN vs DoS Hulk.

| Model             | float F1 |
|-------------------|----------|
| B-spline (PyKAN)  | 0.9985   |
| RBF-KAN (ours)    | 0.9982   |

## LUT results (K=8, all four basis/interp combinations)

All configurations use the same extended quantization pipeline
(`build_rbf_lut_for_edges` + `forward_dense_numpy_rbf`), uint8 values,
int8 derivative table for Hermite. B-spline knots are widened to each
layer's real input domain (measured from `model.acts`) to reproduce
PyKAN extrapolation. B-spline Hermite derivatives are computed by central
finite differences on the full edge function (numerically equivalent to
the analytic Cox-de Boor derivative for compilation purposes).

| base     | interp  | L | mem/edge | F1     | MAE_logit |
|----------|---------|---|----------|--------|-----------|
| B-spline | linear  | 2 | 16 B     | 0.7892 | 7.822     |
| RBF      | linear  | 2 | 16 B     | 0.9002 | 5.686     |
| B-spline | hermite | 2 | 32 B     | 0.7974 | 5.926     |
| RBF      | hermite | 2 | 32 B     | 0.9969 | 0.275     |
| B-spline | linear  | 4 | 32 B     | 0.8310 | 3.689     |
| RBF      | linear  | 4 | 32 B     | 0.9975 | 0.283     |
| B-spline | hermite | 4 | 64 B     | 0.9260 | 1.051     |
| RBF      | hermite | 4 | 64 B     | 0.9979 | 0.006     |
| B-spline | linear  | 8 | 64 B     | 0.9799 | 1.123     |
| RBF      | linear  | 8 | 64 B     | 0.9977 | 0.082     |
| B-spline | hermite | 8 | 128 B    | 0.9987 | 0.025     |
| RBF      | hermite | 8 | 128 B    | 0.9979 | 0.010     |

Reading: Hermite helps both bases, but Gaussians exploit it far more.
RBF+Hermite at L=2 (32 B) already beats B-spline+Hermite at L=4 (64 B).
Two reasons: Gaussian edges are C-infinity (B-spline derivative is
discontinuous at grid knots), and the RBF model normalizes each layer's
input to [0,1] while B-spline covers a wide native domain ([-4.5, 6.4]
and growing per layer) with the same K segments. The inter-layer
normalization is part of what makes the RBF architecture LUT-friendly.

The task saturates quickly (logistic regression reaches F1 0.989), so
MAE on pre-threshold logits is reported as the finer-grained metric.

## Pipeline

Requires the CICIDS2017 `Wednesday-workingHours.pcap_ISCX.csv` in `data/`
(Canadian Institute for Cybersecurity, or Kaggle mirror).

```bash
python preprocessing.py --csv data/Wednesday-workingHours.pcap_ISCX.csv --out dos_data
python train_bspline_baseline.py --data dos_data/dataset.pt --device cuda
python train_rbf.py --data dos_data/dataset.pt --device cuda
python lut_eval_rbf.py --K 8 --L_list 2 4 8 16
python lut_eval_bspline.py --K 8 --L_list 2 4 8 16
```

## Notes

- The RBF layer has centers on [0,1]; a frozen min-max normalization between
  layers keeps each layer's input in that domain (see `rbf_multilayer.py`).
- The non-monotonic Hermite MAE at large L is the int8 derivative quantization
  floor, not an algorithmic effect (`diag_hermite_quant.py` verifies this).
- Analytic Cox-de Boor derivatives for B-spline Hermite and the Lobachevsky3
  interpolator (implemented but untested) are future work.
