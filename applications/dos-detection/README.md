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

## Second dataset: ToN_IoT (normal vs dos)

`Train_Test_Network.csv` (UNSW), binary normal-vs-dos, 20k/class, 16 numeric
features. Preprocessing mirrors CICIDS with two dataset-driven fixes
(`preprocessing_ton.py`): the IQR clip is skipped when IQR=0 (on sparse
columns q1=q3=0 and clipping would collapse them to constants), and scaled
features are winsorized at +-10 sigma (heavy-tailed sparse columns otherwise
reach ~200 sigma, leaving uniform LUT knots no resolution where data lives).
`ts` is dropped (time-clustered attacks = label leakage); IPs are excluded
(labels were assigned by tagging attacker IPs).

Float: B-spline F1 0.9936, RBF-KAN 0.9953. Unlike CICIDS the task does not
saturate at the same level, so F1 differences are visible directly.

| base     | interp  | L | mem/edge | F1     | MAE_logit |
|----------|---------|---|----------|--------|-----------|
| B-spline | linear  | 2 | 16 B     | 0.0000 | 6.622     |
| RBF      | linear  | 2 | 16 B     | 0.1114 | 5.636     |
| B-spline | hermite | 2 | 32 B     | 0.0961 | 4.664     |
| RBF      | hermite | 2 | 32 B     | 0.9900 | 1.842     |
| B-spline | linear  | 4 | 32 B     | 0.9873 | 1.643     |
| RBF      | linear  | 4 | 32 B     | 0.9961 | 0.415     |
| B-spline | hermite | 4 | 64 B     | 0.9932 | 0.235     |
| RBF      | hermite | 4 | 64 B     | 0.9951 | 0.013     |
| B-spline | linear  | 8 | 64 B     | 0.9927 | 0.339     |
| RBF      | linear  | 8 | 64 B     | 0.9960 | 0.119     |
| B-spline | hermite | 8 | 128 B    | 0.9936 | 0.011     |
| RBF      | hermite | 8 | 128 B    | 0.9949 | 0.009     |

At L=2 every other configuration collapses; RBF+Hermite is the only one
alive (F1 0.990). At L=4 it recovers float exactly with MAE 20-125x below
the alternatives. Design note: Hermite benefits from sample spacing
dx = segw/(L-1) not exceeding the learned kernel width h (verified by a
layer-wise bisect during development; with healthy preprocessing the L=2
cell holds even slightly above that threshold).

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
