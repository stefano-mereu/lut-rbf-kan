# RBF-KAN + Hermite LUT: extension to LUT-KAN

This branch extends the [LUT-KAN framework](https://arxiv.org/abs/2601.03332) (Kuznetsov 2026) with two contributions:

1. **RBF-KAN adapter** — Gaussian radial basis functions as edge activations, with adaptive shape parameter initialised via Leave-One-Out Cross-Validation.
2. **Hermite LUT interpolation** — cubic Hermite interpolation inside the LUT, exploiting the analytic derivative of Gaussian edges at zero extra cost during compilation.

## New files

```
src/kernels/rbf_math.py                   Gaussian RBF evaluation + analytic derivative
src/kernels/lut_interp_advanced.py        Hermite cubic interpolation kernel
src/kernels/lut_backend_dense_numpy_rbf.py  Forward pass: linear / hermite (dense NumPy)
src/models/rbf_adapter.py                 RBF-KAN layer + LOOCV h init + PyTorch training layer
src/quant/lut_builder_rbf.py              Extended LUT builder: samples phi and phi' simultaneously
tests/test_gkan_lut_workflow.py           Scientific quality tests (method validation)
tests/test_rbf_hermite_roundtrip.py       Engineering correctness tests (shapes, NPZ, memory)
```

## Key result

At equal memory budget (64 bytes per edge), Hermite LUT achieves **30–86% lower MAE** than linear LUT on the same RBF network. Compared to the original B-spline + linear baseline, RBF + Hermite achieves **6–37× lower MAE** at equal memory.

| method | K | L | mem/edge (bytes) | MAE |
|---|---|---|---|---|
| B-spline + linear | 7 | 8 | 56 | 0.036 |
| RBF + linear | 8 | 8 | 64 | 0.006 |
| **RBF + Hermite** | **8** | **4** | **64** | **0.001** |
| RBF + linear | 8 | 16 | 128 | 0.002 |

*Measured on random-coefficient 4×4 layer, domain [0,1], G=20 RBF centers, h=1.5/19.*

## Why Hermite works here

The Gaussian edge function `φ(x) = Σₖ cₖ exp(-(x-μₖ)²/2σ²)` has an analytic derivative:

```
φ'(x) = Σₖ cₖ · [-(x-μₖ)/σ²] · exp(-(x-μₖ)²/2σ²)
```

The derivative is free: the exp() values computed for φ are reused for φ'. During LUT compilation, both φ and φ' are sampled at the same grid points with no extra cost.

At inference, cubic Hermite interpolation between adjacent LUT entries uses these stored derivatives to reduce the interpolation error from O(dx²) to O(dx⁴), where dx is the spacing between LUT samples. This means the same accuracy can be achieved with 2–4× fewer samples per segment, halving memory while improving accuracy.

The Hermite derivative correction term scales as `|φ'| · dx_lut`. For this to be effective, the function must have significant curvature relative to the segment width. With K=8 segments on [0,1] and h≈0.08, each segment width (0.125) is comparable to the Gaussian scale (3h≈0.24), ensuring meaningful curvature is present within each segment.

## Shape parameter initialisation (LOOCV)

The shape parameter h is initialised via LOOCV in the theoretical interval [0.8/(G-1), 2.5/(G-1)]:

```python
from src.models.rbf_adapter import loocv_optimal_h
h_init = loocv_optimal_h(x_train, y_train, G=20)
```

h can then be refined as a learnable parameter during PyTorch training via `RBFKANLayerTorch(h_learnable=True)`, and frozen before LUT compilation with `layer.freeze_h()`.

## Running the tests

```bash
# Scientific quality tests (method validation)
pytest tests/test_gkan_lut_workflow.py -v

# Engineering correctness tests (shapes, NPZ roundtrip, memory)
pytest tests/test_rbf_hermite_roundtrip.py -v

# Full suite including original tests
pytest tests/ -v
```

Expected output for quality tests:

```
test_loocv_h_in_range          PASSED   h_loocv=0.1316 in [0.042, 0.132]
test_numpy_gkan_forward        PASSED   forward shape (50, 1)
test_edge_has_analytic_deriv   PASSED   max error vs FD: 1.16e-04
test_hermite_beats_linear_...  PASSED   86.4% MAE improvement, 50% less memory
test_torch_layer               PASSED   h drift 11-13%, loss converges
```

## Workflow: train → compile → deploy

```python
from src.models.rbf_adapter import RBFKANLayerTorch, RBFKANSingleLayerAdapter, loocv_optimal_h
from src.quant.lut_builder_rbf import build_rbf_lut_for_edges
from src.kernels.lut_backend_dense_numpy_rbf import forward_dense_numpy_rbf, pack_rbf_dense_layer

# 1. Initialise h via LOOCV
h_init = loocv_optimal_h(x_train.ravel(), y_train.ravel(), G=20)

# 2. Train
layer = RBFKANLayerTorch(input_dim=d, output_dim=1, G=20,
                          h_init=h_init, h_learnable=True)
# ... training loop ...

# 3. Freeze h and export
layer.freeze_h()
adapter = RBFKANSingleLayerAdapter.from_trained_layer(layer)

# 4. Compile LUT with Hermite interpolation
edges = adapter.extract_edges()
art = build_rbf_lut_for_edges(edges, L=8, interp="hermite",
                               dtype="uint8", scheme="asymmetric",
                               qmin=0, qmax=255,
                               y_range_method="minmax",
                               lower_pct=0.0, upper_pct=100.0)

# 5. Inference
packed = pack_rbf_dense_layer(art, edges=edges,
                               in_dim=adapter.in_dim, out_dim=adapter.out_dim)
y = forward_dense_numpy_rbf(x_new, packed)
```

## References

- Kuznetsov O. (2026). *LUT-KAN: Segment-wise LUT Quantization for Fast KAN Inference*. arXiv:2601.03332.
- Noorizadegan A., Wang Y. (2025). *GKAN: Gaussian Kolmogorov-Arnold Networks*.
- Rippa S. (1999). An algorithm for selecting a good value for the parameter c in radial basis function interpolation. *Advances in Computational Mathematics*, 11, 193–210.
- De Rossi A., Cavoretto R. et al. — Lobachevsky spline interpolation and RBF-PUM methods.
