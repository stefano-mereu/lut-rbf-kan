# RBF-KAN + Hermite LUT: extension to LUT-KAN

This branch extends the [LUT-KAN framework](https://arxiv.org/abs/2601.03332) (Kuznetsov 2026) with two contributions:

1. **RBF-KAN adapter** — Gaussian radial basis functions as edge activations, with shape parameter h initialised at the midpoint of the theoretical conditioning interval.
2. **Hermite LUT interpolation** — cubic Hermite interpolation inside the LUT, exploiting the analytic derivative of Gaussian edges at zero extra cost during compilation.

## New files

```
src/kernels/rbf_math.py                     Gaussian RBF evaluation + analytic derivative
src/kernels/lut_interp_advanced.py          Hermite cubic interpolation kernel
src/kernels/lut_backend_dense_numpy_rbf.py  Forward pass: linear / hermite (dense NumPy)
src/models/rbf_adapter.py                   RBF-KAN layer + h init + PyTorch training layer
src/quant/lut_builder_rbf.py                Extended LUT builder: samples phi and phi' simultaneously
scripts/benchmark_final.py                  Benchmark on standard KAN target functions
tests/test_gkan_lut_workflow.py             Scientific quality tests (method validation)
tests/test_rbf_hermite_roundtrip.py         Engineering correctness tests (shapes, NPZ, memory)
```

## Key results

### Model quality after LUT quantization (K=8, L=8, mean over 5 seeds)

| Function | Model | float_err | task_err | degrad% |
|---|---|---|---|---|
| sin(2πx) | B-spline + linear | 0.00700 | 0.03279 | +369% |
| sin(2πx) | RBF + linear | 0.00241 | 0.00252 | +4.9% |
| sin(2πx) | **RBF + Hermite** | **0.00241** | **0.00245** | **+1.8%** |
| \|x−0.5\| | B-spline + linear | 0.00657 | 0.01032 | +57% |
| \|x−0.5\| | RBF + linear | 0.00271 | 0.00260 | −4.1% |
| \|x−0.5\| | **RBF + Hermite** | **0.00271** | **0.00271** | **+0.1%** |
| tanh(5x) | B-spline + linear | 0.01654 | 0.02319 | +40% |
| tanh(5x) | RBF + linear | 0.00670 | 0.00653 | −2.6% |
| tanh(5x) | **RBF + Hermite** | **0.00670** | **0.00674** | **+0.6%** |

*Single-layer architecture. B-spline trained via least-squares. RBF trained via Adam (500 epochs, h learnable).*

**Note on B-spline degradation:** the high task_degrad for B-spline LUT is not an infrastructure bug. Training on a target function produces large coefficients that increase local curvature within each segment, requiring much higher L to achieve low phi_error. With random coefficients the same adapter works well at L=8. This effect only emerges with networks trained on real tasks — RBF edge functions maintain smooth, well-controlled curvature governed by h, making them intrinsically more amenable to LUT quantization.

### LUT approximation quality: Hermite vs linear (phi_err, mean over 5 seeds)

Full sweep K∈{4,8,16} × L∈{4,8,16}.

**sin(2πx)**

| K | L=4 lin | L=4 herm | impr% | L=8 lin | L=8 herm | impr% | L=16 lin | L=16 herm | impr% |
|---|---|---|---|---|---|---|---|---|---|
| 4  | 0.01639 | 0.00076 | 95.3 | 0.00309 | 0.00069 | 77.6 | 0.00109 | 0.00082 | 24.1 |
| 8  | 0.00438 | 0.00028 | 93.6 | 0.00090 | 0.00039 | 57.1 | 0.00040 | 0.00037 |  6.6 |
| 16 | 0.00108 | 0.00015 | 86.0 | 0.00027 | 0.00018 | 35.0 | 0.00019 | 0.00019 | −1.6 |

**\|x−0.5\| (cusp)**

| K | L=4 lin | L=4 herm | impr% | L=8 lin | L=8 herm | impr% | L=16 lin | L=16 herm | impr% |
|---|---|---|---|---|---|---|---|---|---|
| 4  | 0.00646 | 0.00129 | 80.0 | 0.00165 | 0.00022 | 86.5 | 0.00045 | 0.00018 | 60.0 |
| 8  | 0.00236 | 0.00019 | 92.1 | 0.00046 | 0.00009 | 79.9 | 0.00015 | 0.00009 | 38.8 |
| 16 | 0.00058 | 0.00004 | 92.4 | 0.00013 | 0.00004 | 65.9 | 0.00006 | 0.00005 | 16.4 |

**tanh(5x)**

| K | L=4 lin | L=4 herm | impr% | L=8 lin | L=8 herm | impr% | L=16 lin | L=16 herm | impr% |
|---|---|---|---|---|---|---|---|---|---|
| 4  | 0.01648 | 0.00096 | 94.1 | 0.00287 | 0.00038 | 86.6 | 0.00088 | 0.00041 | 53.6 |
| 8  | 0.00394 | 0.00025 | 93.8 | 0.00076 | 0.00020 | 73.8 | 0.00029 | 0.00020 | 30.7 |
| 16 | 0.00098 | 0.00008 | 91.5 | 0.00023 | 0.00011 | 53.1 | 0.00012 | 0.00011 |  9.2 |

**Hermite valid regime: K≥4, L∈[4,8].** With K<4 the segments are too wide relative to the Gaussian scale h and the cubic polynomial can overshoot. At L≥32 the int8 quantization of the derivative table introduces noise that reduces the Hermite advantage — this is a quantization floor effect, not an algorithmic one (with float32 derivatives Hermite always wins).

### 2D case: sin(pi x)cos(pi y)

With a single [2,1] layer both models underfit (float_err ~ 0.178 for B-spline
and RBF alike), and the LUT adds essentially nothing on top (task_err ~ float_err
for every config; e.g. at K=8, L=4: B-spline+linear 0.1789, RBF+linear 0.1764,
RBF+Hermite 0.1779). phi_err still shows the expected ordering (Hermite 0.0007
vs linear 0.0102 at L=4), but when the model itself is the bottleneck the
interpolation scheme is irrelevant for the task. Included for completeness:
this is the regime where Hermite is not needed.

## Why Hermite works here

The Gaussian edge function `φ(x) = Σₖ cₖ exp(-((x-μₖ)/ε)²)` has an analytic derivative:

```
φ'(x) = Σₖ cₖ · [-2(x-μₖ)/ε²] · exp(-((x-μₖ)/ε)²)
```

The derivative is free: the exp() values computed for φ are reused for φ'. During LUT compilation, both φ and φ' are sampled at the same grid points with no extra cost.

At inference, cubic Hermite interpolation between adjacent LUT entries uses these stored derivatives to reduce the interpolation error from O(dx²) to O(dx⁴). The same accuracy is achieved with fewer samples per segment, reducing memory at equal accuracy.

## Shape parameter initialisation

The shape parameter h is initialised at the midpoint of the conditioning interval from Noorizadegan & Wang (2026):

```
ε ∈ [h/2, 3h/2]   where h = (x_max - x_min) / (G - 1)
```

The midpoint `ε = h` keeps the ratio h/ε = 1, ensuring well-conditioned feature matrix. h is then refined as a learnable parameter during PyTorch training.

```python
from src.models.rbf_adapter import theoretical_h_init
h_init = theoretical_h_init(G=20)  # = 1/(G-1)
layer = RBFKANLayerTorch(input_dim=d, output_dim=1, G=20,
                          h_init=h_init, h_learnable=True)
```

## Running the tests

```bash
# Scientific quality tests
pytest tests/test_gkan_lut_workflow.py -v

# Engineering correctness tests
pytest tests/test_rbf_hermite_roundtrip.py -v

# Benchmark on target functions
python scripts/benchmark_final.py
```

## Workflow: train → compile → deploy

```python
from src.models.rbf_adapter import RBFKANLayerTorch, RBFKANSingleLayerAdapter, theoretical_h_init
from src.quant.lut_builder_rbf import build_rbf_lut_for_edges
from src.kernels.lut_backend_dense_numpy_rbf import forward_dense_numpy_rbf, pack_rbf_dense_layer

# 1. Initialise h at midpoint of conditioning interval
h_init = theoretical_h_init(G=20)

# 2. Train
layer = RBFKANLayerTorch(input_dim=d, output_dim=1, G=20,
                          h_init=h_init, h_learnable=True)
# ... training loop ...
layer.freeze_h()

# 3. Export and compile LUT
adapter = RBFKANSingleLayerAdapter.from_trained_layer(layer)
edges = adapter.extract_edges()
art = build_rbf_lut_for_edges(edges, L=8, interp="hermite",
                               dtype="uint8", scheme="asymmetric",
                               qmin=0, qmax=255,
                               y_range_method="minmax",
                               lower_pct=0.0, upper_pct=100.0)

# 4. Inference
packed = pack_rbf_dense_layer(art, edges=edges,
                               in_dim=adapter.in_dim, out_dim=adapter.out_dim)
y = forward_dense_numpy_rbf(x_new, packed)
```

## Future work

The free analytic derivative is not exclusive to Gaussian kernels:
- **B-splines**: derivative via Cox–de Boor degree-reduction recurrence
- **Chebyshev**: T'ₙ(x) = n·Uₙ₋₁(x)

A full Hermite LUT implementation for all three basis types would make this a general-purpose upgrade to the LUT-KAN framework.

## References

- Kuznetsov O. (2026). *LUT-KAN: Segment-wise LUT Quantization for Fast KAN Inference*. arXiv:2601.03332.
- Noorizadegan A., Wang Y. (2026). *Scaling of Gaussian Kolmogorov-Arnold Networks*. arXiv:2604.21174.
- Cavoretto R., De Rossi A., Haider A., Mereu S. (2026). *BO-GKAN: Bayesian Shape Parameter Optimization for Gaussian KAN PINNs*.
- Rippa S. (1999). An algorithm for selecting a good value for the parameter c in radial basis function interpolation. *Advances in Computational Mathematics*, 11, 193–210.
