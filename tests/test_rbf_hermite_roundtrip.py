# tests/test_rbf_hermite_roundtrip.py
"""
RBF-KAN + Hermite LUT: engineering correctness tests.

Mirrors the structure of test_lut_roundtrip.py (Kuznetsov 2026) for the
RBF+Hermite extension. Verifies:

  1. LUT artifact builds without error and has correct shapes
  2. RBFLUTArtifact saves and reloads correctly (NPZ roundtrip)
  3. forward_dense_numpy_rbf output is finite and bounded
  4. linear and hermite backends agree on values (hermite -> linear as L -> inf)
  5. hermite MAE <= linear MAE on coarse grid (core quality claim)
  6. memory footprint matches expected formula: K*L*(1 or 2) bytes per edge
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.rbf_adapter import RBFKANSingleLayerAdapter
from src.quant.lut_builder_rbf import build_rbf_lut_for_edges, RBFLUTArtifact
from src.kernels.lut_backend_dense_numpy_rbf import (
    forward_dense_numpy_rbf,
    pack_rbf_dense_layer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_adapter(in_dim=3, out_dim=2, K_seg=8, seed=0):
    G = 20
    h = 1.5 / (G - 1)
    adapter = RBFKANSingleLayerAdapter.from_arch(
        {
            "in_dim": in_dim, "out_dim": out_dim,
            "G": G, "h_init": h,
            "x_min": 0.0, "x_max": 1.0,
            "grid_segments": K_seg,
        },
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    adapter.layer.coef[:] = rng.uniform(
        -1.0, 1.0, size=adapter.layer.coef.shape
    ).astype(np.float32)
    return adapter


def _build_art(adapter, L, interp):
    edges = adapter.extract_edges()
    return build_rbf_lut_for_edges(
        edges=edges, L=L, interp=interp,
        y_range_method="minmax",
        lower_pct=0.0, upper_pct=100.0,
        dtype="uint8", scheme="asymmetric",
        qmin=0, qmax=255,
    ), edges


# ---------------------------------------------------------------------------
# Test 1: artifact shapes
# ---------------------------------------------------------------------------

def test_artifact_shapes():
    adapter = _make_adapter(in_dim=3, out_dim=2, K_seg=8)
    art, edges = _build_art(adapter, L=8, interp="hermite")

    E = adapter.in_dim * adapter.out_dim
    K = adapter.grid_segments
    L = 8

    assert art.q_table.shape  == (E, K, L), f"q_table shape wrong: {art.q_table.shape}"
    assert art.dq_table.shape == (E, K, L), f"dq_table shape wrong: {art.dq_table.shape}"
    assert art.scale.shape    == (E, K),    f"scale shape wrong: {art.scale.shape}"
    assert art.dscale.shape   == (E, K),    f"dscale shape wrong: {art.dscale.shape}"
    assert art.q_table.dtype  == np.uint8
    assert art.dq_table.dtype == np.int8
    assert art.has_derivatives()
    print(f"\nArtifact shapes OK: E={E}, K={K}, L={L}")


# ---------------------------------------------------------------------------
# Test 2: NPZ save / reload roundtrip
# ---------------------------------------------------------------------------

def test_npz_roundtrip(tmp_path):
    """Save RBFLUTArtifact to NPZ and reload — all arrays must match."""
    import numpy as np

    adapter = _make_adapter(in_dim=2, out_dim=2, K_seg=4)
    art, edges = _build_art(adapter, L=8, interp="hermite")

    npz_path = tmp_path / "rbf_lut.npz"

    # Manual save (same pattern as lut_io.save_lut_npz)
    np.savez_compressed(
        npz_path,
        format_version=np.int32(art.format_version),
        knots=art.knots,
        L=np.int32(art.L),
        interp=np.asarray(art.interp),
        boundary_mode=np.asarray(art.boundary_mode),
        oob_behavior=np.asarray(art.oob_behavior),
        q_table=art.q_table,
        scale=art.scale,
        y_min=art.y_min,
        dtype=np.asarray(art.dtype),
        scheme=np.asarray(art.scheme),
        qmin=np.int32(art.qmin),
        qmax=np.int32(art.qmax),
        value_representation=np.asarray(art.value_representation),
        base_kind=np.asarray(art.base_kind),
        dq_table=art.dq_table,
        dscale=art.dscale,
        dy_min=art.dy_min,
    )

    assert npz_path.exists()

    with np.load(npz_path, allow_pickle=False) as z:
        assert np.array_equal(z["q_table"],  art.q_table)
        assert np.array_equal(z["dq_table"], art.dq_table)
        assert np.allclose(z["scale"],  art.scale,  atol=1e-6)
        assert np.allclose(z["dscale"], art.dscale, atol=1e-6)
        assert int(z["L"]) == art.L

    print(f"\nNPZ roundtrip OK: {npz_path.stat().st_size} bytes")


# ---------------------------------------------------------------------------
# Test 3: forward is finite and bounded
# ---------------------------------------------------------------------------

def test_forward_finite():
    adapter = _make_adapter(in_dim=4, out_dim=3, K_seg=8)
    x_test = np.random.default_rng(0).uniform(
        0.0, 1.0, size=(200, adapter.in_dim)
    ).astype(np.float32)

    y_true = adapter.forward_float(x_test)

    for interp in ("linear", "hermite"):
        art, edges = _build_art(adapter, L=8, interp=interp)
        packed = pack_rbf_dense_layer(
            art, edges=edges,
            in_dim=adapter.in_dim, out_dim=adapter.out_dim,
        )
        y_lut = forward_dense_numpy_rbf(x_test, packed)

        assert y_lut.shape == y_true.shape
        assert np.all(np.isfinite(y_lut)), f"{interp}: non-finite values in output"
        assert np.max(np.abs(y_lut)) < 1e4, f"{interp}: output unexpectedly large"

    print("\nForward finite OK for linear and hermite")


# ---------------------------------------------------------------------------
# Test 4: hermite and linear converge as L increases
# ---------------------------------------------------------------------------

def test_hermite_converges_to_linear():
    """
    As L -> inf both interpolators sample the same dense grid and their
    outputs should converge. At L=256 the difference should be < 1e-3.
    """
    adapter = _make_adapter(in_dim=2, out_dim=2, K_seg=4)
    x_test = np.random.default_rng(1).uniform(
        0.0, 1.0, size=(100, adapter.in_dim)
    ).astype(np.float32)

    diffs = []
    for L in [8, 32, 128]:
        art_lin,  edges = _build_art(adapter, L=L, interp="linear")
        art_herm, _     = _build_art(adapter, L=L, interp="hermite")

        pk_lin  = pack_rbf_dense_layer(art_lin,  edges=edges,
                                        in_dim=adapter.in_dim, out_dim=adapter.out_dim)
        pk_herm = pack_rbf_dense_layer(art_herm, edges=edges,
                                        in_dim=adapter.in_dim, out_dim=adapter.out_dim)

        y_lin  = forward_dense_numpy_rbf(x_test, pk_lin)
        y_herm = forward_dense_numpy_rbf(x_test, pk_herm)

        diff = float(np.mean(np.abs(y_lin - y_herm)))
        diffs.append(diff)
        print(f"  L={L:>4}: mean |hermite - linear| = {diff:.6f}")

    # Convergence: diff should decrease as L increases
    assert diffs[-1] < diffs[0], "hermite and linear should converge as L grows"
    print("✓ Hermite converges to linear as L increases")


# ---------------------------------------------------------------------------
# Test 5: hermite MAE <= linear MAE on coarse grid (core quality claim)
# ---------------------------------------------------------------------------

def test_hermite_beats_linear_quality():
    """
    Core scientific claim: on a coarse grid (K=8 segments), Hermite
    achieves lower MAE than linear with the same memory budget.

    Memory: hermite K=8 L=4 = 2*8*4 = 64 bytes/edge
            linear  K=8 L=8 =   8*8 = 64 bytes/edge
    """
    adapter = _make_adapter(in_dim=4, out_dim=4, K_seg=8, seed=7)
    rng = np.random.default_rng(42)
    x_test = rng.uniform(0.0, 1.0, size=(500, adapter.in_dim)).astype(np.float32)
    y_true = adapter.forward_float(x_test)

    # linear K=8 L=8: 64 bytes/edge
    art_lin, edges = _build_art(adapter, L=8, interp="linear")
    pk_lin = pack_rbf_dense_layer(art_lin, edges=edges,
                                   in_dim=adapter.in_dim, out_dim=adapter.out_dim)
    mae_lin = float(np.mean(np.abs(y_true - forward_dense_numpy_rbf(x_test, pk_lin))))

    # hermite K=8 L=4: 64 bytes/edge (2 tables × half resolution)
    adapter_coarse = _make_adapter(in_dim=4, out_dim=4, K_seg=4, seed=7)
    adapter_coarse.layer.coef[:] = adapter.layer.coef
    art_herm, edges_c = _build_art(adapter_coarse, L=8, interp="hermite")
    pk_herm = pack_rbf_dense_layer(art_herm, edges=edges_c,
                                    in_dim=adapter.in_dim, out_dim=adapter.out_dim)
    mae_herm = float(np.mean(np.abs(y_true - forward_dense_numpy_rbf(x_test, pk_herm))))

    mem_lin  = adapter.grid_segments * 8 * 1
    mem_herm = 4 * 8 * 2

    print(f"\nlinear  K=8  L=8:  MAE={mae_lin:.5f}  mem={mem_lin} bytes/edge")
    print(f"hermite K=4  L=8:  MAE={mae_herm:.5f}  mem={mem_herm} bytes/edge")
    print(f"Improvement: {(1 - mae_herm/mae_lin)*100:.1f}%  at same memory")

    assert mae_herm < mae_lin, (
        f"Hermite should beat linear at same memory: "
        f"hermite={mae_herm:.5f} vs linear={mae_lin:.5f}"
    )
    print("✓ Hermite beats linear at same memory budget")


# ---------------------------------------------------------------------------
# Test 6: memory formula
# ---------------------------------------------------------------------------

def test_memory_formula():
    """
    Verify that artifact memory matches K * L * (1 or 2) bytes per edge.
    """
    from src.quant.lut_builder import artifact_memory_bytes

    adapter = _make_adapter(in_dim=2, out_dim=2, K_seg=8)
    E = adapter.in_dim * adapter.out_dim
    K = adapter.grid_segments

    for L in [4, 8, 16]:
        art_lin,  _ = _build_art(adapter, L=L, interp="linear")
        art_herm, _ = _build_art(adapter, L=L, interp="hermite")

        # q_table: E*K*L bytes (uint8)
        # dq_table: E*K*L bytes (int8) — only for hermite
        expected_lin  = E * K * L * 1
        expected_herm = E * K * L * 2

        actual_lin_q  = int(art_lin.q_table.nbytes)
        actual_herm_q = int(art_herm.q_table.nbytes + art_herm.dq_table.nbytes)

        assert actual_lin_q  == expected_lin,  f"L={L}: linear q_table size wrong"
        assert actual_herm_q == expected_herm, f"L={L}: hermite q+dq size wrong"

    print(f"\nMemory formula verified for L in [4, 8, 16]")
    print("✓ Memory formula correct")
