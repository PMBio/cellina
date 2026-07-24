"""Tests for the `perturb_fraction` parameter of the node-perturbation machinery."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from cellina._spatial_utils import _make_perturbed_expression


def _toy_adata(n_cells=200, n_genes=8, seed=0):
    rng = np.random.default_rng(seed)
    # log1p-like non-negative expression
    X = np.log1p(rng.poisson(5.0, size=(n_cells, n_genes)).astype(np.float32))
    adata = AnnData(csr_matrix(X))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["ct"] = rng.integers(0, 3, size=n_cells).astype(str)
    return adata


# additive log-space shift applied globally (groupby=None)
PERTURB = {"g0": 2.0, "g1": -1.5, "g2": 1.0}


def _perturb(adata, fraction, random_state=0):
    return _make_perturbed_expression(
        adata, perturbations=PERTURB, groupby=None, add_shift=True,
        renormalize=True, perturb_fraction=fraction, random_state=random_state,
    ).toarray()


def _n_changed_rows(orig, pert, tol=1e-5):
    return int((np.abs(orig - pert).max(axis=1) > tol).sum())


def test_fraction_zero_is_identity():
    adata = _toy_adata()
    orig = adata.X.toarray()
    pert = _perturb(adata, 0.0)
    assert np.allclose(orig, pert, atol=1e-5)


def test_fraction_one_perturbs_all_rows():
    adata = _toy_adata()
    orig = adata.X.toarray()
    pert = _perturb(adata, 1.0)
    assert _n_changed_rows(orig, pert) == adata.n_obs


def test_number_of_changed_rows_matches_fraction():
    adata = _toy_adata()
    orig = adata.X.toarray()
    n = adata.n_obs
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        pert = _perturb(adata, f)
        assert _n_changed_rows(orig, pert) == int(round(f * n))


def test_unperturbed_rows_are_exactly_preserved():
    adata = _toy_adata()
    orig = adata.X.toarray()
    pert = _perturb(adata, 0.5, random_state=0)
    unchanged = np.abs(orig - pert).max(axis=1) <= 1e-5
    assert np.allclose(orig[unchanged], pert[unchanged], atol=1e-6)


def test_magnitude_is_monotonic_in_fraction():
    adata = _toy_adata()
    orig = adata.X.toarray()
    mags = []
    for f in (0.0, 0.25, 0.5, 1.0):
        pert = _perturb(adata, f)
        mags.append(np.abs(pert - orig).mean())
    assert mags[0] < 1e-6  # f=0 -> no perturbation (up to renormalisation float noise)
    # strictly increasing perturbation magnitude with fraction
    assert mags[1] < mags[2] < mags[3]


def test_random_state_controls_selection():
    adata = _toy_adata()
    p_a = _perturb(adata, 0.5, random_state=0)
    p_a2 = _perturb(adata, 0.5, random_state=0)
    p_b = _perturb(adata, 0.5, random_state=1)
    assert np.allclose(p_a, p_a2)          # reproducible
    assert not np.allclose(p_a, p_b)       # seed changes which cells are perturbed


def test_invalid_fraction_raises():
    adata = _toy_adata()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            _perturb(adata, bad)
