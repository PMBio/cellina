"""Tests for the explicit ``layer`` argument on the counterfactual / perturbation helpers.

The training ``spatial_x`` is usually built from a normalized representation
(log1p of CP10K) while ``adata.X`` is reset to raw counts for the model. These
tests pin down that the counterfactual and node-perturbation helpers aggregate
the representation the caller asks for.
"""

import warnings

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

from cellina._spatial_utils import (
    compute_spatial_features,
    make_counterfactual_adata,
    make_neighbor_perturbation,
)

N_OBS = 60
N_VARS = 20


def _to_dense(x):
    return np.asarray(x.toarray() if issparse(x) else x, dtype=np.float64)


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=8.0, size=(N_OBS, N_VARS)).astype(np.float32)
    # make sure the raw-count heuristic triggers (integer valued, max > 50)
    counts[0, 0] = 120.0
    coords = rng.uniform(0, 100, size=(N_OBS, 2))

    adata = AnnData(X=csr_matrix(counts))
    adata.obs_names = [f"cell_{i}" for i in range(N_OBS)]
    adata.var_names = [f"gene_{i}" for i in range(N_VARS)]
    adata.obsm["spatial"] = coords
    adata.layers["counts"] = csr_matrix(counts)

    # log1p(CP10K) "training" representation
    lognorm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-8) * 1e4
    lognorm = np.log1p(lognorm).astype(np.float32)
    adata.layers["lognorm"] = csr_matrix(lognorm)

    # simple kNN connectivity (binary, symmetric)
    nn = NearestNeighbors(n_neighbors=6).fit(coords)
    conn = nn.kneighbors_graph(coords, mode="connectivity")
    conn.setdiag(0)
    conn.eliminate_zeros()
    conn = conn.maximum(conn.T).tocsr().astype(np.float32)
    adata.obsp["spatial_connectivities"] = conn

    # training spatial features come from the lognorm layer
    compute_spatial_features(adata, obsm_key="spatial_x", layer="lognorm")
    return adata


def _expected_aggregation(conn, matrix):
    """Degree-normalized neighbour mean of ``matrix`` over ``conn``."""
    C = csr_matrix(conn)
    result = _to_dense(C @ csr_matrix(matrix))
    degree = np.asarray(C.sum(axis=1)).ravel()
    denom = 1.0 / np.where(degree == 0, 1.0, degree)
    return result * denom[:, None]


def test_counterfactual_layer_aggregates_named_layer(adata):
    indices_basal = np.arange(0, 10)
    indices_counterfactual = np.arange(30, 60)

    adata_cf = make_counterfactual_adata(
        adata,
        indices_basal,
        indices_counterfactual,
        spatial_column="spatial_x",
        precomputed=False,
        n_neighbours=5,
        random_state=0,
        layer="lognorm",
    )

    # reuse the cf connectivity the function wrote back to the source adata
    expected = _expected_aggregation(
        adata.obsp["spatial_connectivities_cf"], adata.layers["lognorm"]
    )[indices_basal]
    np.testing.assert_allclose(
        _to_dense(adata_cf.obsm["spatial_x"]), expected, rtol=1e-5, atol=1e-6
    )

    # and it is *not* the raw-count aggregation
    raw_expected = _expected_aggregation(
        adata.obsp["spatial_connectivities_cf"], adata.X
    )[indices_basal]
    assert not np.allclose(_to_dense(adata_cf.obsm["spatial_x"]), raw_expected)


def test_counterfactual_layer_none_matches_previous_behaviour(adata):
    """``layer=None`` keeps the 1.1.1 behaviour: ``adata.X`` is aggregated."""
    indices_basal = np.arange(0, 10)
    indices_counterfactual = np.arange(30, 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata_cf = make_counterfactual_adata(
            adata,
            indices_basal,
            indices_counterfactual,
            spatial_column="spatial_x",
            precomputed=False,
            n_neighbours=5,
            random_state=0,
        )

    expected = _expected_aggregation(
        adata.obsp["spatial_connectivities_cf"], adata.X
    )[indices_basal]
    np.testing.assert_allclose(
        _to_dense(adata_cf.obsm["spatial_x"]), expected, rtol=1e-5, atol=1e-6
    )


def test_raw_counts_warning(adata):
    with pytest.warns(UserWarning, match="raw counts"):
        compute_spatial_features(adata, obsm_key="spatial_x_warn")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_features(adata, obsm_key="spatial_x_nowarn", layer="lognorm")


def test_neighbor_perturbation_sources_from_layer(adata):
    perturbations = {"gene_0": 1.0, "gene_1": -0.5}

    make_neighbor_perturbation(
        adata,
        perturbations=perturbations,
        obsm_key_out="spatial_x_cf",
        layer_key="pert_lognorm",
        add_shift=True,
        renormalize=False,
        layer="lognorm",
    )

    lognorm = _to_dense(adata.layers["lognorm"])
    shift = np.zeros(N_VARS)
    shift[0] = 1.0
    shift[1] = -0.5
    expected_pert = np.clip(lognorm + shift, 0, None)
    np.testing.assert_allclose(
        _to_dense(adata.layers["pert_lognorm"]), expected_pert, rtol=1e-5, atol=1e-6
    )

    expected_cf = _expected_aggregation(
        adata.obsp["spatial_connectivities"], expected_pert
    )
    np.testing.assert_allclose(
        _to_dense(adata.obsm["spatial_x_cf"]), expected_cf, rtol=1e-5, atol=1e-5
    )

    # sourcing from X (raw counts) gives a different perturbed matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        make_neighbor_perturbation(
            adata,
            perturbations=perturbations,
            obsm_key_out="spatial_x_cf_raw",
            layer_key="pert_raw",
            add_shift=True,
            renormalize=False,
        )
    assert not np.allclose(
        _to_dense(adata.layers["pert_raw"]), _to_dense(adata.layers["pert_lognorm"])
    )
