from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from ._spatial_utils import compute_spatial_features

_CF_CONN_TMP = "__cf_conn_tmp"
_CF_OBSM_TMP = "__cf_obsm_tmp"


def make_counterfactual_adata(
    adata,
    indices_basal,
    indices_counterfactual,
    spatial_column,
    precomputed: bool = True,
    n_neighbours: Optional[int] = None,
    random_state: int = 0,
):
    """
    Create a counterfactual AnnData keeping everything from the original
    except .obsm[spatial_column], which is replaced with counterfactual
    spatial neighbourhood features.

    Parameters
    ----------
    adata
        Original AnnData.
    indices_basal
        Indices of basal/control cells to keep in .X and obs.
    indices_counterfactual
        Indices of cells to use as the counterfactual neighbourhood.
    spatial_column
        Key in .obsm where the spatial features are stored / written.
    precomputed
        If True, sample rows from existing .obsm[spatial_column] of
        counterfactual cells (fast, uses precomputedd features).
        If False (default), rebuild spatial features from scratch via
        compute_spatial_features: a counterfactual connectivity matrix is
        constructed where each basal cell's neighbourhood is reassigned to
        cells from indices_counterfactual.
    n_neighbours
        Only used when precomputed=False. Number of neighbours to sample
        per basal cell from indices_counterfactual. If None, all
        counterfactual cells are used with equal weight (mean aggregation).
    random_state
        Seed for reproducibility.

    Returns
    -------
    adata_cf : AnnData
        Copy of original AnnData (basal cells only) with updated
        .obsm[spatial_column].
    """
    rng = np.random.default_rng(random_state)
    n_basal = len(indices_basal)

    if precomputed:
        adata_cf = adata[indices_basal].copy()
        spatial_counts_cf = adata.obsm[spatial_column][indices_counterfactual]
        idx = rng.integers(0, len(indices_counterfactual), size=n_basal)
        adata_cf.obsm[spatial_column] = spatial_counts_cf[idx]
        return adata_cf

    # Build (n_obs, n_obs) counterfactual connectivity: basal rows → counterfactual cols
    n_obs = adata.n_obs
    n_cf = len(indices_counterfactual)

    if n_neighbours is None:
        rows = np.repeat(np.arange(n_basal), n_cf)
        cols = np.tile(indices_counterfactual, n_basal)
    else:
        sampled = rng.choice(indices_counterfactual, size=(n_basal, n_neighbours), replace=True)
        rows = np.repeat(np.arange(n_basal), n_neighbours)
        cols = sampled.ravel()

    data = np.ones(len(rows), dtype=np.float32)
    C_cf = csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))

    # Temporarily attach to adata, delegate aggregation to compute_spatial_features, then clean up
    adata.obsp[_CF_CONN_TMP] = C_cf
    try:
        compute_spatial_features(adata, connectivity_key=_CF_CONN_TMP, obsm_key=_CF_OBSM_TMP)
        adata_cf = adata[indices_basal].copy()
        adata_cf.obsm[spatial_column] = adata_cf.obsm.pop(_CF_OBSM_TMP)
    finally:
        del adata.obsp[_CF_CONN_TMP]
        if _CF_OBSM_TMP in adata.obsm:
            del adata.obsm[_CF_OBSM_TMP]

    return adata_cf
