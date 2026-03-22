from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ._constants import SPATIAL_X_KEY

# Spatial Kernels
def _gaussian(distance_mtx, bandwidth):
    return np.exp(-(distance_mtx ** 2.0) / (2.0 * bandwidth ** 2.0))

def _exponential(distance_mtx, bandwidth):
    return np.exp(-distance_mtx / bandwidth)

def _linear(distance_mtx, bandwidth):
    connectivity = 1 - distance_mtx / bandwidth
    return np.clip(connectivity, a_min=0, a_max=np.inf)

def _spatial_neighbors_core(adata: AnnData,
                           bandwidth=None,
                           cutoff=0.1,
                           max_neighbours=100,
                           kernel='gaussian',
                           set_diag=False,
                           zoi=0,
                           standardize=False,
                           reference=None,
                           spatial_key='spatial'):
    """Core spatial neighbors computation without library_key handling."""
    coordinates = adata.obsm[spatial_key]

    if reference is None:
        _reference = coordinates
    else:
        _reference = reference

    tree = NearestNeighbors(n_neighbors=max_neighbours + 1, # +1 to exclude self
                            algorithm='ball_tree',
                            metric='euclidean').fit(_reference)
    dist = tree.kneighbors_graph(coordinates, mode='distance')

    # prevent float overflow
    bandwidth = np.array(bandwidth, dtype=np.float64)

    # define zone of indifference
    dist.data[dist.data < zoi] = np.inf

    # NOTE: dist gets converted to a connectivity (proximity) matrix
    if kernel == 'gaussian':
        dist.data = _gaussian(dist.data, bandwidth)
    elif kernel == 'misty_rbf':
        dist.data = _misty_rbf(dist.data, bandwidth)
    elif kernel == 'exponential':
        dist.data = _exponential(dist.data, bandwidth)
    elif kernel == 'linear':
        dist.data = _linear(dist.data, bandwidth)
    else:
        raise ValueError("Please specify a valid family to generate connectivity weights")

    if not set_diag:
        dist.setdiag(0)
    if cutoff is not None:
        dist.data = dist.data * (dist.data > cutoff)
    if standardize:
        dist = normalize(dist, axis=1, norm='l1')

    spot_n = dist.shape[0]
    if reference is None:
        assert spot_n == adata.shape[0]
    if spot_n > 1000:
        dist = dist.astype(np.float32)

    return dist


def spatial_neighbors(adata: AnnData,
                      bandwidth=None,
                      cutoff=0.1,
                      max_neighbours=100,
                      kernel='gaussian',
                      set_diag=False,
                      zoi=0,
                      standardize=False,
                      reference=None,
                      spatial_key='spatial',
                      key_added='spatial',
                      library_key=None,
                      inplace=True
                      ):
    """
    Generate spatial connectivity weights using Euclidean distance.

    Parameters
    ----------
    %(adata)s
    bandwidth
         Denotes signaling length (`l`) and controls the maximum distance at which two spots are considered.
         Corresponds to the units in which spatial coordinates are expressed.
    cutoff
        Values below this cutoff will be set to 0.
    max_neighbours
        Maximum nearest neighbours to be considered when generating spatial connectivity weights.
        Essentially, the maximum number of edges in the spatial connectivity graph.
    kernel
        Kernel function used to generate connectivity weights.
        It controls the shape of the connectivity weights.
        The following options are available: ['gaussian', 'exponential', 'linear', 'misty_rbf']
    set_diag
        Logical, sets connectivity diagonal to 0 if `False`. Default is `True`.
    zoi
        Zone of indifference. Values below this cutoff will be set to `np.inf`.
    standardize
        Whether to (l1) standardize spatial proximities (connectivities) so that they sum to 1.
        This plays a role when weighing border regions prior to downstream methods, as the number of spots
        in the border region (and hence the sum of proximities) is smaller than the number of spots in the center.
        Relevant for methods with unstandardized scores (e.g. product). Default is `False`.
    reference
        Reference coordinates to use when generating spatial connectivity weights.
        If `None`, uses the spatial coordinates in `adata.obsm[spatial_key]`.
        This is only relevant if you want to use a different set of coordinates to generate spatial connectivity weights.
    %(spatial_key)s
    key_added
        Key to add to `adata.obsp` if `inplace = True`. If reference is not `None`, key will be added to `adata.obsm`.
    library_key
        Key in adata.obs for grouping samples. If provided, builds separate graphs per sample and concatenates them.
    %(inplace)s

    Notes
    -----
    This function is adapted from mistyR, and is set to be consistent with
    the `squidpy.gr.spatial_neighbors` function in the `squidpy` package.

    Returns
    -------
    If ``inplace = False``, returns an `np.array` with spatial connectivity weights.
    Otherwise, modifies the ``adata`` object with the following key:
        - :attr:`anndata.AnnData.obsp` ``['{key_added}_connectivities']`` with the aforementioned array

    """
    if cutoff is None:
        raise ValueError("`cutoff` must be provided!")
    assert spatial_key in adata.obsm
    families = ['gaussian', 'exponential', 'linear', 'misty_rbf']
    if kernel not in families:
        raise AssertionError(f"{kernel} must be a member of {families}")
    if bandwidth is None:
        raise ValueError("Please specify a bandwidth")

    # Handle library_key for sample-wise graph building
    if library_key is not None:
        from scipy.sparse import block_diag
        from typing import cast
        from anndata.utils import make_index_unique

        libs = adata.obs[library_key].cat.categories
        make_index_unique(adata.obs_names)

        mats = []
        ixs = []
        for lib in libs:
            ixs.extend(np.where(adata.obs[library_key] == lib)[0])
            mats.append(_spatial_neighbors_core(adata[adata.obs[library_key] == lib],
                                        bandwidth=bandwidth, cutoff=cutoff, max_neighbours=max_neighbours,
                                        kernel=kernel, set_diag=set_diag, zoi=zoi, standardize=standardize,
                                        reference=reference, spatial_key=spatial_key))

        ixs = cast(list[int], np.argsort(ixs).tolist())
        dist = block_diag(mats, format="csr")[ixs, :][:, ixs]
    else:
        # Single sample case
        dist = _spatial_neighbors_core(adata, bandwidth=bandwidth, cutoff=cutoff,
                                      max_neighbours=max_neighbours, kernel=kernel,
                                      set_diag=set_diag, zoi=zoi, standardize=standardize,
                                      reference=reference, spatial_key=spatial_key)

    if inplace:
        if reference is not None:
            adata.obsm[f'{key_added}_connectivities'] = dist
        else:
            adata.obsp[f'{key_added}_connectivities'] = dist
        return None
    else:
        return dist


# ---------------------------------------------------------------------------
# Private aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_simple(X_sub: np.ndarray, C: csr_matrix) -> np.ndarray:
    """Degree-normalised mean of neighbour expression."""
    degree = np.array(C.sum(axis=1))  # (n_cells, 1)
    result = C @ X_sub  # (n_cells, n_genes_sub)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.nan_to_num(result / degree)
    return result


def _aggregate_pseudobulk(
    X: np.ndarray,
    C: csr_matrix,
    labels: np.ndarray,
    binarize: bool,
) -> tuple[np.ndarray, list[str]]:
    """Per-cell-type pseudobulk aggregation of neighbour expression.
    
    Returns
    -------
    result : np.ndarray, shape (n_cells, n_cell_types * n_genes)
    spatial_var : list of str, feature names ``<cell_type>_<gene>``
    """
    if binarize:
        X = np.where(X > 0, 1.0, 0.0)

    cell_types = np.unique(labels)
    n_cells, n_genes = X.shape
    n_cell_types = len(cell_types)

    out = np.zeros((n_cells, n_cell_types, n_genes))

    for idx, ct in enumerate(cell_types):
        ct_idx = np.where(labels == ct)[0]
        sp_sub = C[:, ct_idx]
        ct_expr = X[ct_idx, :]
        weighted_sums = sp_sub.dot(ct_expr)
        neighbor_weights = sp_sub.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[:, idx, :] = np.nan_to_num(weighted_sums / neighbor_weights)

    out = out.reshape(n_cells, -1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_spatial_features(
    adata: AnnData,
    sp,
    groupby: Optional[str] = None,
    neighbor_genes: Optional[List[str]] = None,
    binarize: bool = False,
    obsm_key: str = SPATIAL_X_KEY,
) -> None:
    """
    Compute spatial neighbourhood features and store them in ``adata.obsm``.

    Parameters
    ----------
    adata
        AnnData object.
    sp
        Sparse spatial connectivity matrix (n_cells × n_cells).
    groupby
        Column in ``adata.obs`` to use for per-cell-type pseudobulk aggregation.
        When ``None`` (default), a simple degree-normalised mean over all
        neighbours is computed (faster, less memory).
    neighbor_genes
        Subset of genes to aggregate.  ``None`` means all genes.
        Only used when ``groupby=None``.
    binarize
        Binarise counts before aggregation (pseudobulk mode only).
    obsm_key
        Key in ``adata.obsm`` where the result is stored.

    Notes
    -----
    The ``spatial_x`` features are a **linear aggregation of raw counts**.
    Perturbations expressed as logFC are applied directly to counts:
    ``X_cf[j, g] = X[j, g] * 2^logFC``, which propagates correctly through
    the linear aggregation step.
    """
    if not isinstance(sp, csr_matrix):
        sp = csr_matrix(sp)

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    if groupby is None:
        # Simple mean path
        genes = list(adata.var_names) if neighbor_genes is None else list(neighbor_genes)
        gene_idx = [list(adata.var_names).index(g) for g in genes]
        X_sub = X[:, gene_idx]
        result = _aggregate_simple(X_sub, sp)
    else:
        # Per-cell-type pseudobulk path
        labels = adata.obs[groupby].values
        result = _aggregate_pseudobulk(X, sp, labels, binarize)

    adata.obsm[obsm_key] = result


def make_neighbor_perturbation(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    perturbations: Optional[Dict[str, float]] = None,
    cell_type_perturbations: Optional[pd.DataFrame] = None,
    groupby: Optional[str] = None,
    neighbor_genes: Optional[List[str]] = None,
    binarize: bool = False,
    obsm_key_out: str = "spatial_x_cf",
) -> None:
    """
    Apply logFC perturbations to neighbour expression and re-aggregate.

    Perturbed spatial features are written to ``adata.obsm[obsm_key_out]``.
    The original count matrix ``adata.X`` is **not** modified.

    Parameters
    ----------
    adata
        AnnData object.
    connectivity_key
        Key in ``adata.obsp`` for the spatial connectivity matrix.
    perturbations
        Dict mapping gene name → logFC (base-2).  Applied to **all** cells.
        ``X_cf[j, g] = X[j, g] * 2^logFC``.
    cell_type_perturbations
        DataFrame of shape ``(genes × cell_types)`` with logFC values.
        Rows are gene names, columns are cell-type labels matching ``groupby``.
        Requires ``groupby`` to be set.
    groupby
        Column in ``adata.obs`` used for per-cell-type pseudobulk aggregation.
        When ``None``, a simple degree-normalised mean is computed.
    neighbor_genes
        Subset of genes to aggregate.  ``None`` means all genes.
        Only used when ``groupby=None``.
    binarize
        Binarise counts before aggregation (pseudobulk mode only).
    obsm_key_out
        Key in ``adata.obsm`` for the counterfactual spatial features.

    Raises
    ------
    ValueError
        If ``cell_type_perturbations`` is passed without ``groupby``.
    """
    if cell_type_perturbations is not None and groupby is None:
        raise ValueError(
            "cell_type_perturbations requires groupby to be set "
            "(pseudobulk aggregation mode)."
        )

    var_names = list(adata.var_names)
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X_cf = X.copy().astype(float)

    # Apply global perturbations
    if perturbations:
        for gene, logfc in perturbations.items():
            if gene not in var_names:
                raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
            g_idx = var_names.index(gene)
            X_cf[:, g_idx] *= 2.0 ** logfc

    # Apply per-cell-type perturbations
    if cell_type_perturbations is not None:
        labels = adata.obs[groupby].values
        for gene in cell_type_perturbations.index:
            if gene not in var_names:
                raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
            g_idx = var_names.index(gene)
            for ct in cell_type_perturbations.columns:
                logfc = cell_type_perturbations.loc[gene, ct]
                if logfc == 0.0:
                    continue
                ct_mask = labels == ct
                X_cf[ct_mask, g_idx] *= 2.0 ** logfc

    # Re-aggregate with perturbed counts
    C = csr_matrix(adata.obsp[connectivity_key])

    if groupby is None:
        genes = list(adata.var_names) if neighbor_genes is None else list(neighbor_genes)
        gene_idx = [var_names.index(g) for g in genes]
        X_sub = X_cf[:, gene_idx]
        result = _aggregate_simple(X_sub, C)
    else:
        result = _aggregate_pseudobulk(X_cf, C, adata.obs[groupby].values, binarize)

    adata.obsm[obsm_key_out] = result


