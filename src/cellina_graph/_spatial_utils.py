import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)
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
                            test_indices=None,
                            spatial_key='spatial'):
    """Core spatial neighbors computation without library_key handling."""
    coordinates = adata.obsm[spatial_key]
    if test_indices is not None and len(test_indices) > 0:
        coordinates = coordinates.copy()
        extent = np.abs(coordinates).max() + 1.0
        test_arr = np.asarray(test_indices)
        coordinates[test_arr, 0] += extent * 1e6 * (np.arange(len(test_arr)) + 1)
    tree = NearestNeighbors(n_neighbors=max_neighbours + 1, # +1 to exclude self
                            algorithm='ball_tree',
                            metric='euclidean').fit(coordinates)
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
        dist.eliminate_zeros()
    if standardize:
        dist = normalize(dist, axis=1, norm='l1')

    spot_n = dist.shape[0]
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
                      test_indices=None,
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
    test_indices
        Integer indices (into ``adata``) of cells to exclude from being selected as neighbors.
        Their coordinates are displaced to a position that cannot appear in any KNN result.
        Rows for these cells in the output matrix will reflect artificial distances; only the
        column-exclusion (they are never a neighbor of another cell) is guaranteed.
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
        
        test_set = set(test_indices) if test_indices is not None else set()
        mats = []
        ixs = []
        for lib in libs:
            global_ixs = np.where(adata.obs[library_key] == lib)[0]
            ixs.extend(global_ixs)
            local_test = [int(np.searchsorted(global_ixs, gi)) for gi in test_indices
                          if gi in test_set and gi in set(global_ixs.tolist())] if test_indices is not None else None
            mats.append(_spatial_neighbors_core(adata[adata.obs[library_key] == lib],
                                        bandwidth=bandwidth, cutoff=cutoff, max_neighbours=max_neighbours,
                                        kernel=kernel, set_diag=set_diag, zoi=zoi, standardize=standardize,
                                        test_indices=local_test, spatial_key=spatial_key))
        
        ixs = cast(list[int], np.argsort(ixs).tolist())
        dist = block_diag(mats, format="csr")[ixs, :][:, ixs]
    else:
        # Single sample case
        dist = _spatial_neighbors_core(adata, bandwidth=bandwidth, cutoff=cutoff,
                                      max_neighbours=max_neighbours, kernel=kernel,
                                      set_diag=set_diag, zoi=zoi, standardize=standardize,
                                      test_indices=test_indices, spatial_key=spatial_key)

    if inplace:
        if reference is not None:
            adata.obsm[f'{key_added}_connectivities'] = dist
        else:
            adata.obsp[f'{key_added}_connectivities'] = dist
        return None
    else:
        return dist

def _node_perturbation(X, var_idx, perturbations, groupby=None, labels=None,
                       base=np.e, add_shift=False, renormalize=True):
    n_vars = len(var_idx)
    if renormalize:
        row_sums_before = np.asarray(X.sum(axis=1)).ravel()

    neutral = 0.0 if add_shift else 1.0
    if groupby is None:
        transform = np.full(n_vars, neutral, dtype=np.float32)
        for gene, logfc in perturbations.items():
            if gene in var_idx:
                transform[var_idx[gene]] = logfc if add_shift else base ** logfc
    else:
        transform = np.full((X.shape[0], n_vars), neutral, dtype=np.float32)
        for ct, logfc_series in perturbations.items():
            ct_mask = labels == ct
            for gene, logfc in logfc_series.items():
                if gene in var_idx:
                    transform[ct_mask, var_idx[gene]] = logfc if add_shift else base ** logfc

    if add_shift:
        X_arr = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)
        X = np.clip(X_arr + transform, 0, None)
    else:
        # +1 pseudocount so zero-expressed genes can be activated by positive logFC
        X_arr = (X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)) + 1
        X = X_arr * transform

    if renormalize:
        row_sums_after = np.asarray(X.sum(axis=1)).ravel()
        scale_rows = np.where(row_sums_after == 0, 1.0, row_sums_before / row_sums_after)
        X = X.multiply(scale_rows[:, np.newaxis]) if sp.issparse(X) else X * scale_rows[:, np.newaxis]
    return X


def make_perturbed_expression(
    adata: AnnData,
    perturbations: Optional[dict] = None,
    groupby: Optional[str] = None,
    layer_key: str = "counts_cf",
    base: float = np.e,
    add_shift: bool = False,
    renormalize: bool = True,
    inplace: bool = True,
):
    """
    Apply logFC perturbations to counts and store the result as a layer.

    Counts-space analog of cellina's ``make_neighbor_perturbation``. cellina
    scales counts AND aggregates them via the connectivity matrix into a
    precomputed pseudobulk. Here only the scaling step is performed; the GCN
    aggregates over the spatial graph at inference time when this layer is
    supplied via ``cf_layer``.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``adata.X``.
    perturbations
        When ``groupby=None``: ``Dict[str, float]`` mapping gene → logFC,
        applied globally to all cells.
        When ``groupby`` is set: ``Dict[str, pd.Series]`` mapping cell-type
        label → gene-indexed logFC Series; each cell is scaled by its cell
        type's vector. Partial dicts (covering only some cell types) are
        allowed — unspecified cell types pass through unchanged.
        When ``None``, the layer is a copy of ``adata.X``.
    groupby
        Column in ``adata.obs`` used for cell-type-specific perturbations.
    layer_key
        Where to write the perturbed matrix in ``adata.layers``.
    base
        Base for the logFC → fold-change conversion. Default ``np.e``.
    add_shift
        If True, add the logFC value directly to counts rather than
        multiplying by ``base ** logfc``. Result is always a dense matrix.
    renormalize
        If True, rescale each row after perturbation so its sum matches
        the pre-perturbation row sum (library-size invariant).
    inplace
        If True, write to ``adata.layers[layer_key]`` and return None.
        If False, return the perturbed matrix without modifying adata.

    Raises
    ------
    ValueError
        If ``groupby`` is set and ``perturbations`` contains cell-type keys
        not present in ``adata.obs[groupby]``.
    """
    var_idx = {g: i for i, g in enumerate(adata.var_names)}

    if perturbations is not None and groupby is not None:
        obs_cts = set(adata.obs[groupby].unique())
        unknown = set(perturbations) - obs_cts
        if unknown:
            raise ValueError(
                f"perturbations contains cell types not in "
                f"adata.obs['{groupby}']: {unknown}"
            )

    if perturbations:
        var_names_set = set(var_idx)
        if groupby is None:
            skipped = [g for g in perturbations if g not in var_names_set]
        else:
            skipped = [
                g
                for series in perturbations.values()
                for g in series.index
                if g not in var_names_set
            ]
        if skipped:
            logger.warning(
                "%d perturbation gene(s) not in var_names, skipped: %s",
                len(skipped),
                skipped,
            )

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

    if not perturbations:
        X_cf = X.copy()
    else:
        labels = adata.obs[groupby].values if groupby is not None else None
        X_cf = _node_perturbation(
            X, var_idx=var_idx, perturbations=perturbations,
            groupby=groupby, labels=labels, base=base,
            add_shift=add_shift, renormalize=renormalize,
        )

    X_csr = X_cf if sp.issparse(X_cf) else sp.csr_matrix(X_cf)
    if inplace:
        adata.layers[layer_key] = X_csr.tocsr()
        return None
    return X_csr.tocsr()


