import logging
from typing import List, Optional

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ._constants import SPATIAL_X_KEY

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
        The following options are available: ['gaussian', 'exponential', 'linear']
    set_diag
        Logical, sets connectivity diagonal to 0 if `False`. Default is `False`.
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
    families = ['gaussian', 'exponential', 'linear']
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

def _node_perturbation(X, var_idx, perturbations, groupby=None, labels=None, base=np.e, add_shift=False, renormalize=False):
    n_vars = len(var_idx)
    if renormalize:
        row_sums_before = np.asarray(X.sum(axis=1)).ravel()

    neutral = 0.0 if add_shift else 1.0
    if groupby is None:
        transform = np.full(n_vars, neutral)
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
        X = np.asarray(X + transform)
    else:
        X = X.multiply(transform) if issparse(X) else X * transform

    if renormalize:
        row_sums_after = np.asarray(X.sum(axis=1)).ravel()
        scale_rows = np.where(row_sums_after == 0, 1.0, row_sums_before / row_sums_after)
        X = X.multiply(scale_rows[:, np.newaxis]) if issparse(X) else X * scale_rows[:, np.newaxis]
    return X


def compute_spatial_features(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    groupby: Optional[str] = None,
    neighbor_genes: Optional[List[str]] = None,
    obsm_key: str = SPATIAL_X_KEY,
    perturbations: Optional[dict] = None,
    base: float = np.e,
    add_shift: bool = False,
    renormalize: bool = False,
) -> None:
    """
    Compute spatial neighbourhood features and store them in ``adata.obsm``.
    Expects normalized counts in ``adata.X`` and a spatial connectivity matrix in ``adata.obsp[connectivity_key]``.

    Parameters
    ----------
    adata
        AnnData object.
    connectivity_key
        Key in ``adata.obsp`` for the spatial connectivity matrix.
    groupby
        Column in ``adata.obs`` used to apply cell-type-specific perturbations.
        Each cell is scaled by the logFC vector for its cell type before the
        standard degree-normalised mean aggregation.  When ``None``, perturbations
        are applied globally to all cells.
    neighbor_genes
        Subset of genes to aggregate.  ``None`` means all genes.
    obsm_key
        Key in ``adata.obsm`` where the result is stored.

    Notes
    -----
    The ``spatial_x`` features are a **linear aggregation of raw counts**.
    Perturbations expressed as logFC are applied directly to counts:
    ``X_cf[j, g] = X[j, g] * base^logFC``, which propagates correctly through
    the linear aggregation step.
    """
    C = csr_matrix(adata.obsp[connectivity_key])
    var_names = list(adata.var_names)
    var_idx = {g: i for i, g in enumerate(var_names)}
    # TODO: add a warning if not normalized and handle this better...
    X = adata.X if isinstance(adata.X, csr_matrix) else csr_matrix(adata.X)

    if perturbations:
        var_names_set = set(var_idx)
        if groupby is None:
            skipped = [g for g in perturbations if g not in var_names_set]
        else:
            skipped = [g for ct_s in perturbations.values() for g in ct_s.index
                       if g not in var_names_set]
        if skipped:
            logger.warning("%d perturbation gene(s) not in var_names, skipped: %s",
                           len(skipped), skipped)

    if perturbations:
        labels = adata.obs[groupby].values if groupby is not None else None
        X = _node_perturbation(
            X, var_idx=var_idx, perturbations=perturbations,
            groupby=groupby, labels=labels, base=base, add_shift=add_shift, renormalize=renormalize,
        )

    if neighbor_genes is not None:
        gene_idx = [var_idx[g] for g in neighbor_genes if g in var_idx]
        X = X[:, gene_idx]
    result = C @ X
    degree = np.asarray(C.sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1.0 / np.where(degree == 0, 1.0, degree)
        result = result.multiply(denom) if issparse(result) else result * denom
    adata.obsm[obsm_key] = csr_matrix(result).astype(np.float32)


def make_neighbor_perturbation(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    perturbations: Optional[dict] = None,
    groupby: Optional[str] = None,
    neighbor_genes: Optional[List[str]] = None,
    obsm_key_out: str = "spatial_x_cf",
    base: float = np.e,
    add_shift: bool = False,
    renormalize: bool = False,
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
        When ``groupby=None``: ``Dict[str, float]`` mapping gene → logFC,
        applied globally to all cells.
        When ``groupby`` is set: ``Dict[str, pd.Series]`` mapping cell-type label →
        gene-indexed logFC Series; each cell is scaled by its cell type's vector
        before standard degree-normalised aggregation.
    groupby
        Column in ``adata.obs`` used to apply cell-type-specific perturbations.
        When ``None``, a simple degree-normalised mean is computed.
    neighbor_genes
        Subset of genes to aggregate.  ``None`` means all genes.
    obsm_key_out
        Key in ``adata.obsm`` for the counterfactual spatial features.

    Raises
    ------
    ValueError
        If ``perturbations`` contains cell-type keys that are not present in
        ``adata.obs[groupby]``.  Partial dictionaries (covering only a subset of
        cell types) are allowed — unspecified cell types are left unmodified.
    """
    if perturbations is not None and groupby is not None:
        obs_cts = set(adata.obs[groupby].unique())
        unknown = set(perturbations) - obs_cts
        if unknown:
            raise ValueError(
                f"perturbations contains cell types not found in "
                f"adata.obs['{groupby}']: {unknown}"
            )

    compute_spatial_features(
        adata,
        connectivity_key=connectivity_key,
        groupby=groupby,
        neighbor_genes=neighbor_genes,
        obsm_key=obsm_key_out,
        perturbations=perturbations,
        base=base,
        add_shift=add_shift,
        renormalize=renormalize,
    )


def make_counterfactual_adata(
    adata,
    indices_basal,
    indices_counterfactual,
    spatial_column,
    precomputed: bool = True,
    n_neighbours: int = 50,
    random_state: int = 0,
    connectivity_key: str = "spatial_connectivities",
    cf_conn_key: str = "spatial_connectivities_cf",
    cf_obsm_key: str = "spatial_x_cf",
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
        If True (default), sample rows from existing .obsm[spatial_column]
        of counterfactual cells (fast, uses precomputed features).
        If False, rebuild spatial features from scratch: all edges incident
        on seed nodes are removed from the original connectivity graph and
        replaced with bidirectional edges connecting each seed to nodes
        sampled from indices_counterfactual.
    n_neighbours
        Number of neighbours to sample per basal cell from
        indices_counterfactual without replacement. Defaults to 50.
        When precomputed=False: sample exactly n_neighbours edges per seed.
    random_state
        Seed for reproducibility.
    connectivity_key
        Key in adata.obsp for the original spatial connectivity matrix.
        Only used when precomputed=False.
    cf_conn_key
        Key written to adata.obsp for the counterfactual connectivity matrix.
        Only used when precomputed=False.
    cf_obsm_key
        Key written to adata.obsm for the counterfactual spatial features.
        Written in both precomputed=True and precomputed=False.

    Returns
    -------
    adata_cf : AnnData
        Copy of original AnnData (basal cells only) with updated
        .obsm[spatial_column].
    """
    if precomputed:
        rng = np.random.default_rng(random_state)
        n_basal = len(indices_basal)
        adata_cf = adata[indices_basal].copy()
        spatial_counts_cf = adata.obsm[spatial_column][indices_counterfactual]
        idx = rng.integers(0, len(indices_counterfactual), size=n_basal)
        sampled = spatial_counts_cf[idx]
        adata_cf.obsm[cf_obsm_key] = sampled
        adata_cf.obsm[spatial_column] = sampled
        return adata_cf

    # precomputed=False: rewire connectivity graph, recompute spatial features
    indices_basal = np.asarray(indices_basal)
    indices_counterfactual = np.asarray(indices_counterfactual)
    n_basal = len(indices_basal)
    n_cf = len(indices_counterfactual)
    rng = np.random.default_rng(random_state)

    # Remove all edges incident on seed (basal) nodes
    C_coo = csr_matrix(adata.obsp[connectivity_key]).tocoo()
    src_arr, dst_arr, data_arr = C_coo.row, C_coo.col, C_coo.data
    keep = ~(np.isin(src_arr, indices_basal) | np.isin(dst_arr, indices_basal))
    src_f, dst_f, data_f = src_arr[keep], dst_arr[keep], data_arr[keep]

    # Build counterfactual edges: seed <-> counterfactual pool
    if n_neighbours is None or n_neighbours >= n_cf:
        cf_src = np.repeat(indices_basal, n_cf)
        cf_dst = np.tile(indices_counterfactual, n_basal)
    else:
        cf_src_parts, cf_dst_parts = [], []
        for s in indices_basal:
            chosen = rng.choice(indices_counterfactual, size=n_neighbours, replace=False)
            cf_src_parts.append(np.full(n_neighbours, s, dtype=indices_basal.dtype))
            cf_dst_parts.append(chosen)
        cf_src = np.concatenate(cf_src_parts)
        cf_dst = np.concatenate(cf_dst_parts)

    # Make bidirectional and merge with filtered original edges
    cf_src_bi = np.concatenate([cf_src, cf_dst])
    cf_dst_bi = np.concatenate([cf_dst, cf_src])
    cf_data = np.ones(len(cf_src_bi), dtype=np.float32)

    C_cf = csr_matrix(
        (np.concatenate([data_f.astype(np.float32), cf_data]),
         (np.concatenate([src_f, cf_src_bi]), np.concatenate([dst_f, cf_dst_bi]))),
        shape=(adata.n_obs, adata.n_obs),
    )
    C_cf.sum_duplicates()

    adata.obsp[cf_conn_key] = C_cf
    compute_spatial_features(adata, connectivity_key=cf_conn_key, obsm_key=cf_obsm_key)
    adata_cf = adata[indices_basal].copy()
    adata_cf.obsm[spatial_column] = adata_cf.obsm[cf_obsm_key]
    
    return adata_cf
