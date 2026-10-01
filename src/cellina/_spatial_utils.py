import logging
import warnings
from typing import List, Optional, Sequence, Tuple, Union

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
    return np.clip(connectivity, a_min=0, a_max=1.0)

def _spatial_neighbors_core(adata: AnnData,
                           bandwidth=None,
                           cutoff=0.1,
                           max_neighbours=100,
                           kernel='gaussian',
                           set_diag=False,
                           zoi=0,
                           standardize=False,
                           test_indices=None,
                           reference=None,
                           spatial_key='spatial'):
    """Core spatial neighbors computation without library_key handling."""
    coordinates = adata.obsm[spatial_key]

    if test_indices is not None and len(test_indices) > 0:
        coordinates = coordinates.astype(float)
        extent = np.abs(coordinates).max() + 1.0
        test_arr = np.asarray(test_indices)
        coordinates[test_arr, 0] += extent * 1e6 * (np.arange(len(test_arr)) + 1)

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
    # Drop the explicit zeros just introduced
    dist.eliminate_zeros()
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
    test_indices
        Integer indices (into ``adata``) of cells to exclude from being selected as neighbors.
        Each test cell's coordinates are displaced to a unique far-away position so it cannot
        appear in any kNN result — resulting in all-zero rows and columns for those cells.
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

        test_set = set(test_indices) if test_indices is not None else set()
        mats = []
        ixs = []
        for lib in libs:
            global_ixs = np.where(adata.obs[library_key] == lib)[0]
            ixs.extend(global_ixs)
            local_test = (
                [int(np.searchsorted(global_ixs, gi)) for gi in test_indices
                 if gi in test_set and gi in set(global_ixs.tolist())]
                if test_indices is not None else None
            )
            mats.append(_spatial_neighbors_core(adata[adata.obs[library_key] == lib],
                                        bandwidth=bandwidth, cutoff=cutoff, max_neighbours=max_neighbours,
                                        kernel=kernel, set_diag=set_diag, zoi=zoi, standardize=standardize,
                                        test_indices=local_test, reference=reference, spatial_key=spatial_key))

        ixs = cast(list[int], np.argsort(ixs).tolist())
        dist = block_diag(mats, format="csr")[ixs, :][:, ixs]
    else:
        # Single sample case
        dist = _spatial_neighbors_core(adata, bandwidth=bandwidth, cutoff=cutoff,
                                      max_neighbours=max_neighbours, kernel=kernel,
                                      set_diag=set_diag, zoi=zoi, standardize=standardize,
                                      test_indices=test_indices, reference=reference,
                                      spatial_key=spatial_key)

    if inplace:
        if reference is not None:
            adata.obsm[f'{key_added}_connectivities'] = dist
        else:
            adata.obsp[f'{key_added}_connectivities'] = dist
        return None
    else:
        return dist

def _node_perturbation(X, var_idx, perturbations, groupby=None, labels=None,
                       base=np.e, add_shift=False, renormalize=False,
                       perturb_fraction=1.0, random_state=0):
    if not 0.0 <= perturb_fraction <= 1.0:
        raise ValueError(f"perturb_fraction must be in [0, 1], got {perturb_fraction}")
    n_vars = len(var_idx)
    neutral = 0.0 if add_shift else 1.0

    def _encode(logfc):
        return logfc if add_shift else base ** logfc

    if renormalize:
        row_sums_before = np.asarray(X.sum(axis=1)).ravel()

    # Build the per-gene (optionally per-cell-type) transform
    if groupby is None:
        transform = np.full(n_vars, neutral, dtype=np.float32)
        for gene, logfc in perturbations.items():
            if gene in var_idx:
                transform[var_idx[gene]] = _encode(logfc)
    else:
        transform = np.full((X.shape[0], n_vars), neutral, dtype=np.float32)
        for ct, logfc_series in perturbations.items():
            ct_mask = labels == ct
            for gene, logfc in logfc_series.items():
                if gene in var_idx:
                    transform[ct_mask, var_idx[gene]] = _encode(logfc)

    # Densify once
    X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float32)

    # Keep the unperturbed expression so that, when only a fraction of cells is
    # perturbed, the remaining cells can be restored to their original values.
    X_orig = X.copy() if perturb_fraction < 1.0 else None

    if add_shift:  # log1p data + logFC shift
        X = X + transform
    else:          # counts; exp(logFC) applied multiplicatively
        # +1 / -1 avoids zeroing out genes multiplied by zero
        X = (X + 1.0) * transform - 1.0
    X = np.clip(X, 0, None)

    # Fractional perturbation: only a random subset of cells keeps the shift;
    # the rest are reverted to their original expression. Applied before
    # renormalisation so reverted cells trivially keep their original row sums.
    if perturb_fraction < 1.0:
        n_cells = X.shape[0]
        n_sel = int(round(perturb_fraction * n_cells))
        rng = np.random.default_rng(random_state)
        selected = np.zeros(n_cells, dtype=bool)
        if n_sel > 0:
            selected[rng.choice(n_cells, size=n_sel, replace=False)] = True
        X[~selected] = X_orig[~selected]

    if renormalize:
        row_sums_after = np.asarray(X.sum(axis=1)).ravel()
        scale_rows = np.where(row_sums_after == 0, 1.0, row_sums_before / row_sums_after)
        X = X * scale_rows[:, np.newaxis]

    return csr_matrix(X)


def _make_perturbed_expression(
    adata: AnnData,
    perturbations: dict,
    groupby: Optional[str] = None,
    base: float = np.e,
    add_shift: bool = True,
    renormalize: bool = True,
    perturb_fraction: float = 1.0,
    random_state: int = 0,
    layer: Optional[str] = None,
):
    """Apply Node perturbations to the source expression and return the modified matrix.

    The source is ``adata.layers[layer]`` when ``layer`` is given, else ``adata.X``.
    """
    var_names = list(adata.var_names)
    var_idx = {g: i for i, g in enumerate(var_names)}
    var_names_set = set(var_idx)

    if groupby is None:
        skipped = [g for g in perturbations if g not in var_names_set]
    else:
        skipped = [g for ct_s in perturbations.values() for g in ct_s.index
                   if g not in var_names_set]
    if skipped:
        logger.warning("%d perturbation gene(s) not in var_names, skipped: %s",
                       len(skipped), skipped)

    source = adata.layers[layer] if layer is not None else adata.X
    X = source if isinstance(source, csr_matrix) else csr_matrix(source)
    labels = adata.obs[groupby].values if groupby is not None else None
    return _node_perturbation(
        X, var_idx=var_idx, perturbations=perturbations,
        groupby=groupby, labels=labels, base=base, add_shift=add_shift, renormalize=renormalize,
        perturb_fraction=perturb_fraction, random_state=random_state,
    )


def _looks_like_raw_counts(matrix, max_sample: int = 10_000) -> bool:
    """Cheap heuristic: are the (sampled) non-zero values integer-valued and large?

    Samples up to ``max_sample`` stored non-zero entries, so no dense copy of the
    full matrix is made.
    """
    try:
        if issparse(matrix):
            values = np.asarray(csr_matrix(matrix).data).ravel()
        else:
            arr = np.asarray(matrix)
            values = arr.ravel()
            values = values[values != 0]
        if values.size == 0:
            return False
        if values.size > max_sample:
            values = values[:: max(1, values.size // max_sample)][:max_sample]
        values = values.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            return False
        return bool(np.all(values == np.rint(values)) and values.max() > 50)
    except Exception:  # never let the heuristic break the computation
        return False


def _warn_if_raw_counts(matrix) -> None:
    if _looks_like_raw_counts(matrix):
        warnings.warn(
            "Spatial features are being aggregated from what looks like raw counts "
            "(`adata.X` holds integer values with a large maximum). Cellina is usually "
            "trained on spatial features built from a normalized representation "
            "(e.g. log1p of CP10K). Pass `layer=` naming the normalized representation "
            "used at training time to keep training and inference consistent.",
            UserWarning,
            stacklevel=3,
        )


def compute_spatial_features(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    neighbor_genes: Optional[List[str]] = None,
    obsm_key: str = SPATIAL_X_KEY,
    layer: Optional[str] = None,
) -> None:
    """
    Compute spatial neighbourhood features and store them in ``adata.obsm``.
    Expects normalized counts in ``adata.X`` (or ``adata.layers[layer]``) and a
    spatial connectivity matrix in ``adata.obsp[connectivity_key]``.

    Parameters
    ----------
    adata
        AnnData object.
    connectivity_key
        Key in ``adata.obsp`` for the spatial connectivity matrix.
    neighbor_genes
        Subset of genes to aggregate.  ``None`` means all genes.
    obsm_key
        Key in ``adata.obsm`` where the result is stored.
    layer
        Key in ``adata.layers`` to use as expression source.
        When ``None``, ``adata.X`` is used.
    """
    C = csr_matrix(adata.obsp[connectivity_key])
    raw = adata.layers[layer] if layer is not None else adata.X
    if layer is None:
        _warn_if_raw_counts(raw)
    X = raw if isinstance(raw, csr_matrix) else csr_matrix(raw)

    if neighbor_genes is not None:
        var_idx = {g: i for i, g in enumerate(adata.var_names)}
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
    perturbations: dict,
    connectivity_key: str = "spatial_connectivities",
    groupby: Optional[str] = None,
    neighbor_genes: Optional[List[str]] = None,
    obsm_key_out: str = "spatial_x_cf",
    layer_key: str = "counts_cf",
    base: float = np.e,
    add_shift: bool = False,
    renormalize: bool = True,
    perturb_fraction: float = 1.0,
    random_state: int = 0,
    layer: Optional[str] = None,
) -> None:
    """
    Apply Node perturbations to neighbour expression and re-aggregate.

    Perturbed expression is written to ``adata.layers[layer_key]`` and the
    resulting spatial features to ``adata.obsm[obsm_key_out]``.
    The original count matrix ``adata.X`` is **not** modified.

    Parameters
    ----------
    adata
        AnnData object.
    connectivity_key
        Key in ``adata.obsp`` for the spatial connectivity matrix.
    perturbations
        When ``groupby=None``: ``Dict[str, float]`` mapping gene → perturbation value
        (:math:`\delta_g`) applied globally to all cells.
        When ``groupby`` is set: ``Dict[str, pd.Series]`` mapping cell-type label →
        gene-indexed perturbation value (:math:`\delta_g`) Series.
        Values are interpreted as additive shifts when ``add_shift=True``
        (:math:`T_g(x) = x + \delta_g`) or as logFCs when ``add_shift=False``
        (:math:`T_g(x) = x \cdot e^{\delta_g}`).
    groupby
        Column in ``adata.obs`` used to apply cell-type-specific perturbations.
        When ``None``, perturbations are applied globally to all cells.
    neighbor_genes
        Subset of genes to aggregate.  ``None`` means all genes.
    obsm_key_out
        Key in ``adata.obsm`` for the counterfactual spatial features.
    layer_key
        Key in ``adata.layers`` where the perturbed counts are stored.
    perturb_fraction
        Fraction of cells (in ``[0, 1]``) that receive the perturbation before
        re-aggregation. A random subset of this size is perturbed and the rest are
        left at their original expression, so after the degree-normalised neighbour
        mean each focal cell effectively sees ``~perturb_fraction`` of its neighbours
        perturbed. ``1.0`` (default) perturbs every cell (original behaviour); ``0.0``
        applies no change. The subset is drawn over *all* cells, so when every cell
        type is covered by ``perturbations`` this equals the fraction of perturbed
        neighbours.
    random_state
        Seed for the random subset selection when ``perturb_fraction < 1``.
    layer
        Key in ``adata.layers`` used as the *source* expression that the
        perturbation is applied to. When ``None``, ``adata.X`` is used.
        This must be the same representation that the training ``spatial_x``
        was built from (e.g. a ``'lognorm'`` layer holding log1p(CP10K)),
        otherwise perturbed spatial features are on a different scale than
        the ones the model was trained on.

    Raises
    ------
    ValueError
        If ``perturbations`` contains cell-type keys that are not present in
        ``adata.obs[groupby]``.  Partial dictionaries (covering only a subset of
        cell types) are allowed — unspecified cell types are left unmodified.
    """
    if groupby is not None:
        obs_cts = set(adata.obs[groupby].unique())
        unknown = set(perturbations) - obs_cts
        if unknown:
            raise ValueError(
                f"perturbations contains cell types not found in "
                f"adata.obs['{groupby}']: {unknown}"
            )

    adata.layers[layer_key] = _make_perturbed_expression(
        adata, perturbations=perturbations, groupby=groupby,
        base=base, add_shift=add_shift, renormalize=renormalize,
        perturb_fraction=perturb_fraction, random_state=random_state,
        layer=layer,
    )

    compute_spatial_features(
        adata,
        connectivity_key=connectivity_key,
        neighbor_genes=neighbor_genes,
        obsm_key=obsm_key_out,
        layer=layer_key,
    )


def make_counterfactual_adata(
    adata,
    indices_basal,
    indices_counterfactual,
    spatial_column,
    anchor_donors: bool = True,
    n_neighbors: int = 50,
    random_state: int = 0,
    connectivity_key: str = "spatial_connectivities",
    cf_conn_key: str = "spatial_connectivities_cf",
    cf_obsm_key: str = "spatial_x_cf",
    layer: Optional[str] = None,
):
    """Counterfactual AnnData: ``indices_basal`` with their spatial features replaced.

    Parameters
    ----------
    adata
        Original AnnData.
    indices_basal
        Cells to keep; their ``.obsm[spatial_column]`` is replaced.
    indices_counterfactual
        Anchor cells when ``anchor_donors=True``, donor pool when ``False``.
    spatial_column
        ``.obsm`` key of the spatial features.
    anchor_donors
        If True (default), each basal cell is paired with one anchor drawn uniformly with
        replacement from ``indices_counterfactual`` and takes over that anchor's stored
        spatial features, so the anchors must have had neighbours when those features were
        computed (after masking them out with ``spatial_neighbors(test_indices=...)`` use
        ``anchor_donors=False``). If False, ``indices_counterfactual`` is a donor pool: every edge
        incident on a basal cell is removed from ``adata.obsp[connectivity_key]``, the cell
        is rewired (bidirectionally) to ``n_neighbors`` donors drawn uniformly without
        replacement, and its spatial features are re-aggregated over the rewired graph.
    n_neighbors
        Donors per basal cell when ``anchor_donors=False``; must be
        ``< len(indices_counterfactual)``.
    random_state
        Seed for the anchor / donor draw.
    connectivity_key
        ``.obsp`` key of the graph rewired when ``anchor_donors=False``.
    cf_conn_key
        ``.obsp`` key the rewired graph is written to (``anchor_donors=False`` only).
    cf_obsm_key
        ``.obsm`` key the counterfactual spatial features are written to.
    layer
        ``adata.layers`` key aggregated over the rewired graph when ``anchor_donors=False``
        (``None``: ``adata.X``). Must match the representation the training spatial
        features were built from (e.g. log1p(CP10K) while ``adata.X`` holds raw counts).

    Returns
    -------
    AnnData of the basal cells with counterfactual ``.obsm[spatial_column]``.
    """
    if anchor_donors:
        rng = np.random.default_rng(random_state)
        n_basal = len(indices_basal)
        adata_cf = adata[indices_basal].copy()
        spatial_counts_cf = adata.obsm[spatial_column][indices_counterfactual]
        idx = rng.integers(0, len(indices_counterfactual), size=n_basal)
        sampled = spatial_counts_cf[idx]
        adata_cf.obsm[cf_obsm_key] = sampled
        adata_cf.obsm[spatial_column] = sampled
        return adata_cf

    # anchor_donors=False: rewire connectivity graph, recompute spatial features
    indices_basal = np.asarray(indices_basal)
    indices_counterfactual = np.asarray(indices_counterfactual)
    n_cf = len(indices_counterfactual)
    rng = np.random.default_rng(random_state)

    # Remove all edges incident on seed (basal) nodes
    C_coo = csr_matrix(adata.obsp[connectivity_key]).tocoo()
    src_arr, dst_arr, data_arr = C_coo.row, C_coo.col, C_coo.data
    keep = ~(np.isin(src_arr, indices_basal) | np.isin(dst_arr, indices_basal))
    src_f, dst_f, data_f = src_arr[keep], dst_arr[keep], data_arr[keep]

    # Build counterfactual edges: seed <-> counterfactual pool
    if n_neighbors is None or n_neighbors >= n_cf:
        raise ValueError(
            f"n_neighbors must be a finite value < n_cf ({n_cf}); got {n_neighbors}. "
            f"Connecting every basal cell to the full counterfactual pool is not supported."
        )
    cf_src_parts, cf_dst_parts = [], []
    for s in indices_basal:
        chosen = rng.choice(indices_counterfactual, size=n_neighbors, replace=False)
        cf_src_parts.append(np.full(n_neighbors, s, dtype=indices_basal.dtype))
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
    compute_spatial_features(adata, connectivity_key=cf_conn_key, obsm_key=cf_obsm_key,
                             layer=layer)
    adata_cf = adata[indices_basal].copy()
    adata_cf.obsm[spatial_column] = adata_cf.obsm[cf_obsm_key]

    return adata_cf


def make_perturbed_expression(
    adata: AnnData,
    perturbations: Optional[dict] = None,
    groupby: Optional[str] = None,
    layer_key: str = "counts_cf",
    base: float = np.e,
    add_shift: bool = False,
    renormalize: bool = True,
    inplace: bool = True,
    layer: Optional[str] = None,
):
    """
    Apply Node perturbations to counts and store the result as a layer.

    Counts-space analog of :func:`make_neighbor_perturbation`. Cellina scales
    counts; the GCN aggregates over the spatial graph at inference time when
    this layer is supplied via ``cf_layer``.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``adata.X``.
    perturbations
        ``Dict[str, float]`` mapping gene → perturbation value (:math:`\delta_g`, global)
        when ``groupby=None``.
        ``Dict[str, pd.Series]`` mapping cell-type label → gene-indexed perturbation
        value (:math:`\delta_g`) Series when ``groupby`` is set.
        Values are interpreted as additive shifts when ``add_shift=True``
        (:math:`T_g(x) = x + \delta_g`) or as logFCs when ``add_shift=False``
        (:math:`T_g(x) = x \cdot e^{\delta_g}`). ``None`` copies ``adata.X`` unchanged.
    groupby
        Column in ``adata.obs`` for cell-type-specific perturbations.
    layer_key
        Key to write in ``adata.layers``.
    base
        Base for logFC → fold-change conversion. Default ``np.e``.
    add_shift
        If True, add the logFC directly to counts instead of multiplying.
    renormalize
        If True, rescale rows after perturbation to preserve library sizes.
    inplace
        If True, write to ``adata.layers[layer_key]`` and return None.
        If False, return the perturbed matrix without modifying adata.
    layer
        Key in ``adata.layers`` used as the *source* expression. When ``None``
        (default), ``adata.X`` is used.

    Raises
    ------
    ValueError
        If ``groupby`` is set and ``perturbations`` contains unknown cell types.
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

    source = adata.layers[layer] if layer is not None else adata.X
    X = source if issparse(source) else csr_matrix(source)

    if not perturbations:
        X_cf = X.copy()
    else:
        labels = adata.obs[groupby].values if groupby is not None else None
        X_cf = _node_perturbation(
            X, var_idx=var_idx, perturbations=perturbations,
            groupby=groupby, labels=labels, base=base,
            add_shift=add_shift, renormalize=renormalize,
        )

    if add_shift:
        result = np.asarray(X_cf.todense() if issparse(X_cf) else X_cf, dtype=np.float32)
    else:
        result = X_cf.tocsr() if issparse(X_cf) else csr_matrix(X_cf)
    if inplace:
        adata.layers[layer_key] = result
        return None
    return result



def _sample_anchor_donors(connectivity, indices, anchor_indices, exclude=None, seed=0):
    """Per-cell donor sets for the anchor draw: one random anchor per focal cell, whose
    neighbours in ``connectivity`` (minus ``exclude`` and the focal cell) become the donors.

    Anchors with no neighbour outside ``exclude`` (homotypic anchors) are dropped with a
    warning and never drawn. Returns a list of 1-D int64 arrays, one per entry of
    ``indices``. Raises if no anchor is left.
    """
    indices = np.asarray(indices)
    anchor_indices = np.asarray(anchor_indices)
    for name, arr in (("indices", indices), ("neighbour_indices", anchor_indices)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must have an integer dtype, got {arr.dtype}.")
    if anchor_indices.size == 0:
        raise ValueError("neighbour_indices is empty; at least one anchor cell is required.")

    # copy=True: eliminate_zeros / sum_duplicates must not touch the caller's adata.obsp.
    conn = csr_matrix(connectivity, copy=True)
    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError(f"connectivity must be a square matrix, got shape {conn.shape}.")
    n_obs = conn.shape[0]
    for name, arr in (("indices", indices), ("neighbour_indices", anchor_indices)):
        if arr.size and (arr.min() < 0 or arr.max() >= n_obs):
            raise ValueError(f"{name} contains cells outside [0, {n_obs}).")
    conn.sum_duplicates()
    conn.eliminate_zeros()

    excluded = np.zeros(n_obs, dtype=bool)
    if exclude is not None:
        exclude = np.asarray(exclude)
        if exclude.size:
            if exclude.ndim != 1 or not np.issubdtype(exclude.dtype, np.integer):
                raise ValueError("exclude_indices must be a 1-D integer array.")
            excluded[exclude] = True

    # Admissible neighbourhood per anchor; drop anchors that have none (homotypic).
    neigh = {}
    for a in np.unique(anchor_indices):
        nb = conn.indices[conn.indptr[a]:conn.indptr[a + 1]]
        neigh[a] = nb[~excluded[nb]]
    usable = np.array([neigh[a].size > 0 for a in anchor_indices], dtype=bool)
    n_homotypic = int((~usable).sum())
    if n_homotypic == anchor_indices.size:
        raise ValueError(
            "All anchors are homotypic: no neighbour left outside `exclude_indices`. "
            "Check `connectivity_key` (an anchor masked out of the training graph has no "
            "neighbours there)."
        )
    if n_homotypic:
        warnings.warn(
            f"{n_homotypic} of {anchor_indices.size} anchors are homotypic (no neighbour "
            "outside `exclude_indices`) and were not used.",
            UserWarning, stacklevel=3,
        )
    valid_anchors = anchor_indices[usable]

    rng = np.random.default_rng(seed)
    anchors = rng.choice(valid_anchors, size=len(indices), replace=True)

    donors = []
    for focal, anchor in zip(indices, anchors):
        nb = neigh[anchor]
        nb = nb[nb != focal]
        # The anchor's only admissible neighbour may be the focal cell itself: redraw.
        while nb.size == 0:
            anchor = rng.choice(valid_anchors)
            nb = neigh[anchor]
            nb = nb[nb != focal]
        donors.append(np.ascontiguousarray(nb, dtype=np.int64))
    return donors
