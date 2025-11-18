from anndata import AnnData
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

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

def weighted_pseudobulks(adata, sp, groupby, obsm_key='spatial_pseodobulks', binarize=True):
    # Ensure `sp` is a sparse matrix
    if not isinstance(sp, csr_matrix):
        sp = csr_matrix(sp)

    # Get unique cell types and their indices
    cell_types = adata.obs[groupby].unique()
    n_cells, n_genes = adata.shape
    n_cell_types = len(cell_types)

    # Initialize an array to store results
    weighted_pseudobulks = np.zeros((n_cells, n_cell_types, n_genes))

    # Precompute gene expression matrix as dense
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    
    if binarize:
        X = np.where(X > 0, 1., 0.)

    # Precompute masks for cell types
    cell_type_masks = {cell_type: np.where(adata.obs[groupby] == cell_type)[0] for cell_type in cell_types}

    # Loop through each cell type and compute weighted averages
    for idx, cell_type in enumerate(cell_types):
        # Get the indices for cells of this type
        cell_indices = cell_type_masks[cell_type]

        # Subset adjacency matrix for neighbors of this cell type
        sp_subset = sp[:, cell_indices]  # Shape: (n_cells, n_cells_of_type)

        # Extract gene expression for cells of this type
        cell_type_expr = X[cell_indices, :]  # Shape: (n_cells_of_type, n_genes)

        # Compute the weighted sum of gene expressions
        weighted_sums = sp_subset.dot(cell_type_expr)  # Shape: (n_cells, n_genes)

        # Compute the sum of weights for neighbors of this type
        neighbor_weights = sp_subset.sum(axis=1)  # Shape: (n_cells, 1)

        # Normalize to get the weighted average
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_avg = np.nan_to_num(weighted_sums / neighbor_weights)

        # Store the result in the corresponding slice
        weighted_pseudobulks[:, idx, :] = weighted_avg

    # Set names for celltype-gene features
    cell_ids = np.unique(adata.obs[groupby].values)
    gene_ids = adata.var.index.values
    spatial_var = [a + "_" + b for a in cell_ids for b in gene_ids]

    weighted_pseudobulks = weighted_pseudobulks.reshape(weighted_pseudobulks.shape[0], -1)

    adata.obsm[obsm_key] = weighted_pseudobulks
    adata.uns["_spatial_var"] = spatial_var

