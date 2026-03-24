import numpy as np


def make_counterfactual_adata(
    adata,
    indices_basal,
    indices_counterfactual,
    spatial_column,
    sample: bool = False,
    random_state: int = 0,
):
    """
    Create a counterfactual AnnData keeping everything from the original
    except .obsm[spatial_column], which is replaced with sampled spatial counts.

    Parameters
    ----------
    adata
        Original AnnData.
    indices_basal
        Indices of basal/control cells to keep in .X and obs.
    indices_counterfactual
        Indices of counterfactual cells to generate spatial counts from.
    spatial_column
        Column in .obsm containing spatial information (counts of neighbors).
    sample
        If True, generate NB-distributed counts per gene.
        If False, sample rows from existing neighboring cells with replacement.
    random_state
        Seed for reproducibility.

    Returns
    -------
    adata_cf : AnnData
        Copy of original AnnData with updated .obsm[spatial_column] for basal cells.
    """
    rng = np.random.default_rng(random_state)

    # 1. Subset basal cells
    adata_cf = adata[indices_basal].copy()

    # 2. Get spatial counts of counterfactual cells
    spatial_counts_cf = adata.obsm[spatial_column][indices_counterfactual]

    n_basal = len(indices_basal)
    n_genes = spatial_counts_cf.shape[1]

    # 3. Sampling: if true, compute representative NB dist and sample from it
    if sample:
        mu = spatial_counts_cf.mean(axis=0)
        var = spatial_counts_cf.var(axis=0)
        theta = np.maximum((mu**2) / (var - mu + 1e-8), 1e-8)

        spatial_counts_basal_cf = rng.negative_binomial(
            n=theta, p=theta / (theta + mu), size=(n_basal, n_genes)
        )
    # Otherwise just sample from existing neighbors with replacement
    else:
        indices = rng.integers(low=0, high=spatial_counts_cf.shape[0], size=n_basal)
        spatial_counts_basal_cf = spatial_counts_cf[indices]

    # 4. Replace spatial_column in .obsm
    adata_cf.obsm[spatial_column] = spatial_counts_basal_cf

    # 5. Keep original target cells to compare later if needed
    adata_cf.uns["target_cells"] = adata[indices_counterfactual].X.copy()

    return adata_cf
