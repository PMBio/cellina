"""Utilities for perturbation evaluation notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

DEFAULT_GROUPBY = [
    "Fibroblast",
    "Endothelial",
    "Myeloid",
    "T_cell",
    "Epithelial",
    "B_cell",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_LABEL_TO_COARSE = {
    "epi1": "Epithelial", "epi2": "Epithelial",
    "epi3": "Epithelial", "epi4": "Epithelial",
    "fib1": "Fibroblast", "fib2": "Fibroblast",
    "EC": "Endothelial",
    "SMC": "Smooth_muscle",
    "BC": "B_cell",
    "PC_IgA": "Plasma_cell", "PC_IgG": "Plasma_cell", "PC_IgM": "Plasma_cell",
    "TC": "T_cell",
    "mye1": "Myeloid", "mye2": "Myeloid",
    "mast": "Mast_cell",
}


def load_crc_slide(
    slide_id: int = 242,
    data_dir: str = "../../data/crc_wt_cosmx",
    n_top_genes: int = 3000,
    labels_key: str = "coarse_type",
    domains_key: str = "typ",
):
    """Load and preprocess a CRC CosMx slide.

    Parameters
    ----------
    slide_id
        Slide identifier (e.g. 242).
    data_dir
        Directory containing ``crc_{slide_id}.h5ad`` files.
    n_top_genes
        Number of highly variable genes to retain.
    labels_key
        obs column name for coarse cell-type labels.
    domains_key
        obs column name for domain/tissue labels.

    Returns
    -------
    Preprocessed AnnData with:
    - ``obs[labels_key]``: coarse cell-type categories
    - ``obsm['spatial']``: spatial coordinates
    - ``layers['counts']``: raw counts
    - HVG-filtered genes
    """
    adata = sc.read(
        f"{data_dir}/crc_{slide_id}.h5ad",
        backup_url=f"https://zenodo.org/records/15574384/files/{slide_id}.h5ad?download=1",
    )
    adata.obs_names_make_unique()

    adata.obs[labels_key] = adata.obs["ist"].map(_LABEL_TO_COARSE)

    adata = adata[~adata.obs[domains_key].isna()].copy()
    adata = adata[~adata.obs[labels_key].isna()].copy()

    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)

    adata.obs[labels_key] = adata.obs[labels_key].astype("category")
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values
    adata.layers["counts"] = adata.X.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, layer="counts", flavor="seurat_v3", n_top_genes=n_top_genes, subset=True
    )

    return adata


# ---------------------------------------------------------------------------
# Pseudobulk logFC
# ---------------------------------------------------------------------------

def _get_domain_labels(adata, domains_key: str) -> tuple[str, str]:
    """Infer the REF and CRC domain labels from adata.obs[domains_key].

    Scans the unique values for entries containing 'REF' and 'CRC' so the
    exact label format (e.g. '242_REF') never needs to be hardcoded.
    """
    unique = adata.obs[domains_key].astype(str).unique()
    ref_matches = [d for d in unique if "REF" in d]
    crc_matches = [d for d in unique if "CRC" in d]
    if len(ref_matches) != 1:
        raise ValueError(
            f"Expected exactly 1 domain containing 'REF', found: {ref_matches}"
        )
    if len(crc_matches) != 1:
        raise ValueError(
            f"Expected exactly 1 domain containing 'CRC', found: {crc_matches}"
        )
    return ref_matches[0], crc_matches[0]


def compute_pseudobulk_logfc(
    adata,
    labels_key: str = "coarse_type",
    domains_key: str = "typ",
) -> tuple:
    """Pseudobulk sum → normalize → log1p → logFC (CRC − REF) per cell type.

    Parameters
    ----------
    adata
        AnnData with ``layers['counts']``, ``obs[labels_key]``, ``obs[domains_key]``.
    labels_key
        obs column for cell-type labels.
    domains_key
        obs column for domain labels (must contain exactly one value with 'REF'
        and one with 'CRC').

    Returns
    -------
    domain_logfc_df
        DataFrame of shape (n_cell_types, n_genes) with logFC values.
    ref_label
        The REF domain label inferred from the data.
    crc_label
        The CRC domain label inferred from the data.
    """
    import decoupler as dc

    ref_label, crc_label = _get_domain_labels(adata, domains_key)

    pdata = dc.pp.pseudobulk(
        adata=adata,
        sample_col=domains_key,
        groups_col=labels_key,
        mode="sum",
        layer="counts",
    )
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)

    cell_types = [
        ct for ct in pdata.obs[labels_key].unique()
        if (
            ((pdata.obs[domains_key] == ref_label) & (pdata.obs[labels_key] == ct)).any()
            and ((pdata.obs[domains_key] == crc_label) & (pdata.obs[labels_key] == ct)).any()
        )
    ]

    domain_logfc_df = pd.concat(
        [
            pd.Series(
                (
                    pdata[
                        (pdata.obs[domains_key] == crc_label) & (pdata.obs[labels_key] == ct)
                    ].X
                    - pdata[
                        (pdata.obs[domains_key] == ref_label) & (pdata.obs[labels_key] == ct)
                    ].X
                ).flatten(),
                index=pdata.var_names,
                name=ct,
            )
            for ct in cell_types
        ],
        axis=1,
    ).T

    return domain_logfc_df, ref_label, crc_label


# ---------------------------------------------------------------------------
# Expression normalisation helper (used by spatial-prop inference)
# ---------------------------------------------------------------------------

def total_normalize(X, target_sum: float = 1e4) -> np.ndarray:
    """Clip, nan-to-num, then normalise each cell to target_sum counts."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, 0, None)
    X = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-6)
    X = X * target_sum
    return X


# ---------------------------------------------------------------------------
# Train / val / test splitting with a held-out cell type
# ---------------------------------------------------------------------------

def split_indices(adata, holdout_celltype, labels_key='coarse_type', domains_key='typ', seed=0):
    """Create train/val/test splits consistent with notebooks.

    Test: holdout_celltype & typ contains 'CRC'
    Val: 10% of remaining trainval (random)
    """
    if holdout_celltype not in adata.obs[labels_key].unique():
        raise ValueError(f"holdout_celltype '{holdout_celltype}' not found in adata.obs['{labels_key}'] values")

    is_tumor_region = adata.obs[domains_key].astype(str).str.contains('CRC', regex=True)
    is_holdout_ct = adata.obs[labels_key].astype(str) == holdout_celltype
    test_mask = is_tumor_region & is_holdout_ct

    all_idx = np.arange(adata.n_obs)
    test_idx = np.where(test_mask.values)[0]
    trainval_idx = np.setdiff1d(all_idx, test_idx)

    rng = np.random.default_rng(seed)
    n_trainval = trainval_idx.shape[0]
    n_val = max(1, int(0.1 * n_trainval))
    val_idx_rel = rng.choice(np.arange(n_trainval), size=n_val, replace=False)
    val_idx = trainval_idx[val_idx_rel]
    train_idx = np.setdiff1d(trainval_idx, val_idx)

    # annotate is_holdout in adata.obs
    adata.obs['is_holdout'] = False
    if len(test_idx) > 0:
        adata.obs.iloc[test_idx, adata.obs.columns.get_loc('is_holdout')] = True

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Distributional metric helpers
# ---------------------------------------------------------------------------

def _knn_masked_mean(kernel_matrix: torch.Tensor, k: int, exclude_self: bool) -> torch.Tensor:
    """Mean of a kernel matrix restricted to each row's top-k neighbours."""
    mask = torch.zeros_like(kernel_matrix, dtype=torch.bool)
    for i in range(kernel_matrix.shape[0]):
        row = kernel_matrix[i].clone()
        if exclude_self:
            row[i] = -float("inf")
        k_eff = min(k, kernel_matrix.shape[1] - int(exclude_self))
        top_idx = torch.topk(row, k=k_eff).indices
        mask[i, top_idx] = True
    return kernel_matrix[mask].mean()


def _mixing_index(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    n_clusters: int = 2,
    n_pcs: int = 50,
    random_state: int = 0,
) -> float:
    """Mixing index: fraction of predicted cells correctly co-clustered with true cells.

    Reduces dimensionality with PCA, clusters with KMeans, then computes how
    many predicted cells land in clusters that contain the expected proportion
    of true cells.  A value of 1 indicates perfect mixing; 0 indicates no mixing.
    """
    n_pred = X_pred.shape[0]
    n_true = X_true.shape[0]
    expected_proportion = n_pred / n_true

    X_combined = np.vstack([X_pred, X_true])
    n_pcs_eff = min(n_pcs, X_combined.shape[1], X_combined.shape[0] - 1)
    pca = PCA(n_components=n_pcs_eff, random_state=random_state)
    X_reduced = pca.fit_transform(X_combined)

    batch_labels = np.array(["pred"] * n_pred + ["true"] * n_true)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_reduced)

    n_correctly_mixed = 0
    for cluster_id in range(n_clusters):
        in_cluster = cluster_labels == cluster_id
        n_pred_in = (batch_labels[in_cluster] == "pred").sum()
        n_true_in = (batch_labels[in_cluster] == "true").sum()
        n_correctly_mixed += min(n_pred_in, n_true_in * expected_proportion)

    return n_correctly_mixed / n_pred


def _standard_edistance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Standard energy distance between two sample sets.

    E(X, Y) = 2 * E[||x - y||] - E[||x - x'||] - E[||y - y'||]

    A value of 0 means the distributions are identical; larger values indicate
    greater distributional discrepancy.

    Parameters
    ----------
    X, Y
        2-D arrays of shape (n_cells, n_features).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)

    mean_dist_xy = torch.cdist(Xt, Yt, p=2).mean()
    mean_dist_xx = torch.cdist(Xt, Xt, p=2).mean()
    mean_dist_yy = torch.cdist(Yt, Yt, p=2).mean()

    return (2 * mean_dist_xy - mean_dist_xx - mean_dist_yy).item()


# ---------------------------------------------------------------------------
# Core metrics function
# ---------------------------------------------------------------------------

def compute_cf_logfc(
    ref_expr: np.ndarray,
    pert_expr: np.ndarray,
    obs_expr: np.ndarray,
    top_n: int | None = 100,
    gene_names: list | None = None,
    mixing_n_clusters: int = 2,
    mixing_n_pcs: int = 50,
    random_state: int = 0,
) -> dict:
    """Compute logFC metrics comparing perturbation prediction against ground truth.

    Parameters
    ----------
    ref_expr
        Normalized expression of reference (REF) cells, shape (n_ref, n_genes).
    pert_expr
        Model-predicted perturbed expression of REF cells, shape (n_ref, n_genes).
    obs_expr
        Real target (CRC) expression — ground truth, shape (n_crc, n_genes).
    top_n
        Restrict correlation metrics to the top_n genes by absolute real_logfc.
        If None, uses all genes.
    gene_names
        Gene names corresponding to columns of the expression matrices.
    mixing_n_clusters
        Number of KMeans clusters for mixing index.
    mixing_n_pcs
        Number of PCA components before clustering.
    random_state
        Random seed.

    Returns
    -------
    dict with keys:
        pearson_r, pearson_p, spearman_r, spearman_p, precision, mixing_index,
        edistance, real_logfc, pred_logfc, top_n_mask, gene_names
    """
    ref_mean = np.log1p(ref_expr.mean(0))
    crc_mean = np.log1p(obs_expr.mean(0))
    pred_mean = np.log1p(pert_expr.mean(0))

    real_logfc = crc_mean - ref_mean   # ground truth: CRC vs REF
    pred_logfc = pred_mean - ref_mean  # model prediction: perturbed vs REF

    # top-n gene selection by |real_logfc|
    top_n_mask = None
    precision = float("nan")

    if top_n is not None:
        n_select = min(top_n, len(real_logfc))
        top_real_idx = set(np.argsort(np.abs(real_logfc))[-n_select:])
        top_pred_idx = set(np.argsort(np.abs(pred_logfc))[-n_select:])

        top_n_mask = np.zeros(len(real_logfc), dtype=bool)
        top_n_mask[list(top_real_idx)] = True

        precision = len(top_real_idx & top_pred_idx) / n_select

        real_eval = real_logfc[top_n_mask]
        pred_eval = pred_logfc[top_n_mask]
    else:
        real_eval = real_logfc
        pred_eval = pred_logfc

    pearson_r, pearson_p = pearsonr(real_eval, pred_eval)
    spearman_r, spearman_p = spearmanr(real_eval, pred_eval)

    mix_idx = _mixing_index(
        pert_expr, obs_expr,
        n_clusters=mixing_n_clusters,
        n_pcs=mixing_n_pcs,
        random_state=random_state,
    )

    edist = _standard_edistance(pert_expr, obs_expr)

    return dict(
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        precision=precision,
        mixing_index=mix_idx,
        edistance=edist,
        real_logfc=real_logfc,
        pred_logfc=pred_logfc,
        top_n_mask=top_n_mask,
        gene_names=gene_names,
    )
