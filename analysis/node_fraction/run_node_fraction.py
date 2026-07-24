#!/usr/bin/env python
"""Single-run worker for the node-perturbation *fraction* experiment.

Reuses a pre-trained k=200 Cellina checkpoint (bandwidth=inf uniform kNN graph)
and applies the tutorial's node perturbation (tutorial 4.2) to the held-out CRC
Myeloid population, but shifts only a random *fraction* of neighbour cells
(`perturb_fraction`) before re-aggregation. This is pure inference -- no training.

For each (seed, fraction) it records:
  * pearson   -- Pearson r between observed and predicted logFC over a FIXED
                 top-`n_deg` observed-DE gene set (same set across fractions).
  * l2_norm   -- ||predicted logFC||_2 over ALL genes (magnitude of change).
  * mean_abs  -- mean |predicted logFC| over ALL genes.
  * n_genes_gt_{t} -- #{ genes with |predicted logFC| > t } over ALL genes,
                 for t in --thresholds.

One run == one (seed, fraction). Pin the GPU with CUDA_VISIBLE_DEVICES.
Mirrors docs/tutorial.ipynb 4.2; graph is rebuilt with bandwidth=inf, k=200 to
match how the reused checkpoint was trained.
"""
import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True,
                   help="Model seed (selects the k200_seed{seed} checkpoint) and "
                        "the seed for the fraction mask.")
    p.add_argument("--fraction", type=float, required=True,
                   help="perturb_fraction: fraction of neighbour cells shifted.")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--k", type=int, default=200,
                   help="Neighborhood size of the reused checkpoint / graph.")
    p.add_argument("--data", type=str,
                   default="/data/ddimitrov/repos/cellina/docs/data/crc_232.h5ad")
    p.add_argument("--ckpt-root", type=str,
                   default="/data/ddimitrov/repos/cellina/analysis/graph_sensitivity/runs")
    p.add_argument("--n-deg", type=int, default=50)
    p.add_argument("--n-pert-genes", type=int, default=200)
    p.add_argument("--thresholds", type=str, default="0.25 0.5 1.0",
                   help="Space-separated |logFC| thresholds for the gene-count metric.")
    p.add_argument("--batch-size", type=int, default=2048)
    return p.parse_args()


# ---------------------------------------------------------------------------
# logFC helpers (verbatim from docs/tutorial.ipynb)
# ---------------------------------------------------------------------------
def _normalize_counts(x, eps=1e-8, scale=1e4):
    return x / (x.sum(axis=1, keepdims=True) + eps) * scale


def safe_log2_fold_change(a, b, eps=1e-6):
    import numpy as np
    return np.log2((np.asarray(a) + eps) / (np.asarray(b) + eps))


def get_lfc(control, target, counterfactual, normalize_counts=True, n_deg=200):
    import numpy as np
    if normalize_counts:
        control = _normalize_counts(control)
        target = _normalize_counts(target)
        counterfactual = _normalize_counts(counterfactual)
    mean_control = np.nanmean(control, axis=0)
    mean_target = np.nanmean(target, axis=0)
    mean_cf = np.nanmean(counterfactual, axis=0)
    gt_vec = safe_log2_fold_change(mean_target, mean_control)
    cf_vec = safe_log2_fold_change(mean_cf, mean_control)
    top_features = np.argsort(-np.abs(gt_vec))[:n_deg]
    return gt_vec, cf_vec, top_features


def get_perturbation_logfc(adata, control_domain, holdout_domain, labels_key, domains_key):
    import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc, decoupler as dc
    pdata_ct = dc.pp.pseudobulk(adata=adata, sample_col=domains_key,
                                groups_col=labels_key, mode="sum", layer="counts")
    sc.pp.normalize_total(pdata_ct, target_sum=1e4)
    sc.pp.log1p(pdata_ct)
    cell_types_with_both = [
        ct for ct in pdata_ct.obs[labels_key].unique()
        if ((pdata_ct.obs[domains_key] == control_domain) & (pdata_ct.obs[labels_key] == ct)).any()
        and ((pdata_ct.obs[domains_key] == holdout_domain) & (pdata_ct.obs[labels_key] == ct)).any()
    ]
    _ct_rows = []
    for _ct in cell_types_with_both:
        _crc = pdata_ct[(pdata_ct.obs[domains_key] == holdout_domain) & (pdata_ct.obs[labels_key] == _ct)].X
        _ref = pdata_ct[(pdata_ct.obs[domains_key] == control_domain) & (pdata_ct.obs[labels_key] == _ct)].X
        _crc_m = np.asarray(_crc.mean(axis=0)).flatten() if sp.issparse(_crc) else _crc.mean(axis=0).flatten()
        _ref_m = np.asarray(_ref.mean(axis=0)).flatten() if sp.issparse(_ref) else _ref.mean(axis=0).flatten()
        _ct_rows.append(pd.Series(_crc_m - _ref_m, index=pdata_ct.var_names, name=_ct))
    return pd.concat(_ct_rows, axis=1).T


def get_global_perturbation_logfc(adata, control_domain, holdout_domain, labels_key,
                                  domains_key, holdout_ct):
    import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc, decoupler as dc
    adata_sub = adata[adata.obs[labels_key] != holdout_ct]
    pdata = dc.pp.pseudobulk(adata=adata_sub, sample_col=domains_key,
                             groups_col=None, mode="sum", layer="counts")
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)
    _h = pdata[pdata.obs[domains_key] == holdout_domain].X
    _c = pdata[pdata.obs[domains_key] == control_domain].X
    _hm = np.asarray(_h.mean(axis=0)).flatten() if sp.issparse(_h) else _h.mean(axis=0).flatten()
    _cm = np.asarray(_c.mean(axis=0)).flatten() if sp.issparse(_c) else _c.mean(axis=0).flatten()
    return pd.Series(_hm - _cm, index=pdata.var_names)


def main():
    args = parse_args()
    t0 = time.time()
    thresholds = [float(t) for t in args.thresholds.split()]

    import numpy as np
    import scanpy as sc
    import torch
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr

    from cellina import Cellina, make_neighbor_perturbation
    from cellina._spatial_utils import spatial_neighbors, compute_spatial_features

    os.makedirs(args.outdir, exist_ok=True)
    run_tag = f"frac{args.fraction}_seed{args.seed}"

    labels_key, domains_key, batch_key = "coarse_type", "typ", None
    holdout_ct = "Myeloid"
    control_domain, target_domain = "232_REF", "232_CRC"

    # ---- load + preprocess (identical to the training pipeline) ----------
    adata = sc.read(args.data)
    adata.obs_names_make_unique()
    label_to_coarse = {
        "epi1": "Epithelial", "epi2": "Epithelial", "epi3": "Epithelial", "epi4": "Epithelial",
        "fib1": "Fibroblast", "fib2": "Fibroblast",
        "EC": "Endothelial", "SMC": "Smooth_muscle",
        "BC": "B_cell", "PC_IgA": "Plasma_cell", "PC_IgG": "Plasma_cell", "PC_IgM": "Plasma_cell",
        "TC": "T_cell", "mye1": "Myeloid", "mye2": "Myeloid", "mast": "Mast_cell",
    }
    adata.obs["coarse_type"] = adata.obs["ist"].map(label_to_coarse)
    adata = adata[~adata.obs[domains_key].isna()]
    adata = adata[~adata.obs[labels_key].isna()]
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, layer="counts", flavor="seurat_v3",
                                n_top_genes=2000, subset=True)

    is_tumor_region = adata.obs[domains_key].str.contains("CRC", regex=True)
    is_holdout_ct = adata.obs[labels_key] == holdout_ct
    test_idx = np.where((is_tumor_region & is_holdout_ct))[0]

    adata.obs["is_holdout"] = False
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("is_holdout")] = True

    # (train/val split kept identical to training for provenance; unused here)
    trainval_idx = np.setdiff1d(np.arange(adata.n_obs), test_idx)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1,
                                          random_state=0, shuffle=True)

    # ---- spatial graph: bandwidth=inf, k=200 (matches the checkpoint) ----
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values
    spatial_neighbors(adata, bandwidth=float("inf"), max_neighbours=args.k,
                      standardize=False, test_indices=test_idx)
    compute_spatial_features(adata)
    adata.X = adata.layers["counts"].copy()

    # ---- load the pre-trained k=200 checkpoint --------------------------
    Cellina.setup_anndata(adata, batch_key=batch_key, labels_key=labels_key,
                          domains_key=domains_key, spatial_obsm_key="spatial_x",
                          layer="counts")
    ckpt_dir = os.path.join(args.ckpt_root, f"k{args.k}_seed{args.seed}")
    ckpt_name = os.listdir(ckpt_dir)[0]
    model = Cellina.load(os.path.join(ckpt_dir, ckpt_name), adata=adata)

    # ---- control / target populations -----------------------------------
    mask_control = adata.obs[domains_key].str.contains("REF") & is_holdout_ct
    mask_target = is_tumor_region & is_holdout_ct
    idx_control = np.where(mask_control.values)[0]

    # ---- perturbation logFC (tutorial 4.2) -------------------------------
    domain_logfc_df = get_perturbation_logfc(adata, control_domain, target_domain,
                                             labels_key, domains_key)
    global_logfc = get_global_perturbation_logfc(adata, control_domain, target_domain,
                                                 labels_key, domains_key, holdout_ct)
    # held-out cell type shifted by the GLOBAL change (never sees its own)
    domain_logfc_df.loc[holdout_ct, global_logfc.index] = global_logfc
    logfc_series_dict = {}
    for ct in domain_logfc_df.index:
        s = domain_logfc_df.loc[ct]
        top_g = s.abs().nlargest(args.n_pert_genes).index.tolist()
        logfc_series_dict[ct] = s[top_g]

    # ---- node perturbation at the requested fraction ---------------------
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    make_neighbor_perturbation(
        adata, perturbations=logfc_series_dict, groupby=labels_key,
        obsm_key_out="spatial_x_cf", base=np.e, renormalize=True, add_shift=True,
        perturb_fraction=args.fraction, random_state=args.seed,
    )
    pert_expr = model.get_perturbed_expression(
        adata=adata, indices=idx_control, spatial_obsm_key="spatial_x_cf",
        batch_size=args.batch_size, library_size=1e4)

    # Model's UNPERTURBED prediction for the same cells (real neighbourhoods).
    # Magnitude/breadth are measured against THIS baseline so they capture the
    # perturbation-induced shift in the model's own output, not the constant
    # model-vs-observed reconstruction offset.
    baseline_expr = model.get_perturbed_expression(
        adata=adata, indices=idx_control, spatial_obsm_key="spatial_x",
        batch_size=args.batch_size, library_size=1e4)

    # ---- metrics ---------------------------------------------------------
    control = np.array(adata.layers["counts"][mask_control.values, :].todense())
    target = np.array(adata.layers["counts"][mask_target.values, :].todense())

    # direction fidelity: predicted vs OBSERVED control logFC over top observed DEGs
    true_lfc, pred_lfc, deg = get_lfc(control=control, target=target,
                                      counterfactual=pert_expr, n_deg=args.n_deg)
    pearson, _ = pearsonr(true_lfc[deg], pred_lfc[deg])

    # perturbation-induced shift: predicted counterfactual vs predicted BASELINE
    mean_cf = np.nanmean(_normalize_counts(pert_expr), axis=0)
    mean_base = np.nanmean(_normalize_counts(baseline_expr), axis=0)
    shift_lfc = safe_log2_fold_change(mean_cf, mean_base)   # per gene, all genes
    abs_shift = np.abs(shift_lfc)
    # does the induced shift point in the ground-truth direction?
    pearson_shift, _ = pearsonr(true_lfc[deg], shift_lfc[deg])

    result = {
        "fraction": args.fraction,
        "seed": args.seed,
        "k": args.k,
        "pearson": float(pearson),                         # direction (vs observed control)
        "pearson_shift": float(pearson_shift),             # direction of induced shift
        "l2_norm": float(np.linalg.norm(shift_lfc)),       # magnitude of induced shift, all genes
        "mean_abs": float(abs_shift.mean()),
        "n_genes_total": int(shift_lfc.shape[0]),
        "n_deg": args.n_deg,
        "n_pert_genes": args.n_pert_genes,
        "n_control": int(len(idx_control)),
        "n_target": int(mask_target.values.sum()),
        "checkpoint": ckpt_name,
        "runtime_sec": round(time.time() - t0, 1),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for t in thresholds:
        result[f"n_genes_gt_{t}"] = int((abs_shift > t).sum())

    out_path = os.path.join(args.outdir, f"{run_tag}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{run_tag}] r={pearson:.3f} l2={result['l2_norm']:.2f} "
          f"n>0.5={result.get('n_genes_gt_0.5')}  -> {out_path} "
          f"({result['runtime_sec']}s)")


if __name__ == "__main__":
    main()
