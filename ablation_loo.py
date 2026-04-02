"""
Ablation study: LOO cell-type × link_prediction_weight.

Runs 9 = 3 holdout cell types × 3 link_prediction_weight values,
saves results to CSV after every run, then writes a PDF summary.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cellina_graph import CellinaModel
from cellina_graph._spatial_utils import spatial_neighbors
from perturb_utils import load_crc_slide, split_indices, compute_cf_logfc

# ── Config ───────────────────────────────────────────────────────────────────
HOLDOUT_CELLTYPES       = ["Epithelial", "T_cell", "Fibroblast"]
LINK_PREDICTION_WEIGHTS = [0, 0.1, 1]

SLIDE_ID      = 232
LABELS_KEY    = "coarse_type"
DOMAINS_KEY   = "typ"
TOP_N         = 100
MIN_CELLS     = 50
BATCH_SIZE    = 512
LIBRARY_SIZE  = 1e4
DEVICES       = [1]

RESULTS_CSV = "results/ablation_loo.csv"
RESULTS_PDF = "results/ablation_loo.pdf"

METRICS = ["pearson_r", "spearman_r", "precision", "mixing_index", "marginal_ll"]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize_total(arr, target_sum=LIBRARY_SIZE):
    return arr / arr.sum(axis=1, keepdims=True) * target_sum


def run_one(adata_base, holdout_celltype, link_prediction_weight):
    """Train one model and return a list of per-cell-type result dicts."""
    print(
        f"\n{'='*60}\n"
        f"  holdout={holdout_celltype}  "
        f"link_prediction_weight={link_prediction_weight}\n"
        f"{'='*60}"
    )

    adata = adata_base.copy()

    train_idx, val_idx, test_idx = split_indices(
        adata,
        holdout_celltype=holdout_celltype,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
    )
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    CellinaModel.setup_anndata(
        adata,
        batch_key=None,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
        layer="counts",
        spatial_connectivities_key="spatial_connectivities",
    )

    model = CellinaModel(
        adata,
        n_latent=20,
        convolution_type="gat",
        n_layers=3,
        classifier_lambda=1,
        discriminator_lambda=1,
        link_prediction_weight=link_prediction_weight,
        condition_on_intrinsic=False,
    )

    model.train(
        max_epochs=50,
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_monitor="vae_loss_validation",
        train_size=0.9,
        validation_size=0.1,
        plan_kwargs={"lr": 1e-3, "weight_decay": 0.0001, "normalize_losses": True},
        datasplitter_kwargs={"external_indexing": [train_idx, val_idx, test_idx]},
        enable_checkpointing=True,
        batch_size=BATCH_SIZE,
        devices=DEVICES,
    )

    marginal_ll = model.get_marginal_ll(adata, indices=model.test_indices_, reduce="mean")
    print(f"  marginal_ll={marginal_ll:.4f}")

    # Domain labels
    domains = adata.obs[DOMAINS_KEY].astype(str).unique()
    ref_label = next(d for d in domains if "REF" in d)
    crc_label = next(d for d in domains if "CRC" in d)

    cell_types = [
        ct for ct in adata.obs[LABELS_KEY].cat.categories
        if ((adata.obs[DOMAINS_KEY] == ref_label) & (adata.obs[LABELS_KEY] == ct)).any()
        and ((adata.obs[DOMAINS_KEY] == crc_label) & (adata.obs[LABELS_KEY] == ct)).any()
    ]
    crc_all_idx = np.where(adata.obs[DOMAINS_KEY].astype(str).str.contains("CRC"))[0]

    rows = []
    for ct in sorted(cell_types):
        ref_mask = (adata.obs[LABELS_KEY] == ct) & (adata.obs[DOMAINS_KEY] == ref_label)
        crc_mask = (adata.obs[LABELS_KEY] == ct) & (adata.obs[DOMAINS_KEY] == crc_label)

        if ct == holdout_celltype:
            crc_mask = crc_mask & adata.obs["is_holdout"]

        ref_idx = np.where(ref_mask.values)[0]
        crc_idx = np.where(crc_mask.values)[0]

        if len(ref_idx) < MIN_CELLS or len(crc_idx) < MIN_CELLS:
            print(f"  skip {ct}: ref={len(ref_idx)}, crc={len(crc_idx)}")
            continue

        print(f"  {ct}: ref={len(ref_idx)}, crc={len(crc_idx)}")

        ref_arr = adata[ref_idx].layers["counts"]
        ref_expr = ref_arr.toarray() if hasattr(ref_arr, "toarray") else np.asarray(ref_arr)
        ref_expr = _normalize_total(ref_expr)

        crc_arr = adata[crc_idx].layers["counts"]
        cf_expr = crc_arr.toarray() if hasattr(crc_arr, "toarray") else np.asarray(crc_arr)
        cf_expr = _normalize_total(cf_expr)

        pert_expr = model.get_counterfactual_expression(
            indices=ref_idx,
            neighbour_indices=crc_all_idx,
            batch_size=BATCH_SIZE,
            n_neighbors_per_seed=30,
            library_size=LIBRARY_SIZE,
        )

        stats = compute_cf_logfc(
            ref_expr, pert_expr, cf_expr,
            top_n=TOP_N,
            gene_names=adata.var_names.tolist(),
        )

        rows.append(dict(
            holdout_celltype=holdout_celltype,
            link_prediction_weight=link_prediction_weight,
            cell_type=ct,
            is_holdout=(ct == holdout_celltype),
            n_ref=len(ref_idx),
            n_crc=len(crc_idx),
            pearson_r=stats["pearson_r"],
            spearman_r=stats["spearman_r"],
            precision=stats["precision"],
            mixing_index=stats["mixing_index"],
            marginal_ll=marginal_ll,
        ))

    return rows


def generate_pdf(results_df, out_path):
    """One page per holdout cell type; metrics vs link_prediction_weight."""
    ncols = 3
    nrows = 2  # 5 metrics → 2 rows × 3 cols (last slot empty)

    with PdfPages(out_path) as pdf:
        for holdout_ct in HOLDOUT_CELLTYPES:
            sub = results_df[results_df["holdout_celltype"] == holdout_ct]

            holdout_sub  = sub[sub["is_holdout"]].sort_values("link_prediction_weight")
            nonhold_mean = (
                sub[~sub["is_holdout"]]
                .groupby("link_prediction_weight")[METRICS[:-1]]  # marginal_ll is per-run
                .mean()
                .reset_index()
                .sort_values("link_prediction_weight")
            )
            # marginal_ll is the same for all cell types in a run → grab from holdout rows
            mll_sub = (
                sub[["link_prediction_weight", "marginal_ll"]]
                .drop_duplicates()
                .sort_values("link_prediction_weight")
            )

            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
            axes = axes.flatten()

            for i, metric in enumerate(METRICS):
                ax = axes[i]
                if metric == "marginal_ll":
                    ax.plot(
                        mll_sub["link_prediction_weight"],
                        mll_sub["marginal_ll"],
                        marker="o", color="#e05c5c", linewidth=2, label=holdout_ct,
                    )
                else:
                    ax.plot(
                        holdout_sub["link_prediction_weight"],
                        holdout_sub[metric],
                        marker="o", color="#e05c5c", linewidth=2, label=f"{holdout_ct} (holdout)",
                    )
                    ax.plot(
                        nonhold_mean["link_prediction_weight"],
                        nonhold_mean[metric],
                        marker="s", color="#4C72B0", linewidth=1.5,
                        linestyle="--", label="non-holdout mean",
                    )
                    ax.legend(fontsize=8)

                ax.set_title(metric, fontsize=11)
                ax.set_xlabel("link_prediction_weight")
                ax.set_ylabel(metric)
                ax.set_xticks(LINK_PREDICTION_WEIGHTS)

            # hide unused subplot
            for j in range(len(METRICS), len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(f"Holdout: {holdout_ct}", fontsize=14, y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved PDF → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)

    print("Loading data and computing spatial graph...")
    adata_base = load_crc_slide(SLIDE_ID, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY)
    spatial_neighbors(adata_base, bandwidth=100 / 0.12028, max_neighbours=50, standardize=False)

    # Resume from existing CSV if present
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        done = set(zip(existing["holdout_celltype"], existing["link_prediction_weight"]))
        all_rows = existing.to_dict("records")
        print(f"Resuming: {len(done)} run(s) already done → {RESULTS_CSV}")
    else:
        done = set()
        all_rows = []

    for holdout_ct in HOLDOUT_CELLTYPES:
        for lpw in LINK_PREDICTION_WEIGHTS:
            if (holdout_ct, lpw) in done:
                print(f"  skip (already done): holdout={holdout_ct}, lpw={lpw}")
                continue

            rows = run_one(adata_base, holdout_ct, lpw)
            all_rows.extend(rows)

            pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
            print(f"  Saved {len(all_rows)} rows → {RESULTS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    generate_pdf(results_df, RESULTS_PDF)


if __name__ == "__main__":
    main()
