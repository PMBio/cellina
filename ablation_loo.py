"""
Ablation study: LOO cell-type × spatial_loss_type × link_prediction_weight.

Runs HOLDOUT_CELLTYPES × SPATIAL_LOSS_TYPES × LINK_PREDICTION_WEIGHTS combinations
(skipping domain_clf at lpw=0, which is equivalent to supcon).
Evaluates counterfactuals for both CRC and TVA targets.
Saves results to CSV after every run, then writes a PDF summary.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scib_metrics

from cellina_graph import CellinaModel
from cellina_graph._spatial_utils import spatial_neighbors
from perturb_utils import load_crc_slide, split_indices, compute_cf_logfc

# ── Config ───────────────────────────────────────────────────────────────────
HOLDOUT_CELLTYPES    = ["Epithelial", "T_cell", "Myeloid"]
SPATIAL_LOSS_TYPES   = ["supcon", "domain_clf"]
LINK_PREDICTION_WEIGHTS = [0, 0.01, 0.1, 1]

SLIDE_ID      = 242
LABELS_KEY    = "coarse_type"
DOMAINS_KEY   = "typ"
TOP_N         = 50
MIN_CELLS     = 50
BATCH_SIZE    = 512
LIBRARY_SIZE  = 1e4
DEVICES       = [1]
MAX_EPOCHS     = 50

RESULTS_CSV = "results/ablation_loo.csv"
RESULTS_PDF = "results/ablation_loo.pdf"

CF_METRICS   = ["pearson_r", "spearman_r", "precision", "mixing_index"]
RUN_METRICS  = ["marginal_ll", "ari", "nmi"]
METRICS      = CF_METRICS + RUN_METRICS

SLT_COLORS   = {"supcon": "#4C72B0", "domain_clf": "#e05c5c"}
SLT_MARKERS  = {"supcon": "o", "domain_clf": "s"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize_total(arr, target_sum=LIBRARY_SIZE):
    return arr / arr.sum(axis=1, keepdims=True) * target_sum


def run_one(adata_base, holdout_celltype, link_prediction_weight, spatial_loss_type):
    """Train one model and return a list of per-cell-type / per-target result dicts."""
    print(
        f"\n{'='*60}\n"
        f"  holdout={holdout_celltype}  "
        f"spatial_loss_type={spatial_loss_type}  "
        f"link_prediction_weight={link_prediction_weight}\n"
        f"{'='*60}"
    )

    adata = adata_base.copy()

    train_idx, val_idx, test_idx = split_indices(
        adata,
        holdout_celltype=holdout_celltype,
        labels_key=LABELS_KEY,
        domains_key=DOMAINS_KEY,
        holdout_domains=("CRC", "TVA"),
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
        spatial_loss_type=spatial_loss_type,
        condition_on_intrinsic=False,
        use_batch_norm=False if spatial_loss_type == "supcon" else True, # NOTE: batch norm might not be consistent with neigh means in supcon
    )

    model.train(
        max_epochs=MAX_EPOCHS,
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

    # ── Per-run scalars ───────────────────────────────────────────────────────
    marginal_ll = model.get_marginal_ll(adata, indices=model.test_indices_, reduce="mean")
    print(f"  marginal_ll={marginal_ll:.4f}")

    adata.obsm["s"] = model.get_latent_representation(latent_key="s")
    rng = np.random.default_rng(seed=42)
    sub_idx = rng.choice(adata.n_obs, size=int(adata.n_obs * 0.25), replace=False)
    ari_nmi = scib_metrics.nmi_ari_cluster_labels_kmeans(
        labels=adata.obs[DOMAINS_KEY].values[sub_idx],
        X=adata.obsm["s"][sub_idx],
    )
    ari = float(ari_nmi["ari"])
    nmi = float(ari_nmi["nmi"])
    print(f"  ARI={ari:.4f}  NMI={nmi:.4f}")

    # ── Build target domain pools (mirrors notebook cell 22) ─────────────────
    domains = [d for d in adata.obs[DOMAINS_KEY].astype(str).unique() if d != "nan"]
    ref_label  = next(d for d in domains if "REF" in d)
    crc_label  = next(d for d in domains if "CRC" in d)
    tva_labels = [d for d in domains if "TVA" in d]
    target_labels = [crc_label] + tva_labels

    cell_types = [
        ct for ct in adata.obs[LABELS_KEY].cat.categories
        if ((adata.obs[DOMAINS_KEY] == ref_label) & (adata.obs[LABELS_KEY] == ct)).any()
        and any(
            ((adata.obs[DOMAINS_KEY] == tl) & (adata.obs[LABELS_KEY] == ct)).any()
            for tl in target_labels
        )
    ]

    target_pools = {
        tl: np.where(adata.obs[DOMAINS_KEY].astype(str) == tl)[0]
        for tl in target_labels
    }

    rows = []
    for target_label in target_labels:
        target_all_idx = target_pools[target_label]
        target_short = next(k for k in ("CRC", "TVA") if k in target_label)

        for ct in sorted(cell_types):
            ref_mask    = (adata.obs[LABELS_KEY] == ct) & (adata.obs[DOMAINS_KEY] == ref_label)
            target_mask = (adata.obs[LABELS_KEY] == ct) & (adata.obs[DOMAINS_KEY] == target_label)

            if ct == holdout_celltype:
                target_mask = target_mask & adata.obs["is_holdout"]

            ref_idx    = np.where(ref_mask.values)[0]
            target_idx = np.where(target_mask.values)[0]

            if len(ref_idx) < MIN_CELLS or len(target_idx) < MIN_CELLS:
                print(f"  skip {ct} ({target_short}): ref={len(ref_idx)}, target={len(target_idx)}")
                continue

            print(f"  {ct} ({target_short}): ref={len(ref_idx)}, target={len(target_idx)}")

            ref_arr  = adata[ref_idx].layers["counts"]
            ref_expr = ref_arr.toarray() if hasattr(ref_arr, "toarray") else np.asarray(ref_arr)
            ref_expr = _normalize_total(ref_expr)

            tgt_arr  = adata[target_idx].layers["counts"]
            cf_expr  = tgt_arr.toarray() if hasattr(tgt_arr, "toarray") else np.asarray(tgt_arr)
            cf_expr  = _normalize_total(cf_expr)

            pert_expr = model.get_counterfactual_expression(
                indices=ref_idx,
                neighbour_indices=target_all_idx,
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
                spatial_loss_type=spatial_loss_type,
                link_prediction_weight=link_prediction_weight,
                target=target_short,
                cell_type=ct,
                is_holdout=(ct == holdout_celltype),
                n_ref=len(ref_idx),
                n_target=len(target_idx),
                pearson_r=stats["pearson_r"],
                spearman_r=stats["spearman_r"],
                precision=stats["precision"],
                mixing_index=stats["mixing_index"],
                marginal_ll=marginal_ll,
                ari=ari,
                nmi=nmi,
            ))

    return rows


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _plot_lines(ax, df, x_col, y_col, title, xlabel="link_prediction_weight"):
    """Plot one line per spatial_loss_type on ax."""
    for slt, grp in df.groupby("spatial_loss_type"):
        grp = grp.sort_values(x_col)
        ax.plot(
            grp[x_col], grp[y_col],
            marker=SLT_MARKERS[slt],
            color=SLT_COLORS[slt],
            linewidth=1.8,
            label=slt,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.set_xticks(LINK_PREDICTION_WEIGHTS)
    ax.legend(fontsize=7)


def generate_pdf(results_df, out_path):
    """Per-holdout pages + summary pages."""
    with PdfPages(out_path) as pdf:

        # ── Per-holdout pages ─────────────────────────────────────────────────
        for holdout_ct in HOLDOUT_CELLTYPES:
            sub = results_df[results_df["holdout_celltype"] == holdout_ct]
            if sub.empty:
                continue

            for target in sorted(sub["target"].unique()):
                tsub = sub[sub["target"] == target]

                holdout_sub = tsub[tsub["is_holdout"]].copy()
                nonhold_sub = tsub[~tsub["is_holdout"]].copy()

                # Average non-holdout metrics across cell types per (lpw, slt)
                nonhold_mean = (
                    nonhold_sub
                    .groupby(["link_prediction_weight", "spatial_loss_type"])[CF_METRICS]
                    .mean()
                    .reset_index()
                )

                # per-run scalars deduplicated
                run_sub = (
                    tsub.drop_duplicates(["link_prediction_weight", "spatial_loss_type"])
                    [["link_prediction_weight", "spatial_loss_type"] + RUN_METRICS]
                )

                n_metrics = len(METRICS)
                ncols = 3
                nrows = (n_metrics + ncols - 1) // ncols

                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
                axes = axes.flatten()

                for i, metric in enumerate(METRICS):
                    ax = axes[i]
                    if metric in RUN_METRICS:
                        _plot_lines(ax, run_sub, "link_prediction_weight", metric, metric)
                    else:
                        # holdout cell type rows
                        for slt, grp in holdout_sub.groupby("spatial_loss_type"):
                            grp = grp.sort_values("link_prediction_weight")
                            ax.plot(
                                grp["link_prediction_weight"], grp[metric],
                                marker=SLT_MARKERS[slt], color=SLT_COLORS[slt],
                                linewidth=2, label=f"{holdout_ct} (holdout) [{slt}]",
                            )
                        # non-holdout mean
                        for slt, grp in nonhold_mean.groupby("spatial_loss_type"):
                            grp = grp.sort_values("link_prediction_weight")
                            ax.plot(
                                grp["link_prediction_weight"], grp[metric],
                                marker=SLT_MARKERS[slt], color=SLT_COLORS[slt],
                                linewidth=1.5, linestyle="--",
                                label=f"non-holdout mean [{slt}]",
                            )
                        ax.set_title(metric, fontsize=10)
                        ax.set_xlabel("link_prediction_weight", fontsize=8)
                        ax.set_ylabel(metric, fontsize=8)
                        ax.set_xticks(LINK_PREDICTION_WEIGHTS)
                        ax.legend(fontsize=6)

                for j in range(n_metrics, len(axes)):
                    axes[j].set_visible(False)

                fig.suptitle(f"Holdout: {holdout_ct}  |  target: {target}", fontsize=13, y=1.01)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # ── Summary pages (one per target) ────────────────────────────────────
        for target in sorted(results_df["target"].unique()):
            tsub = results_df[results_df["target"] == target]

            holdout_agg = (
                tsub[tsub["is_holdout"]]
                .groupby(["link_prediction_weight", "spatial_loss_type"])[CF_METRICS]
                .mean()
                .reset_index()
            )
            id_agg = (
                tsub[~tsub["is_holdout"]]
                .groupby(["link_prediction_weight", "spatial_loss_type"])[CF_METRICS]
                .mean()
                .reset_index()
            )
            run_agg = (
                tsub
                .drop_duplicates(["holdout_celltype", "link_prediction_weight", "spatial_loss_type"])
                .groupby(["link_prediction_weight", "spatial_loss_type"])[RUN_METRICS]
                .mean()
                .reset_index()
            )

            # Layout: rows = [LOO, ID, run-scalars], cols = metrics per group
            # We'll make 3 sub-figures: LOO panel (4 CF metrics), ID panel (4 CF metrics),
            # run panel (3 run metrics)
            sections = [
                ("LOO cell type (avg)", holdout_agg, CF_METRICS),
                ("ID cell types (avg)",  id_agg,      CF_METRICS),
                ("Per-run scalars",       run_agg,     RUN_METRICS),
            ]

            ncols = 4
            nrows = len(sections)
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))

            for row_i, (section_title, df_agg, metrics) in enumerate(sections):
                for col_i in range(ncols):
                    ax = axes[row_i, col_i]
                    if col_i < len(metrics):
                        metric = metrics[col_i]
                        _plot_lines(ax, df_agg, "link_prediction_weight", metric, metric)
                        if col_i == 0:
                            ax.set_ylabel(section_title, fontsize=9, labelpad=10)
                    else:
                        ax.set_visible(False)

            fig.suptitle(f"Summary  |  target: {target}", fontsize=14, y=1.01)
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
        done = set(zip(
            existing["holdout_celltype"],
            existing["link_prediction_weight"],
            existing["spatial_loss_type"],
        ))
        all_rows = existing.to_dict("records")
        print(f"Resuming: {len(done)} run(s) already done → {RESULTS_CSV}")
    else:
        done = set()
        all_rows = []

    for holdout_ct in HOLDOUT_CELLTYPES:
        for lpw in LINK_PREDICTION_WEIGHTS:
            for slt in SPATIAL_LOSS_TYPES:
                # lpw=0 makes spatial_loss_type irrelevant; run only once under 'supcon'
                if lpw == 0 and slt == "domain_clf":
                    continue

                if (holdout_ct, lpw, slt) in done:
                    print(f"  skip (already done): holdout={holdout_ct}, lpw={lpw}, slt={slt}")
                    continue

                rows = run_one(adata_base, holdout_ct, lpw, slt)
                all_rows.extend(rows)

                pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
                print(f"  Saved {len(all_rows)} rows → {RESULTS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    generate_pdf(results_df, RESULTS_PDF)


if __name__ == "__main__":
    main()
