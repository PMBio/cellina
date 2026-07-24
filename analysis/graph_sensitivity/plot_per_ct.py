#!/usr/bin/env python
"""Aggregate the per-cell-type sensitivity sweep -> CSV, LaTeX table, overlay plot.

Reads ``results_per_ct/<ct>/k*_seed*.json`` (cell type inferred from the parent
directory, robust to reused JSONs that predate the holdout_ct field). Emits:
  * <out>_long.csv    tidy per-run rows
  * <out>_wide.csv    mean +/- SD Pearson r, rows=k, cols=cell type
  * <out>.tex         LaTeX table (rows=k, one column per cell type)
  * <out>.{png,pdf}   overlay: Pearson r vs k, one line per cell type
"""
import argparse
import glob
import json
import os


# user-facing cell-type column order
CT_ORDER = ["Fibroblast", "Endothelial", "Myeloid", "T_cell", "Epithelial"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results_per_ct")
    p.add_argument("--results-root", type=str, default=default_root,
                   help="Root dir containing <cell_type>/k*_seed*.json.")
    p.add_argument("--out-prefix", type=str, default=None,
                   help="Output prefix (default: <results-root>/graph_sensitivity_per_ct).")
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    out_prefix = args.out_prefix or os.path.join(args.results_root,
                                                 "graph_sensitivity_per_ct")

    files = sorted(glob.glob(os.path.join(args.results_root, "*", "k*_seed*.json")))
    if not files:
        raise SystemExit(f"No result JSONs found under {args.results_root}/<ct>/")

    rows = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        d["cell_type"] = os.path.basename(os.path.dirname(f))  # robust source of truth
        rows.append(d)
    df = pd.DataFrame(rows)

    keep = ["cell_type", "k", "seed", "pearson", "n_deg", "n_control",
            "n_target", "n_donor_pool"]
    long = df[[c for c in keep if c in df.columns]].sort_values(
        ["cell_type", "k", "seed"]).reset_index(drop=True)
    long.to_csv(f"{out_prefix}_long.csv", index=False)
    print(f"Wrote {out_prefix}_long.csv  ({len(long)} runs)")

    # aggregate mean +/- SD across seeds
    agg = (df.groupby(["cell_type", "k"])["pearson"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)

    ks = sorted(df["k"].unique())
    cts = [c for c in CT_ORDER if c in set(df["cell_type"])]
    cts += [c for c in sorted(df["cell_type"].unique()) if c not in cts]

    # wide mean +/- SD table (string cells)
    def cell(ct, k):
        r = agg[(agg.cell_type == ct) & (agg.k == k)]
        if r.empty:
            return ""
        return f"{r['mean'].iloc[0]:.3f} +/- {r['std'].iloc[0]:.3f}"

    wide = pd.DataFrame({"k": ks})
    for ct in cts:
        wide[ct] = [cell(ct, k) for k in ks]
    wide.to_csv(f"{out_prefix}_wide.csv", index=False)
    print(f"Wrote {out_prefix}_wide.csv")

    # ---- LaTeX table -----------------------------------------------------
    col_fmt = "r" + "c" * len(cts)
    header = "$k$ & " + " & ".join(c.replace("_", r"\_") for c in cts) + r" \\"
    lines = [
        r"\begin{center}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for k in ks:
        cells = []
        for ct in cts:
            r = agg[(agg.cell_type == ct) & (agg.k == k)]
            cells.append(f"${r['mean'].iloc[0]:.3f} \\pm {r['std'].iloc[0]:.3f}$"
                         if not r.empty else "--")
        lines.append(f"{k:<5} & " + " & ".join(cells) + r" \\")
    n_seed = int(df["seed"].nunique())
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\noindent\small\textit{Table: Sensitivity of Cellina to neighbour-graph "
        r"construction, resolved by cell type. With the kernel bandwidth fixed to "
        r"$\infty$ (unweighted, within-domain $k$NN graph), we vary the neighbourhood "
        r"size $k$ and report the Pearson correlation between observed and predicted "
        r"log-fold changes over the top-50 DE genes for the edge-perturbation "
        r"counterfactual on held-out tumour cells of each type (mean $\pm$ SD over "
        f"{n_seed} seeds). Cross-domain neighbours are excluded. Performance is "
        r"stable across a broad range of $k$ and declines only at extreme $k$ as the "
        r"neighbourhood mean over-smooths toward the domain average.}\normalsize",
    ]
    with open(f"{out_prefix}.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {out_prefix}.tex")

    # ---- overlay plot ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, ct in enumerate(cts):
        a = agg[agg.cell_type == ct].sort_values("k")
        color = cmap(i)
        ax.fill_between(a["k"], a["mean"] - a["std"], a["mean"] + a["std"],
                        alpha=0.15, color=color, linewidth=0)
        ax.plot(a["k"], a["mean"], "-o", color=color, label=ct, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Neighborhood size  k  (max_neighbours, bandwidth = ∞)")
    ax.set_ylabel("Pearson r  (observed vs. predicted logFC, top-50 DE genes)")
    ax.set_title("Cellina sensitivity to neighbor-graph construction, per cell type\n"
                 "(within-domain kNN, edge perturbation on held-out tumour cells)")
    ax.legend(frameon=False, fontsize=9, title="held-out cell type")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Wrote {out_prefix}.{ext}")

    # console summary
    print("\n" + wide.to_string(index=False))


if __name__ == "__main__":
    main()
