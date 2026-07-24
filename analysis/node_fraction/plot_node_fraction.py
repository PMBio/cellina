#!/usr/bin/env python
"""Aggregate the node-fraction sweep and draw the dose-response plots.

Reads every ``frac*_seed*.json`` from run_node_fraction.py, writes a tidy CSV and
a LaTeX table, and plots (a) Pearson r, (b) ||predicted logFC||, and (c) number of
perturbed genes at several |logFC| thresholds -- each as a function of the fraction
of neighbours perturbed, with mean +/- SD bands across seeds.
"""
import argparse
import glob
import json
import os
import re


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    p.add_argument("--results", type=str, default=default_dir)
    p.add_argument("--out-prefix", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_prefix = args.out_prefix or os.path.join(args.results, "node_fraction")

    files = sorted(glob.glob(os.path.join(args.results, "frac*_seed*.json")))
    if not files:
        raise SystemExit(f"No result JSONs found in {args.results}")
    df = pd.DataFrame([json.load(open(f)) for f in files])
    df = df.sort_values(["fraction", "seed"]).reset_index(drop=True)

    thr_cols = sorted([c for c in df.columns if c.startswith("n_genes_gt_")],
                      key=lambda c: float(c.rsplit("_", 1)[1]))

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} runs, {df['fraction'].nunique()} fractions, "
          f"{df['seed'].nunique()} seeds)")

    metrics = ["pearson", "l2_norm"] + thr_cols
    agg = (df.groupby("fraction")[metrics]
             .agg(["mean", "std"]).reset_index().sort_values("fraction"))
    fr = agg["fraction"].values

    def m(col):
        return agg[(col, "mean")].values
    def s(col):
        return np.nan_to_num(agg[(col, "std")].values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Pearson r
    ax = axes[0]
    ax.fill_between(fr, m("pearson") - s("pearson"), m("pearson") + s("pearson"),
                    alpha=0.2, color="#1f77b4", linewidth=0)
    ax.plot(fr, m("pearson"), "-o", color="#1f77b4")
    ax.scatter(df["fraction"], df["pearson"], s=16, color="#1f77b4", alpha=0.3, linewidths=0)
    ax.set_ylabel("Pearson r (obs vs. pred logFC, top-DE genes)")
    ax.set_title("(a) Direction fidelity")

    # (b) magnitude
    ax = axes[1]
    ax.fill_between(fr, m("l2_norm") - s("l2_norm"), m("l2_norm") + s("l2_norm"),
                    alpha=0.2, color="#d62728", linewidth=0)
    ax.plot(fr, m("l2_norm"), "-o", color="#d62728")
    ax.scatter(df["fraction"], df["l2_norm"], s=16, color="#d62728", alpha=0.3, linewidths=0)
    ax.set_ylabel("‖predicted logFC‖₂ (all genes)")
    ax.set_title("(b) Magnitude of predicted change")

    # (c) number of perturbed genes at thresholds
    ax = axes[2]
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(thr_cols):
        t = c.rsplit("_", 1)[1]
        col = cmap(i / max(1, len(thr_cols) - 1))
        ax.fill_between(fr, m(c) - s(c), m(c) + s(c), alpha=0.15, color=col, linewidth=0)
        ax.plot(fr, m(c), "-o", color=col, label=f"|logFC| > {t}")
    ax.set_ylabel("# predicted DE genes (all genes)")
    ax.set_title("(c) Breadth of response")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel("Fraction of neighbours perturbed")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Cellina node perturbation is continuous in the fraction of perturbed neighbours "
                 "(held-out CRC Myeloids, k=200)", y=1.03, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Wrote {out_prefix}.{ext}")

    # ---- LaTeX table -----------------------------------------------------
    def fmt(col):
        return [f"${mm:.3f} \\pm {ss:.3f}$" if col == "pearson"
                else f"${mm:.2f} \\pm {ss:.2f}$" if col == "l2_norm"
                else f"${mm:.0f} \\pm {ss:.0f}$"
                for mm, ss in zip(m(col), s(col))]

    thr_headers = " & ".join([f"$n_{{|\\text{{logFC}}|>{c.rsplit('_',1)[1]}}}$" for c in thr_cols])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{r" + "c" * (2 + len(thr_cols)) + "}", r"\toprule",
        r"Fraction & Pearson $r$ & $\|\text{logFC}\|_2$ & " + thr_headers + r" \\",
        r"\midrule",
    ]
    pear, l2 = fmt("pearson"), fmt("l2_norm")
    thr_fmt = {c: fmt(c) for c in thr_cols}
    for i, f in enumerate(fr):
        row = f"{f:g} & {pear[i]} & {l2[i]} & " + " & ".join(thr_fmt[c][i] for c in thr_cols) + r" \\"
        lines.append(row)
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Node-perturbation dose response. Applying the healthy$\rightarrow$tumour "
        r"logFC shift to an increasing fraction of neighbour cells (held-out CRC Myeloids, "
        r"$k=200$, bandwidth $=\infty$). Direction fidelity (Pearson $r$ over the top-50 "
        r"observed DE genes) stays high while the magnitude ($\|\text{logFC}\|_2$) and the "
        r"number of predicted DE genes grow monotonically with the fraction, showing that "
        r"Cellina models node perturbations continuously. Mean $\pm$ SD over 3 seeds.}",
        r"\label{tab:node_fraction}", r"\end{table}",
    ]
    tex_path = f"{out_prefix}.tex"
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {tex_path}")

    # console summary
    cols_show = ["pearson", "l2_norm"] + thr_cols
    print("\nfraction " + "  ".join(f"{c:>14}" for c in cols_show))
    for i, f in enumerate(fr):
        vals = "  ".join(f"{m(c)[i]:>14.3f}" for c in cols_show)
        print(f"{f:<8.2f} {vals}")


if __name__ == "__main__":
    main()
