#!/usr/bin/env python
"""Aggregate the sweep results and draw the sensitivity line plot.

Reads every ``k*_seed*.json`` produced by run_sensitivity.py, writes a tidy CSV,
and plots Pearson r (observed vs. predicted edge-perturbation logFC) as a function
of the neighborhood size k, with a mean line and a +/- SD band across seeds.
"""
import argparse
import glob
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    p.add_argument("--results", type=str, default=default_dir,
                   help="Directory containing k*_seed*.json result files.")
    p.add_argument("--out-prefix", type=str, default=None,
                   help="Output path prefix (default: <results>/graph_sensitivity).")
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    out_prefix = args.out_prefix or os.path.join(args.results, "graph_sensitivity")

    files = sorted(glob.glob(os.path.join(args.results, "k*_seed*.json")))
    if not files:
        raise SystemExit(f"No result JSONs found in {args.results}")

    rows = []
    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows).sort_values(["k", "seed"]).reset_index(drop=True)

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} runs, "
          f"{df['k'].nunique()} k values, {df['seed'].nunique()} seeds)")

    # aggregate across seeds
    agg = (df.groupby("k")["pearson"]
             .agg(["mean", "std", "count"])
             .reset_index()
             .sort_values("k"))
    agg["std"] = agg["std"].fillna(0.0)  # single-seed k -> no band

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.fill_between(agg["k"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                    alpha=0.2, color="#1f77b4", linewidth=0, label="±1 SD")
    ax.plot(agg["k"], agg["mean"], "-o", color="#1f77b4", zorder=3, label="mean")
    # individual seed points (jitter-free, translucent) for transparency
    ax.scatter(df["k"], df["pearson"], s=18, color="#1f77b4", alpha=0.35,
               zorder=2, linewidths=0, label="per-seed")

    ax.set_xscale("log")
    ax.set_xticks(agg["k"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Neighborhood size  k  (max_neighbours, bandwidth = ∞)")
    ax.set_ylabel("Pearson r  (observed vs. predicted logFC, top-DE genes)")
    ax.set_title("Cellina sensitivity to neighbor-graph construction\n"
                 "(edge perturbation, held-out CRC Myeloids)")
    # de-duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=9, loc="best")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")

    # console summary
    print("\nk        mean_r   sd_r    n")
    for _, r in agg.iterrows():
        print(f"{int(r['k']):<7} {r['mean']:.4f}  {r['std']:.4f}  {int(r['count'])}")


if __name__ == "__main__":
    main()
