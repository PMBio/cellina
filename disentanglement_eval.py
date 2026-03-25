"""Disentanglement evaluation utilities."""

import os
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import ColumnDefinition, Table
from plottable.plots import bar
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"

WEIGHTS = dict(batch_weight=0.2, bio_weight=0.3, nuisance_weight=0.5)

METHOD_STYLES = {
    'KIARA':          dict(color='black',    weight='bold', style='italic'),
    'SCANVI':         dict(color='black',    weight='bold'),
    'contrastiveVI':  dict(color='black',    weight='bold'),
    'scVIVA':         dict(color='black',    weight='bold'),
    'PCA':            dict(color='gray',     weight='bold'),
    'scVI':           dict(color='gray',     weight='bold'),
    'LDVAE':          dict(color='gray',     weight='bold'),
    'Cell-type only': dict(color='darkblue', weight='bold'),
    'Nuisance only':  dict(color='darkred',  weight='bold'),
}


# ── Plotting helpers ──────────────────────────────────────────────────────────

def min_max_scale_metrics(df, exclude_rows=None):
    """Min-max scale metrics DataFrame row-wise.

    Each row (metric) is independently scaled to [0, 1] across columns
    (embeddings).
    """
    if exclude_rows is None:
        exclude_rows = [_METRIC_TYPE]

    out = df.copy()
    for metric_name in out.index:
        if metric_name in exclude_rows:
            continue
        values = pd.to_numeric(out.loc[metric_name], errors='coerce')
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            out.loc[metric_name] = (values - min_val) / (max_val - min_val)
    return out


def _normed_cmap(col_data, cmap, num_stds=2.5):
    """Create a normalized colormap function based on data statistics."""
    mean = col_data.mean()
    std = col_data.std()
    vmin = mean - num_stds * std
    vmax = mean + num_stds * std
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return lambda x: scalar_mappable.to_rgba(x)


def _bar_contrast_text(ax, val, **kwargs):
    """Bar plot that switches annotation text to white over dark bars."""
    xlim = kwargs.get('xlim', (0, 1))
    if val >= 0.5 * xlim[1]:
        cmap = kwargs.get('cmap')
        color = kwargs.get('color', 'C1')
        if cmap is not None:
            color = cmap(float(val))
        r, g, b = mpl.colors.to_rgb(color)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance < 0.5:
            kwargs['textprops'] = {**kwargs.get('textprops', {}), 'color': 'white'}
    return bar(ax, val, **kwargs)


def plot_results_table(
    df,
    show: bool = True,
    save_dir: str | None = None,
    metric_cmap=mpl.cm.PRGn,
    score_cmap=mpl.cm.YlGnBu,
    sort_col='Bio conservation',
    group_colors: dict = None,
    method_styles: dict = None,
) -> Table:
    """Plot benchmarking results with colored metric type groups."""
    if method_styles is None:
        method_styles = METHOD_STYLES

    if group_colors is None:
        group_colors = {
            'Nuisance score \u2b07': 'red',
            'Bio conservation \u2b06': 'blue',
            'Batch correction \u2b06': 'grey',
            'Aggregate score \u2b06': 'black',
        }

    group_cmaps = {
        'Nuisance score \u2b07': mpl.cm.Reds,
        'Bio conservation \u2b06': mpl.cm.Blues,
        'Batch correction \u2b06': mpl.cm.Greys,
        'Aggregate score \u2b06': mpl.cm.YlGnBu,
    }

    num_embeds = df.shape[0] - 1
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    score_cols = df.columns[df.loc[_METRIC_TYPE].str.contains('Aggregate score', na=False)]
    other_cols = df.columns[~df.loc[_METRIC_TYPE].str.contains('Aggregate score', na=False)]

    _score_order = ['Nuisance score', 'Bio conservation', 'Batch correction', 'Total']
    score_cols = sorted(score_cols, key=lambda c: _score_order.index(c) if c in _score_order else len(_score_order))

    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left"}),
    ]

    for col in other_cols:
        metric_type = df.loc[_METRIC_TYPE, col]
        type_cmap = group_cmaps.get(metric_type, metric_cmap)
        column_definitions.append(
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=_normed_cmap(plot_df[col], cmap=type_cmap, num_stds=2.5),
                group=metric_type,
                formatter="{:.2f}",
            )
        )

    _score_cmaps = {
        'Nuisance score': mpl.cm.Reds,
        'Bio conservation': mpl.cm.Blues,
        'Batch correction': mpl.cm.Greys,
        'Total': mpl.cm.YlGnBu,
    }

    for i, col in enumerate(score_cols):
        metric_type = df.loc[_METRIC_TYPE, col]
        col_cmap = _score_cmaps.get(col, score_cmap)
        column_definitions.append(
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=_bar_contrast_text,
                plot_kw={
                    "cmap": col_cmap,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=metric_type,
                border="left" if col == 'Batch correction' else None,
            )
        )

    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)

        for text_obj in ax.texts:
            for metric_type, color in group_colors.items():
                if text_obj.get_text() == metric_type:
                    text_obj.set_color(color)
                    text_obj.set_weight('bold')

        if method_styles:
            method_names = set(plot_df.index)
            default_style = dict(color='gray', weight='bold')
            for text_obj in ax.texts:
                txt = text_obj.get_text()
                if txt in method_names:
                    style = method_styles.get(txt, default_style)
                    text_obj.set_color(style['color'])
                    text_obj.set_weight(style['weight'])
                    if 'style' in style:
                        text_obj.set_style(style['style'])

    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

    return tab


# ── Disentanglement metrics ───────────────────────────────────────────────────

def _calculate_perturbation_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    n_components_pcr: Optional[int] = None,
) -> dict:
    """Calculate perturbation recovery metrics for a single embedding."""
    import scib_metrics

    nmi_ari_result = scib_metrics.nmi_ari_cluster_labels_kmeans(X, labels)
    asw = scib_metrics.silhouette_label(X, labels)
    pcr = scib_metrics.utils.principal_component_regression(
        X, labels, n_components=n_components_pcr
    )

    return {
        'NMI cluster/label': nmi_ari_result['nmi'],
        'ARI cluster/label': nmi_ari_result['ari'],
        'ASW label': asw,
        'PCR': pcr,
    }


def get_perturbation_metrics(
    adata,
    embedding_obsm_keys: List[str],
    nuisance_key: str,
    n_components_pcr: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate nuisance disentanglement metrics across multiple embeddings.

    Returns DataFrame with rows=metrics, columns=embeddings.
    """
    nuisance_labels = pd.Categorical(adata.obs[nuisance_key]).codes
    results = {}

    for emb_key in tqdm(embedding_obsm_keys, desc="Calculating Nuisance metrics"):
        if emb_key not in adata.obsm:
            raise KeyError(f"Embedding key '{emb_key}' not found in adata.obsm")
        results[emb_key] = _calculate_perturbation_metrics(
            adata.obsm[emb_key], nuisance_labels, n_components_pcr=n_components_pcr
        )

    return pd.DataFrame(results)


def benchmark_disentanglement(
    adata,
    embedding_obsm_keys: List[str],
    nuisance_key: str,
    bio_key: str = None,
    batch_key: str = None,
    n_components_pcr: Optional[int] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute individual disentanglement metrics (no aggregation).

    Returns DataFrame with rows=metrics, columns=embeddings, plus a
    'Metric Type' column labelling each metric's category.
    """
    from scib_metrics.benchmark import BatchCorrection, BioConservation, Benchmarker

    # Nuisance metrics
    nuisance_df = get_perturbation_metrics(
        adata=adata,
        embedding_obsm_keys=embedding_obsm_keys,
        nuisance_key=nuisance_key,
        n_components_pcr=n_components_pcr,
    )

    # Batch correction + bio conservation via scib-metrics
    if batch_key is not None or bio_key is not None:
        if batch_key is None:
            adata.obs['_dummy_batch'] = 'batch1'

        bm = Benchmarker(
            adata,
            batch_key=batch_key if batch_key is not None else '_dummy_batch',
            label_key=bio_key,
            bio_conservation_metrics=BioConservation() if bio_key else None,
            batch_correction_metrics=BatchCorrection() if batch_key else None,
            embedding_obsm_keys=embedding_obsm_keys,
            n_jobs=n_jobs,
        )
        bm.benchmark()
        scib_df = bm.get_results(min_max_scale=False).T
    else:
        scib_df = pd.DataFrame()

    # Combine individual metrics
    individual_df = pd.concat([nuisance_df, scib_df])

    # Label each metric with its category
    individual_df[_METRIC_TYPE] = individual_df.get(_METRIC_TYPE, pd.Series(dtype=str))
    individual_df[_METRIC_TYPE] = individual_df[_METRIC_TYPE].fillna("Nuisance score")

    # Drop scib-metrics aggregate rows (we recompute them ourselves)
    individual_df = individual_df[individual_df[_METRIC_TYPE] != _AGGREGATE_SCORE]

    return individual_df


def aggregate_results(
    individual_df: pd.DataFrame,
    batch_weight: float = 0.2,
    bio_weight: float = 0.3,
    nuisance_weight: float = 0.5,
) -> pd.DataFrame:
    """Aggregate individual metrics into category scores and weighted Total.

    Takes the output of ``benchmark_disentanglement`` (rows=metrics,
    cols=embeddings + 'Metric Type') and produces the final results table
    (rows=embeddings + 'Metric Type', cols=individual metrics + aggregates).
    """
    # Validate weights
    total_weight = batch_weight + bio_weight + nuisance_weight
    if not np.isclose(total_weight, 1.0):
        batch_weight /= total_weight
        bio_weight /= total_weight
        nuisance_weight /= total_weight

    # Category means across metrics (rows=embeddings, cols=category names)
    per_class_score = individual_df.groupby(_METRIC_TYPE).mean().transpose()

    # Invert Nuisance so higher = better (matching bio/batch convention)
    if 'Nuisance score' in per_class_score.columns:
        per_class_score['Nuisance score'] = 1 - per_class_score['Nuisance score']

    # Weighted Total (renormalize weights to present categories)
    present = []
    if 'Batch correction' in per_class_score.columns:
        present.append((batch_weight, per_class_score["Batch correction"]))
    if 'Bio conservation' in per_class_score.columns:
        present.append((bio_weight, per_class_score["Bio conservation"]))
    if 'Nuisance score' in per_class_score.columns:
        present.append((nuisance_weight, per_class_score["Nuisance score"]))
    if present:
        w_sum = sum(w for w, _ in present)
        per_class_score["Total"] = sum((w / w_sum) * s for w, s in present)

    # Combine individual metrics (transposed) with aggregate columns
    result = pd.concat(
        [individual_df.drop(columns=_METRIC_TYPE).transpose(), per_class_score],
        axis=1,
    )

    # Add Metric Type row
    metric_type_map = dict(zip(individual_df.index, individual_df[_METRIC_TYPE]))
    result.loc[_METRIC_TYPE] = [
        metric_type_map.get(col, _AGGREGATE_SCORE) for col in result.columns
    ]

    # Direction arrows
    direction_dict = {
        'Nuisance score': 'Nuisance score \u2b07',
        'Batch correction': 'Batch correction \u2b06',
        'Bio conservation': 'Bio conservation \u2b06',
        _AGGREGATE_SCORE: 'Aggregate score \u2b06',
        'Total': 'Total \u2b06',
    }
    result.loc[_METRIC_TYPE] = result.loc[_METRIC_TYPE].replace(direction_dict)

    return result
