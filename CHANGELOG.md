# Changelog

All notable changes to this project will be documented in this file.

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.2.0] - 01.10.2026
### Added
- `anchor_donors` for `CellinaGCN.get_counterfactual_expression` /
  `get_counterfactual_latents` (default `True`): `neighbour_indices` are *anchor* cells;
  each focal cell is paired with one anchor drawn uniformly with replacement and inherits
  that anchor's neighbours in `adata.obsp[connectivity_key]` (new argument, default: the
  graph registered in `setup_anndata`), minus `exclude_indices` (new argument) and itself,
  as its donors; anchors with no neighbour left (homotypic) are skipped with a warning.
  `anchor_donors=False` is the previous behaviour: `neighbour_indices` is a
  donor pool from which each focal cell draws `n_neighbors` donors uniformly without
  replacement (`exclude_indices` is removed from the pool). `neighbour_indices` may also
  be a list/tuple with one 1-D integer array per focal cell, used verbatim.
- `make_counterfactual_adata` gained a `layer` argument (threaded through
  `Cellina._make_counterfactual_adata`, `Cellina.get_counterfactual_latents` and
  `Cellina.get_counterfactual_expression`). When `precomputed=False`, the
  counterfactual spatial features are aggregated from `adata.layers[layer]`
  instead of `adata.X`. This fixes a train/inference mismatch: the training
  `spatial_x` is typically built from log1p(CP10K) while `adata.X` is reset to
  raw counts for the model, so edge-perturbation counterfactuals silently
  aggregated raw counts. Default `None` keeps the previous behaviour.
- `make_neighbor_perturbation` and `make_perturbed_expression` gained an optional
  source `layer` argument: the perturbation is applied to `adata.layers[layer]`
  instead of `adata.X` when given.
- `compute_spatial_features` warns (`UserWarning`) when `layer is None` and
  `adata.X` looks like raw counts (sampled non-zero entries are integer-valued
  with a maximum above 50), pointing at `layer=` for the normalized
  representation used at training time.
- `make_neighbor_perturbation` gained `perturb_fraction` and `random_state`
  arguments: only a random subset of cells (of the requested fraction) receives
  the perturbation before re-aggregation, so each focal cell effectively sees
  `~perturb_fraction` of its neighbours perturbed. `perturb_fraction=1.0`
  (default) keeps the previous behaviour of perturbing every cell.
- `exclude_indices` on `Cellina.get_counterfactual_expression` /
  `get_counterfactual_latents` (and `Cellina._make_counterfactual_adata` /
  `make_counterfactual_adata`): a 1-D integer array of cells that never contribute to an
  anchor's neighbourhood when `anchor_donors=True` — their columns are zeroed in
  `connectivity_key` before the anchors' spatial features are aggregated. Mirrors the
  argument of the same name on `CellinaGCN`, except that the focal cell itself is not
  removed automatically; with `anchor_donors=False` it raises rather than being ignored.
- `CellinaGCN` can return GATv2 attention weights: `get_attention_weights` extracts the
  per-edge attention of the spatial encoder's GATv2 layers and
  `attention_by_group` aggregates them by an `adata.obs` grouping, with a new
  section 3.1 in `docs/tutorial_gat.ipynb` showing the workflow.

### Changed
- `make_counterfactual_adata(..., anchor_donors=True)` (and therefore
  `Cellina.get_counterfactual_expression` / `get_counterfactual_latents`) now recomputes
  each anchor's spatial features from `adata.obsp[connectivity_key]` with the columns of
  `exclude_indices` removed, aggregating `adata.layers[layer]` (or `adata.X`), instead of
  copying the stored `adata.obsm[spatial_column]` rows verbatim. The stored rows are
  all-zero for anchors masked out of the training graph with
  `spatial_neighbors(test_indices=...)`, so the previous behaviour silently fed the model
  empty neighbourhoods under leave-one-out. `connectivity_key` and `layer` are now used in
  both branches.
- `docs/tutorial.ipynb` stores the normalized expression in
  `adata.layers['lognorm']` and passes `layer='lognorm'` to the edge- and
  node-perturbation calls, so both now aggregate the same representation the
  model was trained on (previously the edge-perturbation section aggregated raw
  counts).

- `Cellina.get_counterfactual_expression` / `get_counterfactual_latents` /
  `_make_counterfactual_adata` and `make_counterfactual_adata`: `precomputed` is renamed
  `anchor_donors` and `n_neighbours` is renamed `n_neighbors` (same semantics and
  defaults, no behaviour change). `CellinaGCN`'s `n_neighbors_per_seed` is renamed
  `n_neighbors`. `make_counterfactual_adata`'s first two positional parameters
  `indices_basal` / `indices_counterfactual` are renamed `indices` / `neighbour_indices`,
  the names `Cellina.get_counterfactual_*` and `CellinaGCN.get_counterfactual_*` already
  use. No aliases are kept.

## [1.1.1] — 2026-09-05
### Removed
- **Breaking**: `condition_on_intrinsic` parameter removed from `Cellina`,
  `CellinaModule`, `CellinaGCN`, and `CellinaGCNModule`; behaviour is now always
  the previous `False` (the `s_encoder` receives spatial features only, never
  concatenated with detached `z`). Note that `CellinaModule`/`CellinaGCNModule`
  previously defaulted to `True` when constructed directly.

### Changed
- `CellinaGCNModule.inference` now runs the count (`z_encoder`) and library
  encoders on seed nodes only instead of the full sampled subgraph, cutting
  their per-batch compute by the subgraph blow-up factor (~20–50x) and removing
  the degree-weighted bias in `z_encoder` batch-norm statistics.
- `GraphJointDataSplitter` symmetrizes an asymmetric spatial connectivity
  matrix internally (`adj.maximum(adj.T)`) with a `UserWarning`; previously an
  asymmetric graph silently transposed the GCN receptive field.

### Fixed
- Explicitly stored zeros in the spatial connectivity matrix (e.g. from
  in-place weight thresholding) no longer become message-passing edges; they
  are dropped via `eliminate_zeros()` on a copy, leaving `adata.obsp` untouched.
- `CellinaGCN` counterfactual loaders now add donor -> seed edges only instead of
  bidirectional seed <-> donor edges. With multi-hop sampling, the reverse edges let
  donors aggregate over the (control) seeds and pulled counterfactuals back toward
  the control state. Donor draws use the same RNG stream, so `seed` still reproduces
  the same donor sets; predictions change.

## [1.0.0] — 2026-06-04 - Release

This is the first stable release of Cellina, graduating from the 0.99.x pre-release series.

### Added
- `subgraph_type` parameter on `CellinaGCN` and `GraphJointDataSplitter`
  (`"induced"` | `"directional"`); directional mode cuts counterfactual-inference
  VRAM by ~40% compared to the induced default.
- End-to-end tutorial notebook shipped in the repository at `docs/tutorial.ipynb` and `docs/tutorial_gat.ipynb`,
  covering a full CRC counterfactual workflow from data loading to perturbation
  analysis.

### Changed
- `__version__` is now resilient to missing package metadata (returns `"unknown"`
  instead of raising `PackageNotFoundError`).
- README updated to reference `docs/tutorial.ipynb` as the primary getting-started
  resource.

### Fixed
- ReadTheDocs output path and `attrs` mock in `docs/conf.py`.

## [0.99.3] — 2026-06-02 - Pre-prerelease
- Vectorized SupCon loss for improved training speed; updated tests accordingly.
- Hops and layer mismatches will raise a warning. From tests, this has a negligable effect if batch_size or the number of neighbors is large enough.
- Added log1p for spatial_x in case not normalized.

## [0.99.2] — 2026-06-01

- Sparse matrix only for GCN-based module; updated tests accordingly.


## [0.99.1] — 2026-06-01


- Added minus 1.0 back to multiplicative perturbation to avoid zeroing out genes with large negative logFCs; updated tests accordingly.


## [0.99.0] — 2026-05-27

This is the first public version of Cellina: a dual-encoder variational autoencoder
for spatial transcriptomics with adversarial domain forgetting.

### Added

-   `Cellina` / `CellinaModule`: MLP-based dual-encoder VAE with adversarial
    domain classifier and discriminator for batch-effect removal
-   `CellinaGCN` / `CellinaGCNModule`: GCN-based variant that encodes spatial
    context via a graph convolutional network (`s_encoder`) alongside the count
    encoder (`z_encoder`); supports link prediction for edge-level tasks
-   `CellinaAdversarialTrainingPlan`: unified two-step adversarial training plan
    shared by both module types
-   `GraphJointDataSplitter` / `InferenceBatchLoader` / `JointBatchLoader`:
    graph-aware data loading built on PyTorch Geometric `NeighborLoader` and
    `LinkNeighborLoader`
-   `spatial_neighbors`: builds spatial connectivity graphs (kNN + kernel weighting)
    with support for `gaussian`, `exponential`, and `linear` kernels, per-library
    graph construction via `library_key`, and the new `test_indices` parameter
-   `test_indices` in `spatial_neighbors` / `_spatial_neighbors_core`: isolates
    test cells from the spatial graph at build time via coordinate displacement,
    producing all-zero rows and columns for those cells without post-hoc masking
-   `compute_spatial_features`, `make_neighbor_perturbation`,
    `make_perturbed_expression`: spatial feature computation and in-silico
    perturbation utilities
