# Changelog

All notable changes to this project will be documented in this file.

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.0.0] — 2026-06-04 - Release

This is the first stable release of Cellina, graduating from the 0.99.x pre-release series.

### Added
- `subgraph_type` parameter on `CellinaGCN` and `GraphJointDataSplitter`
  (`"induced"` | `"directional"`); directional mode cuts counterfactual-inference
  VRAM by ~40% compared to the induced default.
- End-to-end tutorial notebook shipped in the repository at `docs/tutorial.ipynb`,
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
