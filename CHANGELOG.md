# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [0.99.0] — 2026-05-27 — Initial public release

This is the first public version of Cellina: a dual-encoder variational autoencoder
for spatial transcriptomics with adversarial domain forgetting.

### Added

-   `CellinaModel` / `CellinaModule`: MLP-based dual-encoder VAE with adversarial
    domain classifier and discriminator for batch-effect removal
-   `CellinaGraph` / `CellinaGraphModule`: GCN-based variant that encodes spatial
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
