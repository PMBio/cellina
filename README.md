<p align="center">
  <img src="https://raw.githubusercontent.com/PMBio/cellina/main/docs/logo.svg" alt="Cellina" width="200">
</p>

# Cellina

[![Tests](https://github.com/PMBio/cellina/actions/workflows/test.yml/badge.svg)](https://github.com/PMBio/cellina/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/PMBio/cellina/branch/main/graph/badge.svg)](https://codecov.io/gh/PMBio/cellina)
[![Documentation Status](https://readthedocs.org/projects/cellina/badge/?version=latest)](https://cellina.readthedocs.io/en/latest/?badge=latest)

Cellina is a dual-encoder variational autoencoder for predicting how a cell's gene expression changes under altered spatial contexts — a class of queries we call *tissue graph counterfactuals*.

In tissues, a cell's transcriptional state is shaped by its local neighborhood: the composition of nearby cells and the signals they emit. Existing perturbation methods typically treat cells as independent and apply perturbations uniformly. Cellina addresses this gap by explicitly separating a cell's **intrinsic state** (*z*, encoding cell identity) from its **spatial context** (*s*, encoding microenvironmental influence), then uses *s* as a conditioning input to render counterfactual predictions under two types of intervention:

- **Edge perturbation** — rewire a cell's neighborhood (replace neighbors with those from a different domain)
- **Node perturbation** — modify the expression of existing neighbors (e.g. pathway activation or knockout)

## Getting started

Follow the worked tutorials for end-to-end examples on colorectal cancer tissue: [Cellina](https://cellina.readthedocs.io/en/latest/tutorial.html) and [Cellina-GAT](https://cellina.readthedocs.io/en/latest/tutorial_gat.html) (or run them locally from [`docs/tutorial.ipynb`](docs/tutorial.ipynb) and [`docs/tutorial_gat.ipynb`](docs/tutorial_gat.ipynb)).

## How it works

**Generative model.** Cellina is a VAE with two latent variables. An MLP encoder $\text{Enc}_z$ maps raw counts to $z \sim q(z \mid x)$; a spatial encoder maps the cell's neighborhood to $s \sim q(s \mid \mathcal{N}(v))$. A shared decoder reconstructs counts from $[z;\, s]$ under a Negative Binomial likelihood. Both latents have standard normal priors.

**Supervised disentanglement.** Optimizing the ELBO alone does not prevent $z$ from absorbing spatially-driven variation. Cellina adds auxiliary objectives:
- A **cell-type classifier** on $z$ anchors it to transcriptional identity.
- An **adversarial discriminator** is trained to predict spatial domain from $z$; the encoder is then trained to fool it, routing microenvironmental variation to $s$ by elimination.
- A **graph-supervised contrastive loss** $s$ *(CellinaGCN only, optional)*, as a biologically grounded inductive bias that promotes similarity within local neighbourhoods. Enabled by setting `link_prediction_weight > 0`.

**Training** alternates between a discriminator step (encoder frozen) and a VAE step (discriminator frozen), following a standard adversarial schedule.

**Two variants** differ in how the spatial encoder is implemented:

| Code class | Paper name | Spatial encoder |
|---|---|---|
| `Cellina` | *Cellina* | Degree-normalized weighted pseudobulk aggregation of neighbor expression → MLP |
| `CellinaGCN` | *Cellina-GAT* | Multi-layer GATv2 on the local subgraph; self-loops excluded so $v$'s own expression is captured by $z$ alone; modified contrastive loss on $s$ |

The two variants perform on par. `Cellina` decouples neighborhood construction from training and scales similarly to non-spatial baselines; `CellinaGCN` learns attention over each subgraph at additional cost per step.

## Tissue graph counterfactuals

Cellina supports two post-training interventions on the spatial graph $\mathcal{G}$:

**Edge perturbation** replaces a cell's spatial neighbourhood with donors sampled from a target tissue domain, while keeping the cell's own expression fixed:

$$\mathcal{N}(v) := \mathcal{N}'$$

Both variants expose this through a single `get_counterfactual_expression` call.

**Node perturbation** modifies the feature vectors of $v$'s neighbours while preserving graph topology. For a target gene set $\mathcal{S}$ and a gene-specific transformation $T_g$:

$$x_{u,g}^{\mathrm{cf}} = \begin{cases} T_g(x_{u,g}) & g \in \mathcal{S} \\ x_{u,g} & g \notin \mathcal{S} \end{cases}$$

$T_g$ can encode any intervention (additive shift, knockout, overexpression, or learned counterfactual values). The two variants differ in how $T_g$ is instantiated and when the perturbed features are consumed: *Cellina* applies an additive shift to log-normalised neighbour expression and pre-aggregates into pseudobulk spatial features before inference, while *CellinaGCN* applies a multiplicative factor to raw counts and lets the GCN aggregate the perturbed counts on the fly. Both also expose `get_*_latents` counterparts for inspecting the spatial latent $s$ directly.

See the [Cellina](https://cellina.readthedocs.io/en/latest/tutorial.html) and [Cellina-GAT](https://cellina.readthedocs.io/en/latest/tutorial_gat.html) tutorials for full worked examples.

## Release notes

See the [changelog](CHANGELOG.md).

## Installation

Cellina ships two conda environments: [`environment.yml`](environment.yml) for the full GPU/CUDA setup, and [`env_minimal.yml`](env_minimal.yml) for a lightweight CPU-only install. Create one with `conda env create`, then follow the tutorials above.

## Citation

> Citation coming soon.

Built on [scvi-tools](https://scvi-tools.org).

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/PMBio/cellina/issues

Copyright (c) 2026, PMBio
