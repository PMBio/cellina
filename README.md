# Cellina

[![Tests][badge-tests]][link-tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/PMBio/cellina/test.yaml?branch=main
[link-tests]: https://github.com/PMBio/cellina/actions/workflows/test.yml

Cellina is a dual-encoder variational autoencoder for predicting how a cell's gene expression changes under altered spatial contexts — a class of queries we call *tissue graph counterfactuals*.

In tissues, a cell's transcriptional state is shaped by its local neighborhood: the composition of nearby cells and the signals they emit. Existing perturbation methods typically treat cells as independent and apply perturbations uniformly. Cellina addresses this gap by explicitly separating a cell's **intrinsic state** (*z*, encoding cell identity) from its **spatial context** (*s*, encoding microenvironmental influence), then uses *s* as a conditioning input to render counterfactual predictions under two types of intervention:

- **Edge perturbation** — rewire a cell's neighborhood (replace neighbors with those from a different domain)
- **Node perturbation** — modify the expression of existing neighbors (e.g. pathway activation or knockout)

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

Both variants share the same call:

```python
expr_cf = model.get_counterfactual_expression(
    indices=ref_idx,              # focal cells (source domain)
    neighbour_indices=donor_idx,  # donor pool (target domain)
)
```

**Node perturbation** modifies the feature vectors of $v$'s neighbours while preserving graph topology. For a target gene set $\mathcal{S}$ and a gene-specific transformation $T_g$:

$$x_{u,g}^{\mathrm{cf}} = \begin{cases} T_g(x_{u,g}) & g \in \mathcal{S} \\ x_{u,g} & g \notin \mathcal{S} \end{cases}$$

$T_g$ can encode any intervention (additive shift, knockout, overexpression, or learned counterfactual values). The two variants differ in both how $T_g$ is instantiated and when the perturbed features are consumed:

*Cellina* — $T_g(x_{u,g}) = x_{u,g} + \delta_g$ applied to log-normalised neighbour expression (`add_shift=True`); pre-aggregate into pseudobulk spatial features, then run inference:

```python
from cellina import make_neighbor_perturbation

make_neighbor_perturbation(adata, {"VEGFA": 2.0, "MYC": -1.5}, add_shift=True)  # values are logFCs
expr_cf = model.get_perturbed_expression(adata)
```

*CellinaGCN* — $T_g(x_{u,g}) = x_{u,g} \cdot e^{\delta_g}$ applied to raw counts (`add_shift=False`); store perturbed counts as a layer and let the GCN aggregate them on the fly at inference:

```python
from cellina import make_perturbed_expression

make_perturbed_expression(adata, {"VEGFA": 2.0, "MYC": -1.5}, layer_key="counts_cf", add_shift=False)  # values are logFCs
expr_cf = gcn_model.get_perturbed_expression(adata, cf_layer="counts_cf")
```

Both perturbation types also expose `get_*_latents` counterparts for inspecting the spatial latent $s$ directly. See `demo.ipynb` for a full worked example.

## Repository contents

```
src/cellina/
  _cellina_model.py          # Cellina (Cellina)
  _cellina_module.py
  _cellina_gcn_model.py    # CellinaGCN (Cellina-GAT)
  _cellina_gcn_module.py
  _spatial_encoder.py        # GATv2-based GraphEncoder
  _edge_data_splitter.py     # Graph-aware data loading (NeighborLoader / LinkNeighborLoader)
  _training_plan.py          # Shared adversarial training plan
  _spatial_utils.py          # spatial_neighbors, compute_spatial_features, perturbation utilities
demo.ipynb                   # End-to-end demo
perturb_utils.py             # Perturbation evaluation helpers
```

## Release notes

See the [changelog](CHANGELOG.md).

## Citation

> Citation coming soon.

Built on [scvi-tools](https://scvi-tools.org):

```
@article{gayoso2022python,
  title={A Python library for probabilistic analysis of single-cell omics data},
  author={Gayoso, Adam and Lopez, Romain and Xing, Galen and Boyeau, Pierre and Valiollah Pour Amiri, Valeh and Hong, Justin and Wu, Katherine and Jayasuriya, Michael and Mehlman, Edouard and Langevin, Maxime and others},
  journal={Nature biotechnology},
  volume={40},
  number={2},
  pages={163--166},
  year={2022},
  publisher={Nature Publishing Group US New York}
}
```

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/PMBio/cellina/issues

Copyright (c) 2026, PMBio
