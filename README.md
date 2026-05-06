# Cellina

<!-- badges omitted for anonymous review -->

Cellina: A spatial-aware variational autoencoder for spatial RNA-seq data with dual encoders.

## Key Features

- **Dual Encoder Architecture**: Separate encoders for count data (z) and spatial features (s)
- **Graph-based Spatial Integration**: GCN/GAT encoder aggregates neighbours' log-normalized expression via message passing over the cell neighbourhood graph to produce spatial latent `s`
- **Supervised Contrastive Spatial Loss**: SupCon loss on `s` pulls spatially adjacent cells together (graph neighbours = positives) while pushing cells from different spatial domains apart (non-neighbour domain cells = negatives)
- **Optional Cell Type Classifier**: Supervised classification head for cell type prediction
- **Adversarial Domain Forgetting**: Optional discriminator for removing unwanted domain effects from latent representations
- **Built on scvi-tools**: Leverages the robust scvi-tools framework for scalability and reliability

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install cellina:

<!--
1) Install the latest release of `cellina` from `PyPI <https://pypi.org/project/cellina/>`_:

```bash
pip install cellina
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/[anonymous]/cellina.git@main
```

2. Install from local directory (for development):

```bash
pip install -e .
```

## Quick Start

```python
import scanpy as sc
import numpy as np
from cellina_graph import CellinaModel, make_perturbed_expression
from cellina_graph._spatial_utils import spatial_neighbors

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Compute spatial connectivity graph (stored in adata.obsp["spatial_connectivities"])
spatial_neighbors(adata, bandwidth=np.inf, max_neighbours=50, standardize=True)

# Setup and train model
CellinaModel.setup_anndata(
    adata,
    batch_key="batch",
    labels_key="cell_type",
    domains_key="niche",
    spatial_connectivities_key="spatial_connectivities",  # default
)

model = CellinaModel(
    adata,
    n_latent=20,
    classifier_lambda=1e4,   # optional: cell type classifier loss weight
    discriminator_lambda=1e4, # optional: adversarial domain-forgetting weight
)

model.train(max_epochs=200)

# Get latent representations
adata.obsm['X_cellina'] = model.get_latent_representation(latent_key='shifted')  # concat(z, s)
adata.obsm['X_cellina_z'] = model.get_latent_representation(latent_key='z')
adata.obsm['X_cellina_s'] = model.get_latent_representation(latent_key='s')
```

### Counterfactual inference (graph rewiring)

Ask "what expression/latent would a cell have if placed in a different spatial neighbourhood?"

```python
target_idx = np.where(adata.obs["cell_type"] == "Tumor")[0]
donor_idx  = np.where(adata.obs["cell_type"] == "Stroma")[0]

# Spatial latent under the counterfactual neighbourhood
cf_latents = model.get_counterfactual_latents(
    indices=target_idx,
    neighbour_indices=donor_idx,
    latent_key="s",        # "s" (default), "z", or "shifted"
)

# Predicted expression under the counterfactual neighbourhood
cf_expr = model.get_counterfactual_expression(
    indices=target_idx,
    neighbour_indices=donor_idx,
    library_size="latent", # "latent" (default) or a float, e.g. 1e4
)
```

### Perturbation inference (feature perturbation)

Ask "what expression/latent would result if neighbours' counts were perturbed?"

```python
# Build a counterfactual count matrix (e.g. silence GeneA with logFC = -10)
make_perturbed_expression(
    adata,
    perturbations={"GeneA": -10.0},
    layer_key="counts_cf",  # default
)

# Propagate perturbed neighbour features through the GCN
pt_latents = model.get_perturbed_latents(adata, cf_layer="counts_cf")
pt_expr    = model.get_perturbed_expression(adata, cf_layer="counts_cf")
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

If you use Cellina in your research, please cite:

```
# Citation to be added
```

Built with [scvi-tools](https://scvi-tools.org):
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

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/[anonymous]/cellina/issues
[changelog]: https://cellina.readthedocs.io/latest/changelog.html
[link-docs]: https://cellina.readthedocs.io
[link-api]: https://cellina.readthedocs.io/latest/api.html

<!-- scverse/NumFOCUS attribution omitted for anonymous review -->

Copyright (c) 2025, [Anonymous]
