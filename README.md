# Cellina

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/PMBio/cellina/test.yaml?branch=main
[link-tests]: https://github.com/PMBio/cellina/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/cellina

Cellina: A spatial-aware variational autoencoder for spatial RNA-seq data with dual encoders.

This package extends [scVI-tools](https://www.nature.com/articles/s41592-018-0229-2) with a spatial encoder that processes spatial features alongside the standard count encoder. The model uses dual latent representations (z from counts, s from spatial+z) that are combined to reconstruct count data while preserving both biological identity and spatial context.

## Key Features

- **Dual Encoder Architecture**: Separate encoders for count data (z) and spatial features (s)
- **Spatial Integration**: Learns spatial context through weighted pseudobulk features
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
pip install git+https://github.com/PMBio/cellina.git@main
```

2. Install from local directory (for development):

```bash
pip install -e .
```

## Quick Start

```python
import scanpy as sc
import numpy as np
from cellina import CellinaModel
from cellina._spatial_utils import spatial_neighbors, weighted_pseudobulks

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Compute spatial features
spatial_neighbors(adata, bandwidth=np.inf, max_neighbours=50, standardize=True)
weighted_pseudobulks(
    adata,
    sp=adata.obsp['spatial_connectivities'],
    groupby="cell_type",
    obsm_key='spatial_x',
)

# Setup and train model
CellinaModel.setup_anndata(
    adata, 
    batch_key="batch", 
    labels_key="cell_type",
    domains_key="niche",
    spatial_obsm_key="spatial_x"
)

model = CellinaModel(
    adata, 
    n_latent=20,
    classifier_lambda=1e4,  # Optional: enable cell type classifier
    discriminator_lambda=1e4  # Optional: enable domain forgetting
)

model.train(max_epochs=200)

# Get latent representations
adata.obsm['X_cellina'] = model.get_latent_representation()  # concat(z, s)
adata.obsm['X_cellina_z'] = model.get_latent_representation(latent_key='z')
adata.obsm['X_cellina_s'] = model.get_latent_representation(latent_key='s')
```

## Model Architecture

Cellina implements a dual-encoder VAE:
- **z encoder**: Processes count data to capture biological identity
- **s encoder**: Processes spatial features (concatenated with z) to capture spatial context
- **Decoder**: Reconstructs counts from shifted = concat(z, s)

Optional components:
- **Cell type classifier**: Supervised head on z for cell type prediction
- **Domain discriminator**: Adversarial training to remove domain effects from z

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
[issue-tracker]: https://github.com/PMBio/cellina/issues
[changelog]: https://cellina.readthedocs.io/latest/changelog.html
[link-docs]: https://cellina.readthedocs.io
[link-api]: https://cellina.readthedocs.io/latest/api.html

[//]: # (numfocus-fiscal-sponsor-attribution)

Cellina is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

Copyright (c) 2025, PMBio
