<p align="center"><img src="https://raw.githubusercontent.com/PMBio/cellina/main/docs/banner.svg" alt="Cellina" width="75%"></p>
<!-- docs-readme-start -->

# Cellina

[![Tests](https://github.com/PMBio/cellina/actions/workflows/test.yml/badge.svg)](https://github.com/PMBio/cellina/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/PMBio/cellina/branch/main/graph/badge.svg)](https://codecov.io/gh/PMBio/cellina)
[![Documentation Status](https://readthedocs.org/projects/cellina/badge/?version=latest)](https://cellina.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2606.08493-b31b1b)](https://arxiv.org/abs/2606.08493v2)

Cellina predicts how a cell's gene expression would change if its spatial neighborhood changed — *tissue graph counterfactuals*. It disentangles a cell's **intrinsic state** (*z*) from its **spatial context** (*s*), then answers two kinds of query on the tissue graph:

- **Edge perturbation** — swap a cell's neighbors (e.g. give healthy cells a tumor neighborhood).
- **Node perturbation** — change what the neighbors express (e.g. activate or knock out a pathway).

Method details are in the paper: [*Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement*](https://arxiv.org/abs/2606.08493v2) (NeurIPS 2026).

## Models

| Class | Paper name | Spatial encoder |
|---|---|---|
| `Cellina` | *Cellina* | Pseudobulk of neighbor expression → MLP; fast, scales like non-spatial baselines |
| `CellinaGCN` | *Cellina-GAT* | GATv2 on the local subgraph; learns attention over neighbors |

Both perform on par.

## Getting started

Install with one of the shipped conda environments — [`environment.yml`](https://github.com/PMBio/cellina/blob/main/environment.yml) (GPU/CUDA) or [`env_minimal.yml`](https://github.com/PMBio/cellina/blob/main/env_minimal.yml) (CPU) — via `conda env create -f <file>`. Without conda, `pip install cellina` gives `Cellina`; `CellinaGCN` additionally needs PyG: install the `torch-scatter`/`torch-sparse` wheels for your torch/CUDA from [data.pyg.org](https://data.pyg.org/whl/), then `pip install "cellina[gat]"`.

Then follow the tutorials on colorectal cancer tissue: [Cellina](https://cellina.readthedocs.io/en/latest/tutorial.html) and [Cellina-GAT](https://cellina.readthedocs.io/en/latest/tutorial_gat.html) (notebooks in [`docs/`](https://github.com/PMBio/cellina/tree/main/docs)). Release notes are in the [changelog](https://github.com/PMBio/cellina/blob/main/CHANGELOG.md).

## Citation

```bibtex
@inproceedings{moeed2026querying,
  title     = {Querying Counterfactuals on Tissue Graphs with Supervised Disentanglement},
  author    = {Moeed, Abdul and Schrod, Stefan and Rohbeck, Martin and Bonder, Marc Jan and Lutsik, Pavlo and Stegle, Oliver and Dimitrov, Daniel},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.08493v2}
}
```

Built on [scvi-tools](https://scvi-tools.org). Bugs and questions: [issue tracker](https://github.com/PMBio/cellina/issues).
