import logging
import os
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import scvi
import torch
from rich.console import Console
from rich.logging import RichHandler

from ._cellina_model import Cellina
from ._cellina_module import CellinaModule
from ._training_plan import CellinaAdversarialTrainingPlan
from ._spatial_utils import (
    compute_spatial_features,
    make_counterfactual_adata,
    make_neighbor_perturbation,
    make_perturbed_expression,
    spatial_neighbors,
)



def __getattr__(name):
    # CellinaGCN needs PyG (optional `gat` extra): import lazily so `Cellina` works without it.
    if name in ("CellinaGCN", "CellinaGCNModule"):
        try:
            from ._cellina_gcn_model import CellinaGCN
            from ._cellina_gcn_module import CellinaGCNModule
        except ImportError as e:
            raise ImportError(
                f"{name} requires PyG (torch-geometric, torch-sparse, torch-scatter): "
                'pip install "cellina[gat]"'
            ) from e
        return {"CellinaGCN": CellinaGCN, "CellinaGCNModule": CellinaGCNModule}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Nice logging outputs with rich
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("cellina: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Prevent double outputs
logger.propagate = False


def seed_everything(seed=0):
    """Seed every RNG cellina touches and force deterministic CUDA kernels.

    Call before any CUDA work so ``CUBLAS_WORKSPACE_CONFIG`` takes effect.
    """
    # Seeds python/numpy/torch(+CUDA)/Lightning, sets cudnn deterministic, and
    # seeds scvi's train/val split (it reads ``scvi.settings.seed``).
    scvi.settings.seed = seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


__all__ = [
    "Cellina",
    "CellinaModule",
    "CellinaAdversarialTrainingPlan",
    "compute_spatial_features",
    "make_counterfactual_adata",
    "make_neighbor_perturbation",
    "make_perturbed_expression",
    "spatial_neighbors",
    "seed_everything",
]

if all(find_spec(m) for m in ("torch_geometric", "torch_sparse", "torch_scatter")):
    __all__ += ["CellinaGCN", "CellinaGCNModule"]

try:
    __version__ = version("cellina")
except PackageNotFoundError:
    # Package not installed (e.g. running from source on a docs build where
    # only PYTHONPATH=src is set). Fall back to a placeholder version.
    __version__ = "0.0.0"
