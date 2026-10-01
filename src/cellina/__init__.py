import logging
from importlib.metadata import PackageNotFoundError, version

from rich.console import Console
from rich.logging import RichHandler

from ._cellina_model import Cellina
from ._cellina_module import CellinaModule
from ._cellina_gcn_model import CellinaGCN
from ._cellina_gcn_module import CellinaGCNModule
from ._training_plan import CellinaAdversarialTrainingPlan
from ._spatial_utils import (
    compute_spatial_features,
    make_counterfactual_adata,
    make_neighbor_perturbation,
    make_perturbed_expression,
    spatial_neighbors,
)

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

__all__ = [
    "Cellina",
    "CellinaModule",
    "CellinaGCN",
    "CellinaGCNModule",
    "CellinaAdversarialTrainingPlan",
    "compute_spatial_features",
    "make_counterfactual_adata",
    "make_neighbor_perturbation",
    "make_perturbed_expression",
    "spatial_neighbors",
]

try:
    __version__ = version("cellina")
except PackageNotFoundError:
    # Package not installed (e.g. running from source on a docs build where
    # only PYTHONPATH=src is set). Fall back to a placeholder version.
    __version__ = "0.0.0"
