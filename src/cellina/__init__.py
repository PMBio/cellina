import logging
from importlib.metadata import version

from rich.console import Console
from rich.logging import RichHandler

from ._cellina_model import CellinaModel
from ._cellina_module import CellinaModule
from ._training_plan import CellinaAdversarialTrainingPlan
from ._spatial_utils import (
    compute_spatial_features,
    make_neighbor_perturbation,
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
    "CellinaModel",
    "CellinaModule",
    "CellinaAdversarialTrainingPlan",
    "compute_spatial_features",
    "make_neighbor_perturbation",
]

__version__ = version("cellina")
