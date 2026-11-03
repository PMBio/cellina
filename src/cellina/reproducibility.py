import os

import scvi
import torch


def seed_everything(seed=0):
    """Seed every RNG cellina touches and force deterministic CUDA kernels.

    Call before any CUDA work so ``CUBLAS_WORKSPACE_CONFIG`` takes effect.
    """
    # Seeds python/numpy/torch(+CUDA)/Lightning, sets cudnn deterministic, and
    # seeds scvi's train/val split (it reads ``scvi.settings.seed``).
    scvi.settings.seed = seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
