import logging
from typing import List, Optional

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    ObsmField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from ._cellina_module import CellinaModule
from ._constants import DOMAINS_KEY, SPATIAL_X_KEY
from ._training_plan import CellinaAdversarialTrainingPlan

logger = logging.getLogger(__name__)

class CellinaModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Cellina model with dual encoders for counts and spatial data.

    This model extends scVI with a spatial encoder that processes spatial features
    alongside the standard count encoder. The two latent representations (z from counts,
    s from spatial+z) are summed element-wise (shifted = z + s) and decoded together 
    to reconstruct the count data.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~cellina.CellinaModel.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer (shared by both encoders).
    n_latent
        Dimensionality of the latent space for both z and s encoders.
    n_layers
        Number of hidden layers (shared by both encoders).
    discriminator_lambda
        Weight for adversarial domain forgetting. Set to 0 (default) to disable.
    **model_kwargs
        Keyword args for :class:`~cellina.CellinaModule`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> # adata.obsm["spatial_x"] should contain spatial features
    >>> CellinaModel.setup_anndata(adata, batch_key="batch", spatial_obsm_key="spatial_x")
    >>> model = CellinaModel(adata, n_latent=10)
    >>> model.train()
    >>> adata.obsm["X_cellina"] = model.get_latent_representation()  # Returns shifted = z + s
    >>> adata.obsm["X_cellina_z"] = model.get_latent_representation(latent_key='z')
    >>> adata.obsm["X_cellina_s"] = model.get_latent_representation(latent_key='s')
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        discriminator_lambda: float = 0.0,
        **model_kwargs,
    ):
        super().__init__(adata)

        library_log_means, library_log_vars = _init_library_size(
            self.adata_manager, self.summary_stats["n_batch"]
        )

        # Get spatial input dimensions from registry
        n_spatial_input = self.summary_stats["n_spatial_x"]

        self.module = CellinaModule(
            n_input=self.summary_stats["n_vars"],
            n_spatial_input=n_spatial_input,
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_labels=self.summary_stats.get("n_labels"),
            discriminator_lambda=discriminator_lambda,
            n_domains=self.summary_stats.get("n_domains"),
            **model_kwargs,
        )
        
        # Update summary string
        adv_str = " with adversarial domain forgetting" if discriminator_lambda > 0 else ""
        self._model_summary_string = (
            f"Cellina Model with {n_latent}-dim latent space (z and s encoders){adv_str}"
        )
        self.init_params_ = self._get_init_params(locals())

        logger.info(f"The Cellina model has been initialized{adv_str}")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spatial_obsm_key: str = "spatial_x",
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        domains_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        spatial_obsm_key
            Key in `adata.obsm` containing spatial features matrix.
        %(param_batch_key)s
        %(param_labels_key)s
        domains_key
            Key in `adata.obs` for domain labels (categorical). Required if discriminator_lambda > 0.
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            ObsmField(SPATIAL_X_KEY, spatial_obsm_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(DOMAINS_KEY, domains_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 400,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Train the model.
        
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        accelerator
            Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            as well as custom accelerator instances.
        devices
            The devices to use. Can be set to a positive number (int or str), a sequence of device indices
            (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
            automatic selection based on the chosen accelerator.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the validation set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in
            the sequential order of the data according to `validation_size` and `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~cellina.CellinaAdversarialTrainingPlan` or :class:`~scvi.train.TrainingPlan`.
            Keyword arguments passed to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        # Ensure plan_kwargs is a dict we can mutate
        if plan_kwargs is None:
            plan_kwargs = {}
        
        # If adversarial training is disabled, remove plan-only keys that would be
        # handled by the adversarial plan (avoid forwarding them to module.loss)
        if self.module.discriminator_lambda == 0:
            plan_kwargs.pop("normalize_losses", None)
            plan_kwargs.pop("scale_adversarial_loss", None)
        
        # Set training plan class
        if self.module.discriminator_lambda > 0:
            # Use adversarial training plan when discriminator is enabled
            self._training_plan_cls = CellinaAdversarialTrainingPlan
        
        super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        give_mean: bool = False,
        batch_size: Optional[int] = None,
        latent_key: Optional[str] = "shifted",
    ):
        """
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use.
        give_mean
            If True, return the mean of the latent distribution. Otherwise, sample.
        batch_size
            Minibatch size for data loading into model.
        latent_key
            Which latent representation to return. Options: 'shifted', 'z', 's'.
            Default: 'shifted' (returns z + s, which is what the decoder uses).

        Returns
        -------
        Latent representation for each cell as numpy array.
        - If latent_key is 'shifted': z + s (what goes into the decoder)
        - If latent_key is 'z': only z encoder output
        - If latent_key is 's': only s encoder output
        """
        if latent_key not in ['shifted', 'z', 's']:
            raise ValueError(f"latent_key must be 'shifted', 'z', or 's', got {latent_key}")

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        latent = []

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if latent_key == 'z':
                lat = outputs["qzm"] if give_mean else outputs["z"]
            elif latent_key == 's':
                lat = outputs["qsm"] if give_mean else outputs["s"]
            else:  # shifted
                if give_mean: # TODO: this should not work like this, potentially inconsistent
                    # For mean, sum the means
                    lat = outputs["qzm"] + outputs["qsm"]
                else:
                    lat = outputs["shifted"]

            latent.append(lat.cpu())

        latent = torch.cat(latent).numpy()
        return latent
    
    def get_marginal_ll(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        n_mc_samples: int = 1000,
        reduce: str | None = None
    ):
        """Get marginal log-likelihood of the data.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use.
        batch_size
            Minibatch size for data loading into model.
        n_mc_samples
            Number of Monte Carlo samples for approximation.
        reduce
            Reduction method to apply to the marginal log-likelihoods. Options are:
            - None: return list of marginal log-likelihoods per batch
            - 'mean': return mean marginal log-likelihood across all batches
            - 'sum': return sum of marginal log-likelihoods across all batches
        Returns
        -------
        Marginal log-likelihood of the data.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        if reduce not in [None, 'mean', 'sum']:
            raise ValueError(f"Reduction must be None, 'mean' or 'sum', got {reduce}")

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        marginal_ll = []

        for tensors in scdl:
            outputs = self.module.marginal_ll(
                tensors, n_mc_samples
            )
            marginal_ll.append(outputs)

        # Apply reduction if specified
        if reduce == 'mean':
            marginal_ll = np.mean(marginal_ll)
        elif reduce == 'sum':
            marginal_ll = np.sum(marginal_ll)

        return marginal_ll
