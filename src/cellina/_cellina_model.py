import logging
from typing import List, Optional, Union

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
from ._utils import make_counterfactual_adata

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
        condition_on_intrinsic: bool = True,
        use_observed_lib_size: bool = True,
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
            condition_on_intrinsic=condition_on_intrinsic,
            use_observed_lib_size=use_observed_lib_size,
            **model_kwargs,
        )

        # Update summary string
        adv_str = " with adversarial domain forgetting" if discriminator_lambda > 0 else ""
        self._model_summary_string = (
            f"Cellina Model with {n_latent}-dim latent space (z and s encoders){adv_str}"
        )
        self.init_params_ = self._get_init_params(locals())

        logger.info(f"The Cellina model has been initialized{adv_str}")

        # TODO: should this be here?
        # Store the obsm key that was registered for spatial features so that
        # perturbation methods can temporarily swap it.
        self._spatial_obsm_key = next(
            f._attr_key
            for f in self.adata_manager.fields
            if getattr(f, '_registry_key', None) == SPATIAL_X_KEY
        )

    def _make_counterfactual_loader(
            self,
            indices: np.ndarray,
            neighbour_indices: np.ndarray,
            batch_size: int = 128,
            seed: int = 0,
            adata: Optional[AnnData] = None,
        ):
            """
            Create a data loader that yields tensors for counterfactual evaluation.

            This function constructs a temporary AnnData containing only the `indices`
            cells with their `.obsm[SPATIAL_X_KEY]` replaced by sampled spatial counts
            from `neighbour_indices` (using :func:`make_counterfactual_adata`). It then
            delegates to the model's `_make_data_loader` to produce the same batch-wise
            tensor tuples used during normal inference.

            Parameters
            ----------
            indices
                Indices of control cells (basal) to evaluate.
            neighbour_indices
                Indices of donor cells to sample spatial information from.
            batch_size
                Batch size for the returned dataloader.
            seed
                Random seed forwarded to the sampling routine.
            adata
                Optional AnnData to use instead of self.adata for generating the counterfactual loader.

            Returns
            -------
            An iterable dataloader yielding the same tensor dicts as ``_make_data_loader``.
            """
            adata_cf = make_counterfactual_adata(
                self.adata if adata is None else adata,  # use provided adata or default to self.adata
                indices, 
                neighbour_indices, 
                spatial_column=SPATIAL_X_KEY, 
                sample=False, 
                random_state=seed
            )
            return self._make_data_loader(adata=adata_cf, batch_size=batch_size)


    @torch.inference_mode()
    def get_counterfactual_latents(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        adata: Optional[AnnData] = None,
        give_mean: bool = False,
        batch_size: Optional[int] = None,
        latent_key: str = "shifted",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Return latent representations under a counterfactual spatial neighbourhood.

        Intrinsic ``z`` is computed from each cell's own counts (unchanged).
        Spatial ``s`` is computed via the neighbors of ``neighbour_indices`` instead of
        their real spatial neighbours.

        Parameters
        ----------
        indices
            Cell indices to compute counterfactual latents for.
        neighbour_indices
            Indices of donor cells to sample spatial information from.
        adata
            Optional AnnData to use instead of self.adata for generating the counterfactual loader.
        give_mean
            If True, return the mean of the latent distribution.
        batch_size
            Minibatch size for the loader.
        latent_key
            Which latent to return: ``'shifted'``, ``'z'``, or ``'s'``.
        seed
            Random seed for neighbour sampling.

        Returns
        -------
        np.ndarray of shape ``(len(indices), n_latent)`` (or ``2*n_latent`` for shifted).
        """
        if latent_key not in ['shifted', 'z', 's']:
            raise ValueError(f"latent_key must be 'shifted', 'z', or 's', got {latent_key}")

        self._check_if_trained(warn=False)
        indices = np.asarray(indices)
        neighbour_indices = np.asarray(neighbour_indices)
        if batch_size is None:
            batch_size = 128

        scdl = self._make_counterfactual_loader(
            indices, neighbour_indices, batch_size, seed, adata
        )

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if latent_key == 'z':
                lat = outputs["qzm"] if give_mean else outputs["z"]
            elif latent_key == 's':
                lat = outputs["qsm"] if give_mean else outputs["s"]
            else:  # shifted
                if give_mean:
                    lat = torch.cat([outputs["qzm"], outputs["qsm"]], dim=-1)
                else:
                    lat = outputs["shifted"]

            latent.append(lat.cpu())

        return torch.cat(latent).numpy()


    @torch.inference_mode()
    def get_counterfactual_expression(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Predict gene expression under a counterfactual spatial neighbourhood.

        Runs the full inference + generative pipeline with rewired graph edges,
        returning the mean of the negative binomial distribution (``px_rate``).

        Parameters
        ----------
        indices
            Cell indices to predict counterfactual expression for.
        neighbour_indices
            Indices of donor cells to sample spatial information from.
        adata
            Optional AnnData to use instead of self.adata for generating the counterfactual loader.
        batch_size
            Minibatch size for the loader.
        seed
            Random seed for neighbour sampling.

        Returns
        -------
        np.ndarray of shape ``(len(indices), n_genes)``.
        """
        self._check_if_trained(warn=False)
        indices = np.asarray(indices)
        neighbour_indices = np.asarray(neighbour_indices)
        if batch_size is None:
            batch_size = 128

        scdl = self._make_counterfactual_loader(
            indices, neighbour_indices, batch_size, seed, adata
        )

        expressions = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = self.module.generative(**generative_inputs)

            expressions.append(generative_outputs["px_rate"].cpu())

        return torch.cat(expressions).numpy()
    

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
    
    @torch.inference_mode()
    def get_perturbed_latents(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        give_mean: bool = False,
        batch_size: Optional[int] = None,
        latent_key: Optional[str] = "s",
        spatial_obsm_key: str = "spatial_x_cf",
    ) -> np.ndarray:
        """
        Return latent representation using counterfactual spatial features.

        Temporarily swaps ``adata.obsm[registered_spatial_key]`` with
        ``adata.obsm[spatial_obsm_key]``, runs inference, then restores the
        original data.

        Parameters
        ----------
        adata
            AnnData object; defaults to the model's registered adata.
        indices
            Cell indices to use.
        give_mean
            Return the mean of the posterior rather than a sample.
        batch_size
            Mini-batch size for inference.
        latent_key
            Which latent to return: ``'shifted'``, ``'z'``, or ``'s'``.
            Default is ``'s'`` (the spatially-informed latent).
        spatial_obsm_key
            Key in ``adata.obsm`` that holds the counterfactual spatial features
            (written by :func:`~cellina.make_neighbor_perturbation`).
        """
        adata = self._validate_anndata(adata)
        orig = adata.obsm[self._spatial_obsm_key].copy()
        adata.obsm[self._spatial_obsm_key] = adata.obsm[spatial_obsm_key]
        try:
            return self.get_latent_representation(
                adata=adata,
                indices=indices,
                give_mean=give_mean,
                batch_size=batch_size,
                latent_key=latent_key,
            )
        finally:
            adata.obsm[self._spatial_obsm_key] = orig

    @torch.inference_mode()
    def get_perturbed_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        spatial_obsm_key: str = "spatial_x_cf",
        library_size: Union[float, str] = 1.0,
    ) -> np.ndarray:
        """
        Return normalised expression using counterfactual spatial features.

        Temporarily swaps ``adata.obsm[registered_spatial_key]`` with
        ``adata.obsm[spatial_obsm_key]``, runs inference and decoding, then
        restores the original data.

        Parameters
        ----------
        adata
            AnnData object; defaults to the model's registered adata.
        indices
            Cell indices to use.
        batch_size
            Mini-batch size for inference.
        spatial_obsm_key
            Key in ``adata.obsm`` that holds the counterfactual spatial features.
        library_size
            Passed directly to :meth:`get_normalized_expression`.
        """
        adata = self._validate_anndata(adata)
        orig = adata.obsm[self._spatial_obsm_key].copy()
        adata.obsm[self._spatial_obsm_key] = adata.obsm[spatial_obsm_key]
        try:
            return self.get_normalized_expression(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
                library_size=library_size,
            )
        finally:
            adata.obsm[self._spatial_obsm_key] = orig

    def get_marginal_ll(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        n_mc_samples: int = 1000,
        return_mean: bool = True,
    ):
        """Get marginal log-likelihood of the data.
        ...
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        per_batch_mlls = []

        for tensors in scdl:
            # returns a 1D tensor per batch (per-cell log-likelihoods)
            batch_mll = self.module.marginal_ll(tensors, n_mc_samples)
            # ensure tensor on CPU
            if not torch.is_tensor(batch_mll):
                batch_mll = torch.as_tensor(batch_mll)
            per_batch_mlls.append(batch_mll.cpu())

        if len(per_batch_mlls) == 0:
            return np.array([])

        # concatenate per-cell log-likelihoods across batches
        all_mll = torch.cat(per_batch_mlls, dim=0).numpy()

        if return_mean:
            return float(np.mean(all_mll))
        else:
            # return per-cell array
            return all_mll


    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        library_size: Union[float, str] = 1.,
    ):
        """
        Return normalized expression like scvi-tools.

        Parameters
        ----------
        library_size
            - float (e.g. 1e4): multiplies px_scale by this constant
            - 1: returns px_scale (pure proportions)
            - "latent": uses inferred latent library size (returns px_rate)
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        exprs = []
        with torch.no_grad():
            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inference_inputs)

                generative_inputs = self.module._get_generative_input(
                    tensors, inference_outputs
                )
                generative_outputs = self.module.generative(**generative_inputs)

                px_scale = generative_outputs["px_scale"]

                if library_size == "latent":
                    # inferred library size per cell
                    lib = torch.exp(inference_outputs["library"])
                    px = px_scale * lib
                else:
                    px = px_scale * library_size
                
                exprs.append(px.cpu())

        exprs = torch.cat(exprs, dim=0)

        if return_numpy:
            return exprs.numpy()

        return exprs
