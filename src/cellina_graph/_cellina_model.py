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
)
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from ._cellina_module import CellinaModule
from ._constants import DOMAINS_KEY, SPATIAL_CONNECTIVITIES_KEY
from ._edge_data_splitter import GraphBatchLoader, GraphJointDataSplitter
from ._training_plan import CellinaAdversarialTrainingPlan

logger = logging.getLogger(__name__)

class CellinaModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Cellina model with dual encoders for counts (MLP) and spatial context (GCN).

    This model extends scVI with a GCN spatial encoder that learns spatial aggregation
    via message passing over the spatial connectivity graph. The two latent representations
    (z from counts, s from GCN) are concatenated (shifted = concat(z, s)) and decoded
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
    link_prediction_weight
        Weight for the spatial loss on ``s``. Set to 0 (default) to disable.
        Applied to whichever ``spatial_loss_type`` is selected.
    spatial_loss_type
        Which spatial loss to apply to ``s``. One of ``"supcon"`` (supervised contrastive,
        default) or ``"domain_clf"`` (cross-entropy classifier predicting domain from ``s``).
    supcon_temperature
        Temperature for the SupCon loss (only used when ``link_prediction_weight > 0``).
        Default: 0.1.
    num_neighbors
        Number of neighbors to sample per node per GCN layer. Default: [-1] (all neighbors).
    condition_on_intrinsic
        Whether to concatenate detached z to the GCN input features before message passing.
        Defaults to False: the spatial encoder aggregates raw log-counts from neighbours,
        keeping spatial context independent of the target cell's intrinsic identity.
        Set to True to replicate non-graph Cellina behaviour.
    use_observed_lib_size
        Use observed library size for normalization. If True, use observed library size.
    convolution_type
        Graph convolution type for the spatial encoder. One of ``"gcn"``, ``"gat"``,
        ``"gin"``, ``"sg"``. Defaults to ``"gcn"``.
    **model_kwargs
        Keyword args for :class:`~cellina.CellinaModule`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> CellinaModel.setup_anndata(adata, batch_key="batch",
    ...     spatial_connectivities_key="spatial_connectivities")
    >>> model = CellinaModel(adata, n_latent=10)
    >>> model.train()
    >>> adata.obsm["X_cellina"] = model.get_latent_representation()
    >>> adata.obsm["X_cellina_z"] = model.get_latent_representation(latent_key='z')
    >>> adata.obsm["X_cellina_s"] = model.get_latent_representation(latent_key='s')
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        discriminator_lambda: float = 1.0,
        condition_on_intrinsic: bool = False,
        link_prediction_weight: float = 1.0,
        spatial_loss_type: str = "supcon",
        classifier_lambda: float = 1.0,
        supcon_temperature: float = 0.25,
        num_neighbors: List[int] = None,
        use_observed_lib_size: bool = True,
        convolution_type: str = "gcn",
        **model_kwargs,
    ):
        super().__init__(adata)

        # Always use graph-aware data splitter (GCN needs neighbors)
        self._data_splitter_cls = GraphJointDataSplitter
        self._num_neighbors = num_neighbors or [-1]
        self._data_splitter_kwargs = {
            'num_neighbors': self._num_neighbors,
        }

        library_log_means, library_log_vars = _init_library_size(
            self.adata_manager, self.summary_stats["n_batch"]
        )

        self.module = CellinaModule(
            n_input=self.summary_stats["n_vars"],
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
            link_prediction_weight=link_prediction_weight,
            spatial_loss_type=spatial_loss_type,
            classifier_lambda=classifier_lambda,
            supcon_temperature=supcon_temperature,
            use_observed_lib_size=use_observed_lib_size,
            convolution_type=convolution_type,
            **model_kwargs,
        )

        # Update summary string
        adv_str = " with adversarial domain forgetting" if discriminator_lambda > 0 else ""
        edge_str = " with edge prediction" if link_prediction_weight > 0 else ""
        self._model_summary_string = (
            f"Cellina Model with {n_latent}-dim latent space (z MLP + s GCN encoders){adv_str}{edge_str}"
        )
        self.init_params_ = self._get_init_params(locals())

        logger.info(f"The Cellina model has been initialized{adv_str}{edge_str}")

    def _make_data_loader(self, adata=None, indices=None, batch_size=None, shuffle=False, data_splitter_kwargs=None):
        """Create graph-aware data loader using NeighborLoader."""
        adata = self._validate_anndata(adata) if adata is not None else self.adata

        if batch_size is None:
            batch_size = 128
        if indices is None:
            indices = np.arange(adata.n_obs)

        # Build a lightweight splitter for inference
        splitter = GraphJointDataSplitter(
            self.adata_manager,
            num_neighbors=self._num_neighbors,
            batch_size=batch_size,
        )
        return splitter.create_inference_loader(
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _make_counterfactual_loader(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        n_neighbors_per_seed: Optional[int] = None,
        batch_size: int = 128,
        seed: int = 0,
    ):
        """
        Create a graph-aware data loader with rewired edges for counterfactual inference.

        Removes all edges involving seed nodes and replaces them with edges
        connecting each seed to nodes in ``neighbour_indices``.

        Parameters
        ----------
        indices
            Seed node indices (cells to query).
        neighbour_indices
            Indices of the counterfactual neighbourhood donor pool.
        n_neighbors_per_seed
            If None, connect each seed to ALL nodes in ``neighbour_indices``.
            Otherwise, randomly sample this many neighbours per seed.
        batch_size
            Minibatch size for the NeighborLoader.
        seed
            Random seed for neighbour sampling.
        """
        # Build PyG data from the registered AnnData (reuses splitter logic)
        splitter = GraphJointDataSplitter(
            self.adata_manager,
            num_neighbors=self._num_neighbors,
            batch_size=batch_size,
        )
        pyg_data = splitter.pyg_data
        edge_index = pyg_data.edge_index.numpy()  # [2, E]

        # Remove all edges involving any seed node (both directions)
        src, dst = edge_index[0], edge_index[1]
        keep_mask = ~(np.isin(src, indices) | np.isin(dst, indices))
        filtered_edges = edge_index[:, keep_mask]

        # Build counterfactual edges: seed <-> neighbour_indices
        rng = np.random.default_rng(seed)
        n_seeds = len(indices)
        neighbour_indices = np.asarray(neighbour_indices)

        if n_neighbors_per_seed is None or n_neighbors_per_seed >= len(neighbour_indices):
            # Connect each seed to ALL donor nodes
            cf_src = np.repeat(indices, len(neighbour_indices))
            cf_dst = np.tile(neighbour_indices, n_seeds)
        else:
            # Sample k neighbours per seed
            cf_src_parts, cf_dst_parts = [], []
            for s in indices:
                chosen = rng.choice(neighbour_indices, size=n_neighbors_per_seed, replace=False)
                cf_src_parts.append(np.full(n_neighbors_per_seed, s))
                cf_dst_parts.append(chosen)
            cf_src = np.concatenate(cf_src_parts)
            cf_dst = np.concatenate(cf_dst_parts)

        # Make bidirectional
        cf_edges = np.stack([
            np.concatenate([cf_src, cf_dst]),
            np.concatenate([cf_dst, cf_src]),
        ], axis=0)

        # Combine with filtered original edges
        new_edge_index = np.concatenate([filtered_edges, cf_edges], axis=1)
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)

        # Create new Data object sharing feature tensors with the original
        cf_data = Data(
            x=pyg_data.x,
            edge_index=new_edge_index,
            batch_labels=pyg_data.batch_labels,
            labels=pyg_data.labels,
            domains=pyg_data.domains,
            num_nodes=pyg_data.num_nodes,
        )

        node_loader = NeighborLoader(
            cf_data,
            num_neighbors=self._num_neighbors,
            input_nodes=torch.tensor(indices, dtype=torch.long),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            directed=False,
        )
        return GraphBatchLoader(node_loader)

    @torch.inference_mode()
    def get_counterfactual_latents(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        n_neighbors_per_seed: Optional[int] = None,
        give_mean: bool = False,
        batch_size: Optional[int] = None,
        latent_key: str = "s",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Return latent representations under a counterfactual spatial neighbourhood.

        Intrinsic ``z`` is computed from each cell's own counts (unchanged).
        Spatial ``s`` is computed via GCN message passing over the rewired graph
        where seed nodes receive messages from ``neighbour_indices`` instead of
        their real spatial neighbours.

        Parameters
        ----------
        indices
            Cell indices to compute counterfactual latents for.
        neighbour_indices
            Indices of the donor neighbourhood pool.
        n_neighbors_per_seed
            Number of donors to wire per seed. None = use all.
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
            indices, neighbour_indices, n_neighbors_per_seed, batch_size, seed
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
        n_neighbors_per_seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        seed: int = 0,
        library_size: Union[float, str] = "latent",
        return_numpy: bool = True,
    ) -> np.ndarray:
        """
        Predict gene expression under a counterfactual spatial neighbourhood.

        Runs the full inference + generative pipeline with rewired graph edges.
        Library normalisation follows the same logic as :meth:`get_normalized_expression`.

        Parameters
        ----------
        indices
            Cell indices to predict counterfactual expression for.
        neighbour_indices
            Indices of the donor neighbourhood pool.
        n_neighbors_per_seed
            Number of donors to wire per seed. None = use all.
        batch_size
            Minibatch size for the loader.
        seed
            Random seed for neighbour sampling.
        library_size
            - ``"latent"``: uses inferred library size (returns px_rate). Default.
            - float (e.g. 1e4): multiplies px_scale by this constant.
            - 1: returns px_scale (pure proportions).
        return_numpy
            If True, return np.ndarray; otherwise return torch.Tensor.

        Returns
        -------
        np.ndarray of shape ``(len(indices), n_genes)``.
        """
        self._check_if_trained(warn=False)
        if batch_size is None:
            batch_size = 128
        scdl = self._make_counterfactual_loader(
            np.asarray(indices), np.asarray(neighbour_indices),
            n_neighbors_per_seed, batch_size, seed,
        )
        return self._compute_expression(scdl, library_size, return_numpy)

    def _make_perturbed_loader(
        self,
        adata,
        indices,
        batch_size: int,
        cf_layer: str,
    ):
        """Create an inference loader that swaps GCN node features with adata.layers[cf_layer]."""
        adata = self._validate_anndata(adata) if adata is not None else self.adata
        if cf_layer not in adata.layers:
            raise ValueError(
                f"cf_layer '{cf_layer}' not found in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        if indices is None:
            indices = np.arange(adata.n_obs)
        splitter = GraphJointDataSplitter(
            self.adata_manager,
            num_neighbors=self._num_neighbors,
            batch_size=batch_size,
            cf_layer=cf_layer,
        )
        return splitter.create_inference_loader(indices=indices, batch_size=batch_size, shuffle=False)

    @torch.inference_mode()
    def get_perturbed_latents(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        give_mean: bool = False,
        batch_size: Optional[int] = None,
        latent_key: str = "s",
        cf_layer: str = "counts_cf",
    ) -> np.ndarray:
        """
        Return latent representations using counterfactual node features for the GCN.

        ``z`` is computed from each cell's own counts (unchanged).
        ``s`` is computed via GCN message passing with node features drawn from
        ``adata.layers[cf_layer]`` instead of ``adata.X``.

        Parameters
        ----------
        adata
            AnnData object; defaults to the model's registered adata.
        indices
            Cell indices to use.
        give_mean
            Return the posterior mean rather than a sample.
        batch_size
            Mini-batch size for inference.
        latent_key
            Which latent to return: ``'shifted'``, ``'z'``, or ``'s'``. Default ``'s'``.
        cf_layer
            Key in ``adata.layers`` holding the counterfactual count matrix
            (raw counts, same shape as ``adata.X``).
        """
        if latent_key not in ['shifted', 'z', 's']:
            raise ValueError(f"latent_key must be 'shifted', 'z', or 's', got {latent_key}")

        self._check_if_trained(warn=False)
        if batch_size is None:
            batch_size = 128

        scdl = self._make_perturbed_loader(adata, indices, batch_size, cf_layer)

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if latent_key == 'z':
                lat = outputs["qzm"] if give_mean else outputs["z"]
            elif latent_key == 's':
                lat = outputs["qsm"] if give_mean else outputs["s"]
            else:
                if give_mean:
                    lat = torch.cat([outputs["qzm"], outputs["qsm"]], dim=-1)
                else:
                    lat = outputs["shifted"]
            latent.append(lat.cpu())

        return torch.cat(latent).numpy()

    @torch.inference_mode()
    def get_perturbed_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        cf_layer: str = "counts_cf",
        library_size: Union[float, str] = "latent",
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict gene expression using counterfactual node features for the GCN.

        Library size is computed from each cell's own observed ``adata.X`` counts.

        Parameters
        ----------
        adata
            AnnData object; defaults to the model's registered adata.
        indices
            Cell indices to use.
        batch_size
            Mini-batch size for inference.
        cf_layer
            Key in ``adata.layers`` holding the counterfactual count matrix.
        library_size
            ``"latent"`` (default), a float scalar, or ``1`` for pure proportions.
        return_numpy
            If True, return np.ndarray; otherwise return torch.Tensor.
        """
        self._check_if_trained(warn=False)
        if batch_size is None:
            batch_size = 128
        scdl = self._make_perturbed_loader(adata, indices, batch_size, cf_layer)
        return self._compute_expression(scdl, library_size, return_numpy)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        domains_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        spatial_connectivities_key: str = "spatial_connectivities",
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        domains_key
            Key in `adata.obs` for domain labels (categorical). Required if discriminator_lambda > 0.
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        spatial_connectivities_key
            Key in `adata.obsp` containing spatial connectivity matrix. Required for GCN message passing.

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(DOMAINS_KEY, domains_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)

        # Store spatial connectivities key for the graph data splitter
        adata.uns[SPATIAL_CONNECTIVITIES_KEY] = spatial_connectivities_key

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
            Supports passing different accelerator types.
        devices
            The devices to use.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the validation set.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size to use during training.
        datasplitter_kwargs
            Additional keyword arguments passed into the data splitter.
        plan_kwargs
            Keyword args for training plan.
        **kwargs
            Other keyword args for Trainer.
        """
        if plan_kwargs is None:
            plan_kwargs = {}

        if self.module.discriminator_lambda == 0:
            plan_kwargs.pop("normalize_losses", None)
            plan_kwargs.pop("scale_adversarial_loss", None)

        if self.module.discriminator_lambda > 0:
            self._training_plan_cls = CellinaAdversarialTrainingPlan

        # Merge stored data splitter kwargs with user-provided ones
        if datasplitter_kwargs is None:
            datasplitter_kwargs = {}
        merged_kwargs = {**self._data_splitter_kwargs, **datasplitter_kwargs}
        datasplitter_kwargs = merged_kwargs

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

        Returns
        -------
        Latent representation for each cell as numpy array.
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
                if give_mean:
                    lat = torch.cat([outputs["qzm"], outputs["qsm"]], dim=-1)
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
        return_mean: bool = True,
    ):
        """Get marginal log-likelihood of the data.

        Parameters
        ----------
        adata
            AnnData object to evaluate. Defaults to the registered training data.
        indices
            Cell indices to evaluate. Defaults to all cells.
        batch_size
            Mini-batch size for the data loader.
        n_mc_samples
            Number of Monte Carlo importance-weighted samples per cell.
        return_mean
            If True (default), return the mean log-likelihood over all cells as a
            float. If False, return a 1D numpy array of per-cell log-likelihoods.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        per_batch_mlls = []
        for tensors in scdl:
            batch_mll = self.module.marginal_ll(tensors, n_mc_samples)
            if not torch.is_tensor(batch_mll):
                batch_mll = torch.as_tensor(batch_mll)
            per_batch_mlls.append(batch_mll.cpu())
        if len(per_batch_mlls) == 0:
            return np.array([])
        all_mll = torch.cat(per_batch_mlls, dim=0).numpy()  # [n_cells]
        if return_mean:
            return float(np.mean(all_mll))
        else:
            return all_mll


    def _compute_expression(
        self,
        scdl,
        library_size: Union[float, str],
        return_numpy: bool,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Shared inference loop used by get_normalized_expression and get_counterfactual_expression."""
        exprs = []
        with torch.no_grad():
            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inference_inputs)
                generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
                generative_outputs = self.module.generative(**generative_inputs)
                px_scale = generative_outputs["px_scale"]
                if library_size == "latent":
                    lib = torch.exp(inference_outputs["library"])
                    px = px_scale * lib
                else:
                    px = px_scale * library_size
                exprs.append(px.cpu())
        exprs = torch.cat(exprs, dim=0)
        return exprs.numpy() if return_numpy else exprs

    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        library_size: Union[float, str] = 'latent',
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
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        return self._compute_expression(scdl, library_size, return_numpy)
