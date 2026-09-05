import logging
import warnings
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

from ._cellina_gcn_module import CellinaGCNModule
from ._constants import DOMAINS_KEY, SPATIAL_CONNECTIVITIES_KEY
from ._edge_data_splitter import GraphBatchLoader, GraphJointDataSplitter
from ._training_plan import CellinaAdversarialTrainingPlan

logger = logging.getLogger(__name__)

class CellinaGCN(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Cellina model with dual encoders for counts (MLP) and spatial context (GCN).

    Extends scVI with a GCN spatial encoder that learns spatial aggregation
    via message passing over the spatial connectivity graph. The two latent
    representations (z from counts, s from GCN) are concatenated
    (shifted = concat(z, s)) and decoded to reconstruct count data.

    Parameters
    ----------
    adata
        AnnData registered via :meth:`~cellina.CellinaGCN.setup_anndata`.
    n_hidden
        Nodes per hidden layer (shared by both encoders).
    n_latent
        Latent dimensionality for both z and s.
    n_layers
        Hidden layers (shared by both encoders).
    discriminator_lambda
        Weight for adversarial domain forgetting. 0 disables it.
    link_prediction_weight
        Weight for spatial loss on s. 0 disables it.
    spatial_loss_type
        ``"supcon"`` (supervised contrastive, default) or ``"domain_clf"``.
    classifier_lambda
        Weight for cell-type classifier loss.
    supcon_temperature
        SupCon temperature.
    num_neighbors
        Neighbors sampled per GCN layer; its length is the number of sampled hops and should
        equal ``n_layers``. Default: ``None`` -> ``[-1] * n_layers`` (all neighbors at every
        hop). Any length other than ``n_layers`` (including length 1) emits a ``UserWarning``
        and is used as-is, sampling that many hops.
    x_spatial_layer
        Optional ``adata.layers`` key for alternative spatial features.
    use_observed_lib_size
        Must be True (graph batches require observed library size).
    convolution_type
        GCN type: ``"gcn"``, ``"gat"``, ``"gin"``, ``"sg"``.
    **model_kwargs
        Keyword args for :class:`~cellina.CellinaGCNModule`.

    Examples
    --------
    >>> CellinaGCN.setup_anndata(adata, batch_key="batch",
    ...     spatial_connectivities_key="spatial_connectivities")
    >>> model = CellinaGCN(adata, n_latent=10)
    >>> model.train()
    >>> adata.obsm["X_cellina_gcn"] = model.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        discriminator_lambda: float = 1.0,
        link_prediction_weight: float = 1.0,
        spatial_loss_type: str = "supcon",
        classifier_lambda: float = 1.0,
        supcon_temperature: float = 0.25,
        num_neighbors: List[int] = None,
        x_spatial_layer: Optional[str] = None,
        use_observed_lib_size: bool = True,
        convolution_type: str = "gat",
        subgraph_type: str = "induced",
        **model_kwargs,
    ):
        super().__init__(adata)

        self._data_splitter_cls = GraphJointDataSplitter
        self.n_layers = n_layers
        self._num_neighbors = _resolve_num_neighbors(num_neighbors, n_layers)
        self._x_spatial_layer = x_spatial_layer
        self._subgraph_type = _validate_subgraph_type(subgraph_type)
        # Lazily-built, reused across inference calls so the spatial graph / sparse X
        # store is constructed once instead of per call (avoids loop RAM creep).
        self._cached_splitter = None

        library_log_means, library_log_vars = _init_library_size(
            self.adata_manager, self.summary_stats["n_batch"]
        )

        self.module = CellinaGCNModule(
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
            link_prediction_weight=link_prediction_weight,
            spatial_loss_type=spatial_loss_type,
            classifier_lambda=classifier_lambda,
            supcon_temperature=supcon_temperature,
            use_observed_lib_size=use_observed_lib_size,
            convolution_type=convolution_type,
            **model_kwargs,
        )

        adv_str = " with adversarial domain forgetting" if discriminator_lambda > 0 else ""
        edge_str = " with edge prediction" if link_prediction_weight > 0 else ""
        self._model_summary_string = (
            f"CellinaGCN Model with {n_latent}-dim latent space "
            f"(z MLP + s GCN encoders){adv_str}{edge_str}"
        )
        self.init_params_ = self._get_init_params(locals())
        logger.info(f"The CellinaGCN model has been initialized{adv_str}{edge_str}")

    def _get_cached_splitter(self, batch_size):
        """Build (once) and reuse the splitter for the model's own adata.

        The splitter holds the spatial graph and a sparse-resident X store; rebuilding it
        per inference call is what made host RAM climb across perturbation loops. The
        per-call ``batch_size`` is supplied to ``create_inference_loader``, so a stale
        cached ``batch_size`` is harmless.
        """
        if self._cached_splitter is None:
            self._cached_splitter = GraphJointDataSplitter(
                self.adata_manager,
                num_neighbors=self._num_neighbors,
                batch_size=batch_size,
                x_spatial_layer=self._x_spatial_layer,
                subgraph_type=self._subgraph_type,
            )
        return self._cached_splitter

    def _make_data_loader(self, adata=None, indices=None, batch_size=None, shuffle=False,
                          x_spatial_layer=None):
        adata = self._validate_anndata(adata) if adata is not None else self.adata

        if batch_size is None:
            batch_size = 128
        if indices is None:
            indices = np.arange(adata.n_obs)

        spatial_layer = x_spatial_layer if x_spatial_layer is not None else self._x_spatial_layer
        splitter = self._get_cached_splitter(batch_size)

        # Perturbation cf_layer: reuse the cached graph + base X store, swapping only the
        # spatial feature store instead of rebuilding the whole splitter.
        override = (
            splitter.load_spatial_store(spatial_layer)
            if spatial_layer != self._x_spatial_layer
            else None
        )
        return splitter.create_inference_loader(
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
            x_spatial_override=override,
        )

    def _make_counterfactual_loader(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        n_neighbors_per_seed: int,
        batch_size: int = 128,
        seed: int = 0,
        subgraph_type: Optional[str] = None,
    ):
        # None inherits the model's subgraph_type (single source of truth); an explicit
        # value overrides per call. Validated against the shared allowed set either way.
        subgraph_type = _validate_subgraph_type(subgraph_type or self._subgraph_type)
        # Reuse the cached graph + sparse X store; only the edges are rewired below.
        splitter = self._get_cached_splitter(batch_size)
        pyg_data = splitter.pyg_data
        edge_index = pyg_data.edge_index.numpy()

        src, dst = edge_index[0], edge_index[1]
        keep_mask = ~(np.isin(src, indices) | np.isin(dst, indices))
        filtered_edges = edge_index[:, keep_mask]

        rng = np.random.default_rng(seed)
        neighbour_indices = np.asarray(neighbour_indices)

        if n_neighbors_per_seed >= len(neighbour_indices):
            raise ValueError(
                f"n_neighbors_per_seed ({n_neighbors_per_seed}) must be less than "
                f"len(neighbour_indices) ({len(neighbour_indices)})"
            )

        # Same RNG stream / draw order as before, so seeds reproduce the same donor sets.
        donors = [rng.choice(neighbour_indices, size=n_neighbors_per_seed, replace=False)
                  for _ in indices]
        cf_src = np.repeat(indices, n_neighbors_per_seed)
        cf_dst = np.concatenate(donors)

        # Donor -> seed only. Bidirectional edges let donors aggregate over the (control)
        # seeds under multi-hop sampling, pulling the counterfactual back toward control.
        cf_edges = np.stack([cf_dst, cf_src], axis=0)

        new_edge_index = np.concatenate([filtered_edges, cf_edges], axis=1)
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)

        # Features are gathered lazily from the splitter's sparse store; only edges are
        # rewired for the counterfactual, so no dense x is attached here.
        cf_data = Data(
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
            subgraph_type=subgraph_type,
        )
        # Counterfactual latents use base X features over the rewired graph (no spatial
        # layer), matching the original behaviour before splitter caching.
        return GraphBatchLoader(node_loader, splitter._x_sparse, None)

    @torch.inference_mode()
    def get_counterfactual_latents(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        n_neighbors_per_seed: int = 20,
        give_mean: bool = False,
        batch_size: Optional[int] = None,
        latent_key: str = "s",
        seed: int = 0,
        subgraph_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return latent representations under a counterfactual spatial neighbourhood.

        Parameters
        ----------
        indices
            Cell indices to compute counterfactual latents for.
        neighbour_indices
            Donor neighbourhood pool indices.
        n_neighbors_per_seed
            Donors per seed. Raises ValueError if >= len(neighbour_indices).
        give_mean
            Return posterior mean rather than a sample.
        batch_size
            Mini-batch size.
        latent_key
            ``'shifted'``, ``'z'``, or ``'s'``.
        seed
            Random seed.
        subgraph_type
            Counterfactual subgraph sampling mode. ``None`` (default) inherits the model's
            ``subgraph_type``; pass ``'directional'`` to keep only sampling-path edges
            (lower VRAM, output-equivalent for counterfactuals) or ``'induced'`` to
            materialise the full induced subgraph. "directional" has not been tested with
            the contrastive loss and may result in undersampled negatives.
        """
        if latent_key not in ['shifted', 'z', 's']:
            raise ValueError(f"latent_key must be 'shifted', 'z', or 's', got {latent_key}")

        self._check_if_trained(warn=False)
        indices = np.asarray(indices)
        neighbour_indices = np.asarray(neighbour_indices)
        if batch_size is None:
            batch_size = 128

        scdl = self._make_counterfactual_loader(
            indices, neighbour_indices, n_neighbors_per_seed, batch_size, seed,
            subgraph_type=subgraph_type,
        )

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
    def get_counterfactual_expression(
        self,
        indices: np.ndarray,
        neighbour_indices: np.ndarray,
        n_neighbors_per_seed: int = 20,
        batch_size: Optional[int] = None,
        seed: int = 0,
        library_size: Union[float, str] = "latent",
        return_numpy: bool = True,
        subgraph_type: Optional[str] = None,
    ) -> np.ndarray:
        """Predict gene expression under a counterfactual spatial neighbourhood.

        ``subgraph_type`` selects the counterfactual graph construction: ``None`` (default)
        inherits the model's ``subgraph_type``; ``'directional'`` keeps only sampling-path
        edges (lower VRAM, output-equivalent for counterfactuals); ``'induced'`` materialises
        the full induced subgraph (higher VRAM).
        """
        self._check_if_trained(warn=False)
        if batch_size is None:
            batch_size = 128
        scdl = self._make_counterfactual_loader(
            np.asarray(indices), np.asarray(neighbour_indices),
            n_neighbors_per_seed, batch_size, seed,
            subgraph_type=subgraph_type,
        )
        return self._compute_expression(scdl, library_size, return_numpy)

    def _make_perturbed_loader(self, adata, indices, batch_size: int, cf_layer: str):
        adata = self._validate_anndata(adata) if adata is not None else self.adata
        if cf_layer not in adata.layers:
            raise ValueError(
                f"cf_layer '{cf_layer}' not found in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        if indices is None:
            indices = np.arange(adata.n_obs)
        return self._make_data_loader(adata, indices, batch_size, x_spatial_layer=cf_layer)

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

        Parameters
        ----------
        adata
            AnnData; defaults to model's adata.
        indices
            Cell indices.
        give_mean
            Return posterior mean.
        batch_size
            Mini-batch size.
        latent_key
            ``'shifted'``, ``'z'``, or ``'s'``.
        cf_layer
            Key in ``adata.layers`` for counterfactual counts.
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
        """Predict gene expression using counterfactual node features for the GCN."""
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
            Key in ``adata.obs`` for domain labels. Required if
            ``discriminator_lambda > 0``.
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        spatial_connectivities_key
            Key in ``adata.obsp`` for the spatial connectivity matrix.

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
            Passes through the dataset.
        accelerator
            Accelerator type.
        devices
            Devices to use.
        train_size
            Training set fraction.
        validation_size
            Validation set size.
        shuffle_set_split
            Shuffle before splitting.
        batch_size
            Minibatch size.
        datasplitter_kwargs
            Extra kwargs for the data splitter.
        plan_kwargs
            Keyword args for training plan.
        """
        if plan_kwargs is None:
            plan_kwargs = {}

        self._training_plan_cls = CellinaAdversarialTrainingPlan

        datasplitter_kwargs = dict(datasplitter_kwargs or {})
        
        forbidden = {"num_neighbors", "x_spatial_layer", "subgraph_type"} & datasplitter_kwargs.keys()
        if forbidden:
            raise ValueError(
                f"{sorted(forbidden)} must be set on the CellinaGCN(...) constructor."
            )

        datasplitter_kwargs = {
            **datasplitter_kwargs,
            "num_neighbors": self._num_neighbors,
            "x_spatial_layer": self._x_spatial_layer,
            "subgraph_type": self._subgraph_type,
        }

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
            AnnData; defaults to training data.
        indices
            Cell indices.
        give_mean
            Return posterior mean.
        batch_size
            Mini-batch size.
        latent_key
            ``'shifted'``, ``'z'``, or ``'s'``.
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
            else:
                if give_mean:
                    lat = torch.cat([outputs["qzm"], outputs["qsm"]], dim=-1)
                else:
                    lat = outputs["shifted"]
            latent.append(lat.cpu())

        return torch.cat(latent).numpy()

    def get_marginal_ll(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        n_mc_samples: int = 1000,
        return_mean: bool = True,
    ):
        """
        Get marginal log-likelihood of the data.

        Parameters
        ----------
        adata
            AnnData to evaluate.
        indices
            Cell indices.
        batch_size
            Mini-batch size.
        n_mc_samples
            Monte Carlo importance-weighted samples per cell.
        return_mean
            If True, return mean over all cells.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        per_batch_mlls = []
        for tensors in scdl:
            batch_mll = self.module.marginal_ll(tensors, n_mc_samples)
            if not torch.is_tensor(batch_mll):
                batch_mll = torch.as_tensor(batch_mll)
            per_batch_mlls.append(batch_mll.cpu())
        if len(per_batch_mlls) == 0:
            return np.array([])
        all_mll = torch.cat(per_batch_mlls, dim=0).numpy()
        if return_mean:
            return float(np.mean(all_mll))
        else:
            return all_mll

    def _compute_expression(self, scdl, library_size, return_numpy):
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
        Return normalized expression.

        Parameters
        ----------
        library_size
            ``"latent"`` (inferred), a float scalar, or ``1`` for pure proportions.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        return self._compute_expression(scdl, library_size, return_numpy)


# NOTE: Resolve this, should be only one place where this can be passed...
_VALID_SUBGRAPH_TYPES = ("induced", "directional")

def _validate_subgraph_type(subgraph_type: str) -> str:
    """Validate a PyG ``NeighborLoader`` ``subgraph_type`` against the supported set.

    Single source of truth for the allowed values, shared by the constructor and the
    counterfactual inference methods so the two cannot drift.
    """
    if subgraph_type not in _VALID_SUBGRAPH_TYPES:
        raise ValueError(
            f"subgraph_type must be one of {_VALID_SUBGRAPH_TYPES}, got {subgraph_type!r}"
        )
    return subgraph_type


def _resolve_num_neighbors(num_neighbors: Optional[List[int]], n_layers: int) -> List[int]:
    """Resolve ``num_neighbors`` to one fan-out per GCN layer.

    ``NeighborLoader`` derives the number of sampled hops from ``len(num_neighbors)``, so it
    should match the GCN's message-passing depth (``n_layers``); a mismatched length truncates
    or pads the receptive field. Contract:

    - ``None``            -> ``[-1] * n_layers`` (all neighbours at every hop)
    - length ``n_layers`` -> used as-is
    - any other length    -> ``UserWarning``
    """
    if num_neighbors is None:
        return [-1] * n_layers
    num_neighbors = list(num_neighbors)
    if len(num_neighbors) != n_layers:
        warnings.warn(
            f"len(num_neighbors)={len(num_neighbors)} != n_layers={n_layers}; NeighborLoader "
            f"samples len(num_neighbors) hops differing from the GCN depth."
            f"Pass a length-{n_layers} list to silence. Got {num_neighbors}.",
            UserWarning,
        )
    return num_neighbors

