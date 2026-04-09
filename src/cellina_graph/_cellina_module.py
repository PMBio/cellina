from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.module._classifier import Classifier
from scvi.nn import DecoderSCVI, Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ._constants import DOMAINS_KEY
from ._spatial_encoder import GraphEncoder

TensorDict = Dict[str, torch.Tensor]


class CellinaModule(BaseModuleClass):
    """
    Cellina module with dual encoders (z from counts, s from spatial GCN).

    This module implements a dual-encoder variational autoencoder where:
    - z_encoder (MLP) processes count data to produce latent representation z
    - s_encoder (GCN) processes node features via spatial graph message passing to produce s
    - decoder reconstructs counts from shifted = concat(z, s)

    Parameters
    ----------
    n_input
        Number of input genes.
    library_log_means
        1 x n_batch array of means of the log library sizes.
    library_log_vars
        1 x n_batch array of variances of the log library sizes.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer (shared by both encoders).
    n_latent
        Dimensionality of the latent space for both z and s encoders.
    n_layers
        Number of hidden layers (shared by both encoders).
    dropout_rate
        Dropout rate for neural networks.
    gene_likelihood
        One of "zinb" or "nb".
    classifier_lambda
        Weight for the supervised classifier loss. Set to 0 (default) to disable classifier.
    classifier_kwargs
        Extra keyword args forwarded to :class:`~scvi.module._classifier.Classifier`.
    n_labels
        Number of labels for the optional classifier head.
    discriminator_lambda
        Weight for the adversarial domain discriminator loss. Set to 0 (default) to disable.
    discriminator_kwargs
        Extra keyword args forwarded to domain discriminator Classifier.
    n_domains
        Number of domain labels.
    condition_on_intrinsic
        Whether to concatenate detached z to the GCN input features before message passing.
        Defaults to True in this module; the graph-aware CellinaModel overrides this to False —
        keeping spatial context independent of the target cell's intrinsic identity.
    link_prediction_weight
        Weight for the spatial loss on ``s``. Set to 0 (default) to disable.
        Applied to whichever ``spatial_loss_type`` is selected.
    spatial_loss_type
        Which spatial loss to apply to ``s``. One of ``"supcon"`` (supervised contrastive,
        default) or ``"domain_clf"`` (cross-entropy classifier predicting domains from ``s``).
    supcon_require_same_domain
        If True, only same-domain neighbours count as positives in the SupCon loss and
        negatives are different-domain nodes. If False (default), all neighbours are
        positives and non-neighbours are negatives (pure spatial contrastive).
        Only used when ``spatial_loss_type="supcon"``.
    convolution_type
        Which graph convolution to use in the spatial encoder. One of ``"gcn"``, ``"gat"``,
        ``"gin"``, ``"sg"``. Defaults to ``"gat"``.
    """

    def __init__(
        self,
        n_input: int,
        library_log_means: torch.Tensor,
        library_log_vars: torch.Tensor,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "zinb",
        classifier_lambda: float = 0.0,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        n_labels: Optional[int] = None,
        discriminator_lambda: float = 0.0,
        discriminator_kwargs: Optional[Dict[str, Any]] = None,
        n_domains: Optional[int] = None,
        condition_on_intrinsic: bool = True,
        link_prediction_weight: float = 0.0,
        spatial_loss_type: str = "supcon",
        supcon_temperature: float = 0.25,
        supcon_require_same_domain: bool = True,
        use_observed_lib_size: bool = True,
        use_batch_norm: bool = False, # TODO: can double check later if GCN batch norm is correctly done for edge cases (e.g. supcon, etc.)
        convolution_type: str = "gcn",
    ):
        super().__init__()
        if not use_observed_lib_size:
            raise NotImplementedError(
                "cellina_graph only supports use_observed_lib_size=True. "
                "Latent library size is not implemented for graph-aware batches."
            )
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        self.classifier_lambda = classifier_lambda
        self.discriminator_lambda = discriminator_lambda
        self.link_prediction_weight = link_prediction_weight
        self.spatial_loss_type = spatial_loss_type
        self.supcon_temperature = supcon_temperature
        self.supcon_require_same_domain = supcon_require_same_domain
        self.latent_distribution = "normal"
        self.use_observed_lib_size = use_observed_lib_size

        self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
        self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        # Batch injection setup
        cat_list = [n_batch] if n_batch > 0 else None

        # Z encoder: counts -> z (MLP)
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        # S encoder: node features -> s (GCN with spatial message passing)
        self.condition_on_intrinsic = condition_on_intrinsic
        n_input_s = n_input + n_latent if condition_on_intrinsic else n_input
        self.s_encoder = GraphEncoder(
            n_input=n_input_s,
            n_output=n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            convolution_type=convolution_type,
        )

        # Library encoder
        self.l_encoder = Encoder(
            n_input,
            1,
            n_cat_list=cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        # Decoder: shifted (z concat s) -> counts
        self.decoder = DecoderSCVI(
            n_latent * 2,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        # Cell type classifier
        self.classifier: Optional[Classifier] = None
        if classifier_lambda > 0:
            classifier_kwargs = dict(classifier_kwargs or {})
            self.classifier = Classifier(
                n_input=n_latent,
                n_labels=n_labels,
                logits=True,
                **classifier_kwargs
            )

        # Domain discriminator
        self.domain_discriminator: Optional[Classifier] = None
        if discriminator_lambda > 0:
            if n_domains is None or n_domains < 2:
                raise ValueError(
                    "discriminator_lambda > 0 requires n_domains >= 2. "
                    "Please provide domains_key in setup_anndata()."
                )
            discriminator_kwargs = dict(discriminator_kwargs or {})
            self.domain_discriminator = Classifier(
                n_input=n_latent,
                n_labels=n_domains,
                n_hidden=discriminator_kwargs.pop("n_hidden", 32),
                n_layers=discriminator_kwargs.pop("n_layers", 2),
                logits=True,
                **discriminator_kwargs
            )

        # Spatial domain classifier on s (positive direction: s should encode domain)
        self.s_domain_classifier: Optional[Classifier] = None
        if link_prediction_weight > 0 and spatial_loss_type == "domain_clf":
            if n_domains is None or n_domains < 2:
                raise ValueError(
                    "spatial_loss_type='domain_clf' requires n_domains >= 2. "
                    "Please provide domains_key in setup_anndata()."
                )
            self.s_domain_classifier = Classifier(
                n_input=n_latent,
                n_labels=n_domains,
                logits=True,
            )

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args."""
        if 'node_batch' not in tensors:
            raise ValueError(
                "CellinaModule requires graph-aware batches with 'node_batch' key. "
                "Use a NeighborLoader-based data loader."
            )

        node_batch = tensors['node_batch']
        return dict(
            x=node_batch['X'],
            batch_index=node_batch['batch_label'],
            edge_index=node_batch['edge_index'],
            batch_size=node_batch['batch_size'],
        )

    def _get_generative_input(self, tensors, inference_outputs):
        shifted = inference_outputs["shifted"]
        library = inference_outputs["library"]

        batch_index = tensors['node_batch']['batch_label']
        batch_size = tensors['node_batch']['batch_size']
        batch_index = batch_index[:batch_size]

        return {
            "shifted": shifted,
            "library": library,
            "batch_index": batch_index,
        }

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        edge_index,
        batch_size,
        n_samples=1,
    ):
        """
        High level inference method.

        Runs the inference (encoder) model for node reconstruction.
        When link_prediction_weight > 0, also returns neighbor means for SupCon.
        """
        x_ = torch.log(1 + x)

        # Encode counts -> z (MLP)
        qzm, qzv, z = self.z_encoder(x_, batch_index)

        # Prepare GCN input features
        if self.condition_on_intrinsic:
            spatial_input = torch.cat([x_, z.detach()], dim=-1)
        else:
            spatial_input = x_

        # Encode spatial -> s (GCN with message passing); slices to seed nodes internally
        qsm, qsv, s, neighbor_means = self.s_encoder(
            spatial_input, edge_index, batch_index,
            batch_size=batch_size,
            return_neighbor_means=(
                self.link_prediction_weight > 0 and self.spatial_loss_type == "supcon"
            ),
        )

        # Library size
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
            qlm, qlv = None, None
        else:
            qlm, qlv, library = self.l_encoder(x_, batch_index)

        # Slice z outputs to actual batch size (not including sampled neighbors)
        qzm = qzm[:batch_size, :]
        qzv = qzv[:batch_size, :]
        z_sliced = z[:batch_size, :]
        shifted = torch.cat([z_sliced, s], dim=-1)
        qlm = qlm[:batch_size, :] if qlm is not None else None
        qlv = qlv[:batch_size, :] if qlv is not None else None
        library = library[:batch_size, :]

        outputs = dict(
            z=z_sliced,
            qzm=qzm,
            qzv=qzv,
            s=s,
            qsm=qsm,
            qsv=qsv,
            shifted=shifted,
            library=library,
            qlm=qlm,
            qlv=qlv,
            edge_index=edge_index,
            neighbor_means=neighbor_means,
        )

        # Cell type classifier (on sliced z)
        if self.classifier is not None:
            outputs["classifier_logits"] = self.classifier(z_sliced)

        # Domain discriminator (on sliced z)
        if self.domain_discriminator is not None:
            outputs["discriminator_logits"] = self.domain_discriminator(z_sliced)

        # Spatial domain classifier (on s)
        if self.s_domain_classifier is not None:
            outputs["s_domain_logits"] = self.s_domain_classifier(s)

        return outputs

    @auto_move_data
    def generative(self, shifted, library, batch_index):
        """Runs the generative model."""
        px_scale, _, px_rate, px_dropout = self.decoder("gene", shifted, library, batch_index)
        px_r = torch.exp(self.px_r)

        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def _compute_classifier_metrics(
        self,
        classifier: Optional[Classifier],
        weight: float,
        inference_outputs: dict,
        labels: torch.Tensor,
        reconst_loss_shape: torch.Tensor,
        metric_name: str,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute loss and accuracy for a classifier (cell type classifier or domain discriminator).
        """
        if weight != 0.0 and classifier is not None:
            logits_key = f"{metric_name}_logits"
            logits = inference_outputs.get(logits_key)
            if logits is None:
                logits = classifier(inference_outputs["z"])

            loss = F.cross_entropy(logits, labels, reduction="none")

            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()

            return loss, accuracy
        else:
            return torch.zeros_like(reconst_loss_shape), 0.0

    def _compute_supcon_loss(
        self,
        qsm: torch.Tensor,
        neighbor_means: torch.Tensor,
        edge_index: torch.Tensor,
        domains_all: torch.Tensor,
        batch_size: int,
        temperature: float,
        require_same_domain: bool = True,
    ) -> torch.Tensor:
        """
        Spatial supervised contrastive loss on s.

        When require_same_domain=False (default):
            P(i): all spatial neighbours j
            N(i): any node j where domains[j] != domains[i]           (different niche)

        When require_same_domain=True:
            P(i): spatial neighbours j where domains[j] == domains[i] (same niche)
            N(i): any node j where domains[j] != domains[i]           (different niche)

        Seed nodes with no valid positive or no valid negative are excluded.
        """
        batch_size = int(batch_size)
        s_all = torch.cat([qsm, neighbor_means], dim=0)   # (N_total, n_latent)
        s_all = F.normalize(s_all, p=2, dim=1)

        src, dst = edge_index[0], edge_index[1]

        loss_total = torch.tensor(0.0, device=qsm.device)
        n_valid = 0

        for i in range(batch_size):
            # Neighbours of seed node i
            edge_mask = src == i
            neighbor_idx = dst[edge_mask]
            if len(neighbor_idx) == 0:
                continue

            if require_same_domain:
                # Positive set: same-niche neighbours only
                pos_mask = domains_all[neighbor_idx] == domains_all[i]
                pos_idx = neighbor_idx[pos_mask]
                if len(pos_idx) == 0:
                    continue
            else:
                # Positive set: all neighbours
                pos_idx = neighbor_idx

            # Negative set: different-domain nodes
            neg_mask = domains_all != domains_all[i]

            if neg_mask.sum() == 0:
                continue

            sim_i = (s_all[i].unsqueeze(0) * s_all).sum(dim=-1) / temperature  # (N_total,)
            sim_i[i] = float('-inf')                                              # exclude self

            # Denominator: positives union negatives (minus self)
            denom_mask = torch.zeros(s_all.size(0), dtype=torch.bool, device=qsm.device)
            denom_mask[pos_idx] = True
            denom_mask = denom_mask | neg_mask
            denom_mask[i] = False

            log_denom = torch.logsumexp(sim_i[denom_mask], dim=0)
            log_pos   = sim_i[pos_idx] - log_denom               # (|P(i)|,)

            loss_total = loss_total + (-log_pos.mean())
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=qsm.device)
        return loss_total / n_valid

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        discriminator_lambda: float = 0.0,
        classifier_scale: float = 1.0,
        discriminator_scale: float = 1.0,
        spatial_scale: float = 1.0,
    ):
        """Loss function."""
        # Graph-aware batch format
        x = tensors["node_batch"]['X']
        batch_index = tensors["node_batch"]['batch_label']
        batch_size = tensors["node_batch"]['batch_size']
        x = x[:batch_size, :]
        batch_index = batch_index[:batch_size]
        labels = tensors["node_batch"].get(
            REGISTRY_KEYS.LABELS_KEY,
            torch.zeros(batch_size, dtype=torch.long, device=x.device)
        ).reshape(-1).long()[:batch_size]
        domain_labels = tensors["node_batch"].get(
            DOMAINS_KEY,
            torch.zeros(batch_size, dtype=torch.long, device=x.device)
        ).reshape(-1).long()[:batch_size]

        qzm = inference_outputs["qzm"]
        qzv = inference_outputs["qzv"]
        qsm = inference_outputs["qsm"]
        qsv = inference_outputs["qsv"]
        qlm = inference_outputs["qlm"]
        qlv = inference_outputs["qlv"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # Reconstruction loss
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)

        # KL divergence for z
        mean = torch.zeros_like(qzm)
        scale = torch.ones_like(qzv)
        kl_divergence_z = kl(Normal(qzm, torch.sqrt(qzv)), Normal(mean, scale)).sum(dim=1)

        # KL divergence for s
        mean_s = torch.zeros_like(qsm)
        scale_s = torch.ones_like(qsv)
        kl_divergence_s = kl(Normal(qsm, torch.sqrt(qsv)), Normal(mean_s, scale_s)).sum(dim=1)

        # KL divergence for library
        #batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        batch_index = tensors['node_batch']['batch_label']
        batch_size = tensors['node_batch']['batch_size']
        batch_index = batch_index[:batch_size]
        if not self.use_observed_lib_size:
            local_library_log_means, local_library_log_vars = self._get_local_library_params(batch_index)
            kl_divergence_l = kl(
                Normal(qlm, torch.sqrt(qlv)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            # no library latent KL when using observed library sizes
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local_for_warmup = kl_divergence_z + kl_divergence_s
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Cell type classifier
        classifier_loss_raw, classifier_accuracy = self._compute_classifier_metrics(
            classifier=self.classifier,
            weight=self.classifier_lambda,
            inference_outputs=inference_outputs,
            labels=labels,
            reconst_loss_shape=reconst_loss,
            metric_name="classifier",
        )
        classifier_loss_scaled = (classifier_loss_raw * classifier_scale * self.classifier_lambda).mean()

        # Domain discriminator (fool loss)
        fool_ce, discriminator_accuracy = self._compute_classifier_metrics(
            classifier=self.domain_discriminator,
            weight=discriminator_lambda,
            inference_outputs=inference_outputs,
            labels=domain_labels,
            reconst_loss_shape=reconst_loss,
            metric_name="discriminator",
        )
        fool_loss_raw = -fool_ce  # negate for adversarial direction (maximize disc CE)
        fool_loss_scaled = (fool_loss_raw * discriminator_scale * discriminator_lambda).mean()

        # Spatial loss on s: either SupCon or domain classifier
        # Labels/domains for ALL subgraph nodes (seeds + neighbours, unsliced)
        domains_all = tensors["node_batch"].get(
            DOMAINS_KEY,
            torch.zeros(tensors["node_batch"]['X'].shape[0], dtype=torch.long, device=reconst_loss.device)
        ).reshape(-1).long()

        spatial_loss_raw = torch.tensor(0.0, device=reconst_loss.device)
        s_domain_accuracy = 0.0
        if self.link_prediction_weight > 0:
            if self.spatial_loss_type == "supcon":
                spatial_loss_raw = self._compute_supcon_loss(
                    qsm=inference_outputs["qsm"],
                    neighbor_means=inference_outputs["neighbor_means"],
                    edge_index=inference_outputs["edge_index"],
                    domains_all=domains_all,
                    batch_size=batch_size,
                    temperature=self.supcon_temperature,
                    require_same_domain=self.supcon_require_same_domain,
                )
            elif self.spatial_loss_type == "domain_clf":
                s_clf_loss, s_domain_accuracy = self._compute_classifier_metrics(
                    classifier=self.s_domain_classifier,
                    weight=self.link_prediction_weight,
                    inference_outputs=inference_outputs,
                    labels=domain_labels,
                    reconst_loss_shape=reconst_loss,
                    metric_name="s_domain",
                )
                spatial_loss_raw = s_clf_loss.mean()
        spatial_loss = spatial_loss_raw * self.link_prediction_weight * spatial_scale

        # Total loss
        vae_loss_tensor = reconst_loss + weighted_kl_local
        vae_loss = torch.mean(vae_loss_tensor)
        loss = vae_loss + classifier_loss_scaled + fool_loss_scaled + spatial_loss

        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_s=kl_divergence_s,
        )

        extra_metrics = {
            'vae_loss': vae_loss,
            'classifier_loss_raw': classifier_loss_raw.mean(),
            'classifier_loss': classifier_loss_scaled,
            'classifier_accuracy': classifier_accuracy,
            'fool_loss_raw': fool_loss_raw.mean(),
            'fool_loss': fool_loss_scaled,
            'fool_accuracy': discriminator_accuracy,
            'spatial_loss_raw': spatial_loss_raw,
            'spatial_loss': spatial_loss,
            's_domain_accuracy': s_domain_accuracy,
        }

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> torch.Tensor:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale samples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        if self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
        else:
            dist = NegativeBinomial(mu=px_rate, theta=px_r)

        if n_samples > 1:
            exprs = dist.sample().permute([1, 2, 0])
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors: dict, n_mc_samples: int):
        """
        Marginal ll approximation via Monte Carlo Importance Sampling.

        Parameters
        ----------
        tensors
            Input tensors from data loader (graph-aware format)
        n_mc_samples
            Number of Monte Carlo samples for approximation
        """
        node_batch = tensors['node_batch']
        sample_batch = node_batch['X']
        batch_index = node_batch['batch_label']
        batch_size = node_batch['batch_size']

        # Slice to seed nodes
        sample_batch = sample_batch[:batch_size]
        batch_index = batch_index[:batch_size]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            inference_outputs, generative_outputs = self.forward(tensors, compute_loss=False)

            # Intrinsic latent
            qz_m = inference_outputs["qzm"]
            qz_v = inference_outputs["qzv"]
            z = inference_outputs["z"]
            # Spatial latent
            qs_m = inference_outputs["qsm"]
            qs_v = inference_outputs["qsv"]
            s = inference_outputs["s"]
            # Library latent
            ql_m = inference_outputs["qlm"]
            ql_v = inference_outputs["qlv"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            px_rate = generative_outputs["px_rate"]
            px_r = generative_outputs["px_r"]
            px_dropout = generative_outputs["px_dropout"]

            if self.gene_likelihood == "zinb":
                reconst_loss = (
                    -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                    .log_prob(sample_batch)
                    .sum(dim=-1)
                )
            elif self.gene_likelihood == "nb":
                reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(sample_batch).sum(dim=-1)

            # Log-probabilities
            p_z = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
            p_s = Normal(torch.zeros_like(qs_m), torch.ones_like(qs_v)).log_prob(s).sum(dim=-1)
            p_x_zsl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_s_x = Normal(qs_m, qs_v.sqrt()).log_prob(s).sum(dim=-1)

            # Library size terms (only when using latent library)
            if not self.use_observed_lib_size:
                n_batch = self.library_log_means.shape[1]
                local_library_log_means = F.linear(
                    F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
                )
                local_library_log_vars = F.linear(
                    F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
                )
                p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
                q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)
                to_sum[:, i] = p_z + p_s + p_l + p_x_zsl - q_z_x - q_s_x - q_l_x
            else:
                to_sum[:, i] = p_z + p_s + p_x_zsl - q_z_x - q_s_x

        # per-cell marginal log-likelihood (numerically stable log-sum-exp estimator)
        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)

        return float(batch_log_lkl.mean().item())
