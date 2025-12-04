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

TensorDict = Dict[str, torch.Tensor]


class CellinaModule(BaseModuleClass):
    """
    Cellina module with dual encoders (z from counts, s from spatial+z).

    This module implements a dual-encoder variational autoencoder where:
    - z_encoder processes count data to produce latent representation z
    - s_encoder processes spatial features concatenated with z to produce latent representation s
    - decoder reconstructs counts from shifted = concat(z, s)

    Parameters
    ----------
    n_input
        Number of input genes.
    n_spatial_input
        Number of spatial features.
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
        When > 0, requires labels_key to be provided in setup_anndata().
    classifier_kwargs
        Extra keyword args forwarded to :class:`~scvi.module._classifier.Classifier`.
    n_labels
        Number of labels for the optional classifier head. Automatically set from adata
        when labels_key is provided in setup_anndata().
    discriminator_lambda
        Weight for the adversarial domain discriminator loss. Set to 0 (default) to disable.
        When > 0, requires domains_key to be provided in setup_anndata().
    discriminator_kwargs
        Extra keyword args forwarded to domain discriminator Classifier.
    n_domains
        Number of domain labels. Automatically set from adata when domains_key is provided.
    link_prediction_weight
        Weight for the edge prediction loss. Set to 0 (default) to disable link prediction.
        When > 0, requires spatial_connectivities_key to be provided in setup_anndata().
    """

    def __init__(
        self,
        n_input: int,
        n_spatial_input: int,
        library_log_means: torch.Tensor,
        library_log_vars: torch.Tensor,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
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
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        self.classifier_lambda = classifier_lambda
        self.discriminator_lambda = discriminator_lambda
        self.link_prediction_weight = link_prediction_weight
        # this is needed to comply with some requirement of the VAEMixin class
        self.latent_distribution = "normal"

        self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
        self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        # setup the parameters of the generative model
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        # Batch injection setup
        cat_list = [n_batch] if n_batch > 0 else None

        # Z encoder: counts -> z
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

        # S encoder: [spatial_x, z] -> s
        # Set whether to condition spatial encoder on intrinsic latent
        self.condition_on_intrinsic = condition_on_intrinsic
        n_input_s = n_spatial_input + n_latent if condition_on_intrinsic else n_spatial_input
        self.s_encoder = Encoder(
            n_input_s,  # spatial features + z OR spatial features only
            n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
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
            n_latent * 2,  # shifted = concat(z, s)
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

    def _compute_edge_scores(
        self,
        s: torch.Tensor,
        pos_edge_index: torch.Tensor | None,
        neg_edge_index: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute edge scores as cosine similarity between s embeddings.
        
        Parameters
        ----------
        s
            Spatial latent embeddings for all nodes [n_nodes, n_latent]
        pos_edge_index
            Positive edge indices [2, n_pos_edges]
        neg_edge_index
            Negative edge indices [2, n_neg_edges]
            
        Returns
        -------
        Edge scores [n_pos_edges + n_neg_edges]
        """
        # Normalize embeddings for cosine similarity
        s_norm = F.normalize(s, p=2, dim=1)
        
        scores = []
        
        # Positive edges
        if pos_edge_index is not None and pos_edge_index.size(1) > 0:
            s_pos_src = s_norm[pos_edge_index[0]]
            s_pos_tgt = s_norm[pos_edge_index[1]]
            pos_scores = (s_pos_src * s_pos_tgt).sum(dim=1)
            scores.append(pos_scores)
        
        # Negative edges
        if neg_edge_index is not None and neg_edge_index.size(1) > 0:
            s_neg_src = s_norm[neg_edge_index[0]]
            s_neg_tgt = s_norm[neg_edge_index[1]]
            neg_scores = (s_neg_src * s_neg_tgt).sum(dim=1)
            scores.append(neg_scores)
        
        if scores:
            return torch.cat(scores, dim=0)
        else:
            return torch.tensor([], device=s.device)

    def _compute_edge_scores_from_embeddings(
        self,
        s_emb: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute edge prediction scores from node embeddings.
        
        Parameters
        ----------
        s_emb
            Node embeddings (s) from edge prediction subgraph
        edge_label_index
            Edge indices [2, num_edges] with source and target nodes
        edge_mask
            Optional mask to filter edges within same batch
            
        Returns
        -------
        Edge scores (logits) for binary classification
        """
        if edge_label_index is None or edge_label_index.size(1) == 0:
            return torch.tensor([], device=s_emb.device)
        
        # Normalize embeddings for cosine similarity
        s_norm = F.normalize(s_emb, p=2, dim=1)
        
        # Get source and target node embeddings
        src_emb = s_norm[edge_label_index[0]]  # [num_edges, latent_dim]
        tgt_emb = s_norm[edge_label_index[1]]  # [num_edges, latent_dim]
        
        # Compute similarity scores (cosine similarity as dot product of normalized vectors)
        scores = (src_emb * tgt_emb).sum(dim=-1)  # [num_edges]
        
        # Apply mask if provided
        if edge_mask is not None:
            scores = scores[edge_mask]
        
        return scores

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # Handle both graph-aware (dict with 'node_batch') and standard (flat dict) formats
        if 'node_batch' in tensors:
            # Graph-aware format from GraphJointDataSplitter
            node_batch = tensors['node_batch']
            edge_batch = tensors.get('edge_batch')
            
            input_dict = dict(
                # Node batch inputs
                x=node_batch['X'],
                spatial_x=node_batch['spatial_x'],
                batch_index=node_batch['batch_label'],
                edge_index=node_batch['edge_index'],
                batch_size=node_batch['batch_size'],
            )
            
            # Add edge batch inputs if available
            if edge_batch is not None:
                input_dict.update(
                    x_link=edge_batch['X'],
                    spatial_x_link=edge_batch['spatial_x'],
                    batch_index_link=edge_batch['batch_label'],
                    edge_index_link=edge_batch['edge_index'],
                    edge_label_index=edge_batch['edge_label_index'],
                    edge_label=edge_batch['edge_label'],
                    edge_mask=edge_batch['edge_mask'],
                )
            else:
                input_dict.update(
                    x_link=None,
                    spatial_x_link=None,
                    batch_index_link=None,
                    edge_index_link=None,
                    edge_label_index=None,
                    edge_label=None,
                    edge_mask=None,
                )
        else:
            # Standard format - no graph data
            from ._constants import SPATIAL_X_KEY
            input_dict = dict(
                x=tensors[REGISTRY_KEYS.X_KEY],
                spatial_x=tensors[SPATIAL_X_KEY],
                batch_index=tensors[REGISTRY_KEYS.BATCH_KEY],
                edge_index=None,
                batch_size=tensors[REGISTRY_KEYS.X_KEY].shape[0],
                x_link=None,
                spatial_x_link=None,
                batch_index_link=None,
                edge_index_link=None,
                edge_label_index=None,
                edge_label=None,
                edge_mask=None,
            )

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        shifted = inference_outputs["shifted"]
        library = inference_outputs["library"]
        
        # Handle both formats
        if 'node_batch' in tensors:
            batch_index = tensors['node_batch']['batch_label']
            # Slice batch_index to match shifted (which is sliced to batch_size in inference)
            batch_size = tensors['node_batch']['batch_size']
            batch_index = batch_index[:batch_size]
        else:
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        
        input_dict = {
            "shifted": shifted,
            "library": library,
            "batch_index": batch_index,
        }
            
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        spatial_x,
        batch_index,
        edge_index,
        batch_size,
        x_link=None,
        spatial_x_link=None,
        batch_index_link=None,
        edge_index_link=None,
        edge_label_index=None,
        edge_label=None,
        edge_mask=None,
        n_samples=1,
    ):
        """
        High level inference method.

        Runs the inference (encoder) model for both node reconstruction and edge prediction.
        
        Two separate forward passes through s_encoder:
        1. Node forward: uses node batch for reconstruction
        2. Edge forward: uses edge batch for link prediction
        """
        # ======= NODE RECONSTRUCTION =======
        # Log transform input
        x_ = torch.log(1 + x)

        # Encode counts -> z
        qzm, qzv, z = self.z_encoder(x_, batch_index)

        # (Optionally) Concatenate spatial_x and z, then encode -> s
        if self.condition_on_intrinsic:
            spatial_input = torch.cat([spatial_x, z.detach()], dim=-1)
        else:
            spatial_input = spatial_x
        qsm, qsv, s = self.s_encoder(spatial_input, batch_index)

        # Compute shifted = concat(z, s) for reconstruction
        shifted = torch.cat([z, s], dim=-1)

        # Library size
        qlm, qlv, library = self.l_encoder(x_, batch_index)

        # Slice outputs to actual batch size (not including sampled neighbors)
        qzm = qzm[:batch_size, :]
        qzv = qzv[:batch_size, :]
        z_sliced = z[:batch_size, :]
        qsm = qsm[:batch_size, :]
        qsv = qsv[:batch_size, :]
        s_sliced = s[:batch_size, :]
        shifted = shifted[:batch_size, :]
        qlm = qlm[:batch_size, :]
        qlv = qlv[:batch_size, :]
        library = library[:batch_size, :]

        outputs = dict(
            z=z_sliced,
            qzm=qzm,
            qzv=qzv,
            s=s_sliced,
            qsm=qsm,
            qsv=qsv,
            shifted=shifted,
            library=library,
            qlm=qlm,
            qlv=qlv,
        )

        # Cell type classifier (on sliced z)
        if self.classifier is not None:
            outputs["classifier_logits"] = self.classifier(z_sliced)

        # Domain discriminator (on sliced z)
        if self.domain_discriminator is not None:
            outputs["discriminator_logits"] = self.domain_discriminator(z_sliced)

        # ======= EDGE PREDICTION - separate forward on edge subgraph =======
        if self.link_prediction_weight > 0 and x_link is not None and edge_label_index is not None:
            # Encode edge subgraph counts -> z_link
            x_link_ = torch.log(1 + x_link)
            _, _, z_link = self.z_encoder(x_link_, batch_index_link)
            
            # Encode edge subgraph spatial features -> s_link
            if self.condition_on_intrinsic:
                spatial_input_link = torch.cat([spatial_x_link, z_link.detach()], dim=-1)
            else:
                spatial_input_link = spatial_x_link
            _, _, s_link = self.s_encoder(spatial_input_link, batch_index_link)
            
            # Compute edge scores using s_link embeddings
            edge_scores = self._compute_edge_scores_from_embeddings(
                s_link, edge_label_index, edge_mask
            )
            outputs["edge_scores"] = edge_scores
            outputs["edge_label"] = edge_label
            outputs["edge_mask"] = edge_mask

        return outputs

    @auto_move_data
    def generative(self, shifted, library, batch_index):
        """Runs the generative model."""
        # Decode using shifted = concat(z, s)
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
        
        Parameters
        ----------
        classifier
            The classifier network (or None if disabled)
        weight
            Weight for the classifier loss (e.g., classifier_lambda or discriminator_weight)
        inference_outputs
            Outputs from inference containing z and optionally pre-computed logits
        labels
            True labels for classification
        reconst_loss_shape
            Shape reference for zero loss tensor
        metric_name
            Base name for metrics (e.g., 'classifier' or 'discriminator')
        
        Returns
        -------
        Tuple of (loss_tensor, accuracy_scalar)
        """
        if weight != 0.0 and classifier is not None:
            # Get or compute logits
            logits_key = f"{metric_name}_logits"
            logits = inference_outputs.get(logits_key)
            if logits is None:
                logits = classifier(inference_outputs["z"])
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, labels, reduction="none")
            loss = weight * loss
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
            return loss, accuracy
        else:
            # Return zeros when classifier is disabled
            return torch.zeros_like(reconst_loss_shape), 0.0

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        discriminator_lambda: float = 0.0,
        classifier_scale: float = 1.0,
        discriminator_scale: float = 1.0,
    ):
        """
        Loss function.
        
        Parameters
        ----------
        tensors
            Input tensors from data loader
        inference_outputs
            Outputs from inference method
        generative_outputs
            Outputs from generative method
        kl_weight
            Weight for KL divergence terms (warmup)
        discriminator_lambda
            Weight multiplier for discriminator loss. Used to scale discriminator contribution.
            Set to 0 to exclude discriminator from loss computation.
        classifier_scale
            EMA-based normalization scale for classifier loss (default 1.0)
        discriminator_scale
            EMA-based normalization scale for discriminator/fool loss (default 1.0)
        """
        # Handle joint batch format
        if "node_batch" in tensors:
            x = tensors["node_batch"]['X']
            batch_index = tensors["node_batch"]['batch_label']
            batch_size = tensors["node_batch"]['batch_size']
            # Slice to actual batch size (exclude sampled neighbors)
            x = x[:batch_size, :]
            batch_index = batch_index[:batch_size]
            labels = tensors["node_batch"].get(REGISTRY_KEYS.LABELS_KEY, torch.zeros(batch_size, dtype=torch.long, device=x.device)).reshape(-1).long()
            domain_labels = tensors["node_batch"].get(DOMAINS_KEY, torch.zeros(batch_size, dtype=torch.long, device=x.device)).reshape(-1).long()
            # Slice labels/domain_labels if they exist and have full subgraph size
            if REGISTRY_KEYS.LABELS_KEY in tensors["node_batch"]:
                labels = labels[:batch_size]
            if DOMAINS_KEY in tensors["node_batch"]:
                domain_labels = domain_labels[:batch_size]
        else:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            labels = tensors.get(REGISTRY_KEYS.LABELS_KEY, torch.zeros(x.shape[0], dtype=torch.long, device=x.device)).reshape(-1).long()
            domain_labels = tensors.get(DOMAINS_KEY, torch.zeros(x.shape[0], dtype=torch.long, device=x.device)).reshape(-1).long()
            
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
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )
        local_library_log_vars = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )

        kl_divergence_l = kl(
            Normal(qlm, torch.sqrt(qlv)),
            Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
        ).sum(dim=1)

        # Total KL for warmup (z and s)
        kl_local_for_warmup = kl_divergence_z + kl_divergence_s
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Cell type classifier
        classifier_loss, classifier_accuracy = self._compute_classifier_metrics(
            classifier=self.classifier,
            weight=self.classifier_lambda,
            inference_outputs=inference_outputs,
            labels=labels,
            reconst_loss_shape=reconst_loss,
            metric_name="classifier",
        )
        classifier_loss_scaled = classifier_loss * classifier_scale

        # Domain discriminator (fool loss - always negative for adversarial training)
        fool_loss, discriminator_accuracy = self._compute_classifier_metrics(
            classifier=self.domain_discriminator,
            weight=discriminator_lambda,
            inference_outputs=inference_outputs,
            labels=domain_labels,
            reconst_loss_shape=reconst_loss,
            metric_name="discriminator",
        )
        fool_loss_scaled = fool_loss * discriminator_scale

        # Edge prediction loss
        edge_loss = torch.tensor(0.0, device=reconst_loss.device)
        if self.link_prediction_weight > 0 and "edge_scores" in inference_outputs:
            edge_scores = inference_outputs["edge_scores"]
            edge_label = inference_outputs.get("edge_label")
            edge_mask = inference_outputs.get("edge_mask")
            
            if edge_label is not None and len(edge_scores) > 0:
                # Apply mask to labels if needed
                if edge_mask is not None:
                    edge_label_filtered = edge_label[edge_mask]
                else:
                    edge_label_filtered = edge_label
                
                # Binary cross-entropy loss for edge prediction
                edge_loss = F.binary_cross_entropy_with_logits(
                    edge_scores,
                    edge_label_filtered.float(),
                    reduction="mean"
                ) * self.link_prediction_weight

        # Total loss
        vae_loss_tensor = reconst_loss + weighted_kl_local
        vae_loss = torch.mean(vae_loss_tensor)
        loss = vae_loss + edge_loss

        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_s=kl_divergence_s,
        )
        
        extra_metrics = {
            'vae_loss': vae_loss,
            'classifier_loss_raw': classifier_loss.mean(),
            'classifier_loss': classifier_loss_scaled.mean(),
            'classifier_accuracy': classifier_accuracy,
            'fool_loss_raw': fool_loss.mean(),
            'fool_loss': fool_loss_scaled.mean(),
            'fool_accuracy': discriminator_accuracy,
            'edge_prediction_loss': edge_loss,
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
    def marginal_ll(self, tensors: TensorDict, n_mc_samples: int):
        """
        Marginal ll approximation via Monte Carlo Importance Sampling.
        Used for model evaluation.
        Parameters
        ----------
        tensors
            Input tensors from data loader
        n_mc_samples
            Number of Monte Carlo samples for approximation
        """
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
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
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
            )
            local_library_log_vars = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
            )
            p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
            p_z = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
            p_s = Normal(torch.zeros_like(qs_m), torch.ones_like(qs_v)).log_prob(s).sum(dim=-1)
            p_x_zsl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_s_x = Normal(qs_m, qs_v.sqrt()).log_prob(s).sum(dim=-1)
            q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

            to_sum[:, i] = p_z + p_s + p_l + p_x_zsl - q_z_x - q_s_x - q_l_x

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl
