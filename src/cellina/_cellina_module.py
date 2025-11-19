from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.module._classifier import Classifier
from scvi.nn import DecoderSCVI, Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

TensorDict = Dict[str, torch.Tensor]


class CellinaModule(BaseModuleClass):
    """
    Cellina module with dual encoders (z from counts, s from spatial+z).

    This module implements a dual-encoder variational autoencoder where:
    - z_encoder processes count data to produce latent representation z
    - s_encoder processes spatial features concatenated with z to produce latent representation s
    - decoder reconstructs counts from shifted = z + s (element-wise sum)

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
        When > 0, requires domain_key to be provided in setup_anndata().
    discriminator_kwargs
        Extra keyword args forwarded to domain discriminator Classifier.
    n_domains
        Number of domain labels. Automatically set from adata when domain_key is provided.
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
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        self.classifier_lambda = classifier_lambda
        self.discriminator_lambda = discriminator_lambda
        # this is needed to comply with some requirement of the VAEMixin class
        self.latent_distribution = "normal"

        self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
        self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        # setup the parameters of the generative model
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        # Z encoder: counts -> z
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # S encoder: [spatial_x, z] -> s
        self.s_encoder = Encoder(
            n_spatial_input + n_latent,  # spatial features + z
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Library encoder
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Decoder: shifted (z + s) -> counts
        self.decoder = DecoderSCVI(
            n_latent,  # shifted = z + s
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
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
                    "Please provide domain_key in setup_anndata()."
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

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[REGISTRY_KEYS.X_KEY]
        spatial_x = tensors["spatial_x"]

        input_dict = dict(x=x, spatial_x=spatial_x)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        shifted = inference_outputs["shifted"]
        library = inference_outputs["library"]

        input_dict = {
            "shifted": shifted,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, spatial_x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)

        # Encode counts -> z
        qzm, qzv, z = self.z_encoder(x_)

        # Concatenate spatial_x and z, then encode -> s
        spatial_z_concat = torch.cat([spatial_x, z], dim=-1)
        qsm, qsv, s = self.s_encoder(spatial_z_concat)

        # Compute shifted = z + s
        shifted = z + s

        # Library size
        qlm, qlv, library = self.l_encoder(x_)

        outputs = dict(
            z=z,
            qzm=qzm,
            qzv=qzv,
            s=s,
            qsm=qsm,
            qsv=qsv,
            shifted=shifted,
            library=library,
            qlm=qlm,
            qlv=qlv,
        )
        
        # Cell type classifier
        if self.classifier is not None:
            outputs["classifier_logits"] = self.classifier(z)
        
        # Domain discriminator
        if self.domain_discriminator is not None:
            outputs["discriminator_logits"] = self.domain_discriminator(z)
        
        return outputs

    @auto_move_data
    def generative(self, shifted, library):
        """Runs the generative model."""
        # Decode using shifted = z + s
        px_scale, _, px_rate, px_dropout = self.decoder("gene", shifted, library)
        px_r = torch.exp(self.px_r)

        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        discriminator_weight: float = 0.0,
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
        discriminator_weight
            Weight for discriminator loss. Sign controls gradient direction:
            - Positive: train discriminator to predict domains (Step 1)
            - Negative: train encoder to fool discriminator (Step 2)
        """
        x = tensors[REGISTRY_KEYS.X_KEY]
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
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
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

        classifier_loss = torch.zeros_like(reconst_loss)
        if self.classifier is not None:
            labels = tensors[REGISTRY_KEYS.LABELS_KEY].reshape(-1).long()
            classifier_logits = inference_outputs.get("classifier_logits")
            if classifier_logits is None:
                classifier_logits = self.classifier(inference_outputs["z"])
            classifier_loss = F.cross_entropy(
                classifier_logits,
                labels,
                reduction="none",
            )
        
        # NOTE: do we also apply warmup to the classifier loss?
        classifier_loss = self.classifier_lambda * classifier_loss # * kl_local_for_warmup

        # Domain discriminator loss
        discriminator_loss = torch.zeros_like(reconst_loss)
        if discriminator_weight != 0.0 and self.domain_discriminator is not None:
            domain_labels = tensors["domain_key"].reshape(-1).long()
            z = inference_outputs["z"]
            discriminator_logits = inference_outputs.get("discriminator_logits")
            if discriminator_logits is None:
                discriminator_logits = self.domain_discriminator(z)
            
            # Standard cross-entropy with true domain labels
            discriminator_loss = F.cross_entropy(
                discriminator_logits,
                domain_labels,
                reduction="none",
            )
            # Sign of discriminator_weight controls gradient direction
            discriminator_loss = discriminator_weight * discriminator_loss

        total_loss = reconst_loss + weighted_kl_local + classifier_loss + discriminator_loss
        loss = torch.mean(total_loss)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_s=kl_divergence_s,
        )
        
        extra_metrics = {}
        extra_metrics['classifier_loss'] = classifier_loss.mean()
        # Note: discriminator_loss sign indicates training phase (pos=train disc, neg=fool disc)
        extra_metrics['discriminator_loss'] = discriminator_loss.mean()
            
        if discriminator_weight != 0.0 and self.domain_discriminator is not None:
            # Compute discriminator accuracy for monitoring
            discriminator_logits = inference_outputs.get("discriminator_logits")
            if discriminator_logits is None:
                discriminator_logits = self.domain_discriminator(z)
            predictions = torch.argmax(discriminator_logits, dim=1)
            accuracy = (predictions == domain_labels).float().mean()
            extra_metrics['discriminator_accuracy'] = accuracy

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
