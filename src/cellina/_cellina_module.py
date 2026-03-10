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
        use_observed_lib_size: bool = True,
        supervised: bool = True,
        mmd_lambda: float = 0.0,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        # If the module is constructed in unsupervised mode, force classifier/discriminator weights to 0
        self.supervised = supervised
        if self.supervised:
            mmd_lambda = 0.0
        else:
            classifier_lambda = 0.0
            discriminator_lambda = 0.0
        self.classifier_lambda = classifier_lambda
        self.discriminator_lambda = discriminator_lambda
        self.mmd_lambda = mmd_lambda
        self.use_observed_lib_size = use_observed_lib_size
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
        classifier_kwargs = dict(classifier_kwargs or {})
        self.classifier = Classifier(
            n_input=n_latent, 
            n_labels=n_labels, 
            logits=True, 
            **classifier_kwargs
        )

        # Domain discriminator
        self.domain_discriminator: Optional[Classifier] = None
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

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[REGISTRY_KEYS.X_KEY]
        spatial_x = tensors["spatial_x"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        input_dict = dict(x=x, spatial_x=spatial_x, batch_index=batch_index)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        shifted = inference_outputs["shifted"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        input_dict = {
            "shifted": shifted,
            "library": library,
            "batch_index": batch_index,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, spatial_x, batch_index):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)

        # Library size
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
            qlm, qlv = None, None
        else:
            qlm, qlv, library = self.l_encoder(x_, batch_index)

        # Encode counts -> z
        qzm, qzv, z = self.z_encoder(x_, batch_index)

        # (Optionally) Concatenate spatial_x and z, then encode -> s
        if self.condition_on_intrinsic:
            spatial_input = torch.cat([spatial_x, z.detach()], dim=-1)
        else:
            spatial_input = spatial_x
        qsm, qsv, s = self.s_encoder(spatial_input, batch_index)

        # Compute shifted = concat(z, s)
        shifted = torch.cat([z, s], dim=-1)

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
    def generative(self, shifted, library, batch_index):
        """Runs the generative model."""
        # Decode using shifted = concat(z, s)
        px_scale, _, px_rate, px_dropout = self.decoder("gene", shifted, library, batch_index)
        px_r = torch.exp(self.px_r)

        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def _get_local_library_params(self, batch_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch-specific library log means and log variances.
        
        Parameters
        ----------
        batch_index
            Batch indices for each sample
            
        Returns
        -------
        Tuple of (local_library_log_means, local_library_log_vars)
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )
        local_library_log_vars = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

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

    def _compute_mmd(self, z: torch.Tensor, s: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
        """
        Compute an RBF MMD statistic between two samples z and s. Returns a scalar tensor.
        """
        # Ensure shapes (B, D)
        if z.ndim > 2:
            z = z.view(z.shape[0], -1)
        if s.ndim > 2:
            s = s.view(s.shape[0], -1)

        # Pairwise squared distances
        def pdist_sq(x):
            xx = (x * x).sum(dim=1, keepdim=True)
            return xx + xx.t() - 2.0 * (x @ x.t())

        zz = pdist_sq(z)
        ss = pdist_sq(s)
        zs = torch.cdist(z, s, p=2) ** 2

        if sigma is None:
            # median heuristic on combined distances
            with torch.no_grad():
                median = torch.cat([zz.flatten(), ss.flatten(), zs.flatten()]).median()
                sigma = float(torch.sqrt(median + 1e-8)) if median > 0 else 1.0

        def rbf_from_sqdist(d2, sigma):
            return torch.exp(-d2 / (2 * (sigma ** 2)))

        Kzz = rbf_from_sqdist(zz, sigma)
        Kss = rbf_from_sqdist(ss, sigma)
        Kzs = rbf_from_sqdist(zs, sigma)

        m = z.shape[0]
        # unbiased estimate
        mmd = Kzz.sum() / (m * m) + Kss.sum() / (m * m) - 2.0 * Kzs.sum() / (m * m)
        return mmd

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        discriminator_lambda: float = 0.0,
        classifier_scale: float = 1.0,
        discriminator_scale: float = 1.0,
        mmd_scale: float = 1.0,
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
        x = tensors[REGISTRY_KEYS.X_KEY]
        qzm = inference_outputs["qzm"]
        qzv = inference_outputs["qzv"]
        qsm = inference_outputs["qsm"]
        qsv = inference_outputs["qsv"]
        qlm = inference_outputs["qlm"]
        qlv = inference_outputs["qlv"]
        z = inference_outputs["z"]
        s = inference_outputs["s"]
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
        if not self.use_observed_lib_size:
            local_library_log_means, local_library_log_vars = self._get_local_library_params(batch_index)
            kl_divergence_l = kl(
                Normal(qlm, torch.sqrt(qlv)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            # no library latent KL when using observed library sizes
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        # Total KL for warmup (z and s)
        kl_local_for_warmup = kl_divergence_z + kl_divergence_s
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Cell type classifier
        labels = tensors[REGISTRY_KEYS.LABELS_KEY].reshape(-1).long()
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
        domain_labels = tensors[DOMAINS_KEY].reshape(-1).long()
        fool_loss, discriminator_accuracy = self._compute_classifier_metrics(
            classifier=self.domain_discriminator,
            weight=discriminator_lambda,
            inference_outputs=inference_outputs,
            labels=domain_labels,
            reconst_loss_shape=reconst_loss,
            metric_name="discriminator",
        )
        fool_loss_scaled = fool_loss * discriminator_scale

        # Add MMD regularization if requested
        mmd_loss_raw = -self._compute_mmd(z, s)
        mmd_loss_scaled = mmd_loss_raw * (self.mmd_lambda * mmd_scale)

        # VAE loss (reconstruction + KL only)
        vae_loss_tensor = reconst_loss + weighted_kl_local
        vae_loss = torch.mean(vae_loss_tensor)
        loss = vae_loss

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
            'mmd_loss_raw': mmd_loss_raw.mean(),
            'mmd_loss': mmd_loss_scaled.mean(),
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
            p_z = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
            p_s = Normal(torch.zeros_like(qs_m), torch.ones_like(qs_v)).log_prob(s).sum(dim=-1)
            p_x_zsl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_s_x = Normal(qs_m, qs_v.sqrt()).log_prob(s).sum(dim=-1)
            
            # Library size terms (only when using latent library)
            if not self.use_observed_lib_size:
                local_library_log_means, local_library_log_vars = self._get_local_library_params(batch_index)
                p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
                q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)
                to_sum[:, i] = p_z + p_s + p_l + p_x_zsl - q_z_x - q_s_x - q_l_x
            else:
                to_sum[:, i] = p_z + p_s + p_x_zsl - q_z_x - q_s_x

        # per-cell marginal log-likelihood (numerically stable log-sum-exp estimator)
        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)

        # RETURN per-cell log-likelihoods (1D tensor) instead of a summed scalar
        return batch_log_lkl.cpu()
