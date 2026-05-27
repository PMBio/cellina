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


class CellinaGCNModule(BaseModuleClass):
    """
    CellinaGCN module with dual encoders (z from counts MLP, s from spatial GCN).

    Parameters
    ----------
    n_input
        Number of input genes.
    library_log_means
        1 x n_batch array of means of the log library sizes.
    library_log_vars
        1 x n_batch array of variances of the log library sizes.
    n_batch
        Number of batches.
    n_hidden
        Nodes per hidden layer.
    n_latent
        Latent dimensionality for both z and s.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    gene_likelihood
        ``"zinb"`` or ``"nb"``.
    classifier_lambda
        Weight for cell-type classifier loss. 0 disables.
    n_labels
        Number of cell-type labels.
    discriminator_lambda
        Weight for adversarial domain discriminator. 0 disables.
    n_domains
        Number of domain labels.
    condition_on_intrinsic
        If True, concatenate detached z to GCN input before message passing.
    link_prediction_weight
        Weight for spatial loss on s. 0 disables.
    spatial_loss_type
        ``"supcon"`` (supervised contrastive) or ``"domain_clf"``.
    supcon_temperature
        SupCon temperature parameter.
    use_observed_lib_size
        Must be True (latent library not supported for graph batches).
    convolution_type
        GCN type: ``"gcn"``, ``"gat"``, ``"gin"``, ``"sg"``.
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
        use_observed_lib_size: bool = True,
        use_batch_norm: bool = False,
        convolution_type: str = "gcn",
    ):
        super().__init__()
        if not use_observed_lib_size:
            raise NotImplementedError(
                "CellinaGCNModule only supports use_observed_lib_size=True."
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
        self.latent_distribution = "normal"
        self.use_observed_lib_size = use_observed_lib_size

        self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
        self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        cat_list = [n_batch] if n_batch > 0 else None

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

        self.classifier: Optional[Classifier] = None
        if classifier_lambda > 0:
            classifier_kwargs = dict(classifier_kwargs or {})
            self.classifier = Classifier(
                n_input=n_latent,
                n_labels=n_labels,
                logits=True,
                **classifier_kwargs
            )

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

        self.s_domain_classifier: Optional[Classifier] = None
        if link_prediction_weight > 0 and spatial_loss_type == "domain_clf":
            if n_domains is None or n_domains < 2:
                raise ValueError(
                    "spatial_loss_type='domain_clf' requires n_domains >= 2."
                )
            self.s_domain_classifier = Classifier(
                n_input=n_latent,
                n_labels=n_domains,
                logits=True,
            )

    def _get_inference_input(self, tensors):
        if 'node_batch' not in tensors:
            raise ValueError(
                "CellinaGCNModule requires graph-aware batches with 'node_batch' key."
            )
        node_batch = tensors['node_batch']
        return dict(
            x=node_batch['X'],
            batch_index=node_batch['batch_label'],
            edge_index=node_batch['edge_index'],
            batch_size=node_batch['batch_size'],
            x_spatial=node_batch.get('x_spatial'),
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
        x_spatial=None,
    ):
        x_ = torch.log(1 + x)

        qzm, qzv, z = self.z_encoder(x_, batch_index)

        x_spatial_ = x_spatial if x_spatial is not None else x_

        if self.condition_on_intrinsic:
            spatial_input = torch.cat([x_spatial_, z.detach()], dim=-1)
        else:
            spatial_input = x_spatial_

        qsm, qsv, s, neighbor_means = self.s_encoder(
            spatial_input, edge_index, batch_index,
            batch_size=batch_size,
            return_neighbor_means=(
                self.link_prediction_weight > 0 and self.spatial_loss_type == "supcon"
            ),
        )

        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
            qlm, qlv = None, None
        else:
            qlm, qlv, library = self.l_encoder(x_, batch_index)

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

        if self.classifier is not None:
            outputs["classifier_logits"] = self.classifier(z_sliced)

        if self.domain_discriminator is not None:
            outputs["discriminator_logits"] = self.domain_discriminator(z_sliced)

        if self.s_domain_classifier is not None:
            outputs["s_domain_logits"] = self.s_domain_classifier(s)

        return outputs

    @auto_move_data
    def generative(self, shifted, library, batch_index):
        px_scale, _, px_rate, px_dropout = self.decoder("gene", shifted, library, batch_index)
        px_r = torch.exp(self.px_r)
        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def _compute_classifier_metrics(
        self,
        classifier: Optional[Classifier],
        weight: float,
        inference_outputs: dict,
        labels: torch.Tensor,
        reconst_loss_shape,
        metric_name: str,
    ) -> tuple[torch.Tensor, float]:
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
            if reconst_loss_shape is not None:
                return torch.zeros_like(reconst_loss_shape), 0.0
            else:
                return torch.tensor(0.0), 0.0

    def _compute_supcon_loss(
        self,
        qsm: torch.Tensor,
        neighbor_means: torch.Tensor,
        edge_index: torch.Tensor,
        domains_all: torch.Tensor,
        batch_size: int,
        temperature: float,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        s_all = torch.cat([qsm, neighbor_means], dim=0)
        s_all = F.normalize(s_all, p=2, dim=1)

        src, dst = edge_index[0], edge_index[1]

        loss_total = torch.tensor(0.0, device=qsm.device)
        n_valid = 0

        for i in range(batch_size):
            edge_mask = src == i
            neighbor_idx = dst[edge_mask]
            if len(neighbor_idx) == 0:
                continue

            pos_idx = neighbor_idx

            neighbor_set = torch.zeros(s_all.size(0), dtype=torch.bool, device=qsm.device)
            neighbor_set[neighbor_idx] = True

            neg_mask = (domains_all != domains_all[i]) & ~neighbor_set
            neg_mask[i] = False

            if neg_mask.sum() == 0:
                continue

            sim_i = (s_all[i].unsqueeze(0) * s_all).sum(dim=-1) / temperature
            sim_i[i] = float('-inf')

            denom_mask = neighbor_set | neg_mask
            denom_mask[i] = False

            log_denom = torch.logsumexp(sim_i[denom_mask], dim=0)
            log_pos   = sim_i[pos_idx] - log_denom

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

        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)

        mean = torch.zeros_like(qzm)
        scale = torch.ones_like(qzv)
        kl_divergence_z = kl(Normal(qzm, torch.sqrt(qzv)), Normal(mean, scale)).sum(dim=1)

        mean_s = torch.zeros_like(qsm)
        scale_s = torch.ones_like(qsv)
        kl_divergence_s = kl(Normal(qsm, torch.sqrt(qsv)), Normal(mean_s, scale_s)).sum(dim=1)

        batch_index_full = tensors['node_batch']['batch_label']
        batch_size_full = tensors['node_batch']['batch_size']
        batch_index = batch_index_full[:batch_size_full]
        if not self.use_observed_lib_size:
            local_library_log_means, local_library_log_vars = self._get_local_library_params(batch_index)
            kl_divergence_l = kl(
                Normal(qlm, torch.sqrt(qlv)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local_for_warmup = kl_divergence_z + kl_divergence_s
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        classifier_loss_raw, classifier_accuracy = self._compute_classifier_metrics(
            classifier=self.classifier,
            weight=self.classifier_lambda,
            inference_outputs=inference_outputs,
            labels=labels,
            reconst_loss_shape=reconst_loss,
            metric_name="classifier",
        )
        classifier_loss_scaled = (classifier_loss_raw * classifier_scale * self.classifier_lambda).mean()

        fool_ce, discriminator_accuracy = self._compute_classifier_metrics(
            classifier=self.domain_discriminator,
            weight=discriminator_lambda,
            inference_outputs=inference_outputs,
            labels=domain_labels,
            reconst_loss_shape=reconst_loss,
            metric_name="discriminator",
        )
        fool_loss_raw = -fool_ce
        fool_loss_scaled = (fool_loss_raw * discriminator_scale * discriminator_lambda).mean()

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
                    batch_size=batch_size_full,
                    temperature=self.supcon_temperature,
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
    def sample(self, tensors, n_samples=1, library_size=1) -> torch.Tensor:
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs = self.forward(
            tensors, inference_kwargs=inference_kwargs, compute_loss=False,
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
    def marginal_ll(self, tensors: dict, n_mc_samples: int) -> torch.Tensor:
        node_batch = tensors['node_batch']
        sample_batch = node_batch['X']
        batch_index = node_batch['batch_label']
        batch_size = node_batch['batch_size']

        sample_batch = sample_batch[:batch_size]
        batch_index = batch_index[:batch_size]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples, device=sample_batch.device)

        for i in range(n_mc_samples):
            inference_outputs, generative_outputs = self.forward(tensors, compute_loss=False)

            qz_m = inference_outputs["qzm"]
            qz_v = inference_outputs["qzv"]
            z = inference_outputs["z"]
            qs_m = inference_outputs["qsm"]
            qs_v = inference_outputs["qsv"]
            s = inference_outputs["s"]
            ql_m = inference_outputs["qlm"]
            ql_v = inference_outputs["qlv"]
            library = inference_outputs["library"]

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

            p_z = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
            p_s = Normal(torch.zeros_like(qs_m), torch.ones_like(qs_v)).log_prob(s).sum(dim=-1)
            p_x_zsl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_s_x = Normal(qs_m, qs_v.sqrt()).log_prob(s).sum(dim=-1)

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

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        return batch_log_lkl.cpu()
