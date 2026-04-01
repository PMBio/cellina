"""Training plan for Cellina with optional adversarial domain discrimination."""

import logging
import numpy as np

import torch
from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ._constants import DOMAINS_KEY

logger = logging.getLogger(__name__)


class CellinaAdversarialTrainingPlan(TrainingPlan):
    """
    Training plan for Cellina with adversarial domain forgetting.

    Implements alternating two-step training:
    1. Train domain discriminator to predict domains (frozen VAE)
    2. Train VAE to fool discriminator (frozen discriminator)

    This training plan should only be used when module.discriminator_lambda > 0.

    Parameters
    ----------
    module
        CellinaModule instance with discriminator_lambda > 0
    normalize_losses
        Whether to normalize classifier/fool/edge losses relative to VAE loss.
        Scales are computed once from epoch-0 warmup statistics.
    **kwargs
        Other arguments passed to base TrainingPlan
    """

    def __init__(
        self,
        module,
        normalize_losses: bool = False,
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)

        # Always use manual optimization for two-step training
        self.automatic_optimization = False

        # Warmup collection: collect first-epoch losses to compute fixed normalization scales
        self._warmup_stats = {"scvi": [], "fool": [], "clf": [], "supcon": []}
        self._warmup_done = False
        self._normalize_losses = normalize_losses

        # Fixed normalization scales (computed once from epoch-0 warmup stats)
        self._scale_clf    = 1.0
        self._scale_fool   = 1.0
        self._scale_supcon = 1.0

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opts_list = opts if isinstance(opts, list) else [opts]
        opt_vae = opts_list[0]
        opt_discriminator = opts_list[1] if len(opts_list) > 1 else None

        kappa = self.module.discriminator_lambda

        # ---------------------- WARMUP COLLECTION ----------------------
        # Epoch 0: ALWAYS no-grad (graph batches include neighbors → large memory footprint)
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            with torch.no_grad():
                inference_outputs, generative_outputs, scvi_loss = self.forward(
                    batch,
                    loss_kwargs={
                        "kl_weight": self.kl_weight,
                        "discriminator_lambda": kappa,
                    }
                )
                if self._normalize_losses:
                    scvi_val   = float(scvi_loss.loss.detach().cpu().item())
                    fool_val   = abs(float(scvi_loss.extra_metrics.get("fool_loss_raw", 0.0)))
                    clf_val    = abs(float(scvi_loss.extra_metrics.get("classifier_loss_raw", 0.0)))
                    supcon_val = abs(float(scvi_loss.extra_metrics.get("supcon_loss_raw", 0.0)))

                    self._warmup_stats["scvi"].append(scvi_val)
                    self._warmup_stats["fool"].append(fool_val)
                    self._warmup_stats["clf"].append(clf_val)
                    self._warmup_stats["supcon"].append(supcon_val)

                return {"loss": scvi_loss.loss}

        # ---------- END OF WARMUP → COMPUTE FIXED SCALES ----------
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) > 0:
            if self._normalize_losses:
                vae_mean = np.mean(self._warmup_stats["scvi"])
                self._scale_clf    = vae_mean / (np.mean(self._warmup_stats["clf"])    + 1e-8)
                self._scale_fool   = vae_mean / (np.mean(self._warmup_stats["fool"])   + 1e-8)
                self._scale_supcon = vae_mean / (np.mean(self._warmup_stats["supcon"]) + 1e-8)
                self._warmup_stats.clear()  # Free memory
            self._warmup_done = True

        # ------------------ STEP 1: Train Discriminator ------------------
        with torch.no_grad():
            inference_inputs = self.module._get_inference_input(batch)
            inference_outputs = self.module.inference(**inference_inputs)
            z_detach = inference_outputs["z"].detach()

        # Handle both graph-aware and standard batch formats
        if 'node_batch' in batch:
            domain_labels = batch['node_batch'].get(DOMAINS_KEY, torch.zeros(1, dtype=torch.long)).reshape(-1).long()
            batch_size = batch['node_batch']['batch_size']
            domain_labels = domain_labels[:batch_size]
        else:
            domain_labels = batch.get(DOMAINS_KEY, torch.zeros(1, dtype=torch.long)).reshape(-1).long()

        disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
            classifier=self.module.domain_discriminator,
            weight=kappa,
            inference_outputs={"z": z_detach},
            labels=domain_labels,
            reconst_loss_shape=None,
            metric_name="discriminator",
        )
        disc_loss = disc_loss_tensor.mean() * kappa  # apply kappa (not applied inside _compute_classifier_metrics)

        if disc_loss.requires_grad:
            opt_discriminator.zero_grad()
            self.manual_backward(disc_loss)
            opt_discriminator.step()

        # Log discriminator metrics
        self.log("discriminator_loss_train", disc_loss, on_step=False, on_epoch=True)
        self.log("discriminator_accuracy_train", disc_accuracy, on_step=False, on_epoch=True)

        # ------------------ STEP 2: Train VAE + Fool Discriminator ------------------
        for p in self.module.domain_discriminator.parameters():
            p.requires_grad = False

        # Fixed normalization scales (computed once from epoch-0 warmup stats)
        scale_clf    = self._scale_clf    if self._normalize_losses else 1.0
        scale_fool   = self._scale_fool   if self._normalize_losses else 1.0
        scale_supcon = self._scale_supcon if self._normalize_losses else 1.0

        # Single forward pass with scales
        inference_outputs, generative_outputs, scvi_loss = self.forward(
            batch,
            loss_kwargs={
                "kl_weight": self.kl_weight,
                "discriminator_lambda": kappa,
                "classifier_scale": scale_clf,
                "discriminator_scale": scale_fool,
                "supcon_scale": scale_supcon,
            }
        )

        # Extract losses (tensors for gradient flow)
        vae_loss    = scvi_loss.extra_metrics["vae_loss"]       # pure VAE
        clf_loss    = scvi_loss.extra_metrics["classifier_loss"] # scaled
        fool_loss   = scvi_loss.extra_metrics["fool_loss"]       # scaled
        supcon_loss = scvi_loss.extra_metrics.get("supcon_loss", torch.tensor(0.0))  # already scaled

        # Total training loss
        total_train_loss = vae_loss + clf_loss + fool_loss + supcon_loss

        # Backward pass
        opt_vae.zero_grad()
        self.manual_backward(total_train_loss)
        opt_vae.step()

        for p in self.module.domain_discriminator.parameters():
            p.requires_grad = True

        # ------------------ LOGGING ------------------
        self.log("train_loss", total_train_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self._normalize_losses:
            self.log("scale_clf_train",    scale_clf,    on_step=False, on_epoch=True)
            self.log("scale_fool_train",   scale_fool,   on_step=False, on_epoch=True)
            self.log("scale_supcon_train", scale_supcon, on_step=False, on_epoch=True)

        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")

        return {"loss": total_train_loss}

    def validation_step(self, batch, batch_idx):
        scale_clf    = self._scale_clf    if self._normalize_losses else 1.0
        scale_fool   = self._scale_fool   if self._normalize_losses else 1.0
        scale_supcon = self._scale_supcon if self._normalize_losses else 1.0

        with torch.no_grad():
            kappa = self.module.discriminator_lambda
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch,
                loss_kwargs={
                    "kl_weight": self.kl_weight,
                    "discriminator_lambda": kappa,
                    "classifier_scale": scale_clf,
                    "discriminator_scale": scale_fool,
                    "supcon_scale": scale_supcon,
                }
            )

            # Manually compute discriminator training loss (positive weight) for monitoring
            if 'node_batch' in batch:
                domain_labels = batch['node_batch'].get(DOMAINS_KEY, torch.zeros(1, dtype=torch.long)).reshape(-1).long()
                batch_size = batch['node_batch']['batch_size']
                domain_labels = domain_labels[:batch_size]
            else:
                domain_labels = batch.get(DOMAINS_KEY, torch.zeros(1, dtype=torch.long)).reshape(-1).long()

            disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
                classifier=self.module.domain_discriminator,
                weight=kappa,
                inference_outputs=inference_outputs,
                labels=domain_labels,
                reconst_loss_shape=None,
                metric_name="discriminator"
            )
            scvi_loss.extra_metrics["discriminator_loss"] = disc_loss_tensor.mean() * kappa
            scvi_loss.extra_metrics["discriminator_accuracy"] = disc_accuracy

        # Mirror train_loss: vae + supcon (in scvi_loss.loss) + clf + fool
        clf_loss_val  = scvi_loss.extra_metrics["classifier_loss"]
        fool_loss_val = scvi_loss.extra_metrics["fool_loss"]
        total_val_loss = scvi_loss.loss + clf_loss_val + fool_loss_val
        self.log("validation_loss", total_val_loss, on_step=False, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")

        return total_val_loss

    def configure_optimizers(self):
        # VAE optimizer (all parameters except discriminator)
        params_vae = [
            p for name, p in self.module.named_parameters()
            if p.requires_grad and "domain_discriminator" not in name
        ]

        optimizer_vae = self.get_optimizer_creator()(params_vae)
        config_vae = {"optimizer": optimizer_vae}

        # Add learning rate scheduler if requested
        if self.reduce_lr_on_plateau:
            scheduler_vae = ReduceLROnPlateau(
                optimizer_vae,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )
            config_vae["lr_scheduler"] = {
                "scheduler": scheduler_vae,
                "monitor": self.lr_scheduler_metric,
            }

        # Discriminator optimizer
        params_discriminator = filter(
            lambda p: p.requires_grad,
            self.module.domain_discriminator.parameters()
        )
        optimizer_discriminator = torch.optim.Adam(
            params_discriminator,
            lr=1e-3,
            eps=0.01,
            weight_decay=self.weight_decay,
        )

        opts = [config_vae["optimizer"], optimizer_discriminator]

        if "lr_scheduler" in config_vae:
            return opts, [config_vae["lr_scheduler"]]
        else:
            return opts

    def on_train_epoch_end(self):
        """Log fixed normalization scales after warmup epoch."""
        if self._warmup_done and getattr(self, "current_epoch", 0) == 1:
            self.log("scale_clf_fixed",    self._scale_clf,    on_step=False, on_epoch=True)
            self.log("scale_fool_fixed",   self._scale_fool,   on_step=False, on_epoch=True)
            self.log("scale_supcon_fixed", self._scale_supcon, on_step=False, on_epoch=True)
