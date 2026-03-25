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
    
    This training plan should only be used when module.discriminator_lambda > 0 or module.mmd_lambda > 0.

    Parameters
    ----------
    module
        CellinaModule instance with discriminator_lambda > 0 or mmd_lambda > 0
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

        # Warmup collection: collect first-epoch losses to initialize EMA
        self._warmup_stats = {"scvi": [], "fool": [], "clf": [], 'mmd': []}
        self._warmup_done = False
        self._normalize_losses = normalize_losses

        # EMA tracking for loss normalization
        self._ema = {"vae": 1.0, "clf": 1.0, "fool": 1.0, "mmd": 1.0}
        self._ema_alpha = 0.01

    def _ema_update(self, old, new):
        return old * (1 - self._ema_alpha) + new * self._ema_alpha

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opts_list = opts if isinstance(opts, list) else [opts]
        opt_vae = opts_list[0]
        opt_discriminator = opts_list[1] if len(opts_list) > 1 else None

        kappa = self.module.discriminator_lambda

        # ---------------------- WARMUP COLLECTION ----------------------
        if self._normalize_losses and (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            with torch.no_grad():
                inference_outputs, generative_outputs, scvi_loss = self.forward(
                    batch,
                    loss_kwargs={
                        "kl_weight": self.kl_weight,
                        "discriminator_lambda": kappa if kappa != 0.0 else 0.0,
                    }
                )
                scvi_val = float(scvi_loss.loss.detach().cpu().item())
                fool_val = abs(float(scvi_loss.extra_metrics.get("fool_loss_raw", 0.0)))
                clf_val = abs(float(scvi_loss.extra_metrics.get("classifier_loss_raw", 0.0)))
                mmd_val = abs(float(scvi_loss.extra_metrics.get("mmd_loss_raw", 0.0)))

                self._warmup_stats["scvi"].append(scvi_val)
                self._warmup_stats["fool"].append(fool_val)
                self._warmup_stats["clf"].append(clf_val)
                self._warmup_stats["mmd"].append(mmd_val)

                return {"loss": scvi_loss.loss}

        # ---------- END OF WARMUP → INITIALIZE EMA ----------
        if self._normalize_losses and (not self._warmup_done) and getattr(self, "current_epoch", 0) > 0:
            self._ema["vae"] = np.mean(self._warmup_stats["scvi"])
            self._ema["clf"] = np.mean(self._warmup_stats["clf"])
            self._ema["fool"] = np.mean(self._warmup_stats["fool"])
            self._ema["mmd"] = np.mean(self._warmup_stats["mmd"])
            self._warmup_done = True
            self._warmup_stats.clear()  # Free memory

        # ------------------ STEP 1: Train Discriminator ------------------
        if self.module.domain_discriminator is not None:
            with torch.no_grad():
                inference_inputs = self.module._get_inference_input(batch)
                inference_outputs = self.module.inference(**inference_inputs)
                z_detach = inference_outputs["z"].detach()

            domain_labels = batch[DOMAINS_KEY].reshape(-1).long()
            disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
                classifier=self.module.domain_discriminator,
                weight=kappa,
                inference_outputs={"z": z_detach},
                labels=domain_labels,
                reconst_loss_shape=batch[DOMAINS_KEY].squeeze(dim=1),
                metric_name="discriminator",
            )
            disc_loss = disc_loss_tensor.mean()

            if disc_loss.requires_grad:
                opt_discriminator.zero_grad()
                self.manual_backward(disc_loss)
                opt_discriminator.step()

            # Log discriminator metrics
            self.log("discriminator_loss_train", disc_loss, on_step=False, on_epoch=True)
            self.log("discriminator_accuracy_train", disc_accuracy, on_step=False, on_epoch=True)

        # ------------------ STEP 2: Train VAE + Fool Discriminator ------------------
        if self.module.domain_discriminator is not None:
            for p in self.module.domain_discriminator.parameters():
                p.requires_grad = False

        # EMA-based scales for clf/fool (stable magnitudes); mmd uses per-batch scale below
        scale_clf = scale_fool = 1.0
        if self._normalize_losses:
            scale_clf  = self._ema["vae"] / (self._ema["clf"]  + 1e-8)
            scale_fool = self._ema["vae"] / (self._ema["fool"] + 1e-8)

        # Forward pass (mmd_scale=1.0; per-batch scale applied after)
        inference_outputs, generative_outputs, scvi_loss = self.forward(
            batch,
            loss_kwargs={
                "kl_weight": self.kl_weight,
                "discriminator_lambda": kappa,
                "classifier_scale": scale_clf,
                "discriminator_scale": scale_fool,
                "mmd_scale": 1.0,
            }
        )

        # Extract losses (tensors for gradient flow)
        vae_loss = scvi_loss.loss
        clf_loss_raw = scvi_loss.extra_metrics["classifier_loss_raw"]
        clf_loss     = scvi_loss.extra_metrics["classifier_loss"]
        fool_loss_raw = scvi_loss.extra_metrics["fool_loss_raw"]
        fool_loss    = scvi_loss.extra_metrics["fool_loss"]
        mmd_loss_raw = scvi_loss.extra_metrics["mmd_loss_raw"]

        # Per-batch MMD scale: |mmd_scaled| ≈ ema_vae * mmd_lambda regardless of how large MMD grows
        scale_mmd = (self._ema["vae"] / (abs(mmd_loss_raw.item()) + 1e-8)) if self._normalize_losses else 1.0
        mmd_loss  = mmd_loss_raw * scale_mmd * self.module.mmd_lambda
        scvi_loss.extra_metrics["mmd_loss"] = mmd_loss  # keep logging in sync

        # Total training loss (with gradients)
        total_train_loss = vae_loss + clf_loss + fool_loss + mmd_loss
        #print(f"vae_loss: {vae_loss.item():.4f}, clf_loss: {clf_loss.item():.4f}, fool_loss: {fool_loss.item():.4f}, mmd_loss: {mmd_loss.item():.4f}")

        # Backward pass
        opt_vae.zero_grad()
        self.manual_backward(total_train_loss)
        opt_vae.step()

        if self.module.domain_discriminator is not None:
            for p in self.module.domain_discriminator.parameters():
                p.requires_grad = True

        # ------------------ UPDATE EMA WITH RAW LOSSES ------------------
        # Track raw losses to compute normalization for next iteration
        self._ema["vae"] = self._ema_update(self._ema["vae"], float(vae_loss.item()))
        self._ema["clf"] = self._ema_update(self._ema["clf"], abs(float(clf_loss_raw.item())))
        self._ema["fool"] = self._ema_update(self._ema["fool"], abs(float(fool_loss_raw.item())))
        self._ema["mmd"] = self._ema_update(self._ema["mmd"], abs(float(mmd_loss_raw.item())))

        # ------------------ LOGGING ------------------
        # Log total loss (sum of vae + scaled classifier + scaled fool)
        self.log("train_loss", total_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log normalization scales (not in extra_metrics)
        if self._normalize_losses:
            self.log("scale_classifier_train", scale_clf, on_step=False, on_epoch=True)
            self.log("scale_fool_train", scale_fool, on_step=False, on_epoch=True)
            self.log("scale_mmd_train", scale_mmd, on_step=False, on_epoch=True)

        # Log all metrics from extra_metrics (raw/scaled losses, accuracies, reconstruction, KL)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        
        return {"loss": total_train_loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step with discriminator metrics.
        Uses training EMA scales for consistent evaluation.
        """
        scale_clf = scale_fool = 1.0
        if self._normalize_losses and self._warmup_done:
            scale_clf  = self._ema["vae"] / (self._ema["clf"]  + 1e-8)
            scale_fool = self._ema["vae"] / (self._ema["fool"] + 1e-8)

        with torch.no_grad():
            kappa = self.module.discriminator_lambda + 1e-10
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch,
                loss_kwargs={
                    "kl_weight": self.kl_weight,
                    "discriminator_lambda": kappa,
                    "classifier_scale": scale_clf,
                    "discriminator_scale": scale_fool,
                    "mmd_scale": 1.0,
                }
            )
            mmd_loss_raw = scvi_loss.extra_metrics["mmd_loss_raw"]
            scale_mmd = (self._ema["vae"] / (abs(mmd_loss_raw.item()) + 1e-8)) if (self._normalize_losses and self._warmup_done) else 1.0
            scvi_loss.extra_metrics["mmd_loss"] = mmd_loss_raw * scale_mmd * self.module.mmd_lambda
            
            # Manually compute discriminator training loss (positive weight) for monitoring
            domain_labels = batch[DOMAINS_KEY].reshape(-1).long()
            disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
                classifier=self.module.domain_discriminator,
                weight=kappa,
                inference_outputs=inference_outputs,
                labels=domain_labels,
                reconst_loss_shape=batch[DOMAINS_KEY].squeeze(dim=1),
                metric_name="discriminator"
            )
            scvi_loss.extra_metrics["discriminator_loss"] = disc_loss_tensor.mean()
            scvi_loss.extra_metrics["discriminator_accuracy"] = disc_accuracy

        self.log("validation_loss", scvi_loss.loss, on_step=False, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")
        
        return scvi_loss.loss

    def configure_optimizers(self):
        """
        Configure 2 optimizers for adversarial training:
        - opt1: VAE parameters (encoders, decoder, classifiers)
        - opt2: Domain discriminator parameters only
        """
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
        
        # Discriminator optimizer (only when domain_discriminator exists)
        if self.module.domain_discriminator is not None:
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
        else:
            opts = [config_vae["optimizer"]]

        if "lr_scheduler" in config_vae:
            return opts, [config_vae["lr_scheduler"]]
        else:
            return opts

    def on_train_epoch_end(self):
        """Log EMA values at end of warmup epoch."""
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            # Log initial EMA values for transparency
            self.log("ema_vae_init", self._ema["vae"], on_step=False, on_epoch=True)
            self.log("ema_clf_init", self._ema["clf"], on_step=False, on_epoch=True)
            self.log("ema_fool_init", self._ema["fool"], on_step=False, on_epoch=True)
            self.log("ema_mmd_init", self._ema["mmd"], on_step=False, on_epoch=True)
