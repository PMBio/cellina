"""Training plan for Cellina with optional adversarial domain discrimination."""

import logging
from typing import Literal
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
    scale_adversarial_loss
        How to scale adversarial loss over training:
        - "auto": Use inverse of KL warmup (1 - kl_weight)
        - float: Fixed weight
    **kwargs
        Other arguments passed to base TrainingPlan
    """

    def __init__(
        self,
        module,
        scale_adversarial_loss: float | Literal["auto"] = "auto",
        normalize_losses: bool = False,
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)
        
        self.scale_adversarial_loss = scale_adversarial_loss
        
        # Always use manual optimization for two-step training
        self.automatic_optimization = False

        # Warmup collection: collect first-epoch losses to initialize EMA
        self._warmup_stats = {"scvi": [], "fool": [], "clf": [], "edge": []}
        self._warmup_done = False
        self._normalize_losses = normalize_losses

        # EMA tracking for loss normalization
        self._ema = {"vae": 1.0, "clf": 1.0, "fool": 1.0, "edge": 1.0}
        self._ema_alpha = 0.01

    def _ema_update(self, old, new):
        return old * (1 - self._ema_alpha) + new * self._ema_alpha

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        if not isinstance(opts, list):
            raise ValueError("Expected 2 optimizers for adversarial training")
        opt_vae, opt_discriminator = opts

        kappa = self.module.discriminator_lambda

        # ---------------------- WARMUP COLLECTION ----------------------
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            with torch.no_grad():
                inference_outputs, generative_outputs, scvi_loss = self.forward(
                    batch,
                    loss_kwargs={
                        "kl_weight": self.kl_weight,
                        "discriminator_lambda": -kappa if kappa != 0.0 else 0.0,
                    }
                )
                scvi_val = float(scvi_loss.loss.detach().cpu().item())
                fool_val = abs(float(scvi_loss.extra_metrics.get("fool_loss", 0.0)))
                clf_val = abs(float(scvi_loss.extra_metrics.get("classifier_loss", 0.0)))
                edge_val = abs(float(scvi_loss.extra_metrics.get("edge_prediction_loss", 0.0)))

                self._warmup_stats["scvi"].append(scvi_val)
                self._warmup_stats["fool"].append(fool_val)
                self._warmup_stats["clf"].append(clf_val)
                self._warmup_stats["edge"].append(edge_val)

                return {"loss": scvi_loss.loss}

        # ---------- END OF WARMUP → INITIALIZE EMA ----------
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) > 0:
            self._ema["vae"] = np.mean(self._warmup_stats["scvi"])
            self._ema["clf"] = np.mean(self._warmup_stats["clf"])
            self._ema["fool"] = np.mean(self._warmup_stats["fool"])
            self._ema["edge"] = np.mean(self._warmup_stats["edge"])
            self._warmup_done = True
            self._warmup_stats.clear()  # Free memory

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
        disc_loss = disc_loss_tensor.mean()

        opt_discriminator.zero_grad()
        self.manual_backward(disc_loss)
        opt_discriminator.step()

        # Log discriminator metrics 
        self.log("discriminator_loss_train", disc_loss, on_step=False, on_epoch=True) 
        self.log("discriminator_accuracy_train", disc_accuracy, on_step=False, on_epoch=True)

        # ------------------ STEP 2: Train VAE + Fool Discriminator ------------------
        for p in self.module.domain_discriminator.parameters():
            p.requires_grad = False

        # Compute normalization scales from EMA
        if self._normalize_losses:
            target = self._ema["vae"]
            scale_clf = target / (self._ema["clf"] + 1e-8)
            scale_fool = target / (self._ema["fool"] + 1e-8)
            scale_edge = target / (self._ema["edge"] + 1e-8)
        else:
            scale_clf = 1.0
            scale_fool = 1.0
            scale_edge = 1.0

        # Single forward pass with scales
        inference_outputs, generative_outputs, scvi_loss = self.forward(
            batch,
            loss_kwargs={
                "kl_weight": self.kl_weight,
                "discriminator_lambda": -kappa,  # Negative for adversarial (fool) loss
                "classifier_scale": scale_clf,
                "discriminator_scale": scale_fool,
            }
        )

        # Extract losses (tensors for gradient flow)
        vae_loss = scvi_loss.extra_metrics["vae_loss"]
        clf_loss_raw = scvi_loss.extra_metrics["classifier_loss_raw"]
        clf_loss = scvi_loss.extra_metrics["classifier_loss"]  # scaled is default
        fool_loss_raw = scvi_loss.extra_metrics["fool_loss_raw"]
        fool_loss = scvi_loss.extra_metrics["fool_loss"]  # scaled is default
        edge_loss_raw = scvi_loss.extra_metrics.get("edge_prediction_loss", torch.tensor(0.0))
        edge_loss = edge_loss_raw * scale_edge

        # Total training loss (with gradients)
        total_train_loss = vae_loss + clf_loss + fool_loss + edge_loss

        # Backward pass
        opt_vae.zero_grad()
        self.manual_backward(total_train_loss)
        opt_vae.step()

        for p in self.module.domain_discriminator.parameters():
            p.requires_grad = True

        # ------------------ UPDATE EMA WITH RAW LOSSES ------------------
        # Track raw losses to compute normalization for next iteration
        self._ema["vae"] = self._ema_update(self._ema["vae"], float(vae_loss.item()))
        self._ema["clf"] = self._ema_update(self._ema["clf"], abs(float(clf_loss_raw.item())))
        self._ema["fool"] = self._ema_update(self._ema["fool"], abs(float(fool_loss_raw.item())))
        self._ema["edge"] = self._ema_update(self._ema["edge"], abs(float(edge_loss_raw.item())))

        # ------------------ LOGGING ------------------
        # Log total loss (sum of vae + scaled classifier + scaled fool)
        self.log("train_loss", total_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log normalization scales (not in extra_metrics)
        if self._normalize_losses:
            self.log("scale_classifier_train", scale_clf, on_step=False, on_epoch=True)
            self.log("scale_fool_train", scale_fool, on_step=False, on_epoch=True)
            self.log("scale_edge_train", scale_edge, on_step=False, on_epoch=True)
        
        # Log all metrics from extra_metrics (raw/scaled losses, accuracies, reconstruction, KL)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        
        return {"loss": total_train_loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step with discriminator metrics.
        Uses training EMA scales for consistent evaluation.
        """
        # Compute normalization scales from training EMA (for consistency)
        if self._normalize_losses and self._warmup_done:
            target = self._ema["vae"]
            scale_clf = target / (self._ema["clf"] + 1e-8)
            scale_fool = target / (self._ema["fool"] + 1e-8)
        else:
            scale_clf = 1.0
            scale_fool = 1.0
        
        with torch.no_grad():
            # Forward pass with negative discriminator_lambda for fool loss
            kappa = self.module.discriminator_lambda
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch,
                loss_kwargs={
                    "kl_weight": self.kl_weight,
                    "discriminator_lambda": -kappa,  # Negative for adversarial
                    "classifier_scale": scale_clf,
                    "discriminator_scale": scale_fool,
                }
            )
            
            # Manually compute discriminator training loss (positive weight) for monitoring
            # Handle both graph-aware and standard batch formats
            if 'node_batch' in batch:
                domain_labels = batch['node_batch'].get(DOMAINS_KEY, torch.zeros(1, dtype=torch.long)).reshape(-1).long()
                # Slice to batch_size to match inference outputs (which are already sliced)
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
        
        # Return both optimizers
        opts = [config_vae["optimizer"], optimizer_discriminator]
        
        if "lr_scheduler" in config_vae:
            return opts, [config_vae["lr_scheduler"]]
        else:
            return opts

    def _get_kappa(self) -> float: # NOTE: not used - but probably a better way to anneal
        """Get current adversarial weight (kappa)."""
        if self.scale_adversarial_loss == "auto":
            # Inverse of KL warmup: strong at end of training
            return (1 - self.kl_weight) * self.module.discriminator_lambda

    def on_train_epoch_end(self):
        """Log EMA values at end of warmup epoch."""
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            # Log initial EMA values for transparency
            self.log("ema_vae_init", self._ema["vae"], on_step=False, on_epoch=True)
            self.log("ema_clf_init", self._ema["clf"], on_step=False, on_epoch=True)
            self.log("ema_fool_init", self._ema["fool"], on_step=False, on_epoch=True)
            self.log("ema_edge_init", self._ema["edge"], on_step=False, on_epoch=True)
