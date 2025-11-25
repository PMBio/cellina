"""Training plan for Cellina with optional adversarial domain discrimination."""

import logging
from typing import Literal

import torch
import torch.nn.functional as F
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
        normalize_losses: bool = True,
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)
        
        self.scale_adversarial_loss = scale_adversarial_loss
        
        # Always use manual optimization for two-step training
        self.automatic_optimization = False

        # Warmup collection: collect first-epoch losses without backward to compute
        # normalization constants for scvi, fool and classifier losses.
        self._warmup_stats = {"scvi": [], "fool": [], "clf": []}
        self._warmup_done = False
        self._norm_constants = {"scvi": 1.0, "fool": 1.0, "clf": 1.0}
        self._normalize_losses = normalize_losses

    def training_step(self, batch, batch_idx):
        """
        Two-step adversarial training.
        
        Step 1: Train discriminator to predict domains (frozen VAE)
        Step 2: Train VAE to fool discriminator (frozen discriminator)
        """
        opts = self.optimizers()
        if not isinstance(opts, list):
            raise ValueError("Expected 2 optimizers for adversarial training")
        opt_vae, opt_discriminator = opts
        
        # Compute kappa (adversarial weight)
        # Use module.discriminator_lambda directly here (scaling handled later)
        kappa = self.module.discriminator_lambda
        
        # Log KL weight and kappa
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=False, on_epoch=True)
        self.log("adversarial_kappa", kappa, on_step=False, on_epoch=True)

        # ---------------------- WARMUP COLLECTION ----------------------
        # If we're in the very first epoch and haven't completed warmup, only
        # compute losses (no backward / optimizer steps) and accumulate them.
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            with torch.no_grad():
                # Compute full forward loss (VAE + extra metrics)
                inference_outputs, generative_outputs, scvi_loss = self.forward(
                    batch,
                    loss_kwargs={
                        "kl_weight": self.kl_weight,
                        "discriminator_lambda": kappa,
                    }
                )
                # Extract scalar values (ensure floats)
                scvi_val = float(scvi_loss.loss.detach().cpu().item())
                fool_val = float(scvi_loss.extra_metrics.get("fool_loss", 0.0))
                clf_val = float(scvi_loss.extra_metrics.get("classifier_loss", 0.0))
                # Append to warmup lists
                self._warmup_stats["scvi"].append(scvi_val)
                self._warmup_stats["fool"].append(fool_val)
                self._warmup_stats["clf"].append(clf_val)
                # Log per-step values for monitoring
                self.log("warmup_scvi_step", scvi_val, on_step=False, on_epoch=True)
                self.log("warmup_fool_step", fool_val, on_step=False, on_epoch=True)
                self.log("warmup_clf_step", clf_val, on_step=False, on_epoch=True)
                # Return without performing any backward/optimizer steps
                return {"loss": scvi_loss.loss}

        # ------------------ STANDARD ADVERSARIAL TRAINING ------------------
        # STEP 1: Train Domain Discriminator (Frozen VAE)
        with torch.no_grad():
            inference_inputs = self.module._get_inference_input(batch)
            inference_outputs = self.module.inference(**inference_inputs)
            z_detach = inference_outputs["z"].detach()  # NOTE: overkill but just to be safe
        
        # Compute discriminator loss and accuracy using shared method
        domain_labels = batch[DOMAINS_KEY].reshape(-1).long()
        disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
            classifier=self.module.domain_discriminator,
            weight=kappa,
            inference_outputs={"z": z_detach},
            labels=domain_labels,
            reconst_loss_shape=None,
            metric_name="discriminator",
        )
        disc_loss = disc_loss_tensor.mean()
        
        # Backward only to discriminator (VAE frozen above)
        opt_discriminator.zero_grad()
        self.manual_backward(disc_loss)
        opt_discriminator.step()
        
        # Log discriminator metrics
        self.log("discriminator_loss_train", disc_loss, on_step=False, on_epoch=True)
        self.log("discriminator_accuracy_train", disc_accuracy, on_step=False, on_epoch=True)
        
        # STEP 2: Train VAE + Fool Discriminator (Frozen Discriminator)
        # Freeze discriminator parameters
        for param in self.module.domain_discriminator.parameters():
            param.requires_grad = False
        
        # Compute VAE loss WITHOUT discriminator (discriminator_lambda=-kappa)
        inference_outputs, generative_outputs, scvi_loss = self.forward(
            batch,
            loss_kwargs={
                "kl_weight": self.kl_weight,
                "discriminator_lambda": kappa,  # Exclude discriminator from validation loss
            }
        )
        
        scvi_loss = self.module.loss(
            batch,
            inference_outputs,
            generative_outputs,
            kl_weight=self.kl_weight,
            discriminator_lambda=-kappa,
        )
        # Extract raw terms
        vae_loss = scvi_loss.loss
        fool_loss = scvi_loss.extra_metrics["fool_loss"]
        clf_loss = scvi_loss.extra_metrics["classifier_loss"]
        
        if self._normalize_losses:
            # Normalize each term by warmup constants (avoid division by zero)
            scvi_norm = self._norm_constants.get("scvi", 1.0)
            fool_norm = self._norm_constants.get("fool", 1.0)
            clf_norm = self._norm_constants.get("clf", 1.0)
            
            # Apply normalization and relevant weights
            vae_loss = vae_loss / scvi_norm
            clf_loss = (clf_loss / clf_norm) * self.module.classifier_lambda
            fool_loss = (fool_loss / fool_norm) * self.module.discriminator_lambda
        
        # Total training loss
        total_train_loss = vae_loss + clf_loss + fool_loss
        
        # Backward only to VAE (discriminator was frozen during fool_loss computation)
        opt_vae.zero_grad()
        self.manual_backward(total_train_loss)
        opt_vae.step()
        
        # Unfreeze discriminator for next iteration
        for param in self.module.domain_discriminator.parameters():
            param.requires_grad = True
        
        # Log total and adversarial loss
        self.log("train_loss", total_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("adversarial_loss_train", fool_loss, on_step=False, on_epoch=True)
        
        # Log standard metrics (reconstruction, KL, vae_loss, classifier_loss, classifier_accuracy)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        
        # Return dict for logging (Lightning expects this format)
        return {"loss": total_train_loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step with discriminator metrics.
        """
        # Compute kappa for discriminator weighting
        kappa = self._get_kappa()
        
        with torch.no_grad():
            # Forward pass WITHOUT discriminator in loss (consistent with training)
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch,
                loss_kwargs={
                    "kl_weight": self.kl_weight,
                    "discriminator_lambda": kappa,  # Exclude discriminator from validation loss
                }
            )
            
            # Manually compute discriminator metrics for logging only
            domain_labels = batch[DOMAINS_KEY].reshape(-1).long()
            disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
                classifier=self.module.domain_discriminator,
                weight=-kappa,
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

    def _get_kappa(self) -> float:
        """Get current adversarial weight (kappa)."""
        if self.scale_adversarial_loss == "auto":
            # Inverse of KL warmup: strong at end of training
            return (1 - self.kl_weight) * self.module.discriminator_lambda
        else:
            return self.scale_adversarial_loss * self.module.discriminator_lambda

    def on_train_epoch_end(self):
        # Only compute norms after the first epoch's collection
        if (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            # Compute mean constants, protect against empty lists
            for key in ("scvi", "fool", "clf"):
                vals = self._warmup_stats.get(key, [])
                if len(vals) > 0:
                    mean_val = float(sum(vals) / len(vals))
                    # Use absolute value for norms to avoid zero/negative normalization
                    self._norm_constants[key] = max(abs(mean_val), 1e-8)
                else:
                    self._norm_constants[key] = 1.0
            # Mark warmup as done and clear buffers
            self._warmup_done = True
            self._warmup_stats = {"scvi": [], "fool": [], "clf": []}
            # Log normalization constants
            self.log("norm_scvi", self._norm_constants["scvi"], on_step=False, on_epoch=True)
            self.log("norm_fool", self._norm_constants["fool"], on_step=False, on_epoch=True)
            self.log("norm_clf", self._norm_constants["clf"], on_step=False, on_epoch=True)
