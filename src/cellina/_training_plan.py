"""Training plan for Cellina with optional adversarial domain discrimination."""

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)
        
        self.scale_adversarial_loss = scale_adversarial_loss
        
        # Always use manual optimization for two-step training
        self.automatic_optimization = False

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
        kappa = self._get_kappa()
        
        # Log KL weight and kappa
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=False, on_epoch=True)
        self.log("adversarial_kappa", kappa, on_step=False, on_epoch=True)
        
        # =====================================================================
        # STEP 1: Train Domain Discriminator (Frozen VAE)
        # =====================================================================
        # Get z without gradients to VAE
        with torch.no_grad():
            inference_inputs = self.module._get_inference_input(batch)
            inference_outputs = self.module.inference(**inference_inputs)
            z = inference_outputs["z"].detach()  # Explicitly detach
        
        # Compute discriminator loss (predicting true domains)
        domain_labels = batch["domain_key"].reshape(-1).long()
        disc_logits = self.module.domain_discriminator(z)
        disc_loss = F.cross_entropy(
            disc_logits,
            domain_labels,
            reduction="mean",
        )
        disc_loss = kappa * disc_loss
        
        # Backward only to discriminator
        opt_discriminator.zero_grad()
        self.manual_backward(disc_loss)
        opt_discriminator.step()
        
        # Log discriminator training loss (the actual loss for training discriminator)
        # Note: compute_and_log_metrics below will log discriminator_loss_train=0 from extra_metrics
        # This is the real discriminator training loss
        self.log(
            "discriminator_loss_train",
            disc_loss,
            on_step=False,
            on_epoch=True,
        )
        
        # =====================================================================
        # STEP 2: Train VAE + Fool Discriminator (Frozen Discriminator)
        # =====================================================================
        # Full forward pass through VAE (computes z with gradients)
        inference_inputs = self.module._get_inference_input(batch)
        inference_outputs = self.module.inference(**inference_inputs)
        generative_inputs = self.module._get_generative_input(batch, inference_outputs)
        generative_outputs = self.module.generative(**generative_inputs)
        
        # Compute VAE loss WITHOUT discriminator
        scvi_loss = self.module.loss(
            batch,
            inference_outputs,
            generative_outputs,
            kl_weight=self.kl_weight,
            discriminator_weight=0.0,  # Don't include discriminator in main loss
        )
        
        # Compute adversarial loss separately (gradients to encoder only)
        z = inference_outputs["z"]
        
        # Freeze discriminator parameters
        for param in self.module.domain_discriminator.parameters():
            param.requires_grad = False
        
        # Compute discriminator predictions on z (with gradients to encoder)
        fool_logits = self.module.domain_discriminator(z)
        
        # Loss to fool discriminator (negative cross-entropy)
        fool_loss = F.cross_entropy(
            fool_logits,
            domain_labels,
            reduction="mean",
        )
        fool_loss = -kappa * fool_loss  # Negative to fool
        
        # Unfreeze discriminator for next iteration
        for param in self.module.domain_discriminator.parameters():
            param.requires_grad = True
        
        # Total VAE loss = main loss + adversarial loss
        total_vae_loss = scvi_loss.loss + fool_loss
        
        # Backward only to VAE (discriminator was frozen during fool_loss computation)
        opt_vae.zero_grad()
        self.manual_backward(total_vae_loss)
        opt_vae.step()
        
        # Log metrics
        self.log("train_loss", total_vae_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("adversarial_loss_train", fool_loss, on_step=False, on_epoch=True)
        
        # Remove discriminator metrics from extra_metrics to avoid redundant logging
        # (we log discriminator_train_loss manually above)
        scvi_loss.extra_metrics.pop("discriminator_loss", None)
        scvi_loss.extra_metrics.pop("discriminator_accuracy", None)
        
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        
        # Return dict for logging (Lightning expects this format)
        return {"loss": total_vae_loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step with discriminator metrics.
        """
        # Prepare loss kwargs for validation (only valid parameters)
        loss_kwargs = {"kl_weight": self.kl_weight}
        
        # Forward pass
        inference_outputs, generative_outputs, scvi_loss = self.forward(
            batch,
            loss_kwargs=loss_kwargs
        )
        loss = scvi_loss.loss
        
        # Log standard metrics
        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")
        
        # Compute discriminator metrics
        domain_labels = batch["domain_key"].reshape(-1).long()
        disc_logits = inference_outputs.get("discriminator_logits")
        
        if disc_logits is not None:
            # Compute discriminator loss (weighted for consistency with training)
            kappa = self._get_kappa()
            disc_loss_val = F.cross_entropy(
                disc_logits,
                domain_labels,
                reduction="mean",
            )
            disc_loss_val = kappa * disc_loss_val  # Apply same weighting as training
            
            # Compute discriminator accuracy
            disc_preds = disc_logits.argmax(dim=-1)
            disc_acc = (disc_preds == domain_labels).float().mean()
            
            # Log discriminator metrics (consistent naming with _validation suffix)
            self.log(
                "discriminator_loss_validation",
                disc_loss_val,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "discriminator_accuracy_validation",
                disc_acc,
                on_step=False,
                on_epoch=True,
            )
        
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers.
        
        Returns 2 optimizers if discriminator_lambda > 0:
        - opt1: VAE parameters (encoders, decoder, classifiers)
        - opt2: Domain discriminator parameters only
        """
        # VAE optimizer (all parameters except discriminator)
        if self.discriminator_lambda > 0:
            # Exclude discriminator parameters
            params_vae = [
                p for name, p in self.module.named_parameters()
                if p.requires_grad and "domain_discriminator" not in name
            ]
        else:
            params_vae = filter(lambda p: p.requires_grad, self.module.parameters())
        
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
