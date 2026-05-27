"""Training plan for Cellina with optional adversarial domain discrimination."""

import logging
import numpy as np

import torch
from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ._cellina_gcn_module import CellinaGCNModule
from ._constants import DOMAINS_KEY

logger = logging.getLogger(__name__)


class CellinaAdversarialTrainingPlan(TrainingPlan):
    """
    Training plan for Cellina with adversarial domain forgetting.

    Implements alternating two-step training:
    1. Train domain discriminator to predict domains (frozen VAE)
    2. Train VAE to fool discriminator (frozen discriminator)

    Compatible with both CellinaModule (original MLP spatial encoder) and
    CellinaGCNModule (GCN spatial encoder). The plan auto-detects the module
    type and passes the appropriate loss kwargs.

    Parameters
    ----------
    module
        CellinaModule or CellinaGCNModule instance.
    normalize_losses
        Whether to normalize classifier/fool/spatial losses relative to VAE loss.
        Scales are computed once from epoch-0 warmup statistics.
    **kwargs
        Other arguments passed to base TrainingPlan.
    """

    def __init__(
        self,
        module,
        normalize_losses: bool = False,
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)

        self.automatic_optimization = False

        self._warmup_stats = {"scvi": [], "fool": [], "clf": [], "spatial": []}
        self._warmup_done = False
        self._normalize_losses = normalize_losses

        # Fixed normalization scales (computed once from epoch-0 warmup stats)
        self._scale_clf               = 1.0
        self._scale_fool              = 1.0
        self._scale_spatial           = 1.0
        # Alias kept for backward compat with Cellina tests
        self._scale_domain_classifier = 1.0

        self._is_graph_module = isinstance(module, CellinaGCNModule)

    def _get_domain_labels(self, batch):
        """Extract domain labels from either graph-aware or standard batch."""
        if 'node_batch' in batch:
            domain_labels = batch['node_batch'].get(
                DOMAINS_KEY, torch.zeros(1, dtype=torch.long)
            ).reshape(-1).long()
            batch_size = batch['node_batch']['batch_size']
            return domain_labels[:batch_size]
        else:
            return batch[DOMAINS_KEY].reshape(-1).long()

    def _build_loss_kwargs(self, kl_weight, kappa, scale_clf, scale_fool, scale_spatial):
        """Build loss_kwargs appropriate for the module type."""
        base = {
            "kl_weight": kl_weight,
            "classifier_scale": scale_clf,
            "discriminator_scale": scale_fool,
        }
        if self._is_graph_module:
            base["discriminator_lambda"] = kappa
            base["spatial_scale"] = scale_spatial
        else:
            base["domain_classifier_scale"] = scale_spatial
        return base

    def _get_secondary_loss(self, extra_metrics):
        """Read the secondary spatial/domain-classifier loss from extra_metrics."""
        spatial = extra_metrics.get("spatial_loss")
        if spatial is not None:
            return spatial
        return extra_metrics.get("domain_classifier_loss", torch.tensor(0.0))

    @staticmethod
    def _batch_size(batch) -> int:
        """Return the number of seed nodes (graph) or cells (MLP) in the batch."""
        if 'node_batch' in batch:
            return int(batch['node_batch']['batch_size'])
        # Standard scVI batch: first tensor in the dict gives the cell count.
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        return 1

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opts_list = opts if isinstance(opts, list) else [opts]
        opt_vae = opts_list[0]
        opt_discriminator = opts_list[1] if len(opts_list) > 1 else None

        kappa = self.module.discriminator_lambda
        batch_size = self._batch_size(batch)

        # ── WARMUP COLLECTION ────────────────────────────────────────────────
        # Graph modules: always run epoch 0 as no-grad (memory optimization).
        # MLP modules: only run no-grad when normalize_losses=True.
        should_warmup = self._is_graph_module or self._normalize_losses
        if should_warmup and (not self._warmup_done) and getattr(self, "current_epoch", 0) == 0:
            with torch.no_grad():
                inference_outputs, generative_outputs, scvi_loss = self.forward(
                    batch,
                    loss_kwargs=self._build_loss_kwargs(
                        self.kl_weight, kappa, 1.0, 1.0, 1.0
                    ),
                )
                if self._normalize_losses:
                    scvi_val    = float(scvi_loss.extra_metrics.get("vae_loss", scvi_loss.loss).detach().cpu().item())
                    fool_val    = abs(float(scvi_loss.extra_metrics.get("fool_loss_raw", 0.0)))
                    clf_val     = abs(float(scvi_loss.extra_metrics.get("classifier_loss_raw", 0.0)))
                    # Accept either graph (spatial_loss_raw) or original (domain_classifier_loss_raw) key
                    spatial_val = abs(float(scvi_loss.extra_metrics.get(
                        "spatial_loss_raw",
                        scvi_loss.extra_metrics.get("domain_classifier_loss_raw", 0.0),
                    )))

                    self._warmup_stats["scvi"].append(scvi_val)
                    self._warmup_stats["fool"].append(fool_val)
                    self._warmup_stats["clf"].append(clf_val)
                    self._warmup_stats["spatial"].append(spatial_val)

            return {"loss": scvi_loss.loss}

        # ── END OF WARMUP → COMPUTE FIXED SCALES ────────────────────────────
        if should_warmup and (not self._warmup_done) and getattr(self, "current_epoch", 0) > 0:
            if self._normalize_losses:
                vae_mean = np.mean(self._warmup_stats["scvi"])
                self._scale_clf               = vae_mean / (np.mean(self._warmup_stats["clf"])     + 1e-8)
                self._scale_fool              = vae_mean / (np.mean(self._warmup_stats["fool"])    + 1e-8)
                self._scale_spatial           = vae_mean / (np.mean(self._warmup_stats["spatial"]) + 1e-8)
                self._scale_domain_classifier = self._scale_spatial  # keep in sync
                self._warmup_stats.clear()
            self._warmup_done = True

        # ── STEP 1: Train Discriminator ──────────────────────────────────────
        if self.module.domain_discriminator is not None:
            with torch.no_grad():
                inference_inputs = self.module._get_inference_input(batch)
                inference_outputs = self.module.inference(**inference_inputs)
                z_detach = inference_outputs["z"].detach()

            domain_labels = self._get_domain_labels(batch)
            # Pass weight=1.0 so both module types return raw cross-entropy; apply kappa below.
            # (CellinaModule multiplies by weight internally; CellinaGCNModule does not — so
            # using weight=1.0 normalises the two and lets us always do .mean() * kappa.)
            disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
                classifier=self.module.domain_discriminator,
                weight=1.0,
                inference_outputs={"z": z_detach},
                labels=domain_labels,
                reconst_loss_shape=None,
                metric_name="discriminator",
            )
            disc_loss = disc_loss_tensor.mean() * kappa

            if disc_loss.requires_grad:
                opt_discriminator.zero_grad()
                self.manual_backward(disc_loss)
                opt_discriminator.step()

            self.log("discriminator_loss_train", disc_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("discriminator_accuracy_train", disc_accuracy, on_step=False, on_epoch=True, batch_size=batch_size)

        # ── STEP 2: Train VAE + Fool Discriminator ───────────────────────────
        if self.module.domain_discriminator is not None:
            for p in self.module.domain_discriminator.parameters():
                p.requires_grad = False

        scale_clf     = self._scale_clf     if self._normalize_losses else 1.0
        scale_fool    = self._scale_fool    if self._normalize_losses else 1.0
        scale_spatial = self._scale_spatial if self._normalize_losses else 1.0

        inference_outputs, generative_outputs, scvi_loss = self.forward(
            batch,
            loss_kwargs=self._build_loss_kwargs(
                self.kl_weight, kappa, scale_clf, scale_fool, scale_spatial
            ),
        )

        vae_loss     = scvi_loss.extra_metrics.get("vae_loss", scvi_loss.loss)
        clf_loss     = scvi_loss.extra_metrics["classifier_loss"]
        fool_loss    = scvi_loss.extra_metrics["fool_loss"]
        secondary    = self._get_secondary_loss(scvi_loss.extra_metrics)

        total_train_loss = vae_loss + clf_loss + fool_loss + secondary

        opt_vae.zero_grad()
        self.manual_backward(total_train_loss)
        opt_vae.step()

        if self.module.domain_discriminator is not None:
            for p in self.module.domain_discriminator.parameters():
                p.requires_grad = True

        self.log("train_loss", total_train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        if self._normalize_losses:
            self.log("scale_clf_train",     scale_clf,     on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("scale_fool_train",    scale_fool,    on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("scale_spatial_train", scale_spatial, on_step=False, on_epoch=True, batch_size=batch_size)

        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return {"loss": total_train_loss}

    def validation_step(self, batch, batch_idx):
        batch_size = self._batch_size(batch)
        scale_clf     = self._scale_clf     if self._normalize_losses else 1.0
        scale_fool    = self._scale_fool    if self._normalize_losses else 1.0
        scale_spatial = self._scale_spatial if self._normalize_losses else 1.0

        with torch.no_grad():
            kappa = self.module.discriminator_lambda
            inference_outputs, generative_outputs, scvi_loss = self.forward(
                batch,
                loss_kwargs=self._build_loss_kwargs(
                    self.kl_weight, kappa, scale_clf, scale_fool, scale_spatial
                ),
            )

            if self.module.domain_discriminator is not None:
                domain_labels = self._get_domain_labels(batch)
                disc_loss_tensor, disc_accuracy = self.module._compute_classifier_metrics(
                    classifier=self.module.domain_discriminator,
                    weight=1.0,
                    inference_outputs=inference_outputs,
                    labels=domain_labels,
                    reconst_loss_shape=None,
                    metric_name="discriminator",
                )
                scvi_loss.extra_metrics["discriminator_loss"] = disc_loss_tensor.mean() * kappa
                scvi_loss.extra_metrics["discriminator_accuracy"] = disc_accuracy

        clf_loss  = scvi_loss.extra_metrics["classifier_loss"]
        fool_loss = scvi_loss.extra_metrics["fool_loss"]
        secondary = self._get_secondary_loss(scvi_loss.extra_metrics)
        # Use pure VAE loss whether the module returns it in loss (original) or extra_metrics (graph)
        vae_loss = scvi_loss.extra_metrics.get("vae_loss", scvi_loss.loss)
        total_val_loss = vae_loss + clf_loss + fool_loss + secondary
        self.log("validation_loss", total_val_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")
        return total_val_loss

    def configure_optimizers(self):
        params_vae = [
            p for name, p in self.module.named_parameters()
            if p.requires_grad and "domain_discriminator" not in name
        ]
        optimizer_vae = self.get_optimizer_creator()(params_vae)
        config_vae = {"optimizer": optimizer_vae}

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

        opts = [config_vae["optimizer"]]
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
            opts.append(optimizer_discriminator)

        if "lr_scheduler" in config_vae:
            return opts, [config_vae["lr_scheduler"]]
        else:
            return opts

    def on_train_epoch_end(self):
        """Log fixed normalization scales after warmup epoch."""
        if self._warmup_done and getattr(self, "current_epoch", 0) == 1:
            self.log("scale_clf_fixed",     self._scale_clf,     on_step=False, on_epoch=True)
            self.log("scale_fool_fixed",    self._scale_fool,    on_step=False, on_epoch=True)
            self.log("scale_spatial_fixed", self._scale_spatial, on_step=False, on_epoch=True)
