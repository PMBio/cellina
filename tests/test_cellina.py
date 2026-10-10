import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid
from sklearn.neighbors import NearestNeighbors

from cellina import Cellina
from cellina._spatial_utils import (
    _node_perturbation,
    compute_spatial_features,
    make_counterfactual_adata,
    make_neighbor_perturbation,
    spatial_neighbors,
)


@pytest.fixture
def adata_with_spatial():
    """Create synthetic AnnData with spatial features and connectivity."""
    adata = synthetic_iid()
    rng = np.random.default_rng(0)
    adata.obsm["spatial_x"] = rng.standard_normal((adata.n_obs, 20)).astype(np.float32)
    adata.obs["cell_labels"] = rng.integers(0, 3, size=adata.n_obs).astype(str)
    adata.obs["domain"]      = rng.integers(0, 3, size=adata.n_obs).astype(str)
    adata.obsm["spatial"]    = rng.standard_normal((adata.n_obs, 2)) * 100
    spatial_neighbors(adata, bandwidth=50.0, cutoff=0.1, max_neighbours=10, kernel="gaussian",
                      spatial_key="spatial", inplace=True)
    return adata


def test_cellina_model(adata_with_spatial):
    """Test basic Cellina functionality."""
    n_latent = 5
    
    Cellina.setup_anndata(adata_with_spatial,
                               batch_key="batch",
                               spatial_obsm_key="spatial_x",
                               labels_key="cell_labels",
                               domains_key="domain"
                               )
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    
    # Test architecture
    assert model.module.n_latent == n_latent
    
    # Test training
    model.train(max_epochs=1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_reconstruction_error()
    model.history
    
    # Test __repr__
    print(model)


def test_cellina_s_encoder_architecture(adata_with_spatial):
    """Test that s_encoder receives concatenated [spatial_x, z] as input."""
    n_latent = 5
    n_spatial = adata_with_spatial.obsm["spatial_x"].shape[1]
    
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)

    # Test forward pass produces correct outputs
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    
    inference_inputs = model.module._get_inference_input(batch)
    
    # Verify spatial_x is in inference inputs
    assert "spatial_x" in inference_inputs
    assert inference_inputs["spatial_x"].shape[1] == n_spatial
    
    inference_outputs = model.module.inference(**inference_inputs)
    
    # Verify z and s have correct shapes
    assert inference_outputs["z"].shape[1] == n_latent
    assert inference_outputs["s"].shape[1] == n_latent
    assert all(k in inference_outputs for k in ["z", "s", "qzm", "qzv", "qsm", "qsv"])


def test_cellina_losses(adata_with_spatial):
    """Test that loss includes KL divergence for both z and s, and classifier loss when enabled."""
    n_latent = 5
    
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", labels_key="cell_labels", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=1.0)
    
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    
    inference_inputs = model.module._get_inference_input(batch)
    inference_outputs = model.module.inference(**inference_inputs)
    generative_inputs = model.module._get_generative_input(batch, inference_outputs)
    generative_outputs = model.module.generative(**generative_inputs)
    loss_output = model.module.loss(batch, inference_outputs, generative_outputs)
    
    # Both KL divergences should be present
    assert "kl_divergence_z" in loss_output.kl_local
    assert "kl_divergence_s" in loss_output.kl_local
    assert "kl_divergence_l" in loss_output.kl_local
    
    # Explicit loss components should be in extra_metrics
    assert "vae_loss" in loss_output.extra_metrics
    assert "classifier_loss" in loss_output.extra_metrics
    assert "fool_loss" in loss_output.extra_metrics
    
    # Verify vae_loss is just reconstruction + KL (no classifier)
    vae_loss = loss_output.extra_metrics["vae_loss"]
    assert vae_loss > 0
    
    # Classifier loss should be positive when enabled
    assert loss_output.extra_metrics["classifier_loss"] > 0
    
    # Check we can compute accuracy
    classifier_logits = inference_outputs["classifier_logits"]
    labels = batch[REGISTRY_KEYS.LABELS_KEY].reshape(-1).long()
    predictions = torch.argmax(classifier_logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    assert 0 <= accuracy <= 1


def test_classifier_disabled_by_default(adata_with_spatial):
    """Test that classifier is disabled when classifier_lambda=0."""
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")

    # Should work fine without labels when classifier_lambda=0
    model = Cellina(adata_with_spatial, n_latent=5, classifier_lambda=0.0)
    assert model.module.classifier is None
    assert model.module.classifier_lambda == 0.0


def test_discriminator_enabled_by_default(adata_with_spatial):
    """Test that discriminator is enabled by default (discriminator_lambda=1.0)."""
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")

    model = Cellina(adata_with_spatial, n_latent=5)
    assert model.module.domain_discriminator is not None
    assert model.module.discriminator_lambda == 1.0

    # Explicitly set to 0 disables it
    model2 = Cellina(adata_with_spatial, n_latent=5, discriminator_lambda=0.0)
    assert model2.module.domain_discriminator is None
    assert model2.module.discriminator_lambda == 0.0


def test_discriminator_enabled(adata_with_spatial):
    """Test that discriminator works when discriminator_lambda > 0."""
    Cellina.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        spatial_obsm_key="spatial_x",
        domains_key="domain",
    )

    n_latent = 5
    model = Cellina(adata_with_spatial, n_latent=n_latent, discriminator_lambda=1.0)

    # Check discriminator is initialized
    assert model.module.domain_discriminator is not None
    assert model.module.discriminator_lambda == 1.0

    # Test training with adversarial plan
    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)

    # Check that discriminator metrics are logged
    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"

    # Verify inference outputs include discriminator logits
    model.module.eval()
    model.module.to("cpu")  # Move to CPU for testing
    n_spatial_features = adata_with_spatial.obsm["spatial_x"].shape[1]
    test_batch = {
        "x": torch.abs(torch.randn(10, adata_with_spatial.n_vars)),
        "spatial_x": torch.randn(10, n_spatial_features),
        "batch_index": torch.zeros(10, 1, dtype=torch.long),
    }
    with torch.no_grad():
        outputs = model.module.inference(**test_batch)

    n_domains = adata_with_spatial.obs["domain"].nunique()
    assert "discriminator_logits" in outputs
    assert outputs["discriminator_logits"].shape == (10, n_domains)


def test_cellina_latent_representation(adata_with_spatial):
    """Test latent representation returns correct shapes and uses latent_key."""
    n_latent = 5
    
    Cellina.setup_anndata(adata_with_spatial, 
                               batch_key="batch", spatial_obsm_key="spatial_x",
                               labels_key="cell_labels", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)
    model.train(max_epochs=1, check_val_every_n_epoch=1, train_size=0.5)
    
    # Test separate representations
    latent_z = model.get_latent_representation(latent_key='z')
    latent_s = model.get_latent_representation(latent_key='s')
    latent_shifted = model.get_latent_representation(latent_key='shifted')
    
    assert latent_z.shape == (adata_with_spatial.n_obs, n_latent)
    assert latent_s.shape == (adata_with_spatial.n_obs, n_latent)
    assert latent_shifted.shape == (adata_with_spatial.n_obs, n_latent * 2)  # concat(z, s)
    
    # Default should be shifted (what goes into decoder)
    latent_default = model.get_latent_representation()
    assert latent_default.shape == latent_shifted.shape
    
    # Test error handling
    with pytest.raises(ValueError, match="latent_key must be"):
        model.get_latent_representation(latent_key='invalid')
    
    # Test error handling
    with pytest.raises(ValueError, match="latent_key must be"):
        model.get_latent_representation(latent_key='invalid')


def test_spatial_neighbors(adata_with_spatial):
    """Test spatial_neighbors function."""
    n_obs = adata_with_spatial.n_obs

    # Test basic functionality
    spatial_neighbors(
        adata_with_spatial,
        bandwidth=50.0,
        cutoff=0.1,
        max_neighbours=10,
        kernel='gaussian',
        spatial_key='spatial',
        key_added='spatial',
        inplace=True
    )
    
    # Check that connectivity matrix was added
    assert 'spatial_connectivities' in adata_with_spatial.obsp
    
    # Check that it's a sparse matrix
    assert issparse(adata_with_spatial.obsp['spatial_connectivities'])
    
    # Check shape
    assert adata_with_spatial.obsp['spatial_connectivities'].shape == (n_obs, n_obs)
    
    # Test return value when inplace=False
    conn_matrix = spatial_neighbors(
        adata_with_spatial,
        bandwidth=50.0,
        cutoff=0.1,
        max_neighbours=10,
        kernel='gaussian',
        spatial_key='spatial',
        inplace=False
    )
    
    assert conn_matrix is not None
    assert conn_matrix.shape == (n_obs, n_obs)


def test_spatial_neighbors_test_indices(adata_with_spatial):
    """Test that test_indices cells have zero connectivity (no edges in or out)."""
    test_idx = [0, 1, 2]
    conn = spatial_neighbors(
        adata_with_spatial,
        bandwidth=50.0,
        cutoff=0.1,
        max_neighbours=10,
        kernel='gaussian',
        spatial_key='spatial',
        test_indices=test_idx,
        inplace=False,
    )

    conn_dense = conn.toarray()
    # Test cells must not be selected as neighbors of anyone
    assert conn_dense[:, test_idx].sum() == 0, "test cells appear as neighbors of other cells"
    # Test cells must not have any neighbors themselves
    assert conn_dense[test_idx, :].sum() == 0, "test cells have outgoing edges"
    # Non-test cells must still have neighbors
    non_test = [i for i in range(adata_with_spatial.n_obs) if i not in test_idx]
    assert conn_dense[non_test, :].sum() > 0, "non-test cells lost all connectivity"


def test_compute_spatial_features(adata_with_spatial):
    """Test compute_spatial_features function (pseudobulk mode)."""
    n_obs = adata_with_spatial.n_obs

    # Add cell type labels
    cell_types = ['TypeA', 'TypeB', 'TypeC']
    adata_with_spatial.obs['cell_type'] = np.random.default_rng(1).choice(cell_types, n_obs)

    # Create spatial connectivity matrix
    spatial_neighbors(
        adata_with_spatial,
        bandwidth=50.0,
        cutoff=0.1,
        max_neighbours=10,
        kernel='gaussian',
        spatial_key='spatial',
        inplace=True
    )

    # Test pseudobulk aggregation
    compute_spatial_features(
        adata_with_spatial,
        connectivity_key='spatial_connectivities',
        obsm_key='spatial_pseudobulks',
    )

    # Check that pseudobulks were added
    assert 'spatial_pseudobulks' in adata_with_spatial.obsm

    # Check shape: should be (n_obs, n_genes)
    n_genes = adata_with_spatial.n_vars
    assert adata_with_spatial.obsm['spatial_pseudobulks'].shape == (n_obs, n_genes)
    

def test_marginal_ll(adata_with_spatial):
    """Test get_marginal_ll method and underlying module.marginal_ll."""
    n_latent = 5
    
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)
    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)
    
    # Test basic computation (returns list by default)
    marginal_ll_list = model.get_marginal_ll(n_mc_samples=100, return_mean=False)
    assert isinstance(marginal_ll_list, np.ndarray)
    assert marginal_ll_list.shape[0] == adata_with_spatial.n_obs
    
    # Test mean reduction
    marginal_ll_mean = model.get_marginal_ll(n_mc_samples=100, return_mean=True)
    assert isinstance(marginal_ll_mean, (float, np.floating))
    
    # Test underlying module.marginal_ll
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    model.module.eval()
    with torch.no_grad():
        log_lkl = model.module.marginal_ll(batch, n_mc_samples=100)
    assert isinstance(log_lkl, torch.Tensor) and np.isfinite(log_lkl).all()


def test_s_encoder_spatial_only_input(adata_with_spatial):
    """s_encoder input is the raw spatial feature dimension (plus injected batch covariate)."""
    n_latent = 5
    n_spatial = adata_with_spatial.obsm["spatial_x"].shape[1]

    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")

    model = Cellina(adata_with_spatial, n_latent=n_latent)
    n_batch = model.summary_stats["n_batch"]
    input_dim = model.module.s_encoder.encoder.fc_layers[0][0].in_features
    assert input_dim == n_spatial + n_batch, \
        f"s_encoder input should be n_spatial + n_batch = {n_spatial + n_batch}, got {input_dim}"

    model.train(max_epochs=2, train_size=0.5)

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    model.module.eval()
    with torch.no_grad():
        inference_outputs = model.module.inference(**model.module._get_inference_input(batch))
    assert inference_outputs["z"].shape[1] == n_latent
    assert inference_outputs["s"].shape[1] == n_latent
    assert inference_outputs["shifted"].shape[1] == 2 * n_latent

def test_make_counterfactual_adata(adata_with_spatial):
    """Test make_counterfactual_adata with anchor_donors=False and anchor_donors=True."""
    # Compute spatial_x from gene expression so feature dim matches anchor_donors=False output
    compute_spatial_features(adata_with_spatial, connectivity_key="spatial_connectivities", obsm_key="spatial_x")

    to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    n_obs = adata_with_spatial.n_obs
    indices = np.arange(0, n_obs // 2)
    indices_cf = np.arange(n_obs // 2, n_obs)
    spatial_col = "spatial_x"

    def _cf(**kw):
        return make_counterfactual_adata(
            adata_with_spatial, indices, indices_cf, spatial_col, **kw
        )

    # anchor_donors=False: rebuild via compute_spatial_features
    adata_cf = _cf(anchor_donors=False)
    assert adata_cf.n_obs == len(indices)
    assert adata_cf.n_vars == adata_with_spatial.n_vars
    assert adata_cf.obsm[spatial_col].shape[0] == len(indices)
    np.testing.assert_array_equal(adata_cf.X, adata_with_spatial[indices].X)

    # reproducibility: with n_neighbors the RNG is used; same random_state → same result
    np.testing.assert_array_equal(
        to_dense(_cf(anchor_donors=False, n_neighbors=3, random_state=7).obsm[spatial_col]),
        to_dense(_cf(anchor_donors=False, n_neighbors=3, random_state=7).obsm[spatial_col]),
    )

    # anchor_donors=True: rows sampled from existing obsm; reproducible with same random_state
    adata_cf_pre = _cf(anchor_donors=True, random_state=0)
    np.testing.assert_array_equal(
        to_dense(adata_cf_pre.obsm[spatial_col]),
        to_dense(_cf(anchor_donors=True, random_state=0).obsm[spatial_col]),
    )
    cf_rows = to_dense(adata_with_spatial.obsm[spatial_col][indices_cf])
    result_rows = to_dense(adata_cf_pre.obsm[spatial_col])
    assert np.all(
        np.any(np.all(cf_rows[:, None] == result_rows[None], axis=-1), axis=0)
    ), "anchor_donors=True rows must come from counterfactual obsm rows"

    # anchor_donors=True also writes spatial_x_cf; rows must come from the cf pool
    assert "spatial_x_cf" in adata_cf_pre.obsm, "spatial_x_cf should exist for anchor_donors=True"
    cf_obsm_rows = to_dense(adata_cf_pre.obsm["spatial_x_cf"])
    assert np.all(
        np.any(np.all(cf_rows[:, None] == cf_obsm_rows[None], axis=-1), axis=0)
    ), "spatial_x_cf rows must come from counterfactual obsm rows"

    # Regardless of n_neighbors, the per-gene mean of spatial_x_cf should be close to
    # the full-neighbourhood result (law of large numbers over basal cells).
    mean_full = to_dense(_cf(anchor_donors=False, n_neighbors=50, random_state=0).obsm[spatial_col]).mean(axis=0)
    mean_sub = to_dense(_cf(anchor_donors=False, n_neighbors=10, random_state=0).obsm[spatial_col]).mean(axis=0)
    np.testing.assert_allclose(mean_sub, mean_full, atol=1.0, err_msg=(
        "Per-gene mean of spatial_x_cf should be similar regardless of n_neighbors"
    ))


def test_normalize_losses_true(adata_with_spatial):
    """Test normalize_losses parameter in adversarial training plan."""
    Cellina.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        spatial_obsm_key="spatial_x",
        labels_key="cell_labels",
        domains_key="domain",
    )

    # Create model with discriminator, classifier, and domain_classifier enabled (non-unity lambdas)
    classifier_lambda          = 0.5
    discriminator_lambda       = 2.0
    domain_classifier_lambda   = 1.5
    model = Cellina(adata_with_spatial, n_latent=5,
                         discriminator_lambda=discriminator_lambda,
                         classifier_lambda=classifier_lambda,
                         domain_classifier_lambda=domain_classifier_lambda)

    # Train with normalize_losses=True
    model.train(
        max_epochs=2,
        train_size=0.5,
        plan_kwargs={"normalize_losses": True}
    )

    # Access the training plan from the trainer
    training_plan = model.trainer.strategy.model

    # Check warmup completed (should be done after epoch 0)
    assert training_plan._warmup_done == True, "Warmup should be completed after epoch 0"

    # Check fixed scales were computed (should be positive after warmup)
    assert training_plan._scale_clf        > 0, "Fixed scale for clf loss should be positive"
    assert training_plan._scale_fool       > 0, "Fixed scale for fool loss should be positive"
    assert training_plan._scale_domain_classifier > 0, "Fixed scale for domain_classifier loss should be positive"

    # Check normalize_losses flag is set correctly
    assert training_plan._normalize_losses == True

    # Verify training completed successfully
    # Note: warmup epoch (epoch 0) is not logged in history, so we expect 1 entry for epoch 1
    assert len(model.history_["train_loss"]) >= 1

    # Verify discriminator metrics are logged
    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"

    # --- Scale correctness: scaled == raw * fixed_scale * lambda ---
    expected_scale_clf               = training_plan._scale_clf
    expected_scale_fool              = training_plan._scale_fool
    expected_scale_domain_classifier = training_plan._scale_domain_classifier

    model.module.eval()
    model.module.to("cpu")
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    with torch.no_grad():
        inf_in  = model.module._get_inference_input(batch)
        inf_out = model.module.inference(**inf_in)
        gen_in  = model.module._get_generative_input(batch, inf_out)
        gen_out = model.module.generative(**gen_in)
        loss_out = model.module.loss(
            batch, inf_out, gen_out,
            classifier_scale=expected_scale_clf,
            discriminator_scale=expected_scale_fool,
            domain_classifier_scale=expected_scale_domain_classifier,
        )

    clf_raw                   = loss_out.extra_metrics["classifier_loss_raw"].item()
    clf_scaled                = loss_out.extra_metrics["classifier_loss"].item()
    fool_raw                  = loss_out.extra_metrics["fool_loss_raw"].item()
    fool_scaled               = loss_out.extra_metrics["fool_loss"].item()
    domain_classifier_raw     = loss_out.extra_metrics["domain_classifier_loss_raw"].item()
    domain_classifier_scaled  = loss_out.extra_metrics["domain_classifier_loss"].item()

    np.testing.assert_allclose(clf_scaled,               clf_raw               * expected_scale_clf               * classifier_lambda,        rtol=1e-4)
    np.testing.assert_allclose(fool_scaled,              fool_raw              * expected_scale_fool              * discriminator_lambda,      rtol=1e-4)
    np.testing.assert_allclose(domain_classifier_scaled, domain_classifier_raw * expected_scale_domain_classifier * domain_classifier_lambda,  rtol=1e-4)
    # make sure that disc is roughly 4x scaled compared to clf (since discriminator_lambda is 4x classifier_lambda)
    # fool_scaled is negative (adversarial weight=-1), so compare absolute magnitudes
    np.testing.assert_allclose(abs(fool_scaled / clf_scaled), discriminator_lambda / classifier_lambda, rtol=0.2)
    # assert fool is negative (since it's an adversarial loss)
    assert fool_scaled < 0, "Fool loss should be negative (adversarial weight is -1)"
    # domain_classifier is a positive supervised loss
    assert domain_classifier_scaled > 0, "domain_classifier loss should be positive (non-adversarial)"


def test_get_normalized_expression(adata_with_spatial):
    """Test get_normalized_expression method returns correct shapes."""
    n_latent = 5
    
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent)
    model.train(max_epochs=2, train_size=0.5)
    
    # Test default (numpy array)
    normalized_expr = model.get_normalized_expression()
    assert isinstance(normalized_expr, np.ndarray)
    assert normalized_expr.shape == (adata_with_spatial.n_obs, adata_with_spatial.n_vars)
    assert np.all(normalized_expr >= 0)  # Expression should be non-negative
    
    # Test return_numpy=False
    normalized_expr_tensor = model.get_normalized_expression(return_numpy=False)
    assert isinstance(normalized_expr_tensor, torch.Tensor)
    assert normalized_expr_tensor.shape == (adata_with_spatial.n_obs, adata_with_spatial.n_vars)


def test_get_counterfactual_latents(adata_with_spatial):
    """get_counterfactual_latents returns correct shape for all latent_key options."""
    n_latent = 5
    # anchor_donors=True re-aggregates the anchors' neighbourhoods, so the registered
    # spatial features must be a neighbourhood aggregation of the same expression matrix.
    compute_spatial_features(adata_with_spatial, obsm_key="spatial_x")
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    model.train(max_epochs=1, train_size=0.5)

    n_obs = adata_with_spatial.n_obs
    indices = np.arange(n_obs // 2)
    neighbour_indices = np.arange(n_obs // 2, n_obs)

    for key, expected_dim in (("s", n_latent), ("z", n_latent), ("shifted", 2 * n_latent)):
        result = model.get_counterfactual_latents(indices, neighbour_indices, latent_key=key)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(indices), expected_dim), f"latent_key={key!r}"


def test_get_counterfactual_expression(adata_with_spatial):
    """get_counterfactual_expression returns (n_indices, n_vars) of non-negative values."""
    n_latent = 5
    compute_spatial_features(adata_with_spatial, obsm_key="spatial_x")
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    model.train(max_epochs=1, train_size=0.5)

    n_obs = adata_with_spatial.n_obs
    indices = np.arange(n_obs // 2)
    neighbour_indices = np.arange(n_obs // 2, n_obs)

    result = model.get_counterfactual_expression(indices, neighbour_indices)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(indices), adata_with_spatial.n_vars)
    assert np.all(result >= 0)


def test_get_perturbed_latents(adata_with_spatial):
    """get_perturbed_latents returns (n_obs, n_latent) when given a counterfactual obsm key."""
    n_latent = 5
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    model.train(max_epochs=1, train_size=0.5)

    adata_with_spatial.obsm["spatial_x_cf"] = adata_with_spatial.obsm["spatial_x"].copy()

    result = model.get_perturbed_latents(spatial_obsm_key="spatial_x_cf")
    assert isinstance(result, np.ndarray)
    assert result.shape == (adata_with_spatial.n_obs, n_latent)


def test_get_perturbed_expression(adata_with_spatial):
    """get_perturbed_expression returns (n_obs, n_vars) of non-negative values."""
    n_latent = 5
    Cellina.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = Cellina(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    model.train(max_epochs=1, train_size=0.5)
    
    assert model.module.classifier is None, "Classifier should be disabled when classifier_lambda=0.0"
    assert model.module.domain_discriminator is None, "Discriminator should be disabled when discriminator_lambda=0.0"

    adata_with_spatial.obsm["spatial_x_cf"] = adata_with_spatial.obsm["spatial_x"].copy()

    result = model.get_perturbed_expression(spatial_obsm_key="spatial_x_cf")
    assert isinstance(result, np.ndarray)
    assert result.shape == (adata_with_spatial.n_obs, adata_with_spatial.n_vars)
    assert np.all(result >= 0)


def test_make_neighbor_perturbation(adata_with_spatial):
    """Partial perturbations dict (only some cell types) runs without error."""
    genes = list(adata_with_spatial.var_names[:3])
    perturbations = {"0": pd.Series([1.0, -0.5, 0.5], index=genes)}  # only cell type "0"

    make_neighbor_perturbation(
        adata_with_spatial,
        perturbations=perturbations,
        groupby="cell_labels",
        obsm_key_out="spatial_x_cf",
    )

    assert "spatial_x_cf" in adata_with_spatial.obsm
    assert adata_with_spatial.obsm["spatial_x_cf"].shape == (
        adata_with_spatial.n_obs, adata_with_spatial.n_vars
    )

    make_neighbor_perturbation(
        adata_with_spatial,
        perturbations=perturbations,
        groupby="cell_labels",
        obsm_key_out="spatial_x_cf_add",
        add_shift=True,
    )

    assert "spatial_x_cf_add" in adata_with_spatial.obsm
    assert adata_with_spatial.obsm["spatial_x_cf_add"].shape == (
        adata_with_spatial.n_obs, adata_with_spatial.n_vars
    )


def test_node_perturbation_row_sum_invariance():
    """_node_perturbation preserves row sums for both add_shift modes."""
    rng = np.random.default_rng(42)
    X = csr_matrix(rng.random((10, 5)).astype(np.float32))
    var_idx = {f"gene{i}": i for i in range(5)}
    row_sums_before = np.asarray(X.sum(axis=1)).ravel()
    pert = {"gene0": 3.0, "gene1": -1.0}

    for add_shift in (False, True):
        X_out = _node_perturbation(X, var_idx, pert, add_shift=add_shift, renormalize=True)
        np.testing.assert_allclose(
            np.asarray(X_out.sum(axis=1)).ravel(), row_sums_before, rtol=1e-5
        )


def test_node_perturbation_fraction():
    """perturb_fraction perturbs a seeded random subset of rows; 1.0 is the old behaviour."""
    rng = np.random.default_rng(0)
    X = rng.random((100, 5)).astype(np.float32)
    var_idx = {f"gene{i}": i for i in range(5)}
    pert = {"gene0": 1.0}

    def run(**kw):
        return _node_perturbation(X, var_idx, pert, add_shift=True, **kw).toarray()

    np.testing.assert_array_equal(run(perturb_fraction=1.0), run())
    np.testing.assert_array_equal(run(perturb_fraction=0.0), X)

    half = run(perturb_fraction=0.5, random_state=1)
    assert (half != X).any(axis=1).sum() == 50
    np.testing.assert_array_equal(half, run(perturb_fraction=0.5, random_state=1))

    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            _node_perturbation(X, var_idx, pert, perturb_fraction=bad)


def test_make_neighbor_perturbation_unknown(adata_with_spatial):
    """Unknown cell-type key in perturbations raises ValueError."""
    perturbations = {"nonexistent_type": pd.Series([1.0], index=[adata_with_spatial.var_names[0]])}

    with pytest.raises(ValueError, match="nonexistent_type"):
        make_neighbor_perturbation(
            adata_with_spatial,
            perturbations=perturbations,
            groupby="cell_labels",
        )


# ---------------------------------------------------------------------------
# Tests for the explicit ``layer`` argument on the counterfactual /
# perturbation helpers.
#
# The training ``spatial_x`` is usually built from a normalized representation
# (log1p of CP10K) while ``adata.X`` is reset to raw counts for the model. These
# tests pin down that the counterfactual and node-perturbation helpers aggregate
# the representation the caller asks for.
# ---------------------------------------------------------------------------

N_OBS = 60
N_VARS = 20


def _to_dense(x):
    return np.asarray(x.toarray() if issparse(x) else x, dtype=np.float64)


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=8.0, size=(N_OBS, N_VARS)).astype(np.float32)
    # make sure the raw-count heuristic triggers (integer valued, max > 50)
    counts[0, 0] = 120.0
    coords = rng.uniform(0, 100, size=(N_OBS, 2))

    adata = AnnData(X=csr_matrix(counts))
    adata.obs_names = [f"cell_{i}" for i in range(N_OBS)]
    adata.var_names = [f"gene_{i}" for i in range(N_VARS)]
    adata.obsm["spatial"] = coords
    adata.layers["counts"] = csr_matrix(counts)

    # log1p(CP10K) "training" representation
    lognorm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-8) * 1e4
    lognorm = np.log1p(lognorm).astype(np.float32)
    adata.layers["lognorm"] = csr_matrix(lognorm)

    # simple kNN connectivity (binary, symmetric)
    nn = NearestNeighbors(n_neighbors=6).fit(coords)
    conn = nn.kneighbors_graph(coords, mode="connectivity")
    conn.setdiag(0)
    conn.eliminate_zeros()
    conn = conn.maximum(conn.T).tocsr().astype(np.float32)
    adata.obsp["spatial_connectivities"] = conn

    # training spatial features come from the lognorm layer
    compute_spatial_features(adata, obsm_key="spatial_x", layer="lognorm")
    return adata


def _expected_aggregation(conn, matrix):
    """Degree-normalized neighbour mean of ``matrix`` over ``conn``."""
    C = csr_matrix(conn)
    result = _to_dense(C @ csr_matrix(matrix))
    degree = np.asarray(C.sum(axis=1)).ravel()
    denom = 1.0 / np.where(degree == 0, 1.0, degree)
    return result * denom[:, None]


def test_counterfactual_layer_aggregates_named_layer(adata):
    indices = np.arange(0, 10)
    neighbour_indices = np.arange(30, 60)

    adata_cf = make_counterfactual_adata(
        adata,
        indices,
        neighbour_indices,
        spatial_column="spatial_x",
        anchor_donors=False,
        n_neighbors=5,
        random_state=0,
        layer="lognorm",
    )

    # reuse the cf connectivity the function wrote back to the source adata
    expected = _expected_aggregation(
        adata.obsp["spatial_connectivities_cf"], adata.layers["lognorm"]
    )[indices]
    np.testing.assert_allclose(
        _to_dense(adata_cf.obsm["spatial_x"]), expected, rtol=1e-5, atol=1e-6
    )

    # and it is *not* the raw-count aggregation
    raw_expected = _expected_aggregation(
        adata.obsp["spatial_connectivities_cf"], adata.X
    )[indices]
    assert not np.allclose(_to_dense(adata_cf.obsm["spatial_x"]), raw_expected)


def test_counterfactual_layer_none_matches_previous_behaviour(adata):
    """``layer=None`` keeps the 1.1.1 behaviour: ``adata.X`` is aggregated."""
    indices = np.arange(0, 10)
    neighbour_indices = np.arange(30, 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata_cf = make_counterfactual_adata(
            adata,
            indices,
            neighbour_indices,
            spatial_column="spatial_x",
            anchor_donors=False,
            n_neighbors=5,
            random_state=0,
        )

    expected = _expected_aggregation(
        adata.obsp["spatial_connectivities_cf"], adata.X
    )[indices]
    np.testing.assert_allclose(
        _to_dense(adata_cf.obsm["spatial_x"]), expected, rtol=1e-5, atol=1e-6
    )


def test_raw_counts_warning(adata):
    with pytest.warns(UserWarning, match="raw counts"):
        compute_spatial_features(adata, obsm_key="spatial_x_warn")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_features(adata, obsm_key="spatial_x_nowarn", layer="lognorm")


def test_neighbor_perturbation_sources_from_layer(adata):
    perturbations = {"gene_0": 1.0, "gene_1": -0.5}

    make_neighbor_perturbation(
        adata,
        perturbations=perturbations,
        obsm_key_out="spatial_x_cf",
        layer_key="pert_lognorm",
        add_shift=True,
        renormalize=False,
        layer="lognorm",
    )

    lognorm = _to_dense(adata.layers["lognorm"])
    shift = np.zeros(N_VARS)
    shift[0] = 1.0
    shift[1] = -0.5
    expected_pert = np.clip(lognorm + shift, 0, None)
    np.testing.assert_allclose(
        _to_dense(adata.layers["pert_lognorm"]), expected_pert, rtol=1e-5, atol=1e-6
    )

    expected_cf = _expected_aggregation(
        adata.obsp["spatial_connectivities"], expected_pert
    )
    np.testing.assert_allclose(
        _to_dense(adata.obsm["spatial_x_cf"]), expected_cf, rtol=1e-5, atol=1e-5
    )

    # sourcing from X (raw counts) gives a different perturbed matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        make_neighbor_perturbation(
            adata,
            perturbations=perturbations,
            obsm_key_out="spatial_x_cf_raw",
            layer_key="pert_raw",
            add_shift=True,
            renormalize=False,
        )
    assert not np.allclose(
        _to_dense(adata.layers["pert_raw"]), _to_dense(adata.layers["pert_lognorm"])
    )


def test_neighbor_perturbation_layer_and_fraction_compose(adata):
    """``layer=`` (#41) and ``perturb_fraction=`` (#39) work together."""
    perturbations = {"gene_0": 1.0, "gene_1": -0.5}
    perturb_fraction = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_neighbor_perturbation(
            adata,
            perturbations=perturbations,
            obsm_key_out="spatial_x_cf_frac",
            layer_key="pert_lognorm_frac",
            add_shift=True,
            renormalize=False,
            perturb_fraction=perturb_fraction,
            random_state=0,
            layer="lognorm",
        )

    lognorm = _to_dense(adata.layers["lognorm"])
    perturbed = _to_dense(adata.layers["pert_lognorm_frac"])
    assert perturbed.shape == (N_OBS, N_VARS)

    shift = np.zeros(N_VARS)
    shift[0] = 1.0
    shift[1] = -0.5
    full_expected = np.clip(lognorm + shift, 0, None)

    changed = ~np.isclose(perturbed, lognorm, rtol=1e-5, atol=1e-6).all(axis=1)
    # exactly the requested fraction of cells is perturbed
    assert changed.sum() == int(round(perturb_fraction * N_OBS))
    # perturbed cells carry the layer-sourced shift, the rest keep the layer values
    np.testing.assert_allclose(
        perturbed[changed], full_expected[changed], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        perturbed[~changed], lognorm[~changed], rtol=1e-5, atol=1e-6
    )

    # the aggregated features follow the partially perturbed layer, and sit
    # strictly between "nothing perturbed" and "everything perturbed"
    expected_cf = _expected_aggregation(
        adata.obsp["spatial_connectivities"], perturbed
    )
    np.testing.assert_allclose(
        _to_dense(adata.obsm["spatial_x_cf_frac"]), expected_cf, rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(
        _to_dense(adata.obsm["spatial_x_cf_frac"]),
        _expected_aggregation(adata.obsp["spatial_connectivities"], full_expected),
        rtol=1e-5,
        atol=1e-5,
    )

# ------------------------------------------------- anchor draw: recomputed anchor features
def _anchor_case(adata):
    """Focal cells, anchors and the 'held-out type' (focal + anchors) of the fixture graph."""
    n = adata.n_obs
    focal = np.arange(0, 10)
    anchors = np.arange(n - 15, n)
    exclude = np.concatenate([focal, anchors])
    return focal, anchors, exclude


def _mask_cols(conn, idx):
    """Copy of ``conn`` with the columns in ``idx`` zeroed (and dropped from the pattern)."""
    C = csr_matrix(conn, copy=True)
    excluded = np.zeros(C.shape[1], dtype=bool)
    excluded[np.asarray(idx)] = True
    C.data = np.where(excluded[C.indices], 0, C.data)
    C.eliminate_zeros()
    return C


def _mask_rows_cols(conn, idx):
    """Drop every edge touching ``idx`` — what ``spatial_neighbors(test_indices=...)`` does."""
    coo = csr_matrix(conn).tocoo()
    keep = ~(np.isin(coo.row, idx) | np.isin(coo.col, idx))
    return csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape)


def _loo_adata(adata, exclude):
    """Mask ``exclude`` out of the registered graph, keeping the unmasked graph aside."""
    conn = csr_matrix(adata.obsp["spatial_connectivities"])
    adata.obsp["spatial_connectivities_orig"] = conn
    adata.obsp["spatial_connectivities"] = _mask_rows_cols(conn, exclude)
    compute_spatial_features(adata, obsm_key="spatial_x", layer="lognorm")
    return adata


def _anchor_cf(adata, focal, anchors, exclude, **kw):
    opts = dict(
        anchor_donors=True,
        connectivity_key="spatial_connectivities_orig",
        exclude_indices=exclude,
        layer="lognorm",
        random_state=0,
    )
    opts.update(kw)
    return make_counterfactual_adata(adata, focal, anchors, "spatial_x", **opts)


def _expected_anchor_rows(adata, anchors, exclude, n_draw, seed=0):
    """What the anchor draw must produce: compute_spatial_features over the masked graph."""
    adata.obsp["_masked"] = _mask_cols(adata.obsp["spatial_connectivities_orig"], exclude)
    compute_spatial_features(adata, connectivity_key="_masked", obsm_key="_ref", layer="lognorm")
    kept = np.diff(csr_matrix(adata.obsp["_masked"])[anchors].indptr) > 0
    ref = _to_dense(csr_matrix(adata.obsm["_ref"])[anchors[kept]])
    idx = np.random.default_rng(seed).integers(0, kept.sum(), size=n_draw)
    return ref[idx], ref, kept


def test_anchor_rows_match_compute_spatial_features(adata):
    """Anchor features are bit-identical to compute_spatial_features on the masked graph."""
    focal, anchors, exclude = _anchor_case(adata)
    _loo_adata(adata, exclude)

    # the masked anchors' *stored* rows are all-zero: the old verbatim copy fed zeros
    assert csr_matrix(adata.obsm["spatial_x"])[anchors].nnz == 0

    adata_cf = _anchor_cf(adata, focal, anchors, exclude)
    expected, ref, kept = _expected_anchor_rows(adata, anchors, exclude, len(focal))
    assert kept.all(), "fixture must have no homotypic anchor here"
    assert ref.any(), "reference anchor rows are all zero — vacuous check"

    got = adata_cf.obsm["spatial_x"]
    assert issparse(got) and got.dtype == np.float32
    np.testing.assert_array_equal(_to_dense(got), expected)
    np.testing.assert_array_equal(_to_dense(adata_cf.obsm["spatial_x_cf"]), expected)
    assert adata_cf.n_obs == len(focal) and adata_cf.n_vars == adata.n_vars
    # and it is emphatically not the stored (zero) copy the old code handed over
    assert _to_dense(got).any()


def test_anchor_excluded_cells_never_reach_the_counterfactual(adata):
    """Excluded cells are inert: scrambling their expression leaves the cf features intact."""
    focal, anchors, exclude = _anchor_case(adata)
    _loo_adata(adata, exclude)

    base = _to_dense(_anchor_cf(adata, focal, anchors, exclude).obsm["spatial_x"])
    _, ref, kept = _expected_anchor_rows(adata, anchors, exclude, len(focal))
    # every drawn row is one of the (kept) anchors' neighbourhood rows
    assert all(np.any(np.all(ref == row, axis=1)) for row in base)

    # positive control: without the exclusion the excluded cells do contribute
    leaky = _to_dense(_anchor_cf(adata, focal, anchors, exclude, exclude_indices=None)
                      .obsm["spatial_x"])
    assert not np.allclose(leaky, base)

    lognorm = _to_dense(adata.layers["lognorm"])
    scrambled = lognorm.copy()
    scrambled[exclude] = np.random.default_rng(1).permutation(scrambled[exclude], axis=0) + 3.0
    assert not np.allclose(scrambled[exclude], lognorm[exclude]), "scramble was a no-op"
    adata.layers["lognorm"] = csr_matrix(scrambled.astype(np.float32))

    after = _to_dense(_anchor_cf(adata, focal, anchors, exclude).obsm["spatial_x"])
    np.testing.assert_array_equal(
        base, after,
        err_msg="excluded cells' expression changed the counterfactual features",
    )


def test_anchor_homotypic_anchors_are_dropped(adata):
    """An anchor with only excluded neighbours warns, is never drawn, and alone raises."""
    focal, anchors, exclude = _anchor_case(adata)
    _loo_adata(adata, exclude)
    conn = csr_matrix(adata.obsp["spatial_connectivities_orig"])
    masked = _mask_cols(conn, exclude)

    # pick the anchor with the fewest surviving neighbours and exclude them too
    deg = np.diff(masked[anchors].indptr)
    a_iso = int(anchors[np.argmin(deg)])
    nb = masked.indices[masked.indptr[a_iso]:masked.indptr[a_iso + 1]]
    exclude_iso = np.unique(np.concatenate([exclude, nb]))

    with pytest.warns(UserWarning, match="1 of 15 anchors are homotypic"):
        cf = _anchor_cf(adata, focal, anchors, exclude_iso)
    good = _anchor_cf(adata, focal, anchors[anchors != a_iso], exclude_iso)
    # dropping the homotypic anchor by hand gives exactly the same draw
    np.testing.assert_array_equal(_to_dense(cf.obsm["spatial_x"]),
                                  _to_dense(good.obsm["spatial_x"]))
    assert _to_dense(cf.obsm["spatial_x"]).any(), "no all-zero (homotypic) row may be drawn"
    assert (_to_dense(cf.obsm["spatial_x"]) != 0).any(axis=1).all()

    with pytest.raises(ValueError, match="All anchors are homotypic"):
        _anchor_cf(adata, focal, np.array([a_iso]), exclude_iso)
    with pytest.raises(ValueError, match="at least one anchor"):
        _anchor_cf(adata, focal, np.array([], dtype=int), exclude)
    with pytest.raises(ValueError, match="1-D integer"):
        _anchor_cf(adata, focal, anchors, np.isin(np.arange(adata.n_obs), exclude))
    with pytest.raises(ValueError, match="outside"):
        _anchor_cf(adata, focal, anchors, np.array([adata.n_obs]))
    with pytest.raises(KeyError, match="connectivity_key"):
        _anchor_cf(adata, focal, anchors, exclude, connectivity_key="no_such_graph")


def test_anchor_unmasked_graph_reproduces_stored_spatial_x(adata):
    """Backward compatibility: unmasked graph, no exclusion, matching layer -> stored rows."""
    focal, anchors, _ = _anchor_case(adata)
    stored = _to_dense(csr_matrix(adata.obsm["spatial_x"])[anchors])

    adata_cf = make_counterfactual_adata(
        adata, focal, anchors, "spatial_x", anchor_donors=True, random_state=0,
        layer="lognorm",
    )
    idx = np.random.default_rng(0).integers(0, len(anchors), size=len(focal))
    np.testing.assert_array_equal(_to_dense(adata_cf.obsm["spatial_x"]), stored[idx])
    # the source AnnData is untouched
    assert "spatial_x_cf" not in adata.obsm
    assert "spatial_connectivities_cf" not in adata.obsp


def test_anchor_draw_is_seeded(adata):
    focal, anchors, exclude = _anchor_case(adata)
    _loo_adata(adata, exclude)
    a = _to_dense(_anchor_cf(adata, focal, anchors, exclude, random_state=3).obsm["spatial_x"])
    b = _to_dense(_anchor_cf(adata, focal, anchors, exclude, random_state=3).obsm["spatial_x"])
    c = _to_dense(_anchor_cf(adata, focal, anchors, exclude, random_state=4).obsm["spatial_x"])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_anchor_exclude_indices_threaded_through_the_model(adata_with_spatial):
    """exclude_indices reaches make_counterfactual_adata from both model entry points."""
    adata = adata_with_spatial
    focal, anchors, exclude = _anchor_case(adata)
    conn = csr_matrix(adata.obsp["spatial_connectivities"])
    adata.obsp["spatial_connectivities_orig"] = conn
    adata.obsp["spatial_connectivities"] = _mask_rows_cols(conn, exclude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        compute_spatial_features(adata, obsm_key="spatial_x")

    Cellina.setup_anndata(adata, batch_key="batch", spatial_obsm_key="spatial_x",
                          domains_key="domain")
    model = Cellina(adata, n_latent=5, classifier_lambda=0.0, discriminator_lambda=0.0)
    model.train(max_epochs=1, train_size=0.5)

    kw = dict(connectivity_key="spatial_connectivities_orig", exclude_indices=exclude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expr = model.get_counterfactual_expression(focal, anchors, **kw)
        lat = model.get_counterfactual_latents(focal, anchors, latent_key="s", **kw)
        cf = model._make_counterfactual_adata(focal, anchors, seed=0, **kw)
        leaky = model.get_counterfactual_expression(
            focal, anchors, connectivity_key="spatial_connectivities_orig",
            exclude_indices=None,
        )
    assert expr.shape == (len(focal), adata.n_vars) and np.all(expr >= 0)
    assert lat.shape == (len(focal), 5)
    assert _to_dense(cf.obsm["spatial_x"]).any(), "model path still handed over zero rows"
    assert not np.allclose(expr, leaky), \
        "exclude_indices had no effect when passed through the model"


def test_anchor_exclude_indices_rejected_by_the_pooled_branch(adata):
    """exclude_indices is anchor-only: the pooled draw refuses it instead of ignoring it."""
    focal, anchors, exclude = _anchor_case(adata)
    with pytest.raises(ValueError, match="only used when anchor_donors=True"):
        make_counterfactual_adata(
            adata, focal, np.arange(20, 50), "spatial_x", anchor_donors=False,
            n_neighbors=5, layer="lognorm", exclude_indices=exclude,
        )


def test_anchor_rejects_a_narrower_stored_spatial_representation(adata):
    """A reduced / gene-subset spatial_x cannot be reproduced by the anchor path."""
    focal, anchors, _ = _anchor_case(adata)
    adata.obsm["spatial_reduced"] = np.zeros((adata.n_obs, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="neighbor_genes"):
        make_counterfactual_adata(
            adata, focal, anchors, "spatial_reduced", anchor_donors=True,
            random_state=0, layer="lognorm",
        )
    # the gene-subset case the message points at
    compute_spatial_features(adata, obsm_key="spatial_subset", layer="lognorm",
                             neighbor_genes=list(adata.var_names[:4]))
    with pytest.raises(ValueError, match="4 columns"):
        make_counterfactual_adata(
            adata, focal, anchors, "spatial_subset", anchor_donors=True,
            random_state=0, layer="lognorm",
        )


def test_anchor_layer_must_match_the_stored_spatial_features(adata):
    """``layer`` must name the representation spatial_x was built from, else rows differ."""
    focal, anchors, _ = _anchor_case(adata)
    # the fixture builds spatial_x from layers['lognorm'] while adata.X holds counts
    stored = _to_dense(csr_matrix(adata.obsm["spatial_x"])[anchors])
    idx = np.random.default_rng(0).integers(0, len(anchors), size=len(focal))

    matched = make_counterfactual_adata(adata, focal, anchors, "spatial_x",
                                        anchor_donors=True, random_state=0, layer="lognorm")
    np.testing.assert_array_equal(_to_dense(matched.obsm["spatial_x"]), stored[idx])

    with pytest.warns(UserWarning, match="raw counts"):
        mismatched = make_counterfactual_adata(adata, focal, anchors, "spatial_x",
                                               anchor_donors=True, random_state=0)
    assert not np.allclose(_to_dense(mismatched.obsm["spatial_x"]), stored[idx]), \
        "layer=None must not silently reproduce features built from a layer"
