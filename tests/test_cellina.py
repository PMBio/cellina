import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid

from cellina import CellinaModel


@pytest.fixture
def adata_with_spatial():
    """Create synthetic AnnData with spatial features and connectivity."""
    from cellina._spatial_utils import spatial_neighbors
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
    """Test basic CellinaModel functionality."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial,
                               batch_key="batch",
                               spatial_obsm_key="spatial_x",
                               labels_key="cell_labels",
                               domains_key="domain"
                               )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    
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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)

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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", labels_key="cell_labels", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=1.0)
    
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
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")

    # Should work fine without labels when classifier_lambda=0
    model = CellinaModel(adata_with_spatial, n_latent=5, classifier_lambda=0.0)
    assert model.module.classifier is None
    assert model.module.classifier_lambda == 0.0


def test_discriminator_enabled_by_default(adata_with_spatial):
    """Test that discriminator is enabled by default (discriminator_lambda=1.0)."""
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")

    model = CellinaModel(adata_with_spatial, n_latent=5)
    assert model.module.domain_discriminator is not None
    assert model.module.discriminator_lambda == 1.0

    # Explicitly set to 0 disables it
    model2 = CellinaModel(adata_with_spatial, n_latent=5, discriminator_lambda=0.0)
    assert model2.module.domain_discriminator is None
    assert model2.module.discriminator_lambda == 0.0


def test_discriminator_enabled(adata_with_spatial):
    """Test that discriminator works when discriminator_lambda > 0."""
    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        spatial_obsm_key="spatial_x",
        domains_key="domain",
    )

    n_latent = 5
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, discriminator_lambda=1.0)

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
    
    CellinaModel.setup_anndata(adata_with_spatial, 
                               batch_key="batch", spatial_obsm_key="spatial_x",
                               labels_key="cell_labels", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)
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
    from cellina._spatial_utils import spatial_neighbors
    from scipy.sparse import issparse

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


def test_compute_spatial_features(adata_with_spatial):
    """Test compute_spatial_features function (pseudobulk mode)."""
    from cellina._spatial_utils import compute_spatial_features, spatial_neighbors

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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)
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


def test_condition_on_intrinsic_false(adata_with_spatial):
    """Test s_encoder architecture changes with condition_on_intrinsic=False."""
    n_latent = 5
    n_spatial = adata_with_spatial.obsm["spatial_x"].shape[1]
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    
    # Test with condition_on_intrinsic=True (default)
    model_true = CellinaModel(adata_with_spatial, n_latent=n_latent, condition_on_intrinsic=True)
    assert model_true.module.condition_on_intrinsic == True
    
    # Test with condition_on_intrinsic=False
    model_false = CellinaModel(adata_with_spatial, n_latent=n_latent, condition_on_intrinsic=False)
    assert model_false.module.condition_on_intrinsic == False
    
    # Check the difference in encoder input dimensions
    # The encoder has inject_covariates=True, so batch info is also injected
    # But we can verify the difference is exactly n_latent
    input_dim_true = model_true.module.s_encoder.encoder.fc_layers[0][0].in_features
    input_dim_false = model_false.module.s_encoder.encoder.fc_layers[0][0].in_features
    assert input_dim_true - input_dim_false == n_latent, \
        f"Difference in input dims should be n_latent={n_latent}, got {input_dim_true - input_dim_false}"
    
    # Test training works with condition_on_intrinsic=False
    model_false.train(max_epochs=2, train_size=0.5)
    
    # Test inference outputs have correct shapes
    dataloader = model_false._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    model_false.module.eval()
    with torch.no_grad():
        inference_outputs = model_false.module.inference(**model_false.module._get_inference_input(batch))

def test_make_counterfactual_adata(adata_with_spatial):
    """Test make_counterfactual_adata with precomputed=False and precomputed=True."""
    from cellina._spatial_utils import make_counterfactual_adata
    from cellina._spatial_utils import compute_spatial_features

    # Compute spatial_x from gene expression so feature dim matches precomputed=False output
    compute_spatial_features(adata_with_spatial, connectivity_key="spatial_connectivities", obsm_key="spatial_x")

    to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    n_obs = adata_with_spatial.n_obs
    indices_basal = np.arange(0, n_obs // 2)
    indices_cf = np.arange(n_obs // 2, n_obs)
    spatial_col = "spatial_x"

    def _cf(**kw):
        return make_counterfactual_adata(
            adata_with_spatial, indices_basal, indices_cf, spatial_col, **kw
        )

    # precomputed=False: rebuild via compute_spatial_features
    adata_cf = _cf(precomputed=False)
    assert adata_cf.n_obs == len(indices_basal)
    assert adata_cf.n_vars == adata_with_spatial.n_vars
    assert adata_cf.obsm[spatial_col].shape[0] == len(indices_basal)
    np.testing.assert_array_equal(adata_cf.X, adata_with_spatial[indices_basal].X)

    # reproducibility: with n_neighbours the RNG is used; same random_state → same result
    np.testing.assert_array_equal(
        to_dense(_cf(precomputed=False, n_neighbours=3, random_state=7).obsm[spatial_col]),
        to_dense(_cf(precomputed=False, n_neighbours=3, random_state=7).obsm[spatial_col]),
    )

    # precomputed=True: rows sampled from existing obsm; reproducible with same random_state
    adata_cf_pre = _cf(precomputed=True, random_state=0)
    np.testing.assert_array_equal(
        to_dense(adata_cf_pre.obsm[spatial_col]),
        to_dense(_cf(precomputed=True, random_state=0).obsm[spatial_col]),
    )
    cf_rows = to_dense(adata_with_spatial.obsm[spatial_col][indices_cf])
    result_rows = to_dense(adata_cf_pre.obsm[spatial_col])
    assert np.all(
        np.any(np.all(cf_rows[:, None] == result_rows[None], axis=-1), axis=0)
    ), "precomputed=True rows must come from counterfactual obsm rows"

    # precomputed=True also writes spatial_x_cf; rows must come from the cf pool
    assert "spatial_x_cf" in adata_cf_pre.obsm, "spatial_x_cf should exist for precomputed=True"
    cf_obsm_rows = to_dense(adata_cf_pre.obsm["spatial_x_cf"])
    assert np.all(
        np.any(np.all(cf_rows[:, None] == cf_obsm_rows[None], axis=-1), axis=0)
    ), "spatial_x_cf rows must come from counterfactual obsm rows"

    # Regardless of n_neighbours, the per-gene mean of spatial_x_cf should be close to
    # the full-neighbourhood result (law of large numbers over basal cells).
    mean_full = to_dense(_cf(precomputed=False, n_neighbours=50, random_state=0).obsm[spatial_col]).mean(axis=0)
    mean_sub = to_dense(_cf(precomputed=False, n_neighbours=10, random_state=0).obsm[spatial_col]).mean(axis=0)
    np.testing.assert_allclose(mean_sub, mean_full, atol=1.0, err_msg=(
        "Per-gene mean of spatial_x_cf should be similar regardless of n_neighbours"
    ))


def test_normalize_losses_true(adata_with_spatial):
    """Test normalize_losses parameter in adversarial training plan."""
    CellinaModel.setup_anndata(
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
    model = CellinaModel(adata_with_spatial, n_latent=5,
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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
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
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
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
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
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
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
    model.train(max_epochs=1, train_size=0.5)

    adata_with_spatial.obsm["spatial_x_cf"] = adata_with_spatial.obsm["spatial_x"].copy()

    result = model.get_perturbed_latents(spatial_obsm_key="spatial_x_cf")
    assert isinstance(result, np.ndarray)
    assert result.shape == (adata_with_spatial.n_obs, n_latent)


def test_get_perturbed_expression(adata_with_spatial):
    """get_perturbed_expression returns (n_obs, n_vars) of non-negative values."""
    n_latent = 5
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", domains_key="domain")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)
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
    import pandas as pd
    from cellina._spatial_utils import make_neighbor_perturbation

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
    from scipy.sparse import csr_matrix
    from cellina._spatial_utils import _node_perturbation

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


def test_make_neighbor_perturbation_unknown(adata_with_spatial):
    """Unknown cell-type key in perturbations raises ValueError."""
    import pandas as pd
    from cellina._spatial_utils import make_neighbor_perturbation

    perturbations = {"nonexistent_type": pd.Series([1.0], index=[adata_with_spatial.var_names[0]])}

    with pytest.raises(ValueError, match="nonexistent_type"):
        make_neighbor_perturbation(
            adata_with_spatial,
            perturbations=perturbations,
            groupby="cell_labels",
        )
