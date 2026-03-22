import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid

from cellina import CellinaModel


@pytest.fixture
def adata_with_spatial():
    """Create synthetic AnnData with spatial features."""
    adata = synthetic_iid()
    n_spatial_features = 20
    adata.obsm["spatial_x"] = np.random.randn(adata.n_obs, n_spatial_features).astype(np.float32)
    n_labels = 3
    adata.obs["cell_labels"] = np.random.randint(0, n_labels, size=adata.n_obs).astype(str)
    return adata


def test_cellina_model(adata_with_spatial):
    """Test basic CellinaModel functionality."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial,
                               batch_key="batch",
                               spatial_obsm_key="spatial_x",
                               labels_key="cell_labels"
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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", labels_key="cell_labels")
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


def test_classifier_disabled_by_default():
    """Test that classifier is disabled when classifier_lambda=0."""
    adata = synthetic_iid()
    n_spatial_features = 20
    adata.obsm["spatial_x"] = np.random.randn(adata.n_obs, n_spatial_features).astype(np.float32)
    
    CellinaModel.setup_anndata(adata, batch_key="batch", spatial_obsm_key="spatial_x")
    
    # Should work fine without labels when classifier_lambda=0
    model = CellinaModel(adata, n_latent=5, classifier_lambda=0.0)
    assert model.module.classifier is None
    assert model.module.classifier_lambda == 0.0


def test_discriminator_disabled_by_default():
    """Test that discriminator is disabled when discriminator_lambda=0 (default)."""
    adata = synthetic_iid()
    n_spatial_features = 20
    adata.obsm["spatial_x"] = np.random.randn(adata.n_obs, n_spatial_features).astype(np.float32)
    
    CellinaModel.setup_anndata(adata, batch_key="batch", spatial_obsm_key="spatial_x")
    
    # Default discriminator_lambda should be 0
    model = CellinaModel(adata, n_latent=5)
    assert model.module.domain_discriminator is None
    assert model.module.discriminator_lambda == 0.0
    
    # Explicitly set to 0
    model2 = CellinaModel(adata, n_latent=5, discriminator_lambda=0.0)
    assert model2.module.domain_discriminator is None
    assert model2.module.discriminator_lambda == 0.0


def test_discriminator_enabled():
    """Test that discriminator works when discriminator_lambda > 0."""
    adata = synthetic_iid()
    n_spatial_features = 20
    adata.obsm["spatial_x"] = np.random.randn(adata.n_obs, n_spatial_features).astype(np.float32)
    
    # Add domain labels (required for discriminator)
    n_domains = 3
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)
    
    CellinaModel.setup_anndata(
        adata, 
        batch_key="batch", 
        spatial_obsm_key="spatial_x",
        domains_key="domain"
    )
    
    n_latent = 5
    model = CellinaModel(adata, n_latent=n_latent, discriminator_lambda=1.0)
    
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
    test_batch = {
        "x": torch.abs(torch.randn(10, adata.n_vars)),  # Use positive values
        "spatial_x": torch.randn(10, n_spatial_features),
        "batch_index": torch.zeros(10, 1, dtype=torch.long),  # Add batch_index
    }
    with torch.no_grad():
        outputs = model.module.inference(**test_batch)
    
    assert "discriminator_logits" in outputs
    assert outputs["discriminator_logits"].shape == (10, n_domains)


def test_cellina_latent_representation(adata_with_spatial):
    """Test latent representation returns correct shapes and uses latent_key."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x", labels_key="cell_labels")
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
    
    # Add spatial coordinates to adata
    n_obs = adata_with_spatial.n_obs
    adata_with_spatial.obsm['spatial'] = np.random.rand(n_obs, 2) * 100
    
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


def test_weighted_pseudobulks(adata_with_spatial):
    """Test compute_spatial_features function (pseudobulk mode)."""
    from cellina._spatial_utils import compute_spatial_features, spatial_neighbors

    # Setup spatial data
    n_obs = adata_with_spatial.n_obs
    adata_with_spatial.obsm['spatial'] = np.random.rand(n_obs, 2) * 100

    # Add cell type labels
    cell_types = ['TypeA', 'TypeB', 'TypeC']
    adata_with_spatial.obs['cell_type'] = np.random.choice(cell_types, n_obs)

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
        groupby='cell_type',
        obsm_key='spatial_pseudobulks',
        binarize=True
    )
    
    # Check that pseudobulks were added
    assert 'spatial_pseudobulks' in adata_with_spatial.obsm
    
    # Check shape: should be (n_obs, n_cell_types * n_genes)
    n_genes = adata_with_spatial.n_vars
    n_cell_types = len(cell_types)
    assert adata_with_spatial.obsm['spatial_pseudobulks'].shape == (n_obs, n_cell_types * n_genes)
    


def test_marginal_ll(adata_with_spatial):
    """Test get_marginal_ll method and underlying module.marginal_ll."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
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
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
    
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
    """Test make_counterfactual_adata function with both sampling modes."""
    from cellina._utils import make_counterfactual_adata
    
    # Replace spatial_x with positive count data for NB sampling
    n_spatial_features = 20
    adata_with_spatial.obsm["spatial_x"] = np.random.poisson(5, size=(adata_with_spatial.n_obs, n_spatial_features)).astype(np.float32)
    
    n_obs = adata_with_spatial.n_obs
    indices_basal = np.arange(0, n_obs // 2)
    indices_cf = np.arange(n_obs // 2, n_obs)
    spatial_col = "spatial_x"
    
    # Test sample=True (NB sampling)
    adata_cf_sampled = make_counterfactual_adata(
        adata_with_spatial, indices_basal, indices_cf, spatial_col, sample=True, random_state=42
    )
    
    # Check basic properties
    assert adata_cf_sampled.n_obs == len(indices_basal)
    assert adata_cf_sampled.n_vars == adata_with_spatial.n_vars
    assert spatial_col in adata_cf_sampled.obsm
    assert adata_cf_sampled.obsm[spatial_col].shape == (len(indices_basal), adata_with_spatial.obsm[spatial_col].shape[1])
    
    # Verify .X is from basal cells
    np.testing.assert_array_equal(adata_cf_sampled.X, adata_with_spatial[indices_basal].X)
    
    # Verify .obs is from basal cells
    assert all(adata_cf_sampled.obs.index == adata_with_spatial[indices_basal].obs.index)
    
    # Test sample=False (row sampling with replacement)
    adata_cf_rows = make_counterfactual_adata(
        adata_with_spatial, indices_basal, indices_cf, spatial_col, sample=False, random_state=42
    )
    
    # Check shape is correct
    assert adata_cf_rows.obsm[spatial_col].shape == (len(indices_basal), adata_with_spatial.obsm[spatial_col].shape[1])
    
    # Verify each row comes from counterfactual cells (should match at least one row)
    cf_spatial = adata_with_spatial.obsm[spatial_col][indices_cf]
    for i in range(adata_cf_rows.n_obs):
        row = adata_cf_rows.obsm[spatial_col][i]
        # Check if this row exists in counterfactual spatial features
        matches = np.any(np.all(cf_spatial == row, axis=1))
        assert matches, f"Row {i} does not match any counterfactual spatial features"
    
    # Test reproducibility with same random_state
    adata_cf2 = make_counterfactual_adata(
        adata_with_spatial, indices_basal, indices_cf, spatial_col, sample=True, random_state=42
    )
    np.testing.assert_array_equal(adata_cf_sampled.obsm[spatial_col], adata_cf2.obsm[spatial_col])


def test_normalize_losses_true():
    """Test normalize_losses parameter in adversarial training plan."""
    # Create synthetic data with domain labels for adversarial training
    adata = synthetic_iid()
    n_spatial_features = 20
    adata.obsm["spatial_x"] = np.random.randn(adata.n_obs, n_spatial_features).astype(np.float32)
    
    n_domains = 3
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)
    
    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        spatial_obsm_key="spatial_x",
        domains_key="domain"
    )
    
    # Create model with discriminator enabled
    model = CellinaModel(adata, n_latent=5, discriminator_lambda=1.0, classifier_lambda=0.0)
    
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
    
    # Check EMA values were initialized (should be positive)
    assert training_plan._ema["vae"] > 0, "EMA for vae loss should be positive"
    assert training_plan._ema["clf"] >= 0, "EMA for clf loss should be non-negative"
    assert training_plan._ema["fool"] >= 0, "EMA for fool loss should be non-negative"
    
    # Check normalize_losses flag is set correctly
    assert training_plan._normalize_losses == True
    
    # Verify training completed successfully
    # Note: warmup epoch (epoch 0) is not logged in history, so we expect 1 entry for epoch 1
    assert len(model.history_["train_loss"]) >= 1
    
    # Verify discriminator metrics are logged
    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"


def test_get_normalized_expression(adata_with_spatial):
    """Test get_normalized_expression method returns correct shapes."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
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
