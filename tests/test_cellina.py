import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid

from cellina import CellinaModel


@pytest.fixture
def adata_with_spatial():
    """Create synthetic AnnData with spatial features."""
    adata = synthetic_iid()
    n_spatial_features = 20
    adata.obsm["spatial_x"] = np.random.randn(adata.n_obs, n_spatial_features).astype(np.float32)
    return adata


def test_cellina_model(adata_with_spatial):
    """Test basic CellinaModel functionality."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
    
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
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
    
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


def test_cellina_dual_kl_divergence(adata_with_spatial):
    """Test that loss includes KL divergence for both z and s."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
    
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


def test_cellina_latent_representation(adata_with_spatial):
    """Test latent representation returns correct shapes and uses latent_key."""
    n_latent = 5
    
    CellinaModel.setup_anndata(adata_with_spatial, batch_key="batch", spatial_obsm_key="spatial_x")
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
    model.train(max_epochs=1, check_val_every_n_epoch=1, train_size=0.5)
    
    # Test separate representations
    latent_z = model.get_latent_representation(latent_key='z')
    latent_s = model.get_latent_representation(latent_key='s')
    latent_shifted = model.get_latent_representation(latent_key='shifted')
    
    assert latent_z.shape == (adata_with_spatial.n_obs, n_latent)
    assert latent_s.shape == (adata_with_spatial.n_obs, n_latent)
    assert latent_shifted.shape == (adata_with_spatial.n_obs, n_latent)
    
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
    """Test weighted_pseudobulks function."""
    from cellina._spatial_utils import weighted_pseudobulks, spatial_neighbors
    from scipy.sparse import csr_matrix
    
    # Setup spatial data
    n_obs = adata_with_spatial.n_obs
    adata_with_spatial.obsm['spatial'] = np.random.rand(n_obs, 2) * 100
    
    # Add cell type labels
    cell_types = ['TypeA', 'TypeB', 'TypeC']
    adata_with_spatial.obs['cell_type'] = np.random.choice(cell_types, n_obs)
    
    # Create spatial connectivity matrix
    sp = spatial_neighbors(
        adata_with_spatial,
        bandwidth=50.0,
        cutoff=0.1,
        max_neighbours=10,
        kernel='gaussian',
        spatial_key='spatial',
        inplace=False
    )
    
    # Test weighted pseudobulks
    weighted_pseudobulks(
        adata_with_spatial,
        sp=sp,
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
    
    # Check that spatial_var was added to uns
    assert '_spatial_var' in adata_with_spatial.uns
