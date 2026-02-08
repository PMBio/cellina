import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid

from cellina import CellinaModel


def _add_spatial_connectivity(adata, max_neighbors=5):
    """Add a random spatial connectivity matrix to adata."""
    n_obs = adata.n_obs
    rows, cols = [], []
    for i in range(n_obs):
        n_neigh = min(max_neighbors, n_obs - 1)
        neighbors = np.random.choice(
            [j for j in range(n_obs) if j != i], size=n_neigh, replace=False
        )
        for j in neighbors:
            rows.append(i)
            cols.append(j)
    connectivity = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_obs, n_obs)
    )
    # Make symmetric
    connectivity = connectivity + connectivity.T
    connectivity.data[:] = 1.0
    connectivity.eliminate_zeros()
    adata.obsp["spatial_connectivities"] = connectivity


@pytest.fixture
def adata_with_spatial():
    """Create synthetic AnnData with spatial connectivity."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)
    n_labels = 3
    adata.obs["cell_labels"] = np.random.randint(0, n_labels, size=adata.n_obs).astype(str)
    return adata


def test_cellina_model(adata_with_spatial):
    """Test basic CellinaModel functionality."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0, discriminator_lambda=0.0)

    assert model.module.n_latent == n_latent

    model.train(max_epochs=1, check_val_every_n_epoch=1, train_size=0.5)
    model.history

    print(model)


def test_cellina_s_encoder_architecture(adata_with_spatial):
    """Test that s_encoder is a GCN (GraphEncoder)."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)

    # s_encoder should be a GraphEncoder with GCN layers
    from cellina._spatial_encoder import GraphEncoder
    assert isinstance(model.module.s_encoder, GraphEncoder)
    assert hasattr(model.module.s_encoder.encoder, 'gcn_layers')

    # Test forward pass produces correct outputs
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    inference_inputs = model.module._get_inference_input(batch)

    # Verify edge_index is in inference inputs (for GCN message passing)
    assert "edge_index" in inference_inputs

    inference_outputs = model.module.inference(**inference_inputs)

    # Verify z and s have correct shapes
    assert inference_outputs["z"].shape[1] == n_latent
    assert inference_outputs["s"].shape[1] == n_latent
    assert all(k in inference_outputs for k in ["z", "s", "qzm", "qzv", "qsm", "qsv"])


def test_cellina_losses(adata_with_spatial):
    """Test that loss includes KL divergence for both z and s, and classifier loss when enabled."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        spatial_connectivities_key="spatial_connectivities",
    )
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

    vae_loss = loss_output.extra_metrics["vae_loss"]
    assert vae_loss > 0

    assert loss_output.extra_metrics["classifier_loss"] > 0

    # Check we can compute accuracy
    classifier_logits = inference_outputs["classifier_logits"]
    labels = batch['node_batch'][REGISTRY_KEYS.LABELS_KEY]
    batch_size = batch['node_batch']['batch_size']
    labels = labels[:batch_size].reshape(-1).long()
    predictions = torch.argmax(classifier_logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    assert 0 <= accuracy <= 1


def test_classifier_disabled_by_default():
    """Test that classifier is disabled when classifier_lambda=0."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)

    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata, n_latent=5, classifier_lambda=0.0)
    assert model.module.classifier is None
    assert model.module.classifier_lambda == 0.0


def test_discriminator_disabled_by_default():
    """Test that discriminator is disabled when discriminator_lambda=0 (default)."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)

    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata, n_latent=5)
    assert model.module.domain_discriminator is None
    assert model.module.discriminator_lambda == 0.0

    model2 = CellinaModel(adata, n_latent=5, discriminator_lambda=0.0)
    assert model2.module.domain_discriminator is None
    assert model2.module.discriminator_lambda == 0.0


def test_discriminator_enabled():
    """Test that discriminator works when discriminator_lambda > 0."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)

    n_domains = 3
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)

    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    n_latent = 5
    model = CellinaModel(adata, n_latent=n_latent, discriminator_lambda=1.0)

    assert model.module.domain_discriminator is not None
    assert model.module.discriminator_lambda == 1.0

    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)

    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"

    # Verify inference outputs include discriminator logits
    model.module.eval()
    dataloader = model._make_data_loader(adata, batch_size=10)
    batch = next(iter(dataloader))

    with torch.no_grad():
        inference_inputs = model.module._get_inference_input(batch)
        outputs = model.module.inference(**inference_inputs)

    assert "discriminator_logits" in outputs
    assert outputs["discriminator_logits"].shape[1] == n_domains


def test_cellina_latent_representation(adata_with_spatial):
    """Test latent representation returns correct shapes and uses latent_key."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)
    model.train(max_epochs=1, check_val_every_n_epoch=1, train_size=0.5)

    latent_z = model.get_latent_representation(latent_key='z')
    latent_s = model.get_latent_representation(latent_key='s')
    latent_shifted = model.get_latent_representation(latent_key='shifted')

    assert latent_z.shape == (adata_with_spatial.n_obs, n_latent)
    assert latent_s.shape == (adata_with_spatial.n_obs, n_latent)
    assert latent_shifted.shape == (adata_with_spatial.n_obs, n_latent * 2)

    latent_default = model.get_latent_representation()
    assert latent_default.shape == latent_shifted.shape

    with pytest.raises(ValueError, match="latent_key must be"):
        model.get_latent_representation(latent_key='invalid')


def test_spatial_neighbors(adata_with_spatial):
    """Test spatial_neighbors function."""
    from cellina._spatial_utils import spatial_neighbors
    from scipy.sparse import issparse

    n_obs = adata_with_spatial.n_obs
    adata_with_spatial.obsm['spatial'] = np.random.rand(n_obs, 2) * 100

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

    assert 'spatial_connectivities' in adata_with_spatial.obsp
    assert issparse(adata_with_spatial.obsp['spatial_connectivities'])
    assert adata_with_spatial.obsp['spatial_connectivities'].shape == (n_obs, n_obs)

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

    n_obs = adata_with_spatial.n_obs
    adata_with_spatial.obsm['spatial'] = np.random.rand(n_obs, 2) * 100

    cell_types = ['TypeA', 'TypeB', 'TypeC']
    adata_with_spatial.obs['cell_type'] = np.random.choice(cell_types, n_obs)

    sp_mat = spatial_neighbors(
        adata_with_spatial,
        bandwidth=50.0,
        cutoff=0.1,
        max_neighbours=10,
        kernel='gaussian',
        spatial_key='spatial',
        inplace=False
    )

    weighted_pseudobulks(
        adata_with_spatial,
        sp=sp_mat,
        groupby='cell_type',
        obsm_key='spatial_pseudobulks',
        binarize=True
    )

    assert 'spatial_pseudobulks' in adata_with_spatial.obsm

    n_genes = adata_with_spatial.n_vars
    n_cell_types = len(cell_types)
    assert adata_with_spatial.obsm['spatial_pseudobulks'].shape == (n_obs, n_cell_types * n_genes)

    assert '_spatial_var' in adata_with_spatial.uns


def test_marginal_ll(adata_with_spatial):
    """Test get_marginal_ll method and underlying module.marginal_ll."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, classifier_lambda=0.0)
    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)

    marginal_ll_list = model.get_marginal_ll(n_mc_samples=10)
    assert isinstance(marginal_ll_list, list)
    assert len(marginal_ll_list) > 0
    assert all(isinstance(ll, float) and np.isfinite(ll) for ll in marginal_ll_list)

    marginal_ll_mean = model.get_marginal_ll(n_mc_samples=10, reduce='mean')
    assert isinstance(marginal_ll_mean, (float, np.floating))

    marginal_ll_sum = model.get_marginal_ll(n_mc_samples=10, reduce='sum')
    assert isinstance(marginal_ll_sum, (float, np.floating))

    with pytest.raises(ValueError, match="Reduction must be None, 'mean' or 'sum'"):
        model.get_marginal_ll(reduce='invalid')

    # Test underlying module.marginal_ll
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    model.module.eval()
    with torch.no_grad():
        log_lkl = model.module.marginal_ll(batch, n_mc_samples=10)
    assert isinstance(log_lkl, float) and np.isfinite(log_lkl)


def test_condition_on_intrinsic_false(adata_with_spatial):
    """Test s_encoder architecture changes with condition_on_intrinsic=False."""
    n_latent = 5
    n_vars = adata_with_spatial.n_vars

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        spatial_connectivities_key="spatial_connectivities",
    )

    # Test with condition_on_intrinsic=True (default)
    model_true = CellinaModel(adata_with_spatial, n_latent=n_latent, condition_on_intrinsic=True)
    assert model_true.module.condition_on_intrinsic == True

    # Test with condition_on_intrinsic=False
    model_false = CellinaModel(adata_with_spatial, n_latent=n_latent, condition_on_intrinsic=False)
    assert model_false.module.condition_on_intrinsic == False

    # Check GCN input dimensions differ by n_latent
    # First GCN layer input size includes covariates (batch injection)
    first_gcn_true = model_true.module.s_encoder.encoder.gcn_layers[0]
    first_gcn_false = model_false.module.s_encoder.encoder.gcn_layers[0]
    input_dim_true = first_gcn_true.in_channels
    input_dim_false = first_gcn_false.in_channels

    # The difference in base input should be n_latent
    # (both have the same covariate injection, so the difference is purely from n_input_s)
    n_cov = model_true.module.s_encoder.encoder.n_cov
    assert (input_dim_true - n_cov) - (input_dim_false - n_cov) == n_latent, \
        f"Difference in input dims should be n_latent={n_latent}, got {(input_dim_true - n_cov) - (input_dim_false - n_cov)}"

    # Test training works with condition_on_intrinsic=False
    model_false.train(max_epochs=2, train_size=0.5)

    # Test inference outputs have correct shapes
    dataloader = model_false._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    model_false.module.eval()
    with torch.no_grad():
        inference_outputs = model_false.module.inference(**model_false.module._get_inference_input(batch))
    assert inference_outputs["z"].shape[1] == n_latent
    assert inference_outputs["s"].shape[1] == n_latent


def test_normalize_losses_true():
    """Test normalize_losses parameter in adversarial training plan."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)

    n_domains = 3
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)

    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    model = CellinaModel(adata, n_latent=5, discriminator_lambda=1.0, classifier_lambda=0.0)

    model.train(
        max_epochs=2,
        train_size=0.5,
        plan_kwargs={"normalize_losses": True}
    )

    training_plan = model.trainer.strategy.model

    assert training_plan._warmup_done == True
    assert training_plan._ema["vae"] > 0
    assert training_plan._ema["clf"] >= 0
    assert training_plan._ema["fool"] >= 0
    assert training_plan._normalize_losses == True

    assert len(model.history_["train_loss"]) >= 1

    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"


def test_edge_prediction_loss():
    """Test edge prediction functionality when link_prediction_weight > 0."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)

    n_labels = 3
    n_domains = 2
    adata.obs["cell_type"] = np.random.randint(0, n_labels, size=adata.n_obs).astype(str)
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)

    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="cell_type",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    n_latent = 5
    link_prediction_weight = 0.1
    model = CellinaModel(
        adata,
        n_latent=n_latent,
        link_prediction_weight=link_prediction_weight,
        classifier_lambda=1.0,
        discriminator_lambda=1.0,
    )

    assert model.module.link_prediction_weight == link_prediction_weight
    assert model.module.link_prediction_weight > 0

    from cellina._edge_data_splitter import GraphJointDataSplitter
    assert hasattr(model, '_data_splitter_cls')
    assert issubclass(model._data_splitter_cls, GraphJointDataSplitter)

    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)

    history_keys = list(model.history_.keys())
    edge_loss_keys = [k for k in history_keys if "edge_prediction_loss" in k]
    assert len(edge_loss_keys) > 0, f"No edge prediction metrics found in history. Keys: {history_keys}"

    edge_train_loss = model.history_["edge_prediction_loss_train"].iloc[-1].item()
    assert edge_train_loss >= 0, f"Edge prediction train loss should be non-negative, got {edge_train_loss}"

    edge_val_loss = model.history_["edge_prediction_loss_validation"].iloc[-1].item()
    assert edge_val_loss >= 0, f"Edge prediction validation loss should be non-negative, got {edge_val_loss}"


def test_edge_prediction_disabled_by_default():
    """Test that edge prediction is disabled when link_prediction_weight=0 (default)."""
    adata = synthetic_iid()
    _add_spatial_connectivity(adata)

    CellinaModel.setup_anndata(
        adata,
        batch_key="batch",
        spatial_connectivities_key="spatial_connectivities",
    )

    model = CellinaModel(adata, n_latent=5)
    assert model.module.link_prediction_weight == 0.0

    # Always uses GraphJointDataSplitter now (GCN needs graph)
    from cellina._edge_data_splitter import GraphJointDataSplitter
    assert issubclass(model._data_splitter_cls, GraphJointDataSplitter)

    # But use_edge_prediction should be False
    assert model._data_splitter_kwargs['use_edge_prediction'] == False

    model2 = CellinaModel(adata, n_latent=5, link_prediction_weight=0.0)
    assert model2.module.link_prediction_weight == 0.0
