import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid

from cellina_graph import CellinaModel
from cellina_graph._cellina_module import CellinaModule


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
    n_domains = 3
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)
    return adata


def test_cellina_model(adata_with_spatial):
    """Test basic CellinaModel functionality."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)

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
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)

    # s_encoder should be a GraphEncoder with GCN layers
    from cellina_graph._spatial_encoder import GraphEncoder
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
        domains_key="domain",
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


def test_classifier_disabled(adata_with_spatial):
    """Test that classifier is disabled when classifier_lambda=0."""
    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=5, classifier_lambda=0.0)
    assert model.module.classifier is None
    assert model.module.classifier_lambda == 0.0


def test_discriminator_disabled_by(adata_with_spatial):
    """Test that discriminator is disabled when discriminator_lambda=0 (default)."""

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=5, discriminator_lambda=0.0)
    assert model.module.domain_discriminator is None
    assert model.module.discriminator_lambda == 0.0

    model2 = CellinaModel(adata_with_spatial, n_latent=5, discriminator_lambda=0.0)
    assert model2.module.domain_discriminator is None
    assert model2.module.discriminator_lambda == 0.0


def test_discriminator_enabled(adata_with_spatial):
    """Test that discriminator works when discriminator_lambda > 0."""

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    n_latent = 5
    model = CellinaModel(adata_with_spatial, n_latent=n_latent, discriminator_lambda=1.0, classifier_lambda=0.0)

    assert model.module.domain_discriminator is not None
    assert model.module.discriminator_lambda == 1.0

    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)

    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"

    # Verify inference outputs include discriminator logits
    model.module.eval()
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=10)
    batch = next(iter(dataloader))

    with torch.no_grad():
        inference_inputs = model.module._get_inference_input(batch)
        outputs = model.module.inference(**inference_inputs)

    assert "discriminator_logits" in outputs
    assert outputs["discriminator_logits"].shape[1] == adata_with_spatial.obs["domain"].nunique()


def test_cellina_latent_representation(adata_with_spatial):
    """Test latent representation returns correct shapes and uses latent_key."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
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
    from cellina_graph._spatial_utils import spatial_neighbors
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


def test_marginal_ll(adata_with_spatial):
    """Test get_marginal_ll method and underlying module.marginal_ll."""
    n_latent = 5

    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=n_latent)
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
        labels_key="cell_labels",
        domains_key="domain",
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



def test_normalize_losses_true(adata_with_spatial):
    """Test normalize_losses parameter in adversarial training plan."""
    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    # Create model with both discriminator and classifier enabled (non-unity lambdas)
    classifier_lambda    = 0.5
    discriminator_lambda = 2.0
    model = CellinaModel(adata_with_spatial, n_latent=5,
                         discriminator_lambda=discriminator_lambda,
                         classifier_lambda=classifier_lambda)

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
    assert training_plan._scale_clf  > 0, "Fixed scale for clf loss should be positive"
    assert training_plan._scale_fool > 0, "Fixed scale for fool loss should be positive"
    assert training_plan._scale_spatial > 0, "Fixed scale for spatial loss should be positive"

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
    expected_scale_clf  = training_plan._scale_clf
    expected_scale_fool = training_plan._scale_fool

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
            discriminator_lambda=discriminator_lambda,  # module negates fool_ce internally
            classifier_scale=expected_scale_clf,
            discriminator_scale=expected_scale_fool,
        )

    clf_raw    = loss_out.extra_metrics["classifier_loss_raw"].item()
    clf_scaled = loss_out.extra_metrics["classifier_loss"].item()
    fool_raw   = loss_out.extra_metrics["fool_loss_raw"].item()
    fool_scaled = loss_out.extra_metrics["fool_loss"].item()

    np.testing.assert_allclose(clf_scaled,  clf_raw  * expected_scale_clf  * classifier_lambda,   rtol=1e-4)
    np.testing.assert_allclose(fool_scaled, fool_raw * expected_scale_fool * discriminator_lambda, rtol=1e-4)
    # make sure that disc is roughly 4x scaled compared to clf (since discriminator_lambda is 4x classifier_lambda)
    # fool_scaled is negative (adversarial weight=-1), so compare absolute magnitudes
    np.testing.assert_allclose(abs(fool_scaled / clf_scaled), discriminator_lambda / classifier_lambda, rtol=0.2)
    # assert fool is negative (since it's an adversarial loss)
    assert fool_scaled < 0, "Fool loss should be negative (adversarial weight is -1)"


# ── SupCon unit tests ────────────────────────────────────────────────────────

def _supcon_module():
    return CellinaModule(
        n_input=10,
        library_log_means=np.zeros((1, 1)),
        library_log_vars=np.ones((1, 1)),
    )


def test_supcon_loss():
    """_compute_supcon_loss: positive, no-negatives, and no-edges cases."""
    module = _supcon_module()
    n_latent = 5

    def _call(qsm, nbr, edge_index, domains_all):
        return module._compute_supcon_loss(
            qsm=qsm, neighbor_means=nbr, edge_index=edge_index,
            domains_all=domains_all, batch_size=len(qsm), temperature=0.1,
        )

    # ── case 1: valid pairs → loss > 0
    batch_size, n_nbr = 4, 4
    qsm = torch.randn(batch_size, n_latent)
    nbr = torch.randn(n_nbr, n_latent)
    src = torch.arange(batch_size)
    dst = torch.arange(batch_size, batch_size + n_nbr)
    ei  = torch.stack([src, dst])
    # seeds 0,1 + nbrs 0,1 → domain 0; seeds 2,3 + nbrs 2,3 → domain 1
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    loss = _call(qsm, nbr, ei, domains)
    assert loss.shape == torch.Size([]) and loss.item() > 0.0

    # ── case 2: all same domain → no negatives → loss == 0
    domains_same = torch.zeros(batch_size + n_nbr, dtype=torch.long)
    assert _call(qsm, nbr, ei, domains_same).item() == 0.0

    # ── case 3: no edges → no neighbours → loss == 0
    assert _call(
        torch.randn(2, n_latent),
        torch.zeros(0, n_latent),
        torch.zeros(2, 0, dtype=torch.long),
        torch.tensor([0, 1]),
    ).item() == 0.0


def test_supcon_model(adata_with_spatial):
    """spatial_loss_raw > 0 when link_prediction_weight > 0 and domains differ (supcon)."""
    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(adata_with_spatial, n_latent=5, link_prediction_weight=1.0)

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    inference_outputs  = model.module.inference(**model.module._get_inference_input(batch))
    generative_outputs = model.module.generative(**model.module._get_generative_input(batch, inference_outputs))
    loss_output = model.module.loss(batch, inference_outputs, generative_outputs)

    assert "spatial_loss_raw" in loss_output.extra_metrics
    assert loss_output.extra_metrics["spatial_loss_raw"] > 0


def test_domain_clf_model(adata_with_spatial):
    """spatial_loss_raw > 0 when spatial_loss_type='domain_clf' and link_prediction_weight > 0."""
    CellinaModel.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaModel(
        adata_with_spatial, n_latent=5,
        link_prediction_weight=1.0,
        spatial_loss_type="domain_clf",
    )

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    inference_outputs  = model.module.inference(**model.module._get_inference_input(batch))
    generative_outputs = model.module.generative(**model.module._get_generative_input(batch, inference_outputs))
    loss_output = model.module.loss(batch, inference_outputs, generative_outputs)

    assert "spatial_loss_raw" in loss_output.extra_metrics
    assert loss_output.extra_metrics["spatial_loss_raw"] > 0
    assert "s_domain_accuracy" in loss_output.extra_metrics


def test_normalize_losses_spatial_scale(adata_with_spatial):
    """spatial_loss is scaled by _scale_spatial when normalize_losses=True, for both loss types."""
    for spatial_loss_type in ("supcon", "domain_clf"):
        CellinaModel.setup_anndata(
            adata_with_spatial,
            batch_key="batch",
            labels_key="cell_labels",
            domains_key="domain",
            spatial_connectivities_key="spatial_connectivities",
        )
        link_prediction_weight = 1.0
        model = CellinaModel(
            adata_with_spatial, n_latent=5,
            discriminator_lambda=1.0,
            link_prediction_weight=link_prediction_weight,
            spatial_loss_type=spatial_loss_type,
        )
        model.train(max_epochs=2, train_size=0.5, plan_kwargs={"normalize_losses": True})

        training_plan = model.trainer.strategy.model
        assert training_plan._warmup_done
        assert training_plan._scale_spatial > 0, \
            f"_scale_spatial should be positive for spatial_loss_type={spatial_loss_type!r}"

        expected_scale = training_plan._scale_spatial
        model.module.eval()
        model.module.to("cpu")
        dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
        batch = next(iter(dataloader))

        with torch.no_grad():
            inf_out = model.module.inference(**model.module._get_inference_input(batch))
            gen_out = model.module.generative(**model.module._get_generative_input(batch, inf_out))
            loss_out = model.module.loss(
                batch, inf_out, gen_out,
                spatial_scale=expected_scale,
            )

        spatial_raw    = loss_out.extra_metrics["spatial_loss_raw"].item()
        spatial_scaled = loss_out.extra_metrics["spatial_loss"].item()
        np.testing.assert_allclose(
            spatial_scaled, spatial_raw * expected_scale * link_prediction_weight, rtol=1e-4,
            err_msg=f"Scale mismatch for spatial_loss_type={spatial_loss_type!r}",
        )
