import numpy as np
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid

from cellina import CellinaGCN, make_perturbed_expression
from cellina._cellina_gcn_module import CellinaGCNModule


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
    connectivity = connectivity + connectivity.T
    connectivity.data[:] = 1.0
    connectivity.eliminate_zeros()
    adata.obsp["spatial_connectivities"] = connectivity


@pytest.fixture
def adata_with_spatial():
    """Create synthetic AnnData with spatial connectivity."""
    adata = synthetic_iid()
    adata.X = sp.csr_matrix(adata.X)
    _add_spatial_connectivity(adata)
    n_labels = 3
    adata.obs["cell_labels"] = np.random.randint(0, n_labels, size=adata.n_obs).astype(str)
    n_domains = 3
    adata.obs["domain"] = np.random.randint(0, n_domains, size=adata.n_obs).astype(str)
    return adata


def test_cellina_model(adata_with_spatial):
    """Test basic CellinaGCN functionality."""
    n_latent = 5

    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=n_latent)

    assert model.module.n_latent == n_latent

    model.train(max_epochs=1, check_val_every_n_epoch=1, train_size=0.5)
    model.history

    print(model)


def test_num_neighbors_resolution(adata_with_spatial):
    """num_neighbors contract: None -> [-1]*n_layers; mismatched length warns and is used as-is."""
    import warnings

    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    # Default: None -> [-1] * n_layers, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model_default = CellinaGCN(adata_with_spatial, n_latent=5, n_layers=3)
    assert model_default._num_neighbors == [-1, -1, -1]

    # Length 1 != n_layers -> warns, used as-is (no broadcast).
    with pytest.warns(UserWarning):
        model_short = CellinaGCN(adata_with_spatial, n_latent=5, n_layers=3, num_neighbors=[-1])
    assert model_short._num_neighbors == [-1]

    # Length == n_layers -> no warning, trains.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model_full = CellinaGCN(
            adata_with_spatial, n_latent=5, n_layers=3, num_neighbors=[-1, -1, -1]
        )
    assert model_full._num_neighbors == [-1, -1, -1]
    model_full.train(max_epochs=1, train_size=0.5)


def test_cellina_s_encoder_architecture(adata_with_spatial):
    """Test that s_encoder is a GCN (GraphEncoder)."""
    n_latent = 5

    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=n_latent)

    from cellina._spatial_encoder import GraphEncoder
    assert isinstance(model.module.s_encoder, GraphEncoder)
    assert hasattr(model.module.s_encoder.encoder, 'gcn_layers')

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    inference_inputs = model.module._get_inference_input(batch)

    assert "edge_index" in inference_inputs

    inference_outputs = model.module.inference(**inference_inputs)

    assert inference_outputs["z"].shape[1] == n_latent
    assert inference_outputs["s"].shape[1] == n_latent
    assert all(k in inference_outputs for k in ["z", "s", "qzm", "qzv", "qsm", "qsv"])


def test_cellina_losses(adata_with_spatial):
    """Test that loss includes KL divergence for both z and s, and classifier loss when enabled."""
    n_latent = 5

    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=n_latent, classifier_lambda=1.0)

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    inference_inputs = model.module._get_inference_input(batch)
    inference_outputs = model.module.inference(**inference_inputs)
    generative_inputs = model.module._get_generative_input(batch, inference_outputs)
    generative_outputs = model.module.generative(**generative_inputs)
    loss_output = model.module.loss(batch, inference_outputs, generative_outputs)

    assert "kl_divergence_z" in loss_output.kl_local
    assert "kl_divergence_s" in loss_output.kl_local
    assert "kl_divergence_l" in loss_output.kl_local

    assert "vae_loss" in loss_output.extra_metrics
    assert "classifier_loss" in loss_output.extra_metrics
    assert "fool_loss" in loss_output.extra_metrics

    vae_loss = loss_output.extra_metrics["vae_loss"]
    assert vae_loss > 0

    assert loss_output.extra_metrics["classifier_loss"] > 0

    classifier_logits = inference_outputs["classifier_logits"]
    labels = batch['node_batch'][REGISTRY_KEYS.LABELS_KEY]
    batch_size = batch['node_batch']['batch_size']
    labels = labels[:batch_size].reshape(-1).long()
    predictions = torch.argmax(classifier_logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    assert 0 <= accuracy <= 1


def test_classifier_disabled(adata_with_spatial):
    """Test that classifier is disabled when classifier_lambda=0."""
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=5, classifier_lambda=0.0)
    assert model.module.classifier is None
    assert model.module.classifier_lambda == 0.0


def test_discriminator_disabled_by(adata_with_spatial):
    """Test that discriminator is disabled when discriminator_lambda=0."""
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=5, discriminator_lambda=0.0)
    assert model.module.domain_discriminator is None
    assert model.module.discriminator_lambda == 0.0

    model2 = CellinaGCN(adata_with_spatial, n_latent=5, discriminator_lambda=0.0)
    assert model2.module.domain_discriminator is None
    assert model2.module.discriminator_lambda == 0.0


def test_discriminator_enabled(adata_with_spatial):
    """Test that discriminator works when discriminator_lambda > 0."""
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    n_latent = 5
    model = CellinaGCN(adata_with_spatial, n_latent=n_latent, discriminator_lambda=1.0, classifier_lambda=0.0)

    assert model.module.domain_discriminator is not None
    assert model.module.discriminator_lambda == 1.0

    model.train(max_epochs=2, check_val_every_n_epoch=1, train_size=0.5)

    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"

    model.module.eval()
    dataloader = model._make_data_loader(adata_with_spatial, batch_size=10)
    batch = next(iter(dataloader))

    with torch.no_grad():
        inference_inputs = model.module._get_inference_input(batch)
        outputs = model.module.inference(**inference_inputs)

    assert "discriminator_logits" in outputs
    assert outputs["discriminator_logits"].shape[1] == adata_with_spatial.obs["domain"].nunique()


def test_cellina_latent_representation(trained_model):
    """Test latent representation returns correct shapes and uses latent_key."""
    model, adata = trained_model
    n_latent = model.module.n_latent

    latent_z = model.get_latent_representation(latent_key='z')
    latent_s = model.get_latent_representation(latent_key='s')
    latent_shifted = model.get_latent_representation(latent_key='shifted')

    assert latent_z.shape == (adata.n_obs, n_latent)
    assert latent_s.shape == (adata.n_obs, n_latent)
    assert latent_shifted.shape == (adata.n_obs, n_latent * 2)

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


def test_spatial_neighbors_test_indices(adata_with_spatial):
    """Test that test_indices cells have zero connectivity (no edges in or out)."""
    from cellina._spatial_utils import spatial_neighbors
    import numpy as np

    n_obs = adata_with_spatial.n_obs
    adata_with_spatial.obsm['spatial'] = np.random.rand(n_obs, 2) * 100

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
    assert conn_dense[:, test_idx].sum() == 0, "test cells appear as neighbors of other cells"
    assert conn_dense[test_idx, :].sum() == 0, "test cells have outgoing edges"
    non_test = [i for i in range(n_obs) if i not in test_idx]
    assert conn_dense[non_test, :].sum() > 0, "non-test cells lost all connectivity"


def test_marginal_ll(trained_model):
    """Test get_marginal_ll method and underlying module.marginal_ll."""
    model, adata = trained_model

    marginal_ll_arr = model.get_marginal_ll(n_mc_samples=10, return_mean=False)
    assert isinstance(marginal_ll_arr, np.ndarray)
    assert marginal_ll_arr.ndim == 1
    assert len(marginal_ll_arr) == adata.n_obs
    assert np.all(np.isfinite(marginal_ll_arr))

    marginal_ll_mean = model.get_marginal_ll(n_mc_samples=10, return_mean=True)
    assert isinstance(marginal_ll_mean, float)
    assert np.isfinite(marginal_ll_mean)

    dataloader = model._make_data_loader(adata, batch_size=32)
    batch = next(iter(dataloader))
    with torch.no_grad():
        log_lkl = model.module.marginal_ll(batch, n_mc_samples=10)
    assert isinstance(log_lkl, torch.Tensor)
    assert log_lkl.ndim == 1
    assert torch.all(torch.isfinite(log_lkl))


def test_seed_sliced_inference(adata_with_spatial):
    """z/s are seed-sized and the s_encoder input is the raw gene dimension."""
    n_latent = 5

    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    model = CellinaGCN(adata_with_spatial, n_latent=n_latent)
    first_gcn = model.module.s_encoder.encoder.gcn_layers[0]
    assert first_gcn.in_channels == adata_with_spatial.n_vars

    model.train(max_epochs=2, train_size=0.5)

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))
    batch_size = batch["node_batch"]["batch_size"]
    assert batch["node_batch"]["X"].shape[0] >= batch_size  # subgraph includes neighbours
    model.module.eval()
    with torch.no_grad():
        inference_outputs = model.module.inference(**model.module._get_inference_input(batch))
    assert inference_outputs["z"].shape == (batch_size, n_latent)
    assert inference_outputs["s"].shape == (batch_size, n_latent)
    assert inference_outputs["shifted"].shape == (batch_size, 2 * n_latent)


def test_normalize_losses_true(adata_with_spatial):
    """Test normalize_losses parameter in adversarial training plan."""
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )

    classifier_lambda    = 0.5
    discriminator_lambda = 2.0
    model = CellinaGCN(adata_with_spatial, n_latent=5,
                         discriminator_lambda=discriminator_lambda,
                         classifier_lambda=classifier_lambda)

    model.train(
        max_epochs=2,
        train_size=0.5,
        plan_kwargs={"normalize_losses": True}
    )

    training_plan = model.trainer.strategy.model

    assert training_plan._warmup_done == True, "Warmup should be completed after epoch 0"

    assert training_plan._scale_clf  > 0, "Fixed scale for clf loss should be positive"
    assert training_plan._scale_fool > 0, "Fixed scale for fool loss should be positive"
    assert training_plan._scale_spatial > 0, "Fixed scale for spatial loss should be positive"

    assert training_plan._normalize_losses == True

    assert len(model.history_["train_loss"]) >= 1

    history_keys = list(model.history_.keys())
    assert any("discriminator" in key for key in history_keys), \
        f"No discriminator metrics found in history. Keys: {history_keys}"

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
            discriminator_lambda=discriminator_lambda,
            classifier_scale=expected_scale_clf,
            discriminator_scale=expected_scale_fool,
        )

    clf_raw    = loss_out.extra_metrics["classifier_loss_raw"].item()
    clf_scaled = loss_out.extra_metrics["classifier_loss"].item()
    fool_raw   = loss_out.extra_metrics["fool_loss_raw"].item()
    fool_scaled = loss_out.extra_metrics["fool_loss"].item()

    np.testing.assert_allclose(clf_scaled,  clf_raw  * expected_scale_clf  * classifier_lambda,   rtol=1e-4)
    np.testing.assert_allclose(fool_scaled, fool_raw * expected_scale_fool * discriminator_lambda, rtol=1e-4)
    np.testing.assert_allclose(abs(fool_scaled / clf_scaled), discriminator_lambda / classifier_lambda, rtol=0.2)
    assert fool_scaled < 0, "Fool loss should be negative (adversarial weight is -1)"


# ── SupCon unit tests ─────────────────────────────────────────────────────────

def _supcon_module():
    return CellinaGCNModule(
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

    # case 1: valid pairs → loss > 0
    batch_size, n_nbr = 4, 4
    qsm = torch.randn(batch_size, n_latent)
    nbr = torch.randn(n_nbr, n_latent)
    src = torch.arange(batch_size)
    dst = torch.arange(batch_size, batch_size + n_nbr)
    ei  = torch.stack([src, dst])
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    loss = _call(qsm, nbr, ei, domains)
    assert loss.shape == torch.Size([]) and loss.item() > 0.0

    # case 2: all same domain → no negatives → loss == 0
    domains_same = torch.zeros(batch_size + n_nbr, dtype=torch.long)
    assert _call(qsm, nbr, ei, domains_same).item() == 0.0

    # case 3: no edges → no neighbours → loss == 0
    assert _call(
        torch.randn(2, n_latent),
        torch.zeros(0, n_latent),
        torch.zeros(2, 0, dtype=torch.long),
        torch.tensor([0, 1]),
    ).item() == 0.0


def _supcon_loss_reference(qsm, neighbor_means, edge_index, domains_all,
                           batch_size, temperature):
    """Original per-anchor loop, kept as the parity reference for the vectorized form."""
    batch_size = int(batch_size)
    s_all = torch.cat([qsm, neighbor_means], dim=0)
    s_all = torch.nn.functional.normalize(s_all, p=2, dim=1)

    src, dst = edge_index[0], edge_index[1]
    loss_total = torch.tensor(0.0, device=qsm.device)
    n_valid = 0
    for i in range(batch_size):
        neighbor_idx = dst[src == i]
        if len(neighbor_idx) == 0:
            continue
        pos_idx = neighbor_idx
        neighbor_set = torch.zeros(s_all.size(0), dtype=torch.bool, device=qsm.device)
        neighbor_set[neighbor_idx] = True
        neg_mask = (domains_all != domains_all[i]) & ~neighbor_set
        neg_mask[i] = False
        if neg_mask.sum() == 0:
            continue
        sim_i = (s_all[i].unsqueeze(0) * s_all).sum(dim=-1) / temperature
        sim_i[i] = float('-inf')
        denom_mask = neighbor_set | neg_mask
        denom_mask[i] = False
        log_denom = torch.logsumexp(sim_i[denom_mask], dim=0)
        log_pos = sim_i[pos_idx] - log_denom
        loss_total = loss_total + (-log_pos.mean())
        n_valid += 1
    if n_valid == 0:
        return torch.tensor(0.0, device=qsm.device)
    return loss_total / n_valid


def test_supcon_loss_vectorized_parity():
    """Vectorized _compute_supcon_loss matches the reference loop in value and gradient."""
    module = _supcon_module()
    n_latent = 6
    temperature = 0.2

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # A few randomized subgraphs, plus the two degenerate cases (no negatives, no edges).
    cases = []
    for batch_size, n_nbr, n_domains in [(8, 12, 3), (5, 20, 2), (16, 30, 4), (3, 0, 2)]:
        n_all = batch_size + n_nbr
        # random edges from seed anchors to any node (no self-loops)
        n_edges = max(0, n_nbr)
        if n_edges > 0:
            edge_src = rng.integers(0, batch_size, size=n_edges)
            edge_dst = rng.integers(0, n_all, size=n_edges)
            # No self-loops and unique (src, dst) pairs — the real spatial graph is a
            # simple graph; the boolean-mask vectorization assumes edge uniqueness.
            keep = edge_src != edge_dst
            pairs = np.unique(np.stack([edge_src[keep], edge_dst[keep]], axis=1), axis=0)
            ei = torch.from_numpy(pairs.T).long()
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        domains = torch.from_numpy(rng.integers(0, n_domains, size=n_all)).long()
        cases.append((batch_size, n_nbr, ei, domains))
    # explicit all-same-domain case (no negatives anywhere → loss 0)
    cases.append((4, 4, torch.stack([torch.arange(4), torch.arange(4, 8)]),
                  torch.zeros(8, dtype=torch.long)))

    for batch_size, n_nbr, ei, domains in cases:
        base_qsm = torch.randn(batch_size, n_latent)
        base_nbr = torch.randn(n_nbr, n_latent)

        qsm_v = base_qsm.clone().requires_grad_(True)
        nbr_v = base_nbr.clone().requires_grad_(True)
        out_v = module._compute_supcon_loss(
            qsm=qsm_v, neighbor_means=nbr_v, edge_index=ei,
            domains_all=domains, batch_size=batch_size, temperature=temperature,
        )

        qsm_r = base_qsm.clone().requires_grad_(True)
        nbr_r = base_nbr.clone().requires_grad_(True)
        out_r = _supcon_loss_reference(
            qsm=qsm_r, neighbor_means=nbr_r, edge_index=ei,
            domains_all=domains, batch_size=batch_size, temperature=temperature,
        )

        assert torch.allclose(out_v, out_r, atol=1e-5), (out_v.item(), out_r.item())

        if out_r.requires_grad and out_r.item() != 0.0:
            out_v.backward()
            out_r.backward()
            assert torch.allclose(qsm_v.grad, qsm_r.grad, atol=1e-5)
            assert torch.allclose(nbr_v.grad, nbr_r.grad, atol=1e-5)


def test_supcon_model(adata_with_spatial):
    """spatial_loss_raw > 0 when link_prediction_weight > 0 and domains differ (supcon)."""
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=5, link_prediction_weight=1.0)

    dataloader = model._make_data_loader(adata_with_spatial, batch_size=32)
    batch = next(iter(dataloader))

    inference_outputs  = model.module.inference(**model.module._get_inference_input(batch))
    generative_outputs = model.module.generative(**model.module._get_generative_input(batch, inference_outputs))
    loss_output = model.module.loss(batch, inference_outputs, generative_outputs)

    assert "spatial_loss_raw" in loss_output.extra_metrics
    assert loss_output.extra_metrics["spatial_loss_raw"] > 0


def test_domain_clf_model(adata_with_spatial):
    """spatial_loss_raw > 0 when spatial_loss_type='domain_clf' and link_prediction_weight > 0."""
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(
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
        CellinaGCN.setup_anndata(
            adata_with_spatial,
            batch_key="batch",
            labels_key="cell_labels",
            domains_key="domain",
            spatial_connectivities_key="spatial_connectivities",
        )
        link_prediction_weight = 1.0
        model = CellinaGCN(
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


# ── Perturbation API ──────────────────────────────────────────────────────────

@pytest.fixture
def trained_model(adata_with_spatial):
    CellinaGCN.setup_anndata(
        adata_with_spatial,
        batch_key="batch",
        labels_key="cell_labels",
        domains_key="domain",
        spatial_connectivities_key="spatial_connectivities",
    )
    model = CellinaGCN(adata_with_spatial, n_latent=5, discriminator_lambda=0.0, classifier_lambda=0.0)
    model.train(max_epochs=1, train_size=0.8, check_val_every_n_epoch=1)
    return model, adata_with_spatial


def test_make_perturbed_expression():
    X = np.arange(1, 10, dtype=float).reshape(3, 3)
    adata = AnnData(X=X.copy())
    adata.var_names = ["G0", "G1", "G2"]

    # multiplicative with pseudocount (GCN path: add_shift=False, renormalize=False)
    make_perturbed_expression(adata, perturbations={"G0": 1.5, "G2": -0.5}, layer_key="cf", base=np.e,
                              add_shift=False, renormalize=False)
    expected = (X + 1).copy()
    expected[:, 0] *= np.e ** 1.5
    expected[:, 2] *= np.e ** -0.5
    expected -= 1.0
    np.testing.assert_allclose(np.asarray(adata.layers["cf"].toarray()), expected, rtol=1e-6)

    # additive shift (no renormalize so expected values are exact)
    make_perturbed_expression(adata, perturbations={"G0": 1.5, "G2": -0.5}, layer_key="cf_shift",
                              add_shift=True, renormalize=False)
    expected_shift = X.copy()
    expected_shift[:, 0] += 1.5
    expected_shift[:, 2] += -0.5
    np.testing.assert_allclose(np.asarray(adata.layers["cf_shift"]), expected_shift, rtol=1e-6)

    # renormalize: row sums after perturbation must match original row sums
    make_perturbed_expression(adata, perturbations={"G0": 1.5, "G2": -0.5}, layer_key="cf_renorm",
                              renormalize=True)
    row_sums_before = X.sum(axis=1)
    row_sums_after = np.asarray(adata.layers["cf_renorm"].sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums_after, row_sums_before, rtol=1e-6)


def test_perturbed_latents(trained_model):
    model, adata = trained_model
    adata.layers["cf_zero"] = sp.csr_matrix(np.zeros(adata.shape, dtype=np.float32))
    z_base = model.get_latent_representation(latent_key="z", give_mean=True)
    s_base = model.get_latent_representation(latent_key="s", give_mean=True)
    z_cf = model.get_perturbed_latents(cf_layer="cf_zero", latent_key="z", give_mean=True)
    s_cf = model.get_perturbed_latents(cf_layer="cf_zero", latent_key="s", give_mean=True)
    np.testing.assert_allclose(z_cf, z_base, rtol=1e-5)
    assert not np.allclose(s_cf, s_base, atol=1e-3)


def test_perturbed_expression(trained_model):
    model, adata = trained_model
    adata.layers["cf_zero"] = sp.csr_matrix(np.zeros(adata.shape, dtype=np.float32))
    expr_base = model.get_normalized_expression(library_size=1e4)
    expr_cf = model.get_perturbed_expression(cf_layer="cf_zero", library_size=1e4)
    assert expr_cf.shape == adata.shape
    assert not np.allclose(expr_cf, expr_base, atol=1e-3)
