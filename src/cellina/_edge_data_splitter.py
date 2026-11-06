"""Graph data splitter for CellinaGCN model."""

import warnings

import torch
import numpy as np
import scipy.sparse as sp
from typing import Optional
from scvi.dataloaders import DataSplitter
from scvi import REGISTRY_KEYS
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import remove_self_loops

from ._constants import SPATIAL_CONNECTIVITIES_KEY, DOMAINS_KEY


def _csr_to_device(x, device):
    """Any (sparse or dense) matrix -> CSR store ``(indptr, indices, data, n_cols)`` on ``device``."""
    x = sp.csr_matrix(x)
    return (torch.from_numpy(x.indptr.astype(np.int64)).to(device),
            torch.from_numpy(x.indices.astype(np.int32)).to(device),  # int32: ~1/3 less memory
            torch.from_numpy(x.data.astype(np.float32)).to(device),
            x.shape[1])


def _gather_dense_rows(store, node_ids):
    """Rows ``node_ids`` of the CSR ``store`` densified, on the store's device.

    Only the sampled subgraph rows are materialised, so peak memory scales with the
    batch's subgraph size rather than the full ``n_cells x n_genes`` matrix. Gathering on
    the training device avoids a host-side scipy gather + H2D copy of the dense subgraph
    (~15 ms -> ~2 ms per batch on crc_232 with 2000 HVGs).
    """
    indptr, indices, data, n_cols = store
    dev = indptr.device
    node_ids = node_ids.to(dev)
    start, nnz = indptr[node_ids], indptr[node_ids + 1] - indptr[node_ids]
    row = torch.repeat_interleave(torch.arange(len(node_ids), device=dev), nnz)
    # position of each gathered entry in indices/data: its row's start + offset within the row
    pos = torch.arange(row.numel(), device=dev) + torch.repeat_interleave(start - (nnz.cumsum(0) - nnz), nnz)
    out = torch.zeros(len(node_ids), n_cols, device=dev)
    out[row, indices[pos].long()] = data[pos]
    return out


class GraphBatchLoader:
    """
    Iterator for graph-aware batches.

    Wraps a NeighborLoader and yields dicts in the format CellinaGCNModule expects.
    Node features live as device-resident CSR stores in ``x_store``/``x_spatial_store``
    (see ``_csr_to_device``) and only the sampled subgraph's rows are densified per batch
    (via ``node_batch.n_id``), on that device.
    """

    def __init__(self, node_loader, x_store, x_spatial_store=None, seed=None):
        self.node_loader = node_loader
        self.x_store = x_store
        self.x_spatial_store = x_spatial_store
        # Inference loaders are seeded so neighbour sampling (and posterior draws)
        # do not depend on the global RNG state at call time; None = untouched (training).
        self.seed = seed

    def _node_batch_to_dict(self, node_batch):
        nid = node_batch.n_id
        d = {
            'X': _gather_dense_rows(self.x_store, nid),
            'edge_index': node_batch.edge_index,
            'node_indices': node_batch.input_id,
            # local -> global id for every sampled node (seeds first); maps per-edge
            # quantities such as GAT attention back onto adata's indexing.
            'n_id': node_batch.n_id,
            'batch_size': node_batch.batch_size,
            'batch_label': node_batch.batch_labels,
            REGISTRY_KEYS.LABELS_KEY: node_batch.labels,
            DOMAINS_KEY: node_batch.domains,
        }
        if self.x_spatial_store is not None:
            d['x_spatial'] = _gather_dense_rows(self.x_spatial_store, nid)
        return d

    def __iter__(self):
        if self.seed is None:
            yield from self._iter()
            return
        devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            yield from self._iter()

    def _iter(self):
        for node_batch in self.node_loader:
            yield {'node_batch': self._node_batch_to_dict(node_batch)}

    def __len__(self):
        return len(self.node_loader)


class GraphJointDataSplitter(DataSplitter):
    """
    Graph data splitter for node-level tasks.

    Creates synchronized train/val/test splits and yields batches with
    node-centric subgraphs (seed nodes + sampled spatial neighbours).

    Parameters
    ----------
    adata_manager
        AnnData manager.
    num_neighbors
        Fan-out sampled per GCN hop; its length should equal the model's ``n_layers``.
        ``CellinaGCN`` passes a resolved list (see ``_resolve_num_neighbors``). 
    x_spatial_layer
        Optional key in ``adata.layers`` for alternative spatial features.
    device
        Device holding the CSR feature stores; batches' ``X`` / ``x_spatial`` are gathered
        and densified there (the training / inference device).
    **kwargs
        Additional keyword arguments passed to DataSplitter (batch_size, etc.).
    """

    def __init__(
        self,
        adata_manager,
        num_neighbors=None,
        x_spatial_layer: Optional[str] = None,
        subgraph_type: str = "induced",
        device="cpu",
        **kwargs,
    ):
        super().__init__(adata_manager, **kwargs)

        self.num_neighbors = num_neighbors
        self.batch_size = kwargs.get('batch_size', 128)
        self.x_spatial_layer = x_spatial_layer
        # NeighborLoader subgraph_type for train/val/test/inference loaders.
        # 'induced' keeps all edges among sampled nodes; 'directional' keeps only
        # sampling-path edges (lower VRAM, but changes message passing since real
        # spatial neighbours are densely interconnected). See _make_neighbor_loader.
        self.subgraph_type = subgraph_type
        self.device = torch.device(device)

        self.pyg_data = self._adata_to_pyg_data()

    def _adata_to_pyg_data(self) -> Data:
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        # CSR store on self.device; only sampled subgraph rows are densified per batch
        # (see GraphBatchLoader). Avoids materialising the full n_cells x n_genes dense matrix.
        self._x_shape = x.shape
        self._x_store = _csr_to_device(x, self.device)
        self._x_spatial_store = None

        n_cells = x.shape[0]

        batch_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        batch_labels = torch.from_numpy(np.asarray(batch_data)).long()

        labels_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = torch.from_numpy(np.asarray(labels_data)).long()

        domains_data = self.adata_manager.get_from_registry(DOMAINS_KEY)
        domains = torch.from_numpy(np.asarray(domains_data)).long()

        spatial_key = self.adata_manager.adata.uns.get(SPATIAL_CONNECTIVITIES_KEY)
        if spatial_key is None or spatial_key not in self.adata_manager.adata.obsp:
            key_name = spatial_key or "<not set>"
            raise ValueError(
                f"Spatial connectivity matrix '{key_name}' not found in adata.obsp. "
                f"Run spatial_neighbors(adata, key_added='{key_name}') before training."
            )
        adj_matrix = self.adata_manager.adata.obsp[spatial_key]
        if not sp.issparse(adj_matrix):
            adj_matrix = sp.csr_matrix(adj_matrix)

        # NeighborLoader samples incoming edges and GCNLayers builds adj_t from them, so
        # an asymmetric graph silently transposes the receptive field. Symmetrize and warn.
        if (adj_matrix != adj_matrix.T).nnz > 0:
            warnings.warn(
                f"Spatial connectivity matrix '{spatial_key}' is not symmetric; "
                "symmetrizing internally with adj.maximum(adj.T).",
                UserWarning,
            )
            adj_matrix = adj_matrix.maximum(adj_matrix.T)

        # Explicitly stored zeros (e.g. from in-place weight thresholding) are non-edges;
        # tocoo() would otherwise turn them into message-passing edges. Copy before
        # eliminate_zeros() so the user's adata.obsp matrix is never mutated.
        if np.any(adj_matrix.data == 0):
            adj_matrix = adj_matrix.copy()
            adj_matrix.eliminate_zeros()
        adj_coo = adj_matrix.tocoo()
        edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)

        # NeighborLoader expands the subgraph from edge_index + num_nodes alone; features
        # are gathered lazily by node id, so no dense x is attached to Data here.
        data = Data(
            edge_index=edge_index,
            batch_labels=batch_labels,
            labels=labels,
            domains=domains,
            num_nodes=n_cells,
        )

        if self.x_spatial_layer is not None:
            self._x_spatial_store = self.load_spatial_store(self.x_spatial_layer)

        return data

    def load_spatial_store(self, layer):
        """Return the device CSR feature store for ``adata.layers[layer]``.

        Validation matches the X matrix shape. No densification — the store is gathered
        per batch like the main X store.
        """
        adata = self.adata_manager.adata
        if layer not in adata.layers:
            raise ValueError(
                f"x_spatial_layer '{layer}' not in adata.layers. "
                f"Available: {list(adata.layers.keys())}"
            )
        x_sp = adata.layers[layer]
        if x_sp.shape != self._x_shape:
            raise ValueError(
                f"x_spatial_layer shape {x_sp.shape} does not match X shape "
                f"{self._x_shape}"
            )
        return _csr_to_device(x_sp, self.device)

    def _make_neighbor_loader(self, node_indices, batch_size, shuffle, drop_last,
                              num_neighbors=None):
        # An explicit num_neighbors overrides the fan-out for this loader only; the
        # splitter is cached and shared, so it is never mutated.
        return NeighborLoader(
            self.pyg_data,
            num_neighbors=self.num_neighbors if num_neighbors is None else num_neighbors,
            input_nodes=torch.tensor(node_indices, dtype=torch.long),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            subgraph_type=self.subgraph_type,
        )

    def _create_loader(self, node_indices, shuffle=False):
        node_loader = self._make_neighbor_loader(node_indices, self.batch_size, shuffle, self.drop_last)
        return GraphBatchLoader(node_loader, self._x_store, self._x_spatial_store)

    def train_dataloader(self):
        return self._create_loader(self.train_idx, shuffle=True)

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return self._create_loader(self.val_idx, shuffle=False)
        return None

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return self._create_loader(self.test_idx, shuffle=False)
        return None

    def create_inference_loader(self, indices, batch_size=None, shuffle=False,
                                x_spatial_override=None, num_neighbors=None, seed=0):
        """Create a node-only loader for inference.

        ``x_spatial_override`` lets callers reuse this splitter's cached graph and base
        X store while swapping in a different spatial feature store (e.g. a perturbation
        ``cf_layer``), avoiding a full splitter rebuild. ``num_neighbors`` overrides the
        fan-out for this loader only. ``seed`` makes neighbour sampling deterministic
        regardless of global RNG state.
        """
        batch_size = batch_size or self.batch_size
        node_loader = self._make_neighbor_loader(indices, batch_size, shuffle, drop_last=False,
                                                 num_neighbors=num_neighbors)
        spatial_store = (
            x_spatial_override if x_spatial_override is not None else self._x_spatial_store
        )
        return GraphBatchLoader(node_loader, self._x_store, spatial_store, seed=seed)
