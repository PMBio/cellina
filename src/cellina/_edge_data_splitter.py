"""Graph data splitter for CellinaGCN model."""

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


def _gather_dense_rows(store, node_ids):
    """Gather ``node_ids`` rows from the sparse feature store and densify them.

    Only the sampled subgraph rows are materialised, so peak memory scales with the
    batch's subgraph size rather than the full ``n_cells x n_genes`` matrix.
    """
    return torch.from_numpy(store[node_ids].toarray()).float()


class GraphBatchLoader:
    """
    Iterator for graph-aware batches.

    Wraps a NeighborLoader and yields dicts in the format CellinaGCNModule expects.
    Node features are kept sparse-resident in ``x_sparse``/``x_spatial_sparse`` and only
    the sampled subgraph's rows are densified per batch (via ``node_batch.n_id``).
    """

    def __init__(self, node_loader, x_sparse, x_spatial_sparse=None):
        self.node_loader = node_loader
        self.x_sparse = x_sparse
        self.x_spatial_sparse = x_spatial_sparse

    def _node_batch_to_dict(self, node_batch):
        nid = node_batch.n_id.cpu().numpy()
        d = {
            'X': _gather_dense_rows(self.x_sparse, nid),
            'edge_index': node_batch.edge_index,
            'node_indices': node_batch.input_id,
            'batch_size': node_batch.batch_size,
            'batch_label': node_batch.batch_labels,
            REGISTRY_KEYS.LABELS_KEY: node_batch.labels,
            DOMAINS_KEY: node_batch.domains,
        }
        if self.x_spatial_sparse is not None:
            d['x_spatial'] = _gather_dense_rows(self.x_spatial_sparse, nid)
        return d

    def __iter__(self):
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
    **kwargs
        Additional keyword arguments passed to DataSplitter (batch_size, etc.).
    """

    def __init__(
        self,
        adata_manager,
        num_neighbors=None,
        x_spatial_layer: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(adata_manager, **kwargs)

        self.num_neighbors = num_neighbors
        self.batch_size = kwargs.get('batch_size', 128)
        self.x_spatial_layer = x_spatial_layer

        self.pyg_data = self._adata_to_pyg_data()

    def _adata_to_pyg_data(self) -> Data:
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        # Keep features sparse-resident; only sampled subgraph rows are densified per
        # batch (see GraphBatchLoader). Avoids materialising the full n_cells x n_genes
        # dense matrix, which otherwise dominates host RAM on large datasets.
        self._x_sparse = x.tocsr()
        self._x_spatial_sparse = None

        n_cells = self._x_sparse.shape[0]

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
            self._x_spatial_sparse = self.load_spatial_store(self.x_spatial_layer)

        return data

    def load_spatial_store(self, layer):
        """Return the sparse feature store for ``adata.layers[layer]``.

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
        if x_sp.shape != self._x_sparse.shape:
            raise ValueError(
                f"x_spatial_layer shape {x_sp.shape} does not match X shape "
                f"{self._x_sparse.shape}"
            )
        return x_sp.tocsr()

    def _make_neighbor_loader(self, node_indices, batch_size, shuffle, drop_last):
        return NeighborLoader(
            self.pyg_data,
            num_neighbors=self.num_neighbors,
            input_nodes=torch.tensor(node_indices, dtype=torch.long),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            directed=False,
        )

    def _create_loader(self, node_indices, shuffle=False):
        node_loader = self._make_neighbor_loader(node_indices, self.batch_size, shuffle, self.drop_last)
        return GraphBatchLoader(node_loader, self._x_sparse, self._x_spatial_sparse)

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
                                x_spatial_override=None):
        """Create a node-only loader for inference.

        ``x_spatial_override`` lets callers reuse this splitter's cached graph and base
        X store while swapping in a different spatial feature store (e.g. a perturbation
        ``cf_layer``), avoiding a full splitter rebuild.
        """
        batch_size = batch_size or self.batch_size
        node_loader = self._make_neighbor_loader(indices, batch_size, shuffle, drop_last=False)
        spatial_store = (
            x_spatial_override if x_spatial_override is not None else self._x_spatial_sparse
        )
        return GraphBatchLoader(node_loader, self._x_sparse, spatial_store)
