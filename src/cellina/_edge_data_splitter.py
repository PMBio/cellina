"""Graph data splitter for CellinaGraph model."""

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


class GraphBatchLoader:
    """
    Iterator for graph-aware batches.

    Wraps a NeighborLoader and yields dicts in the format CellinaGraphModule expects.
    """

    def __init__(self, node_loader):
        self.node_loader = node_loader

    @staticmethod
    def _node_batch_to_dict(node_batch):
        d = {
            'X': node_batch.x,
            'edge_index': node_batch.edge_index,
            'node_indices': node_batch.input_id,
            'batch_size': node_batch.batch_size,
            'batch_label': node_batch.batch_labels,
            REGISTRY_KEYS.LABELS_KEY: node_batch.labels,
            DOMAINS_KEY: node_batch.domains,
        }
        if hasattr(node_batch, 'x_spatial'):
            d['x_spatial'] = node_batch.x_spatial
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
        Number of neighbors to sample per node per layer. Default: [-1] (all neighbors).
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

        self.num_neighbors = num_neighbors or [-1]
        self.batch_size = kwargs.get('batch_size', 128)
        self.x_spatial_layer = x_spatial_layer

        self.pyg_data = self._adata_to_pyg_data()

    def _adata_to_pyg_data(self) -> Data:
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        if sp.issparse(x):
            x = torch.tensor(x.toarray(), dtype=torch.float32)
        else:
            x = torch.tensor(np.asarray(x), dtype=torch.float32)

        n_cells = x.shape[0]

        batch_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        batch_labels = torch.from_numpy(np.asarray(batch_data)).long()

        labels_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = torch.from_numpy(np.asarray(labels_data)).long()

        domains_data = self.adata_manager.get_from_registry(DOMAINS_KEY)
        domains = torch.from_numpy(np.asarray(domains_data)).long()

        spatial_key = self.adata_manager.adata.uns.get(SPATIAL_CONNECTIVITIES_KEY)
        if spatial_key is None or spatial_key not in self.adata_manager.adata.obsp:
            raise ValueError(
                f"Spatial connectivity key '{spatial_key}' not found in adata.obsp. "
                f"Available keys: {list(self.adata_manager.adata.obsp.keys())}. "
                "Please provide spatial_connectivities_key in setup_anndata()."
            )
        adj_matrix = self.adata_manager.adata.obsp[spatial_key]
        if not sp.issparse(adj_matrix):
            adj_matrix = sp.csr_matrix(adj_matrix)

        adj_coo = adj_matrix.tocoo()
        edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)

        data = Data(
            x=x,
            edge_index=edge_index,
            batch_labels=batch_labels,
            labels=labels,
            domains=domains,
            num_nodes=n_cells,
        )

        if self.x_spatial_layer is not None:
            adata = self.adata_manager.adata
            if self.x_spatial_layer not in adata.layers:
                raise ValueError(
                    f"x_spatial_layer '{self.x_spatial_layer}' not in adata.layers. "
                    f"Available: {list(adata.layers.keys())}"
                )
            x_sp = adata.layers[self.x_spatial_layer]
            if sp.issparse(x_sp):
                x_sp = x_sp.toarray()
            x_sp = np.asarray(x_sp)
            if x_sp.shape != tuple(data.x.shape):
                raise ValueError(
                    f"x_spatial_layer shape {x_sp.shape} does not match X shape {tuple(data.x.shape)}"
                )
            data.x_spatial = torch.tensor(x_sp, dtype=torch.float32)

        return data

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
        return GraphBatchLoader(node_loader)

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

    def create_inference_loader(self, indices, batch_size=None, shuffle=False):
        """Create a node-only loader for inference."""
        batch_size = batch_size or self.batch_size
        node_loader = self._make_neighbor_loader(indices, batch_size, shuffle, drop_last=False)
        return GraphBatchLoader(node_loader)
