"""Graph data splitter for Cellina model (node + edge tasks)."""

import torch
import numpy as np
import scipy.sparse as sp
from scvi.dataloaders import DataSplitter
from scvi import REGISTRY_KEYS
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, CachedLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import RandomLinkSplit

from ._constants import SPATIAL_CONNECTIVITIES_KEY, DOMAINS_KEY


class GraphBatchLoader:
    """
    Iterator for graph-aware batches with optional edge prediction.

    Wraps a NeighborLoader and yields dicts in the format CellinaModule expects.
    When an edge_loader is provided, edge batches are included (cycling via CachedLoader).
    """

    def __init__(self, node_loader, edge_loader=None):
        self.node_loader = node_loader
        self.edge_loader = CachedLoader(edge_loader) if edge_loader is not None else None

    @staticmethod
    def _node_batch_to_dict(node_batch):
        """Convert PyG node batch to dict format expected by module."""
        return {
            'X': node_batch.x,
            'edge_index': node_batch.edge_index,
            'node_indices': node_batch.input_id,
            'batch_size': node_batch.batch_size,
            'batch_label': node_batch.batch_labels,
            REGISTRY_KEYS.LABELS_KEY: node_batch.labels,
            DOMAINS_KEY: node_batch.domains,
        }

    def __iter__(self):
        edge_iter = iter(self.edge_loader) if self.edge_loader is not None else None

        for node_batch in self.node_loader:
            batch_dict = {'node_batch': self._node_batch_to_dict(node_batch)}

            if edge_iter is not None:
                try:
                    edge_batch = next(edge_iter)
                except StopIteration:
                    edge_iter = iter(self.edge_loader)
                    edge_batch = next(edge_iter)

                # Filter edges to same batch (physical slide)
                edge_label_index = edge_batch.edge_label_index
                batch_label = edge_batch.batch_labels
                src_batch = batch_label[edge_label_index[0]]
                tgt_batch = batch_label[edge_label_index[1]]
                mask = (src_batch == tgt_batch).flatten()

                batch_dict['edge_batch'] = {
                    'X': edge_batch.x,
                    'edge_index': edge_batch.edge_index,
                    'edge_label_index': edge_batch.edge_label_index,
                    'edge_label': edge_batch.edge_label,
                    'batch_label': batch_label,
                    'edge_mask': mask,
                }

            yield batch_dict

    def __len__(self):
        return len(self.node_loader)


class GraphJointDataSplitter(DataSplitter):
    """
    Graph data splitter for both node-level and edge-level tasks.

    Creates synchronized train/val/test splits and yields batches with both
    node-centric and (optionally) edge-centric subgraphs. Iteration is driven
    by the node loader, with edge batches cycling as needed.

    Parameters
    ----------
    adata_manager
        AnnData manager
    num_neighbors
        Number of neighbors to sample per node per layer. Default: [-1] (all neighbors)
    val_ratio
        Fraction of edges for validation
    test_ratio
        Fraction of edges for test
    neg_sampling_ratio
        Ratio of negative samples to positive samples for edge prediction
    use_edge_prediction
        Whether to create edge loaders for link prediction
    **kwargs
        Additional keyword arguments passed to DataSplitter (batch_size, etc.)
    """

    def __init__(
        self,
        adata_manager,
        num_neighbors=None,
        val_ratio=0.1,
        test_ratio=0.1,
        neg_sampling_ratio=1.0,
        use_edge_prediction=False,
        **kwargs,
    ):
        super().__init__(adata_manager, **kwargs)

        self.num_neighbors = num_neighbors or [-1]
        self.batch_size = kwargs.get('batch_size', 128)
        self.neg_sampling_ratio = neg_sampling_ratio
        self.use_edge_prediction = use_edge_prediction

        self.pyg_data = self._adata_to_pyg_data()
        if self.use_edge_prediction:
            self._split_data(val_ratio, test_ratio)

    def _adata_to_pyg_data(self) -> Data:
        """Convert AnnData to PyG Data for Cellina."""
        # Extract count features
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        if sp.issparse(x):
            x = torch.tensor(x.toarray(), dtype=torch.float32)
        else:
            x = torch.tensor(np.asarray(x), dtype=torch.float32)

        n_cells = x.shape[0]

        # Batch labels for batch correction
        batch_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        batch_labels = torch.from_numpy(np.asarray(batch_data)).long()

        # Cell type labels for classifier
        labels_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = torch.from_numpy(np.asarray(labels_data)).long()

        # Domain labels for discriminator
        domains_data = self.adata_manager.get_from_registry(DOMAINS_KEY)
        domains = torch.from_numpy(np.asarray(domains_data)).long()

        # Spatial adjacency matrix
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

        return Data(
            x=x,
            edge_index=edge_index,
            batch_labels=batch_labels,
            labels=labels,
            domains=domains,
            num_nodes=n_cells,
        )

    def _split_data(self, val_ratio, test_ratio):
        """Split edges using RandomLinkSplit for train/val/test."""
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=0.0,
        )
        self.train_data, self.val_data, self.test_data = transform(self.pyg_data)

        # Share feature tensors across splits to save memory
        for split_data in [self.val_data, self.test_data]:
            split_data.x = self.train_data.x
            split_data.batch_labels = self.train_data.batch_labels
            split_data.labels = self.train_data.labels
            split_data.domains = self.train_data.domains

    def _make_neighbor_loader(self, node_indices, batch_size, shuffle, drop_last):
        """Create a NeighborLoader for the given node indices."""
        return NeighborLoader(
            self.pyg_data,
            num_neighbors=self.num_neighbors,
            input_nodes=torch.tensor(node_indices, dtype=torch.long),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            directed=False,
        )

    def _create_loader(self, node_indices, edge_data=None, shuffle=False):
        """Create a GraphBatchLoader with optional edge prediction."""
        node_loader = self._make_neighbor_loader(node_indices, self.batch_size, shuffle, self.drop_last)

        edge_loader = None
        if edge_data is not None and edge_data.edge_label_index.size(1) > 0:
            edge_loader = LinkNeighborLoader(
                data=edge_data,
                num_neighbors=self.num_neighbors,
                edge_label_index=edge_data.edge_label_index,
                edge_label=getattr(edge_data, 'edge_label', None),
                batch_size=self.batch_size,
                shuffle=shuffle,
                neg_sampling_ratio=self.neg_sampling_ratio,
                drop_last=self.drop_last,
            )

        return GraphBatchLoader(node_loader, edge_loader)

    def train_dataloader(self):
        """Create training dataloader."""
        edge_data = self.train_data if self.use_edge_prediction else None
        return self._create_loader(self.train_idx, edge_data, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        if len(self.val_idx) > 0:
            edge_data = self.val_data if self.use_edge_prediction else None
            return self._create_loader(self.val_idx, edge_data, shuffle=False)
        return None

    def test_dataloader(self):
        """Create test dataloader."""
        if len(self.test_idx) > 0:
            edge_data = self.test_data if self.use_edge_prediction else None
            return self._create_loader(self.test_idx, edge_data, shuffle=False)
        return None

    def create_inference_loader(self, indices, batch_size=None, shuffle=False):
        """
        Create a node-only loader for inference (no edge prediction).

        Parameters
        ----------
        indices
            Node indices to include in batches
        batch_size
            Batch size (defaults to self.batch_size)
        shuffle
            Whether to shuffle
        """
        batch_size = batch_size or self.batch_size
        node_loader = self._make_neighbor_loader(indices, batch_size, shuffle, drop_last=False)
        return GraphBatchLoader(node_loader)
