"""Joint graph data splitter for Cellina model (node + edge tasks)."""

import torch
import numpy as np
import scipy.sparse as sp
from itertools import cycle
from typing import Optional, List
from scvi.dataloaders import DataSplitter
from scvi import REGISTRY_KEYS
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import RandomLinkSplit

from ._constants import SPATIAL_CONNECTIVITIES_KEY, DOMAINS_KEY


class JointBatchLoader:
    """
    Iterator that yields combined node and edge batches.
    
    Node loader drives iteration count. Edge loader cycles/resets as needed.
    Spatial features are included in BOTH node and edge batches.
    Domains are included in node batches only.
    """
    
    def __init__(self, node_loader, edge_loader, adata_manager):
        self.node_loader = node_loader
        self.edge_loader = edge_loader
        self.adata_manager = adata_manager
        
    def __iter__(self):
        # Create cycling iterator for edge batches
        if self.edge_loader is not None:
            edge_iter = cycle(self.edge_loader)
        else:
            edge_iter = None
        
        for node_batch in self.node_loader:
            batch_dict = {
                'node_batch': {
                    'X': node_batch.x,
                    'edge_index': node_batch.edge_index,
                    'node_indices': node_batch.input_id,
                    'batch_size': node_batch.batch_size,
                    'batch_label': node_batch.batch_labels,
                    'spatial_x': node_batch.spatial_x,
                    'domains': node_batch.domains,
                }
            }
            
            # Add edge batch if available
            if edge_iter is not None:
                edge_batch = next(edge_iter)

                # Keep only edges within the same batch (sample)
                edge_label_index = edge_batch['edge_label_index']
                batch_label = edge_batch['batch_labels']
                src_batch = batch_label[edge_label_index[0]]
                tgt_batch = batch_label[edge_label_index[1]]
                
                mask = (src_batch == tgt_batch).flatten()
                
                batch_dict['edge_batch'] = {
                    'X': edge_batch.x,
                    'edge_index': edge_batch.edge_index,
                    'edge_label_index': edge_batch.edge_label_index,
                    'edge_label': edge_batch.edge_label,
                    'batch_label': batch_label,
                    'spatial_x': edge_batch.spatial_x,
                    'edge_mask': mask
                }
            
            yield batch_dict
            
    def __len__(self):
        return len(self.node_loader)


class GraphJointDataSplitter(DataSplitter):
    """
    Joint graph data splitter for both node-level and edge-level tasks.
    
    Creates synchronized train/val/test splits and yields batches with both
    node-centric and edge-centric subgraphs. Iteration is driven by the node
    loader, with edge batches cycling as needed.
    
    Spatial features are included in both node and edge batches.
    Domains are included in node batches only.
    
    Parameters
    ----------
    adata_manager
        AnnData manager
    num_neighbors
        Number of neighbors to sample per node. Default: [-1] (all neighbors)
    val_ratio
        Fraction of edges for validation
    test_ratio
        Fraction of edges for test
    neg_sampling_ratio
        Ratio of negative samples to positive samples for edge prediction
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
        **kwargs
    ):
        super().__init__(adata_manager, **kwargs)
        
        # Configuration
        self.num_neighbors = num_neighbors or [-1]
        self.batch_size = kwargs.get('batch_size', 128)
        self.neg_sampling_ratio = neg_sampling_ratio
        
        # Create PyG data and perform splits
        self.pyg_data = self._adata_to_pyg_data()
        self._split_data(val_ratio, test_ratio)
        
    def _adata_to_pyg_data(self) -> Data:
        """Convert AnnData to PyG Data for Cellina."""
        # Extract features from the registered count layer
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        if sp.issparse(x):
            x = torch.tensor(x.toarray(), dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)
            
        # Get spatial features
        spatial_key = self.adata_manager.registry["setup_args"]["spatial_obsm_key"]
        if sp.issparse(self.adata_manager.adata.obsm[spatial_key]):
            spatial_x = torch.tensor(self.adata_manager.adata.obsm[spatial_key].toarray(), dtype=torch.float32)
        else:
            spatial_x = torch.tensor(self.adata_manager.adata.obsm[spatial_key], dtype=torch.float32)
        
        # Get domains
        domains_data = self.adata_manager.get_from_registry(DOMAINS_KEY)
        domains = torch.from_numpy(domains_data).long()
        
        # Get batch data and ensure proper shape
        batch_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        batch = torch.from_numpy(batch_data).long()
        
        # Verify tensor shapes match
        n_cells = x.shape[0]
        if spatial_x.shape[0] != n_cells:
            raise ValueError(f"Spatial features have {spatial_x.shape[0]} rows but data has {n_cells} cells")
        if domains.shape[0] != n_cells:
            raise ValueError(f"Domains have {domains.shape[0]} entries but data has {n_cells} cells")
        if batch.shape[0] != n_cells:
            raise ValueError(f"Batch data has {batch.shape[0]} entries but data has {n_cells} cells")
        
        # Extract spatial adjacency matrix using configured key
        spatial_key = self.adata_manager.adata.uns.get(SPATIAL_CONNECTIVITIES_KEY)
        
        if spatial_key not in self.adata_manager.adata.obsp.keys():
            raise ValueError(
                f"spatial_key '{spatial_key}' not found in adata.obsp. "
                "Available keys: " + ", ".join(self.adata_manager.adata.obsp.keys())
            )
        adj_matrix = self.adata_manager.adata.obsp[spatial_key]
        if not sp.issparse(adj_matrix):
            adj_matrix = sp.csr_matrix(adj_matrix)
        
        # Convert to edge_index
        adj_coo = adj_matrix.tocoo()
        edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
        
        # Remove self-loops for link prediction
        edge_index, _ = remove_self_loops(edge_index)

        return Data(x=x, spatial_x=spatial_x, domains=domains, edge_index=edge_index, batch_labels=batch, num_nodes=x.shape[0])

    def _split_data(self, val_ratio, test_ratio):
        """
        Split edges using RandomLinkSplit, creating train/val/test data objects.
        Node splits are inherited from parent DataSplitter (self.train_idx, etc.)
        """
        # Use PyG's RandomLinkSplit for edge splitting
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=0.0,  # We'll do neg sampling in LinkNeighborLoader
        )
        
        # Apply the split - creates separate data objects for train/val/test
        self.train_data, self.val_data, self.test_data = transform(self.pyg_data)
        
    def _create_joint_loader(self, split_name, node_indices, edge_data, shuffle=False):
        """
        Create combined node and edge loaders that iterate together.
        
        Parameters
        ----------
        split_name
            'train', 'val', or 'test'
        node_indices
            Indices for node-level sampling
        edge_data
            PyG Data object with edge splits
        shuffle
            Whether to shuffle
        """
        node_loader = NeighborLoader(
            self.pyg_data,
            num_neighbors=self.num_neighbors,
            input_nodes=torch.tensor(node_indices, dtype=torch.long),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            directed=False,
        )
        
        # Create edge loader (no self-loops for link prediction)
        edge_loader = None
        if edge_data.edge_label_index.size(1) > 0:
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
        
        return JointBatchLoader(node_loader, edge_loader, self.adata_manager)

    def train_dataloader(self):
        """Create training dataloader."""
        return self._create_joint_loader('train', self.train_idx, self.train_data, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        if len(self.val_idx) > 0:
            return self._create_joint_loader('val', self.val_idx, self.val_data, shuffle=False)
        return None

    def test_dataloader(self):
        """Create test dataloader."""
        if len(self.test_idx) > 0:
            return self._create_joint_loader('test', self.test_idx, self.test_data, shuffle=False)
        return None