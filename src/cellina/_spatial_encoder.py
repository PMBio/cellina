from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, SGConv, SimpleConv
from torch_sparse import SparseTensor


def _identity(x):
    return x


_SPARSE_COMPAT_TYPES = frozenset({"gcn", "gat", "sg", "sum", "mean"})


def _make_conv_layer(
    conv_type: str,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    """Create one graph-conv layer of the requested type."""
    if conv_type == "gcn":
        return GCNConv(in_channels, out_channels, bias=bias, add_self_loops=False)
    elif conv_type == "gat":
        return GATv2Conv(in_channels, out_channels, bias=bias, add_self_loops=False, concat=False)
    elif conv_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels, bias=bias),
        )
        return GINConv(mlp)
    elif conv_type == "sg":
        return SGConv(in_channels, out_channels, bias=bias, add_self_loops=False)
    elif conv_type in ("sum", "mean"):
        aggr = "add" if conv_type == "sum" else "mean"

        class _LinearAggConv(nn.Module):
            def __init__(self, in_ch, out_ch, bias, aggr):
                super().__init__()
                self.linear = nn.Linear(in_ch, out_ch, bias=bias)
                self.conv = SimpleConv(aggr=aggr, combine_root=None)
            def forward(self, x, edge_index):
                return self.conv(self.linear(x), edge_index)

        return _LinearAggConv(in_channels, out_channels, bias, aggr)
    else:
        raise ValueError(
            f"Unknown convolution_type '{conv_type}'. Choose from: gcn, gat, gin, sg, sum, mean."
        )


class GCNLayers(nn.Module):
    """Multi-layer GCN network with covariate injection.

    Parameters
    ----------
    n_in
        Input dimensionality.
    n_out
        Output dimensionality.
    n_cat_list
        Number of categories per categorical covariate.
    n_cont
        Dimensionality of continuous covariates.
    n_layers
        Number of GCN layers.
    n_hidden
        Nodes per hidden layer.
    dropout_rate
        Dropout rate.
    bias
        Whether to learn bias in GCN layers.
    inject_covariates
        Whether to inject covariates in each layer or only the first.
    use_batch_norm
        Whether to use batch normalization.
    convolution_type
        Graph convolution type: ``"gcn"``, ``"gat"``, ``"gin"``, ``"sg"``.
    activation_fn
        Activation function class.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_cont: int = 0,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        bias: bool = True,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        convolution_type: str = "gat",
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        self.n_layers = n_layers
        self.convolution_type = convolution_type
        self.activation_fn = activation_fn()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.n_cov = n_cont + sum(self.n_cat_list)

        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.gcn_layers = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            self.gcn_layers.append(
                _make_conv_layer(convolution_type, dim_in, dim_out, bias=bias)
            )

        self.cov_layers = nn.ModuleList([
            nn.Linear(self.n_cov, dim_out, bias=False)
            if (self.n_cov > 0 and self._inject_into_layer(i))
            else None
            for i, (_, dim_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim_out) if i < n_layers - 1 else nn.Identity()
            for i, (_, dim_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
        ])

    def _inject_into_layer(self, layer_num: int) -> int:
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return int(user_cond)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *cat_list: int,
        cont: torch.Tensor | None = None,
    ):
        one_hot_cat_list = []
        cont_list = [cont] if cont is not None else []
        cat_list = cat_list or []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")

        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]

        cov_list = cont_list + one_hot_cat_list

        if self.convolution_type in _SPARSE_COMPAT_TYPES:
            num_nodes = x.size(0)
            adj = SparseTensor(
                row=edge_index[1], col=edge_index[0],
                sparse_sizes=(num_nodes, num_nodes),
            )
        else:
            adj = edge_index

        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            if self.cov_layers[i] is not None and cov_list:
                cov = torch.cat(cov_list, dim=-1)
                x = x + self.cov_layers[i](cov.float())
            x = self.layer_norms[i](x)
            x = self.activation_fn(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x


class GraphEncoder(nn.Module):
    """Encode node features into a latent space via graph convolutions.

    Parameters
    ----------
    n_input
        Input dimensionality.
    n_output
        Latent dimensionality.
    n_cat_list
        Number of categories per categorical covariate.
    n_layers
        Number of GCN hidden layers.
    n_hidden
        Nodes per hidden layer.
    dropout_rate
        Dropout rate.
    distribution
        Latent distribution (``"normal"`` or ``"ln"``).
    var_eps
        Minimum variance for numerical stability.
    var_activation
        Callable for variance positivity. Defaults to :func:`torch.exp`.
    return_dist
        If True, return the distribution object instead of parameters.
    convolution_type
        Graph convolution type: ``"gcn"``, ``"gat"``, ``"gin"``, ``"sg"``.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        use_batch_norm: bool = True,
        convolution_type: str = "gat",
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = GCNLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            convolution_type=convolution_type,
            **kwargs,
        )
        self.bn = nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001) if use_batch_norm else None
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *cat_list: int,
        batch_size: int,
        return_neighbor_means: bool = False,
    ):
        q = self.encoder(x, edge_index, *cat_list)
        q_seed = q[:batch_size]
        if self.bn is not None:
            q_seed = self.bn(q_seed)
        neighbor_means = None
        if return_neighbor_means:
            neighbor_means = self.mean_encoder(q[batch_size:])

        q_m = self.mean_encoder(q_seed)
        q_v = self.var_activation(self.var_encoder(q_seed)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent, neighbor_means
        return q_m, q_v, latent, neighbor_means
