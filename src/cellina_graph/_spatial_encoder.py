from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor


def _identity(x):
    return x


class GCNLayers(nn.Module):
    """A helper class to build multi-layer GCN network with covariates support.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_cont
        The dimensionality of the continuous covariates
    n_layers
        The number of GCN layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    bias
        Whether to learn bias in GCN layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
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
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        self.n_layers = n_layers
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
            layer_n_in = dim_in + self.n_cov * self._inject_into_layer(i)
            self.gcn_layers.append(GCNConv(layer_n_in, dim_out, bias=bias, add_self_loops=False))

    def _inject_into_layer(self, layer_num: int) -> int:
        """Helper to determine if covariates should be injected into a layer."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return int(user_cond)

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        if hook_first_layer and len(self.gcn_layers) > 0:
            gcn_first = self.gcn_layers[0]
            w = gcn_first.lin.weight.register_hook(_hook_fn_weight)
            self.hooks.append(w)
            if hasattr(gcn_first, 'bias') and gcn_first.bias is not None:
                b = gcn_first.bias.register_hook(_hook_fn_zero_out)
                self.hooks.append(b)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *cat_list: int,
        cont: torch.Tensor | None = None,
    ):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_nodes, n_in)``
        edge_index
            tensor of edge indices with shape ``(2, n_edges)``
        cat_list
            list of category membership(s) for this sample
        cont
            tensor of continuous covariates with shape ``(n_nodes, n_cont)``

        Returns
        -------
        tensor of shape ``(n_nodes, n_out)``
        """
        # Prepare covariates
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

        # Convert edge_index to SparseTensor for memory-efficient message passing
        num_nodes = x.size(0)
        adj_t = SparseTensor(
            row=edge_index[1], col=edge_index[0],
            sparse_sizes=(num_nodes, num_nodes),
        )

        for i, gcn_layer in enumerate(self.gcn_layers):
            if self._inject_into_layer(i) and cov_list:
                x = torch.cat((x, *cov_list), dim=-1)

            x = gcn_layer(x, adj_t)
            x = self.activation_fn(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x


class GraphEncoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a graph convolutional network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of GCN hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~GCNLayers`
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
            **kwargs,
        )
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
    ):
        q = self.encoder(x, edge_index, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent