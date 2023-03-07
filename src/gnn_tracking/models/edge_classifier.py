from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from torch_geometric.nn import MessagePassing


from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP
from gnn_tracking.models.resin import ResIN


class EdgeClassifier(nn.Module):
    def __init__(
        self,
        node_indim,
        edge_indim,
        L=4,
        node_latentdim=8,
        edge_latentdim=12,
        r_hidden_size=32,
        o_hidden_size=32,
    ):
        super().__init__()
        self.node_encoder = MLP(node_indim, node_latentdim, 64, L=1)
        self.edge_encoder = MLP(edge_indim, edge_latentdim, 64, L=1)
        gnn_layers = []
        for _l in range(L):
            # fixme: Wrong parameters?
            gnn_layers.append(
                IN(
                    node_latentdim,
                    edge_latentdim,
                    node_outdim=node_latentdim,
                    edge_outdim=edge_latentdim,
                    edge_hidden_dim=r_hidden_size,
                    node_hidden_dim=o_hidden_size,
                )
            )
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.W = MLP(edge_latentdim, 1, 32, L=2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            node_latent, edge_latent = layer(node_latent, edge_index, edge_latent)
        edge_weights = torch.sigmoid(self.W(edge_latent))
        return edge_weights


def symmetrize_edge_weights(edge_indices: Tensor, edge_weights: Tensor) -> Tensor:
    """
    Symmetrizes the edge_weights based on edge_indices
    Args:
    edge_indices: A 2xN tensor containing the indices of the edges
    edge_weights: A 1D tensor containing the weights of the edges
    
    Returns:
    sym_edge_weights: A 1D tensor containing the symmetrized edge weights
    """
    ei = edge_indices.tolist()
    ew = edge_weights.tolist()
    seen = set()

    ei_dict = {(x[0], x[1]): i for i, x in enumerate(ei)}
    for i, e in enumerate(ei):
        if (e[1], e[0]) in ei_dict and (e[1], e[0]) not in seen:
            flip_idx = ei_dict[(e[1], e[0])]
            avg = (ew[i] + ew[flip_idx]) / 2
            ew[i] = avg
            ew[flip_idx] = avg
            seen.add((e[1], e[0]))
            seen.add((e[0], e[1]))

    return Tensor(ew)


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)


class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)


class InteractionNetwork(MessagePassing):
    def __init__(self, hidden_size):
        super(InteractionNetwork, self).__init__(aggr='add',
                                                 flow='source_to_target')
        self.R1 = RelationalModel(16, 4, hidden_size)
        self.O = ObjectModel(10, 3, hidden_size)
        self.R2 = RelationalModel(10, 1, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:

        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=None)

        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2))

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c)


class ECForGraphTCN(nn.Module):
    def __init__(
        self,
        *,
        node_indim: int,
        edge_indim: int,
        interaction_node_hidden_dim=5,
        interaction_edge_hidden_dim=4,
        h_dim=5,
        e_dim=4,
        hidden_dim=40,
        L_ec=3,
        alpha_ec: float = 0.5,
    ):
        """Edge classification step to be used for Graph Track Condensor network
        (Graph TCN)

        Args:
            node_indim: Node feature dim
            edge_indim: Edge feature dim
            interaction_node_hidden_dim: Hidden dimension of interaction networks.
                Defaults to 5 for backward compatibility, but this is probably
                not reasonable.
            interaction_edge_hidden_dim: Hidden dimension of interaction networks
                Defaults to 4 for backward compatibility, but this is probably
                not reasonable.
            h_dim: node dimension in latent space
            e_dim: edge dimension in latent space
            hidden_dim: width of hidden layers in all perceptrons
            L_ec: message passing depth for edge classifier
            alpha_ec: strength of residual connection for EC
        """
        super().__init__()
        self.relu = nn.ReLU()

        # specify the edge classifier
        self.ec_node_encoder = MLP(
            node_indim, h_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.ec_edge_encoder = MLP(
            edge_indim, e_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.ec_resin = ResIN.identical_in_layers(
            node_indim=h_dim,
            edge_indim=e_dim,
            node_outdim=h_dim,
            edge_outdim=e_dim,
            node_hidden_dim=interaction_node_hidden_dim,
            edge_hidden_dim=interaction_edge_hidden_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_ec,
            n_layers=L_ec,
        )

        self.W = MLP(
            e_dim + self.ec_resin.length_concatenated_edge_attrs, 1, hidden_dim, L=3
        )

    def forward(
        self,
        data: Data,
    ) -> Tensor:
        # apply the edge classifier to generate edge weights
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h_ec = self.relu(self.ec_node_encoder(x))
        edge_attr_ec = self.relu(self.ec_edge_encoder(edge_attr))
        h_ec, _, edge_attrs_ec = self.ec_resin(h_ec, edge_index, edge_attr_ec)

        # append edge weights as new edge features
        edge_attrs_ec = torch.cat(edge_attrs_ec, dim=1)
        edge_weights = torch.sigmoid(self.W(edge_attrs_ec))
        return edge_weights


class PerfectEdgeClassification(nn.Module):
    def __init__(self, tpr=1.0, tnr=1.0, false_below_pt=0.0):
        """An edge classifier that is perfect because it uses the truth information.
        If TPR or TNR is not 1.0, noise is added to the truth information.

        Args:
            tpr: True positive rate
            tnr: False positive rate
            false_below_pt: If not 0.0, all true edges between hits corresponding to
                particles with a pt lower than this threshold are set to false.
                This is not counted towards the TPR/TNR but applied afterwards.
        """
        super().__init__()
        assert 0.0 <= tpr <= 1.0
        self.tpr = tpr
        assert 0.0 <= tnr <= 1.0
        self.tnr = tnr
        self.false_below_pt = false_below_pt

    def forward(self, data: Data) -> Tensor:
        r = data.y.bool()
        if not np.isclose(self.tpr, 1.0):
            true_mask = r.detach().clone()
            rand = torch.rand(int(true_mask.sum()), device=r.device)
            r[true_mask] = rand <= self.tpr
        if not np.isclose(self.tnr, 1.0):
            false_mask = (~r).detach().clone()
            rand = torch.rand(int(false_mask.sum()), device=r.device)
            r[false_mask] = ~(rand <= self.tnr)
        if self.false_below_pt > 0.0:
            false_mask = data.pt < self.false_below_pt
            r[false_mask] = False
        return r
