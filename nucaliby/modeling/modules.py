"""All-atom residue featurization and message-passing blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import MAX_NUM_ATOMS
from .graph_ops import cat_neighbors_nodes, gather_edges, gather_nodes, knn_graph

NUM_RBF = 16
RBF_MIN, RBF_MAX = 2.0, 22.0
NUM_POSITIONAL_EMBEDDINGS = 16
MAX_RELATIVE_FEATURE = 32


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff)
        self.W_out = nn.Linear(num_ff, num_hidden)
        self.act = nn.GELU()

    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))


class PositionalEncodings(nn.Module):
    """Relative sequence-offset encoding, clipped to +/- MAX_RELATIVE_FEATURE.

    Cross-chain pairs collapse into one extra bin, giving 2*32 + 1 + 1 = 66 inputs.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2 * MAX_RELATIVE_FEATURE + 2, NUM_POSITIONAL_EMBEDDINGS)

    def forward(self, offset, same_chain):
        m = MAX_RELATIVE_FEATURE
        d = torch.clip(offset + m, 0, 2 * m) * same_chain + (1 - same_chain) * (2 * m + 1)
        return self.linear(F.one_hot(d, 2 * m + 2).float())


class TokenFeatures(nn.Module):
    """Build the residue k-NN graph and its edge features.

    Edge features concatenate
      * relative positional encoding                       (16)
      * all-atom RBF: 23x23 atom pairs x 16 radial bins    (8464)
      * a token-bond indicator, always 0 for protein-only  (1)
    giving the 8481-wide input of `edge_embedding`.
    """

    def __init__(self, n_channel: int, k_neighbors: int):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.embeddings = PositionalEncodings()
        edge_in = NUM_POSITIONAL_EMBEDDINGS + NUM_RBF * MAX_NUM_ATOMS**2 + 1
        self.edge_embedding = nn.Linear(edge_in, n_channel, bias=False)
        self.norm_edges = nn.LayerNorm(n_channel)

    def _all_atom_rbf(self, x_all, edge_idx):
        """[B, N, K, 23*23*16] radial features over every ordered atom pair.

        Gathers neighbour coordinates first and only then takes distances, so the
        cost is O(N*K*23^2) rather than the O(N^2*23^2) of forming full pairwise
        matrices and discarding all but K columns.
        """
        b, n, a, _ = x_all.shape
        k = edge_idx.shape[-1]
        x_j = gather_nodes(x_all.reshape(b, n, a * 3), edge_idx).reshape(b, n, k, a, 3)
        # [B, N, K, 23(i), 23(j)]
        d = torch.sqrt(((x_all[:, :, None, :, None, :] - x_j[:, :, :, None, :, :]) ** 2).sum(-1) + 1e-6)

        mu = torch.linspace(RBF_MIN, RBF_MAX, NUM_RBF, device=d.device)
        sigma = (RBF_MAX - RBF_MIN) / NUM_RBF
        rbf = torch.exp(-(((d.unsqueeze(-1) - mu) / sigma) ** 2))
        return rbf.reshape(b, n, k, a * a * NUM_RBF)

    def forward(self, batch: dict):
        mask = batch["token_exists_mask"]
        x_center = batch["token_center_coords"]  # [B, N, 3]
        d_neighbors, edge_idx = knn_graph(x_center, mask.float(), self.k_neighbors)

        # Missing atoms fall back to the token centre, so the pair distance stays
        # finite and degenerates rather than producing NaN.
        x_all = torch.where(
            batch["token_atom_mask"].unsqueeze(-1).bool(),
            batch["token_atom_coords"],
            x_center[..., None, :],
        )
        rbf = self._all_atom_rbf(x_all, edge_idx)

        residue_index = batch["residue_index"]
        offset = residue_index[:, :, None] - residue_index[:, None, :]
        offset = gather_edges(offset[..., None], edge_idx)[..., 0]
        asym_id = batch["asym_id"]
        same_chain = ((asym_id[:, :, None] - asym_id[:, None, :]) == 0).long()
        same_chain = gather_edges(same_chain[..., None], edge_idx)[..., 0]
        e_positional = self.embeddings(offset.long(), same_chain)

        # Protein-only inputs contain no token bonds, so this channel is zero.
        token_bonds = torch.zeros_like(e_positional[..., :1])

        e = self.edge_embedding(torch.cat([e_positional, rbf, token_bonds], dim=-1))
        return self.norm_edges(e), edge_idx, d_neighbors


class EncLayer(nn.Module):
    """Encoder block: node update, then edge update."""

    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout(dropout) for _ in range(3))
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, edge_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, edge_idx)
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1), h_EV], dim=-1)
        m = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            m = mask_attend.unsqueeze(-1) * m
        h_V = self.norm1(h_V + self.dropout1(m.sum(-2) / self.scale))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, edge_idx)
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1), h_EV], dim=-1)
        m = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        return h_V, self.norm3(h_E + self.dropout3(m))


class DecLayer(nn.Module):
    """Decoder block.

    Same shape as `EncLayer`, but the incoming edge tensor already carries the
    neighbour's node and sequence embeddings (width `num_in`), so the edge update
    writes back at that width.

    `scale` normalises summed messages by node degree: 30 for the residue graph,
    143 for the nucleotide graph.
    """

    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: float = 30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout(dropout) for _ in range(3))
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden * 2 + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_in)
        self.norm3 = nn.LayerNorm(num_in)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, edge_idx=None, mask_attend=None):
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1), h_E], dim=-1)
        m = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            m = mask_attend.unsqueeze(-1) * m
        h_V = self.norm1(h_V + self.dropout1(m.sum(-2) / self.scale))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, edge_idx)
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1), h_EV], dim=-1)
        m = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        return h_V, self.norm3(h_E + self.dropout3(m))
