"""Conditional Potts model on a k-NN graph.

    E(x) = sum_v h_v(x_v) + sum_(u,v) J_uv(x_u, x_v)

`h` comes from node features and `J` from edge features through the factored
parameterization `J = L @ R`, with `L, R` in R^{AxA}.

Both heads are instances of this class: the amino-acid head over 32 states and
the nucleotide head over 4.
"""

import torch
import torch.nn as nn

from .graph_ops import collect_edges_transpose, collect_neighbors


def mask_couplings(edge_idx: torch.Tensor, mask_i: torch.Tensor, mask_ij: torch.Tensor) -> torch.Tensor:
    """Edge mask for J: drop self-edges and any edge touching a missing node."""
    ii = torch.arange(edge_idx.shape[1], device=edge_idx.device).view(1, -1, 1)
    not_self = torch.ne(edge_idx, ii).float()
    self_present = mask_i.unsqueeze(-1)
    neighbor_present = collect_neighbors(self_present, edge_idx).squeeze(-1)
    mask_J = not_self * self_present * neighbor_present
    return mask_J if mask_ij is None else mask_ij * mask_J


INIT_SCALE = 0.1
DROPOUT = 0.1


class GraphPotts(nn.Module):
    """Predict Potts fields h [B, N, A] and couplings J [B, N, K, A, A].

    Instantiated twice: over 32 amino-acid states, and over 4 nucleotide states.
    """

    def __init__(self, dim_nodes: int, dim_edges: int, num_states: int):
        super().__init__()
        self.num_states = num_states
        self.log_scale = nn.Parameter(torch.log(torch.tensor(INIT_SCALE)) * torch.ones(1))
        self.W_h = nn.Linear(dim_nodes, num_states)
        self.W_J_left = nn.Linear(dim_edges, num_states**2)
        self.W_J_right = nn.Linear(dim_edges, num_states**2)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, node_h, edge_h, edge_idx, mask_i, mask_ij):
        a = self.num_states
        mask_J = mask_couplings(edge_idx, mask_i, mask_ij)
        scale = torch.exp(self.log_scale)

        h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
        mask_J = scale * mask_J.unsqueeze(-1)
        shape_J = list(edge_h.shape)[:3] + [a, a]
        J = torch.matmul(
            (mask_J * self.W_J_left(edge_h)).view(shape_J),
            (mask_J * self.W_J_right(edge_h)).view(shape_J),
        )
        J = self.dropout(J)

        # Zero-sum gauge: removes the reparameterisation freedom that would
        # otherwise let constant shifts move between h and J.
        h = h - h.mean(-1, keepdim=True)
        J = J - J.mean(-1, keepdim=True) - J.mean(-2, keepdim=True) + J.mean(dim=(-1, -2), keepdim=True)

        return h, self._symmetrize(J, edge_idx, mask_ij)

    def _symmetrize(self, J, edge_idx, mask_ij):
        """J_ij <- 0.5 * (J_ij + J_ji^T), and ZERO where the reverse edge is absent.

        k-NN graphs are not symmetric, so this drops one-directional edges rather
        than averaging them against nothing. `potts_energy`'s 0.5 double-count
        correction is only valid because of that.
        """
        b, n, k, a, _ = J.shape
        J_t, mask_ji = collect_edges_transpose(J.reshape(b, n, k, -1), edge_idx, mask_ij)
        J_t = J_t.reshape(b, n, k, a, a).transpose(-2, -1)
        return (0.5 * mask_ji).view(b, n, k, 1, 1) * (J + J_t)


def potts_energy(
    S: torch.Tensor, h: torch.Tensor, J: torch.Tensor, edge_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Total energy and per-site conditional energies.

    Args:
        S: [B, N] state indices.

    Returns:
        U: [B] total energy (lower is better).
        U_i: [B, N, A] site-conditional energies, i.e. the energy of setting site
            i to each state with the rest of S held fixed. This is what the DLMC
            proposal differentiates.
    """
    S_j = collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1], -1)
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)

    J_i = J_ij.sum(2)
    U_i = h + J_i

    s = S[..., None]
    U = (torch.gather(U_i, -1, s) - 0.5 * torch.gather(J_i, -1, s)).sum((1, 2))
    return U, U_i
