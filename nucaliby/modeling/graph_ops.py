"""Gather / scatter primitives for k-NN graphs stored as neighbour-index tensors.

A graph is `edge_idx [B, N, K]`, where `edge_idx[b, i, k]` is the node index of
the k-th neighbour of node i. Edge features live at `[B, N, K, C]`.

`transpose_edge_idx` and `collect_edges_transpose` map directed edges to their
reverse-edge slots.
"""

import torch


def gather_nodes(nodes: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Node features [B, N, C] at neighbour indices [B, N, K] -> [B, N, K, C]."""
    b, n, k = edge_idx.shape
    flat = edge_idx.reshape(b, n * k, 1).expand(-1, -1, nodes.size(-1))
    return torch.gather(nodes, 1, flat).reshape(b, n, k, -1)


def gather_edges(edges: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Dense pair features [B, N, N, C] -> sparse edge features [B, N, K, C]."""
    idx = edge_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, idx)


def cat_neighbors_nodes(h_nodes: torch.Tensor, h_neighbors: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Concat neighbour node features onto edge features.

    Order matters: the result is `[h_neighbors, h_nodes_j]`, so the 384-wide edge
    tensor consumed by the Potts heads is laid out as `[h_E_ij, h_S_j, h_V_j]`.
    """
    return torch.cat([h_neighbors, gather_nodes(h_nodes, edge_idx)], dim=-1)


def transpose_edge_idx(edge_idx: torch.Tensor, mask_ij: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index the reverse edge j->i in place at each forward edge i->j.

    Returns:
        ij_to_ji: [B, N*K] flat indices into the flattened edge tensor.
        mask_ji: [B, N, K], 1 only where *both* i->j and j->i exist. k-NN graphs
            are not symmetric, so this is genuinely sparse.
    """
    b, n, k = edge_idx.shape

    # Neighbours-of-neighbours: [b, i, j, k'] is the k'-th neighbour of i's j-th neighbour.
    flat = edge_idx.reshape(b, n * k, 1).expand(-1, -1, k)
    nbr_of_nbr = torch.gather(edge_idx, 1, flat).reshape(b, n, k, k)

    # Which k' at node j points back at i?
    residue_i = torch.arange(n, device=edge_idx.device).reshape(1, -1, 1, 1)
    match = (nbr_of_nbr == residue_i).to(torch.float32)
    return_mask, return_idx = torch.max(match, dim=-1)

    ij_to_ji = (edge_idx * k + return_idx).reshape(b, -1)

    mask_ji = torch.gather(mask_ij.reshape(b, -1), -1, ij_to_ji).reshape(b, n, k)
    mask_ji = mask_ij * return_mask * mask_ji
    return ij_to_ji, mask_ji


def collect_edges_transpose(
    edge_h: torch.Tensor, edge_idx: torch.Tensor, mask_ij: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the features of edge j->i into the slot of edge i->j."""
    b, n, k, c = edge_h.shape
    ij_to_ji, mask_ji = transpose_edge_idx(edge_idx, mask_ij)
    flat = edge_h.reshape(b, n * k, -1)
    idx = ij_to_ji.unsqueeze(-1).expand(-1, -1, c)
    out = torch.gather(flat, 1, idx).reshape(b, n, k, c)
    return mask_ji.unsqueeze(-1) * out, mask_ji


def collect_neighbors(node_h: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Gather neighboring node features."""
    return gather_nodes(node_h, edge_idx)


def knn_graph(coords: torch.Tensor, mask: torch.Tensor, k: int, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a k-NN graph over token centre coordinates.

    Masked-out nodes are pushed to the maximum observed distance so they sort
    last, rather than being dropped; `k` is clamped to N. Self-edges are kept and
    always land in slot 0 (distance 0), which the nucleotide graph relies on.

    Returns:
        d_neighbors: [B, N, K] distances, sorted ascending.
        edge_idx: [B, N, K] neighbour indices.
    """
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    d = mask_2d * torch.sqrt(((coords.unsqueeze(1) - coords.unsqueeze(2)) ** 2).sum(-1) + eps)
    d_max, _ = torch.max(d, dim=-1, keepdim=True)
    d_adjust = d + (1.0 - mask_2d) * d_max
    return torch.topk(d_adjust, min(k, coords.shape[1]), dim=-1, sorted=True, largest=False)
