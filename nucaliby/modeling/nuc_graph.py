"""Lift the residue graph to a nucleotide graph.

N residue nodes with K neighbours each become 3N nucleotide nodes with
K_TOTAL = 2 + 3(K-1) = 143 neighbours each:

    u_{i,p}            = h_i  + E_pos(p) + W_nuc_s(onehot(nuc_type))
    w_{(i,p)->(j,q)}   = e_ij + E_type(...)

Intra-codon edges (i == j) occupy slots [0, 2). There is no residue-level edge
i->i, so they are built from the *self* edge h_ESV[:, :, 0]. Inter-codon edges
fill the rest, tiling each of the K-1 non-self residue edges across the 3 target
codon positions.

Edge types use the unordered intra-codon position pair (three types) and the
destination inter-codon position (three types), matching the module
parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import N_NUC_TOKENS, NUC_MASK_IDX, NUC_UNK_IDX


class ResidueToNucleotideGraph(nn.Module):
    def __init__(self):
        super().__init__()
        from .model import EDGE_N_CHANNEL, N_CHANNEL

        self.E_pos = nn.Embedding(3, N_CHANNEL)
        self.W_nuc_s = nn.Linear(N_NUC_TOKENS, N_CHANNEL, bias=False)
        self.E_intra = nn.Embedding(3, EDGE_N_CHANNEL)
        self.E_inter = nn.Embedding(3, EDGE_N_CHANNEL)

        # For codon position p, the two intra-codon edges point at the other two
        # positions, with the unordered-pair type index. Reordering these
        # desynchronises E_intra from the intra-codon slots that the
        # codon-marginalised Potts loss indexes by hand.
        self.register_buffer("_intra_dst_q", torch.tensor([[1, 2], [0, 2], [0, 1]]))
        self.register_buffer("_intra_type_idx", torch.tensor([[0, 1], [0, 2], [1, 2]]))

    def nuc_token_types(self, nuc_seq, seq_cond_mask, token_unk_mask, b, n, device):
        """Per-nucleotide input token [B, 3N], applying MASK then UNK precedence.

        UNK wins: an unresolved residue is UNK at every codon position whether or
        not it was selected for masking.
        """
        n3 = 3 * n
        if nuc_seq is None or seq_cond_mask is None:
            return torch.full((b, n3), NUC_MASK_IDX, device=device, dtype=torch.long)
        types = nuc_seq.clone()
        cond_3n = seq_cond_mask.unsqueeze(-1).expand(b, n, 3).reshape(b, n3)
        types[~cond_3n.bool()] = NUC_MASK_IDX
        unk_3n = token_unk_mask.unsqueeze(-1).expand(b, n, 3).reshape(b, n3)
        types[unk_3n.bool()] = NUC_UNK_IDX
        return types

    def embed_nuc_tokens(self, types):
        return self.W_nuc_s(F.one_hot(types.long(), num_classes=N_NUC_TOKENS).float())

    def forward(self, h_V, h_ESV, edge_idx, token_mask, nuc_types):
        """
        Args:
            h_V: [B, N, D] residue node features.
            h_ESV: [B, N, K, D_e] residue edge features; slot 0 is the self edge.
            edge_idx: [B, N, K] residue neighbour indices.
            token_mask: [B, N] valid-residue mask.
            nuc_types: [B, 3N] nucleotide input tokens.

        Returns:
            nuc_h_V [B, 3N, D], nuc_h_ESV [B, 3N, 143, D_e], nuc_edge_idx,
            nuc_mask_i [B, 3N], nuc_mask_ij [B, 3N, 143].
        """
        from .model import N_INTRA_SLOTS

        b, n, d_node = h_V.shape
        d_edge = h_ESV.shape[-1]
        k = edge_idx.shape[2]
        device = h_V.device
        n3 = 3 * n

        pos_idx = torch.arange(n3, device=device) % 3
        res_of_nuc = torch.arange(n3, device=device) // 3

        # --- nodes: N -> 3N -------------------------------------------------
        pos_emb = self.E_pos(torch.arange(3, device=device))
        nuc_h_V = (h_V.unsqueeze(2).expand(b, n, 3, d_node) + pos_emb).reshape(b, n3, d_node)
        nuc_h_V = nuc_h_V + self.embed_nuc_tokens(nuc_types)
        nuc_mask_i = token_mask.unsqueeze(-1).expand(b, n, 3).reshape(b, n3)

        k_intra = N_INTRA_SLOTS
        k_inter_per_q = k - 1
        k_inter = 3 * k_inter_per_q
        k_total = k_intra + k_inter

        nuc_edge_idx = torch.zeros(b, n3, k_total, dtype=torch.long, device=device)
        nuc_h_ESV = torch.zeros(b, n3, k_total, d_edge, device=device, dtype=h_ESV.dtype)
        nuc_mask_ij = torch.zeros(b, n3, k_total, device=device, dtype=nuc_mask_i.dtype)

        # --- intra-codon edges, slots [0, 2) --------------------------------
        intra_dst = res_of_nuc[:, None] * 3 + self._intra_dst_q[pos_idx]
        nuc_edge_idx[:, :, :k_intra] = intra_dst[None].expand(b, -1, -1)
        nuc_h_ESV[:, :, :k_intra] = h_ESV[:, :, 0, :][:, res_of_nuc].unsqueeze(2) + self.E_intra(
            self._intra_type_idx[pos_idx]
        )
        nuc_mask_ij[:, :, :k_intra] = nuc_mask_i.unsqueeze(-1).expand(-1, -1, k_intra)

        # --- inter-codon edges, slots [2, K_TOTAL) --------------------------
        # Residue slot 0 is the self edge; it is consumed above, not tiled here.
        edge_idx_inter = edge_idx[:, :, 1:]
        h_ESV_inter = h_ESV[:, :, 1:]
        qq = torch.arange(3, device=device)

        dst = (edge_idx_inter[:, :, None, :] * 3 + qq[None, None, :, None]).reshape(b, n, k_inter)
        nuc_edge_idx[:, :, k_intra:] = dst[:, res_of_nuc]

        feat = h_ESV_inter.unsqueeze(2).expand(b, n, 3, k_inter_per_q, d_edge).reshape(b, n, k_inter, d_edge)
        feat = feat[:, res_of_nuc] + self.E_inter(qq.repeat_interleave(k_inter_per_q))
        nuc_h_ESV[:, :, k_intra:] = feat

        batch_ar = torch.arange(b, device=device)[:, None, None]
        edge_mask_res = token_mask[:, :, None] * token_mask[batch_ar, edge_idx_inter]
        mask_q3 = edge_mask_res.unsqueeze(2).expand(b, n, 3, k_inter_per_q).reshape(b, n, k_inter)
        nuc_mask_ij[:, :, k_intra:] = mask_q3[:, res_of_nuc]

        return nuc_h_V, nuc_h_ESV, nuc_edge_idx, nuc_mask_i, nuc_mask_ij
