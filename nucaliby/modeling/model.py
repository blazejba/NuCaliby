"""NuCaliby: a structure-conditioned Potts model at residue and nucleotide level.

    structure -> encoder -> +-- amino-acid decoder -> Potts (32 states)
                            |
                            +-- nucleotide graph lift -> nucleotide decoder
                                                      -> Potts (4 states)

Architecture dimensions are fixed by checkpoint tensor shapes. Changing them
requires training a new model.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import GAP_IDX, N_TOKENS, UNK_IDX
from .graph_ops import cat_neighbors_nodes, gather_nodes
from .modules import DecLayer, EncLayer, TokenFeatures
from .nuc_graph import ResidueToNucleotideGraph
from .potts import GraphPotts

N_CHANNEL = 128
K_NEIGHBORS = 48
N_LAYERS = 3  # encoder, amino-acid decoder, and nucleotide decoder alike
DROPOUT = 0.1
EDGE_N_CHANNEL = 3 * N_CHANNEL  # edges carry (h_E_ij, h_S_j, h_V_j) = 384

# Nucleotide graph: 2 intra-codon edges + 3*(K-1) inter-codon edges per node.
N_INTRA_SLOTS = 2
NUC_K_TOTAL = N_INTRA_SLOTS + 3 * (K_NEIGHBORS - 1)  # 143

AUGMENT_EPS = 0.3  # coordinate noise sigma, training only


@dataclass
class PottsParams:
    """Fields, couplings, graph indices, and masks for a Potts model."""

    h: torch.Tensor  # [B, N, A]
    J: torch.Tensor  # [B, N, K, A, A]
    edge_idx: torch.Tensor  # [B, N, K]
    mask_i: torch.Tensor  # [B, N]
    mask_ij: torch.Tensor  # [B, N, K]

    @property
    def num_states(self) -> int:
        return self.h.shape[-1]

    def to(self, device) -> "PottsParams":
        return PottsParams(*(t.to(device) for t in (self.h, self.J, self.edge_idx, self.mask_i, self.mask_ij)))


@dataclass
class Output:
    logits: torch.Tensor  # [B, N, 32] amino-acid decoder head
    aa_potts: PottsParams
    nuc_logits: torch.Tensor  # [B, 3N, 4] nucleotide decoder head
    nuc_potts: PottsParams


class NuCaliby(nn.Module):
    def __init__(self):
        super().__init__()
        d, d_e = N_CHANNEL, EDGE_N_CHANNEL

        self.token_features = TokenFeatures(N_CHANNEL, K_NEIGHBORS)
        self.W_e = nn.Linear(d, d, bias=False)
        self.W_s = nn.Linear(N_TOKENS, d, bias=False)

        self.encoder_layers = nn.ModuleList([EncLayer(d, d * 2) for _ in range(N_LAYERS)])
        self.decoder_layers = nn.ModuleList([DecLayer(d, d_e) for _ in range(N_LAYERS)])
        self.decoder_S_potts = GraphPotts(d, d_e, N_TOKENS)
        self.W_out = nn.Linear(d, N_TOKENS)

        self.nuc_graph_converter = ResidueToNucleotideGraph()
        # Scale by the 143 nucleotide neighbors to keep message magnitudes
        # independent of node degree.
        self.nuc_decoder_layers = nn.ModuleList([DecLayer(d, d_e, scale=NUC_K_TOTAL) for _ in range(N_LAYERS)])
        self.nuc_potts = GraphPotts(d, d_e, 4)
        self.W_out_nuc = nn.Linear(d, 4)

    # -- checkpoint ---------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "NuCaliby":
        """Load a native state dict or a Lightning checkpoint."""
        obj = torch.load(path, map_location=map_location, weights_only=False)
        state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        prefix = "model.denoiser.atom_mpnn."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        model = cls()
        model.load_state_dict(state, strict=True)
        return model.eval()

    # -- forward ------------------------------------------------------------

    def encode(self, batch: dict):
        """Structure + (masked) sequence -> encoder node/edge features."""
        b, n, _ = batch["restype"].shape
        device = batch["restype"].device

        # Masked residues become the gap token, which doubles as [MASK].
        gap = F.one_hot(torch.full((b, n), GAP_IDX, device=device), num_classes=N_TOKENS).float()
        restype = torch.where(batch["seq_cond_mask"].unsqueeze(-1).bool(), batch["restype"], gap)
        h_S = self.W_s(restype)

        h_E, edge_idx, _ = self.token_features(batch)
        h_V = torch.zeros(b, n, N_CHANNEL, device=device, dtype=h_S.dtype) + h_S
        h_E = self.W_e(h_E)

        token_mask = batch["token_exists_mask"]
        mask_2d = token_mask.unsqueeze(-1) * gather_nodes(token_mask.unsqueeze(-1), edge_idx).squeeze(-1)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, edge_idx, token_mask, mask_2d)
        return h_V, h_E, h_S, edge_idx, mask_2d

    def forward(self, batch: dict) -> Output:
        token_mask = batch["token_exists_mask"]
        h_V, h_E, h_S, edge_idx, mask_2d = self.encode(batch)

        # Snapshot BEFORE the amino-acid decoder: it mutates h_V through residual
        # connections, and the nucleotide branch must see encoder output.
        h_V_enc, h_E_enc = h_V.clone(), h_E.clone()

        # --- amino-acid branch ---------------------------------------------
        h_ESV = cat_neighbors_nodes(h_V, cat_neighbors_nodes(h_S, h_E, edge_idx), edge_idx)
        for layer in self.decoder_layers:
            h_V, h_ESV = layer(h_V, h_ESV, token_mask, edge_idx)

        h_aa, J_aa = self.decoder_S_potts(h_V, h_ESV, edge_idx, token_mask, mask_2d)
        logits = self.W_out(h_V)

        # --- nucleotide branch ---------------------------------------------
        b, n = token_mask.shape
        token_unk_mask = (batch["restype"].argmax(dim=-1) == UNK_IDX).float()

        conv = self.nuc_graph_converter
        nuc_types = conv.nuc_token_types(
            batch.get("nuc_seq"), batch.get("seq_cond_mask"), token_unk_mask, b, n, token_mask.device
        )
        # Edge features carry the neighbour's *nucleotide* identity, pooled over
        # its three codon positions. This is what makes the nucleotide Potts
        # nucleotide-conditioned rather than amino-acid-conditioned.
        h_S_nuc = conv.embed_nuc_tokens(nuc_types).reshape(b, n, 3, -1).mean(dim=2)
        h_ESV_enc = cat_neighbors_nodes(h_V_enc, cat_neighbors_nodes(h_S_nuc, h_E_enc, edge_idx), edge_idx)

        nuc_h_V, nuc_h_ESV, nuc_edge_idx, nuc_mask_i, nuc_mask_ij = conv(
            h_V_enc, h_ESV_enc, edge_idx, token_mask, nuc_types
        )
        for layer in self.nuc_decoder_layers:
            # mask_attend is required here, and must NOT be passed to the
            # amino-acid decoder, or padded residues leak into nucleotide messages.
            nuc_h_V, nuc_h_ESV = layer(nuc_h_V, nuc_h_ESV, nuc_mask_i, nuc_edge_idx, mask_attend=nuc_mask_ij)

        h_nuc, J_nuc = self.nuc_potts(nuc_h_V, nuc_h_ESV, nuc_edge_idx, nuc_mask_i, nuc_mask_ij)

        return Output(
            logits=logits,
            aa_potts=PottsParams(h_aa, J_aa, edge_idx, token_mask, mask_2d),
            nuc_logits=self.W_out_nuc(nuc_h_V),
            nuc_potts=PottsParams(h_nuc, J_nuc, nuc_edge_idx, nuc_mask_i, nuc_mask_ij),
        )
