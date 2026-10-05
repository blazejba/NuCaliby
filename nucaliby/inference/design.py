"""Design sequences for a backbone: run the model, then anneal its Potts model.

Supports residue-level and nucleotide-level design:
  "aa"  sample the 32-state residue Potts and read the protein off directly
  "nt"  sample the 4-state nucleotide Potts and translate; this is the branch that
        accepts nucleotide-level guidance

Ensemble conditioning averages the Potts parameters over several structures of the
same protein before sampling.
"""

from typing import Sequence

import torch

from ..constants import TOKENS, aa_idx_to_string, nuc_idx_to_string, translate
from ..modeling import NuCaliby, PottsParams
from .sampling import Guidance, sample

# Non-protein tokens: RNA, DNA and the gap/mask token. Banned when sampling the
# amino-acid Potts, which shares AF3's 32-token alphabet with nucleic acids.
NON_PROTEIN_TOKENS = tuple(range(20, len(TOKENS)))


def potts_for(model: NuCaliby, batch: dict, alphabet: str) -> PottsParams:
    with torch.no_grad():
        out = model(batch)
    return out.aa_potts if alphabet == "aa" else out.nuc_potts


def aggregate(members: Sequence[PottsParams]) -> PottsParams:
    """Average Potts parameters over an ensemble of structures of one protein.

    Averaging energies is a geometric mean in probability space, so the aggregate
    favours sequences that are good for *every* member rather than for one.

    A node survives only if it is present in all members (a stricter rule than for
    a single structure, and a large part of why ensemble recovery drops from ~39%
    to ~29%). An edge survives if any member has it, and its coupling is divided by
    the member count -- so a member lacking that edge contributes zero, not a gap.

    The reference implementation scattered J into a dense [G, N, N, C, C] tensor,
    which is ~1 GB per member at N=500 over the 32-state alphabet and is the
    documented cause of OOM on 16 GB GPUs. Building the union of the members' edge
    sets instead is algebraically identical and roughly an order of magnitude smaller.
    """
    if len(members) == 1:
        return members[0]

    g = len(members)
    n = members[0].h.shape[1]
    device = members[0].h.device
    assert all(m.h.shape[1] == n for m in members), "ensemble members must have equal length"

    h = torch.stack([m.h for m in members]).mean(0)
    mask_i = torch.stack([m.mask_i for m in members]).prod(0)

    # Union of neighbour sets, per node. Absent slots point at the node itself and
    # are masked out, so they contribute nothing.
    per_node = torch.cat([m.edge_idx[0] for m in members], dim=1)  # [N, G*K]
    valid = torch.cat([m.mask_ij[0] for m in members], dim=1).bool()
    self_idx = torch.arange(n, device=device)[:, None]
    per_node = torch.where(valid, per_node, self_idx)

    union, counts = [], []
    for i in range(n):
        u = torch.unique(per_node[i][valid[i]]) if valid[i].any() else self_idx[i]
        union.append(u)
        counts.append(len(u))
    k_union = max(counts)
    edge_idx = self_idx.expand(n, k_union).clone()
    mask_ij = torch.zeros(n, k_union, device=device)
    for i, u in enumerate(union):
        edge_idx[i, : len(u)] = u
        mask_ij[i, : len(u)] = 1.0

    a = members[0].num_states
    J = torch.zeros(n, k_union, a, a, device=device, dtype=members[0].J.dtype)
    rows = torch.arange(n, device=device)[:, None]
    for m in members:
        # [N, K, k_union]: which of this member's slots holds each union neighbour
        match = (m.edge_idx[0][:, :, None] == edge_idx[:, None, :]) & m.mask_ij[0].bool()[:, :, None]
        slot = match.float().argmax(dim=1)
        present = match.any(dim=1)
        J = J + m.J[0][rows, slot] * present[..., None, None]
    J = J / g

    mask_ij = mask_ij * mask_i[0][:, None] * mask_i[0][edge_idx]
    return PottsParams(h, J[None], edge_idx[None], mask_i, mask_ij[None])


def design(
    potts: PottsParams,
    alphabet: str,
    *,
    num_seqs: int = 1,
    num_sweeps: int = 500,
    temperature: float = 0.01,
    guidance: Sequence[Guidance] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Sample `num_seqs` sequences and return one record per sample.

    Records carry the protein sequence, the coding sequence when designing at the
    nucleotide level, the Potts energy, and whether the design contains an internal
    stop codon.
    """
    banned = NON_PROTEIN_TOKENS if alphabet == "aa" else None
    records = []
    for i in range(num_seqs):
        gen = torch.Generator(device=potts.h.device).manual_seed(seed + i)
        S, U = sample(
            potts.h,
            potts.J,
            potts.edge_idx,
            potts.mask_i,
            num_sweeps=num_sweeps,
            temperature=temperature,
            guidance=guidance,
            banned_states=banned,
            generator=gen,
        )
        keep = potts.mask_i[0].bool()
        if alphabet == "aa":
            aa = S[0][keep]
            nt = None
        else:
            nt_keep = keep.repeat_interleave(3) if keep.numel() * 3 == S.shape[1] else keep
            nt = S[0][nt_keep]
            aa = translate(nt)
        records.append(
            {
                "sample": i,
                "seed": seed + i,
                "energy": U.item(),
                "protein": aa_idx_to_string(aa),
                "dna": nuc_idx_to_string(nt) if nt is not None else None,
                "has_internal_stop": bool((aa == 20).any().item()),
            }
        )
    return records
