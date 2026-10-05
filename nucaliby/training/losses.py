"""NuCaliby training objectives.

    L = seq + omega(t) * potts_composite + nuc_aa_consistency + omega(t) * nuc_potts_codon

The nucleotide objectives marginalize probability over every synonymous codon
for the target amino acid, so no single synonymous codon is designated as the
target.
"""

import torch
import torch.nn.functional as F

from ..constants import AA_CODON_MASK, UNK_IDX
from ..modeling.graph_ops import collect_neighbors
from ..modeling.model import N_INTRA_SLOTS, Output, PottsParams
from ..modeling.potts import mask_couplings

LABEL_SMOOTHING = 0.1
T_WEIGHT_ALPHA = 1.25


def omega(t: torch.Tensor) -> torch.Tensor:
    """Potts loss weight (1 - t)^1.25.

    `t` is the probability of keeping a residue visible, so `1 - t` is the mask
    fraction. This weights heavily masked examples more strongly.
    """
    return (1.0 - t).pow(T_WEIGHT_ALPHA)


def _reduce(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum over tokens, divide by the PADDED width -- not the valid-token count.

    Every example is padded to `data.CROP_TOKENS`, so the divisor is constant
    and longer proteins contribute proportionally more total gradient. Changing
    this normalization changes the effective loss scale.
    """
    return (per_token * mask).sum(dim=-1) / mask.shape[1]


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, label_smoothing: float = LABEL_SMOOTHING
) -> torch.Tensor:
    """Label-smoothed CE over `logits` [B, N, C] against `target` [B, N] -> [B]."""
    n_classes = logits.shape[-1]
    target_oh = F.one_hot(target, num_classes=n_classes).float()
    target_oh = (1.0 - label_smoothing) * target_oh + label_smoothing / n_classes
    cel = -(F.log_softmax(logits, dim=-1) * target_oh).sum(dim=-1)
    return _reduce(cel, mask)


def _log_composite_likelihood(
    S: torch.Tensor, potts: PottsParams, smoothing_alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise composite likelihood log p(s_i, s_j | rest) for every edge.

    The pair conditional needs the fields on i and j with J_ij itself removed,
    so it is assembled from the full site
    fields r_i, r_j minus the one coupling that the pair marginal must not
    double-count.

    Returns:
        logp_ij: [B, N, K] log-likelihood of the observed pair on each edge.
        mask_p_ij: [B, N, K] edges that actually exist.
    """
    h, J, edge_idx = potts.h, potts.J, potts.edge_idx
    b, n, k, a, _ = J.shape

    S_j = collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, a, -1)
    J_clamp_j = torch.gather(J, -1, S_j).squeeze(-1)  # [B, i, j, A_i]

    S_i = S.view(b, n, 1, 1, 1).expand(-1, -1, k, a, a)
    J_clamp_i = torch.gather(J, -2, S_i)  # [B, i, j, 1, A_j]

    r_i = h + J_clamp_j.sum(2)
    r_j = collect_neighbors(r_i, edge_idx)

    r_i_minus_ij = r_i.view(b, n, 1, a, 1) - J_clamp_j.unsqueeze(-1)
    r_j_minus_ji = r_j.view(b, n, k, 1, a) - J_clamp_i

    logits_ij = (r_i_minus_ij + r_j_minus_ji + J).view(b, n, k, -1)
    logp = F.log_softmax(-logits_ij, dim=-1).view(b, n, k, a, a)

    logp_j = torch.gather(logp, -1, S_j).squeeze(-1)
    logp_ij = torch.gather(logp_j, -1, S.view(b, n, 1, 1).expand(-1, -1, k, -1)).squeeze(-1)

    if smoothing_alpha > 0.0:
        # Smoothing is applied over the A^2 pair bins; the second term corrects
        # for the observed bin being counted once in the background sum too.
        prob_no_smooth = (1.0 - smoothing_alpha) ** 2
        prob_background = (1.0 - prob_no_smooth) / float(a**2 - 1)
        logp_ij = (prob_no_smooth - prob_background) * logp_ij + prob_background * logp.sum([-2, -1])

    mask_p_ij = mask_couplings(edge_idx, potts.mask_i, potts.mask_ij)
    return mask_p_ij * logp_ij, mask_p_ij


def potts_composite_loss(S: torch.Tensor, potts: PottsParams, label_smoothing: float = LABEL_SMOOTHING) -> torch.Tensor:
    """Amino-acid Potts composite pseudo-likelihood [B].

    The 2.0 in the denominator undoes the double-count of each undirected pair,
    which appears once as i->j and once as j->i.
    """
    logp_ij, mask_p_ij = _log_composite_likelihood(S, potts, label_smoothing)
    logp_i = potts.mask_i * (mask_p_ij * logp_ij).sum(-1) / (2.0 * mask_p_ij.sum(-1) + 1e-3)
    return -_reduce(logp_i, potts.mask_i)


def _log_codon_probs(nuc_logits: torch.Tensor) -> torch.Tensor:
    """Independent per-position nucleotide logits [B, 3N, 4] -> log p(codon) [B, N, 64].

    Codon index is c0*16 + c1*4 + c2, matching `constants.CODON_TO_AA_IDX`.
    """
    b, n3, _ = nuc_logits.shape
    log_p = F.log_softmax(nuc_logits, dim=-1).view(b, n3 // 3, 3, 4)
    return (log_p[:, :, 0, :, None, None] + log_p[:, :, 1, None, :, None] + log_p[:, :, 2, None, None, :]).reshape(
        b, n3 // 3, 64
    )


def codon_aa_consistency_loss(
    nuc_logits: torch.Tensor,
    target_restype: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = LABEL_SMOOTHING,
) -> torch.Tensor:
    """CE between the codon-marginal amino-acid distribution and the native AA [B].

    log p(a) = logsumexp over syn(a) of log p(codon), so the loss is blind to
    which synonymous codon carries the mass. This is the nucleotide decoder's
    only tie to the amino-acid target.
    """
    log_codon = _log_codon_probs(nuc_logits)
    syn = AA_CODON_MASK.to(nuc_logits.device)  # [21, 64]
    log_aa = torch.logsumexp(log_codon.unsqueeze(2).masked_fill(~syn, float("-inf")), dim=-1)  # [B, N, 21]

    target_oh = F.one_hot(target_restype, num_classes=21).float()
    target_oh = (1.0 - label_smoothing) * target_oh + label_smoothing / 21
    return _reduce(-(log_aa * target_oh).sum(dim=-1), mask)


def codon_marginalised_potts_loss(
    nuc_seq: torch.Tensor,
    target_restype: torch.Tensor,
    potts: PottsParams,
    label_smoothing: float = LABEL_SMOOTHING,
) -> torch.Tensor:
    """Codon-block pseudo-likelihood, marginalised over synonymous codons [B].

    Scores whole codons rather than single nucleotides: the block energy of codon
    c at residue r, conditioned on every other codon, is

        E_r(c) = E_h(c) + 0.5 * E_intra(c) + E_inter(c)

    and the target is the *set* syn(a_r), giving -log P(codon_r in syn(a_r) | rest).
    Probability is therefore summed over all synonymous codons for each target
    residue rather than selecting a single codon as correct.

    The 0.5 halves the six directed intra-codon edges into three pairs; inter-codon
    edges are not halved because their far end is clamped to the observed
    neighbour, so they are a field on the block, not a pair inside it.
    """
    h, J = potts.h, potts.J
    b, n3, _ = h.shape
    n = n3 // 3
    k_intra = N_INTRA_SLOTS

    h_codons = h.view(b, n, 3, 4)
    E_h = (
        h_codons[:, :, 0, :, None, None] + h_codons[:, :, 1, None, :, None] + h_codons[:, :, 2, None, None, :]
    ).reshape(b, n, 64)

    # Intra-codon slots, laid out by `nuc_graph._intra_dst_q`:
    #   p=0: slot 0 -> q=1, slot 1 -> q=2
    #   p=1: slot 0 -> q=0, slot 1 -> q=2
    #   p=2: slot 0 -> q=0, slot 1 -> q=1
    # Transposes bring every matrix into (lower position, higher position) order.
    J_intra = J[:, :, :k_intra].reshape(b, n, 3, k_intra, 4, 4)
    c0c1 = J_intra[:, :, 0, 0] + J_intra[:, :, 1, 0].transpose(-1, -2)
    c0c2 = J_intra[:, :, 0, 1] + J_intra[:, :, 2, 0].transpose(-1, -2)
    c1c2 = J_intra[:, :, 1, 1] + J_intra[:, :, 2, 1].transpose(-1, -2)
    E_intra = 0.5 * (c0c1.unsqueeze(-1) + c0c2.unsqueeze(-2) + c1c2.unsqueeze(-3)).reshape(b, n, 64)

    S_j = collect_neighbors(nuc_seq.unsqueeze(-1), potts.edge_idx)[:, :, k_intra:]  # [B, 3N, K_inter, 1]
    J_inter = torch.gather(J[:, :, k_intra:], -1, S_j.unsqueeze(-1).expand(-1, -1, -1, 4, -1)).squeeze(-1)
    J_inter = (J_inter * potts.mask_ij[:, :, k_intra:].unsqueeze(-1)).sum(dim=2).view(b, n, 3, 4)
    E_inter = (
        J_inter[:, :, 0, :, None, None] + J_inter[:, :, 1, None, :, None] + J_inter[:, :, 2, None, None, :]
    ).reshape(b, n, 64)

    neg_E = -(E_h + E_intra + E_inter)
    syn_mask = AA_CODON_MASK.to(h.device)[target_restype]  # [B, N, 64]
    res_mask = potts.mask_i.view(b, n, 3).min(dim=-1).values

    if label_smoothing > 0:
        # Soft target: uniform over syn(a_r), smoothed towards uniform over all 64.
        target = syn_mask.float() / syn_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
        target = (1.0 - label_smoothing) * target + label_smoothing / 64.0
        nll = -(target * F.log_softmax(neg_E, dim=-1)).sum(dim=-1)
    else:
        nll = torch.logsumexp(neg_E, dim=-1) - torch.logsumexp(neg_E.masked_fill(~syn_mask, float("-inf")), dim=-1)

    return _reduce(nll, res_mask)


def nucaliby_loss(out: Output, batch: dict, t: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total training loss and the four unweighted components, for logging.

    `t` is the per-example keep probability that produced `batch["seq_cond_mask"]`.
    """
    target_restype = batch["restype"].argmax(dim=-1)
    token_mask = batch["token_exists_mask"]
    not_unk = (target_restype != UNK_IDX).float()

    # Losses are scored only where the model had to guess, i.e. masked positions.
    seq_mask = token_mask * (1.0 - batch["seq_cond_mask"]) * not_unk

    # AA_CODON_MASK row 20 represents both stop codons and UNK, so an UNK
    # residue would otherwise be trained to emit a stop. Drop it from every
    # nucleotide loss by hand -- the table cannot express the difference.
    nuc_potts = PottsParams(
        out.nuc_potts.h,
        out.nuc_potts.J,
        out.nuc_potts.edge_idx,
        out.nuc_potts.mask_i * not_unk.repeat_interleave(3, dim=-1),
        out.nuc_potts.mask_ij,
    )

    w = omega(t)
    parts = {
        "seq_loss": masked_cross_entropy(out.logits, target_restype, seq_mask),
        "potts_composite_loss": potts_composite_loss(target_restype, out.aa_potts),
        "nuc_aa_consistency_loss": codon_aa_consistency_loss(out.nuc_logits, target_restype, seq_mask),
        "nuc_potts_codon_loss": codon_marginalised_potts_loss(batch["nuc_seq"], target_restype, nuc_potts),
    }
    total = (
        parts["seq_loss"]
        + w * parts["potts_composite_loss"]
        + parts["nuc_aa_consistency_loss"]
        + w * parts["nuc_potts_codon_loss"]
    ).mean()
    return total, {k: v.mean().detach() for k, v in parts.items()}
