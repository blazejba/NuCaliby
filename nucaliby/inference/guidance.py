"""Composable inference-time energy biases.

Each term is a function of a soft one-hot sequence returning one energy per batch
element, in the same units and sign convention as the Potts energy (lower is
better):

    E_bias : [B, L, C] -> [B]

Differentiable terms are evaluated on a relaxed one-hot inside the DLMC proposal,
and autograd supplies the straight-through gradient that shifts the per-site energy
gaps. Acceptance-only terms (`differentiable = False`) are evaluated on hard
sequences inside the Metropolis-Hastings ratio instead, which is what lets a
black-box scorer with no analytic gradient steer the chain.

Terms compose by summation, so several objectives can be applied in one run.

Specs are parsed from strings so CLIs stay flat:

    stop                            internal stop-codon penalty, default weight
    stop:weight=200
    tai:weight=50,organism=bl21_de3
    motif:file=assets/motifs/sira.txt,weight=50,beta=2
    vienna:weight=10                acceptance-only
    lcp                             low-complexity penalty (amino-acid branch only)
"""

import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from ..constants import NUC_TOKENS, dna_to_idx

ASSETS = Path(__file__).resolve().parents[1] / "assets"

STOP_WEIGHT = 100.0
TAI_WEIGHT = 50.0
MOTIF_WEIGHT, MOTIF_BETA = 50.0, 2.0
VIENNA_WEIGHT = 10.0

# A window of 30 residues whose effective
# alphabet size falls below exp(2.32) ~ 10 is penalised quadratically.
LCP_WINDOW = 30
LCP_ENTROPY_MIN = 2.32
LCP_MIN_COVERAGE = 0.9


class Guidance:
    """Base class. Subclasses set `differentiable` and implement `energy`."""

    differentiable: bool = True
    name: str = "guidance"

    def __call__(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        return self.energy(seq_onehot)

    def energy(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StopCodonPenalty(Guidance):
    """Penalise internal stop codons, proportionally to their soft probability.

    Nucleotide branch only: it reads the sequence as consecutive codons.
    """

    name = "stop"

    def __init__(self, weight: float = STOP_WEIGHT):
        self.weight = weight

    def energy(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        b, ln, _ = seq_onehot.shape
        p = seq_onehot.reshape(b, ln // 3, 3, 4)
        p0, p1, p2 = p[:, :, 0], p[:, :, 1], p[:, :, 2]
        # TAA, TAG, TGA with A=0, C=1, G=2, T=3
        stop = (
            p0[..., 3] * p1[..., 0] * p2[..., 0]
            + p0[..., 3] * p1[..., 0] * p2[..., 2]
            + p0[..., 3] * p1[..., 2] * p2[..., 0]
        )
        return self.weight * stop.sum(-1)


class TaiBias(Guidance):
    """Push codon usage toward the host tRNA pool.

    E_tAI(c) = -w_tAI(c), averaged over codons. The expected log-weight is taken
    through the codon's product structure, so the bias stays differentiable in the
    relaxed one-hot.
    """

    name = "tai"

    def __init__(self, weight: float = TAI_WEIGHT, organism: str = "bl21_de3"):
        self.weight = weight
        self.organism = organism
        self.log_w = _tai_log_weights(organism)

    def energy(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        b, ln, _ = seq_onehot.shape
        p = seq_onehot.reshape(b, ln // 3, 3, 4)
        log_w = self.log_w.to(seq_onehot.device)
        expected = torch.einsum("bnx,bny,bnz,xyz->bn", p[:, :, 0], p[:, :, 1], p[:, :, 2], log_w)
        return -self.weight * expected.mean(-1)


class MotifBias(Guidance):
    """Embed a sequence motif anywhere in the design.

    The Hamming distance to the motif is computed at every window by convolution,
    and the windows are combined with a smooth minimum

        E = -(lambda / beta) * log sum_mu exp(-beta * d_mu)

    so the sampler is free to discover the placement rather than being told one.
    Larger beta approaches a hard minimum over windows.
    """

    name = "motif"

    def __init__(self, motif: str | Path, weight: float = MOTIF_WEIGHT, beta: float = MOTIF_BETA, valid_len=None):
        idx = dna_to_idx(_read_motif(motif))
        self.motif = F.one_hot(idx, num_classes=len(NUC_TOKENS)).float()
        self.weight, self.beta = weight, beta
        self.valid_len = valid_len

    def energy(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        b, ln, _ = seq_onehot.shape
        m = self.motif.to(seq_onehot.device)
        lm = m.shape[0]
        if ln < lm:
            # The motif cannot fit. Return the constant worst-case energy rather
            # than letting conv1d raise, so a short chain in a batch is survivable.
            return self.weight * torch.full((b,), float(lm), device=seq_onehot.device)
        scores = F.conv1d(seq_onehot.transpose(1, 2), m.T.unsqueeze(0)).squeeze(1)  # [B, W]

        hamming = lm - scores
        valid_len = self.valid_len if self.valid_len is not None else torch.full((b,), ln, device=seq_onehot.device)
        windows = torch.arange(scores.shape[1], device=seq_onehot.device)
        # A window running off the end of the real sequence is not a placement.
        outside = (windows[None, :] + lm) > valid_len.to(seq_onehot.device)[:, None]
        hamming = hamming.masked_fill(outside, float(lm))

        return self.weight * -torch.logsumexp(-self.beta * hamming, dim=-1) / self.beta

    def min_hamming(self, seq_idx: torch.Tensor) -> torch.Tensor:
        """Exact minimum Hamming distance for a hard sequence."""
        onehot = F.one_hot(seq_idx, num_classes=len(NUC_TOKENS)).float()
        m = self.motif.to(onehot.device)
        scores = F.conv1d(onehot.transpose(1, 2), m.T.unsqueeze(0)).squeeze(1)
        return (m.shape[0] - scores).min(dim=-1).values


class ViennaFreeEnergy(Guidance):
    """Bias designs toward thermodynamically stable mRNA.

    The ViennaRNA ensemble free energy has no closed form gradient with respect to
    a discrete sequence, so this term is acceptance-only: the sampler evaluates it
    at the current and proposed sequence and folds the difference into the
    Metropolis-Hastings ratio. Normalised per nucleotide so the scale does not
    drift with protein length.
    """

    name = "vienna"
    differentiable = False

    def __init__(self, weight: float = VIENNA_WEIGHT, valid_len=None):
        self.weight = weight
        self.valid_len = valid_len

    def energy(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        import RNA  # lazy: ViennaRNA is an optional dependency

        idx = seq_onehot.argmax(-1).cpu()
        b, ln = idx.shape
        lens = self.valid_len.cpu().tolist() if self.valid_len is not None else [ln] * b
        out = []
        for row, n in zip(idx, lens):
            seq = "".join("ACGU"[int(c)] for c in row[: int(n)])
            try:
                _, dg = RNA.pf_fold(seq)
            except Exception:
                dg = 0.0
            out.append(dg / max(int(n), 1))
        return self.weight * torch.tensor(out, device=seq_onehot.device, dtype=torch.float32)


class LowComplexityPenalty(Guidance):
    """Discourage low-complexity stretches in the designed protein.

    Amino-acid branch only. A four-letter nucleotide alphabet is intrinsically
    low complexity under this objective.
    """

    name = "lcp"

    def __init__(self, mask: torch.Tensor | None = None):
        self.mask = mask

    def energy(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        b, n, a = seq_onehot.shape
        device = seq_onehot.device
        w = min(LCP_WINDOW, n)
        mask_i = self.mask.to(device) if self.mask is not None else torch.ones(b, n, device=device)

        offsets = torch.arange(w, device=device) - w // 2
        idx = torch.arange(n, device=device)[None, :, None] + offsets[None, None, :]
        inside = (idx >= 0) & (idx < n)
        idx = idx.clamp(0, n - 1).expand(b, -1, -1)

        mask_j = torch.gather(mask_i, 1, idx.reshape(b, -1)).reshape(b, n, w)
        mask_ij = inside.float() * mask_j * mask_i[..., None]

        counts = (
            mask_ij[..., None]
            * torch.gather(
                (mask_i[..., None] * seq_onehot), 1, idx.reshape(b, -1)[..., None].expand(-1, -1, a)
            ).reshape(b, n, w, a)
        ).sum(2)

        covered = counts.sum(-1) > LCP_MIN_COVERAGE * w
        p = counts / (counts.sum(-1, keepdim=True) + 1e-5)
        entropy = -(p * torch.log(p + 1e-11)).sum(-1)
        penalty = covered * (entropy.exp() - math.exp(LCP_ENTROPY_MIN)).clamp(max=0).square()
        return (mask_i * penalty).sum(1)


# --- construction -----------------------------------------------------------


def _read_motif(motif: str | Path) -> str:
    """Accept either a literal sequence or a path to a one-line motif file."""
    text = str(motif)
    if set(text.strip().upper()) <= set("ACGTUN") and len(text.strip()) > 3:
        return text.strip()
    path = Path(text)
    if not path.exists():
        path = ASSETS / "motifs" / f"{text}.txt"
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith((">", "#")):
            return line
    raise ValueError(f"no motif sequence found in {path}")


def _tai_log_weights(organism: str) -> torch.Tensor:
    """[4, 4, 4] log relative-adaptiveness, indexed by (base0, base1, base2).

    ATG and stop codons stay at 0 so the bias is neutral there: the start codon is
    not a free choice, and stop codons are handled by their own penalty.
    """
    from .tai import codon_weights

    weights = codon_weights(organism)
    log_w = torch.zeros(4, 4, 4)
    for i, b0 in enumerate(NUC_TOKENS):
        for j, b1 in enumerate(NUC_TOKENS):
            for k, b2 in enumerate(NUC_TOKENS):
                w = weights.get(b0 + b1 + b2)
                if w and w > 0:
                    log_w[i, j, k] = math.log(w)
    return log_w


_REGISTRY: dict[str, Callable[..., Guidance]] = {
    "stop": StopCodonPenalty,
    "tai": TaiBias,
    "motif": MotifBias,
    "vienna": ViennaFreeEnergy,
    "lcp": LowComplexityPenalty,
}


def build(spec: str, *, mask: torch.Tensor | None = None) -> Guidance:
    """Build one guidance term from a spec string such as `tai:weight=50`.

    `mask` is the Potts node mask [B, L]; terms that must not run off the end of a
    padded sequence take their valid lengths from it.
    """
    name, _, rest = spec.partition(":")
    name = name.strip()
    if name not in _REGISTRY:
        raise ValueError(f"unknown guidance '{name}'; choose from {sorted(_REGISTRY)}")

    kwargs: dict[str, object] = {}
    for field in filter(None, (f.strip() for f in rest.split(","))):
        key, _, value = field.partition("=")
        try:
            kwargs[key] = float(value) if "." in value or value.lstrip("-").isdigit() else value
        except ValueError:
            kwargs[key] = value

    cls = _REGISTRY[name]
    if cls is LowComplexityPenalty:
        return cls(mask=mask)
    if cls in (MotifBias, ViennaFreeEnergy) and mask is not None:
        kwargs.setdefault("valid_len", mask.sum(-1))
    if cls is MotifBias and "file" in kwargs:
        kwargs["motif"] = kwargs.pop("file")
    return cls(**kwargs)


def build_all(specs, *, mask: torch.Tensor | None = None) -> list[Guidance]:
    return [build(s, mask=mask) for s in specs or []]
