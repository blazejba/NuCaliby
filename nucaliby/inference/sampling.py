"""Sampling from a Potts model with Discrete Langevin Monte Carlo.

DLMC (Sun et al. 2023) computes per-site energy gaps for every candidate state and
turns them into local transition probabilities, so a full sweep updates all sites
at once instead of one at a time.

Guidance supports two modes:

  differentiable    E_bias is evaluated on a soft one-hot relaxation and its
                    gradient is added to the per-site energy gaps (straight-through
                    estimator). Cheap; used for tAI, motifs, stop codons.

  acceptance-only   E_bias is evaluated on hard sequences only, inside the
                    Metropolis-Hastings ratio. Needs no gradient, so it supports
                    black-box scorers like the ViennaRNA partition function.

Temperature anneals linearly from `temperature_init` to `temperature` over the
first 80% of sweeps, then holds.
"""

from typing import Callable, Protocol, Sequence

import torch
import torch.nn.functional as F

from ..modeling.potts import potts_energy

ANNEALING_FRACTION = 0.8
DLMC_DT = 0.1


class Guidance(Protocol):
    """An auxiliary energy term. Lower is better, like the Potts energy itself."""

    differentiable: bool

    def __call__(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        """[B, L, C] (soft) one-hot -> [B] energy."""
        ...


def _combined(guidance: Sequence[Guidance] | None, differentiable: bool) -> Callable | None:
    """Sum the guidance terms of one kind, or None if there are none."""
    terms = [g for g in (guidance or []) if g.differentiable == differentiable]
    if not terms:
        return None
    return lambda x: sum(g(x) for g in terms)


def _dlmc_proposal(S, h, J, edge_idx, T, bias=None):
    """Return (total energy, log transition probabilities [B, N, A])."""
    U, U_i = potts_energy(S, h, J, edge_idx)
    A = h.shape[-1]

    if bias is not None:
        with torch.enable_grad():
            one_hot = F.one_hot(S, A).float().requires_grad_(True)
            U_bias = bias(one_hot)
            grad = torch.autograd.grad(U_bias.sum(), [one_hot])[0].detach()
        # Centre on the current state: only *differences* between candidate states
        # matter, and centring keeps the gaps on the same scale as the Potts term.
        U_i = U_i + grad - torch.gather(grad, -1, S[..., None])
        U = U + U_bias.detach()

    logP_j = F.log_softmax(-U_i / T, dim=-1)
    logP_i = torch.gather(logP_j, -1, S[..., None])

    # Sigmoid balancing function: Q_ij = sigmoid(logP_j - logP_i), the locally
    # balanced choice that keeps the chain reversible w.r.t. the Potts measure.
    log_Q_ij = F.logsigmoid(logP_j - logP_i)
    rate = torch.exp(log_Q_ij - logP_j)

    one_hot = F.one_hot(S, A).float()
    logP_ij = logP_j + (-(-DLMC_DT * rate).expm1()).log()
    p_flip = ((1.0 - one_hot) * logP_ij.exp()).sum(-1, keepdim=True)
    logP_ii = (1.0 - p_flip).clamp(1e-5).log()
    return U, (1.0 - one_hot) * logP_ij + one_hot * logP_ii


@torch.no_grad()
def sample(
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.Tensor,
    mask_i: torch.Tensor,
    *,
    num_sweeps: int = 500,
    temperature: float = 0.01,
    temperature_init: float = 1.0,
    guidance: Sequence[Guidance] | None = None,
    banned_states: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anneal a Potts chain and return (sequence [B, N], energy [B]).

    Positions with `mask_i == 0` are padding and are frozen at state 0.
    """
    b, n, a = h.shape
    device = h.device

    allowed_states = None
    if banned_states:
        # Banned states are pushed to a large finite energy, never -inf: the DLMC
        # arithmetic takes differences of log-probabilities and calls expm1, both
        # of which produce NaN if any entry is infinite.
        h = h.clone()
        h[:, :, list(banned_states)] = h.max() + 1e3 * max(1.0, temperature)
        banned = set(banned_states)
        allowed_states = torch.tensor([i for i in range(a) if i not in banned], device=device)

    if allowed_states is None:
        S = torch.randint(0, a, (b, n), device=device, generator=generator)
    else:
        S = allowed_states[torch.randint(0, len(allowed_states), (b, n), device=device, generator=generator)]
    S = torch.where(mask_i.bool(), S, torch.zeros_like(S))

    diff_bias = _combined(guidance, differentiable=True)
    acc_bias = _combined(guidance, differentiable=False)

    n_anneal = int(ANNEALING_FRACTION * num_sweeps)
    temperatures = torch.cat(
        [
            torch.linspace(temperature_init, temperature, n_anneal),
            torch.full((num_sweeps - n_anneal,), temperature),
        ]
    ).tolist()

    for T in temperatures:
        U, logP = _dlmc_proposal(S, h, J, edge_idx, T, diff_bias)
        probs = torch.softmax(logP, dim=-1)
        S_new = torch.multinomial(probs.reshape(-1, a), 1, generator=generator).reshape(b, n)
        S_new = torch.where(mask_i.bool(), S_new, S)

        if acc_bias is None:
            S = S_new
            continue

        # Metropolis-Hastings on the full proposal, so a non-differentiable bias
        # can steer the chain using only its values at S and S'.
        U_new, logP_new = _dlmc_proposal(S_new, h, J, edge_idx, T, diff_bias)
        E_old = acc_bias(F.one_hot(S, a).float())
        E_new = acc_bias(F.one_hot(S_new, a).float())

        def flux(energy, logp, target, bias_energy):
            moved = (mask_i * torch.gather(logp, -1, target[..., None])[..., 0]).sum(1)
            return -(energy + bias_energy) / T + moved

        log_acc = flux(U_new, logP_new, S, E_new) - flux(U, logP, S_new, E_old)
        accept = torch.bernoulli(log_acc.exp().clamp(max=1.0), generator=generator)
        S = torch.where(accept[:, None] > 0, S_new, S)

    U, _ = potts_energy(S, h, J, edge_idx)
    return S, U
