"""Distributed PyTorch training loop for NuCaliby.

The model is trained end to end. Checkpoints contain both raw and
exponential-moving-average weights; use the EMA checkpoint for evaluation.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..modeling import NuCaliby
from .data import DEFAULT_EXCLUDE_IDS, DEFAULT_VALIDATION_IDS, SolubleShards, make_dataloader
from .losses import nucaliby_loss

BATCH_SIZE = 4  # per device
MAX_STEPS = 120000
SEED = 0
NUM_WORKERS = 8

NOAM_FACTOR = 2.0
NOAM_WARMUP = 4000
NOAM_MODEL_SIZE = 128  # MPNN hidden width
ADAM_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-8
GRAD_CLIP = 0.0

EMA_DECAY = 0.99

LOG_EVERY = 500
CKPT_EVERY = 2500
VAL_EVERY = 6250
VAL_BATCHES = 64


def noam_lr(step: int) -> float:
    """Noam schedule: linear warmup then inverse-sqrt decay.

    `step` counts completed optimizer steps. The clamp at 1 defines the learning
    rate before the first update.
    """
    step = max(step, 1)
    return NOAM_FACTOR * NOAM_MODEL_SIZE**-0.5 * min(step**-0.5, step * NOAM_WARMUP**-1.5)


class EMA:
    """Shadow copy of the weights, updated as `ema <- d*ema + (1-d)*w` every step.

    No bias correction or decay warmup is applied. Non-float entries are copied
    rather than averaged.
    """

    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if shadow.is_floating_point():
                shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(v)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state.items()}


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(path: Path, model: NuCaliby, ema: EMA, optimizer: torch.optim.Optimizer, step: int) -> None:
    """Write raw and EMA weights as two files, both loadable by `from_checkpoint`.

    It keys on "state_dict", so the extra resume state riding along in the raw
    file is ignored on load. Evaluate the `-ema` file, not this one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )
    torch.save({"state_dict": ema.state_dict(), "step": step}, path.with_name(path.stem + "-ema.pt"))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, max_batches: int = VAL_BATCHES) -> dict:
    """Mean held-out loss over the first `max_batches` validation batches.

    Validation examples carry the same masking and augmentation fields as training
    examples. Dropout is disabled while evaluating.
    """
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        if n_batches == max_batches:
            break
        batch = to_device(batch, device)
        t = batch["t"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total, parts = nucaliby_loss(model(batch), batch, t)
        for k, v in {"loss": total, **parts}.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1

    model.train(was_training)
    means = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if dist.is_initialized():
        # Ranks hold different shards, so the reported number must be the mean of
        # the means, not rank 0's slice.
        keys = sorted(means)
        stacked = torch.tensor([means[k] for k in keys], device=device)
        dist.all_reduce(stacked, op=dist.ReduceOp.AVG)
        means = dict(zip(keys, stacked.tolist()))
    return means


def _ddp_setup() -> tuple[int, int, torch.device]:
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; a bare `python` run sets none."""
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), torch.device("cuda", local_rank)


def _epochs(loader):
    """Yield batches forever, reshuffling each pass."""
    for epoch in range(1 << 30):
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from loader


def train(
    out_dir: str,
    data_root: str,
    batch_size: int = BATCH_SIZE,
    max_steps: int = MAX_STEPS,
    resume: str | None = None,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
    validation_ids: str | Path = DEFAULT_VALIDATION_IDS,
    exclude_ids: tuple[str | Path, ...] = DEFAULT_EXCLUDE_IDS,
) -> None:
    rank, world_size, device = _ddp_setup()
    # Per-rank seed: a shared one gives every rank the same masks and coordinate
    # noise. Weight init still agrees, because DDP broadcasts rank 0's copy.
    torch.manual_seed(seed + rank)

    model = NuCaliby().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=noam_lr(0), betas=ADAM_BETAS, eps=ADAM_EPS)
    ema = EMA(model)

    step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        step = ckpt["step"]

    train_loader = make_dataloader(
        SolubleShards(
            data_root,
            phase="train",
            seed=seed + rank,
            validation_ids=validation_ids,
            exclude_ids=exclude_ids,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = make_dataloader(
        SolubleShards(
            data_root,
            phase="val",
            seed=seed + rank,
            validation_ids=validation_ids,
            exclude_ids=exclude_ids,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    net = model
    if world_size > 1:
        net = DistributedDataParallel(model, device_ids=[device.index])
    net.train()

    ckpt_dir = Path(out_dir) / "checkpoints"
    running: dict[str, float] = {}
    n_running = 0
    t_last = time.time()

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    log(f"rank {rank}/{world_size} | {sum(p.numel() for p in model.parameters())} params | resuming at step {step}")

    for batch in _epochs(train_loader):
        if step >= max_steps:
            break

        lr = noam_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = to_device(batch, device)
        t = batch["t"]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total, parts = nucaliby_loss(net(batch), batch, t)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        ema.update(model)
        step += 1

        for k, v in {"loss": total, **parts}.items():
            running[k] = running.get(k, 0.0) + float(v)
        n_running += 1

        if step % LOG_EVERY == 0:
            per_step = (time.time() - t_last) / n_running
            terms = " ".join(f"{k} {v / n_running:.4f}" for k, v in sorted(running.items()))
            log(f"step {step:>7d} lr {lr:.3e} {terms} | {per_step * 1e3:.0f} ms/step")
            running, n_running, t_last = {}, 0, time.time()

        if step % VAL_EVERY == 0:
            means = evaluate(net, val_loader, device)
            log(f"step {step:>7d} val " + " ".join(f"{k} {v:.4f}" for k, v in sorted(means.items())))
            t_last = time.time()  # exclude validation from the ms/step figure

        if step % CKPT_EVERY == 0 and rank == 0:
            save_checkpoint(ckpt_dir / f"step{step}.pt", model, ema, optimizer, step)

    if rank == 0:
        save_checkpoint(ckpt_dir / f"step{step}.pt", model, ema, optimizer, step)
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NuCaliby")
    ap.add_argument("--out-dir", required=True, help="checkpoints go to <out-dir>/checkpoints/")
    ap.add_argument("--data-root", required=True, help="store produced by nucaliby.preprocessing")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="per device")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--resume", help="path to a stepN.pt checkpoint")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--validation-ids", default=str(DEFAULT_VALIDATION_IDS))
    ap.add_argument(
        "--exclude-ids",
        action="append",
        dest="exclude_ids",
        help="exclusion-list path; repeat for multiple lists (defaults to all packaged lists)",
    )
    args = ap.parse_args()
    train(
        out_dir=args.out_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        resume=args.resume,
        num_workers=args.num_workers,
        seed=args.seed,
        validation_ids=args.validation_ids,
        exclude_ids=tuple(args.exclude_ids) if args.exclude_ids else DEFAULT_EXCLUDE_IDS,
    )


if __name__ == "__main__":
    main()
