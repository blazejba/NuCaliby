"""Design sequences for one or many backbones.

    # Amino-acid design
    python scripts/inference/design.py --ckpt nucaliby.pt --pdb-dir $CLEAN_PDB_DIR \
        --test-ids assets/test_pdb_ids.txt --alphabet aa --guide lcp --out designs.csv

    # Nucleotide design
    python scripts/inference/design.py --ckpt nucaliby.pt --pdb-dir $CLEAN_PDB_DIR \
        --test-ids assets/test_pdb_ids.txt --alphabet nt --guide stop:weight=100 --out designs.csv

    # tAI-guided, E. coli BL21(DE3)
    ... --alphabet nt --guide stop:weight=100 --guide tai:weight=50,organism=bl21_de3

    # embed the SiRA aptamer in the coding sequence
    ... --alphabet nt --guide stop:weight=200 --guide motif:file=assets/motifs/sira.txt,weight=50,beta=2

    # Ensemble conditioning: add the same flag to either command
    ... --ensemble-dir $ENSEMBLE_DIR
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch

from nucaliby.constants import aa_idx_to_string
from nucaliby.inference.design import aggregate, design, potts_for
from nucaliby.inference.guidance import build_all
from nucaliby.inference.structure import load_structure
from nucaliby.inference.tai import sequence_tai
from nucaliby.modeling import NuCaliby

FIELDS = [
    "pdb_id", "chain", "sample", "seed", "alphabet", "guidance", "n_members",
    "energy", "recovery", "gc", "tai", "has_internal_stop", "protein", "dna", "native_aa",
]  # fmt: skip


def ensemble_paths(ensemble_dir: str | None, pdb_id: str) -> list[Path]:
    """Structure variants for one backbone, or [] if there is no ensemble."""
    if not ensemble_dir:
        return []
    d = Path(ensemble_dir) / pdb_id
    return sorted(p for p in d.glob("*.pdb")) + sorted(p for p in d.glob("*.cif")) if d.is_dir() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pdb", nargs="*", default=[], help="explicit structure files")
    ap.add_argument("--pdb-dir", default=os.environ.get("CLEAN_PDB_DIR"))
    ap.add_argument("--test-ids", help="file of test PDB IDs, one per line; used with --pdb-dir")
    ap.add_argument("--ensemble-dir", default=None, help="<dir>/<pdb_id>/*.pdb structure variants")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphabet", choices=["aa", "nt"], default="nt")
    ap.add_argument(
        "--guide",
        action="append",
        default=[],
        metavar="SPEC",
        help="repeatable, e.g. stop:weight=100 or tai:weight=50,organism=bl21_de3",
    )
    ap.add_argument("--num-seqs", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--chains", default="A", help="comma-separated, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.alphabet == "aa" and any(g.split(":")[0] in {"stop", "tai", "motif", "vienna"} for g in args.guide):
        ap.error("nucleotide guidance requires --alphabet nt")
    if args.alphabet == "nt" and any(g.split(":")[0] == "lcp" for g in args.guide):
        ap.error("the low-complexity penalty is defined on the amino-acid alphabet; drop it or use --alphabet aa")

    chains = None if args.chains == "all" else tuple(args.chains.split(","))

    targets: list[tuple[str, Path]] = [(Path(p).stem, Path(p)) for p in args.pdb]
    if args.test_ids:
        ids = [ln.strip() for ln in Path(args.test_ids).read_text().split() if ln.strip()]
        for pid in ids:
            hit = next(
                (
                    Path(args.pdb_dir) / f"{pid}{s}"
                    for s in (".cif", ".pdb")
                    if (Path(args.pdb_dir) / f"{pid}{s}").exists()
                ),
                None,
            )
            if hit is None:
                print(f"  skip {pid}: no structure in {args.pdb_dir}")
                continue
            targets.append((pid, hit))
    if not targets:
        ap.error("no input structures; pass --pdb or --pdb-dir with --test-ids")

    model = NuCaliby.from_checkpoint(args.ckpt).to(args.device)
    print(f"{len(targets)} backbones | alphabet={args.alphabet} | guidance={args.guide or ['none']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        for i, (pdb_id, path) in enumerate(targets, 1):
            t0 = time.time()
            try:
                members = [load_structure(p, chains=chains) for p in [path, *ensemble_paths(args.ensemble_dir, pdb_id)]]
                params = [
                    potts_for(model, {k: v.to(args.device) for k, v in b.items()}, args.alphabet) for b in members
                ]
                potts = aggregate(params)

                native = members[0]["restype"][0].argmax(-1)
                guidance = build_all(args.guide, mask=potts.mask_i)
                records = design(
                    potts,
                    args.alphabet,
                    num_seqs=args.num_seqs,
                    num_sweeps=args.sweeps,
                    temperature=args.temperature,
                    guidance=guidance,
                    seed=args.seed,
                )
            except Exception as exc:  # one bad structure must not sink a complete evaluation sweep
                print(f"[{i}/{len(targets)}] {pdb_id}: FAILED ({type(exc).__name__}: {exc})")
                continue

            for r in records:
                designed = r["protein"]
                keep = min(len(designed), len(native))
                native_str = aa_idx_to_string(native)[:keep]
                r["recovery"] = sum(a == b for a, b in zip(designed[:keep], native_str)) / max(keep, 1)
                r["gc"] = (r["dna"].count("G") + r["dna"].count("C")) / len(r["dna"]) if r["dna"] else None
                r["tai"] = sequence_tai(r["dna"]) if r["dna"] else None
                writer.writerow(
                    {
                        **r,
                        "pdb_id": pdb_id,
                        "chain": args.chains,
                        "alphabet": args.alphabet,
                        "guidance": ";".join(args.guide),
                        "n_members": len(members),
                        "native_aa": native_str,
                    }
                )
            fh.flush()

            best = min(records, key=lambda r: r["energy"])
            print(
                f"[{i}/{len(targets)}] {pdb_id}: recovery={best['recovery']:.3f} "
                f"E={best['energy']:.1f} n={len(members)} ({time.time() - t0:.1f}s)"
            )

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
