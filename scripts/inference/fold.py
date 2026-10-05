"""Designed sequences -> AlphaFold2 predictions.

    designs.csv --(translate + select)--> designs.fasta --> colabfold_batch

The designs CSV is what the sampler writes, one row per design:

    pdb_id      backbone id, e.g. 7qnl
    chain       chain the design was sampled for, e.g. A
    sample      0-based sample index within (pdb_id, chain)
    dna         designed DNA; empty for amino-acid-branch designs
    protein     designed protein
    native_aa   native protein sequence, for sequence recovery

The default ColabFold settings use single-sequence AF2 (`alphafold2_ptm`), four
recycles, and model 1. The script invokes `colabfold_batch` on the complete
FASTA; use `--colabfold` to select another executable or container command.
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from nucaliby.constants import aa_idx_to_string, dna_to_idx, translate

COLABFOLD_FLAGS = [
    "--msa-mode", "single_sequence",
    "--num-models", "1",
    "--num-recycle", "4",
    "--model-type", "alphafold2_ptm",
]  # fmt: skip


def translate_design(seq_nt: str) -> tuple[str, bool]:
    """Designed DNA -> (protein, has_premature_stop), with a trailing stop dropped.

    `translate` maps stop codons onto the UNK token index, which renders as "X".
    A four-state nucleotide design cannot produce a genuine UNK residue, so every
    X coming out of here is a stop codon.
    """
    aa = aa_idx_to_string(translate(dna_to_idx(seq_nt))).replace("X", "*")
    return aa.removesuffix("*"), "*" in aa[:-1]


def resolve(designs: pd.DataFrame) -> pd.DataFrame:
    """Add `design_id`, `seq_aa` and `premature_stop` to every row."""
    out = designs.copy()
    out["seq_nt"] = out["dna"].fillna("").astype(str)
    if out["seq_nt"].str.len().gt(0).any():
        seq_nt = out["seq_nt"]
        translated = seq_nt.map(lambda s: translate_design(s) if s else ("", False))
        out["seq_aa"] = [aa for aa, _ in translated]
        out["premature_stop"] = [stop for _, stop in translated]
    else:
        out["seq_aa"] = out["protein"].fillna("").astype(str)
        out["premature_stop"] = out["seq_aa"].str.contains(r"\*")
        out["seq_aa"] = out["seq_aa"].str.replace("*", "", regex=False)

    out["design_id"] = out["pdb_id"].astype(str) + "_" + out["chain"].astype(str) + "_s" + out["sample"].astype(str)
    return out


def select(designs: pd.DataFrame) -> pd.DataFrame:
    """Retain every supplied design and describe its sampling multiplicity."""
    out = designs.reset_index(drop=True)
    counts = designs.groupby(["pdb_id", "chain"]).size()
    out["protocol"] = "one-sample" if counts.eq(1).all() else "multi-sample"
    return out


def write_fasta(selected: pd.DataFrame, path: Path) -> int:
    """Write the foldable designs; returns how many. Premature stops are skipped."""
    foldable = selected[~selected["premature_stop"] & (selected["seq_aa"].str.len() > 0)]
    path.write_text("".join(f">{r.design_id}\n{r.seq_aa}\n" for r in foldable.itertuples()))
    return len(foldable)


def already_predicted(predictions: Path) -> set[str]:
    """Design ids that ColabFold has already produced a rank-1 model for."""
    done = set()
    for pdb in predictions.glob("*_rank_001_*.pdb"):
        name = pdb.name.split("_rank_")[0]
        for suffix in ("_unrelaxed", "_relaxed"):
            name = name.removesuffix(suffix)
        done.add(name)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", required=True, help="designs CSV written by the sampler")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--colabfold", default="colabfold_batch", help="colabfold_batch command")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the FASTA and selection, do not fold")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    predictions = out / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)

    selected = select(resolve(pd.read_csv(args.designs)))
    selected.to_csv(out / "selected.csv", index=False)
    n_stop = int(selected["premature_stop"].sum())
    print(f"selected {len(selected)} designs [{selected['protocol'].iloc[0]}], {n_stop} with a premature stop")

    to_fold = selected
    if args.skip_existing:
        done = already_predicted(predictions)
        to_fold = selected[~selected["design_id"].isin(done)]
        print(f"{len(done)} already predicted, {len(to_fold)} to go")

    fasta = out / "designs.fasta"
    n = write_fasta(to_fold, fasta)
    print(f"wrote {n} sequences to {fasta}")
    if n == 0 or args.dry_run:
        return

    cmd = args.colabfold.split() + [str(fasta), str(predictions)] + COLABFOLD_FLAGS
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
