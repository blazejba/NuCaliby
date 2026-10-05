"""Compute sequence and structure metrics for folded designs.

The input is the ``selected.csv`` written by ``fold.py`` and the corresponding
ColabFold prediction directory. Every input row is scored by default. Optional
ID and chain filters let a caller define an evaluation subset without embedding
dataset policy in the scorer.

Per-design output includes sequence recovery, pLDDT, pTM, scTM, scRMSD, GC
content, and tAI when the required sequence, prediction, or reference is
available. scTM and scRMSD require TM-align and an input-backbone directory.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd

from nucaliby.inference.tai import sequence_tai

METRIC_COLUMNS = (
    "sequence_recovery",
    "plddt",
    "ptm",
    "scTM",
    "scRMSD",
    "gc_content",
    "tai",
)
IDENTITY_COLUMNS = ("design_id", "pdb_id", "chain", "sample", "protocol", "premature_stop")
METADATA_COLUMNS = ("alphabet", "guidance", "n_members", "energy")


def as_text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def gc_content(seq_nt: str) -> float:
    return (seq_nt.count("G") + seq_nt.count("C")) / len(seq_nt)


def recovery(designed: str, native: str) -> float | None:
    """Return sequence recovery, or None when the sequences are not comparable."""
    if not designed or not native or len(designed) != len(native):
        return None
    return sum(a == b for a, b in zip(designed, native)) / len(native)


def read_structure(path: Path):
    if path.suffix.lower() == ".pdb":
        return pdb.PDBFile.read(str(path)).get_structure(model=1)
    return pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)


def find_backbone(pdb_dir: Path, pdb_id: str) -> Path | None:
    for suffix in (".cif", ".pdb"):
        path = pdb_dir / f"{pdb_id}{suffix}"
        if path.is_file():
            return path
    return None


def write_ca_pdb(structure: Path, chain: str, out: Path) -> None:
    """Write the reference chain as a CA-only PDB for TM-align."""
    atoms = read_structure(structure)
    ca = atoms[(atoms.atom_name == "CA") & (atoms.chain_id == chain)]
    if len(ca) == 0:
        raise ValueError(f"{structure} has no chain {chain}")
    lines = [
        f"ATOM  {i + 1:5d}  CA  {str(ca.res_name[i]):>3s} A{ca.res_id[i]:4d}    "
        f"{ca.coord[i, 0]:8.3f}{ca.coord[i, 1]:8.3f}{ca.coord[i, 2]:8.3f}  1.00  0.00\n"
        for i in range(len(ca))
    ]
    out.write_text("".join(lines) + "END\n")


def tmalign(prediction: str, reference: str, command: str) -> tuple[float, float]:
    """Return TM-score normalized by the reference and the aligned RMSD."""
    stdout = subprocess.run([command, prediction, reference], capture_output=True, text=True, check=True).stdout
    tm = re.findall(r"TM-score=\s*([\d.]+)", stdout)
    rmsd = re.search(r"RMSD=\s*([\d.]+)", stdout)
    if len(tm) < 2 or rmsd is None:
        raise ValueError("could not parse TM-align output")
    return float(tm[1]), float(rmsd.group(1))


def find_prediction(predictions: Path, design_id: str) -> Path | None:
    hits = sorted(predictions.glob(f"{design_id}_*rank_001_*.pdb"))
    return hits[0] if hits else None


def prediction_scores(prediction: Path) -> tuple[float, float]:
    """Return mean pLDDT and pTM from ColabFold's companion score file."""
    path = prediction.with_name(re.sub(r"_(un)?relaxed_", "_scores_", prediction.stem) + ".json")
    scores = json.loads(path.read_text())
    return float(np.mean(scores["plddt"])), float(scores["ptm"])


def design_metrics(row: dict, reference: str | None, predictions: str, tmalign_command: str) -> dict:
    out = {key: row[key] for key in (*IDENTITY_COLUMNS, *METADATA_COLUMNS) if key in row}
    seq_aa = as_text(row.get("seq_aa"))
    seq_nt = as_text(row.get("seq_nt"))
    out["length"] = len(seq_aa)
    out["sequence_recovery"] = recovery(seq_aa, as_text(row.get("native_aa")))
    if seq_nt:
        out["gc_content"] = gc_content(seq_nt)
        out["tai"] = sequence_tai(seq_nt)

    prediction = find_prediction(Path(predictions), str(row["design_id"]))
    if prediction is not None:
        out["plddt"], out["ptm"] = prediction_scores(prediction)
        if reference is not None:
            out["scTM"], out["scRMSD"] = tmalign(str(prediction), reference, tmalign_command)
    return out


def prepare_references(selected: pd.DataFrame, pdb_dir: Path, workdir: Path) -> dict[str, str]:
    """Map design IDs to temporary CA-only references."""
    refs: dict[str, str] = {}
    cache: dict[tuple[str, str], str | None] = {}
    for row in selected.itertuples():
        key = (str(row.pdb_id), str(row.chain))
        if key not in cache:
            structure = find_backbone(pdb_dir, key[0])
            out = workdir / f"{key[0]}_{key[1]}.pdb"
            try:
                if structure is None:
                    raise FileNotFoundError(f"no .cif or .pdb file for {key[0]}")
                write_ca_pdb(structure, key[1], out)
                cache[key] = str(out)
            except (FileNotFoundError, ValueError) as error:
                print(f"no reference for {key[0]} chain {key[1]}: {error}")
                cache[key] = None
        if cache[key] is not None:
            refs[str(row.design_id)] = cache[key]
    return refs


def read_ids(path: str) -> set[str]:
    return {Path(line.strip()).stem.lower() for line in Path(path).read_text().splitlines() if line.strip()}


def filter_designs(selected: pd.DataFrame, ids_path: str | None, chains: list[str] | None) -> pd.DataFrame:
    keep = pd.Series(True, index=selected.index)
    if ids_path:
        ids = read_ids(ids_path)
        pdb_ids = selected["pdb_id"].astype(str).map(lambda value: Path(value).stem.lower())
        keep &= pdb_ids.isin(ids)
    if chains:
        keep &= selected["chain"].astype(str).isin(chains)
    filtered = selected[keep].reset_index(drop=True)
    if filtered.empty:
        raise SystemExit("no designs remain after filtering")
    print(f"selected {len(filtered)}/{len(selected)} designs")
    return filtered


def summarise(metrics: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    missing = [column for column in group_by if column not in metrics.columns]
    if missing:
        raise SystemExit(f"cannot group by missing columns: {missing}")

    groups = metrics.groupby(group_by, dropna=False, sort=False) if group_by else [((), metrics)]
    rows = []
    for key, group in groups:
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_by, values))
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            observed = pd.to_numeric(group[metric], errors="coerce").dropna()
            if observed.empty:
                continue
            rows.append(
                {**identity, "metric": metric, "mean": observed.mean(), "std": observed.std(), "n": len(observed)}
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("no numeric metrics available to summarise")
    else:
        print("\n" + summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, help="selected.csv written by fold.py")
    ap.add_argument("--predictions", required=True, help="ColabFold output directory")
    ap.add_argument("--pdb-dir", default=os.environ.get("CLEAN_PDB_DIR"), help="optional reference-backbone directory")
    ap.add_argument("--out", required=True, help="per-design metrics CSV")
    ap.add_argument("--summary-out", help="optional aggregate metrics CSV")
    ap.add_argument("--ids", help="optional file containing PDB IDs to retain")
    ap.add_argument("--chains", nargs="+", help="optional chain IDs to retain")
    ap.add_argument("--group-by", nargs="*", default=[], help="metadata columns for aggregate summaries")
    ap.add_argument("--tmalign", default="TMalign", help="TM-align executable")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    selected = filter_designs(pd.read_csv(args.selected), args.ids, args.chains)
    with tempfile.TemporaryDirectory() as tmp:
        refs = prepare_references(selected, Path(args.pdb_dir), Path(tmp)) if args.pdb_dir else {}
        jobs = [
            (row, refs.get(row["design_id"]), args.predictions, args.tmalign)
            for row in selected.to_dict("records")
        ]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(design_metrics, *zip(*jobs)))
        else:
            rows = [design_metrics(*job) for job in jobs]

    metrics = pd.DataFrame(rows)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    print(f"wrote {len(metrics)} rows to {out}")

    summary = summarise(metrics, args.group_by)
    if args.summary_out:
        summary_out = Path(args.summary_out).expanduser().resolve()
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_out, index=False)
        print(f"wrote {len(summary)} rows to {summary_out}")


if __name__ == "__main__":
    main()
