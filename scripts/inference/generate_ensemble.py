"""Generate backbone ensembles by Protpardelle-1c partial denoising."""

import argparse
import importlib
import os
import shutil
import tempfile
from pathlib import Path

from biotite.structure.io.pdb import PDBFile

from nucaliby.inference.structure import clean, read_atoms

STRUCTURE_SUFFIXES = (".pdb", ".cif", ".bcif", ".ent")


def prepare_backbone(source: Path, destination: Path) -> None:
    """Write exactly the residues NuCaliby will retain as a Protpardelle PDB."""
    atoms = clean(read_atoms(source))
    if atoms.array_length() == 0:
        raise ValueError(f"no complete standard protein residues in {source}")
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(destination)


def resolve_targets(pdbs: list[str], pdb_dir: str | None, test_ids: str | None) -> list[Path]:
    targets = [Path(path) for path in pdbs]
    if not pdb_dir:
        return targets

    root = Path(pdb_dir)
    if test_ids:
        for raw in Path(test_ids).read_text().splitlines():
            pdb_id = raw.strip().lower()
            if not pdb_id:
                continue
            hit = next(
                (path for suffix in STRUCTURE_SUFFIXES for path in root.rglob(f"{pdb_id}{suffix}")),
                None,
            )
            if hit is None:
                raise FileNotFoundError(f"no structure for {pdb_id} below {root}")
            targets.append(hit)
    else:
        targets.extend(path for path in root.iterdir() if path.suffix.lower() in STRUCTURE_SUFFIXES)
    return targets


def generate(
    source: Path,
    out_root: Path,
    weights_dir: Path,
    config: Path,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> Path:
    """Generate conformers for one structure in `<out_root>/<stem>/`."""
    weights_dir = weights_dir.resolve()
    required = (weights_dir / "configs" / "cc95.yaml", weights_dir / "weights" / "cc95_epoch3490.pth")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Protpardelle model files: " + ", ".join(missing))

    destination = out_root / source.stem
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nucaliby_protpardelle_") as tmp_name:
        tmp = Path(tmp_name)
        prepared = tmp / f"{source.stem}.pdb"
        prepare_backbone(source, prepared)

        os.environ["PROTPARDELLE_OUTPUT_DIR"] = str(tmp / "raw_output")
        os.environ["PROTPARDELLE_MODEL_PARAMS"] = str(weights_dir)
        os.environ.setdefault("FOLDSEEK_BIN", ".")
        os.environ.setdefault("ESMFOLD_PATH", ".")
        os.environ.setdefault("PROTEINMPNN_WEIGHTS", ".")
        try:
            protpardelle_sample = importlib.import_module("protpardelle.sample")
        except ImportError as exc:
            raise RuntimeError("install the ensemble dependencies with `pip install '.[ensemble]'`") from exc
        # The package reads this path at import time. Assign it explicitly too,
        # so repeated calls in one process each use their own temporary folder.
        protpardelle_sample.PROTPARDELLE_OUTPUT_DIR = tmp / "raw_output"

        save_dirs = protpardelle_sample.sample(
            sampling_yaml_path=config.resolve(),
            motif_pdb=prepared,
            batch_size=batch_size,
            num_samples=num_samples,
            num_mpnn_seqs=0,
            seed=seed,
            use_wandb=False,
        )
        generated = [
            path for save_dir in save_dirs for path in Path(save_dir).glob("*.pdb") if path.name != prepared.name
        ]
        if len(generated) != num_samples:
            raise RuntimeError(f"Protpardelle returned {len(generated)} conformers; expected {num_samples}")
        for index, path in enumerate(sorted(generated)):
            shutil.copy2(path, destination / f"conformer_{index:04d}.pdb")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", action="append", default=[], help="input PDB/mmCIF; repeat for multiple targets")
    parser.add_argument("--pdb-dir", help="directory of structures, optionally selected by --test-ids")
    parser.add_argument("--test-ids", help="one PDB ID per line")
    parser.add_argument(
        "--weights-dir",
        default=str(Path(__file__).resolve().parents[2] / "assets/protpardelle-1c"),
        help="Protpardelle directory containing configs/ and weights/ (default: bundled assets)",
    )
    parser.add_argument("--out", required=True, help="output root; conformers go to <out>/<pdb_stem>/")
    parser.add_argument("--config", default="assets/protpardelle_partial_diffusion.yaml")
    parser.add_argument(
        "--num-samples", type=int, default=31, help="generated conformers (plus the original during design)"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    targets = resolve_targets(args.pdb, args.pdb_dir, args.test_ids)
    if not targets:
        parser.error("pass --pdb or --pdb-dir")
    for index, source in enumerate(targets):
        destination = generate(
            source,
            Path(args.out),
            Path(args.weights_dir),
            Path(args.config),
            args.num_samples,
            args.batch_size,
            args.seed + index,
        )
        print(f"{source}: {args.num_samples} conformers -> {destination}", flush=True)


if __name__ == "__main__":
    main()
