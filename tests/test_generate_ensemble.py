from pathlib import Path
from types import SimpleNamespace

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from nucaliby.inference.structure import read_atoms
from scripts.inference import generate_ensemble


def test_resolve_targets_supports_an_id_list(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    structure = pdb_dir / "1abc.pdb"
    structure.touch()
    ids = tmp_path / "ids.txt"
    ids.write_text("1ABC\n")

    assert generate_ensemble.resolve_targets([], str(pdb_dir), str(ids)) == [structure]


def test_generate_writes_only_sampled_conformers(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    (weights / "configs").mkdir(parents=True)
    (weights / "weights").mkdir()
    (weights / "configs" / "cc95.yaml").touch()
    (weights / "weights" / "cc95_epoch3490.pth").touch()
    source = tmp_path / "target.cif"
    source.touch()
    config = tmp_path / "config.yaml"
    config.touch()

    monkeypatch.setattr(generate_ensemble, "prepare_backbone", lambda source, destination: destination.touch())

    def sample(**kwargs):
        save_dir = Path(generate_ensemble.importlib.import_module("protpardelle.sample").PROTPARDELLE_OUTPUT_DIR)
        save_dir.mkdir(parents=True)
        (save_dir / "target.pdb").touch()
        (save_dir / "sample_0.pdb").touch()
        (save_dir / "sample_1.pdb").touch()
        return [save_dir]

    fake_module = SimpleNamespace(PROTPARDELLE_OUTPUT_DIR=None, sample=sample)
    monkeypatch.setattr(generate_ensemble.importlib, "import_module", lambda name: fake_module)

    destination = generate_ensemble.generate(source, tmp_path / "out", weights, config, 2, 2, 0)

    assert [path.name for path in destination.iterdir()] == ["conformer_0000.pdb", "conformer_0001.pdb"]


def test_generated_pdb_format_is_readable_by_nucaliby(tmp_path):
    atoms = AtomArray(1)
    atoms.coord = np.zeros((1, 3))
    atoms.chain_id = np.array(["A"])
    atoms.res_id = np.array([1])
    atoms.ins_code = np.array([""])
    atoms.res_name = np.array(["GLY"])
    atoms.hetero = np.array([False])
    atoms.atom_name = np.array(["CA"])
    atoms.element = np.array(["C"])
    pdb = PDBFile()
    pdb.set_structure(atoms)
    path = tmp_path / "conformer.pdb"
    pdb.write(path)

    loaded = read_atoms(path)

    assert loaded.array_length() == 1
    assert loaded.res_name[0] == "GLY"
