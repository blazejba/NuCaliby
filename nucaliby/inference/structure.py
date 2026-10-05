"""Load PDB/mmCIF protein chains into model input tensors.

MSE is converted to MET; water, hydrogen, and zero-occupancy atoms are removed;
residues missing N, CA, or C are dropped; coordinates are centered; and atoms
are mapped to the fixed slots in `constants.RESIDUE_ATOMS`.

`residue_index` is a dense 0-based rank *within each chain*, recomputed after the
frame-incomplete residues are dropped. It is not the mmCIF `res_id`. Feeding
Using mmCIF `res_id` instead changes every relative positional edge feature.

Sequence design conditions on backbone atoms only, so side-chain slots remain
masked even when the structure contains side chains.
"""

from pathlib import Path
from typing import Sequence

import biotite.structure as struc
import numpy as np
import torch
import torch.nn.functional as F
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, get_structure

from ..constants import BACKBONE_ATOMS, MAX_NUM_ATOMS, N_TOKENS, NUC_MASK_IDX, RESIDUE_ATOMS, TOKEN_TO_IDX, UNK_IDX

WATERS = ("HOH", "DOD")
FRAME_ATOMS = ("N", "CA", "C")  # a residue without a complete backbone frame is not a token
CENTER_SLOT = 1  # CA, first in every RESIDUE_ATOMS entry after N

# name -> {atom_name: slot}. Residue types outside this table are encoded as UNK,
# which keeps them as graph nodes (and as neighbours) instead of silently
# renumbering the chain around them.
_SLOTS: dict[str, dict[str, int]] = {res: {a: i for i, a in enumerate(atoms)} for res, atoms in RESIDUE_ATOMS.items()}

assert all(tuple(atoms[:4]) == BACKBONE_ATOMS for atoms in RESIDUE_ATOMS.values())
assert all(len(atoms) <= MAX_NUM_ATOMS for atoms in RESIDUE_ATOMS.values())


def read_atoms(path: str | Path) -> AtomArray:
    """Read the first model of a structure file, keeping occupancies."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".bcif":
        f = BinaryCIFFile.read(path)
    elif suffix in (".pdb", ".ent"):
        f = PDBFile.read(path)
        return f.get_structure(model=1, extra_fields=["occupancy"])
    else:
        f = CIFFile.read(path)
    return get_structure(f, model=1, extra_fields=["occupancy"], use_author_fields=True)


def clean(atoms: AtomArray, chains: Sequence[str] | None = None) -> AtomArray:
    """Protein-only, hydrogen-free, frame-complete residues of the requested chains.

    `chains` selects author chain IDs; None keeps every protein chain.
    """
    atoms = atoms[~np.isin(atoms.element, ("H", "D"))]
    atoms = atoms[~np.isin(atoms.res_name, WATERS)]
    atoms = atoms[atoms.occupancy > 0]

    # Selenomethionine is a crystallography artefact, not a 21st amino acid.
    mse = atoms.res_name == "MSE"
    atoms.atom_name[mse & (atoms.atom_name == "SE")] = "SD"
    atoms.element[mse & (atoms.element == "SE")] = "S"
    atoms.res_name[mse] = "MET"
    atoms.hetero[mse] = False

    atoms = atoms[struc.filter_amino_acids(atoms)]
    if chains is not None:
        atoms = atoms[np.isin(atoms.chain_id, list(chains))]

    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    keep = np.zeros(atoms.array_length(), dtype=bool)
    for lo, hi in zip(starts[:-1], starts[1:]):
        keep[lo:hi] = np.isin(FRAME_ATOMS, atoms.atom_name[lo:hi]).all()
    return atoms[keep]


def featurise(atoms: AtomArray, backbone_only: bool = True) -> dict[str, torch.Tensor]:
    """Cleaned AtomArray -> a batch of one. Every tensor carries a leading B=1."""
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    n = len(starts) - 1
    if n == 0:
        raise ValueError("no protein residues survived cleaning")

    coord = torch.from_numpy(atoms.coord - atoms.coord.mean(axis=0)).float()

    restype = torch.zeros(n, N_TOKENS)
    token_atom_coords = torch.zeros(n, MAX_NUM_ATOMS, 3)
    token_atom_mask = torch.zeros(n, MAX_NUM_ATOMS)

    for i, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        res = atoms.res_name[lo]
        restype[i, TOKEN_TO_IDX.get(res, UNK_IDX)] = 1.0
        slots = _SLOTS.get(res, _SLOTS["UNK"])
        for a in range(lo, hi):
            slot = slots.get(atoms.atom_name[a])
            if slot is None:  # e.g. a non-standard side-chain atom on an UNK token
                continue
            token_atom_coords[i, slot] = coord[a]
            token_atom_mask[i, slot] = 1.0

    if backbone_only:
        token_atom_mask[:, len(BACKBONE_ATOMS) :] = 0.0
    token_atom_coords = token_atom_coords * token_atom_mask.unsqueeze(-1)

    chain_id = atoms.chain_id[starts[:-1]]
    chains_seen, asym_id = np.unique(chain_id, return_inverse=True)
    residue_index = np.zeros(n, dtype=np.int64)
    for c in chains_seen:
        in_chain = chain_id == c
        residue_index[in_chain] = np.arange(in_chain.sum())

    return {
        "restype": restype[None],
        "seq_cond_mask": torch.zeros(1, n),  # design everything; callers pin positions by flipping entries
        "token_exists_mask": torch.ones(1, n),
        "residue_index": torch.from_numpy(residue_index)[None].long(),
        "asym_id": torch.from_numpy(asym_id)[None].long(),
        "token_center_coords": token_atom_coords[None, :, CENTER_SLOT],
        "token_atom_coords": token_atom_coords[None],
        "token_atom_mask": token_atom_mask[None],
        # No nucleotide conditioning. The model would derive this from an
        # all-zero seq_cond_mask anyway; it is here so samplers can write into it.
        "nuc_seq": torch.full((1, 3 * n), NUC_MASK_IDX, dtype=torch.long),
    }


def load_structure(
    path: str | Path, chains: Sequence[str] | None = None, backbone_only: bool = True
) -> dict[str, torch.Tensor]:
    """Structure file -> a batch of one, ready for `NuCaliby.forward`."""
    return featurise(clean(read_atoms(path), chains), backbone_only)


def collate(examples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad single-structure batches to a common length and stack them.

    Padded tokens carry `token_exists_mask = 0`, which is what keeps them out of
    the k-NN graph's message passing.
    """
    n = max(e["token_exists_mask"].shape[1] for e in examples)
    return {k: torch.cat([_pad(e[k], 3 * n if k == "nuc_seq" else n) for e in examples]) for k in examples[0]}


def load_batch(
    paths: Sequence[str | Path], chains: Sequence[str] | None = None, backbone_only: bool = True
) -> dict[str, torch.Tensor]:
    """Several structure files -> one padded batch."""
    return collate([load_structure(p, chains, backbone_only) for p in paths])


def _pad(x: torch.Tensor, n: int) -> torch.Tensor:
    """Zero-pad dim 1 up to length n. F.pad counts dims from the last one back."""
    return F.pad(x, (0, 0) * (x.dim() - 2) + (0, n - x.shape[1]))
