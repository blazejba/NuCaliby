"""Training dataset backed by the preprocessed structure store.

The store is `$NUCALIBY_SOLUBLE_DATA_DIR`: one clustered metadata parquet plus one
pickled example per PDB entry under `cached_examples/`.

Cached examples contain AtomWorks `TransformedDict` and biotite `AtomArray`
objects, so training requires AtomWorks. Rows are restricted to protein chains
and sampled with weights inversely proportional to cluster size.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..constants import AA_CODON_MASK, MAX_NUM_ATOMS, N_TOKENS, TOKEN_TO_IDX, UNK_IDX
from ..modeling.model import AUGMENT_EPS

CROP_TOKENS = 1024
TRANSLATION_SCALE = 1.0

# t = keep probability, drawn as (U(0, 0.95))^0.5. The square root piles draws up
# near t=1, i.e. mostly-visible sequences; `losses.omega` then damps exactly those.
T_MAX = 0.95
T_POWER = 0.5

# atomworks ChainType: CYCLIC_PSEUDO_PEPTIDE, POLYPEPTIDE_D, POLYPEPTIDE_L.
PROTEIN_CHAIN_TYPES = (0, 5, 6)
# A residue whose N, CA or C is unresolved has no frame, so the whole residue goes.
FRAME_ATOMS = ("N", "CA", "C")
# The checkpoint config filters with 16 < length < 2048.
MIN_LENGTH, MAX_LENGTH = 17, 2048
TRAINING_LIST_DIR = Path(__file__).parents[1] / "resources" / "training_lists"
DEFAULT_VALIDATION_IDS = TRAINING_LIST_DIR / "training_validation_pdb_ids.txt"
DEFAULT_EXCLUDE_IDS = tuple(
    TRAINING_LIST_DIR / name
    for name in (
        "excluded_transmembrane_pdb_ids.txt",
        "excluded_unknown_residue_pdb_ids.txt",
        "excluded_unknown_residue_chain_ids.txt",
    )
)

TRAIN_FILTERS = ["num_polymer_pn_units < 50", "release_date <= '2021-09-30'", "resolution < 9.0"]
VAL_FILTERS = [
    "num_polymer_pn_units < 50",
    "'2021-09-30' < release_date <= '2023-01-13'",
    "resolution < 4.5",
]

_COLUMNS = [
    "example_id",
    "pdb_id",
    "q_pn_unit_iid",
    "q_pn_unit_type",
    "q_pn_unit_sequence_length",
    "q_pn_unit_cluster_id",
    "num_polymer_pn_units",
    "release_date",
    "resolution",
]

# Codon indices per amino acid, from the frozen genetic code. Row 20 (stop/UNK)
# is excluded: UNK residues get a placeholder codon and are masked out of the
# nucleotide losses instead.
SYNONYMOUS_CODONS: list[np.ndarray] = [np.nonzero(AA_CODON_MASK[a].numpy())[0] for a in range(20)]


class SolubleShards(Dataset):
    """Protein chains from the clustered soluble store.

    One item is one query chain: cropped, augmented, masked, and given a freshly
    sampled synonymous nucleotide sequence. Sampling is by cluster (weight
    1/cluster_size), which the accompanying `WeightedRandomSampler` applies.
    """

    def __init__(
        self,
        root: str | Path,
        phase: str = "train",
        crop_tokens: int = CROP_TOKENS,
        seed: int = 0,
        validation_ids: str | Path = DEFAULT_VALIDATION_IDS,
        exclude_ids: tuple[str | Path, ...] = DEFAULT_EXCLUDE_IDS,
    ):
        self.root = Path(root)
        self.phase = phase
        self.crop_tokens = crop_tokens
        self.seed = seed
        self._rng: np.random.Generator | None = None

        df = pd.read_parquet(self.root / "metadata_clustered.parquet", columns=_COLUMNS)
        val_ids = _read_ids(Path(validation_ids))
        in_val = df["pdb_id"].str.lower().isin(val_ids)
        df = df[in_val if phase == "val" else ~in_val]

        # The exclude lists contain bare PDB IDs and full chain example IDs.
        # Normalize both metadata columns because the source lists mix case.
        excluded = set()
        for path in exclude_ids:
            excluded |= _read_ids(Path(path))
        df = df[~df["pdb_id"].str.lower().isin({x for x in excluded if len(x) == 4})]
        df = df[~df["example_id"].str.lower().isin({x for x in excluded if len(x) > 4})]

        for query in TRAIN_FILTERS if phase == "train" else VAL_FILTERS:
            df = df.query(query)
        df = df[df["q_pn_unit_type"].isin(PROTEIN_CHAIN_TYPES)]
        df = df[df["q_pn_unit_sequence_length"].between(MIN_LENGTH, MAX_LENGTH - 1)]
        df = df[df["q_pn_unit_cluster_id"] >= 0]

        self.df = df.reset_index(drop=True)
        cluster_size = self.df["q_pn_unit_cluster_id"].map(self.df["q_pn_unit_cluster_id"].value_counts())
        self.sampling_weights = (1.0 / cluster_size).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if self._rng is None:
            # Fresh per worker: torch.initial_seed() already differs per worker and epoch.
            self._rng = np.random.default_rng((torch.initial_seed() + self.seed) % 2**32)

        row = self.df.iloc[idx]
        example = torch.load(
            self.root / "cached_examples" / f"{row['pdb_id']}.pt", map_location="cpu", weights_only=False
        )
        feats = featurise(example["atom_array"], row["q_pn_unit_iid"], self._rng, self.crop_tokens)
        feats["example_id"] = row["example_id"]
        return feats


def featurise(atom_array, pn_unit_iid: str, rng: np.random.Generator, crop_tokens: int = CROP_TOKENS) -> dict:
    """One cached AtomArray -> one padded training example.

    Args:
        atom_array: biotite AtomArray from a cached example.
        pn_unit_iid: the query chain instance to design.
        rng: per-worker generator; drives the crop, the augmentation, the mask
            schedule and the synonymous codon draw.
        crop_tokens: crop AND pad width. Padding to a fixed width rather than to
            the batch maximum is what makes `losses._reduce`'s constant divisor
            constant.
    """
    a = atom_array
    resolved = (a.occupancy > 0) & ~a.atomize

    starts = _residue_starts(a)
    res_of_atom = np.zeros(len(a), dtype=np.int64)
    res_of_atom[starts[1:]] = 1
    res_of_atom = np.cumsum(res_of_atom)

    # A residue without a full backbone frame is dropped whole, not atom by atom.
    has_frame = np.ones(len(starts), dtype=bool)
    for name in FRAME_ATOMS:
        is_named = a.atom_name == name
        present = np.zeros(len(starts), dtype=bool)
        present[res_of_atom[is_named]] = resolved[is_named]
        has_frame &= present

    keep_res = has_frame & np.isin(a.chain_type[starts], PROTEIN_CHAIN_TYPES) & (a.pn_unit_iid[starts] == pn_unit_iid)
    if keep_res.sum() == 0:
        raise ValueError(f"no designable residue left in chain {pn_unit_iid}")

    kept = np.nonzero(keep_res)[0]
    if len(kept) > crop_tokens:
        offset = rng.integers(0, len(kept) - crop_tokens + 1)
        kept = kept[offset : offset + crop_tokens]
    sub = np.isin(res_of_atom, kept)
    a, resolved, res_of_atom = a[sub], resolved[sub], np.searchsorted(kept, res_of_atom[sub])

    n = len(kept)
    starts = _residue_starts(a)
    slot = np.arange(len(a)) - starts[res_of_atom]

    restype_idx = np.array([TOKEN_TO_IDX.get(name, UNK_IDX) for name in a.res_name[starts]])
    coords = _augment(torch.from_numpy(a.coord).float(), torch.from_numpy(resolved), rng)

    centre_atom = starts.copy()
    is_ca = a.atom_name == "CA"
    centre_atom[res_of_atom[is_ca]] = np.nonzero(is_ca)[0]

    t = float((T_MAX * rng.random()) ** T_POWER)
    seq_cond = torch.from_numpy(rng.random(n) < t).float()
    if seq_cond.sum() == n:
        # Every example must expose at least one residue to the sequence losses.
        seq_cond[rng.integers(n)] = 0.0

    # Sidechains are hidden wherever the sequence is hidden, and additionally at
    # rate 1-p with p ~ U(0,1) drawn once per example, so the model sees the whole
    # spectrum from backbone-only to fully packed.
    keep_scn = torch.from_numpy(rng.random(n) < rng.random()).float() * seq_cond
    atom_cond = torch.from_numpy(resolved & (a.is_backbone_atom | keep_scn.numpy()[res_of_atom].astype(bool)))

    in_slot = slot < MAX_NUM_ATOMS
    token_atom_coords = torch.zeros(n, MAX_NUM_ATOMS, 3)
    token_atom_mask = torch.zeros(n, MAX_NUM_ATOMS)
    idx = (res_of_atom[in_slot], slot[in_slot])
    token_atom_mask[idx] = atom_cond[in_slot].float()
    token_atom_coords[idx] = coords[in_slot] * atom_cond[in_slot, None].float()

    feats = {
        "restype": torch.nn.functional.one_hot(torch.from_numpy(restype_idx), N_TOKENS).float(),
        "residue_index": torch.from_numpy(a.within_chain_res_idx[starts].astype(np.int64)),
        "asym_id": torch.from_numpy(np.unique(a.pn_unit_iid[starts], return_inverse=True)[1]),
        "token_center_coords": coords[centre_atom],
        "token_atom_coords": token_atom_coords,
        "token_atom_mask": token_atom_mask,
        # Every surviving residue has a frame, so existence is just "not padding".
        "token_exists_mask": torch.ones(n),
        "seq_cond_mask": seq_cond,
        "nuc_seq": _sample_synonymous_dna(restype_idx, rng),
    }
    feats = {k: _pad(v, crop_tokens * (3 if k == "nuc_seq" else 1)) for k, v in feats.items()}
    feats["t"] = torch.tensor(t)
    return feats


def _residue_starts(a) -> np.ndarray:
    """Index of the first atom of each residue, in array order."""
    same = (a.chain_iid[1:] == a.chain_iid[:-1]) & (a.res_id[1:] == a.res_id[:-1]) & (a.ins_code[1:] == a.ins_code[:-1])
    return np.concatenate([[0], np.nonzero(~same)[0] + 1])


def _augment(coords: torch.Tensor, resolved: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Centre on resolved atoms, then apply a random rigid motion plus atom noise.

    The AUGMENT_EPS noise is per-atom and independent, so drawing it here rather
    than in the training step is the same distribution.
    """
    coords = coords - coords[resolved].mean(dim=0)

    q = torch.from_numpy(rng.standard_normal(4)).float()
    w, x, y, z = q / q.norm()
    rot = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    coords = coords @ rot.T + TRANSLATION_SCALE * torch.from_numpy(rng.standard_normal(3)).float()
    return coords + AUGMENT_EPS * torch.from_numpy(rng.standard_normal(coords.shape)).float()


def _sample_synonymous_dna(restype_idx: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """Draw one uniform synonymous codon per residue -> nucleotide indices [3N].

    Redrawn every epoch on purpose: the model must never see a fixed codon choice
    as the "right" one, since the losses that consume this are marginalised over
    syn(a_i) anyway.
    """
    codons = np.zeros(len(restype_idx), dtype=np.int64)  # UNK -> codon 0 (AAA), masked out downstream
    for i, aa in enumerate(restype_idx):
        if aa < 20:
            codons[i] = rng.choice(SYNONYMOUS_CODONS[aa])
    nuc = np.stack([codons // 16, (codons // 4) % 4, codons % 4], axis=-1)
    return torch.from_numpy(nuc.reshape(-1))


def _pad(x: torch.Tensor, width: int) -> torch.Tensor:
    out = torch.zeros(width, *x.shape[1:], dtype=x.dtype)
    out[: x.shape[0]] = x[:width]
    return out


def collate(items: list[dict]) -> dict:
    """Stack fixed-width examples; example_id stays a plain list of strings."""
    keys = [k for k in items[0] if k != "example_id"]
    batch = {k: torch.stack([item[k] for item in items]) for k in keys}
    batch["example_id"] = [item["example_id"] for item in items]
    return batch


def make_dataloader(
    dataset: SolubleShards, batch_size: int = 4, samples_per_epoch: int = 6250, num_workers: int = 8
) -> DataLoader:
    """Cluster-weighted sampling with replacement for train, in-order for val."""
    sampler = None
    if dataset.phase == "train":
        weights = torch.from_numpy(dataset.sampling_weights)
        sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate,
    )


def _read_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Required training split or exclusion list not found: {path}")
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}
