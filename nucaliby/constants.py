"""Fixed alphabets and lookup tables used by NuCaliby.

The token alphabet is AF3's 32-token set. It cannot be shrunk to the 21 protein
tokens without breaking the checkpoint: `W_s` is [128, 32], `W_out` is [32, 128]
and the amino-acid Potts head emits 32x32 couplings.
"""

from typing import Final

import torch

# --- Token alphabet (AF3 ordering) -----------------------------------------

TOKENS: Final[tuple[str, ...]] = (
    # 20 standard amino acids + unknown
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
    # 4 RNA + unknown
    "A", "C", "G", "U", "N",
    # 4 DNA + unknown
    "DA", "DC", "DG", "DT", "DN",
    # gap / mask token
    "<G>",
)  # fmt: skip

N_TOKENS: Final[int] = len(TOKENS)  # 32
TOKEN_TO_IDX: Final[dict[str, int]] = {t: i for i, t in enumerate(TOKENS)}

UNK_IDX: Final[int] = TOKEN_TO_IDX["UNK"]  # 20
GAP_IDX: Final[int] = TOKEN_TO_IDX["<G>"]  # 31, used as the [MASK] token

ONE_TO_THREE: Final[dict[str, str]] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}  # fmt: skip
THREE_TO_ONE: Final[dict[str, str]] = {v: k for k, v in ONE_TO_THREE.items()}

# --- Nucleotide alphabet ----------------------------------------------------

NUC_TOKENS: Final[tuple[str, ...]] = ("A", "C", "G", "T")
NUC_TO_IDX: Final[dict[str, int]] = {n: i for i, n in enumerate(NUC_TOKENS)}
NUC_MASK_IDX: Final[int] = 4
NUC_UNK_IDX: Final[int] = 5
N_NUC_TOKENS: Final[int] = 6  # A, C, G, T, MASK, UNK

# --- Genetic code -----------------------------------------------------------
# Codon index = c1 * 16 + c2 * 4 + c3, with A=0, C=1, G=2, T=3.
# Value is the AF3 token index of the encoded amino acid; 20 (UNK) marks a stop
# codon. Note that index 20 therefore means *both* "stop" and "unknown residue" —
# see `AA_CODON_MASK` below and the UNK masking in `losses.py`.

CODON_TO_AA_IDX: Final[torch.Tensor] = torch.tensor(
    [11,  2, 11,  2, 16, 16, 16, 16,  1, 15,  1, 15,  9,  9, 12,  9,
      5,  8,  5,  8, 14, 14, 14, 14,  1,  1,  1,  1, 10, 10, 10, 10,
      6,  3,  6,  3,  0,  0,  0,  0,  7,  7,  7,  7, 19, 19, 19, 19,
     20, 18, 20, 18, 15, 15, 15, 15, 20,  4, 17,  4, 10, 13, 10, 13],
    dtype=torch.long,
)  # fmt: skip

STOP_CODON_IDX: Final[tuple[int, ...]] = (48, 50, 56)  # TAA, TAG, TGA


def _build_aa_codon_mask() -> torch.Tensor:
    """[21, 64] bool: row a is True at every codon translating to AA index a.

    Row 20 collects the three stop codons. Because UNK also has index 20, any
    UNK residue must be excluded from the nucleotide losses explicitly rather
    than relying on this table.
    """
    mask = torch.zeros(21, 64, dtype=torch.bool)
    mask[CODON_TO_AA_IDX, torch.arange(64)] = True
    return mask


AA_CODON_MASK: Final[torch.Tensor] = _build_aa_codon_mask()

# --- Structure --------------------------------------------------------------

MAX_NUM_ATOMS: Final[int] = 23  # per-token atom slots used by the all-atom RBF
BACKBONE_ATOMS: Final[tuple[str, ...]] = ("N", "CA", "C", "O")

# Heavy-atom order per residue, taken from the CCD (hydrogens removed). This is
# the order atomworks lays atoms out in, and the RBF edge features are indexed
# by slot, so the order is load-bearing. OXT is only present on the C-terminal
# residue; missing atoms leave their slot masked out.
RESIDUE_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "ALA": ("N", "CA", "C", "O", "CB", "OXT"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "OXT"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "OXT"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "OXT"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "OXT"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"),
    "GLY": ("N", "CA", "C", "O", "OXT"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "OXT"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OXT"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "OXT"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "OXT"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2", "OXT"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "OXT"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"),
    "UNK": ("N", "CA", "C", "O", "CB", "CG", "OXT"),
}

# The token whose coordinate represents the residue in the k-NN graph.
CENTER_ATOM: Final[str] = "CA"


def translate(nuc_idx: torch.Tensor) -> torch.Tensor:
    """Translate a nucleotide index sequence [..., 3N] to AF3 AA indices [..., N]."""
    *lead, n3 = nuc_idx.shape
    assert n3 % 3 == 0, f"nucleotide length {n3} is not a multiple of 3"
    codons = nuc_idx.reshape(*lead, n3 // 3, 3)
    codon_idx = codons[..., 0] * 16 + codons[..., 1] * 4 + codons[..., 2]
    return CODON_TO_AA_IDX.to(nuc_idx.device)[codon_idx]


def aa_idx_to_string(idx: torch.Tensor) -> str:
    """AF3 token indices -> one-letter amino-acid string."""
    return "".join(THREE_TO_ONE.get(TOKENS[int(i)], "X") for i in idx)


def nuc_idx_to_string(idx: torch.Tensor) -> str:
    """Nucleotide indices -> DNA string."""
    return "".join(NUC_TOKENS[int(i)] for i in idx)


def dna_to_idx(seq: str) -> torch.Tensor:
    """DNA/RNA string -> nucleotide indices (U is folded to T)."""
    seq = seq.strip().upper().replace("U", "T")
    return torch.tensor([NUC_TO_IDX[c] for c in seq], dtype=torch.long)
