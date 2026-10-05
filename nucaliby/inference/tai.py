"""tRNA Adaptation Index weights (dos Reis et al. 2004).

tAI scores how well a codon is served by the host's tRNA pool, correcting for
wobble pairing at the third position. It provides a differentiable proxy for translation
efficiency, applied as an inference-time energy bias rather than learned from
genomic data.

Counts are tRNA gene copy numbers indexed by the codon each tRNA reads (the
reverse complement of its anticodon), from GtRNAdb.
"""

import math
from typing import Final

# Wobble efficiencies p = 1 - s, dos Reis et al. (2004) Table 2, keyed by the
# third base of the codon.
WOBBLE_P: Final[dict[str, float]] = {"T": 0.59, "C": 0.72, "A": 0.0001, "G": 0.32}

# Isoleucine ATA is read by a lysidine-modified tRNA in bacteria, so it gets its
# own efficiency rather than the generic A-wobble term.
ISOLEUCINE_P: Final[float] = 1 - 0.89

STOP_CODONS: Final[tuple[str, ...]] = ("TAA", "TAG", "TGA")

TRNA_COUNTS: Final[dict[str, dict[str, int]]] = {
    # GtRNAdb sacCer3 (S288C); excludes initiator Met and undetermined tRNAs.
    "s_cerevisiae": {
        "TTT": 0,  "TTC": 10, "TTA": 7,  "TTG": 10,
        "TCT": 11, "TCC": 0,  "TCA": 3,  "TCG": 1,
        "TAT": 0,  "TAC": 8,
        "TGT": 0,  "TGC": 4,  "TGG": 6,
        "CTT": 0,  "CTC": 1,  "CTA": 3,  "CTG": 0,
        "CCT": 2,  "CCC": 0,  "CCA": 10, "CCG": 0,
        "CAT": 0,  "CAC": 7,  "CAA": 9,  "CAG": 1,
        "CGT": 6,  "CGC": 0,  "CGA": 0,  "CGG": 1,
        "ATT": 13, "ATC": 0,  "ATA": 2,
        "ACT": 11, "ACC": 0,  "ACA": 4,  "ACG": 1,
        "AAT": 0,  "AAC": 10, "AAA": 7,  "AAG": 14,
        "AGT": 0,  "AGC": 2,  "AGA": 11, "AGG": 1,
        "GTT": 14, "GTC": 0,  "GTA": 2,  "GTG": 2,
        "GCT": 11, "GCC": 0,  "GCA": 5,  "GCG": 0,
        "GAT": 0,  "GAC": 16, "GAA": 14, "GAG": 2,
        "GGT": 0,  "GGC": 16, "GGA": 3,  "GGG": 2,
    },
    # E. coli K-12 MG1655.
    "e_coli": {
        "TTT": 0, "TTC": 2, "TTA": 1, "TTG": 1,
        "CTT": 0, "CTC": 1, "CTA": 1, "CTG": 4,
        "ATT": 0, "ATC": 3, "ATA": 0, "GTT": 0,
        "GTC": 2, "GTA": 5, "GTG": 0, "TCT": 0,
        "TCC": 2, "TCA": 1, "TCG": 1, "CCT": 0,
        "CCC": 1, "CCA": 1, "CCG": 1, "ACT": 0,
        "ACC": 2, "ACA": 1, "ACG": 2, "GCT": 0,
        "GCC": 2, "GCA": 3, "GCG": 0, "TAT": 0,
        "TAC": 3, "CAT": 0, "CAC": 1, "CAA": 2,
        "CAG": 2, "AAT": 0, "AAC": 4, "AAA": 6,
        "AAG": 0, "GAT": 0, "GAC": 3, "GAA": 4,
        "GAG": 0, "TGT": 0, "TGC": 1, "TGG": 1,
        "CGT": 4, "CGC": 0, "CGA": 0, "CGG": 1,
        "AGT": 0, "AGC": 1, "AGA": 1, "AGG": 1,
        "GGT": 0, "GGC": 4, "GGA": 1, "GGG": 1,
    },
    # E. coli BL21(DE3) -- the expression strain used for the wet-lab validation,
    # and therefore the default organism for tAI guidance.
    "bl21_de3": {
        "TTT": 0, "TTC": 2, "TTA": 1, "TTG": 1,
        "CTT": 0, "CTC": 1, "CTA": 1, "CTG": 4,
        "ATT": 0, "ATC": 3, "ATA": 0, "GTT": 0,
        "GTC": 2, "GTA": 5, "GTG": 0, "TCT": 1,
        "TCC": 2, "TCA": 1, "TCG": 1, "CCT": 0,
        "CCC": 1, "CCA": 1, "CCG": 1, "ACT": 0,
        "ACC": 2, "ACA": 1, "ACG": 1, "GCT": 0,
        "GCC": 2, "GCA": 3, "GCG": 0, "TAT": 0,
        "TAC": 3, "CAT": 0, "CAC": 1, "CAA": 2,
        "CAG": 2, "AAT": 0, "AAC": 4, "AAA": 6,
        "AAG": 0, "GAT": 0, "GAC": 3, "GAA": 4,
        "GAG": 0, "TGT": 0, "TGC": 1, "TGG": 1,
        "CGT": 4, "CGC": 0, "CGA": 0, "CGG": 1,
        "AGT": 0, "AGC": 1, "AGA": 1, "AGG": 1,
        "GGT": 0, "GGC": 4, "GGA": 1, "GGG": 1,
    },
}  # fmt: skip

_BASES = "TCAG"
SENSE_CODONS: Final[tuple[str, ...]] = tuple(
    a + b + c for a in _BASES for b in _BASES for c in _BASES if a + b + c not in STOP_CODONS
)


def codon_weights(organism: str) -> dict[str, float]:
    """Relative adaptiveness w_i for each sense codon, normalised to max 1.

    ATG is excluded (the start codon is not a design choice). Codons left with zero
    weight are replaced by the geometric mean of the non-zero ones, so that a
    missing tRNA does not send log w to negative infinity.
    """
    if organism not in TRNA_COUNTS:
        raise ValueError(f"unknown organism '{organism}'; have {sorted(TRNA_COUNTS)}")

    # Stop codons are included at zero because the wobble rules read the sibling
    # codon `pair + base`, which can itself be a stop (TGG looks up TGA).
    trna = dict.fromkeys(SENSE_CODONS, 0) | TRNA_COUNTS[organism]
    trna |= dict.fromkeys(STOP_CODONS, 0)

    absolute = {}
    for codon in SENSE_CODONS:
        if codon == "ATG":
            continue
        pair, third = codon[:2], codon[2]
        if third == "T":
            absolute[codon] = trna[codon] + WOBBLE_P["T"] * trna[pair + "C"]
        elif third == "C":
            absolute[codon] = trna[codon] + WOBBLE_P["C"] * trna[pair + "T"]
        elif third == "A":
            absolute[codon] = trna[codon] + WOBBLE_P["A"] * trna[pair + "T"]
        else:
            absolute[codon] = trna[codon] + WOBBLE_P["G"] * trna[pair + "A"]

    if organism == "e_coli":
        absolute["ATA"] = ISOLEUCINE_P

    w_max = max(absolute.values())
    weights = {c: w / w_max for c, w in absolute.items()}

    nonzero = [w for w in weights.values() if w > 1e-10]
    geo_mean = math.exp(sum(map(math.log, nonzero)) / len(nonzero))
    return {c: (w if w > 1e-10 else geo_mean) for c, w in weights.items()}


def sequence_tai(dna: str, organism: str = "bl21_de3") -> float:
    """Geometric mean of codon weights, excluding the initiator codon."""
    weights = codon_weights(organism)
    logs = [math.log(weights[c]) for i in range(3, len(dna) - 2, 3) if (c := dna[i : i + 3].upper()) in weights]
    return math.exp(sum(logs) / len(logs)) if logs else 0.0
