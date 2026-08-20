"""Molecular specifications for the acrylamide + methanethiolate study.

Follows the CovAngelo protocol (arXiv:2604.10487) on the points that are
reproducible with open tooling:

* two geometries -- transition-state-like encounter (TS) and separated
  reactants (FAR) -- so the deliverable is an energy *difference*, not a single
  point whose absolute value means nothing on its own;
* MP2 natural orbitals for active-space selection, which is what the paper used
  and what keeps the active space on the C-S reaction region instead of on the
  diffuse tail of the anion;
* PCM solvation at epsilon = 4;
* a ladder of active-space sizes, 8 / 12 / 16 qubits.

What this deliberately does NOT reproduce is the paper's ECC-DMET embedding with
quantum-information-optimized bath orbitals. That is the paper's actual novelty,
it is not available in PySCF or Qiskit, and rebuilding it is a research project
rather than a port. Absolute energies here are therefore NOT comparable to the
paper's Table 2; the VQE-vs-FCI *agreement* is what carries over.
"""

from __future__ import annotations

from chemistry import MoleculeSpec


# Acrylamide atoms 0-9, then methanethiolate S/C/H/H/H.
# Fragments from PubChem, rigidly placed with S--C_beta = 2.35 A, the
# transition-state distance reported by Evenseth et al. (arXiv:2604.10487).
_ACRYLAMIDE = [
    ("O", (-0.6119, -1.2994, 0.0001)),
    ("N", (-1.5884, 0.7844, 0.0001)),
    ("C", (-0.5159, -0.0771, -0.0003)),  # carbonyl carbon
    ("C", (0.7845, 0.6387, 0.0000)),     # alpha carbon
    ("C", (1.9318, -0.0466, 0.0001)),    # beta carbon
    ("H", (0.7929, 1.7231, 0.0000)),
    ("H", (-1.4856, 1.7945, 0.0002)),
    ("H", (-2.5388, 0.4287, 0.0004)),
    ("H", (2.8793, 0.4821, 0.0002)),
    ("H", (1.9647, -1.1304, 0.0000)),
]

# Methanethiolate at the transition-state separation.
_METHANETHIOLATE_TS = [
    ("S", (1.9318, -0.0466, 2.3501)),
    ("C", (1.9318, -0.0466, 4.1541)),
    ("H", (2.9419, 0.1434, 4.5256)),
    ("H", (1.2626, 0.7335, 4.5257)),
    ("H", (1.5910, -1.0163, 4.5252)),
]

TS_SEPARATION_ANGSTROM = 2.35
FAR_SEPARATION_ANGSTROM = 8.00


def _translated(atoms, dz: float):
    return [(symbol, (x, y, z + dz)) for symbol, (x, y, z) in atoms]


# Atom indices used by the active-space localization guard: carbonyl O, carbonyl
# C, alpha C, beta C, and the attacking S.
_REACTION_ATOMS = (0, 2, 3, 4, 10)

GEOMETRIES = {
    "ts": (
        _ACRYLAMIDE + _METHANETHIOLATE_TS,
        TS_SEPARATION_ANGSTROM,
        "Transition-state-like encounter complex, S--C_beta = 2.35 A.",
    ),
    "far": (
        _ACRYLAMIDE
        + _translated(_METHANETHIOLATE_TS, FAR_SEPARATION_ANGSTROM - TS_SEPARATION_ANGSTROM),
        FAR_SEPARATION_ANGSTROM,
        "Separated reactants, S--C_beta = 8.00 A, fragments rigid.",
    ),
}

# Active-space ladder. Qubits = 2 * spatial orbitals under Jordan-Wigner.
ACTIVE_SPACES = {
    "8q": (4, 4),    # (electrons, spatial orbitals)
    "12q": (6, 6),
    "16q": (8, 8),
}


def make_spec(
    geometry: str = "ts",
    # 8q = CAS(4e,4o) by default: it is the size the paper ran on hardware, and
    # the fast one to validate the pipeline on. Measured cost of a 50-iteration
    # VQE on an 8-core CPU: 8q 0.6 h, 12q 1.8 h, 14q 13.5 h. Select a larger
    # space per run with --active-space.
    active_space: str = "8q",
    basis: str = "aug-cc-pvdz",
    orbital_selection: str = "mp2_natural",
    solvent_epsilon: float | None = 4.0,
) -> MoleculeSpec:
    """Build one point of the study.

    Defaults follow the paper: MP2 natural orbitals, PCM epsilon = 4,
    aug-cc-pVDZ. `orbital_selection="canonical_hf_frontier"` and
    `solvent_epsilon=None` reproduce the earlier vacuum/canonical setup for
    comparison -- the difference between the two is worth measuring, since the
    paper found orbital choice alone moved their VQE energy by 0.887 Ha.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}; choose from {sorted(GEOMETRIES)}")
    if active_space not in ACTIVE_SPACES:
        raise ValueError(
            f"Unknown active space {active_space!r}; choose from {sorted(ACTIVE_SPACES)}"
        )
    atoms, separation, description = GEOMETRIES[geometry]
    n_electrons, n_orbitals = ACTIVE_SPACES[active_space]
    solvation = (
        "vacuum" if solvent_epsilon is None else f"PCM epsilon={solvent_epsilon:g}"
    )
    return MoleculeSpec(
        name=(
            f"Acrylamide + methanethiolate [{geometry.upper()}] "
            f"CAS({n_electrons}e,{n_orbitals}o) {solvation}"
        ),
        atoms=atoms,
        basis=basis,
        charge=-1,
        spin=0,
        n_active_orbitals=n_orbitals,
        n_active_electrons=n_electrons,
        geometry_source=(
            "Acrylamide: PubChem CID 6579; methanethiolate: PubChem CID 878 with "
            "S-H removed and charge -1. "
            f"{description} Separation {separation:.2f} A. "
            "TS distance from arXiv:2604.10487. Fragments are rigid; this is a "
            "rigid-scan energy difference, not a relaxed reaction barrier."
        ),
        model_note=(
            f"Active-space CAS({n_electrons}e,{n_orbitals}o) with "
            f"{orbital_selection} orbitals and {solvation}. Independent analogue "
            "of arXiv:2604.10487; that work used a proprietary ECC-DMET embedding "
            "with quantum-information-optimized bath orbitals, so absolute "
            "energies here are NOT comparable to its Table 2."
        ),
        orbital_selection=orbital_selection,
        solvent_epsilon=solvent_epsilon,
        diagnostic_atom_indices=_REACTION_ATOMS,
    )


# The default study point, kept as a module-level name so `run.py` and the tests
# have something stable to import.
ACRYLAMIDE_THIOLATE = make_spec()
