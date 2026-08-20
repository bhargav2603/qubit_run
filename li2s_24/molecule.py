"""The system under study: Li2S, CAS(12e,12o)/STO-3G, 24 qubits.

The only file in this folder specific to this molecule. Everything else is
driven by the MoleculeSpec defined here.

Li2S is the discharge product of a lithium-sulfur battery, and the Li-S bond
dissociation is the reaction whose energetics decide whether that chemistry is
reversible. It is also genuinely multireference at long bond length -- pulling
one lithium off gives two open-shell fragments, Li(2S) and LiS(2Pi), which no
single determinant can describe -- so Hartree-Fock fails by hundreds of
millihartree exactly where the answer matters. That is the regime HI-VQE was
built for, and the reason arXiv:2503.06292 uses this system as its 24-qubit
benchmark.
"""

from hamiltonian import MoleculeSpec


# Pellow-Jarman et al., "HIVQE: Handover Iterative Variational Quantum
# Eigensolver for Efficient Quantum Chemistry Calculations", arXiv:2503.06292
# (Qunova Computing). Table 1 lists Li2S as the 24-qubit benchmark: 12 orbitals,
# 12 electrons, STO-3G, 853,776 determinants in the full CAS -- which is exactly
# C(12,6)^2, so the active space is unambiguous. Figure 6 shows the dissociation
# curve produced by "removing a lithium atom", with HI-VQE inside chemical
# accuracy of CASCI at every bond length while Hartree-Fock is off by hundreds
# of millihartree at long range.
#
# The paper publishes no geometry, no per-point energies and no orbital
# specification beyond the active-space size, so the TOTAL energies here are not
# expected to match theirs and are not claimed to. What is comparable, and is
# what this folder actually benchmarks, is stated in the README: the error
# against our own CASCI at each geometry, the fraction of the determinant space
# needed to reach it, and the shape of the dissociation curve.
PUBLISHED_REFERENCE = {
    "source": "Pellow-Jarman et al., arXiv:2503.06292, Table 1 and Figure 6",
    "active_space": "CAS(12e,12o)/STO-3G, 24 qubits",
    "full_cas_determinants": 853776,
    "pauli_words_for_conventional_vqe": 15697,
    "reference_method": "CASCI",
    "claim": (
        "HI-VQE stays inside chemical accuracy (1.6 mHa) of CASCI at every bond "
        "length of the Li-S dissociation, using a small fraction of the "
        "determinant space; Hartree-Fock is off by hundreds of mHa at long range."
    ),
    "note": (
        "No geometry, per-point energies or orbital set are published, so total "
        "energies are not reproducible. Error vs CASCI and determinant count are."
    ),
}

# Linear Li-S-Li. The gas-phase molecule is D-infinity-h with a very flat bending
# potential; the equilibrium bond length below is the minimum of this folder's
# own STO-3G CASCI symmetric stretch, which `python run.py geometry` recomputes
# and prints. It is deliberately *not* an experimental number: the benchmark is
# internal consistency with CASCI on one stated surface, and an STO-3G minimum is
# the right reference point for an STO-3G study.
EQUILIBRIUM_BOND_ANGSTROM = 2.10

LI2S = MoleculeSpec(
    name="Li2S",
    basis="sto-3g",
    charge=0,
    spin=0,
    n_active_orbitals=12,
    n_active_electrons=12,
    equilibrium_bond_angstrom=EQUILIBRIUM_BOND_ANGSTROM,
    geometry_kind="linear_symmetric_triatomic",
    central_atom="S",
    terminal_atom="Li",
    core_atom_index=0,
    geometry_source=(
        "Linear D-infinity-h Li-S-Li, bond length taken from this folder's own "
        "CASCI(12e,12o)/STO-3G symmetric stretch (`python run.py geometry`). "
        "The dissociation coordinate stretches one Li-S bond and holds the other."
    ),
    model_note=(
        "CAS(12e,12o) in canonical RHF orbitals with STO-3G, 24 qubits under "
        "Jordan-Wigner, matching the active space stated for Li2S in "
        "arXiv:2503.06292 Table 1. Li2S has 22 electrons and 19 STO-3G orbitals, "
        "so 12 active electrons fixes the frozen core at the 5 lowest orbitals -- "
        "the sulfur 1s, 2s and 2p shell -- and drops the 2 highest virtuals. "
        "`prepare` measures that the frozen orbitals really are the sulfur core "
        "rather than assuming it."
    ),
    # The sulfur core sits near -91, -9 and -6.7 Ha while nothing else in the
    # molecule is below -3 Ha, so both bars are comfortable at equilibrium and
    # are there to catch a stretched geometry where the ordering has changed.
    min_core_localization=0.90,
    min_core_gap_hartree=1.0,
    published_reference=PUBLISHED_REFERENCE,
)

# The dissociation scan of Figure 6: hold one Li-S bond at equilibrium and pull
# the other lithium away. Dense near the minimum where the curvature matters,
# sparse in the flat tail where it does not.
DISSOCIATION_DISTANCES = (
    1.70,
    1.90,
    2.00,
    2.10,
    2.20,
    2.40,
    2.60,
    2.90,
    3.20,
    3.60,
    4.20,
    5.00,
    6.00,
)
