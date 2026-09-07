"""The molecule under study: imipramine, CASCI(6e,6o), 12 qubits.

The only file in this folder that is specific to this molecule. Everything else
is driven by the MoleculeSpec defined here.
"""

from hamiltonian import MoleculeSpec


# The dibenzazepine pi system: the twelve aromatic carbons of the two fused
# benzene rings plus the bridging azepine nitrogen, whose lone pair is
# conjugated with them. Derived from the hydrogen-attachment pattern below --
# aromatic CH carbons carry one hydrogen, ring-junction carbons none, and every
# excluded carbon carries two or three.
#
#   3, 4, 7, 8            ring-junction carbons, no hydrogens
#   11..18                aromatic CH carbons, one hydrogen each
#   0                     azepine N, conjugated with both rings
#
# Deliberately excluded, because an active space that lands on them is not
# describing what makes imipramine a tricyclic antidepressant:
#
#   1                     dimethylamino N of the side chain
#   5, 6                  the sp3 ethano bridge (C10-C11)
#   2, 9, 10              the propyl tail
#   19, 20                the N-methyls
AROMATIC_CORE_ATOMS = (0, 3, 4, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18)

# Koziell-Pipe et al., arXiv:2607.22468, Table 1: imipramine CAS(6e,6o)/6-31G,
# 12 qubits, Hartree-Fock orbitals, Jordan-Wigner -- the same construction this
# folder performs. Their CASCI reference is -841.953249 Ha and their statevector
# VQE reached -841.952101 Ha, an error of 1.15 mHa with ADAPT-GQE circuits.
#
# Their geometries are unpublished MD/NEB snapshots and this folder uses a
# single PubChem conformer, so the TOTAL energies are not expected to agree:
# conformational differences move them by tens of mHa and that difference is
# physically meaningful, not an error. The active-space correlation energy is
# the conformer-insensitive quantity, and it is what shows whether the same six
# frontier orbitals were selected.
PUBLISHED_REFERENCE = {
    "source": "Koziell-Pipe et al., arXiv:2607.22468, Table 1",
    "active_space": "CAS(6e,6o)/6-31G, 12 qubits, RHF orbitals, Jordan-Wigner",
    "casci_energy_hartree": -841.953249,
    "statevector_vqe_energy_hartree": -841.952101,
    "statevector_error_millihartree": 1.15,
    "hardware_energy_hartree": -841.752841,
    "trained_accuracy_millihartree": [5.0, 10.0],
    "note": (
        "Different conformer: compare E_CASCI - E_HF, not the total energy. "
        "Their 1.15 mHa statevector error used ADAPT-GQE circuits, so a shallow "
        "hardware-efficient ansatz should be expected to need depth."
    ),
}

# Neutral imipramine free base, PubChem CID 3696 conformer 00000E7000000001.
# Coordinates are stored explicitly so every Hamiltonian build is reproducible
# and no compute node ever needs network access.
IMIPRAMINE = MoleculeSpec(
    name="Imipramine",
    atoms=[
        ("N", (-0.1325, -0.0619, -0.3888)),
        ("N", (4.1861, -0.5873, 0.1367)),
        ("C", (1.2487, -0.4584, -0.7068)),
        ("C", (-1.1039, -1.0939, -0.2694)),
        ("C", (-0.3398, 1.3418, -0.3868)),
        ("C", (-2.7993, 0.2743, 1.1690)),
        ("C", (-1.7043, 1.1694, 1.7256)),
        ("C", (-2.3403, -0.9504, 0.4063)),
        ("C", (-1.0916, 1.9752, 0.6167)),
        ("C", (1.8347, -1.3205, 0.4108)),
        ("C", (3.2928, -1.7379, 0.1962)),
        ("C", (-0.8693, -2.3461, -0.8985)),
        ("C", (0.2181, 2.1531, -1.3988)),
        ("C", (-3.2733, -2.0173, 0.3972)),
        ("C", (-1.2686, 3.3703, 0.5982)),
        ("C", (-1.7941, -3.3932, -0.8780)),
        ("C", (0.0366, 3.5373, -1.4122)),
        ("C", (-3.0035, -3.2266, -0.2317)),
        ("C", (-0.7072, 4.1467, -0.4114)),
        ("C", (5.5772, -1.0213, -0.0048)),
        ("C", (4.0336, 0.2463, 1.3305)),
        ("H", (1.3177, -0.9449, -1.6870)),
        ("H", (1.9043, 0.4166, -0.7887)),
        ("H", (-3.4797, 0.8467, 0.5245)),
        ("H", (-3.3894, -0.0642, 2.0317)),
        ("H", (-0.9300, 0.5947, 2.2490)),
        ("H", (-2.1434, 1.8418, 2.4735)),
        ("H", (1.2736, -2.2557, 0.5120)),
        ("H", (1.6950, -0.8291, 1.3817)),
        ("H", (3.3590, -2.3059, -0.7402)),
        ("H", (3.5842, -2.4195, 1.0066)),
        ("H", (0.0356, -2.5422, -1.4676)),
        ("H", (0.7822, 1.7107, -2.2177)),
        ("H", (-4.2317, -1.9044, 0.9018)),
        ("H", (-1.8497, 3.8609, 1.3754)),
        ("H", (-1.5675, -4.3279, -1.3823)),
        ("H", (0.4720, 4.1342, -2.2081)),
        ("H", (-3.7350, -4.0280, -0.2162)),
        ("H", (-0.8526, 5.2227, -0.4166)),
        ("H", (5.7008, -1.6260, -0.9102)),
        ("H", (5.9164, -1.6119, 0.8540)),
        ("H", (6.2414, -0.1566, -0.1138)),
        ("H", (4.8037, 1.0274, 1.3481)),
        ("H", (4.1270, -0.3302, 2.2582)),
        ("H", (3.0844, 0.7908, 1.3374)),
    ],
    basis="6-31g",
    charge=0,
    spin=0,
    n_active_orbitals=6,
    n_active_electrons=6,
    geometry_source=(
        "NIH PubChem CID 3696, conformer 00000E7000000001, retrieved 2026-08-12; "
        "https://pubchem.ncbi.nlm.nih.gov/compound/3696"
    ),
    model_note=(
        "CASCI(6e,6o) in canonical RHF frontier orbitals with 6-31G, matching "
        "the Hamiltonian construction stated in arXiv:2607.22468. This single "
        "PubChem conformer is not one of the paper's unpublished MD/NEB snapshots."
    ),
    diagnostic_atom_indices=AROMATIC_CORE_ATOMS,
    # A genuine occupied dibenzazepine pi orbital sits almost entirely on the
    # aromatic core, so 0.80 is a real test rather than a formality. If
    # HOMO-2..HOMO instead land on the dimethylamino lone pair or the ethano
    # bridge, the run is measuring the wrong molecule and `prepare` refuses.
    min_active_orbital_localization=0.80,
    # Virtual orbitals are held to a looser bar for a real reason, not for
    # convenience: Mulliken populations of virtuals carry long basis-set tails
    # and are only weakly meaningful. 0.65 still rejects a pi* orbital that has
    # drifted onto the propyl tail while accepting the genuine dibenzazepine
    # pi* manifold. Run `prepare --diagnose` before trusting either number on a
    # new geometry.
    min_active_virtual_localization=0.65,
    published_reference=PUBLISHED_REFERENCE,
)
