"""LiH/STO-3G specifications for the local Qiskit Aer workflow.

The geometry and basis are experimental values with a cited source, so the
molecule is fixed independently of any simulator. Only the active space is
stated explicitly rather than left implicit, because the two choices below give
genuinely different problems and the reference energy follows the choice.
"""

from chemistry import MoleculeSpec


# r(Li-H) = 1.595 Angstrom, the experimental equilibrium bond length. STO-3G on
# LiH gives 6 spatial orbitals (Li 1s/2s/2p_x/2p_y/2p_z, H 1s) holding 4
# electrons, so the full space is 12 spin orbitals -> 12 qubits.
_ATOMS = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.595))]
_GEOMETRY_SOURCE = (
    "Experimental equilibrium bond length r_e(LiH) = 1.5949 Angstrom "
    "(Huber & Herzberg, via NIST CCCBDB), rounded to 1.595. Vacuum, neutral, "
    "closed shell."
)

# Default. Li's 1s pair is chemically inert and contributes no correlation worth
# recovering, so freezing it drops the problem from 12 qubits and 4 electrons to
# 10 qubits and 2 electrons -- a much easier optimization for a
# hardware-efficient ansatz, at the cost of a reference energy that is CASCI in
# this space rather than full-space FCI. `validate` checks against whichever one
# the space implies, so the comparison stays honest either way.
LIH_FROZEN_CORE = MoleculeSpec(
    name="LiH CASCI(2e,5o), frozen 1s core",
    atoms=_ATOMS,
    basis="sto-3g",
    charge=0,
    spin=0,
    n_active_orbitals=5,
    n_active_electrons=2,
    geometry_source=_GEOMETRY_SOURCE,
    model_note=(
        "Vacuum CASCI(2e,5o) in canonical RHF orbitals with Li 1s frozen. The "
        "reference energy is CASCI in this active space, NOT full-space FCI: an "
        "energy below it is proof the state left the two-electron sector, not "
        "proof of better chemistry."
    ),
    # A Mulliken localization check is meaningful when an active orbital could
    # sit on the wrong fragment of a large molecule. LiH has two atoms, so every
    # orbital is trivially on the target set and the check would always pass.
    diagnostic_atom_indices=(),
)

# The whole STO-3G space: no frozen core, so CASCI(4e,6o) is exactly FCI and the
# reference is the basis-set FCI energy. 12 qubits, still exactly diagonalizable
# by `validate`, but 4 electrons in 12 spin orbitals is a harder VQE.
LIH_FULL_SPACE = MoleculeSpec(
    name="LiH FCI(4e,6o), full STO-3G space",
    atoms=_ATOMS,
    basis="sto-3g",
    charge=0,
    spin=0,
    n_active_orbitals=6,
    n_active_electrons=4,
    geometry_source=_GEOMETRY_SOURCE,
    model_note=(
        "Vacuum full STO-3G space in canonical RHF orbitals. With every orbital "
        "active, CASCI is FCI, so the reference is the exact STO-3G energy. "
        "STO-3G itself is roughly 1 kcal/mol from the basis-set-limit LiH "
        "energy, so 'chemical accuracy' here means against FCI in this basis, "
        "not against experiment."
    ),
    diagnostic_atom_indices=(),
)

SPACES = {"frozen-core": LIH_FROZEN_CORE, "full": LIH_FULL_SPACE}
DEFAULT_SPACE = "frozen-core"
