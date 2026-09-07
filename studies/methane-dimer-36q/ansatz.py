"""The LUCJ ansatz and its Qiskit circuit.

The paper prepares its samples with a truncated local unitary cluster Jastrow
(LUCJ) ansatz seeded from CCSD amplitudes, built with ffsim.  This module does
the same and nothing more -- ffsim owns the definition of the operator and its
circuit, so there is no reimplementation here to drift out of agreement.

Where the qubit count comes from
--------------------------------
16 spatial orbitals occupy 32 qubits under Jordan-Wigner.  The LUCJ ansatz also
needs alpha-beta density-density interactions, and on a heavy-hex lattice those
are routed through ancillas.  The standard heavy-hex layout permits an
alpha-beta interaction on every fourth orbital, so

    len(range(0, 16, 4)) = 4 ancillas   ->   32 + 4 = 36 qubits

which is exactly the figure in the paper's Figure 2B.  `validate_layout` below
asserts that identity rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import paper

# The paper does not state n_reps. Two repetitions is the common LUCJ default
# and the value used in the IBM SQD tutorials; it is exposed as a flag and
# recorded in every result file. See paper.SQD_HYPERPARAMETER_GAP.
DEFAULT_N_REPS = 2

# Heavy-hex permits an alpha-beta interaction every fourth orbital.
ALPHA_BETA_STRIDE = 4


@dataclass(frozen=True)
class LucjLayout:
    """Connectivity restrictions imposed on the diagonal Coulomb operators."""

    norb: int
    alpha_alpha: list[tuple[int, int]]
    alpha_beta: list[tuple[int, int]]
    n_reps: int

    @property
    def n_occupation_qubits(self) -> int:
        return 2 * self.norb

    @property
    def n_ancilla_qubits(self) -> int:
        return len(self.alpha_beta)

    @property
    def n_qubits_total(self) -> int:
        return self.n_occupation_qubits + self.n_ancilla_qubits

    @property
    def interaction_pairs(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        return (self.alpha_alpha, self.alpha_beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "norb": self.norb,
            "n_reps": self.n_reps,
            "n_occupation_qubits": self.n_occupation_qubits,
            "n_ancilla_qubits": self.n_ancilla_qubits,
            "n_qubits_total": self.n_qubits_total,
            "alpha_alpha_pairs": len(self.alpha_alpha),
            "alpha_beta_pairs": len(self.alpha_beta),
        }


def heavy_hex_layout(norb: int, *, n_reps: int = DEFAULT_N_REPS) -> LucjLayout:
    """Interaction pairs for a heavy-hex device.

    alpha-alpha interactions run along the nearest-neighbour chain within each
    spin block; alpha-beta interactions are restricted to every fourth orbital,
    which is what the heavy-hex lattice can route without extra swaps.
    """
    alpha_alpha = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta = [(p, p) for p in range(0, norb, ALPHA_BETA_STRIDE)]
    return LucjLayout(
        norb=norb, alpha_alpha=alpha_alpha, alpha_beta=alpha_beta, n_reps=n_reps
    )


def validate_layout(layout: LucjLayout) -> None:
    """Check the layout reproduces the paper's stated qubit budget."""
    if layout.norb != paper.N_ORBITALS:
        raise AssertionError(f"norb is {layout.norb}, expected {paper.N_ORBITALS}")
    if layout.n_occupation_qubits != paper.N_QUBITS_OCCUPATION:
        raise AssertionError(
            f"{layout.n_occupation_qubits} occupation qubits, "
            f"expected {paper.N_QUBITS_OCCUPATION}"
        )
    if layout.n_ancilla_qubits != paper.N_QUBITS_ANCILLA:
        raise AssertionError(
            f"heavy-hex layout implies {layout.n_ancilla_qubits} ancillas, but the "
            f"paper's Figure 2B shows {paper.N_QUBITS_ANCILLA}. The alpha-beta "
            "stride no longer matches the device topology assumed here."
        )
    if layout.n_qubits_total != paper.N_QUBITS_TOTAL:
        raise AssertionError(
            f"total is {layout.n_qubits_total} qubits, expected {paper.N_QUBITS_TOTAL}"
        )


def build_operator(mol_data, layout: LucjLayout, *, optimize: bool = False):
    """Construct the LUCJ operator from the CCSD amplitudes.

    Args:
        mol_data: an ``ffsim.MolecularData`` with ``ccsd_t1``/``ccsd_t2`` set
            (call ``reference.run_ccsd(..., store_amplitudes=True)`` first).
        layout: connectivity restrictions from ``heavy_hex_layout``.
        optimize: refine the double factorization numerically. Improves the
            ansatz at real CPU cost; off by default because the sampled
            configurations, not the ansatz energy, are what SQD consumes.
    """
    import ffsim

    if mol_data.ccsd_t2 is None:
        raise ValueError(
            "mol_data has no CCSD t2 amplitudes; run reference.run_ccsd(..., "
            "store_amplitudes=True) before building the ansatz."
        )
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        t1=mol_data.ccsd_t1,
        n_reps=layout.n_reps,
        interaction_pairs=layout.interaction_pairs,
        optimize=optimize,
    )


def build_circuit(mol_data, operator, *, measure: bool = True):
    """Hartree-Fock preparation followed by the LUCJ unitary."""
    import ffsim
    from qiskit import QuantumCircuit, QuantumRegister

    norb, nelec = mol_data.norb, mol_data.nelec
    register = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(register)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), register)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), register)
    if measure:
        circuit.measure_all()
    return circuit


def transpile_for_backend(circuit, backend, layout: LucjLayout, **pass_manager_kwargs):
    """Map the ansatz onto real hardware using ffsim's LUCJ pass manager.

    Returns ``(transpiled_circuit, chosen_qubit_pairs)``. The transpiled circuit
    is where the ancillas actually appear, so its qubit count -- not the
    logical circuit's -- is what should be compared against the paper's 36.
    """
    import ffsim

    pass_manager, pairs = ffsim.qiskit.generate_lucj_pass_manager(
        backend,
        norb=layout.norb,
        connectivity="heavy-hex",
        interaction_pairs=layout.interaction_pairs,
        **pass_manager_kwargs,
    )
    return pass_manager.run(circuit), pairs


def circuit_metrics(circuit) -> dict[str, Any]:
    """Depth and two-qubit gate counts -- the paper's Figure 3 quantities."""
    ops = circuit.count_ops()
    two_qubit = sum(
        count
        for instruction, count in ops.items()
        if instruction in {"cx", "cz", "ecr", "cy", "swap", "iswap", "rzz"}
    )
    return {
        "n_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "two_qubit_gates": two_qubit,
        "operations": dict(ops),
    }
