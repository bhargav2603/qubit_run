#!/usr/bin/env python3
"""ADAPT-VQE: grow the ansatz one operator at a time, largest gradient first.

Grimsley et al., Nat. Commun. 10, 3007 (2019). Instead of committing to a fixed
excitation list, ADAPT starts from the Hartree-Fock determinant and repeatedly
appends whichever pool operator has the largest energy gradient, re-optimizing
every parameter after each addition. It converges toward FCI with far fewer
parameters than UCCSD, at the cost of the screening step.

**Why this is implemented here rather than taken from qiskit-algorithms.** Its
`AdaptVQE` is built on the same gradient machinery whose
`ReverseEstimatorGradient` took 619 s for a single 26-component UCCSD gradient in
this environment -- roughly 450x slower than the batched finite differences this
workflow uses. Screening a pool of ~100 operators through that path every macro
iteration is not viable.

**How the screening is made cheap.** For a Hermitian pool operator P and
U(theta) = exp(-i theta P), the gradient at theta = 0 is

    dE/dtheta = i <psi| [P, H] |psi> = -2 Im( <P psi | H psi> )

which needs one sparse matrix-vector product per pool operator against a
statevector we already have. At 12 qubits that is a 4096-element vector and ~100
matvecs -- milliseconds, against one circuit execution per operator if it were
done on the simulator. Aer still prepares the state; only the screening
arithmetic moves to scipy.

Honest cost note: screening does not make the O(N^4) operator count disappear,
it only makes each screen cheap. ADAPT's win over k-UpCCGSD here is a smaller
final parameter count, not a smaller pool.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np


def build_operator_pool(n_qubits: int, n_electrons: int) -> tuple[list[Any], list[Any]]:
    """The UCCSD singles and doubles generators, as Hermitian SparsePauliOps.

    qiskit-nature already emits exactly these for `UCCSD`, in the same blocked
    spin-orbital convention the rest of this workflow maps operators into, so
    the pool needs no reordering.
    """
    from qiskit_nature.second_q.circuit.library import UCCSD
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if n_electrons % 2:
        raise ValueError("this workflow assumes a closed-shell singlet")
    n_spatial = n_qubits // 2
    particles = (n_electrons // 2, n_electrons // 2)
    ansatz = UCCSD(n_spatial, particles, JordanWignerMapper())
    return list(ansatz.operators), list(ansatz.excitation_list)


class AdaptState:
    """The growing circuit, its parameters, and the chosen excitations."""

    def __init__(self, n_qubits: int, n_electrons: int) -> None:
        from qiskit_runtime import _nature_pieces

        _, _, _, self.initial_state = _nature_pieces(n_qubits, n_electrons)
        self.n_qubits = n_qubits
        self.operators: list[Any] = []
        self.excitations: list[Any] = []
        self.values: np.ndarray = np.zeros(0)

    def circuit(self) -> Any:
        """HF reference followed by exp(-i theta_j P_j) for each chosen operator."""
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import PauliEvolutionGate

        circuit = self.initial_state.copy()
        if not self.operators:
            return circuit
        angles = ParameterVector("t", len(self.operators))
        for angle, operator in zip(angles, self.operators):
            circuit.append(
                PauliEvolutionGate(operator, time=angle), range(self.n_qubits)
            )
        return circuit


def screen_pool(
    statevector: np.ndarray, hamiltonian_sparse: Any, pool_sparse: Sequence[Any]
) -> np.ndarray:
    """|dE/dtheta| at theta = 0 for every pool operator.

    dE/dtheta = -2 Im( <P psi | H psi> ), one sparse matvec per operator.
    """
    h_psi = hamiltonian_sparse @ statevector
    gradients = np.empty(len(pool_sparse), dtype=float)
    for index, operator in enumerate(pool_sparse):
        gradients[index] = -2.0 * float(np.imag(np.vdot(operator @ statevector, h_psi)))
    return gradients


def run_adapt(
    n_qubits: int,
    n_electrons: int,
    hamiltonian: Any,
    statevector_of: Callable[[Any, np.ndarray], np.ndarray],
    minimize: Callable[[Any, np.ndarray], tuple[float, np.ndarray, dict[str, Any]]],
    max_operators: int = 40,
    gradient_tolerance: float = 1.0e-3,
    energy_tolerance: float = 1.0e-8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Grow and optimize an ADAPT ansatz.

    `statevector_of(circuit, values)` must return the simulated statevector, and
    `minimize(circuit, start)` must run a VQE over that circuit. Both are
    injected so this module stays independent of the Aer plumbing and can be
    tested against exact linear algebra.
    """
    from scipy.sparse import csr_matrix

    pool, excitations = build_operator_pool(n_qubits, n_electrons)
    pool_sparse = [csr_matrix(operator.to_matrix(sparse=True)) for operator in pool]
    hamiltonian_sparse = csr_matrix(hamiltonian.to_matrix(sparse=True))

    state = AdaptState(n_qubits, n_electrons)
    history: list[dict[str, Any]] = []
    energy = float("nan")
    started = time.perf_counter()

    for iteration in range(max_operators):
        statevector = statevector_of(state.circuit(), state.values)
        gradients = screen_pool(statevector, hamiltonian_sparse, pool_sparse)
        magnitudes = np.abs(gradients)
        best = int(np.argmax(magnitudes))
        norm = float(np.linalg.norm(gradients))

        if magnitudes[best] < gradient_tolerance:
            if verbose:
                print(
                    f"  ADAPT converged: largest |gradient| {magnitudes[best]:.3e} "
                    f"< {gradient_tolerance:.1e} after {len(state.operators)} operators",
                    flush=True,
                )
            break

        state.operators.append(pool[best])
        state.excitations.append(excitations[best])
        start = np.concatenate([state.values, [0.0]])
        previous_energy = energy
        energy, state.values, diagnostics = minimize(state.circuit(), start)
        history.append({
            "operators": len(state.operators),
            "chosen_excitation": str(excitations[best]),
            "largest_gradient": float(magnitudes[best]),
            "gradient_norm": norm,
            "energy_hartree": energy,
            "optimizer_iterations": diagnostics.get("iterations"),
        })
        if verbose:
            print(
                f"  [{len(state.operators):>3}] {str(excitations[best]):<26} "
                f"|g|max={magnitudes[best]:.3e}  E={energy:.10f} Ha",
                flush=True,
            )
        if np.isfinite(previous_energy) and abs(previous_energy - energy) < energy_tolerance:
            if verbose:
                print(
                    f"  ADAPT stopped: energy improved by less than "
                    f"{energy_tolerance:.1e} Ha",
                    flush=True,
                )
            break
    else:
        if verbose:
            print(f"  ADAPT stopped: reached the {max_operators}-operator cap", flush=True)

    return {
        "energy_hartree": energy,
        "values": state.values,
        "circuit": state.circuit(),
        "n_operators": len(state.operators),
        "n_parameters": int(state.values.size),
        "pool_size": len(pool),
        "excitations": [str(e) for e in state.excitations],
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "gradient_tolerance": gradient_tolerance,
        "energy_tolerance": energy_tolerance,
    }
