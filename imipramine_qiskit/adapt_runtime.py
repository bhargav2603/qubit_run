#!/usr/bin/env python3
"""ADAPT-VQE: the reference method the published imipramine circuits imitate.

`qiskit_runtime.py` runs a fixed hardware-efficient ansatz. That ansatz does not
conserve particle number, which is why this workflow carries a penalty term, a
leakage measurement and a contamination budget -- and it is also why a shallow
HEA is unlikely to reach 1.6 mHa on a twelve-qubit active space.

Koziell-Pipe et al. (arXiv:2607.22468) do not use a hardware-efficient ansatz.
They run **ADAPT-VQE with a UCCGSD operator pool** to produce reference circuits,
then train transformer models to emit similar circuits without re-running the
optimization. The transformer is an amortization trick for their ~15,000
geometries; for a single molecule the reference method *is* the target, so this
module implements that method directly.

Three consequences, all of which simplify the accuracy claim rather than
complicate it:

* **Every pool operator conserves particle number and Sz exactly**, so the state
  cannot leave the six-electron sector and the number penalty -- with all the
  machinery that hangs off it -- becomes unnecessary. <N> is still measured at
  the optimum rather than assumed.

  S^2 is a weaker story and is deliberately not overclaimed. Spin-complementing
  makes the *singles* proper singlet operators, and the `pair` pool is
  seniority-zero, so both commute with S^2 exactly; a general spin-complemented
  double does **not** -- complementing gives invariance under flipping Sz, which
  is not the same as commuting with S^2, and measurably 27 of 54 uccsd doubles
  and 315 of 420 uccgsd doubles fail it. So <S^2> is measured at the optimum and
  priced into the same contamination budget the hardware-efficient path uses.
  In practice it comes back at 1e-19: the Hamiltonian is spin free and the
  reference is a closed-shell singlet, so the variational minimum is the singlet
  ground state. But that is a result of the optimization, not a property of the
  ansatz, and the self-test checks it as one.
* **The ansatz grows until it stops helping.** ADAPT adds the single operator
  with the largest energy gradient, re-optimizes every parameter, and stops when
  no operator in the pool has a gradient above a threshold. Depth is an output,
  not a knob you scan.
* **The optimization is done in sparse linear algebra, not by simulating
  circuits.** At 12 qubits the state is 4096 amplitudes, so exp(theta A)|psi>
  is one `expm_multiply` (~1 ms) and the *entire* gradient costs 2k of them by
  the adjoint method -- against 2N+1 full circuit simulations for parameter
  shift. The Qiskit circuit is still built at the end, transpiled for a gate
  count, and its energy checked against the algebra: if the two disagree the
  result is rejected, so the speed is never bought with a weaker claim.
"""

from __future__ import annotations

import argparse
import platform
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from hamiltonian import (
    CHEMICAL_ACCURACY_MHA,
    CONTAMINATION_ENERGY_BUDGET_HA,
    CONTAMINATION_PRICE_HA,
    SECTOR_ENERGY_TOLERANCE_HA,
    CachedHamiltonian,
    MoleculeSpec,
    electron_number_operator,
    hartree_fock_occupation,
    load_cache,
    number_deviation_operator,
    physics_fingerprint,
    require_validation_receipt,
    spin_squared_qubit_operator,
    workflow_fingerprint,
    write_json_atomic,
)

POOLS = ("uccgsd", "uccsd", "pair")
# Below this largest-gradient value the pool has nothing left worth adding.
# 1e-3 Ha is the usual ADAPT setting; the energy is converged well past chemical
# accuracy by then, and the final gradient norm is recorded either way.
DEFAULT_GRADIENT_TOLERANCE = 1.0e-3
# The circuit must reproduce the algebra's energy this closely or the run is
# rejected. Exponentials of a single fermionic excitation generator synthesize
# exactly -- the Pauli strings in one generator mutually commute -- so any
# disagreement means that assumption broke, not that precision ran out.
CIRCUIT_AGREEMENT_HA = 1.0e-9


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


# --------------------------------------------------------------------------
# Operator pool
# --------------------------------------------------------------------------


def _spin_partner(spin_orbital: int) -> int:
    """Alpha <-> beta in OpenFermion's interleaved ordering (2p, 2p+1)."""
    return spin_orbital + 1 if spin_orbital % 2 == 0 else spin_orbital - 1


def _spin(spin_orbital: int) -> int:
    return spin_orbital % 2


def _excitation(creations: Sequence[int], annihilations: Sequence[int]) -> Any:
    """The anti-Hermitian generator a+_p a+_q ... a_s a_r  -  h.c."""
    from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

    term = " ".join(f"{index}^" for index in creations)
    term += " " + " ".join(str(index) for index in annihilations)
    operator = FermionOperator(term.strip())
    generator = normal_ordered(operator - hermitian_conjugated(operator))
    generator.compress(abs_tol=1.0e-12)
    return generator


def _spin_complement(indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(_spin_partner(index) for index in indices)


def build_pool(
    n_spin_orbitals: int, n_electrons: int, kind: str = "uccgsd"
) -> list[dict[str, Any]]:
    """Spin-complemented, particle-number- and Sz-conserving excitation pool.

    ``uccgsd`` is the paper's choice for twelve qubits: *generalized* singles and
    doubles, meaning every excitation between orbital pairs is allowed rather
    than only occupied -> virtual. That is a much larger pool, and it is what
    lets ADAPT recover correlation the reference determinant's occupation
    pattern would otherwise hide.

    ``uccsd`` restricts to occupied -> virtual relative to the Hartree-Fock
    determinant, which is an order of magnitude smaller and usually enough at
    this size. ``pair`` is the k-UpCCGSD paired-doubles set, the cheapest of the
    three.

    Each entry pairs an excitation with its spin complement under one shared
    parameter. That is what makes the *singles* proper singlet operators -- a
    lone spin-orbital single breaks S^2, the complemented pair does not -- and it
    halves the parameter count everywhere else.

    It does not make the general doubles S^2 eigenoperators, and this docstring
    used to claim otherwise. Complementing buys invariance under flipping Sz,
    which is strictly weaker than commuting with S^2; only the seniority-zero
    `pair` doubles get there. <S^2> is therefore measured at the optimum and
    priced, not assumed away. Particle number and Sz *are* exact for every entry
    in every pool.
    """
    from openfermion import jordan_wigner

    if kind not in POOLS:
        raise ValueError(f"Unknown pool {kind!r}; choose from {POOLS}")
    occupied = set(hartree_fock_occupation(n_electrons))
    orbitals = range(n_spin_orbitals)

    def allowed_single(target: int, source: int) -> bool:
        if kind == "pair":
            return False
        if kind == "uccsd":
            return source in occupied and target not in occupied
        return True

    def allowed_double(creations: tuple[int, int], annihilations: tuple[int, int]) -> bool:
        if kind == "uccsd":
            return all(index in occupied for index in annihilations) and all(
                index not in occupied for index in creations
            )
        if kind == "pair":
            # Both electrons of a spatial pair move together.
            return (
                creations[1] == _spin_partner(creations[0])
                and annihilations[1] == _spin_partner(annihilations[0])
            )
        return True

    seen: set[tuple] = set()
    pool: list[dict[str, Any]] = []

    def _canonical(
        creations: Sequence[int], annihilations: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Direction-independent key: T(a<-b) and T(b<-a) differ only by sign.

        The sign is absorbed into the parameter, so keeping both would put two
        copies of the same operator in the pool and let ADAPT pick the same
        excitation twice.
        """
        return min(
            (tuple(sorted(creations)), tuple(sorted(annihilations))),
            (tuple(sorted(annihilations)), tuple(sorted(creations))),
        )

    def add(indices: tuple[tuple[int, ...], tuple[int, ...]], label: str) -> None:
        creations, annihilations = indices
        partner = (_spin_complement(creations), _spin_complement(annihilations))
        signature = (
            min(
                _canonical(creations, annihilations),
                _canonical(partner[0], partner[1]),
            ),
            len(creations),
        )
        if signature in seen:
            return
        seen.add(signature)
        generator = _excitation(creations, annihilations)
        if partner != indices:
            generator = generator + _excitation(partner[0], partner[1])
            generator.compress(abs_tol=1.0e-12)
        if not generator.terms:
            return
        qubit_operator = jordan_wigner(generator)
        qubit_operator.compress(abs_tol=1.0e-12)
        if not qubit_operator.terms:
            return
        pool.append(
            {
                "label": label,
                "creations": list(creations),
                "annihilations": list(annihilations),
                "fermion": generator,
                "qubit": qubit_operator,
                "rank": len(creations),
            }
        )

    # Singles: same spin only, or Sz is not conserved.
    for target in orbitals:
        for source in orbitals:
            if source >= target or _spin(target) != _spin(source):
                continue
            if not allowed_single(target, source):
                continue
            add(((target,), (source,)), f"S {target}<-{source}")

    # Doubles: any two-in, two-out with Sz conserved. Both directions are
    # enumerated -- a UCCSD double moves electrons from low indices to high,
    # which is the *opposite* lexicographic order from a generalized one -- and
    # the canonical key above removes the duplicates.
    pairs = [(i, j) for i in orbitals for j in orbitals if i < j]
    for creations in pairs:
        for annihilations in pairs:
            if set(creations) == set(annihilations):
                continue
            if _spin(creations[0]) + _spin(creations[1]) != _spin(annihilations[0]) + _spin(
                annihilations[1]
            ):
                continue
            if not allowed_double(creations, annihilations):
                continue
            add(
                (creations, annihilations),
                f"D {creations[0]},{creations[1]}<-{annihilations[0]},{annihilations[1]}",
            )
    return pool


# --------------------------------------------------------------------------
# Sparse engine
# --------------------------------------------------------------------------


class SparseEngine:
    """Exact statevector algebra for one Hamiltonian and one operator pool.

    Nothing here is an approximation: the state is the full 2^n vector, every
    operator is its exact sparse matrix, and `expm_multiply` applies a matrix
    exponential to a vector without ever forming it. The Qiskit circuit built at
    the end is checked against this, not the other way round.
    """

    def __init__(
        self, chemistry: CachedHamiltonian, pool: Sequence[dict[str, Any]]
    ) -> None:
        from openfermion import get_sparse_operator

        self.n_qubits = chemistry.n_qubits
        self.n_electrons = chemistry.n_active_electrons
        self.dimension = 1 << self.n_qubits
        # Kept in operator form as well: the circuit check needs the observable
        # as Paulis, not as a matrix.
        self.qubit_hamiltonian = chemistry.hamiltonian
        self.hamiltonian = get_sparse_operator(
            chemistry.hamiltonian, n_qubits=self.n_qubits
        ).tocsr()
        self.number = get_sparse_operator(
            electron_number_operator(self.n_qubits), n_qubits=self.n_qubits
        ).tocsr()
        self.deviation = get_sparse_operator(
            number_deviation_operator(self.n_qubits, self.n_electrons),
            n_qubits=self.n_qubits,
        ).tocsr()
        self.spin_squared = get_sparse_operator(
            spin_squared_qubit_operator(self.n_qubits), n_qubits=self.n_qubits
        ).tocsr()
        self.pool = list(pool)
        self.pool_matrices = [
            get_sparse_operator(entry["qubit"], n_qubits=self.n_qubits).tocsr()
            for entry in self.pool
        ]
        self.reference = self._determinant(hartree_fock_occupation(self.n_electrons))
        self.exponentials = 0

    def _determinant(self, occupied: Iterable[int]) -> np.ndarray:
        """The Jordan-Wigner computational basis state with these orbitals filled.

        OpenFermion writes qubit 0 into the most significant bit of the vector
        index, which is the convention every sparse operator above already uses.
        """
        index = 0
        for qubit in occupied:
            index |= 1 << (self.n_qubits - 1 - int(qubit))
        vector = np.zeros(self.dimension, dtype=complex)
        vector[index] = 1.0
        return vector

    def _apply(self, matrix: Any, angle: float, vector: np.ndarray) -> np.ndarray:
        from scipy.sparse.linalg import expm_multiply

        if angle == 0.0:
            return vector
        self.exponentials += 1
        return expm_multiply(angle * matrix, vector)

    def evolve(self, angles: Sequence[float], operators: Sequence[int]) -> list[np.ndarray]:
        """Every intermediate state, so the adjoint sweep needs no recomputation."""
        states = [self.reference]
        for angle, index in zip(angles, operators):
            states.append(self._apply(self.pool_matrices[index], float(angle), states[-1]))
        return states

    def state(self, angles: Sequence[float], operators: Sequence[int]) -> np.ndarray:
        return self.evolve(angles, operators)[-1]

    def expectation(self, matrix: Any, vector: np.ndarray) -> float:
        return float(np.real(np.vdot(vector, matrix @ vector)))

    def energy(self, angles: Sequence[float], operators: Sequence[int]) -> float:
        return self.expectation(self.hamiltonian, self.state(angles, operators))

    def energy_and_gradient(
        self, angles: Sequence[float], operators: Sequence[int]
    ) -> tuple[float, np.ndarray]:
        """Adjoint (reverse-mode) gradient: 2k exponentials for k parameters.

        Parameter shift would need 2k+1 *full* evaluations, each of which is
        itself k exponentials -- so this is O(k) rather than O(k^2), and exact
        rather than exact-only-for-RY. The self-test checks it against finite
        differences.
        """
        states = self.evolve(angles, operators)
        final = states[-1]
        energy = self.expectation(self.hamiltonian, final)
        sigma = self.hamiltonian @ final
        gradient = np.zeros(len(operators))
        for position in reversed(range(len(operators))):
            matrix = self.pool_matrices[operators[position]]
            gradient[position] = 2.0 * float(
                np.real(np.vdot(sigma, matrix @ states[position + 1]))
            )
            sigma = self._apply(matrix, -float(angles[position]), sigma)
        return energy, gradient

    def pool_gradients(self, vector: np.ndarray) -> np.ndarray:
        """dE/dtheta at theta = 0 for every pool operator appended at the end.

        For anti-Hermitian A this is <psi|[H, A]|psi>, computed as one sparse
        matrix-vector product per candidate.
        """
        sigma = self.hamiltonian @ vector
        return np.array(
            [
                2.0 * float(np.real(np.vdot(sigma, matrix @ vector)))
                for matrix in self.pool_matrices
            ]
        )

    def sector_observables(self, vector: np.ndarray) -> tuple[float, float, float]:
        return (
            self.expectation(self.number, vector),
            self.expectation(self.deviation, vector),
            self.expectation(self.spin_squared, vector),
        )


# --------------------------------------------------------------------------
# The ADAPT loop
# --------------------------------------------------------------------------


def adapt_vqe(
    engine: SparseEngine,
    max_operators: int,
    gradient_tolerance: float,
    optimizer: str,
    optimizer_tolerance: float,
    maxiter: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Grow the ansatz one operator at a time until the pool has nothing left."""
    from scipy.optimize import minimize

    operators: list[int] = []
    angles: list[float] = []
    history: list[dict[str, Any]] = []
    energy = engine.expectation(engine.hamiltonian, engine.reference)
    converged = False
    stop_reason = "operator budget exhausted"

    if verbose:
        print(f"  {'it':>3}  {'|grad|max':>11}  {'energy (Ha)':>18}  {'dE (mHa)':>11}  operator",
              flush=True)

    for iteration in range(1, max_operators + 1):
        state = engine.state(angles, operators)
        gradients = engine.pool_gradients(state)
        best = int(np.argmax(np.abs(gradients)))
        largest = float(abs(gradients[best]))
        if largest < gradient_tolerance:
            converged = True
            stop_reason = f"largest pool gradient {largest:.3e} < {gradient_tolerance:g}"
            break

        operators.append(best)
        angles.append(0.0)

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = engine.energy_and_gradient(values, operators)
            return value, gradient

        outcome = minimize(
            objective,
            np.asarray(angles, dtype=float),
            jac=True,
            method="L-BFGS-B" if optimizer == "l-bfgs-b" else "BFGS",
            options=(
                {"maxiter": maxiter, "ftol": optimizer_tolerance, "gtol": optimizer_tolerance}
                if optimizer == "l-bfgs-b"
                else {"maxiter": maxiter, "gtol": optimizer_tolerance}
            ),
        )
        angles = [float(value) for value in np.asarray(outcome.x).reshape(-1)]
        previous, energy = energy, float(outcome.fun)
        history.append(
            {
                "iteration": iteration,
                "operator_index": best,
                "operator": engine.pool[best]["label"],
                "rank": engine.pool[best]["rank"],
                "pool_gradient": largest,
                "energy_hartree": energy,
                "energy_change_millihartree": 1000.0 * (energy - previous),
                "parameters": len(angles),
                "optimizer_iterations": int(getattr(outcome, "nit", -1)),
                "optimizer_converged": bool(outcome.success),
            }
        )
        if verbose:
            print(
                f"  {iteration:3d}  {largest:11.3e}  {energy:18.12f}"
                f"  {1000.0 * (energy - previous):+11.4f}  {engine.pool[best]['label']}",
                flush=True,
            )

    state = engine.state(angles, operators)
    final_gradients = engine.pool_gradients(state)
    return {
        "operators": operators,
        "angles": angles,
        "energy_hartree": engine.expectation(engine.hamiltonian, state),
        "history": history,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_pool_gradient_max": float(np.max(np.abs(final_gradients)))
        if len(final_gradients)
        else 0.0,
        "state": state,
    }


# --------------------------------------------------------------------------
# Circuit emission and verification
# --------------------------------------------------------------------------


def to_circuit(
    engine: SparseEngine, operators: Sequence[int], angles: Sequence[float]
) -> Any:
    """The same state as a Qiskit circuit: X gates, then exp(theta A) factors.

    ``exp(theta A)`` with A anti-Hermitian is ``exp(-i theta B)`` for the
    Hermitian ``B = i A``, which is what `PauliEvolutionGate` takes. The Pauli
    strings inside one fermionic excitation generator mutually commute, so
    first-order synthesis is exact rather than a Trotter approximation -- and
    `verify_circuit` proves that rather than trusting it.

    That same commutation is what lets `preserve_order=False` be safe. Qiskit's
    default synthesis emits one CNOT staircase per Pauli string in source order;
    allowing the synthesizer to reorder them exposes cancellations between
    adjacent staircases. The reordering happens strictly WITHIN one gate, so it
    can only permute strings that commute -- the excitation generators
    themselves do not commute with each other and are never reordered.

    Both settings are needed together and neither helps alone. Measured on the
    40-operator uccsd ansatz, two-qubit gates after `optimization_level=3`:

        chain  + preserve_order=True   4179   (Qiskit default)
        fountain                       4791
        chain  + preserve_order=False  5395
        fountain + preserve_order=False 2758  <-- 34% below the default

    The optimized circuit's statevector matches the default one to a fidelity of
    1 - 1.8e-15, and `verify_circuit` re-checks the energy at 1e-9 Ha on every
    run, so a future Qiskit that changes this synthesis cannot pass silently.
    """
    return emit_circuit(
        engine.n_qubits,
        engine.n_electrons,
        [engine.pool[index]["qubit"] for index in operators],
        angles,
    )


def emit_circuit(
    n_qubits: int,
    n_electrons: int,
    generators: Sequence[Any],
    angles: Sequence[float],
) -> Any:
    """Circuit for a list of anti-Hermitian generators, without a SparseEngine.

    `to_circuit` needs an engine because that is where the pool lives, and an
    engine needs a Hamiltonian. Rebuilding a circuit from a stored result needs
    neither -- the operators and angles are in the JSON, and the pool is pure
    combinatorics. Splitting the emission out is what lets `visualize` measure
    gate counts along a finished run's trajectory on a machine that has no cache
    and no PySCF.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    from qiskit_runtime import to_sparse_pauli_op

    synthesis = LieTrotter(cx_structure="fountain", preserve_order=False)
    circuit = QuantumCircuit(n_qubits)
    for qubit in hartree_fock_occupation(n_electrons):
        circuit.x(qubit)
    for generator, angle in zip(generators, angles):
        hermitian = to_sparse_pauli_op(1.0j * generator, n_qubits)
        circuit.append(
            PauliEvolutionGate(hermitian, time=float(angle), synthesis=synthesis),
            range(n_qubits),
        )
    return circuit


def circuit_from_result(result: dict[str, Any], n_operators: int | None = None) -> Any:
    """Rebuild a finished ADAPT run's circuit, optionally truncated to a prefix.

    ADAPT builds its ansatz by appending, so the first *k* operators of a
    finished run are exactly the ansatz that run had at iteration *k*. That makes
    every prefix a real, already-optimized data point rather than a hypothetical
    -- which is what turns one run into a whole accuracy-versus-cost curve, at
    the price of a transpile per point instead of a re-optimization.

    ADAPT re-optimizes *every* parameter at each step, so a prefix of the final
    angle vector is not the state the run passed through at iteration k. The
    result schema records `parameters` per iteration as a count, not as the angle
    vector, so on current results this always truncates the final angles and the
    reconstructed prefix is a valid k-operator ansatz rather than a replay of
    iteration k.

    That is the right circuit for a *cost* measurement and the wrong one for an
    energy: gate count is set by which operators are present and their
    transpiled structure, not by the numeric angles, while the energy depends on
    the angles entirely. So `visualize` reads gate counts from these circuits and
    energies from `adapt.history`, and never takes an energy off a prefix. If the
    schema later grows a per-iteration angle vector this picks it up
    automatically and the prefix becomes an exact replay.
    """
    chemistry = result["chemistry"]
    ansatz = result["ansatz"]
    n_qubits = int(chemistry["qubits"])
    n_electrons = int(chemistry["active_electrons"])
    pool = build_pool(n_qubits, n_electrons, str(ansatz["pool"]))
    by_label = {entry["label"]: entry for entry in pool}

    count = n_operators or len(ansatz["operators"])
    labels = list(ansatz["operators"])[:count]
    angles = list(ansatz["angles"])[:count]
    history = (result.get("adapt") or {}).get("history") or []
    if count <= len(history):
        # `parameters` is a count in the current schema; only a real angle vector
        # of the right length is usable, and anything else falls through.
        recorded = history[count - 1].get("parameters")
        if isinstance(recorded, (list, tuple)) and len(recorded) == count:
            angles = [float(angle) for angle in recorded]
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise KeyError(
            f"{len(missing)} operator(s) in the result are not in a freshly built "
            f"{ansatz['pool']!r} pool (first: {missing[0]!r}). The pool definition "
            "changed since this result was written; re-run `adapt` rather than "
            "charting a circuit that is not the one that produced the energy."
        )
    return emit_circuit(
        n_qubits, n_electrons, [by_label[label]["qubit"] for label in labels], angles
    )


def circuit_cost(circuit: Any, optimization_level: int = 3) -> dict[str, int]:
    """Two-qubit gates and depth after transpilation, the published cost measure."""
    from qiskit import transpile

    basis = ["rz", "ry", "rx", "h", "cx"]
    decomposed = transpile(
        circuit, basis_gates=basis, optimization_level=optimization_level
    )
    counts = decomposed.count_ops()
    return {
        "two_qubit_gates": int(counts.get("cx", 0)),
        "depth": int(decomposed.depth()),
    }


def verify_circuit(
    engine: SparseEngine,
    circuit: Any,
    reference_energy: float,
    optimization_level: int = 3,
) -> dict[str, Any]:
    """Re-derive the energy from the circuit and count what it would cost.

    Two independent things happen here, and only the first gates the result:

    * the circuit's own statevector is contracted against the Hamiltonian and
      compared with the energy the sparse algebra reported. Agreement proves the
      exponentials synthesized exactly rather than Trotterized;
    * the circuit is transpiled to a two-qubit basis for a gate count, which is
      the number the published work reports (1440 -> 244 two-qubit gates after
      pytket optimization at twelve qubits).

    The transpile dominates the cost at `optimization_level=3`, so callers that
    only need the synthesis proof -- the self-test -- can lower it.
    """
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    from qiskit_runtime import to_sparse_pauli_op

    basis = ["rz", "ry", "rx", "h", "cx"]
    decomposed = transpile(
        circuit, basis_gates=basis, optimization_level=optimization_level
    )
    statevector = Statevector(circuit)
    observable = to_sparse_pauli_op(engine.qubit_hamiltonian, engine.n_qubits)
    energy = float(np.real(statevector.expectation_value(observable)))
    counts = decomposed.count_ops()
    return {
        "energy_hartree": energy,
        "energy_error_hartree": energy - reference_energy,
        "exact_synthesis": bool(abs(energy - reference_energy) <= CIRCUIT_AGREEMENT_HA),
        "two_qubit_gates": int(counts.get("cx", 0)),
        "single_qubit_gates": int(sum(v for k, v in counts.items() if k != "cx")),
        "depth": int(decomposed.depth()),
        "basis": basis,
        "optimization_level": int(optimization_level),
    }


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


def adapt_main(
    spec: MoleculeSpec,
    default_cache: str,
    default_results: str,
    title: str,
) -> int:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--cache", type=Path, default=Path(default_cache))
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument(
        "--pool",
        choices=POOLS,
        default="uccgsd",
        help="Operator pool. uccgsd is the paper's choice at twelve qubits; "
        "uccsd is an order of magnitude smaller and usually enough; pair is "
        "the cheapest.",
    )
    parser.add_argument(
        "--max-operators",
        type=int,
        default=40,
        help="Hard cap on ansatz length. ADAPT normally stops on the gradient "
        "threshold well before this; the published circuits used a median of "
        "8-19 operators at twelve qubits.",
    )
    parser.add_argument(
        "--gradient-tolerance", type=float, default=DEFAULT_GRADIENT_TOLERANCE
    )
    parser.add_argument("--optimizer", choices=("l-bfgs-b", "bfgs"), default="l-bfgs-b")
    parser.add_argument("--optimizer-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument(
        "--no-circuit",
        action="store_true",
        help="Skip building and verifying the Qiskit circuit. The energy is "
        "unaffected; you lose the gate count and the synthesis proof.",
    )
    args = parser.parse_args()
    if args.max_operators < 1 or args.maxiter < 1:
        parser.error("--max-operators and --maxiter must be positive")
    if args.gradient_tolerance <= 0:
        parser.error("--gradient-tolerance must be positive")
    if not args.cache.is_file():
        raise FileNotFoundError(
            f"Missing {args.cache}. Run `python run.py prepare` where PySCF is "
            "available, then `python run.py validate`."
        )
    result_path = args.result or Path(default_results) / f"adapt_{args.pool}.json"

    # ADAPT applies no penalty -- the pool cannot leave the sector -- so the
    # receipt entry it needs is the unconstrained one validation already records.
    validation_receipt = require_validation_receipt(args.cache, spec, 0.0, 0.0)

    started = time.perf_counter()
    chemistry = load_cache(args.cache, spec)

    print("=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)
    print(f"Active space / qubits      : {chemistry.n_active_electrons}e,"
          f"{chemistry.n_qubits // 2}o / {chemistry.n_qubits}", flush=True)
    print(f"Basis / Pauli terms        : {chemistry.metadata['basis']} / "
          f"{len(chemistry.hamiltonian.terms)}", flush=True)
    print(f"HF / CASCI energies        : {chemistry.hartree_fock_energy:.12f} / "
          f"{chemistry.reference_energy:.12f} Ha", flush=True)

    pool_started = time.perf_counter()
    pool = build_pool(chemistry.n_qubits, chemistry.n_active_electrons, args.pool)
    engine = SparseEngine(chemistry, pool)
    pool_seconds = time.perf_counter() - pool_started
    singles = sum(1 for entry in pool if entry["rank"] == 1)
    print(f"Pool / singles / doubles   : {args.pool} / {singles} / {len(pool) - singles}"
          f"  ({pool_seconds:.1f} s to build)", flush=True)

    reference_energy = engine.expectation(engine.hamiltonian, engine.reference)
    reference_error = reference_energy - chemistry.hartree_fock_energy
    if abs(reference_error) > 1.0e-6 + float(
        chemistry.metadata["truncation_l1_bound_hartree"]
    ):
        raise RuntimeError(
            f"The reference determinant is not Hartree-Fock (error {reference_error:+.3e} Ha). "
            "Run `python run.py selftest` before trusting any result."
        )
    print(f"Reference determinant      : {reference_energy:.12f} Ha "
          f"({reference_error:+.3e} vs RHF)", flush=True)
    print("-" * 72, flush=True)

    outcome = adapt_vqe(
        engine,
        args.max_operators,
        args.gradient_tolerance,
        args.optimizer,
        args.optimizer_tolerance,
        args.maxiter,
    )
    state = outcome.pop("state")
    energy = outcome["energy_hartree"]
    electron_count, leakage, spin_squared = engine.sector_observables(state)
    elapsed_optimization = time.perf_counter() - started

    circuit_report: dict[str, Any] | None = None
    if not args.no_circuit:
        circuit_report = verify_circuit(
            engine, to_circuit(engine, outcome["operators"], outcome["angles"]), energy
        )

    error_mha = 1000.0 * (energy - chemistry.reference_energy)
    contamination_energy = CONTAMINATION_PRICE_HA * (leakage + abs(spin_squared))
    sector_ground = validation_receipt.get("sector_ground_energy_hartree")
    sector_deficit = None if sector_ground is None else float(sector_ground) - energy
    sector_bound_ok = sector_deficit is None or sector_deficit <= max(
        contamination_energy, SECTOR_ENERGY_TOLERANCE_HA
    )
    circuit_ok = circuit_report is None or circuit_report["exact_synthesis"]

    verified = (
        abs(error_mha) <= CHEMICAL_ACCURACY_MHA
        and contamination_energy <= CONTAMINATION_ENERGY_BUDGET_HA
        and sector_bound_ok
        and circuit_ok
    )
    if verified:
        status = "verified_chemical_accuracy"
    elif not sector_bound_ok:
        status = "inconsistent"
    elif not circuit_ok:
        status = "circuit_disagrees"
    elif contamination_energy > CONTAMINATION_ENERGY_BUDGET_HA:
        # Two different failures wear the same budget. Particle-number leakage is
        # structurally impossible for this pool and means something is broken;
        # spin contamination just means the optimization stopped short of the
        # singlet minimum. They get different names because they get different
        # fixes.
        status = "symmetry_broken" if leakage > 1.0e-12 else "spin_contaminated"
    elif energy > chemistry.hartree_fock_energy:
        status = "above_hartree_fock"
    else:
        status = "outside_chemical_accuracy"
    elapsed = time.perf_counter() - started

    result = {
        "status": status,
        "accuracy_definition": (
            "ADAPT-VQE within 1.6 mHa of CASCI for the identical cached "
            "active-space Hamiltonian, with symmetry breaking priced below "
            f"{1000.0 * CONTAMINATION_ENERGY_BUDGET_HA:.2f} mHa"
        ),
        "backend": "exact sparse statevector algebra (SciPy), circuit verified in Qiskit",
        "method": "ADAPT-VQE",
        "workflow_sha256": workflow_fingerprint(),
        "physics_sha256": physics_fingerprint(),
        # ADAPT has no stochastic element: the operator choice is the largest
        # pool gradient, every new angle starts at exactly zero, and L-BFGS-B is
        # deterministic. There is nothing for a seed to control, so re-running
        # this command reproduces the result bit for bit.
        "deterministic": True,
        "chemistry": chemistry.metadata,
        "published_reference": spec.published_reference,
        "validation": validation_receipt,
        "ansatz": {
            "name": f"adapt-vqe ({args.pool})",
            "layers": None,
            "parameters": len(outcome["operators"]),
            "pool": args.pool,
            "pool_size": len(pool),
            "pool_singles": singles,
            "pool_doubles": len(pool) - singles,
            "hartree_fock_frame": False,
            "start": "Hartree-Fock determinant, theta = 0",
            "number_penalty_hartree": 0.0,
            "spin_squared_penalty_hartree": 0.0,
            "contamination_price_hartree": CONTAMINATION_PRICE_HA,
            "operators": [engine.pool[index]["label"] for index in outcome["operators"]],
            "angles": outcome["angles"],
        },
        "adapt": {
            "gradient_tolerance": args.gradient_tolerance,
            "max_operators": args.max_operators,
            "converged": outcome["converged"],
            "stop_reason": outcome["stop_reason"],
            "final_pool_gradient_max": outcome["final_pool_gradient_max"],
            "history": outcome["history"],
            "matrix_exponentials": engine.exponentials,
            "pool_build_seconds": pool_seconds,
            "optimization_seconds": elapsed_optimization,
        },
        "circuit": circuit_report,
        "optimizer": {
            "optimizer": args.optimizer,
            "tolerance": args.optimizer_tolerance,
            "maxiter_requested": args.maxiter,
            "iterations": len(outcome["history"]),
            "gradient": "adjoint (exact)",
            "converged": bool(outcome["converged"]),
            "message": outcome["stop_reason"],
        },
        "vqe": {
            "objective_energy_hartree": energy,
            "physical_energy_hartree": energy,
            "electron_number": electron_count,
            "number_deviation_squared": leakage,
            "spin_squared": spin_squared,
            "contamination_energy_hartree": contamination_energy,
            "contamination_energy_budget_hartree": CONTAMINATION_ENERGY_BUDGET_HA,
            "reference_energy_hartree": chemistry.reference_energy,
            "hartree_fock_energy_hartree": chemistry.hartree_fock_energy,
            "sector_ground_energy_hartree": sector_ground,
            "sector_energy_deficit_hartree": sector_deficit,
            "sector_bound_respected": bool(sector_bound_ok),
            "error_millihartree": error_mha,
            "within_chemical_accuracy": bool(verified),
            "parameters": list(outcome["angles"]),
        },
        "runtime_seconds": elapsed,
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "versions": {
            name: _package_version(name)
            for name in ("qiskit", "qiskit-aer", "openfermion", "pyscf", "numpy", "scipy")
        },
    }
    write_json_atomic(result_path, result)

    print("-" * 72, flush=True)
    print(f"Operators / stop            : {len(outcome['operators'])} / {outcome['stop_reason']}", flush=True)
    print(f"Final energy                : {energy:.12f} Ha", flush=True)
    print(f"Electron number / leakage   : {electron_count:.10f} / {leakage:.3e}", flush=True)
    print(f"Spin contamination <S^2>    : {spin_squared:.3e}", flush=True)
    print(
        f"Contamination / budget      : {1000.0 * contamination_energy:.4f}"
        f" / {1000.0 * CONTAMINATION_ENERGY_BUDGET_HA:.4f} mHa",
        flush=True,
    )
    if sector_deficit is not None:
        print(f"Below exact sector ground   : {1000.0 * sector_deficit:+.6f} mHa"
              f" ({'ok' if sector_bound_ok else 'IMPOSSIBLE'})", flush=True)
    print(f"Correlation recovered       : "
          f"{100.0 * (energy - chemistry.hartree_fock_energy) / (chemistry.reference_energy - chemistry.hartree_fock_energy):.4f}%", flush=True)
    print(f"Error vs CASCI              : {error_mha:+.6f} mHa", flush=True)
    if circuit_report is not None:
        print(f"Circuit 2q gates / depth    : {circuit_report['two_qubit_gates']}"
              f" / {circuit_report['depth']}", flush=True)
        print(f"Circuit vs algebra          : {1000.0 * circuit_report['energy_error_hartree']:+.3e} mHa"
              f" ({'exact synthesis' if circuit_report['exact_synthesis'] else 'MISMATCH'})", flush=True)
    print(f"Matrix exponentials         : {engine.exponentials}", flush=True)
    print(f"Runtime / result            : {elapsed:.1f} s / {result_path}", flush=True)

    reference = spec.published_reference or {}
    if reference:
        published = reference.get("statevector_error_millihartree")
        print("-" * 72, flush=True)
        print(f"Published anchor            : {reference.get('source', 'literature')}", flush=True)
        if published is not None:
            print(f"  their statevector ADAPT-GQE: {float(published):+.2f} mHa vs their CASCI", flush=True)
            print(f"  this run                   : {error_mha:+.6f} mHa vs this CASCI", flush=True)
        print("  totals are NOT comparable (different conformer); compare these errors.", flush=True)
    print("-" * 72, flush=True)
    if status == "verified_chemical_accuracy":
        print("VERIFIED: within active-space chemical accuracy.", flush=True)
    elif status == "circuit_disagrees":
        print("CIRCUIT DISAGREES: the emitted circuit does not reproduce the "
              "optimized energy, so the exponentials did not synthesize exactly. "
              "The energy is still correct; the circuit is not.", flush=True)
    elif status == "inconsistent":
        print("INCONSISTENT: energy below the exact sector ground state. Re-run "
              "`validate` and `selftest`; do not report this number.", flush=True)
    elif status == "symmetry_broken":
        print("SYMMETRY BROKEN: the state left the six-electron sector. Every "
              "pool operator conserves particle number exactly, so this is not "
              "an ansatz limitation -- the pool, the reference determinant or "
              "the cache is wrong. Run `python run.py selftest`.", flush=True)
    elif status == "spin_contaminated":
        print("SPIN CONTAMINATED: particle number is exact but <S^2> is not "
              "zero. General spin-complemented doubles are not S^2 "
              "eigenoperators, so the singlet is a property of the variational "
              "minimum rather than of the ansatz, and this run did not reach "
              "it. Lower --gradient-tolerance, raise --max-operators, or use "
              "--pool pair, whose seniority-zero doubles do commute with S^2.",
              flush=True)
    else:
        print("OUTSIDE CHEMICAL ACCURACY: lower --gradient-tolerance, raise "
              "--max-operators, or use the larger --pool uccgsd.", flush=True)
    return 0
