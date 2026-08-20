#!/usr/bin/env python3
"""A NumPy statevector engine with exact adjoint gradients.

This is an *alternative* to the Aer engine, not a replacement. Both compute the
same quantity; select with `--engine`. Aer stays the reference, and
`selftest.py` fails the build if the two disagree.

**Why it exists.** The VQE spends nearly all its time computing gradients. With
finite differences a gradient costs 2N+1 full simulations -- 241 of them at 16
qubits with 120 parameters -- and each one re-applies the whole Hamiltonian.
Adjoint (reverse-mode) differentiation gets every component from a single
forward pass and a single backward pass, so the gradient costs about 4 passes
regardless of the parameter count.

Aer cannot do this: it is a closed box that takes a circuit and returns a
number, whereas the backward sweep has to reach inside and undo gates one at a
time. qiskit-algorithms' `ReverseEstimatorGradient` does expose it, but took
619 s for a single 26-parameter gradient here. Hence a small engine of our own.

**The maths.** The ansatz is a Hartree-Fock determinant followed by
U_j = exp(-i t_j P_j) for Hermitian generators P_j, so with psi = U_m...U_1|HF>
and lambda = H psi,

    dE/dt_j = 2 Re < U_j+1^dag ... U_m^dag H psi | (-i P_j) U_j-1...U_1|HF> >

which the loop below evaluates by walking psi and lambda backwards together,
applying each U_j^dag once. Three vector passes total instead of 2N+1.

**Why it is fast in practice.** Applying a Pauli string to a statevector is an
index permutation plus a sign flip -- no matrices are built or stored -- and the
Pauli strings inside a single fermionic excitation generator all commute, so
exp(-i t P) factorises exactly into a product of cos/sin rotations. Nothing here
is an approximation; the commuting property is asserted, not assumed.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

# How much memory the observable's precomputed diagonals may occupy. One 2**n
# complex vector per distinct X mask is a big win at 12-14 qubits and a
# multi-gigabyte mistake at 18+, so it is budgeted rather than unconditional.
DIAGONAL_CACHE_BYTES = 512 * 1024 * 1024
# Sign patterns are cached separately and far more cheaply: one *byte* per
# amplitude instead of sixteen, keyed by Z mask and shared across every group
# and operator. Recomputing them was what actually dominated the 16-qubit
# energy evaluation once the gathers had been grouped away.
PARITY_CACHE_BYTES = 512 * 1024 * 1024


def _popcount_parity(values: np.ndarray, mask: int) -> np.ndarray:
    """(-1)**popcount(values & mask) for every index, vectorised.

    Recomputed on demand rather than cached: caching one 2**n array per Pauli
    string is a memory bomb -- at 20 qubits a k-UpCCGSD ansatz has ~1500 strings,
    which would be tens of gigabytes of sign vectors.
    """
    if mask == 0:
        return np.ones(values.shape, dtype=np.int8)
    masked = values & mask
    try:
        counts = np.bitwise_count(masked)          # NumPy >= 2.0, single pass
    except AttributeError:                          # pragma: no cover
        counts = masked
        shift = 1
        while shift < 64:
            counts = counts ^ (counts >> shift)
            shift <<= 1
    return np.where((counts & 1).astype(bool), np.int8(-1), np.int8(1))


class PauliTerm:
    """One Pauli string, stored as the masks needed to apply it in place.

    A Qiskit Pauli is (-i)^phase * Z^z X^x, and acting on a statevector

        (P psi)[a] = (-i)^phase * (-1)^(z . a) * psi[a XOR x]

    so applying it is a gather plus a sign vector -- no matrix.
    """

    __slots__ = ("x_mask", "z_mask", "phase", "coefficient")

    def __init__(self, x_mask: int, z_mask: int, phase: complex, coefficient: complex) -> None:
        self.x_mask = x_mask
        self.z_mask = z_mask
        self.phase = phase
        self.coefficient = coefficient

    def apply(self, vector: np.ndarray, indices: np.ndarray, scale: complex = 1.0) -> np.ndarray:
        gathered = vector if self.x_mask == 0 else vector[indices ^ self.x_mask]
        weight = scale * self.coefficient * self.phase
        if self.z_mask == 0:
            return weight * gathered
        return (weight * _popcount_parity(indices, self.z_mask)) * gathered


def decompose_operator(operator: Any, n_qubits: int) -> list[PauliTerm]:
    """Turn a SparsePauliOp into applicable Pauli terms."""
    terms: list[PauliTerm] = []
    for pauli, coefficient in zip(operator.paulis, operator.coeffs):
        x_bits = np.asarray(pauli.x, dtype=bool)
        z_bits = np.asarray(pauli.z, dtype=bool)
        x_mask = int(sum(1 << i for i in range(n_qubits) if x_bits[i]))
        z_mask = int(sum(1 << i for i in range(n_qubits) if z_bits[i]))
        # Qiskit's `Pauli.phase` is the *group* phase, not the internal exponent.
        # The operator is (-i)^(phase + count_Y) Z^z X^x, where count_Y is the
        # number of qubits carrying both x and z. Dropping the count_Y term
        # silently rotates every Y-containing string -- which is most of them for
        # a fermionic excitation -- and was the original bug here.
        count_y = int(np.count_nonzero(x_bits & z_bits))
        phase = (-1j) ** ((int(pauli.phase) + count_y) % 4)
        terms.append(PauliTerm(x_mask, z_mask, phase, complex(coefficient)))
    return terms


class Generator:
    """exp(-i t P) for a Hermitian P whose Pauli strings mutually commute.

    Commuting strings mean the exponential factorises exactly:
        exp(-i t sum_k c_k S_k) = prod_k [cos(t c_k) I - i sin(t c_k) S_k]
    which is applied string by string with no matrix and no Trotter error.
    """

    def __init__(self, operator: Any, n_qubits: int) -> None:
        if not _strings_commute(operator):
            raise ValueError(
                "Generator contains non-commuting Pauli strings; the exact "
                "factorisation used here does not apply."
            )
        self.terms = decompose_operator(operator, n_qubits)
        self.operator = operator

    def apply(self, vector: np.ndarray, angle: float, indices: np.ndarray,
              conjugate: bool = False) -> np.ndarray:
        sign = -1.0 if conjugate else 1.0
        out = vector
        for term in self.terms:
            weight = complex(term.coefficient * term.phase)
            # The string acts as (weight * S) with S^2 = I, so the rotation
            # angle for this string is |weight| * t and its axis is S/|weight|.
            magnitude = abs(weight)
            if magnitude < 1e-15:
                continue
            theta = sign * angle * magnitude
            unit = PauliTerm(term.x_mask, term.z_mask, term.phase,
                             term.coefficient / magnitude)
            out = np.cos(theta) * out - 1j * np.sin(theta) * unit.apply(out, indices)
        return out

    def apply_derivative(self, vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """(-i P) applied to a vector."""
        out = np.zeros_like(vector)
        for term in self.terms:
            out += term.apply(vector, indices)
        return -1j * out


def generators_from_circuit(circuit: Any) -> list[Any]:
    """Read the evolution generators back off a circuit, in application order."""
    from qiskit.circuit.library import PauliEvolutionGate

    found: list[Any] = []
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, PauliEvolutionGate):
            found.append(operation.operator)
    return found


def _strings_commute(operator: Any) -> bool:
    paulis = operator.paulis
    for i in range(len(paulis)):
        for j in range(i + 1, len(paulis)):
            if not paulis[i].commutes(paulis[j]):
                return False
    return True


class NumpyEvaluator:
    """Same interface as `AerEvaluator`, backed by NumPy with adjoint gradients."""

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        ansatz_kind: str = "uccsd",
        layers: int = 4,
        k: int = 2,
        circuit: Any | None = None,
        seed: int = 7,
        **_ignored: Any,
    ) -> None:
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import build_ansatz

        if ansatz_kind == "hea":
            raise ValueError(
                "The NumPy engine supports the fermionic ansatze (uccsd, "
                "kupccgsd, adapt) whose generators are exponentials of Pauli "
                "sums. Use --engine aer for the hardware-efficient ansatz."
            )
        self.n_qubits = n_qubits
        self.ansatz_kind = ansatz_kind
        self.method = "numpy-adjoint"
        self.dimension = 1 << n_qubits
        self.indices = np.arange(self.dimension, dtype=np.int64)

        ansatz = circuit if circuit is not None else build_ansatz(
            n_qubits, n_electrons, ansatz_kind, layers, k
        )
        operators = getattr(ansatz, "operators", None)
        if operators is None:
            # ADAPT hands over a plain circuit it assembled itself, so the
            # generators are read back off its evolution gates.
            operators = generators_from_circuit(ansatz)
        if not operators and ansatz.num_parameters:
            raise ValueError(f"Ansatz {ansatz_kind!r} exposes no generator list")
        # k-UpCCGSD is the same operator list repeated `reps` times with
        # independent parameters, so the generator sequence is the list tiled
        # out -- `operators` alone would silently give k times too few.
        reps = int(getattr(ansatz, "reps", 1) or 1)
        built = [Generator(op, n_qubits) for op in operators]
        self.generators = [built[i % len(built)] for i in range(len(built) * reps)]
        self.n_parameters = len(self.generators)
        if ansatz.num_parameters != self.n_parameters:
            raise ValueError(
                f"Generator count {self.n_parameters} does not match the circuit's "
                f"{ansatz.num_parameters} parameters for {ansatz_kind!r}"
            )

        # The reference determinant, as a single computational basis state. For
        # a named ansatz it is the declared initial state; for a bare circuit,
        # every generator is the identity at t = 0, so binding zeros exposes it.
        initial = getattr(ansatz, "initial_state", None)
        if initial is not None:
            amplitudes = np.asarray(Statevector(initial).data)
        elif ansatz.num_parameters:
            amplitudes = np.asarray(
                Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters))).data
            )
        else:
            amplitudes = np.asarray(Statevector(ansatz).data)
        reference_index = int(np.argmax(np.abs(amplitudes)))
        if not np.isclose(abs(amplitudes[reference_index]), 1.0, atol=1e-9):
            raise ValueError("Reference state is not a single determinant")
        self.reference_index = reference_index

        self.depth = sum(len(g.terms) for g in self.generators)
        self.two_qubit_gates = 0
        self.exact_gradient = False   # parameter-shift is not used by this engine
        self.circuit_count = 0
        self.batch_count = 0
        self._observables: dict[int, list[tuple[int, Any]]] = {}
        self._parity: dict[int, np.ndarray] = {}
        self._parity_budget = PARITY_CACHE_BYTES

    def _parity_for(self, z_mask: int) -> np.ndarray:
        """(-1)**popcount(index & z_mask), cached as int8 while the budget lasts."""
        cached = self._parity.get(z_mask)
        if cached is not None:
            return cached
        signs = _popcount_parity(self.indices, z_mask)
        if self._parity_budget >= self.dimension:
            self._parity[z_mask] = signs
            self._parity_budget -= self.dimension
        return signs

    # -- observables ------------------------------------------------------
    def _groups(self, observable: Any) -> list[tuple[int, np.ndarray]]:
        """Group the observable's Pauli strings by their X mask.

        Every string with the same X mask permutes the amplitudes identically,
        so the whole group needs a single gather -- the expensive part -- and
        differs only in a diagonal sign pattern that can be summed first. A
        molecular Hamiltonian is roughly half diagonal (X mask zero), so this
        collapses thousands of gathers into a few hundred.

        The diagonals are cached only while they fit in `DIAGONAL_CACHE_BYTES`;
        one 2**n vector per group would otherwise become gigabytes at 16+ qubits.
        """
        key = id(observable)
        cached = self._observables.get(key)
        if cached is not None:
            return cached

        buckets: dict[int, list[tuple[complex, int]]] = {}
        for term in decompose_operator(observable, self.n_qubits):
            weight = complex(term.coefficient * term.phase)
            buckets.setdefault(term.x_mask, []).append((weight, term.z_mask))

        groups: list[tuple[int, np.ndarray]] = []
        budget = DIAGONAL_CACHE_BYTES
        per_group = self.dimension * 16
        for x_mask, entries in buckets.items():
            if budget >= per_group:
                groups.append((x_mask, self._diagonal(entries)))
                budget -= per_group
            else:
                groups.append((x_mask, entries))  # type: ignore[arg-type]
        self._observables[key] = groups
        return groups

    def _diagonal(self, entries: Sequence[tuple[complex, int]]) -> np.ndarray:
        diagonal = np.zeros(self.dimension, dtype=np.complex128)
        for weight, z_mask in entries:
            if z_mask == 0:
                diagonal += weight
            else:
                diagonal += weight * self._parity_for(z_mask)
        return diagonal

    def _apply_observable(self, observable: Any, vector: np.ndarray) -> np.ndarray:
        out = np.zeros_like(vector)
        for x_mask, payload in self._groups(observable):
            diagonal = payload if isinstance(payload, np.ndarray) else self._diagonal(payload)
            gathered = vector if x_mask == 0 else vector[self.indices ^ x_mask]
            out += diagonal * gathered
        return out

    # -- state ------------------------------------------------------------
    def statevector(self, values: np.ndarray) -> np.ndarray:
        angles = np.asarray(values, dtype=float).reshape(-1)
        if angles.size != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, received {angles.size}"
            )
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[self.reference_index] = 1.0
        for generator, angle in zip(self.generators, angles):
            state = generator.apply(state, float(angle), self.indices)
        self.circuit_count += 1
        self.batch_count += 1
        return state

    # -- energies ---------------------------------------------------------
    def energy(self, values: np.ndarray, observable: Any) -> float:
        state = self.statevector(values)
        return float(np.real(np.vdot(state, self._apply_observable(observable, state))))

    def run(self, jobs: Sequence[tuple[np.ndarray, Any]]) -> list[float]:
        """Evaluate many (theta, observable) pairs, reusing the state per theta."""
        if not jobs:
            return []
        energies = [0.0] * len(jobs)
        cache: dict[bytes, np.ndarray] = {}
        for index, (values, observable) in enumerate(jobs):
            angles = np.asarray(values, dtype=float).reshape(-1)
            key = angles.tobytes()
            state = cache.get(key)
            if state is None:
                state = self.statevector(angles)
                cache[key] = state
            energies[index] = float(
                np.real(np.vdot(state, self._apply_observable(observable, state)))
            )
        return energies

    def energy_and_gradient(
        self, values: np.ndarray, observable: Any, method: str = "adjoint"
    ) -> tuple[float, np.ndarray]:
        """Exact energy and gradient from one forward and one backward sweep."""
        if method == "parameter-shift":
            raise ValueError("The NumPy engine uses adjoint gradients, not parameter-shift")
        angles = np.asarray(values, dtype=float).reshape(-1)
        state = self.statevector(angles)
        h_state = self._apply_observable(observable, state)
        energy = float(np.real(np.vdot(state, h_state)))

        # With psi_j = U_j...U_0|ref> and lambda_j = U_j+1^dag...U_m-1^dag H psi,
        #     dE/dt_j = 2 Re < lambda_j | (-i P_j) psi_j >
        # so each component is read off *before* U_j is undone on either vector.
        # Undoing phi first and pairing it with a not-yet-undone lambda is an
        # off-by-one that leaves the energy correct and the gradient wrong.
        gradient = np.zeros(self.n_parameters, dtype=float)
        phi = state
        lam = h_state
        for j in range(self.n_parameters - 1, -1, -1):
            generator = self.generators[j]
            angle = float(angles[j])
            derivative = generator.apply_derivative(phi, self.indices)
            gradient[j] = 2.0 * float(np.real(np.vdot(lam, derivative)))
            phi = generator.apply(phi, angle, self.indices, conjugate=True)
            lam = generator.apply(lam, angle, self.indices, conjugate=True)
        return energy, gradient


def compare_engines(
    n_qubits: int,
    n_electrons: int,
    observable: Any,
    ansatz_kind: str = "uccsd",
    k: int = 2,
    seed: int = 7,
    n_points: int = 3,
) -> dict[str, Any]:
    """Cross-check the NumPy engine against Aer on energies and gradients."""
    from qiskit_runtime import AerEvaluator

    aer = AerEvaluator(n_qubits, n_electrons, ansatz_kind, method="statevector",
                       seed=seed, k=k)
    fast = NumpyEvaluator(n_qubits, n_electrons, ansatz_kind, k=k, seed=seed)
    if aer.n_parameters != fast.n_parameters:
        raise RuntimeError(
            f"Parameter counts differ: Aer {aer.n_parameters}, NumPy {fast.n_parameters}"
        )

    rng = np.random.default_rng(seed)
    energy_error = gradient_error = 0.0
    aer_seconds = fast_seconds = 0.0
    for point in range(n_points):
        angles = np.zeros(fast.n_parameters) if point == 0 else rng.normal(
            0.0, 0.25, fast.n_parameters
        )
        started = time.perf_counter()
        aer_energy, aer_gradient = aer.energy_and_gradient(
            angles, observable, "finite-difference"
        )
        aer_seconds += time.perf_counter() - started

        started = time.perf_counter()
        fast_energy, fast_gradient = fast.energy_and_gradient(angles, observable)
        fast_seconds += time.perf_counter() - started

        energy_error = max(energy_error, abs(aer_energy - fast_energy))
        gradient_error = max(gradient_error, float(np.max(np.abs(aer_gradient - fast_gradient))))

    return {
        "n_qubits": n_qubits,
        "ansatz": ansatz_kind,
        "parameters": fast.n_parameters,
        "max_energy_difference": energy_error,
        "max_gradient_difference": gradient_error,
        "aer_seconds": aer_seconds,
        "numpy_seconds": fast_seconds,
        "speedup": aer_seconds / fast_seconds if fast_seconds > 0 else float("inf"),
    }
