#!/usr/bin/env python3
"""The quantum half of HI-VQE: a configuration *proposer*, not an energy source.

HI-VQE asks the quantum device one question -- "which electron configurations
matter?" -- and never asks it for an expectation value. That is the whole point
of the method, and it changes what the circuit has to be good at. It does not
need to approximate the ground state to chemical accuracy; it needs to put
probability mass on the determinants that carry the correlation energy, because
the amplitudes are then set exactly by a classical diagonalisation.

The ansatz here is a **hardware-efficient unitary cluster Jastrow**: alternating
layers of

* nearest-neighbour Givens rotations (``XXPlusYY``) inside each spin block --
  these are orbital-rotation-like and move electrons between spatial orbitals of
  the same spin; and
* a diagonal number-number "Jastrow" layer (``RZ`` + ``RZZ``) -- these are the
  gates that correlate the two spin blocks with each other.

Every gate in it commutes with the alpha and the beta number operators
separately, so **every sampled bitstring is a valid configuration by
construction** on a noiseless simulator: no electron-count filtering, no wasted
shots. That is not a convenience -- it is what makes the noiseless leakage
exactly zero rather than merely small, and it is checked in `selftest.py` from
the samples rather than asserted from this docstring.

Qubit layout, matching `determinants.py` exactly: qubit ``p`` is spatial orbital
``p`` with alpha spin, qubit ``M + p`` is spatial orbital ``p`` with beta spin.

Only this module imports Qiskit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


DEFAULT_REPS = 1
DEFAULT_SHOTS = 8192


@dataclass
class SamplerSettings:
    """Everything that decides what comes back from the quantum layer."""

    shots: int = DEFAULT_SHOTS
    seed: int = 1234
    method: str = "statevector"
    max_parallel_threads: int = 0
    readout_error: float = 0.0
    depolarizing_error: float = 0.0
    recover: bool = True

    @property
    def noisy(self) -> bool:
        return self.readout_error > 0.0 or self.depolarizing_error > 0.0


@dataclass
class SampleBatch:
    """One batch of measurements, already split into alpha/beta strings."""

    strings_a: np.ndarray
    strings_b: np.ndarray
    counts: np.ndarray
    n_shots: int
    n_valid: int
    n_recovered: int
    n_discarded: int

    @property
    def leakage(self) -> float:
        """Fraction of shots that left the target particle-number sector."""
        if self.n_shots == 0:
            return 0.0
        return (self.n_shots - self.n_valid) / self.n_shots

    def determinants(self) -> list[tuple[int, int]]:
        return [
            (int(a), int(b)) for a, b in zip(self.strings_a, self.strings_b)
        ]


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------


def _adjacent_pairs(offset: int, count: int, base: int) -> list[tuple[int, int]]:
    return [
        (base + index, base + index + 1)
        for index in range(offset, count - 1, 2)
    ]


def givens_gate_count(n_orbitals: int, depth: int) -> int:
    """Gates in one brickwork network of `depth` alternating sublayers."""
    return sum(
        len(_adjacent_pairs(layer % 2, n_orbitals, 0)) for layer in range(depth)
    )


def default_givens_depth(n_orbitals: int) -> int:
    """A brickwork this deep spans the whole orbital block.

    Depth M of nearest-neighbour Givens rotations is exactly what the Clements
    decomposition needs to realise an arbitrary M-mode orbital rotation, so an
    electron in orbital 0 can reach orbital M-1 within a single repetition. A
    shallower network cannot: with the nearest-neighbour brickwork a depth-d
    layer moves an electron at most d orbitals, which at d = 2 leaves most of a
    twelve-orbital active space unreachable and the proposal distribution
    collapsed onto Hartree-Fock and its near neighbours.
    """
    return n_orbitals


def parameter_count(
    n_orbitals: int,
    reps: int,
    cross_spin: bool = True,
    givens_depth: int | None = None,
) -> int:
    """How many free angles the ansatz has, without building it."""
    depth = default_givens_depth(n_orbitals) if givens_depth is None else givens_depth
    givens_per_network = 2 * givens_gate_count(n_orbitals, depth)
    jastrow_per_rep = 2 * n_orbitals + 2 * (n_orbitals - 1)
    if cross_spin:
        jastrow_per_rep += n_orbitals
    # reps x [Givens, Jastrow], then one closing Givens network.
    return reps * (givens_per_network + jastrow_per_rep) + givens_per_network


def build_ansatz(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    reps: int = DEFAULT_REPS,
    cross_spin: bool = True,
    givens_depth: int | None = None,
) -> tuple[Any, Any]:
    """The Hartree-Fock state followed by `reps` cluster-Jastrow layers.

    Returns ``(circuit, parameters)``. At all-zero angles the circuit is the
    identity on the Hartree-Fock determinant -- ``XXPlusYY(0)``, ``RZ(0)`` and
    ``RZZ(0)`` are each the identity -- so the sampler provably starts at the
    reference configuration and the first HI-VQE subspace provably contains it.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import XXPlusYYGate

    depth = default_givens_depth(n_orbitals) if givens_depth is None else givens_depth
    n_qubits = 2 * n_orbitals
    total = parameter_count(n_orbitals, reps, cross_spin, depth)
    parameters = ParameterVector("t", total)
    circuit = QuantumCircuit(n_qubits, name="he-ucj")

    for orbital in range(n_alpha):
        circuit.x(orbital)
    for orbital in range(n_beta):
        circuit.x(n_orbitals + orbital)
    circuit.barrier()

    cursor = 0

    def givens_network(cursor: int) -> int:
        """Givens / hopping brickwork inside each spin block.

        Depth M spans the block, so one network can move an electron from any
        orbital to any other -- which is what a nearest-neighbour brickwork
        needs in order to propose configurations across the whole active space
        rather than only near the Fermi edge.
        """
        for layer in range(depth):
            for block in (0, n_orbitals):
                for left, right in _adjacent_pairs(layer % 2, n_orbitals, block):
                    circuit.append(XXPlusYYGate(parameters[cursor]), [left, right])
                    cursor += 1
        return cursor

    def jastrow_layer(cursor: int) -> int:
        """Diagonal number-number layer: single-qubit phases plus ZZ terms."""
        for qubit in range(n_qubits):
            circuit.rz(parameters[cursor], qubit)
            cursor += 1
        for block in (0, n_orbitals):
            for left, right in _adjacent_pairs(0, n_orbitals, block) + _adjacent_pairs(
                1, n_orbitals, block
            ):
                circuit.rzz(parameters[cursor], left, right)
                cursor += 1
        if cross_spin:
            # The only gates that correlate the two spin blocks with each other.
            for orbital in range(n_orbitals):
                circuit.rzz(parameters[cursor], orbital, n_orbitals + orbital)
                cursor += 1
        return cursor

    for _ in range(reps):
        cursor = givens_network(cursor)
        circuit.barrier()
        cursor = jastrow_layer(cursor)
        circuit.barrier()
    # The circuit closes on a Givens network, never on a Jastrow layer. A
    # diagonal layer changes only phases, so it leaves every measurement
    # probability untouched -- a trailing Jastrow would cost gates and shots
    # while being exactly invisible to the thing this circuit exists to do.
    cursor = givens_network(cursor)

    if cursor != total:  # pragma: no cover - guards the parameter_count formula
        raise RuntimeError(
            f"parameter_count said {total}, the circuit used {cursor}"
        )
    return circuit, parameters


def initial_parameters(
    n_parameters: int, scale: float = 0.05, seed: int = 1234
) -> np.ndarray:
    """A small random kick away from the exact Hartree-Fock fixed point.

    Exactly zero is the right *reference* -- it is provably the HF determinant --
    but it is a poor starting point: the sampler would return only the reference
    configuration, so the first iteration would have nothing for the classical
    expansion to work with beyond its own single-reference guess. A N(0, 0.05)
    kick spreads a few percent of the probability over the singly and doubly
    excited configurations while leaving HF dominant.
    """
    if scale == 0.0:
        return np.zeros(n_parameters, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=n_parameters)


def circuit_metrics(circuit: Any, linear_coupling: bool = True) -> dict[str, Any]:
    """Depth and two-qubit gate count, raw and transpiled to a device basis.

    The transpiled numbers are the ones comparable with a hardware paper: a
    linear coupling map is the pessimistic case for the cross-spin Jastrow
    terms, which connect qubit p to qubit M+p and therefore need routing.
    """
    from qiskit import transpile

    raw = {
        "qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "two_qubit_gates": int(
            sum(
                1
                for instruction in circuit.data
                if len(instruction.qubits) == 2
                and instruction.operation.name != "barrier"
            )
        ),
        "parameters": int(circuit.num_parameters),
    }
    try:
        coupling = None
        if linear_coupling:
            coupling = [
                [index, index + 1] for index in range(circuit.num_qubits - 1)
            ]
            coupling += [[b, a] for a, b in coupling]
        transpiled = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cz"],
            coupling_map=coupling,
            optimization_level=2,
            seed_transpiler=11,
        )
        raw["transpiled_depth"] = int(transpiled.depth())
        raw["transpiled_two_qubit_gates"] = int(
            transpiled.count_ops().get("cz", 0)
        )
        raw["transpiled_basis"] = "rz, sx, x, cz on a linear chain"
    except Exception as error:  # pragma: no cover - transpiler availability
        raw["transpiled_error"] = str(error)
    return raw


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _build_noise_model(settings: SamplerSettings, n_qubits: int) -> Any:
    if not settings.noisy:
        return None
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
    )

    model = NoiseModel()
    if settings.depolarizing_error > 0.0:
        two = depolarizing_error(settings.depolarizing_error, 2)
        one = depolarizing_error(settings.depolarizing_error / 10.0, 1)
        model.add_all_qubit_quantum_error(two, ["xx_plus_yy", "rzz", "cx", "cz"])
        model.add_all_qubit_quantum_error(one, ["rz", "sx", "x", "u"])
    if settings.readout_error > 0.0:
        probability = settings.readout_error
        error = ReadoutError(
            [[1 - probability, probability], [probability, 1 - probability]]
        )
        model.add_all_qubit_readout_error(error)
    return model


class ConfigurationSampler:
    """Binds angles to the ansatz, runs Aer, and returns configurations.

    The simulator and the transpiled circuit are built once and reused across
    every HI-VQE iteration and every optimizer evaluation, because transpiling a
    24-qubit circuit repeatedly costs more than simulating it.
    """

    def __init__(
        self,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
        reps: int = DEFAULT_REPS,
        settings: SamplerSettings | None = None,
        cross_spin: bool = True,
        givens_depth: int | None = None,
    ) -> None:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.settings = settings or SamplerSettings()
        self.circuit, self.parameters = build_ansatz(
            n_orbitals, n_alpha, n_beta, reps, cross_spin, givens_depth
        )
        self.metrics = circuit_metrics(self.circuit)

        measured = self.circuit.copy()
        measured.measure_all()
        options: dict[str, Any] = {"method": self.settings.method}
        if self.settings.max_parallel_threads:
            options["max_parallel_threads"] = self.settings.max_parallel_threads
        noise_model = _build_noise_model(self.settings, self.circuit.num_qubits)
        if noise_model is not None:
            options["noise_model"] = noise_model
        self.simulator = AerSimulator(**options)
        self.transpiled = transpile(measured, self.simulator, optimization_level=1)
        self._rng = np.random.default_rng(self.settings.seed)
        self.n_circuit_runs = 0

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def sample(
        self,
        angles: Sequence[float],
        shots: int | None = None,
        occupancy_prior: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> SampleBatch:
        """Run the circuit and return the configurations it proposed."""
        shots = int(shots or self.settings.shots)
        bound = self.transpiled.assign_parameters(
            {p: float(v) for p, v in zip(self.parameters, angles)}
        )
        seed = int(self._rng.integers(0, 2**31 - 1))
        job = self.simulator.run(bound, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()
        self.n_circuit_runs += 1
        return self._to_configurations(counts, shots, occupancy_prior)

    def _to_configurations(
        self,
        counts: dict[str, int],
        shots: int,
        occupancy_prior: tuple[np.ndarray, np.ndarray] | None,
    ) -> SampleBatch:
        mask = (1 << self.n_orbitals) - 1
        accumulated: dict[tuple[int, int], int] = {}
        n_valid = 0
        n_recovered = 0
        n_discarded = 0

        for bits, count in counts.items():
            # Qiskit prints qubit 0 as the rightmost character, so int(bits, 2)
            # already has bit q equal to the measurement on qubit q.
            value = int(bits.replace(" ", ""), 2)
            string_a = value & mask
            string_b = (value >> self.n_orbitals) & mask
            good = (
                string_a.bit_count() == self.n_alpha
                and string_b.bit_count() == self.n_beta
            )
            if good:
                n_valid += count
            elif occupancy_prior is not None and self.settings.recover:
                string_a = _recover_string(
                    string_a, self.n_alpha, self.n_orbitals, occupancy_prior[0]
                )
                string_b = _recover_string(
                    string_b, self.n_beta, self.n_orbitals, occupancy_prior[1]
                )
                n_recovered += count
            else:
                n_discarded += count
                continue
            key = (string_a, string_b)
            accumulated[key] = accumulated.get(key, 0) + count

        if not accumulated:  # pragma: no cover - only with pathological noise
            hf_a = (1 << self.n_alpha) - 1
            hf_b = (1 << self.n_beta) - 1
            accumulated[(hf_a, hf_b)] = 0

        keys = sorted(accumulated)
        return SampleBatch(
            strings_a=np.array([k[0] for k in keys], dtype=np.int64),
            strings_b=np.array([k[1] for k in keys], dtype=np.int64),
            counts=np.array([accumulated[k] for k in keys], dtype=np.int64),
            n_shots=shots,
            n_valid=n_valid,
            n_recovered=n_recovered,
            n_discarded=n_discarded,
        )


def _recover_string(
    string: int, n_electrons: int, n_orbitals: int, prior: np.ndarray
) -> int:
    """Restore the electron count of a corrupted string, guided by occupancies.

    The SQD-style configuration recovery, in its simplest honest form: when the
    measured string has too many electrons, empty the occupied orbitals the
    reference wavefunction says are least likely to be occupied; when it has too
    few, fill the empty orbitals it says are most likely. `prior` is the
    orbital-occupancy vector of the previous iteration's classical ground state,
    so the correction uses information the quantum device did not have to
    provide.
    """
    occupied = [p for p in range(n_orbitals) if (string >> p) & 1]
    empty = [p for p in range(n_orbitals) if not (string >> p) & 1]
    excess = len(occupied) - n_electrons
    if excess > 0:
        order = sorted(occupied, key=lambda p: float(prior[p]))
        for orbital in order[:excess]:
            string ^= 1 << orbital
    elif excess < 0:
        order = sorted(empty, key=lambda p: -float(prior[p]))
        for orbital in order[: -excess]:
            string |= 1 << orbital
    return string
