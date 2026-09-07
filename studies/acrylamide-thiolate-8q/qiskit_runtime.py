#!/usr/bin/env python3
"""Qiskit Aer execution layer: operator conversion, UCCSD, and VQE.

Follows the CovAngelo simulator protocol (arXiv:2604.10487): **UCCSD** on a
Hartree-Fock reference, benchmarked against exact CASCI/FCI for the identical
Hamiltonian. The paper tested a hardware-efficient ansatz, a symmetry-preserving
ansatz and ADAPT-VQE, and reported UCCSD as the most reliable; `--ansatz hea`
keeps the old hardware-efficient path available for comparison.

Three engineering decisions here are measurements, not preferences:

* **Gradients are central finite differences, batched.** UCCSD reuses each
  parameter across ~8 gates, so the two-term parameter-shift rule is simply
  invalid for it -- `parameter_shift_is_exact()` detects this and refuses.
  qiskit-algorithms' adjoint `ReverseEstimatorGradient` is numerically correct
  (agrees with finite differences to 1.7e-11) but took **619 s** for a single
  26-component gradient, against ~2.8 s for batched finite differences. It is
  therefore not used.
* **Parallelism is method-dependent, and the naive answer is wrong.** Measured
  on this machine for the UCCSD circuit (8 qubits, depth 1646, 361 Pauli terms):

      statevector   threads=1 exp=8   70.8 ms/circuit
      statevector   threads=8 exp=1   36.0 ms/circuit   <- best
      MPS           threads=1 exp=8   83.9 ms/circuit   <- best for MPS
      MPS           threads=8 exp=1  153.9 ms/circuit

  A *shallow* circuit is too small to thread, so circuit-level parallelism wins;
  a *deep* one is substantial work that threads well. UCCSD is deep, so the
  defaults are set per method rather than assumed.
* **The default method is statevector, not MPS.** Also measured: MPS is 2.3x
  slower here. UCCSD's 1376 entangling gates saturate the bond dimension
  (chi <= 2^(n/2) = 16 at 8 qubits), so MPS compresses nothing and pays pure
  overhead. MPS remains available -- it is the method that scales to weakly
  entangled circuits at larger n -- and the converged energy is always
  cross-checked against exact statevector either way.

Qubit ordering is the subtle part. OpenFermion emits spin orbitals
**interleaved** (2p = alpha_p, 2p+1 = beta_p); qiskit-nature's JordanWignerMapper
expects them **blocked** (0..n-1 alpha, n..2n-1 beta). Feeding one to the other
produces a Hamiltonian with an identical spectrum and completely wrong energies,
so the permutation is applied explicitly and checked determinant by determinant
in `selftest.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from chemistry import (
    CHEMICAL_ACCURACY_MHA,
    COMPRESSION_TOLERANCE,
    CONTAMINATION_ENERGY_BUDGET_HA,
    CONTAMINATION_PRICE_HA,
    ChemistryResult,
    MoleculeSpec,
    add_electron_number_penalty,
    electron_number_operator,
    load_chemistry,
    number_deviation_operator,
    require_validation_receipt,
    spin_squared_qubit_operator,
    to_hartree_fock_frame,
    workflow_fingerprint,
    write_json_atomic,
)

DEFAULT_TRUNCATION_THRESHOLD = 1.0e-16
MPS_AGREEMENT_BUDGET_HA = 0.1 * CHEMICAL_ACCURACY_MHA / 1000.0
SIMULATION_METHODS = ("statevector", "matrix_product_state")
# Measured optimum per method: (max_parallel_threads, max_parallel_experiments)
# as a function of the CPU count. See the module docstring for the numbers.
PARALLELISM = {
    "statevector": lambda cpus: (cpus, 1),
    "matrix_product_state": lambda cpus: (1, cpus),
}
OPTIMIZERS = ("l-bfgs-b", "slsqp", "cobyla")
ANSATZE = ("auto", "uccsd", "kupccgsd", "adapt", "hea")
FINITE_DIFFERENCE_STEP = 1.0e-5
# `aer` simulates the circuit and differentiates by finite differences; `numpy`
# propagates the statevector directly and differentiates by the adjoint method,
# which costs ~4 passes instead of 2N+1 simulations. They agree exactly -- see
# `fastsim.compare_engines` and the cross-engine check in selftest.py.
ENGINES = ("aer", "numpy")
DEFAULT_KUPCCGSD_K = 2

# Above this register size full UCCSD stops being affordable on a laptop.
# Measured for a 50-iteration VQE on 8 cores: 8q 0.6 h, 12q 1.8 h, 14q 13.5 h.
# The gradient costs (2N+1) circuits of depth D, so it scales as N x D, and both
# grow fast with the active space:
#
#   12q (6e,6o)   UCCSD          117 params   depth 11,127
#               k-UpCCGSD k=1     33 params   depth  1,622
#               k-UpCCGSD k=2     66 params   depth  3,226
#
# so k = 2 is roughly 6x cheaper than UCCSD there, and k = 1 about 24x.
AUTO_UCCSD_MAX_QUBITS = 10


def resolve_ansatz(kind: str, n_qubits: int) -> str:
    """Pick an ansatz when the caller asked for `auto`."""
    if kind != "auto":
        return kind
    return "uccsd" if n_qubits <= AUTO_UCCSD_MAX_QUBITS else "kupccgsd"


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


# Aer recurses in proportion to circuit *depth*, and a Windows thread gets a
# 1 MB stack by default (Linux gives 8 MB). A 12-qubit UCCSD circuit is 11,127
# deep and overflows it: the process dies with 0xC00000FD and no Python
# traceback, so it cannot be caught or retried. Measured on this machine:
#
#     UCCSD  8q  depth  1,646  -> fine
#     UCCSD 10q  depth  4,286  -> fine
#     UCCSD 12q  depth 11,127  -> hard crash on statevector AND MPS
#     random 12q depth  3,139 with 12,000 gates -> fine
#
# so it tracks depth, not qubit count or gate count. Running Aer on a thread
# with a 64 MB stack fixes it (12q then completes in 0.71 s). Values much above
# 64 MB fail to allocate the thread on Windows, so this is not a knob to raise
# casually.
STACK_BYTES = 64 * 1024 * 1024


def run_with_deep_stack(work: Callable[[], Any]) -> Any:
    """Execute `work` on a thread with a stack large enough for deep circuits."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = work()
        except BaseException as error:  # noqa: BLE001 - re-raised on the caller
            outcome["error"] = error

    previous = threading.stack_size(STACK_BYTES)
    try:
        thread = threading.Thread(target=target, name="aer-deep-stack")
        thread.start()
        thread.join()
    finally:
        threading.stack_size(previous)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


# --------------------------------------------------------------------------
# Spin-orbital ordering
# --------------------------------------------------------------------------


def interleaved_to_blocked(n_qubits: int) -> list[int]:
    """Map OpenFermion's interleaved spin-orbital index to qiskit-nature's blocked one.

    OpenFermion: 2p -> alpha_p, 2p+1 -> beta_p.
    qiskit-nature: alpha_p -> p, beta_p -> n_spatial + p.

    Sanity check this preserves the reference determinant: for 4 electrons the
    interleaved occupation is {0,1,2,3} = alpha_0, beta_0, alpha_1, beta_1, which
    maps to blocked {0, 4, 1, 5} -- exactly the qubits qiskit-nature's
    HartreeFock circuit excites.
    """
    if n_qubits % 2:
        raise ValueError("a spin-orbital register must have an even number of qubits")
    n_spatial = n_qubits // 2
    mapping = [0] * n_qubits
    for p in range(n_spatial):
        mapping[2 * p] = p
        mapping[2 * p + 1] = n_spatial + p
    return mapping


def identity_mapping(n_qubits: int) -> list[int]:
    return list(range(n_qubits))


# --------------------------------------------------------------------------
# Operator conversion
# --------------------------------------------------------------------------


def to_sparse_pauli_op(
    operator: Any, n_qubits: int, qubit_map: Sequence[int] | None = None
) -> Any:
    """Convert an OpenFermion QubitOperator to a Qiskit SparsePauliOp.

    `qubit_map[i]` is the Qiskit qubit that OpenFermion qubit `i` becomes. The
    sparse-list form takes explicit indices, so the conversion cannot be
    silently reversed by Qiskit's little-endian label convention.

    A non-real coefficient is a hard error: `save_expectation_value` requires a
    Hermitian operator, and quietly dropping an imaginary part would hide a bug.
    """
    from qiskit.quantum_info import SparsePauliOp

    mapping = list(qubit_map) if qubit_map is not None else identity_mapping(n_qubits)
    if len(mapping) != n_qubits or sorted(mapping) != list(range(n_qubits)):
        raise ValueError(f"qubit_map must be a permutation of 0..{n_qubits - 1}")

    sparse_list = []
    max_imaginary = 0.0
    for term, coefficient in operator.terms.items():
        coefficient = complex(coefficient)
        max_imaginary = max(max_imaginary, abs(coefficient.imag))
        indices = []
        for qubit, _ in term:
            index = int(qubit)
            if index < 0 or index >= n_qubits:
                raise ValueError(
                    f"Pauli term {term} acts outside the {n_qubits}-qubit register"
                )
            indices.append(mapping[index])
        sparse_list.append(("".join(p for _, p in term), indices, coefficient.real))
    if max_imaginary > 1.0e-10:
        raise ValueError(
            f"Operator is not Hermitian: largest imaginary coefficient {max_imaginary:.3e}"
        )
    if not sparse_list:
        sparse_list = [("", [], 0.0)]
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits).simplify()


# --------------------------------------------------------------------------
# Ansatz
# --------------------------------------------------------------------------


def _nature_pieces(n_qubits: int, n_electrons: int):
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if n_electrons % 2:
        raise ValueError("this workflow assumes a closed-shell singlet")
    n_spatial = n_qubits // 2
    particles = (n_electrons // 2, n_electrons // 2)
    mapper = JordanWignerMapper()
    return n_spatial, particles, mapper, HartreeFock(n_spatial, particles, mapper)


def build_ansatz(
    n_qubits: int,
    n_electrons: int,
    kind: str = "uccsd",
    layers: int = 4,
    k: int = DEFAULT_KUPCCGSD_K,
) -> Any:
    """A particle-number-conserving ansatz on a Hartree-Fock reference.

    `uccsd` is the CovAngelo protocol's choice and the most accurate, but its
    parameter count and depth both grow as O(N^4) and it becomes unaffordable
    past ~12 qubits on a laptop.

    `kupccgsd` is k repetitions of generalized paired doubles plus generalized
    singles. Parameters grow as O(k*N^2) instead, and because each repetition is
    a shallow block the circuit depth falls even faster than the parameter count.
    Accuracy is controlled by k rather than being fixed.

    Both conserve particle number exactly, so neither needs a penalty term, and
    both start at the reference determinant at theta = 0. That deletes the whole
    penalty / leakage / contamination apparatus the hardware-efficient path
    requires.
    """
    if kind == "uccsd":
        from qiskit_nature.second_q.circuit.library import UCCSD

        n_spatial, particles, mapper, initial_state = _nature_pieces(n_qubits, n_electrons)
        return UCCSD(n_spatial, particles, mapper, initial_state=initial_state)
    if kind == "kupccgsd":
        from qiskit_nature.second_q.circuit.library import PUCCD

        if k < 1:
            raise ValueError("k must be positive")
        n_spatial, particles, mapper, initial_state = _nature_pieces(n_qubits, n_electrons)
        # Paired doubles + singles, generalized (excitations are not restricted
        # to occupied -> virtual), repeated k times with independent parameters.
        return PUCCD(
            n_spatial,
            particles,
            mapper,
            initial_state=initial_state,
            generalized=True,
            include_singles=(True, True),
            reps=k,
        )
    if kind == "hea":
        from qiskit.circuit.library import real_amplitudes

        if layers < 1:
            raise ValueError("layers must be positive")
        return real_amplitudes(n_qubits, reps=layers, entanglement="linear")
    raise ValueError(f"Unknown ansatz {kind!r}; choose from {ANSATZE}")


def parameter_shift_is_exact(circuit: Any) -> bool:
    """True when every parameter is a lone RY angle used exactly once.

    Holds for `real_amplitudes`. Does **not** hold for UCCSD, where one
    parameter drives roughly eight Pauli rotations, so the two-term shift rule
    silently returns the wrong gradient. Checked rather than assumed.
    """
    from qiskit.circuit import Parameter

    seen: set[Any] = set()
    for instruction in circuit.data:
        for expression in instruction.operation.params:
            parameters = getattr(expression, "parameters", ())
            if not parameters:
                continue
            if instruction.operation.name != "ry" or len(parameters) != 1:
                return False
            parameter = next(iter(parameters))
            if not isinstance(parameter, Parameter) or expression != parameter:
                return False
            if parameter in seen:
                return False
            seen.add(parameter)
    return True


# --------------------------------------------------------------------------
# Aer evaluation
# --------------------------------------------------------------------------


class AerEvaluator:
    """Batched, exact expectation values for one fixed ansatz on one method."""

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        ansatz_kind: str = "uccsd",
        layers: int = 4,
        method: str = "statevector",
        max_bond_dimension: int | None = None,
        truncation_threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
        threads: int | None = None,
        parallel_experiments: int | None = None,
        seed: int = 7,
        k: int = DEFAULT_KUPCCGSD_K,
        circuit: Any | None = None,
    ) -> None:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        if method not in SIMULATION_METHODS:
            raise ValueError(f"Unsupported method {method!r}; choose from {SIMULATION_METHODS}")
        cpus = os.cpu_count() or 1
        default_threads, default_experiments = PARALLELISM[method](cpus)
        options: dict[str, Any] = {
            "method": method,
            "precision": "double",
            "seed_simulator": seed,
            "max_parallel_threads": int(threads or default_threads),
            "max_parallel_experiments": int(parallel_experiments or default_experiments),
        }
        if method == "matrix_product_state":
            options["matrix_product_state_truncation_threshold"] = truncation_threshold
            if max_bond_dimension is not None:
                options["matrix_product_state_max_bond_dimension"] = int(max_bond_dimension)

        self.method = method
        self.n_qubits = n_qubits
        self.ansatz_kind = ansatz_kind
        self.backend = AerSimulator(**options)
        # ADAPT hands us a circuit that grows each macro iteration, so an
        # explicit circuit overrides the named-ansatz construction.
        self.ansatz = (
            circuit
            if circuit is not None
            else build_ansatz(n_qubits, n_electrons, ansatz_kind, layers, k)
        )
        self.template = transpile(
            self.ansatz.decompose(reps=5), self.backend, optimization_level=1
        )
        self.layout = self.template.layout
        self.n_parameters = int(self.template.num_parameters)
        self.depth = int(self.template.depth())
        self.two_qubit_gates = int(
            sum(count for name, count in self.template.count_ops().items() if name in ("cx", "cz", "ecr"))
        )
        self.exact_gradient = parameter_shift_is_exact(self.template)
        # Qiskit orders `circuit.parameters` canonically, and `assign_parameters`
        # with a positional array uses that same order, so binding by this list
        # is consistent with everything the optimizer hands us.
        self.parameters = list(self.template.parameters)
        self._save_templates: dict[int, Any] = {}
        self.circuit_count = 0
        self.batch_count = 0

    def _on_device(self, observable: Any) -> Any:
        if self.layout is None:
            return observable
        return observable.apply_layout(self.layout)

    def _template_with_save(self, observable: Any) -> Any:
        """One circuit per observable, carrying its save instruction, cached."""
        key = id(observable)
        cached = self._save_templates.get(key)
        if cached is None:
            cached = self.template.copy()
            cached.save_expectation_value(
                self._on_device(observable),
                qubits=list(range(self.template.num_qubits)),
                label="energy",
            )
            self._save_templates[key] = cached
        return cached

    def _run_one_observable(
        self, values_list: Sequence[np.ndarray], observable: Any
    ) -> list[float]:
        """Many parameter vectors, one observable, via Aer's `parameter_binds`.

        This is the hot path and the single biggest speed decision in the file.
        Binding parameters with `QuantumCircuit.assign_parameters` costs ~419 ms
        per UCCSD circuit, because each of the 26 parameters appears in roughly
        eight gates as a symbolic expression that has to be evaluated in Python.
        Handing Aer the unbound circuit plus a bind table moves all of that into
        C++. Measured on a 53-vector gradient batch: 42.73 s -> 1.36 s, a 31.5x
        speedup, with results identical to 3.9e-14.
        """
        if not values_list:
            return []
        columns: dict[Any, list[float]] = {}
        for index, parameter in enumerate(self.parameters):
            column = []
            for values in values_list:
                angles = np.asarray(values, dtype=float).reshape(-1)
                if angles.size != self.n_parameters:
                    raise ValueError(
                        f"Expected {self.n_parameters} parameters, received {angles.size}"
                    )
                column.append(float(angles[index]))
            columns[parameter] = column
        circuit = self._template_with_save(observable)
        result = run_with_deep_stack(
            lambda: self.backend.run(circuit, parameter_binds=[columns]).result()
        )
        if not result.success:
            raise RuntimeError(f"Aer simulation failed: {result.status}")
        self.circuit_count += len(values_list)
        self.batch_count += 1
        return [float(np.real(result.data(i)["energy"])) for i in range(len(values_list))]

    def run(self, jobs: Sequence[tuple[np.ndarray, Any]]) -> list[float]:
        """Evaluate <psi(theta)|O|psi(theta)> for many (theta, O) pairs.

        Jobs are grouped by observable, because `parameter_binds` binds one
        circuit -- and therefore one save instruction -- across many parameter
        vectors. Gradient batches share a single observable and so collapse to
        one Aer call.
        """
        if not jobs:
            return []
        groups: dict[int, tuple[Any, list[tuple[int, np.ndarray]]]] = {}
        for index, (values, observable) in enumerate(jobs):
            entry = groups.setdefault(id(observable), (observable, []))
            entry[1].append((index, values))
        energies = [0.0] * len(jobs)
        for observable, items in groups.values():
            evaluated = self._run_one_observable([v for _, v in items], observable)
            for (index, _), value in zip(items, evaluated):
                energies[index] = value
        return energies

    def energy(self, values: np.ndarray, observable: Any) -> float:
        return self.run([(values, observable)])[0]

    def statevector(self, values: np.ndarray) -> np.ndarray:
        """Simulate once and return the full statevector.

        ADAPT screens its whole operator pool against this one vector using
        sparse matrix products, which is far cheaper than one circuit execution
        per pool operator.
        """
        circuit = self.template.copy()
        circuit.save_statevector(label="state")
        angles = np.asarray(values, dtype=float).reshape(-1)
        if self.n_parameters:
            if angles.size != self.n_parameters:
                raise ValueError(
                    f"Expected {self.n_parameters} parameters, received {angles.size}"
                )
            binds = [
                {parameter: [float(angles[index])]
                 for index, parameter in enumerate(self.parameters)}
            ]
            result = run_with_deep_stack(
                lambda: self.backend.run(circuit, parameter_binds=binds).result()
            )
        else:
            result = run_with_deep_stack(lambda: self.backend.run(circuit).result())
        self.circuit_count += 1
        self.batch_count += 1
        return np.asarray(result.data(0)["state"], dtype=complex)

    def energy_and_gradient(
        self, values: np.ndarray, observable: Any, method: str = "finite-difference"
    ) -> tuple[float, np.ndarray]:
        """Value and gradient from a single batched Aer call.

        `parameter-shift` is exact but only valid when each parameter drives one
        rotation; `finite-difference` is the general fallback and the default for
        UCCSD.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        if method == "parameter-shift":
            if not self.exact_gradient:
                raise ValueError(
                    "parameter-shift is invalid for this ansatz (parameters are reused); "
                    "use finite-difference"
                )
            step, scale = 0.5 * np.pi, 0.5
        elif method == "finite-difference":
            step = FINITE_DIFFERENCE_STEP
            scale = 1.0 / (2.0 * step)
        else:
            raise ValueError(f"Unknown gradient method {method!r}")

        jobs: list[tuple[np.ndarray, Any]] = [(values, observable)]
        for index in range(self.n_parameters):
            for sign in (1.0, -1.0):
                shifted = values.copy()
                shifted[index] += sign * step
                jobs.append((shifted, observable))
        evaluated = self.run(jobs)
        gradient = np.array(
            [
                scale * (evaluated[1 + 2 * i] - evaluated[2 + 2 * i])
                for i in range(self.n_parameters)
            ]
        )
        return evaluated[0], gradient


def make_evaluator(engine: str, **kwargs: Any) -> Any:
    """Build the requested engine. Both expose the same interface."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; choose from {ENGINES}")
    if engine == "numpy":
        from fastsim import NumpyEvaluator

        return NumpyEvaluator(**kwargs)
    return AerEvaluator(**kwargs)


def _minimize(
    evaluator: AerEvaluator,
    observable: Any,
    start: np.ndarray,
    optimizer: str,
    tolerance: float,
    maxiter: int,
    gradient: str = "finite-difference",
) -> tuple[float, np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    optimizer = optimizer.lower()
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer {optimizer!r}; choose from {OPTIMIZERS}")
    history: list[float] = []

    if optimizer == "cobyla":
        # What the paper used. Gradient-free, so one circuit per evaluation.
        def objective(values: np.ndarray) -> float:
            energy = evaluator.energy(values, observable)
            history.append(energy)
            return energy

        outcome = minimize(
            objective, start, method="COBYLA", tol=tolerance, options={"maxiter": maxiter}
        )
    else:
        def objective_with_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            energy, grad = evaluator.energy_and_gradient(values, observable, gradient)
            history.append(energy)
            return energy, grad

        options: dict[str, Any] = {"maxiter": maxiter, "ftol": tolerance}
        if optimizer == "l-bfgs-b":
            options["gtol"] = tolerance
        outcome = minimize(
            objective_with_gradient,
            start,
            jac=True,
            method="L-BFGS-B" if optimizer == "l-bfgs-b" else "SLSQP",
            options=options,
        )

    return (
        float(outcome.fun),
        np.asarray(outcome.x, dtype=float),
        {
            "optimizer": optimizer,
            "gradient": "none" if optimizer == "cobyla" else gradient,
            "tolerance": tolerance,
            "maxiter_requested": maxiter,
            "iterations": int(getattr(outcome, "nit", -1)),
            "objective_evaluations": len(history),
            "converged": bool(outcome.success),
            "message": str(outcome.message),
            "energy_history": [float(v) for v in history],
        },
    )


def run_adapt_vqe(
    n_qubits: int,
    n_electrons: int,
    operators: dict[str, Any],
    *,
    engine: str,
    gradient: str,
    method: str,
    threads: int | None,
    parallel_experiments: int | None,
    seed: int,
    optimizer: str,
    tolerance: float,
    maxiter: int,
    max_operators: int,
    gradient_tolerance: float,
) -> tuple[float, np.ndarray, Any, dict[str, Any], Any]:
    """Drive ADAPT-VQE, injecting the Aer plumbing it needs.

    Returns the converged energy, its parameters, the final circuit, ADAPT's own
    diagnostics, and an evaluator bound to that final circuit so the caller can
    measure observables on it.
    """
    from adapt import run_adapt

    made: list[Any] = []

    def evaluator_for(circuit: Any) -> Any:
        evaluator = make_evaluator(
            engine, n_qubits=n_qubits, n_electrons=n_electrons, ansatz_kind="adapt",
            method=method, threads=threads, parallel_experiments=parallel_experiments,
            seed=seed, circuit=circuit,
        )
        made.append(evaluator)
        return evaluator

    def statevector_of(circuit: Any, values: np.ndarray) -> np.ndarray:
        # Screening needs an exact amplitude vector, so it always runs on an
        # exact engine regardless of what the production method is.
        exact = make_evaluator(
            engine, n_qubits=n_qubits, n_electrons=n_electrons, ansatz_kind="adapt",
            method="statevector", threads=threads,
            parallel_experiments=parallel_experiments, seed=seed, circuit=circuit,
        )
        made.append(exact)
        return exact.statevector(values)

    def minimize_circuit(circuit: Any, start: np.ndarray):
        evaluator = evaluator_for(circuit)
        return _minimize(
            evaluator, operators["objective"], start, optimizer, tolerance,
            maxiter, gradient,
        )

    outcome = run_adapt(
        n_qubits, n_electrons, operators["hamiltonian"], statevector_of,
        minimize_circuit, max_operators=max_operators,
        gradient_tolerance=gradient_tolerance,
    )

    final = evaluator_for(outcome["circuit"])
    # Roll the per-iteration evaluators' counters into the one we hand back, so
    # the reported circuit count covers the whole ADAPT run, not just the last
    # optimization.
    final.circuit_count = sum(e.circuit_count for e in made)
    final.batch_count = sum(e.batch_count for e in made)
    return (
        float(outcome["energy_hartree"]),
        np.asarray(outcome["values"], dtype=float),
        outcome["circuit"],
        outcome,
        final,
    )


# --------------------------------------------------------------------------
# Objective assembly
# --------------------------------------------------------------------------


def build_objective(
    chemistry: ChemistryResult,
    ansatz_kind: str,
    number_penalty: float = 0.0,
    spin_penalty: float = 0.0,
) -> dict[str, Any]:
    """Hamiltonian and probe observables, in the ansatz's own qubit convention.

    For UCCSD the operators are permuted into qiskit-nature's blocked spin
    ordering and left otherwise alone: the ansatz already starts at the
    reference determinant and conserves particle number, so there is no frame
    rotation and no penalty to add.

    For the hardware-efficient ansatz the old machinery is preserved: penalties
    are folded in and everything is conjugated by X on the occupied spin
    orbitals so that theta = 0 is the reference determinant.
    """
    n_qubits = chemistry.n_qubits
    n_electrons = chemistry.n_active_electrons
    spin_operator = spin_squared_qubit_operator(n_qubits)

    objective = chemistry.hamiltonian
    if number_penalty:
        objective = add_electron_number_penalty(
            objective, n_qubits, n_electrons, number_penalty
        )
    if spin_penalty:
        objective = objective + spin_penalty * spin_operator
        objective.compress(abs_tol=COMPRESSION_TOLERANCE)

    operators = {
        "objective": objective,
        "hamiltonian": chemistry.hamiltonian,
        "number": electron_number_operator(n_qubits),
        "deviation": number_deviation_operator(n_qubits, n_electrons),
        "spin_squared": spin_operator,
    }

    # Every qiskit-nature ansatz -- UCCSD, k-UpCCGSD, and the ADAPT pool -- uses
    # blocked spin-orbital ordering. Only the hardware-efficient path keeps
    # OpenFermion's interleaved ordering and the X-conjugation frame trick.
    if ansatz_kind in ("uccsd", "kupccgsd", "adapt"):
        qubit_map = interleaved_to_blocked(n_qubits)
    else:
        operators = {
            name: to_hartree_fock_frame(op, n_electrons) for name, op in operators.items()
        }
        qubit_map = None
    return {
        name: to_sparse_pauli_op(op, n_qubits, qubit_map) for name, op in operators.items()
    }


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def vqe_main(
    spec: MoleculeSpec,
    default_cache: str,
    default_result: str,
    title: str,
    default_layers: int = 4,
) -> int:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--cache", type=Path, default=Path(default_cache))
    parser.add_argument("--result", type=Path, default=Path(default_result))
    parser.add_argument(
        "--ansatz", choices=ANSATZE, default="auto",
        help=f"auto = UCCSD up to {AUTO_UCCSD_MAX_QUBITS} qubits, k-UpCCGSD above.",
    )
    parser.add_argument(
        "--layers", type=int, default=default_layers, help="Hardware-efficient ansatz only."
    )
    parser.add_argument(
        "-k", "--kupccgsd-k", type=int, default=DEFAULT_KUPCCGSD_K,
        help="Repetitions of the k-UpCCGSD block. Higher k is more accurate and "
        "proportionally more expensive.",
    )
    parser.add_argument(
        "--adapt-max-operators", type=int, default=40,
        help="ADAPT only: cap on how many operators may be appended.",
    )
    parser.add_argument(
        "--adapt-gradient-tolerance", type=float, default=1.0e-3,
        help="ADAPT only: stop once the largest pool gradient falls below this.",
    )
    parser.add_argument(
        "--number-penalty", type=float, default=None,
        help="Default: 0 for UCCSD (it conserves particle number exactly), 1 Ha for the "
        "hardware-efficient ansatz (it does not).",
    )
    parser.add_argument("--spin-penalty", type=float, default=0.0)
    parser.add_argument(
        "--method", choices=SIMULATION_METHODS, default="statevector",
        help="statevector is the measured-faster method for UCCSD (2.3x); MPS is "
        "kept for weakly entangled circuits at larger qubit counts.",
    )
    parser.add_argument("--max-bond-dimension", type=int, default=None)
    parser.add_argument("--truncation-threshold", type=float, default=DEFAULT_TRUNCATION_THRESHOLD)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="l-bfgs-b")
    parser.add_argument("--optimizer-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument(
        "--engine", choices=ENGINES, default="aer",
        help="aer = simulate the circuit, finite-difference gradients (the "
        "reference). numpy = propagate the statevector directly with exact "
        "adjoint gradients; same answers, far faster, and the one to use for "
        "production runs. Not available for --ansatz hea.",
    )
    parser.add_argument(
        "--gradient", choices=("finite-difference", "parameter-shift", "adjoint"),
        default=None,
        help="Defaults to adjoint for --engine numpy and finite-difference for "
        "aer. parameter-shift is exact but valid only for the hardware-efficient "
        "ansatz; UCCSD and k-UpCCGSD reuse parameters, so it is refused there.",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Aer threads per circuit. Default is method-aware: CPU count for "
        "statevector (deep circuits thread well), 1 for MPS.",
    )
    parser.add_argument(
        "--parallel-experiments", type=int, default=None,
        help="Aer circuit-level parallelism. Default is method-aware: 1 for "
        "statevector, CPU count for MPS.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--start-noise", type=float, default=0.0,
        help="Kick added to the theta = 0 start. UCCSD does not need one; the "
        "hardware-efficient ansatz does, because a number-conserving Hamiltonian "
        "leaves its deeper rotation layers with exactly zero gradient at theta = 0.",
    )
    args = parser.parse_args()

    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if args.optimizer_tolerance <= 0:
        parser.error("--optimizer-tolerance must be positive")
    if args.start_noise < 0:
        parser.error("--start-noise must be non-negative")
    if not args.cache.is_file():
        raise FileNotFoundError(
            f"Missing {args.cache}. Run `python run.py prepare` where PySCF is "
            "available (Linux/macOS/WSL/Colab), copy the cache here, then "
            "`python run.py validate`."
        )
    started = time.perf_counter()

    # The register size decides the ansatz, and the ansatz decides the default
    # penalty, so the cache has to be read before those defaults are settled.
    # It is read again below only after the receipt has been checked.
    args.ansatz = resolve_ansatz(args.ansatz, load_chemistry(args.cache, spec).n_qubits)
    if args.number_penalty is None:
        args.number_penalty = 1.0 if args.ansatz == "hea" else 0.0
    if args.ansatz == "hea" and args.start_noise == 0.0:
        args.start_noise = 0.05
    if args.number_penalty < 0 or args.spin_penalty < 0:
        parser.error("penalties must be non-negative")

    validation_receipt = require_validation_receipt(
        args.cache, spec, args.number_penalty, args.spin_penalty
    )
    chemistry = load_chemistry(args.cache, spec)
    operators = build_objective(
        chemistry, args.ansatz, args.number_penalty, args.spin_penalty
    )

    if args.engine == "numpy" and args.ansatz == "hea":
        parser.error(
            "--engine numpy supports the fermionic ansatze only; use --engine aer "
            "for --ansatz hea"
        )
    if args.gradient is None:
        args.gradient = "adjoint" if args.engine == "numpy" else "finite-difference"

    # ADAPT builds its circuit as it goes, so it has no fixed ansatz to stand up
    # front; a one-operator stand-in supplies the reference-determinant checks.
    evaluator = make_evaluator(
        args.engine,
        n_qubits=chemistry.n_qubits,
        n_electrons=chemistry.n_active_electrons,
        ansatz_kind="kupccgsd" if args.ansatz == "adapt" else args.ansatz,
        layers=args.layers,
        method=args.method,
        max_bond_dimension=args.max_bond_dimension,
        truncation_threshold=args.truncation_threshold,
        threads=args.threads,
        parallel_experiments=args.parallel_experiments,
        seed=args.seed,
        k=args.kupccgsd_k,
    )
    if args.gradient == "parameter-shift" and not evaluator.exact_gradient:
        print(
            "WARNING: parameter-shift is invalid for this ansatz (parameters are "
            "reused across gates); falling back to finite differences.",
            flush=True,
        )
        args.gradient = "finite-difference"

    rng = np.random.default_rng(args.seed)
    zeros = np.zeros(evaluator.n_parameters)
    start = (
        rng.normal(0.0, args.start_noise, evaluator.n_parameters)
        if args.start_noise > 0
        else zeros
    )

    print("=" * 76, flush=True)
    print(title, flush=True)
    print("=" * 76, flush=True)
    print(f"Ansatz / parameters        : {args.ansatz.upper()} / {evaluator.n_parameters}", flush=True)
    print(f"Circuit depth / 2q gates   : {evaluator.depth} / {evaluator.two_qubit_gates}", flush=True)
    if args.engine == "aer":
        print(f"Engine / method            : aer / {args.method}, "
              f"{evaluator.backend.options.max_parallel_threads} threads x "
              f"{evaluator.backend.options.max_parallel_experiments} experiments", flush=True)
    else:
        print("Engine / method            : numpy / direct statevector + adjoint gradients",
              flush=True)
    print(f"Active space / qubits      : {chemistry.n_active_electrons}e,{chemistry.n_qubits // 2}o / {chemistry.n_qubits}", flush=True)
    print(f"Basis / orbitals           : {chemistry.metadata['basis']} / {chemistry.metadata.get('orbital_selection')}", flush=True)
    print(f"Solvation                  : {chemistry.metadata.get('solvation_model', 'vacuum')}", flush=True)
    print(f"Pauli terms                : {len(operators['objective'])}", flush=True)
    print(f"SCF / reference det / CASCI: {chemistry.hartree_fock_energy:.10f} / {chemistry.reference_determinant_energy:.10f} / {chemistry.reference_energy:.10f} Ha", flush=True)
    print(f"Optimizer / gradient       : {args.optimizer} / {args.gradient if args.optimizer != 'cobyla' else 'none'}", flush=True)

    # theta = 0 must be the reference determinant. With MP2 natural orbitals
    # that is NOT the SCF energy, which is exactly why the determinant energy is
    # computed and stored separately.
    start_energy = evaluator.energy(zeros, operators["hamiltonian"])
    start_error = start_energy - chemistry.reference_determinant_energy
    truncation = float(chemistry.metadata.get("truncation_l1_bound_hartree", 0.0))
    print(f"theta=0 energy / det error : {start_energy:.10f} Ha / {start_error:+.3e} Ha", flush=True)
    if abs(start_error) > 1.0e-6 + truncation:
        raise RuntimeError(
            f"theta = 0 does not reproduce the reference determinant "
            f"(error {start_error:+.3e} Ha). The spin-orbital ordering, the "
            "Hartree-Fock reference or the qubit mapping is wrong. Run "
            "`python run.py selftest` before trusting any result."
        )

    adapt_report: dict[str, Any] | None = None
    if args.ansatz == "adapt":
        print("-" * 76, flush=True)
        print("ADAPT-VQE: growing the ansatz one operator at a time", flush=True)
        objective_value, parameters, adapt_circuit, adapt_report, evaluator = run_adapt_vqe(
            chemistry.n_qubits,
            chemistry.n_active_electrons,
            operators,
            engine=args.engine,
            gradient=args.gradient,
            method=args.method,
            threads=args.threads,
            parallel_experiments=args.parallel_experiments,
            seed=args.seed,
            optimizer=args.optimizer,
            tolerance=args.optimizer_tolerance,
            maxiter=args.maxiter,
            max_operators=args.adapt_max_operators,
            gradient_tolerance=args.adapt_gradient_tolerance,
        )
        optimizer_diagnostics = {
            "optimizer": args.optimizer,
            "gradient": args.gradient,
            "iterations": adapt_report["n_operators"],
            "converged": True,
            "message": f"ADAPT selected {adapt_report['n_operators']} of "
                       f"{adapt_report['pool_size']} pool operators",
        }
    else:
        adapt_circuit = None
        objective_value, parameters, optimizer_diagnostics = _minimize(
            evaluator, operators["objective"], start, args.optimizer,
            args.optimizer_tolerance, args.maxiter, args.gradient,
        )

    physical_energy, electron_count, leakage, spin_squared = evaluator.run(
        [
            (parameters, operators["hamiltonian"]),
            (parameters, operators["number"]),
            (parameters, operators["deviation"]),
            (parameters, operators["spin_squared"]),
        ]
    )

    # Always re-evaluate the converged state with exact Aer statevector. When the
    # production run used MPS this catches truncation error; when it used the
    # NumPy engine it is a full cross-engine check, which is the stronger test.
    exact_evaluator = AerEvaluator(
        n_qubits=chemistry.n_qubits,
        n_electrons=chemistry.n_active_electrons,
        ansatz_kind="kupccgsd" if args.ansatz == "adapt" else args.ansatz,
        layers=args.layers,
        method="statevector",
        threads=args.threads,
        seed=args.seed,
        k=args.kupccgsd_k,
        circuit=adapt_circuit,
    )
    reference_physical = exact_evaluator.energy(parameters, operators["hamiltonian"])
    mps_error = physical_energy - reference_physical

    error_mha = 1000.0 * (physical_energy - chemistry.reference_energy)
    correlation_recovered = (
        100.0
        * (chemistry.reference_determinant_energy - physical_energy)
        / (chemistry.reference_determinant_energy - chemistry.reference_energy)
        if abs(chemistry.reference_determinant_energy - chemistry.reference_energy) > 1e-12
        else float("nan")
    )
    contamination = max(args.number_penalty, CONTAMINATION_PRICE_HA) * leakage + max(
        args.spin_penalty, CONTAMINATION_PRICE_HA
    ) * abs(spin_squared)
    verified = (
        abs(error_mha) <= CHEMICAL_ACCURACY_MHA
        and contamination <= CONTAMINATION_ENERGY_BUDGET_HA
        and abs(mps_error) <= MPS_AGREEMENT_BUDGET_HA
    )
    elapsed = time.perf_counter() - started

    result = {
        "status": "verified_chemical_accuracy" if verified else "outside_chemical_accuracy",
        "accuracy_definition": (
            "VQE within 1.6 mHa of CASCI for the identical cached active-space "
            "Hamiltonian, with symmetry breaking and MPS truncation each priced "
            f"below {1000.0 * CONTAMINATION_ENERGY_BUDGET_HA:.2f} mHa"
        ),
        "protocol_note": (
            "UCCSD/HF following arXiv:2604.10487's simulator protocol. This is "
            "NOT a reproduction of that paper's ECC-DMET numbers; absolute "
            "energies are not comparable to its Table 2."
        ),
        "backend": (
            f"Qiskit Aer ({args.method})" if args.engine == "aer"
            else "NumPy statevector engine with adjoint gradients"
        ),
        "engine": args.engine,
        "workflow_sha256": workflow_fingerprint(),
        "seed": args.seed,
        "chemistry": chemistry.metadata,
        "validation": validation_receipt,
        "ansatz": {
            "name": args.ansatz,
            "kupccgsd_k": args.kupccgsd_k if args.ansatz == "kupccgsd" else None,
            "adapt": adapt_report and {
                key: adapt_report[key] for key in
                ("n_operators", "n_parameters", "pool_size", "excitations",
                 "history", "runtime_seconds", "gradient_tolerance")
            },
            "parameters": evaluator.n_parameters,
            "depth": evaluator.depth,
            "two_qubit_gates": evaluator.two_qubit_gates,
            "particle_number_conserving": args.ansatz != "hea",
            "start_noise": args.start_noise,
            "theta_zero_energy_hartree": start_energy,
            "theta_zero_error_hartree": start_error,
            "number_penalty_hartree": args.number_penalty,
            "spin_squared_penalty_hartree": args.spin_penalty,
        },
        "simulator": {
            "engine": args.engine,
            "method": args.method if args.engine == "aer" else "numpy-adjoint",
            "max_bond_dimension": args.max_bond_dimension,
            "truncation_threshold": args.truncation_threshold,
            "parallel_experiments": (
                int(evaluator.backend.options.max_parallel_experiments)
                if args.engine == "aer" else None
            ),
            "state_preparations": evaluator.circuit_count,
            "batches": evaluator.batch_count,
            # For engine=aer this is the MPS truncation error; for engine=numpy
            # it is the difference between the two independent engines.
            "vs_exact_aer_statevector_hartree": mps_error,
            "agreement_budget_hartree": MPS_AGREEMENT_BUDGET_HA,
            "exact_aer_statevector_energy_hartree": reference_physical,
        },
        "optimizer": optimizer_diagnostics,
        "vqe": {
            "objective_energy_hartree": objective_value,
            "physical_energy_hartree": physical_energy,
            "reference_determinant_energy_hartree": chemistry.reference_determinant_energy,
            "reference_energy_hartree": chemistry.reference_energy,
            "electron_number": electron_count,
            "number_deviation_squared": leakage,
            "spin_squared": spin_squared,
            "contamination_energy_hartree": contamination,
            "contamination_energy_budget_hartree": CONTAMINATION_ENERGY_BUDGET_HA,
            "error_millihartree": error_mha,
            "correlation_energy_recovered_percent": correlation_recovered,
            "within_chemical_accuracy": bool(verified),
            "parameters": np.asarray(parameters, dtype=float).reshape(-1).tolist(),
        },
        "runtime_seconds": elapsed,
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "versions": {
            name: _package_version(name)
            for name in ("qiskit", "qiskit-aer", "qiskit-nature", "openfermion", "pyscf", "numpy", "scipy")
        },
    }
    write_json_atomic(args.result, result)

    print("-" * 76, flush=True)
    print(f"VQE energy                  : {physical_energy:.10f} Ha", flush=True)
    print(f"CASCI reference             : {chemistry.reference_energy:.10f} Ha", flush=True)
    print(f"Error vs CASCI              : {error_mha:+.6f} mHa", flush=True)
    print(f"Correlation recovered       : {correlation_recovered:.4f} %", flush=True)
    print(f"Electron number / leakage   : {electron_count:.10f} / {leakage:.3e}", flush=True)
    print(f"Spin contamination <S^2>    : {spin_squared:.3e}", flush=True)
    print(f"vs exact Aer statevector    : {1000.0 * mps_error:+.6f} mHa"
          f"{'  (cross-engine check)' if args.engine == 'numpy' else ''}", flush=True)
    print(f"State preps / batches       : {evaluator.circuit_count} / {evaluator.batch_count}", flush=True)
    print(f"Iterations / runtime        : {optimizer_diagnostics['iterations']} / {elapsed:.1f} s", flush=True)
    print(f"Result                      : {args.result}", flush=True)
    if verified:
        print("VERIFIED: within active-space chemical accuracy.", flush=True)
    elif abs(mps_error) > MPS_AGREEMENT_BUDGET_HA:
        print(
            "ENGINE DISAGREEMENT: this run and an exact Aer statevector differ by "
            f"{1000.0 * mps_error:+.6f} mHa. For --method matrix_product_state raise "
            "--max-bond-dimension; otherwise the two engines disagree and the "
            "result must not be reported.",
            flush=True,
        )
    elif physical_energy > chemistry.reference_determinant_energy:
        print("FAILURE: result is above the reference determinant; optimizer failure.", flush=True)
    elif contamination > CONTAMINATION_ENERGY_BUDGET_HA:
        print("SYMMETRY BROKEN: raise --number-penalty (or use --ansatz uccsd).", flush=True)
    else:
        print("OUTSIDE CHEMICAL ACCURACY: ansatz-limited; raise --maxiter first.", flush=True)
    return 0


def backend_main(target_qubits: int, label: str) -> int:
    """Local analogue of a cluster preflight: prove Aer works before spending time."""
    parser = argparse.ArgumentParser(description="Inspect and smoke-test the local Aer backend")
    parser.add_argument("--method", choices=SIMULATION_METHODS, default="statevector")
    parser.add_argument("--ansatz", choices=ANSATZE, default="uccsd")
    args = parser.parse_args()

    from qiskit_aer import AerSimulator

    available = AerSimulator().available_methods()
    if args.method not in available:
        raise RuntimeError(f"Aer lacks method {args.method!r}; available: {available}")

    n_electrons = target_qubits // 2
    evaluator = AerEvaluator(target_qubits, n_electrons, args.ansatz, method=args.method)
    exact = AerEvaluator(target_qubits, n_electrons, args.ansatz, method="statevector")

    from qiskit.quantum_info import SparsePauliOp

    rng = np.random.default_rng(11)
    observable = SparsePauliOp.from_sparse_list(
        [("ZZ", [0, 1], 0.7), ("XX", [2, 5], -0.4), ("YY", [3, 7], 0.25), ("", [], 0.1)],
        num_qubits=target_qubits,
    )
    angles = rng.uniform(-0.4, 0.4, evaluator.n_parameters)

    started = time.perf_counter()
    method_energy = evaluator.energy(angles, observable)
    single = time.perf_counter() - started
    started = time.perf_counter()
    evaluator.run([(angles, observable)] * 32)
    batched = (time.perf_counter() - started) / 32
    exact_energy = exact.energy(angles, observable)

    failures = []
    if not np.isclose(method_energy, exact_energy, atol=1.0e-9):
        failures.append(f"{args.method} vs statevector: {method_energy - exact_energy:.3e}")

    report = {
        "label": label,
        "workflow_sha256": workflow_fingerprint(),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "aer_available_methods": list(available),
        "method": args.method,
        "ansatz": args.ansatz,
        "max_parallel_threads": int(evaluator.backend.options.max_parallel_threads),
        "max_parallel_experiments": int(evaluator.backend.options.max_parallel_experiments),
        "qubits": target_qubits,
        "parameters": evaluator.n_parameters,
        "circuit_depth": evaluator.depth,
        "two_qubit_gates": evaluator.two_qubit_gates,
        "parameter_shift_valid": evaluator.exact_gradient,
        "gradient_method": "parameter-shift" if evaluator.exact_gradient else "finite-difference",
        "energy_method": method_energy,
        "energy_statevector": exact_energy,
        "ms_per_circuit_single": 1000.0 * single,
        "ms_per_circuit_batched": 1000.0 * batched,
        "batching_speedup": single / batched if batched > 0 else None,
        "estimated_seconds_per_vqe_iteration": (2 * evaluator.n_parameters + 1) * batched,
        "versions": {
            name: _package_version(name)
            for name in ("qiskit", "qiskit-aer", "qiskit-nature", "openfermion", "pyscf", "numpy", "scipy")
        },
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("Aer backend check failed: " + "; ".join(failures))
    print(f"\n{label}: Aer backend check passed.")
    return 0
