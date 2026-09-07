#!/usr/bin/env python3
"""Qiskit Aer execution layer: operator conversion, MPS evaluation, and VQE.

This is the only module that touches a simulator; `chemistry.py`,
`integrals.py` and `validation.py` import no Qiskit at all. One hard-won lesson
is designed into it: starting the ansatz at the |0...0> *vacuum* makes the
optimizer spend its budget climbing out of a particle-number penalty and
converge to an energy above Hartree-Fock. The Hartree-Fock frame rotation below
is the fix.

Three properties this layer guarantees rather than assumes:

* **theta = 0 is guaranteed to be the Hartree-Fock determinant.** The
  Hamiltonian is conjugated by X on the occupied spin orbitals, and
  `real_amplitudes` at theta = 0 is the identity, so the optimizer provably
  starts at E_HF with both penalties at exactly zero -- no runtime probe and no
  fail-closed preflight is needed to establish it.
* **Gradients are exact.** Every ansatz parameter is a single RY angle, so the
  parameter-shift rule is exact rather than a finite-difference approximation,
  and all 2N shifted circuits go to Aer in one batched `run` call.
* **Observables at the optimum are always available**, so the physical energy,
  <N>, <S^2> and the contamination budget can never come back `unverified`.

Matrix product state is the default method. At 10 or 12 qubits it buys nothing
over `statevector` -- 4096 amplitudes is 64 KB -- but it is the method that
scales to the larger active spaces this workflow is a warm-up for, so it is the
one worth measuring. MPS is an *approximate* method whenever bond dimension is
capped, so `vqe` re-evaluates the final energy exactly with `statevector` and
refuses to report chemical accuracy if the two disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

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
    hartree_fock_occupation,
    load_chemistry,
    number_deviation_operator,
    require_validation_receipt,
    spin_squared_qubit_operator,
    to_hartree_fock_frame,
    workflow_fingerprint,
    write_json_atomic,
)

# Aer's MPS default. Kept explicit so a result records the truncation it used.
DEFAULT_TRUNCATION_THRESHOLD = 1.0e-16
# How far the approximate MPS energy may sit from the exact statevector energy
# before the accuracy claim is withdrawn. One tenth of chemical accuracy, the
# same budget the workflow already applies to symmetry breaking.
MPS_AGREEMENT_BUDGET_HA = 0.1 * CHEMICAL_ACCURACY_MHA / 1000.0
SIMULATION_METHODS = ("matrix_product_state", "statevector")
OPTIMIZERS = ("l-bfgs-b", "slsqp", "cobyla")
ANSATZE = ("real-amplitudes", "uccsd")


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


# --------------------------------------------------------------------------
# Operator conversion
# --------------------------------------------------------------------------


def to_sparse_pauli_op(operator: Any, n_qubits: int) -> Any:
    """Convert an OpenFermion QubitOperator to a Qiskit SparsePauliOp.

    Both libraries index qubits the same way -- OpenFermion's ``((0, 'X'),)``
    and Qiskit's ``from_sparse_list([("X", [0], c)])`` both mean X on qubit 0 --
    so no endianness correction is needed or wanted here. (Qiskit's *label*
    strings are little-endian, which is why the sparse-list form is used
    instead: it takes explicit indices and cannot be silently reversed.)

    The result must be Hermitian for `save_expectation_value`, so a non-real
    coefficient is a hard error rather than something quietly discarded.
    """
    from qiskit.quantum_info import SparsePauliOp

    sparse_list = []
    max_imaginary = 0.0
    for term, coefficient in operator.terms.items():
        coefficient = complex(coefficient)
        max_imaginary = max(max_imaginary, abs(coefficient.imag))
        indices = [int(qubit) for qubit, _ in term]
        if any(index < 0 or index >= n_qubits for index in indices):
            raise ValueError(
                f"Pauli term {term} acts outside the {n_qubits}-qubit register"
            )
        sparse_list.append(("".join(pauli for _, pauli in term), indices, coefficient.real))
    if max_imaginary > 1.0e-10:
        raise ValueError(
            f"Operator is not Hermitian: largest imaginary coefficient {max_imaginary:.3e}"
        )
    if not sparse_list:
        sparse_list = [("", [], 0.0)]
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits).simplify()


# --------------------------------------------------------------------------
# Aer evaluation
# --------------------------------------------------------------------------


def build_uccsd_ansatz(n_qubits: int, n_electrons: int) -> Any:
    """Spin-adapted UCCSD, reordered into OpenFermion's spin-orbital convention.

    Two conventions have to be reconciled. qiskit-nature emits UCCSD over
    *blocked* spin orbitals -- all alpha, then all beta -- while this workflow's
    Hamiltonian comes from OpenFermion's ``spinorb_from_spatial``, which
    *interleaves* them (spatial orbital p occupies qubits 2p and 2p+1). The two
    differ by a fixed qubit permutation, applied once here when the template is
    built rather than anywhere near the optimizer.

    The X ladder appears on *both* sides of the evolution, which is not a typo.
    `build_objective` measures the Hamiltonian conjugated into the Hartree-Fock
    frame, H' = U H U with U the X ladder on the occupied spin orbitals. For a
    circuit A the energy in that frame is <0|C' H' C'|0> with C' = U A U, so the
    evolution has to be conjugated by the same U as the operator.

    Dropping the trailing X ladder is a silent failure rather than a loud one:
    at theta = 0 the evolution is the identity and U U = I, so the energy is
    still exactly E_HF and the frame check still passes -- but the excitation
    operators would then act on the qubit vacuum, where every annihilation
    operator gives zero, and the whole gradient vanishes. The optimizer stops at
    Hartree-Fock reporting clean convergence.
    """
    from qiskit import QuantumCircuit
    from qiskit_nature.second_q.circuit.library import UCCSD
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if n_qubits % 2:
        raise ValueError("A spin-orbital register must have an even qubit count")
    if n_electrons % 2:
        raise ValueError("UCCSD here is closed-shell singlet only")
    n_spatial = n_qubits // 2
    if not 0 < n_electrons <= n_qubits:
        raise ValueError("The electron count does not fit the register")
    pairs = n_electrons // 2

    blocked = UCCSD(n_spatial, (pairs, pairs), JordanWignerMapper())
    # blocked qubit j holds spatial orbital (j % n_spatial), spin (j // n_spatial),
    # which interleaved calls 2 * spatial + spin.
    order = [2 * (j % n_spatial) + (j // n_spatial) for j in range(n_qubits)]
    occupied = hartree_fock_occupation(n_electrons)
    interleaved = QuantumCircuit(n_qubits)
    for qubit in occupied:
        interleaved.x(qubit)
    interleaved.compose(blocked, qubits=order, inplace=True)
    for qubit in occupied:
        interleaved.x(qubit)
    return interleaved


def build_ansatz(
    n_qubits: int,
    layers: int,
    kind: str = "real-amplitudes",
    n_electrons: int | None = None,
) -> Any:
    """Build the variational circuit.

    `real-amplitudes` is RY rotations plus a linear CX ladder: the standard
    hardware-efficient baseline, and real-valued because the Hamiltonian is. At
    theta = 0 every RY is the identity and the CX ladder leaves |0...0> alone,
    so the circuit prepares the Hartree-Fock determinant of the rotated frame.

    `uccsd` is the chemistry-inspired alternative. It conserves particle number
    and spin by construction, so it does not need the number penalty that the
    hardware-efficient circuit does, and `--layers` does not apply to it.
    """
    kind = kind.lower()
    if kind not in ANSATZE:
        raise ValueError(f"Unsupported ansatz {kind!r}; choose from {ANSATZE}")
    if kind == "uccsd":
        if n_electrons is None:
            raise ValueError("UCCSD needs the active electron count")
        return build_uccsd_ansatz(n_qubits, n_electrons)

    from qiskit.circuit.library import real_amplitudes

    if layers < 1:
        raise ValueError("layers must be positive")
    return real_amplitudes(n_qubits, reps=layers, entanglement="linear")


def parameter_shift_is_exact(circuit: Any) -> bool:
    """True when every parameter is a lone RY angle used exactly once.

    The parameter-shift rule dE/dt = [E(t + pi/2) - E(t - pi/2)] / 2 holds for a
    gate exp(-i t P / 2) with P^2 = I. It is exact for RY. It stops being exact
    the moment a parameter is shared between two gates or feeds an expression
    like 2*theta, so this is checked rather than assumed -- swapping the ansatz
    should degrade to finite differences, not to silently wrong gradients.
    """
    from qiskit.circuit import Parameter

    seen: set[Any] = set()
    for instruction in circuit.data:
        for parameter_expression in instruction.operation.params:
            parameters = getattr(parameter_expression, "parameters", ())
            if not parameters:
                continue
            if instruction.operation.name != "ry":
                return False
            if len(parameters) != 1:
                return False
            parameter = next(iter(parameters))
            if not isinstance(parameter, Parameter):
                return False
            # Reject 2*theta, theta + 0.3, and any other non-identity mapping.
            if parameter_expression != parameter:
                return False
            if parameter in seen:
                return False
            seen.add(parameter)
    return True


class AerEvaluator:
    """Batched, exact expectation values for one fixed ansatz on one method."""

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        method: str = "matrix_product_state",
        max_bond_dimension: int | None = None,
        truncation_threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
        threads: int | None = None,
        seed: int = 7,
        ansatz: str = "real-amplitudes",
        n_electrons: int | None = None,
    ) -> None:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        if method not in SIMULATION_METHODS:
            raise ValueError(f"Unsupported method {method!r}; choose from {SIMULATION_METHODS}")
        options: dict[str, Any] = {"method": method, "precision": "double", "seed_simulator": seed}
        if method == "matrix_product_state":
            options["matrix_product_state_truncation_threshold"] = truncation_threshold
            if max_bond_dimension is not None:
                options["matrix_product_state_max_bond_dimension"] = int(max_bond_dimension)
        if threads is not None:
            options["max_parallel_threads"] = int(threads)

        self.method = method
        self.n_qubits = n_qubits
        self.layers = layers
        self.ansatz_kind = ansatz
        self.backend = AerSimulator(**options)
        self.ansatz = build_ansatz(n_qubits, layers, ansatz, n_electrons)
        # Transpile the *parameterised* template once; binding a parameter
        # vector afterwards costs nothing, and save-instructions need no
        # translation, so no circuit is ever transpiled inside the optimizer.
        self.template = transpile(self.ansatz, self.backend, optimization_level=1)
        self.layout = self.template.layout
        self.n_parameters = int(self.template.num_parameters)
        self.exact_gradient = parameter_shift_is_exact(self.template)
        self.circuit_count = 0
        self.batch_count = 0

    def _observable_on_device(self, observable: Any) -> Any:
        """Relabel an observable onto the transpiled circuit's physical qubits."""
        if self.layout is None:
            return observable
        return observable.apply_layout(self.layout)

    def run(self, jobs: Sequence[tuple[np.ndarray, Any]]) -> list[float]:
        """Evaluate <psi(theta)|O|psi(theta)> for many (theta, O) pairs at once.

        One Aer `run` call for the whole batch: the per-circuit Python overhead
        dominates a 10-qubit simulation, so batching the 2N parameter-shift
        circuits is what makes gradient-based optimization cheap here.
        """
        if not jobs:
            return []
        circuits = []
        for values, observable in jobs:
            angles = np.asarray(values, dtype=float).reshape(-1)
            if angles.size != self.n_parameters:
                raise ValueError(
                    f"Expected {self.n_parameters} parameters, received {angles.size}"
                )
            circuit = self.template.assign_parameters(angles)
            circuit.save_expectation_value(
                self._observable_on_device(observable),
                qubits=list(range(self.template.num_qubits)),
                label="energy",
            )
            circuits.append(circuit)
        result = self.backend.run(circuits).result()
        if not result.success:
            raise RuntimeError(f"Aer simulation failed: {result.status}")
        self.circuit_count += len(circuits)
        self.batch_count += 1
        return [float(np.real(result.data(index)["energy"])) for index in range(len(circuits))]

    def energy(self, values: np.ndarray, observable: Any) -> float:
        return self.run([(values, observable)])[0]

    def energy_and_gradient(
        self, values: np.ndarray, observable: Any, finite_difference: bool = False
    ) -> tuple[float, np.ndarray]:
        """Value and gradient from a single batched Aer call."""
        values = np.asarray(values, dtype=float).reshape(-1)
        step = 1.0e-6 if finite_difference else 0.5 * np.pi
        scale = 1.0 / (2.0 * step) if finite_difference else 0.5
        jobs: list[tuple[np.ndarray, Any]] = [(values, observable)]
        for index in range(self.n_parameters):
            for sign in (1.0, -1.0):
                shifted = values.copy()
                shifted[index] += sign * step
                jobs.append((shifted, observable))
        evaluated = self.run(jobs)
        gradient = np.array(
            [
                scale * (evaluated[1 + 2 * index] - evaluated[2 + 2 * index])
                for index in range(self.n_parameters)
            ]
        )
        return evaluated[0], gradient


def _minimize(
    evaluator: AerEvaluator,
    observable: Any,
    start: np.ndarray,
    optimizer: str,
    tolerance: float,
    maxiter: int,
    finite_difference: bool,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    optimizer = optimizer.lower()
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer {optimizer!r}; choose from {OPTIMIZERS}")
    history: list[float] = []

    if optimizer == "cobyla":
        def objective(values: np.ndarray) -> float:
            energy = evaluator.energy(values, observable)
            history.append(energy)
            return energy

        outcome = minimize(
            objective,
            start,
            method="COBYLA",
            tol=tolerance,
            options={"maxiter": maxiter},
        )
    else:
        def objective_with_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            energy, gradient = evaluator.energy_and_gradient(
                values, observable, finite_difference
            )
            history.append(energy)
            return energy, gradient

        options = {"maxiter": maxiter}
        if optimizer == "l-bfgs-b":
            options["ftol"] = tolerance
            options["gtol"] = tolerance
        else:
            options["ftol"] = tolerance
        outcome = minimize(
            objective_with_gradient,
            start,
            jac=True,
            method="L-BFGS-B" if optimizer == "l-bfgs-b" else "SLSQP",
            options=options,
        )

    diagnostics = {
        "optimizer": optimizer,
        "tolerance": tolerance,
        "maxiter_requested": maxiter,
        "iterations": int(getattr(outcome, "nit", -1)),
        "objective_evaluations": len(history),
        "converged": bool(outcome.success),
        "message": str(outcome.message),
        "gradient": (
            "none"
            if optimizer == "cobyla"
            else "finite_difference" if finite_difference else "parameter_shift"
        ),
    }
    return float(outcome.fun), np.asarray(outcome.x, dtype=float), diagnostics


# --------------------------------------------------------------------------
# Objective assembly
# --------------------------------------------------------------------------


def build_objective(
    chemistry: ChemistryResult, number_penalty: float, spin_penalty: float
) -> dict[str, Any]:
    """Penalised, Hartree-Fock-framed Hamiltonian plus the probe observables.

    Every operator is rotated into the same frame as the ansatz, so a single
    parameter vector is meaningful for all of them.
    """
    n_qubits = chemistry.n_qubits
    n_electrons = chemistry.n_active_electrons
    objective = chemistry.hamiltonian
    if number_penalty:
        objective = add_electron_number_penalty(
            objective, n_qubits, n_electrons, number_penalty
        )
    spin_operator = spin_squared_qubit_operator(n_qubits)
    if spin_penalty:
        objective = objective + spin_penalty * spin_operator
        objective.compress(abs_tol=COMPRESSION_TOLERANCE)
    framed = {
        "objective": to_hartree_fock_frame(objective, n_electrons),
        "hamiltonian": to_hartree_fock_frame(chemistry.hamiltonian, n_electrons),
        "number": to_hartree_fock_frame(electron_number_operator(n_qubits), n_electrons),
        "deviation": to_hartree_fock_frame(
            number_deviation_operator(n_qubits, n_electrons), n_electrons
        ),
        "spin_squared": to_hartree_fock_frame(spin_operator, n_electrons),
    }
    return {name: to_sparse_pauli_op(op, n_qubits) for name, op in framed.items()}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def vqe_main(
    spec: MoleculeSpec,
    default_cache: str,
    default_result: str,
    title: str,
    default_layers: int,
) -> int:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--cache", type=Path, default=Path(default_cache))
    parser.add_argument("--result", type=Path, default=Path(default_result))
    parser.add_argument("--layers", type=int, default=default_layers)
    parser.add_argument(
        "--ansatz",
        choices=ANSATZE,
        default="real-amplitudes",
        help="Variational circuit. 'real-amplitudes' is the hardware-efficient "
        "RY + linear-CX ladder and needs --layers and a number penalty. 'uccsd' "
        "is the chemistry-inspired alternative: it conserves particle number and "
        "spin by construction, ignores --layers, and is what the benchmark "
        "literature compares against.",
    )
    parser.add_argument("--number-penalty", type=float, default=1.0)
    parser.add_argument(
        "--spin-penalty",
        type=float,
        default=0.0,
        help="Off by default: S^2 adds a few hundred Pauli terms to every energy "
        "evaluation to guard a secondary risk. <S^2> is still measured at the "
        "optimum and still gates the accuracy claim. Enable only if it comes back bad.",
    )
    parser.add_argument(
        "--method",
        choices=SIMULATION_METHODS,
        default="matrix_product_state",
        help="Aer simulation method. MPS is the default and is what scales; "
        "the final energy is re-checked against statevector either way.",
    )
    parser.add_argument(
        "--max-bond-dimension",
        type=int,
        default=None,
        help="Cap the MPS bond dimension to bound memory. Uncapped by default: "
        "at 10 qubits nothing needs capping, and a cap makes the method approximate.",
    )
    parser.add_argument(
        "--truncation-threshold", type=float, default=DEFAULT_TRUNCATION_THRESHOLD
    )
    parser.add_argument(
        "--optimizer", choices=OPTIMIZERS, default="l-bfgs-b"
    )
    parser.add_argument("--optimizer-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument(
        "--finite-difference",
        action="store_true",
        help="Use finite-difference instead of exact parameter-shift gradients.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Aer parallel threads. A 10-qubit state is 16 KB and sits in L1, so "
        "extra threads only add fork/join cost; leave unset or use 1.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--start-noise",
        type=float,
        default=0.05,
        help="Standard deviation of the kick added to the theta = 0 start. "
        "theta = 0 is the right *reference* but a bad *start*: on a "
        "number-conserving Hamiltonian the deepest rotation layers have exactly "
        "zero gradient there, so a pure zero start leaves much of the ansatz "
        "frozen and the optimizer stalls after a few iterations. A kick this "
        "small moves the energy by well under a millihartree while making every "
        "parameter live. Set 0 to start at exactly theta = 0.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Start from angles uniform in [-pi, pi] instead of near theta = 0. "
        "In the Hartree-Fock frame theta = 0 is the reference determinant, so "
        "this reproduces the vacuum-start failure mode on purpose.",
    )
    args = parser.parse_args()
    if args.layers < 1 or args.maxiter < 1:
        parser.error("--layers and --maxiter must be positive")
    if args.number_penalty < 0 or args.spin_penalty < 0 or args.optimizer_tolerance <= 0:
        parser.error("penalty must be non-negative and tolerance positive")
    if args.max_bond_dimension is not None and args.max_bond_dimension < 1:
        parser.error("--max-bond-dimension must be positive")
    if args.start_noise < 0:
        parser.error("--start-noise must be non-negative")
    if not args.cache.is_file():
        raise FileNotFoundError(
            f"Missing {args.cache}. Run `python run.py prepare` where PySCF is "
            "available (Linux/macOS/WSL/Colab), copy the cache here, then "
            "`python run.py validate`."
        )
    validation_receipt = require_validation_receipt(
        args.cache, spec, args.number_penalty, args.spin_penalty
    )

    started = time.perf_counter()
    np.random.seed(args.seed)
    chemistry = load_chemistry(args.cache, spec)
    operators = build_objective(chemistry, args.number_penalty, args.spin_penalty)

    evaluator = AerEvaluator(
        n_qubits=chemistry.n_qubits,
        layers=args.layers,
        method=args.method,
        max_bond_dimension=args.max_bond_dimension,
        truncation_threshold=args.truncation_threshold,
        threads=args.threads,
        seed=args.seed,
        ansatz=args.ansatz,
        n_electrons=chemistry.n_active_electrons,
    )
    rng = np.random.default_rng(args.seed)
    if args.random_start:
        start = rng.uniform(-np.pi, np.pi, evaluator.n_parameters)
        start_description = "random in [-pi, pi]"
    elif args.start_noise > 0:
        start = rng.normal(0.0, args.start_noise, evaluator.n_parameters)
        start_description = f"theta=0 + N(0, {args.start_noise}) kick"
    else:
        start = np.zeros(evaluator.n_parameters)
        start_description = "theta=0 exactly (Hartree-Fock)"

    print("=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)
    print(f"Aer method / threads       : {args.method} / {args.threads or 'auto'}", flush=True)
    print(f"Active space / qubits      : {chemistry.n_active_electrons}e,{chemistry.n_qubits // 2}o / {chemistry.n_qubits}", flush=True)
    print(f"Basis / Pauli terms        : {chemistry.metadata['basis']} / {len(operators['objective'])}", flush=True)
    print(f"HF / CASCI energies        : {chemistry.hartree_fock_energy:.12f} / {chemistry.reference_energy:.12f} Ha", flush=True)
    layer_note = "n/a" if args.ansatz == "uccsd" else str(args.layers)
    print(f"Ansatz / layers / parameters: {args.ansatz} / {layer_note} / {evaluator.n_parameters}", flush=True)
    print(f"Start / gradient           : {start_description} / {'finite-difference' if args.finite_difference else 'parameter-shift'}", flush=True)

    if not args.finite_difference and not evaluator.exact_gradient and args.optimizer != "cobyla":
        print(
            "WARNING: the ansatz does not satisfy the parameter-shift "
            "preconditions; falling back to finite differences.",
            flush=True,
        )
        args.finite_difference = True

    # theta = 0 must reproduce Hartree-Fock exactly, or the frame rotation is
    # wrong and every energy downstream is measured from the wrong place.
    zeros = np.zeros(evaluator.n_parameters)
    start_energy, zero_gradient = evaluator.energy_and_gradient(
        zeros, operators["hamiltonian"], args.finite_difference
    )
    start_error = start_energy - chemistry.hartree_fock_energy
    live_parameters = int(np.count_nonzero(np.abs(zero_gradient) > 1.0e-9))
    print(f"theta=0 energy / HF error  : {start_energy:.12f} Ha / {start_error:+.3e} Ha", flush=True)
    print(
        f"theta=0 live parameters    : {live_parameters}/{evaluator.n_parameters}"
        f" (|grad| = {np.linalg.norm(zero_gradient):.3e})",
        flush=True,
    )
    if abs(start_error) > 1.0e-6 + float(chemistry.metadata["truncation_l1_bound_hartree"]):
        raise RuntimeError(
            f"theta = 0 does not reproduce Hartree-Fock (error {start_error:+.3e} Ha). "
            "The Hartree-Fock frame rotation or the qubit ordering is wrong; "
            "run `python run.py selftest` before trusting any result."
        )
    # A number-conserving Hamiltonian kills the gradient of the deeper rotation
    # layers at theta = 0, because a single RY there can only reach states whose
    # electron count differs from the reference. Starting exactly at zero
    # therefore optimizes a fraction of the circuit and stops early looking
    # converged, which is why --start-noise defaults to a small kick.
    if live_parameters < evaluator.n_parameters and args.start_noise == 0 and not args.random_start:
        print(
            f"WARNING: only {live_parameters} of {evaluator.n_parameters} parameters have a "
            "non-zero gradient at theta = 0, so an exact zero start freezes the rest. "
            "Use --start-noise 0.05 unless you are deliberately reproducing that.",
            flush=True,
        )

    objective_value, parameters, optimizer_diagnostics = _minimize(
        evaluator,
        operators["objective"],
        start,
        args.optimizer,
        args.optimizer_tolerance,
        args.maxiter,
        args.finite_difference,
    )

    # One batched call for every observable at the optimum.
    physical_energy, electron_count, leakage, spin_squared = evaluator.run(
        [
            (parameters, operators["hamiltonian"]),
            (parameters, operators["number"]),
            (parameters, operators["deviation"]),
            (parameters, operators["spin_squared"]),
        ]
    )

    # MPS is exact only while nothing is truncated. Re-evaluate the converged
    # state with a method that has no truncation at all and price the gap.
    exact_evaluator = AerEvaluator(
        n_qubits=chemistry.n_qubits,
        layers=args.layers,
        method="statevector",
        threads=args.threads,
        seed=args.seed,
        ansatz=args.ansatz,
        n_electrons=chemistry.n_active_electrons,
    )
    reference_physical_energy = exact_evaluator.energy(parameters, operators["hamiltonian"])
    mps_error = physical_energy - reference_physical_energy

    error_mha = 1000.0 * (physical_energy - chemistry.reference_energy)
    # How much energy the broken symmetries could be hiding. A hardware-efficient
    # ansatz always leaks some particle number; what matters is that the leak is
    # too small to explain the reported error. Each broken symmetry is priced at
    # no less than CONTAMINATION_PRICE_HA, so turning a penalty off cannot buy a
    # weaker accuracy claim.
    contamination_energy = max(args.number_penalty, CONTAMINATION_PRICE_HA) * leakage + max(
        args.spin_penalty, CONTAMINATION_PRICE_HA
    ) * abs(spin_squared)
    verified = (
        abs(error_mha) <= CHEMICAL_ACCURACY_MHA
        and contamination_energy <= CONTAMINATION_ENERGY_BUDGET_HA
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
        "backend": f"Qiskit Aer ({args.method})",
        "workflow_sha256": workflow_fingerprint(),
        "seed": args.seed,
        "chemistry": chemistry.metadata,
        "validation": validation_receipt,
        "ansatz": {
            # Record the ansatz that actually ran. Hardcoding a name here made
            # every result file claim `real_amplitudes` even for a UCCSD run,
            # which is exactly the kind of silent provenance error the rest of
            # this workflow's fingerprinting exists to prevent.
            "name": args.ansatz.replace("-", "_"),
            "entanglement": "linear" if args.ansatz == "real-amplitudes" else None,
            "layers": args.layers if args.ansatz == "real-amplitudes" else None,
            "parameters": evaluator.n_parameters,
            "hartree_fock_frame": True,
            "start": start_description,
            "start_noise": 0.0 if args.random_start else args.start_noise,
            "theta_zero_energy_hartree": start_energy,
            "theta_zero_error_hartree": start_error,
            "theta_zero_live_parameters": live_parameters,
            "theta_zero_gradient_norm": float(np.linalg.norm(zero_gradient)),
            "number_penalty_hartree": args.number_penalty,
            "spin_squared_penalty_hartree": args.spin_penalty,
            "contamination_price_hartree": CONTAMINATION_PRICE_HA,
        },
        "simulator": {
            "method": args.method,
            "max_bond_dimension": args.max_bond_dimension,
            "truncation_threshold": args.truncation_threshold,
            "threads_requested": args.threads,
            "circuits_executed": evaluator.circuit_count,
            "aer_batches": evaluator.batch_count,
            "mps_vs_statevector_hartree": mps_error,
            "mps_agreement_budget_hartree": MPS_AGREEMENT_BUDGET_HA,
            "statevector_energy_hartree": reference_physical_energy,
        },
        "optimizer": optimizer_diagnostics,
        "vqe": {
            "objective_energy_hartree": objective_value,
            "physical_energy_hartree": physical_energy,
            "electron_number": electron_count,
            "number_deviation_squared": leakage,
            "spin_squared": spin_squared,
            "contamination_energy_hartree": contamination_energy,
            "contamination_energy_budget_hartree": CONTAMINATION_ENERGY_BUDGET_HA,
            "reference_energy_hartree": chemistry.reference_energy,
            "error_millihartree": error_mha,
            "within_chemical_accuracy": bool(verified),
            "parameters": np.asarray(parameters, dtype=float).reshape(-1).tolist(),
        },
        "runtime_seconds": elapsed,
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "versions": {
            name: _package_version(name)
            for name in ("qiskit", "qiskit-aer", "openfermion", "pyscf", "numpy", "scipy")
        },
    }
    write_json_atomic(args.result, result)

    print("-" * 72, flush=True)
    print(f"Objective energy            : {objective_value:.12f} Ha", flush=True)
    print(f"Physical energy             : {physical_energy:.12f} Ha", flush=True)
    print(f"Electron number / leakage   : {electron_count:.10f} / {leakage:.3e}", flush=True)
    print(f"Spin contamination <S^2>    : {spin_squared:.3e}", flush=True)
    print(
        f"Contamination / budget      : {1000.0 * contamination_energy:.4f}"
        f" / {1000.0 * CONTAMINATION_ENERGY_BUDGET_HA:.4f} mHa",
        flush=True,
    )
    print(
        f"MPS vs statevector          : {1000.0 * mps_error:+.6f} mHa"
        f" (budget {1000.0 * MPS_AGREEMENT_BUDGET_HA:.4f})",
        flush=True,
    )
    print(f"Error vs CASCI              : {error_mha:+.6f} mHa", flush=True)
    print(
        f"Circuits / batches          : {evaluator.circuit_count} / {evaluator.batch_count}",
        flush=True,
    )
    print(f"Runtime / result            : {elapsed:.3f} s / {args.result}", flush=True)
    if verified:
        print("VERIFIED: within active-space chemical accuracy.", flush=True)
    elif abs(mps_error) > MPS_AGREEMENT_BUDGET_HA:
        print(
            "MPS TRUNCATION TOO COARSE: the approximate and exact energies "
            "disagree by more than the budget; raise --max-bond-dimension or "
            "lower --truncation-threshold.",
            flush=True,
        )
    elif physical_energy > chemistry.hartree_fock_energy:
        print(
            "FAILURE: result is above Hartree-Fock; this is optimizer failure, "
            "not chemistry. Do not add layers.",
            flush=True,
        )
    elif contamination_energy > CONTAMINATION_ENERGY_BUDGET_HA:
        print(
            "SYMMETRY BROKEN: the state leaks too much particle number or spin "
            "for the energy to be meaningful; raise --number-penalty.",
            flush=True,
        )
    else:
        print(
            "OUTSIDE CHEMICAL ACCURACY: working but ansatz-limited. Add layers.",
            flush=True,
        )
    return 0


def backend_main(target_qubits: int, label: str) -> int:
    """Local analogue of the cluster preflight: prove Aer works before spending time.

    Checks the two things that can silently corrupt a result: that MPS agrees
    with an exact statevector on a non-trivial entangled state, and that the
    parameter-shift gradient matches finite differences.
    """
    parser = argparse.ArgumentParser(description="Inspect and smoke-test the local Aer backend")
    parser.add_argument("--method", choices=SIMULATION_METHODS, default="matrix_product_state")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    from qiskit_aer import AerSimulator

    available = AerSimulator().available_methods()
    if args.method not in available:
        raise RuntimeError(
            f"Aer does not offer method {args.method!r}; available: {available}"
        )

    observable = to_sparse_pauli_op_from_pairs(
        target_qubits,
        [((0, "Z"), (1, "Z")), ((2, "X"), (5, "X")), ((3, "Y"), (7, "Y")), ()],
        [0.7, -0.4, 0.25, 0.1],
    )
    evaluator = AerEvaluator(target_qubits, layers=3, method=args.method, threads=args.threads)
    exact = AerEvaluator(target_qubits, layers=3, method="statevector", threads=args.threads)
    rng = np.random.default_rng(11)
    angles = rng.uniform(-np.pi, np.pi, evaluator.n_parameters)

    method_energy = evaluator.energy(angles, observable)
    exact_energy = exact.energy(angles, observable)
    zero_energy = evaluator.energy(np.zeros(evaluator.n_parameters), observable)

    _, shift_gradient = evaluator.energy_and_gradient(angles, observable, finite_difference=False)
    _, difference_gradient = exact.energy_and_gradient(angles, observable, finite_difference=True)
    gradient_error = float(np.max(np.abs(shift_gradient - difference_gradient)))

    failures = []
    if not np.isclose(method_energy, exact_energy, atol=1.0e-10):
        failures.append(
            f"{args.method} disagrees with statevector by {method_energy - exact_energy:.3e}"
        )
    if not evaluator.exact_gradient:
        failures.append("parameter-shift preconditions are not met by the ansatz")
    if gradient_error > 1.0e-5:
        failures.append(f"parameter-shift gradient error {gradient_error:.3e}")
    # theta = 0 must be |0...0>: <ZZ> = +0.7 and every off-diagonal term zero.
    if not np.isclose(zero_energy, 0.8, atol=1.0e-12):
        failures.append(f"theta=0 is not the computational-basis vacuum ({zero_energy:.12f})")

    report = {
        "label": label,
        "workflow_sha256": workflow_fingerprint(),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "aer_available_methods": list(available),
        "method": args.method,
        "target_qubits": target_qubits,
        "ansatz_parameters": evaluator.n_parameters,
        "parameter_shift_exact": evaluator.exact_gradient,
        "energy_method": method_energy,
        "energy_statevector": exact_energy,
        "energy_theta_zero": zero_energy,
        "gradient_max_abs_error": gradient_error,
        "versions": {
            name: _package_version(name)
            for name in ("qiskit", "qiskit-aer", "openfermion", "pyscf", "numpy", "scipy")
        },
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("Aer backend check failed: " + "; ".join(failures))
    print(f"\n{label}: Aer backend check passed.")
    return 0


def to_sparse_pauli_op_from_pairs(
    n_qubits: int, terms: Sequence[tuple], coefficients: Sequence[float]
) -> Any:
    """Small helper so the backend check needs no OpenFermion import."""
    from qiskit.quantum_info import SparsePauliOp

    sparse_list = [
        ("".join(pauli for _, pauli in term), [int(qubit) for qubit, _ in term], float(value))
        for term, value in zip(terms, coefficients)
    ]
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits).simplify()
