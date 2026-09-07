#!/usr/bin/env python3
"""End-to-end proof of the local Qiskit stack, with no PySCF and no cache.

PySCF publishes no Windows wheel, so on Windows the LiH Hamiltonian has to be
built elsewhere and copied in. That leaves a gap: nothing on the Windows machine
would be exercised until the cache arrives, and a mapping or ordering bug would
then surface as a wrong number rather than a failed test.

This command closes that gap. It builds a random but *structurally real*
Hamiltonian at the same active-space size as the selected LiH space --
generated through the identical OpenFermion path `chemistry.py` uses, so it is
Hermitian, particle-number conserving and has a genuine ground state at the
right electron count -- and then runs every piece of the local workflow against
exact classical answers:

1. the OpenFermion -> Qiskit operator conversion, checked spectrally *and*
   determinant by determinant, which is what actually pins down qubit ordering;
2. the Hartree-Fock frame rotation, i.e. that theta = 0 really is the reference
   determinant;
3. matrix product state against exact statevector;
4. parameter-shift gradients against finite differences;
5. a full VQE, checked against exact diagonalization in the target electron
   sector and against the variational bound.

If this passes, the only thing still missing on this machine is the molecule.

One thing step 5 deliberately does NOT assert is that the VQE reaches the exact
answer, or even that it beats Hartree-Fock. The synthetic Hamiltonian is random,
and at the sparse electron counts LiH's frozen-core space uses (2 electrons in
10 spin orbitals) a linear-entanglement hardware-efficient ansatz is measurably
unable to solve it: near theta = 0 the landscape is flat, and further out
L-BFGS-B converges cleanly to minima *above* Hartree-Fock. That is a real
property of the ansatz on a random Hamiltonian, not a bug in the stack, and a
self-test that demanded otherwise would be asserting expressivity it cannot
promise. What step 5 does assert is what correctness guarantees: the variational
bound, clean termination, and that the optimizer never returns a point worse
than the one it started from. Correlation recovery is reported as a diagnostic.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chemistry import (
    ChemistryResult,
    MoleculeSpec,
    conjugate_with_x,
    electron_number_operator,
    hartree_fock_occupation,
    number_deviation_operator,
    save_chemistry,
)
from qiskit_runtime import (
    AerEvaluator,
    DEFAULT_TRUNCATION_THRESHOLD,
    _minimize,
    build_objective,
    to_sparse_pauli_op,
)


def _synthetic_chemistry(seed: int, norb: int, nelec: int) -> ChemistryResult:
    """A random Hamiltonian of the right shape, built the same way the real one is."""
    from openfermion import (
        InteractionOperator,
        get_fermion_operator,
        get_sparse_operator,
        jordan_wigner,
    )
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from scipy.sparse.linalg import eigsh

    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    rng = np.random.default_rng(seed)

    # A bound one-body part: descending diagonal orbital energies plus small
    # symmetric coupling, so the lowest orbitals are genuinely the occupied ones.
    one_body = rng.normal(0.0, 0.15, (norb, norb))
    one_body = 0.5 * (one_body + one_body.T)
    one_body -= np.diag(np.linspace(3.0, 0.5, norb))

    # Chemist-notation (ij|kl) with the full eight-fold symmetry, exactly the
    # symmetry `ao2mo.restore(1, ...)` hands back in chemistry.py.
    two_body = rng.normal(0.0, 0.12, (norb, norb, norb, norb))
    two_body = two_body + two_body.transpose(1, 0, 2, 3)
    two_body = two_body + two_body.transpose(0, 1, 3, 2)
    two_body = two_body + two_body.transpose(2, 3, 0, 1)
    two_body /= 8.0

    core_energy = -12.5
    two_body_openfermion = np.asarray(
        two_body.transpose(0, 2, 3, 1), dtype=float, order="C"
    )
    one_spin, two_spin = spinorb_from_spatial(
        np.asarray(one_body, dtype=float), two_body_openfermion
    )
    hamiltonian = jordan_wigner(
        get_fermion_operator(InteractionOperator(core_energy, one_spin, 0.5 * two_spin))
    )
    hamiltonian.compress(abs_tol=1.0e-12)

    n_qubits = 2 * norb
    matrix = get_sparse_operator(hamiltonian, n_qubits=n_qubits)

    occupied = hartree_fock_occupation(nelec)
    hf_vector = jw_configuration_state(occupied, n_qubits)
    hf_energy = float(np.real(np.vdot(hf_vector, matrix @ hf_vector)))

    number_matrix = get_sparse_operator(
        electron_number_operator(n_qubits), n_qubits=n_qubits
    )
    diagonal = np.real(np.asarray(number_matrix.diagonal()).reshape(-1))
    sector = np.flatnonzero(np.isclose(diagonal, nelec, atol=1.0e-9))
    sector_matrix = matrix[sector][:, sector]
    exact = float(
        np.real(eigsh(sector_matrix, k=1, which="SA", return_eigenvectors=False)[0])
    )

    metadata = {
        "molecule": f"Synthetic ({nelec}e,{norb}o) self-test Hamiltonian",
        "basis": "synthetic",
        "charge": 0,
        "spin_2s": 0,
        "model_note": (
            "Randomly generated one- and two-body integrals with full eight-fold "
            "permutational symmetry, mapped through the same OpenFermion path as "
            "the real workflow. Physically meaningless; structurally identical."
        ),
        "seed": seed,
        "qubits": n_qubits,
        "active_spatial_orbitals": norb,
        "active_electrons": nelec,
        "core_energy_hartree": core_energy,
        "hartree_fock_energy_hartree": hf_energy,
        "reference_energy_hartree": exact,
        "reference_method": "exact sector diagonalization",
        "pauli_terms": len(hamiltonian.terms),
        "truncation_l1_bound_hartree": 0.0,
    }
    return ChemistryResult(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        n_active_electrons=nelec,
        hartree_fock_energy=hf_energy,
        reference_energy=exact,
        reference_method="exact sector diagonalization",
        core_energy=core_energy,
        metadata=metadata,
    )


def _adequate_number_penalty(
    chemistry: ChemistryResult, candidates: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
) -> tuple[float, float]:
    """Smallest penalty whose exact ground state actually sits in the N-electron sector.

    A hardware-efficient ansatz is not particle-number conserving, so the only
    thing stopping the optimizer from wandering into a lower-lying sector is the
    penalty. If the penalised *exact* ground state is already outside the target
    sector, no VQE run against it can mean anything -- it will faithfully find
    the wrong answer. `validate` applies this same test to the real molecule,
    which is why its README tells you to retry with `--number-penalty 4`.

    Sparse rather than dense: only the ground state is wanted, and a dense eigh
    of the 12-qubit space is 4096^3 flops per candidate.
    """
    from openfermion import get_sparse_operator
    from scipy.sparse.linalg import eigsh

    n_qubits = chemistry.n_qubits
    target = chemistry.n_active_electrons
    hamiltonian = get_sparse_operator(chemistry.hamiltonian, n_qubits=n_qubits)
    number = get_sparse_operator(electron_number_operator(n_qubits), n_qubits=n_qubits)
    deviation = get_sparse_operator(
        number_deviation_operator(n_qubits, target), n_qubits=n_qubits
    )
    for penalty in candidates:
        _, vectors = eigsh(hamiltonian + penalty * deviation, k=1, which="SA")
        ground = vectors[:, 0]
        electrons = float(np.real(np.vdot(ground, number @ ground)))
        if abs(electrons - target) < 1.0e-8:
            return penalty, electrons
    raise RuntimeError(
        f"No penalty in {candidates} selects the {target}-electron sector."
    )


def _check(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:<44} {detail}", flush=True)


def selftest_main(spec: MoleculeSpec) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the local Qiskit Aer stack without PySCF or a cache"
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument(
        "--optimizer-tolerance",
        type=float,
        default=1.0e-8,
        help="Looser than the VQE's 1e-10 on purpose: chasing the last few "
        "digits costs most of the iterations and proves nothing extra about "
        "the wiring, which is all this command is checking.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--number-penalty", type=float, default=1.0)
    parser.add_argument(
        "--start-noise",
        type=float,
        default=0.05,
        help="Standard deviation of the kick added to the theta = 0 start, "
        "matching the VQE default. theta = 0 is the right *reference* but a bad "
        "*start*: H conserves particle number, so a lone RY excitation out of the "
        "reference determinant has exactly zero gradient there. The sparser the "
        "sector, the more of the ansatz that freezes -- at 2 electrons in 10 spin "
        "orbitals it is the whole circuit, and the optimizer returns after zero "
        "iterations. Set 0 to reproduce that on purpose.",
    )
    parser.add_argument(
        "--vqe-method",
        choices=("statevector", "matrix_product_state"),
        default="statevector",
        help="Aer method for the closing VQE only. Statevector by default: an "
        "expectation value over ~900 Pauli terms costs about five times more on "
        "MPS, and what step 5 has to prove -- that the optimizer loop, the "
        "batched gradients and the variational bound are wired correctly -- is "
        "method-independent. MPS is still checked against statevector on both "
        "energies (step 3) and gradients (step 4), which is where a method "
        "difference would actually show up.",
    )
    parser.add_argument(
        "--check-penalty-sector",
        action="store_true",
        help="Find the smallest penalty whose exact penalised ground state still "
        "sits in the target electron sector, and raise --number-penalty to it. "
        "Costs one sparse ground-state solve per candidate.",
    )
    parser.add_argument(
        "--write-cache",
        type=str,
        default=None,
        help="Also write the synthetic Hamiltonian as a schema-compatible cache.",
    )
    args = parser.parse_args()

    norb = int(spec.n_active_orbitals)
    nelec = int(spec.n_active_electrons)

    from openfermion import get_sparse_operator

    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    started = time.perf_counter()
    results: list[tuple[str, bool, str]] = []

    print("=" * 72)
    print("Local Qiskit Aer self-test (no PySCF, no cached molecule)")
    print("=" * 72)

    chemistry = _synthetic_chemistry(args.seed, norb, nelec)
    n_qubits = chemistry.n_qubits
    print(f"  Shadowing                   : {spec.name}")
    print(f"  Synthetic system            : {n_qubits} qubits, "
          f"{len(chemistry.hamiltonian.terms)} Pauli terms")
    print(f"  Hartree-Fock / exact        : {chemistry.hartree_fock_energy:.12f} / "
          f"{chemistry.reference_energy:.12f} Ha")
    print("-" * 72)

    if args.write_cache:
        save_chemistry(chemistry, Path(args.write_cache))
        print(f"  Wrote synthetic cache: {args.write_cache}")

    # ---------------------------------------------------------------- 0
    # Before anything is measured against this Hamiltonian, make sure the
    # penalised problem still has its ground state where the chemistry says it
    # should. A random synthetic Hamiltonian is under no obligation to accept the
    # same penalty the real molecule does, so the adequate value is discovered
    # and adopted rather than assumed -- otherwise the VQE below would be scored
    # against a sector it was never constrained to stay in.
    if args.check_penalty_sector:
        penalty, electrons = _adequate_number_penalty(chemistry)
        args.number_penalty = max(args.number_penalty, penalty)
        _check(results, "a penalty selects the target sector", True,
               f"smallest adequate = {penalty} Ha (using {args.number_penalty}), "
               f"<N> = {electrons:.10f}")

    # ---------------------------------------------------------------- 1
    # Operator conversion. The spectral check catches wrong coefficients; the
    # determinant check catches wrong qubit ordering, which a spectral check
    # cannot see because bit-reversal is a similarity transform.
    qiskit_operator = to_sparse_pauli_op(chemistry.hamiltonian, n_qubits)
    openfermion_matrix = get_sparse_operator(chemistry.hamiltonian, n_qubits=n_qubits)
    spectrum_error = float(
        np.max(
            np.abs(
                np.sort(np.linalg.eigvalsh(openfermion_matrix.toarray()))
                - np.sort(np.linalg.eigvalsh(qiskit_operator.to_matrix()))
            )
        )
    )
    _check(results, "operator conversion: spectra agree", spectrum_error < 1.0e-9,
           f"max |dE| = {spectrum_error:.3e} Ha")

    evaluator = AerEvaluator(n_qubits, args.layers, method="matrix_product_state",
                             seed=args.seed)
    exact_evaluator = AerEvaluator(n_qubits, args.layers, method="statevector",
                                   seed=args.seed)

    determinant_error = 0.0
    rng = np.random.default_rng(args.seed + 1)
    for _ in range(6):
        occupation = sorted(
            rng.choice(n_qubits, size=nelec, replace=False).tolist()
        )
        vector = jw_configuration_state(occupation, n_qubits)
        openfermion_value = float(np.real(np.vdot(vector, openfermion_matrix @ vector)))
        # Same determinant in Qiskit: X on exactly those qubit indices, which is
        # the frame rotation trick applied to an arbitrary occupation.
        rotated = to_sparse_pauli_op(
            conjugate_with_x(chemistry.hamiltonian, occupation), n_qubits
        )
        qiskit_value = exact_evaluator.energy(np.zeros(exact_evaluator.n_parameters), rotated)
        determinant_error = max(determinant_error, abs(openfermion_value - qiskit_value))
    _check(results, "qubit ordering: determinants agree", determinant_error < 1.0e-10,
           f"max |dE| = {determinant_error:.3e} Ha over 6 determinants")

    # ---------------------------------------------------------------- 2
    operators = build_objective(chemistry, args.number_penalty, 0.0)
    zeros = np.zeros(evaluator.n_parameters)
    frame_energy = evaluator.energy(zeros, operators["hamiltonian"])
    frame_error = frame_energy - chemistry.hartree_fock_energy
    _check(results, "theta=0 is the Hartree-Fock determinant", abs(frame_error) < 1.0e-10,
           f"error = {frame_error:+.3e} Ha")

    objective_at_zero = evaluator.energy(zeros, operators["objective"])
    penalty_at_zero = objective_at_zero - frame_energy
    _check(results, "penalty vanishes at the reference state", abs(penalty_at_zero) < 1.0e-10,
           f"penalty = {penalty_at_zero:+.3e} Ha")

    electrons_at_zero = evaluator.energy(zeros, operators["number"])
    _check(results, "theta=0 has the target electron number",
           abs(electrons_at_zero - nelec) < 1.0e-10,
           f"<N> = {electrons_at_zero:.10f}")

    # ---------------------------------------------------------------- 3
    angles = rng.uniform(-np.pi, np.pi, evaluator.n_parameters)
    mps_energy = evaluator.energy(angles, operators["objective"])
    statevector_energy = exact_evaluator.energy(angles, operators["objective"])
    mps_error = mps_energy - statevector_energy
    _check(results, f"MPS agrees with statevector (thr {DEFAULT_TRUNCATION_THRESHOLD:.0e})",
           abs(mps_error) < 1.0e-9, f"difference = {mps_error:+.3e} Ha")

    # ---------------------------------------------------------------- 4
    _check(results, "parameter-shift preconditions hold", evaluator.exact_gradient,
           f"{evaluator.n_parameters} RY parameters, each used once")
    _, shift_gradient = evaluator.energy_and_gradient(
        angles, operators["objective"], finite_difference=False
    )
    _, difference_gradient = exact_evaluator.energy_and_gradient(
        angles, operators["objective"], finite_difference=True
    )
    gradient_error = float(np.max(np.abs(shift_gradient - difference_gradient)))
    _check(results, "parameter-shift == finite difference", gradient_error < 1.0e-5,
           f"max |dG| = {gradient_error:.3e}")

    # ---------------------------------------------------------------- 5
    vqe_evaluator = (
        evaluator
        if args.vqe_method == "matrix_product_state"
        else AerEvaluator(n_qubits, args.layers, method=args.vqe_method, seed=args.seed)
    )

    # How much of the ansatz is actually alive at the reference determinant.
    # This is a diagnostic, not a pass/fail: a frozen start is a property of the
    # sector, and the kick below is the workflow's answer to it.
    _, zero_gradient = vqe_evaluator.energy_and_gradient(
        zeros, operators["objective"], finite_difference=False
    )
    live = int(np.count_nonzero(np.abs(zero_gradient) > 1.0e-9))
    if args.start_noise > 0:
        start = np.random.default_rng(args.seed + 2).normal(
            0.0, args.start_noise, vqe_evaluator.n_parameters
        )
        start_description = f"theta=0 + N(0, {args.start_noise}) kick"
    else:
        start = zeros
        start_description = "theta=0 exactly (Hartree-Fock)"

    # The objective at the starting point. A minimizer is allowed to fail to
    # find anything good; it is never allowed to return something worse than
    # where it began, so this is the bound step 5 can actually enforce.
    start_objective = vqe_evaluator.energy(start, operators["objective"])

    print("-" * 72)
    print(f"  Live parameters at theta=0  : {live}/{vqe_evaluator.n_parameters} "
          f"(|grad| = {np.linalg.norm(zero_gradient):.3e})")
    print(f"  Running VQE: {args.layers} layers, {vqe_evaluator.n_parameters} parameters, "
          f"L-BFGS-B, maxiter {args.maxiter}, tol {args.optimizer_tolerance:g}, "
          f"{args.vqe_method}, start {start_description} ...", flush=True)
    vqe_started = time.perf_counter()
    objective_value, parameters, diagnostics = _minimize(
        vqe_evaluator,
        operators["objective"],
        start,
        "l-bfgs-b",
        args.optimizer_tolerance,
        args.maxiter,
        False,
    )
    vqe_elapsed = time.perf_counter() - vqe_started

    physical_energy, electron_count, leakage, spin_squared = vqe_evaluator.run(
        [
            (parameters, operators["hamiltonian"]),
            (parameters, operators["number"]),
            (parameters, operators["deviation"]),
            (parameters, operators["spin_squared"]),
        ]
    )
    error_mha = 1000.0 * (physical_energy - chemistry.reference_energy)
    contamination = args.number_penalty * leakage + abs(spin_squared)
    available_mha = 1000.0 * (chemistry.hartree_fock_energy - chemistry.reference_energy)
    recovered_mha = 1000.0 * (chemistry.hartree_fock_energy - physical_energy)

    print(f"  VQE objective / physical    : {objective_value:.12f} / {physical_energy:.12f} Ha")
    print(f"  <N> / leakage / <S^2>       : {electron_count:.8f} / {leakage:.3e} / {spin_squared:.3e}")
    print(f"  Error vs exact              : {error_mha:+.6f} mHa")
    print(f"  Correlation recovered       : {recovered_mha:+.3f} of {available_mha:.3f} mHa "
          f"({100.0 * recovered_mha / available_mha:.1f}%)")
    print(f"  VQE wall time               : {vqe_elapsed:.2f} s "
          f"({vqe_evaluator.circuit_count} circuits in "
          f"{vqe_evaluator.batch_count} Aer batches)")
    print("-" * 72)

    # The variational bound is the one statement that must hold exactly: a
    # penalised energy minus its own penalty cannot dip below the sector ground
    # state by more than the leak it paid for.
    _check(results, "VQE respects the variational bound",
           physical_energy >= chemistry.reference_energy - max(contamination, 1.0e-9),
           f"E_vqe - E_exact = {error_mha:+.6f} mHa")
    # Not "improves on Hartree-Fock": on a random Hamiltonian in a sparse sector
    # the ansatz may genuinely be unable to, and see the module docstring. What
    # a minimizer must always do is not move uphill from its own start.
    _check(results, "VQE does not return a worse point than its start",
           objective_value <= start_objective + 1.0e-9,
           f"E_start - E_final = {1000.0 * (start_objective - objective_value):+.6f} mHa")
    _check(results, "optimizer terminated cleanly", diagnostics["converged"],
           f"{diagnostics['iterations']} iterations, {diagnostics['message']}")
    if physical_energy >= chemistry.hartree_fock_energy:
        print("  NOTE: the ansatz did not beat Hartree-Fock on this random "
              "Hamiltonian. Expected at sparse electron counts; see the module "
              "docstring. It says nothing about the stack, which the checks "
              "above cover.")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("=" * 72)
    print(f"{passed}/{total} checks passed in {time.perf_counter() - started:.2f} s")
    if passed == total:
        print("SELF-TEST PASSED: the local Qiskit Aer stack is correct.")
        print("The only thing still needed is the LiH Hamiltonian cache.")
    else:
        print("SELF-TEST FAILED: do not run the molecule until this is green.")
    print("=" * 72)
    return 0 if passed == total else 1
