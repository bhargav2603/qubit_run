#!/usr/bin/env python3
"""End-to-end proof of the local Qiskit stack, with no PySCF and no cache.

PySCF publishes no Windows wheel, so on Windows the real Hamiltonian has to be
built elsewhere and copied in. That leaves a gap: nothing local would be
exercised until the cache arrives, and a mapping or ordering bug would surface
as a wrong number rather than a failed test.

This closes that gap. It builds a random but *structurally real* CAS(4e,4o)
Hamiltonian -- through the identical OpenFermion path `chemistry.py` uses, so it
is Hermitian, particle-number conserving and has a genuine four-electron ground
state -- and checks every local component against exact classical answers.

The check that matters most is **spin-orbital ordering**. OpenFermion emits
interleaved spin orbitals (2p = alpha_p, 2p+1 = beta_p); qiskit-nature's
JordanWignerMapper expects them blocked (all alpha, then all beta). Getting that
permutation wrong yields a Hamiltonian with an *identical spectrum* and
completely wrong energies for every determinant, so a spectral comparison cannot
detect it. It is therefore checked determinant by determinant, in both
conventions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chemistry import (
    ChemistryResult,
    closed_shell_determinant_energy,
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
    interleaved_to_blocked,
    to_sparse_pauli_op,
)


N_ACTIVE_ORBITALS = 4
N_ACTIVE_ELECTRONS = 4


def _synthetic_chemistry(seed: int) -> ChemistryResult:
    """A random CAS(4e,4o) Hamiltonian built exactly the way the real one is."""
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

    norb = N_ACTIVE_ORBITALS
    rng = np.random.default_rng(seed)

    one_body = rng.normal(0.0, 0.15, (norb, norb))
    one_body = 0.5 * (one_body + one_body.T)
    one_body -= np.diag(np.linspace(3.0, 0.5, norb))

    # Chemist-notation (ij|kl) with full eight-fold symmetry, matching what
    # `ao2mo.restore(1, ...)` hands back in chemistry.py.
    two_body = rng.normal(0.0, 0.12, (norb, norb, norb, norb))
    two_body = two_body + two_body.transpose(1, 0, 2, 3)
    two_body = two_body + two_body.transpose(0, 1, 3, 2)
    two_body = two_body + two_body.transpose(2, 3, 0, 1)
    two_body /= 8.0

    core_energy = -12.5
    determinant_energy = closed_shell_determinant_energy(
        one_body, two_body, core_energy, N_ACTIVE_ELECTRONS
    )

    one_spin, two_spin = spinorb_from_spatial(
        one_body, np.asarray(two_body.transpose(0, 2, 3, 1), dtype=float, order="C")
    )
    hamiltonian = jordan_wigner(
        get_fermion_operator(InteractionOperator(core_energy, one_spin, 0.5 * two_spin))
    )
    hamiltonian.compress(abs_tol=1.0e-12)

    n_qubits = 2 * norb
    matrix = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    hf_vector = jw_configuration_state(hartree_fock_occupation(N_ACTIVE_ELECTRONS), n_qubits)
    hf_energy = float(np.real(np.vdot(hf_vector, matrix @ hf_vector)))

    number = get_sparse_operator(electron_number_operator(n_qubits), n_qubits=n_qubits)
    diagonal = np.real(np.asarray(number.diagonal()).reshape(-1))
    sector = np.flatnonzero(np.isclose(diagonal, N_ACTIVE_ELECTRONS, atol=1.0e-9))
    exact = float(
        np.real(eigsh(matrix[sector][:, sector], k=1, which="SA", return_eigenvectors=False)[0])
    )

    metadata = {
        "molecule": "Synthetic CAS(4e,4o) self-test Hamiltonian",
        "basis": "synthetic",
        "orbital_selection": "synthetic",
        "solvation_model": "none",
        "seed": seed,
        "qubits": n_qubits,
        "active_electrons": N_ACTIVE_ELECTRONS,
        "reference_energy_hartree": exact,
        "reference_determinant_energy_hartree": determinant_energy,
        "pauli_terms": len(hamiltonian.terms),
        "truncation_l1_bound_hartree": 0.0,
    }
    return ChemistryResult(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        n_active_electrons=N_ACTIVE_ELECTRONS,
        hartree_fock_energy=hf_energy,
        reference_energy=exact,
        reference_method="exact sector diagonalization",
        core_energy=core_energy,
        metadata=metadata,
        reference_determinant_energy=determinant_energy,
    )


def _check(results: list, name: str, passed: bool, detail: str) -> None:
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:<46} {detail}", flush=True)


def _basis_state_energy(occupied, n_qubits: int, operator) -> float:
    """<occ|O|occ> for a computational basis state, by qubit index."""
    from qiskit.quantum_info import Statevector

    index = sum(1 << int(q) for q in occupied)
    state = Statevector.from_int(index, dims=(2,) * n_qubits)
    return float(np.real(state.expectation_value(operator)))


def selftest_main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the local Qiskit Aer stack without PySCF or a cache"
    )
    parser.add_argument("--ansatz", choices=("uccsd", "hea"), default="uccsd")
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--method", default="matrix_product_state")
    parser.add_argument("--write-cache", type=str, default=None)
    args = parser.parse_args()

    from openfermion import get_sparse_operator

    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    started = time.perf_counter()
    results: list = []

    print("=" * 78)
    print("Local Qiskit Aer self-test (no PySCF, no cached molecule)")
    print("=" * 78)

    chemistry = _synthetic_chemistry(args.seed)
    n_qubits = chemistry.n_qubits
    print(f"  Synthetic system            : {n_qubits} qubits, "
          f"{len(chemistry.hamiltonian.terms)} Pauli terms")
    print(f"  Reference determinant       : {chemistry.reference_determinant_energy:.10f} Ha")
    print(f"  Exact (sector FCI)          : {chemistry.reference_energy:.10f} Ha")
    print(f"  Correlation energy          : "
          f"{1000 * (chemistry.reference_energy - chemistry.reference_determinant_energy):.3f} mHa")
    print("-" * 78)

    if args.write_cache:
        save_chemistry(chemistry, Path(args.write_cache))
        print(f"  Wrote synthetic cache: {args.write_cache}")

    # -- 1. the classical determinant-energy formula matches the qubit operator
    determinant_error = chemistry.hartree_fock_energy - chemistry.reference_determinant_energy
    _check(results, "determinant energy formula is correct", abs(determinant_error) < 1e-10,
           f"classical vs qubit: {determinant_error:+.3e} Ha")

    # -- 2. operator conversion preserves the spectrum
    openfermion_matrix = get_sparse_operator(chemistry.hamiltonian, n_qubits=n_qubits)
    interleaved_op = to_sparse_pauli_op(chemistry.hamiltonian, n_qubits)
    spectrum_error = float(np.max(np.abs(
        np.sort(np.linalg.eigvalsh(openfermion_matrix.toarray()))
        - np.sort(np.linalg.eigvalsh(interleaved_op.to_matrix()))
    )))
    _check(results, "operator conversion: spectra agree", spectrum_error < 1e-9,
           f"max |dE| = {spectrum_error:.3e} Ha")

    # -- 3/4. ordering, determinant by determinant, in BOTH conventions.
    #    A bit-permutation is a similarity transform, so it leaves the spectrum
    #    untouched -- only per-determinant energies expose it.
    blocked_map = interleaved_to_blocked(n_qubits)
    blocked_op = to_sparse_pauli_op(chemistry.hamiltonian, n_qubits, blocked_map)
    rng = np.random.default_rng(args.seed + 1)
    interleaved_error = blocked_error = 0.0
    for _ in range(8):
        occupation = sorted(rng.choice(n_qubits, size=N_ACTIVE_ELECTRONS, replace=False).tolist())
        vector = jw_configuration_state(occupation, n_qubits)
        expected = float(np.real(np.vdot(vector, openfermion_matrix @ vector)))
        interleaved_error = max(
            interleaved_error,
            abs(expected - _basis_state_energy(occupation, n_qubits, interleaved_op)),
        )
        blocked_error = max(
            blocked_error,
            abs(expected - _basis_state_energy(
                [blocked_map[q] for q in occupation], n_qubits, blocked_op)),
        )
    _check(results, "interleaved ordering: determinants agree", interleaved_error < 1e-10,
           f"max |dE| = {interleaved_error:.3e} Ha over 8 determinants")
    _check(results, "blocked ordering: determinants agree", blocked_error < 1e-10,
           f"max |dE| = {blocked_error:.3e} Ha over 8 determinants")

    # -- 5. the permutation is a real permutation, and it is not the identity
    _check(results, "blocked map is a non-trivial permutation",
           sorted(blocked_map) == list(range(n_qubits)) and blocked_map != list(range(n_qubits)),
           f"{blocked_map}")

    # -- 6. the ansatz
    evaluator = AerEvaluator(n_qubits, N_ACTIVE_ELECTRONS, args.ansatz,
                             method=args.method, seed=args.seed)
    exact_evaluator = AerEvaluator(n_qubits, N_ACTIVE_ELECTRONS, args.ansatz,
                                   method="statevector", seed=args.seed)
    print(f"  Ansatz                      : {args.ansatz.upper()}, "
          f"{evaluator.n_parameters} parameters, depth {evaluator.depth}, "
          f"{evaluator.two_qubit_gates} 2q gates")

    operators = build_objective(chemistry, args.ansatz, 0.0 if args.ansatz == "uccsd" else 2.0, 0.0)
    zeros = np.zeros(evaluator.n_parameters)

    start_energy = evaluator.energy(zeros, operators["hamiltonian"])
    start_error = start_energy - chemistry.reference_determinant_energy
    _check(results, "theta=0 is the reference determinant", abs(start_error) < 1e-9,
           f"error = {start_error:+.3e} Ha")

    electrons_at_zero = evaluator.energy(zeros, operators["number"])
    _check(results, "theta=0 has the target electron number",
           abs(electrons_at_zero - N_ACTIVE_ELECTRONS) < 1e-9,
           f"<N> = {electrons_at_zero:.10f}")

    # -- 7. symmetry conservation away from theta = 0. This is UCCSD's whole
    #    point: no penalty term is needed because it cannot leave the sector.
    angles = rng.normal(0.0, 0.3, evaluator.n_parameters)
    electrons, leak, spin = evaluator.run([
        (angles, operators["number"]),
        (angles, operators["deviation"]),
        (angles, operators["spin_squared"]),
    ])
    if args.ansatz == "uccsd":
        _check(results, "UCCSD conserves particle number exactly",
               abs(electrons - N_ACTIVE_ELECTRONS) < 1e-9 and abs(leak) < 1e-9,
               f"<N> = {electrons:.10f}, leak = {leak:.3e}")
        # NOT a bug, and not asserted: UCCSD's alpha->alpha and beta->beta
        # amplitudes are independent parameters, so at *arbitrary* angles the
        # state need not be a spin eigenstate. It becomes one at the variational
        # minimum of a spin-free Hamiltonian, which is where it is checked.
        print(f"  <S^2> at random angles      : {spin:.3e} "
              f"(UCCSD conserves N and S_z, not S^2, away from the optimum)")

    # -- 8. MPS vs exact statevector
    mps_energy = evaluator.energy(angles, operators["objective"])
    exact_energy = exact_evaluator.energy(angles, operators["objective"])
    _check(results, f"MPS agrees with statevector (thr {DEFAULT_TRUNCATION_THRESHOLD:.0e})",
           abs(mps_energy - exact_energy) < 1e-9,
           f"difference = {mps_energy - exact_energy:+.3e} Ha")

    # -- 9. the parameter-shift precondition, and the finite-difference gradient
    if args.ansatz == "uccsd":
        _check(results, "parameter-shift correctly rejected for UCCSD",
               not evaluator.exact_gradient,
               "parameters drive multiple rotations; finite differences required")
    _, grad_a = exact_evaluator.energy_and_gradient(angles, operators["objective"], "finite-difference")
    # A second, independent estimate: recompute two components by hand at a
    # different step size. Agreement rules out a stale-cache or indexing bug.
    manual = []
    for k in (0, evaluator.n_parameters // 2):
        h = 1e-4
        up, down = angles.copy(), angles.copy()
        up[k] += h
        down[k] -= h
        e_up, e_down = exact_evaluator.run([(up, operators["objective"]), (down, operators["objective"])])
        manual.append((e_up - e_down) / (2 * h))
    gradient_error = max(abs(manual[0] - grad_a[0]),
                         abs(manual[1] - grad_a[evaluator.n_parameters // 2]))
    _check(results, "finite-difference gradient is step-stable", gradient_error < 1e-5,
           f"max |dG| between step sizes = {gradient_error:.3e}")

    # -- 9b. the two engines must agree. The NumPy engine propagates the
    #        statevector itself and differentiates analytically, so nothing about
    #        it is shared with Aer; agreement is real independent evidence.
    from fastsim import NumpyEvaluator

    fast = NumpyEvaluator(n_qubits, N_ACTIVE_ELECTRONS, args.ansatz, k=2)
    _check(results, "engines expose the same parameter count",
           fast.n_parameters == evaluator.n_parameters,
           f"aer {evaluator.n_parameters}, numpy {fast.n_parameters}")

    engine_energy_error = engine_state_error = 0.0
    for probe in (zeros, angles):
        aer_state = np.asarray(exact_evaluator.statevector(probe))
        fast_state = fast.statevector(probe)
        engine_state_error = max(
            engine_state_error, abs(1.0 - abs(complex(np.vdot(aer_state, fast_state))))
        )
        engine_energy_error = max(
            engine_energy_error,
            abs(fast.energy(probe, operators["objective"])
                - exact_evaluator.energy(probe, operators["objective"])),
        )
    _check(results, "numpy engine reproduces Aer's state", engine_state_error < 1e-9,
           f"1 - |<aer|numpy>| = {engine_state_error:.3e}")
    _check(results, "numpy engine reproduces Aer's energy", engine_energy_error < 1e-9,
           f"max |dE| = {engine_energy_error:.3e} Ha")

    _, adjoint_gradient = fast.energy_and_gradient(angles, operators["objective"])
    adjoint_error = float(np.max(np.abs(adjoint_gradient - grad_a)))
    _check(results, "adjoint gradient matches finite differences", adjoint_error < 1e-5,
           f"max |dG| = {adjoint_error:.3e} (finite-difference step error dominates)")

    # -- 10. the VQE itself
    print("-" * 78)
    print(f"  Running VQE: L-BFGS-B, maxiter {args.maxiter} ...", flush=True)
    vqe_started = time.perf_counter()
    objective_value, parameters, diagnostics = _minimize(
        evaluator, operators["objective"], zeros, "l-bfgs-b", 1e-12, args.maxiter,
        "finite-difference",
    )
    vqe_elapsed = time.perf_counter() - vqe_started

    energy, electrons, leak, spin = evaluator.run([
        (parameters, operators["hamiltonian"]),
        (parameters, operators["number"]),
        (parameters, operators["deviation"]),
        (parameters, operators["spin_squared"]),
    ])
    error_mha = 1000.0 * (energy - chemistry.reference_energy)
    recovered = 100.0 * (chemistry.reference_determinant_energy - energy) / (
        chemistry.reference_determinant_energy - chemistry.reference_energy)

    print(f"  VQE energy / exact          : {energy:.10f} / {chemistry.reference_energy:.10f} Ha")
    print(f"  Error / correlation         : {error_mha:+.4f} mHa / {recovered:.2f} % recovered")
    print(f"  <N> / leak / <S^2>          : {electrons:.8f} / {leak:.2e} / {spin:.2e}")
    print(f"  Wall time                   : {vqe_elapsed:.1f} s "
          f"({evaluator.circuit_count} circuits, {evaluator.batch_count} batches, "
          f"{1000 * vqe_elapsed / max(evaluator.circuit_count, 1):.1f} ms/circuit)")
    print("-" * 78)

    _check(results, "VQE respects the variational bound",
           energy >= chemistry.reference_energy - max(leak, 1e-9),
           f"E_vqe - E_exact = {error_mha:+.4f} mHa")
    _check(results, "VQE improves on the reference determinant",
           energy < chemistry.reference_determinant_energy,
           f"recovered {recovered:.2f} % of the correlation energy")
    if args.ansatz == "uccsd":
        _check(results, "converged state is still in the target sector", leak < 1e-9,
               f"leak = {leak:.3e} (no penalty term was used)")
        # The spin-free Hamiltonian's ground state is a singlet, so a converged
        # variational state must be one too. This is the meaningful S^2 test.
        _check(results, "converged state is a singlet", abs(spin) < 1e-3,
               f"<S^2> = {spin:.3e} at the optimum")
    _check(results, "optimizer made real progress", diagnostics["iterations"] >= 3,
           f"{diagnostics['iterations']} iterations, {diagnostics['message'][:44]}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("=" * 78)
    print(f"{passed}/{total} checks passed in {time.perf_counter() - started:.1f} s")
    if passed == total:
        print("SELF-TEST PASSED: the local Qiskit Aer stack is correct.")
        print("The only thing still needed is the acrylamide Hamiltonian cache.")
    else:
        print("SELF-TEST FAILED: do not run the molecule until this is green.")
    print("=" * 78)
    return 0 if passed == total else 1
