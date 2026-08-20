#!/usr/bin/env python3
"""Exact, classical validation for the cached 12-qubit Hamiltonian.

Everything here is done by dense linear algebra on the full 2^n Hilbert space,
which is affordable to 14 qubits and is the point of the exercise: the VQE
result is only meaningful relative to a reference that was computed a completely
different way. A validation that used the same approximations as the thing it
validates would prove nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from hamiltonian import (
    COMPRESSION_TOLERANCE,
    RECEIPT_SCHEMA,
    SPIN_CONTAMINATION_TOLERANCE,
    MoleculeSpec,
    add_electron_number_penalty,
    electron_number_operator,
    file_fingerprint,
    hartree_fock_occupation,
    load_cache,
    load_validation_receipt,
    penalty_key,
    physics_fingerprint,
    receipt_is_current,
    spec_fingerprint,
    spin_squared_qubit_operator,
    to_hartree_fock_frame,
    validation_receipt_path,
    workflow_fingerprint,
    write_json_atomic,
)

# Above this the dense path stops being affordable and the whole exact-reference
# premise of this module stops holding.
MAX_EXACT_QUBITS = 14


def _dense_hermitian(sparse_matrix: Any) -> np.ndarray:
    """Dense Hermitian array, real when the operator is real.

    The Jordan-Wigner image of a real molecular Hamiltonian has no imaginary
    part -- terms with an odd number of Y factors carry zero coefficient -- so
    the real path is the normal one and halves both memory and eigensolver time.
    """
    dense = np.asarray(sparse_matrix.todense())
    if np.max(np.abs(dense.imag)) < 1.0e-14:
        return np.ascontiguousarray(dense.real)
    return dense


def _ground_state(dense: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Lowest eigenpair plus the full spectrum, by direct diagonalization.

    Deliberately dense rather than ARPACK. A large number penalty is designed to
    push whole particle-number sectors up into a dense band above the ground
    state, which is precisely the spectral profile Lanczos converges slowly on:
    `eigsh` can raise ArpackNoConvergence on exactly the input this check exists
    to examine. At 12 qubits the matrix is 4096x4096 and direct diagonalization
    is seconds, deterministic, and hands back the whole spectrum for free.
    """
    values, vectors = np.linalg.eigh(dense)
    return float(values[0]), np.asarray(vectors[:, 0]), np.asarray(values)


def _check_one_penalty(
    hamiltonian: Any,
    nqubits: int,
    n_active_electrons: int,
    reference_energy: float,
    truncation_bound: float,
    number_penalty: float,
    spin_penalty: float,
    spin_operator: Any,
    nmat: Any,
    spin_mat: Any,
    get_sparse_operator: Any,
) -> dict[str, Any]:
    """Does this penalty setting put the target sector at the global minimum?"""
    if number_penalty <= 0 and spin_penalty <= 0:
        # Nothing is being constrained, so there is nothing to prove about the
        # penalized spectrum. The unpenalized Hamiltonian's global ground state
        # may legitimately sit in another sector; what makes a penalty-free run
        # interpretable is the sector energy bound, which validate_main proves
        # separately and unconditionally.
        return {
            "number_penalty_hartree": number_penalty,
            "spin_penalty_hartree": spin_penalty,
            "penalty_ground_electrons": float(n_active_electrons),
            "penalty_ground_spin_squared": 0.0,
            "penalty_ground_error_hartree": 0.0,
            "selects_target_sector": True,
            "unconstrained": True,
            "checks": "all_passed",
        }

    constrained = hamiltonian
    if number_penalty > 0:
        constrained = add_electron_number_penalty(
            constrained, nqubits, n_active_electrons, number_penalty
        )
    if spin_penalty > 0:
        constrained = constrained + spin_penalty * spin_operator
        constrained.compress(abs_tol=COMPRESSION_TOLERANCE)
    dense = _dense_hermitian(get_sparse_operator(constrained, n_qubits=nqubits))
    ground, vector, spectrum = _ground_state(dense)
    electrons = float(np.real(np.vdot(vector, nmat @ vector)))
    spin_squared = float(np.real(np.vdot(vector, spin_mat @ vector)))
    error = ground - reference_energy
    ok = (
        abs(electrons - n_active_electrons) < 1.0e-6
        and abs(spin_squared) < SPIN_CONTAMINATION_TOLERANCE
        and abs(error) < 1.0e-6 + truncation_bound
    )
    return {
        "number_penalty_hartree": number_penalty,
        "spin_penalty_hartree": spin_penalty,
        "penalty_ground_electrons": electrons,
        "penalty_ground_spin_squared": spin_squared,
        "penalty_ground_error_hartree": error,
        # How much room the penalty has before a wrong-sector state would
        # undercut the right one. A small gap means raise the penalty.
        "penalty_sector_gap_hartree": float(spectrum[1] - spectrum[0])
        if spectrum.size > 1
        else float("inf"),
        "selects_target_sector": bool(ok),
        "unconstrained": False,
        "checks": "all_passed" if ok else "failed",
    }


def validate_main(spec: MoleculeSpec, default_cache: str) -> int:
    """Prove the cached qubit Hamiltonian reproduces the classical references."""
    parser = argparse.ArgumentParser(description="Validate a cached qubit Hamiltonian")
    parser.add_argument("--cache", type=Path, default=Path(default_cache))
    parser.add_argument(
        "--number-penalty",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 4.0],
        help="Validate these number-penalty values and record them all. The "
        "defaults pre-authorise every remedy the README recommends, so a VQE "
        "that turns out to need a different penalty does not have to leave the "
        "allocation to get one validated.",
    )
    parser.add_argument(
        "--spin-penalty",
        type=float,
        nargs="+",
        default=[0.0],
        help="Validate these spin-penalty values against every number penalty. "
        "Defaults to 0 to match the VQE: with no spin penalty the reported "
        "<S^2> proves the target sector's ground state is already a singlet.",
    )
    args = parser.parse_args()
    if any(value < 0 for value in args.number_penalty + args.spin_penalty):
        parser.error("penalties must be non-negative")
    if not args.cache.is_file():
        raise FileNotFoundError(
            f"Missing {args.cache}. Run `python run.py prepare` first."
        )

    from openfermion import get_sparse_operator
    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    result = load_cache(args.cache, spec)
    nqubits = result.n_qubits
    if nqubits > MAX_EXACT_QUBITS:
        raise RuntimeError(
            f"Exact validation is disabled above {MAX_EXACT_QUBITS} qubits"
        )
    hmat = get_sparse_operator(result.hamiltonian, n_qubits=nqubits)
    noperator = electron_number_operator(nqubits)
    nmat = get_sparse_operator(noperator, n_qubits=nqubits)
    hermitian = hmat - hmat.conj().T
    hermitian_error = float(np.max(np.abs(hermitian.data))) if hermitian.nnz else 0.0
    commutator = hmat @ nmat - nmat @ hmat
    commutator_error = float(np.max(np.abs(commutator.data))) if commutator.nnz else 0.0

    occupied = hartree_fock_occupation(result.n_active_electrons)
    hf_state = jw_configuration_state(occupied, nqubits)
    hf_qubit = float(np.real(np.vdot(hf_state, hmat @ hf_state)))
    hf_error = hf_qubit - result.hartree_fock_energy

    ndiag = np.real(np.asarray(nmat.diagonal()).reshape(-1))
    sector = np.flatnonzero(
        np.isclose(ndiag, result.n_active_electrons, atol=1.0e-9)
    )
    sector_dense = _dense_hermitian(hmat[sector][:, sector])
    spin_operator = spin_squared_qubit_operator(nqubits)
    spin_mat = get_sparse_operator(spin_operator, n_qubits=nqubits)
    sector_spin_mat = spin_mat[sector][:, sector]

    # One diagonalization gives both the exact reference and the <S^2> of the
    # state that realizes it. The latter is what justifies leaving the spin
    # penalty off: if the lowest state in the six-electron sector is already a
    # singlet, adding a few hundred Pauli terms to every energy evaluation
    # guards nothing.
    if sector_dense.shape[0] == 1:
        exact = float(np.real(sector_dense[0, 0]))
        sector_spin_squared = float(np.real(sector_spin_mat[0, 0]))
    else:
        exact, sector_vector, _ = _ground_state(sector_dense)
        sector_spin_squared = float(
            np.real(np.vdot(sector_vector, sector_spin_mat @ sector_vector))
        )
    reference_error = exact - result.reference_energy

    rotated = get_sparse_operator(
        to_hartree_fock_frame(result.hamiltonian, result.n_active_electrons),
        n_qubits=nqubits,
    )
    vacuum_energy = float(np.real(rotated[0, 0]))
    frame_error = vacuum_energy - result.hartree_fock_energy
    truncation_bound = float(result.metadata["truncation_l1_bound_hartree"])

    penalty_entries: dict[str, dict[str, Any]] = {}
    for number_penalty in args.number_penalty:
        for spin_penalty in args.spin_penalty:
            entry = _check_one_penalty(
                result.hamiltonian,
                nqubits,
                result.n_active_electrons,
                result.reference_energy,
                truncation_bound,
                float(number_penalty),
                float(spin_penalty),
                spin_operator,
                nmat,
                spin_mat,
                get_sparse_operator,
            )
            entry["sector_ground_energy_hartree"] = exact
            penalty_entries[penalty_key(number_penalty, spin_penalty)] = entry

    structural = (
        hermitian_error < 1.0e-10
        and commutator_error < 1.0e-10
        and abs(hf_error) < 1.0e-7 + truncation_bound
        and abs(reference_error) < 1.0e-6 + truncation_bound
        and abs(frame_error) < 1.0e-7 + truncation_bound
        and abs(sector_spin_squared) < SPIN_CONTAMINATION_TOLERANCE
        and truncation_bound <= 1.0e-4
    )
    usable = {
        key: entry
        for key, entry in penalty_entries.items()
        if entry["checks"] == "all_passed"
    }
    passed = structural and bool(usable)

    print(f"System                    : {result.metadata['molecule']}")
    print(f"Active space              : {result.n_active_electrons}e, {nqubits // 2}o")
    print(f"Qubits / Pauli terms      : {nqubits} / {len(result.hamiltonian.terms)}")
    print(f"Hermiticity residual      : {hermitian_error:.3e}")
    print(f"[H,N] residual            : {commutator_error:.3e}")
    print(f"HF mapping error          : {hf_error:+.3e} Ha")
    print(f"CASCI mapping error       : {reference_error:+.3e} Ha")
    print(f"HF-frame error            : {frame_error:+.3e} Ha")
    print(f"Sector ground energy      : {exact:.12f} Ha")
    print(f"Sector ground <S^2>       : {sector_spin_squared:.3e}")
    print(f"Truncation L1 bound       : {1000.0 * truncation_bound:.6f} mHa")
    print()
    print("  number  spin    <N>            <S^2>        error (Ha)     gap (Ha)   verdict")
    print("  " + "-" * 78)
    for key in sorted(penalty_entries, key=lambda name: penalty_entries[name]["number_penalty_hartree"]):
        entry = penalty_entries[key]
        gap = entry.get("penalty_sector_gap_hartree")
        gap_text = "     n/a" if gap is None or entry["unconstrained"] else f"{gap:9.4f}"
        print(
            f"  {entry['number_penalty_hartree']:6.3g}"
            f"  {entry['spin_penalty_hartree']:6.3g}"
            f"  {entry['penalty_ground_electrons']:12.8f}"
            f"  {entry['penalty_ground_spin_squared']:11.3e}"
            f"  {entry['penalty_ground_error_hartree']:+12.3e}"
            f"  {gap_text}"
            f"   {'ok' if entry['checks'] == 'all_passed' else 'REJECTED'}"
            + ("  (unconstrained)" if entry["unconstrained"] else "")
        )
    print()

    if abs(sector_spin_squared) >= SPIN_CONTAMINATION_TOLERANCE:
        print(
            f"DIAGNOSTIC: the lowest state in the {result.n_active_electrons}-electron "
            f"sector has <S^2> = {sector_spin_squared:.4f}, not 0. The closed-shell "
            "singlet premise of this workflow does not hold for this active space -- "
            "run the VQE with --spin-penalty 1 and re-validate, or choose an active "
            "space whose ground state is a singlet."
        )
    rejected = [key for key, entry in penalty_entries.items() if entry["checks"] != "all_passed"]
    for key in rejected:
        entry = penalty_entries[key]
        if abs(entry["penalty_ground_electrons"] - result.n_active_electrons) >= 1.0e-6:
            print(
                f"DIAGNOSTIC: {key} does not select the target sector "
                f"(<N> = {entry['penalty_ground_electrons']:.6f}). Use a larger "
                "number penalty."
            )
    if not usable:
        print(
            "DIAGNOSTIC: no requested penalty setting was accepted, so the VQE has "
            "nothing to run against. Re-run validate with larger --number-penalty "
            "values, e.g. `--number-penalty 0 1 4 16`."
        )

    print("ALL CHECKS PASSED" if passed else "CHECKS FAILED")
    if not passed:
        return 1

    # Merge rather than overwrite: a receipt that still describes this exact
    # Hamiltonian and physics code accumulates penalty settings across runs, so
    # validating one more penalty never silently revokes the others.
    receipt_path = validation_receipt_path(args.cache)
    existing = load_validation_receipt(args.cache)
    validated = (
        dict(existing.get("validated_penalties", {}))
        if receipt_is_current(existing, args.cache, spec)
        else {}
    )
    validated.update(usable)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "cache_sha256": file_fingerprint(args.cache),
        "spec_sha256": spec_fingerprint(spec),
        # The gate. Narrow by design -- see hamiltonian.physics_fingerprint.
        "physics_sha256": physics_fingerprint(),
        # Provenance only; deliberately not part of the gate.
        "workflow_sha256": workflow_fingerprint(),
        "sector_ground_energy_hartree": exact,
        "sector_ground_spin_squared": sector_spin_squared,
        "hartree_fock_energy_hartree": result.hartree_fock_energy,
        "truncation_l1_bound_hartree": truncation_bound,
        "validated_penalties": validated,
        "checks": "all_passed",
    }
    write_json_atomic(receipt_path, receipt)
    print(f"Validation receipt written: {receipt_path}")
    print(f"Validated penalty settings: {', '.join(sorted(validated))}")
    return 0
