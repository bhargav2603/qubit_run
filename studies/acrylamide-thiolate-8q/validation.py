#!/usr/bin/env python3
"""Exact, classical validation for the cached eight-qubit Hamiltonian."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chemistry import (
    COMPRESSION_TOLERANCE,
    SPIN_CONTAMINATION_TOLERANCE,
    MoleculeSpec,
    add_electron_number_penalty,
    electron_number_operator,
    file_fingerprint,
    hartree_fock_occupation,
    load_chemistry,
    spec_fingerprint,
    spin_squared_qubit_operator,
    to_hartree_fock_frame,
    validation_receipt_path,
    workflow_fingerprint,
    write_json_atomic,
)

def validate_main(spec: MoleculeSpec, default_cache: str) -> int:
    """Exact validation is safe for this workflow's eight-qubit Hamiltonian."""
    parser = argparse.ArgumentParser(description="Validate a cached qubit Hamiltonian")
    parser.add_argument("--cache", type=Path, default=Path(default_cache))
    parser.add_argument("--number-penalty", type=float, default=1.0)
    parser.add_argument(
        "--spin-penalty",
        type=float,
        default=0.0,
        help="Defaults to 0 to match the VQE. With no spin penalty the reported "
        "<S^2> proves the target sector's ground state is already a singlet.",
    )
    args = parser.parse_args()
    if args.number_penalty < 0 or args.spin_penalty < 0:
        parser.error("penalties must be non-negative")
    if not args.cache.is_file():
        raise FileNotFoundError(
            f"Missing {args.cache}. Run `python run.py prepare` first."
        )

    from openfermion import get_sparse_operator
    from scipy.sparse.linalg import eigsh
    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    result = load_chemistry(args.cache, spec)
    nqubits = result.n_qubits
    if nqubits > 16:
        raise RuntimeError("Exact sparse validation is disabled above 16 qubits")
    hmat = get_sparse_operator(result.hamiltonian, n_qubits=nqubits)
    noperator = electron_number_operator(nqubits)
    nmat = get_sparse_operator(noperator, n_qubits=nqubits)
    hermitian = hmat - hmat.getH()
    hermitian_error = float(np.max(np.abs(hermitian.data))) if hermitian.nnz else 0.0
    commutator = hmat @ nmat - nmat @ hmat
    commutator_error = float(np.max(np.abs(commutator.data))) if commutator.nnz else 0.0

    # The qubit Hamiltonian's reference determinant must reproduce the
    # determinant energy computed classically from the same active-space
    # integrals. With MP2 natural orbitals that is NOT the SCF energy -- the
    # lowest active determinant is no longer the SCF determinant -- so comparing
    # against hartree_fock_energy here would fail for a correct Hamiltonian.
    occupied = hartree_fock_occupation(result.n_active_electrons)
    hf_state = jw_configuration_state(occupied, nqubits)
    hf_qubit = float(np.real(np.vdot(hf_state, hmat @ hf_state)))
    hf_error = hf_qubit - result.reference_determinant_energy

    ndiag = np.real(np.asarray(nmat.diagonal()).reshape(-1))
    sector = np.flatnonzero(
        np.isclose(ndiag, result.n_active_electrons, atol=1.0e-9)
    )
    sector_h = hmat[sector][:, sector]
    if sector_h.shape[0] == 1:
        exact = float(np.real(sector_h[0, 0]))
    else:
        exact = float(np.real(eigsh(sector_h, k=1, which="SA", return_eigenvectors=False)[0]))
    reference_error = exact - result.reference_energy

    spin_operator = spin_squared_qubit_operator(nqubits)
    spin_mat = get_sparse_operator(spin_operator, n_qubits=nqubits)
    penalty_energy_error = 0.0
    penalty_electrons = float(result.n_active_electrons)
    penalty_spin_squared = 0.0
    if args.number_penalty > 0 or args.spin_penalty > 0:
        constrained = result.hamiltonian
        if args.number_penalty > 0:
            constrained = add_electron_number_penalty(
                constrained,
                nqubits,
                result.n_active_electrons,
                args.number_penalty,
            )
        if args.spin_penalty > 0:
            constrained = constrained + args.spin_penalty * spin_operator
            constrained.compress(abs_tol=COMPRESSION_TOLERANCE)
        penalized = get_sparse_operator(
            constrained,
            n_qubits=nqubits,
        )
        values, vectors = eigsh(penalized, k=1, which="SA")
        penalty_ground = float(np.real(values[0]))
        penalty_vector = vectors[:, 0]
        penalty_electrons = float(
            np.real(np.vdot(penalty_vector, nmat @ penalty_vector))
        )
        penalty_spin_squared = float(
            np.real(np.vdot(penalty_vector, spin_mat @ penalty_vector))
        )
        penalty_energy_error = penalty_ground - result.reference_energy

    rotated = get_sparse_operator(
        to_hartree_fock_frame(result.hamiltonian, result.n_active_electrons),
        n_qubits=nqubits,
    )
    vacuum_energy = float(np.real(rotated[0, 0]))
    frame_error = vacuum_energy - result.reference_determinant_energy
    truncation_bound = float(result.metadata["truncation_l1_bound_hartree"])
    passed = (
        hermitian_error < 1.0e-10
        and commutator_error < 1.0e-10
        and abs(hf_error) < 1.0e-7 + truncation_bound
        and abs(reference_error) < 1.0e-6 + truncation_bound
        and abs(frame_error) < 1.0e-7 + truncation_bound
        and abs(penalty_electrons - result.n_active_electrons) < 1.0e-6
        and abs(penalty_spin_squared) < SPIN_CONTAMINATION_TOLERANCE
        and abs(penalty_energy_error) < 1.0e-6 + truncation_bound
        and truncation_bound <= 1.0e-4
    )
    print(f"System                    : {result.metadata['molecule']}")
    print(f"Active space              : {result.n_active_electrons}e, {nqubits // 2}o")
    print(f"Orbitals / solvation      : {result.metadata.get('orbital_selection')} / "
          f"{result.metadata.get('solvation_model', 'vacuum')}")
    print(f"Qubits / Pauli terms      : {nqubits} / {len(result.hamiltonian.terms)}")
    print(f"SCF energy                : {result.hartree_fock_energy:.10f} Ha")
    print(f"Reference determinant     : {result.reference_determinant_energy:.10f} Ha")
    print(f"Hermiticity residual      : {hermitian_error:.3e}")
    print(f"[H,N] residual            : {commutator_error:.3e}")
    print(f"Determinant mapping error : {hf_error:+.3e} Ha")
    print(f"CASCI mapping error       : {reference_error:+.3e} Ha")
    print(f"HF-frame error            : {frame_error:+.3e} Ha")
    print(f"Penalty ground <N>        : {penalty_electrons:.10f}")
    print(f"Penalty ground <S^2>      : {penalty_spin_squared:.3e}")
    print(f"Penalty ground error      : {penalty_energy_error:+.3e} Ha")
    print(f"Truncation L1 bound       : {1000.0 * truncation_bound:.6f} mHa")
    print("ALL CHECKS PASSED" if passed else "CHECKS FAILED")
    if passed:
        receipt_path = validation_receipt_path(args.cache)
        receipt = {
            "cache_sha256": file_fingerprint(args.cache),
            "spec_sha256": spec_fingerprint(spec),
            "workflow_sha256": workflow_fingerprint(),
            "number_penalty_hartree": args.number_penalty,
            "spin_penalty_hartree": args.spin_penalty,
            "checks": "all_passed",
        }
        write_json_atomic(receipt_path, receipt)
        print(f"Validation receipt written: {receipt_path}")
    return 0 if passed else 1
