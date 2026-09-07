#!/usr/bin/env python3
"""Molecular specification, CASCI construction, operators, and cache I/O.

Backend-independent: this module knows nothing about Qiskit. The expensive
electronic-structure work runs once and is cached as JSON, and the local Qiskit
Aer run loads that verified qubit Hamiltonian.

Molecular integrals come from one of two interchangeable drivers, chosen by
`_select_cas_data`. PySCF is preferred wherever it installs. It publishes no
Windows wheel, though, so `_cas_data_native` provides the same quantities from
the self-contained STO-3G engine in `integrals.py`; that is what lets `prepare`
run on Windows at all. Both drivers hand back the same `_CasData`, so
everything downstream -- the Jordan-Wigner mapping, the truncation audit, the
metadata -- is shared, and the engine actually used is recorded in the cache as
`integral_backend`.

This diverges from the sibling studies' `chemistry.py`, which are PySCF-only;
the driver split is the difference, and the CASCI construction either side of it
is unchanged.

Imports of PySCF, OpenFermion and integrals.py remain lazy so CLI help, the unit
tests and the cache reader all work in an environment that has none of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CHEMICAL_ACCURACY_MHA = 1.6
# Applied to exact eigenvectors in validate_main, where 1e-6 is the right scale.
SPIN_CONTAMINATION_TOLERANCE = 1.0e-6
# Applied to the VQE state, which a hardware-efficient ansatz does not keep in
# the target particle-number sector. Subtracting the penalty recovers <H> exactly
# by linearity, but <H> is only variational *within* the sector, so the leak must
# be shown to be too small to hide the reported error. Budget it at 10% of
# chemical accuracy rather than picking a round number for the leakage itself.
CONTAMINATION_ENERGY_BUDGET_HA = 0.1 * CHEMICAL_ACCURACY_MHA / 1000.0
# Price per unit of broken symmetry when scoring a result, whether or not that
# symmetry was penalized during the optimization.
CONTAMINATION_PRICE_HA = 1.0
MIN_ACTIVE_ORBITAL_LOCALIZATION = 0.30
CACHE_SCHEMA = 1
COMPRESSION_TOLERANCE = 1.0e-12
DEFAULT_HAMILTONIAN_CUTOFF = 1.0e-6


@dataclass(frozen=True)
class MoleculeSpec:
    """A fixed molecular Hamiltonian specification."""

    name: str
    atoms: Sequence[tuple[str, tuple[float, float, float]]]
    # A plain basis name, or an explicit entry for every element when diffuse
    # functions are wanted only on selected atoms of an anion.
    basis: str | dict[str, str]
    charge: int
    spin: int
    n_active_orbitals: int
    n_active_electrons: int
    geometry_source: str
    model_note: str
    orbital_selection: str = "canonical_hf_frontier"
    diagnostic_atom_indices: Sequence[int] = field(default_factory=tuple)


@dataclass
class ChemistryResult:
    hamiltonian: Any
    n_qubits: int
    n_active_electrons: int
    hartree_fock_energy: float
    reference_energy: float
    reference_method: str
    core_energy: float
    metadata: dict[str, Any]


def _spec_payload(spec: MoleculeSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["atoms"] = [
        [symbol, [float(x), float(y), float(z)]]
        for symbol, (x, y, z) in spec.atoms
    ]
    payload["diagnostic_atom_indices"] = list(spec.diagnostic_atom_indices)
    return payload


def spec_fingerprint(spec: MoleculeSpec) -> str:
    encoded = json.dumps(
        _spec_payload(spec), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workflow_fingerprint() -> str:
    """Hash every scientific/runner module used to produce a result."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "chemistry.py",
        "integrals.py",
        "validation.py",
        "qiskit_runtime.py",
        "system.py",
        "run.py",
    ):
        try:
            content = (root / name).read_bytes()
        except OSError:
            return "unknown"
        digest.update(name.encode("utf-8") + b"\0" + content + b"\0")
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_receipt_path(cache: Path) -> Path:
    return cache.with_name(f"{cache.name}.validated.json")


def require_validation_receipt(
    cache: Path, spec: MoleculeSpec, number_penalty: float, spin_penalty: float
) -> dict[str, Any]:
    receipt_path = validation_receipt_path(cache)
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Missing validation receipt {receipt_path}. Run `python run.py validate` "
            "with the same penalty settings before VQE."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "cache_sha256": file_fingerprint(cache),
        "spec_sha256": spec_fingerprint(spec),
        "workflow_sha256": workflow_fingerprint(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"Stale validation receipt {receipt_path}: {key} does not match. "
                "Run `python run.py validate` again."
            )
    for key, requested in (
        ("number_penalty_hartree", number_penalty),
        ("spin_penalty_hartree", spin_penalty),
    ):
        if not np.isclose(float(receipt.get(key, float("nan"))), requested, atol=1.0e-12):
            raise RuntimeError(
                f"Validation used {key}={receipt.get(key)}, but VQE requested "
                f"{requested}. Re-run validation with matching penalties."
            )
    return receipt


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace a JSON artifact without ever exposing a partial document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open(mode="w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_molecule(spec: MoleculeSpec) -> Any:
    from pyscf import gto

    return gto.M(
        atom=[(symbol, coords) for symbol, coords in spec.atoms],
        basis=spec.basis,
        unit="Angstrom",
        charge=spec.charge,
        spin=spec.spin,
        verbose=0,
    )


def _validate_spec(spec: MoleculeSpec, total_electrons: int, n_mos: int) -> int:
    if spec.spin != 0:
        raise ValueError("This workflow currently supports closed-shell singlets only")
    if spec.n_active_orbitals < 1 or spec.n_active_electrons < 1:
        raise ValueError("The active space must contain orbitals and electrons")
    if spec.n_active_electrons % 2:
        raise ValueError("The active-electron count must be even")
    if spec.n_active_electrons > 2 * spec.n_active_orbitals:
        raise ValueError("The active electrons do not fit in the active orbitals")
    frozen_electrons = total_electrons - spec.n_active_electrons
    if frozen_electrons < 0 or frozen_electrons % 2:
        raise ValueError("The inactive space must contain electron pairs")
    ncore = frozen_electrons // 2
    if ncore + spec.n_active_orbitals > n_mos:
        raise ValueError("The requested core plus active space exceeds the MO space")
    if spec.orbital_selection != "canonical_hf_frontier":
        raise ValueError(f"Unsupported orbital selection: {spec.orbital_selection}")
    return ncore


def _mulliken_orbital_diagnostics(
    overlap: np.ndarray,
    ao_slices: Sequence[tuple[int, int]],
    symbols: Sequence[str],
    mo_coeff: np.ndarray,
    active_indices: Sequence[int],
    atom_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Mulliken population of each active MO on the diagnostic atoms.

    Backend-independent: `ao_slices` and `symbols` come from PySCF's
    ``aoslice_by_atom``/``atom_symbol`` or from ``integrals.ao_slices_by_atom``,
    which produces the same contiguous per-atom ranges.
    """
    if not atom_indices:
        return []
    invalid = [index for index in atom_indices if index < 0 or index >= len(symbols)]
    if invalid:
        raise ValueError(f"Invalid diagnostic atom indices: {invalid}")
    diagnostics: list[dict[str, Any]] = []
    for mo_index in active_indices:
        vector = np.asarray(mo_coeff[:, mo_index])
        s_vector = overlap @ vector
        populations: dict[str, float] = {}
        target_total = 0.0
        for atom_index in atom_indices:
            start, stop = (int(value) for value in ao_slices[atom_index])
            value = float(np.real(np.dot(vector[start:stop], s_vector[start:stop])))
            label = f"{atom_index}:{symbols[atom_index]}"
            populations[label] = value
            target_total += value
        diagnostics.append(
            {
                "mo_index_zero_based": int(mo_index),
                "target_atom_mulliken_population": target_total,
                "per_atom_mulliken_population": populations,
            }
        )
    return diagnostics


def truncate_qubit_operator(operator: Any, abs_tol: float) -> tuple[Any, int, float]:
    from openfermion import QubitOperator

    if abs_tol < 0:
        raise ValueError("Hamiltonian cutoff must be non-negative")
    truncated = QubitOperator()
    discarded_count = 0
    discarded_l1 = 0.0
    for term, coefficient in operator.terms.items():
        if abs(coefficient) < abs_tol:
            discarded_count += 1
            discarded_l1 += float(abs(coefficient))
        else:
            truncated.terms[term] = coefficient
    truncated.compress(abs_tol=COMPRESSION_TOLERANCE)
    return truncated, discarded_count, discarded_l1


@dataclass
class _CasData:
    """Everything an integral backend must supply to build the Hamiltonian."""

    n_electrons: int
    n_orbitals: int
    n_core: int
    active_indices: list[int]
    one_body: np.ndarray  # active-space h1eff, includes the core field
    two_body_chemist: np.ndarray  # active-space (pq|rs)
    core_energy: float
    hartree_fock_energy: float
    orbital_energies: np.ndarray
    nuclear_repulsion: float
    diagnostics: list[dict[str, Any]]
    integral_backend: str
    # PySCF solves CASCI directly; the native path derives the reference by
    # exact diagonalization of the untruncated Hamiltonian instead.
    reference_energy: float | None
    reference_solver: str


def _check_localization(
    diagnostics: list[dict[str, Any]], allow_delocalized_active_space: bool
) -> bool:
    delocalized = [
        entry
        for entry in diagnostics
        if entry["target_atom_mulliken_population"] < MIN_ACTIVE_ORBITAL_LOCALIZATION
    ]
    if delocalized and not allow_delocalized_active_space:
        offenders = ", ".join(
            f"MO {entry['mo_index_zero_based']}"
            f" ({entry['target_atom_mulliken_population']:.3f})"
            for entry in delocalized
        )
        raise RuntimeError(
            "Canonical frontier orbitals carry too little weight on the target atoms "
            f"(threshold {MIN_ACTIVE_ORBITAL_LOCALIZATION}): {offenders}. For a vacuum "
            "anion this usually means the diffuse set, not the reaction center, defines "
            "the frontier. Restrict diffuse functions to the reacting atoms via a "
            "per-element basis dict, or pass --allow-delocalized-active-space to override."
        )
    return bool(delocalized)


def _cas_data_pyscf(spec: MoleculeSpec) -> _CasData:
    from pyscf import ao2mo, mcscf, scf

    molecule = build_molecule(spec)
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-10
    mean_field.max_cycle = 100
    hartree_fock_energy = float(mean_field.kernel())
    if not mean_field.converged:
        newton = mean_field.newton()
        newton.conv_tol = 1.0e-10
        newton.max_cycle = 50
        hartree_fock_energy = float(newton.kernel(mean_field.mo_coeff, mean_field.mo_occ))
        mean_field = newton
    if not mean_field.converged:
        raise RuntimeError(f"PySCF RHF failed to converge for {spec.name}")

    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_mos = int(mo_coeff.shape[1])
    ncore = _validate_spec(spec, int(molecule.nelectron), n_mos)
    active_indices = list(range(ncore, ncore + spec.n_active_orbitals))

    ao_slices = [
        (int(row[2]), int(row[3])) for row in molecule.aoslice_by_atom()
    ]
    diagnostics = _mulliken_orbital_diagnostics(
        molecule.intor_symmetric("int1e_ovlp"),
        ao_slices,
        [molecule.atom_symbol(index) for index in range(molecule.natm)],
        mo_coeff,
        active_indices,
        spec.diagnostic_atom_indices,
    )

    cas = mcscf.CASCI(mean_field, spec.n_active_orbitals, spec.n_active_electrons)
    cas.verbose = 0
    # get_h1eff wants the FULL MO coefficient matrix; get_h2eff wants only the
    # active columns and re-slices internally when given the full matrix. Passing
    # mo_coeff to both is correct but relies on that re-slice.
    one_body, core_energy = cas.get_h1eff(mo_coeff)
    two_body = ao2mo.restore(1, cas.get_h2eff(mo_coeff), spec.n_active_orbitals)
    return _CasData(
        n_electrons=int(molecule.nelectron),
        n_orbitals=n_mos,
        n_core=ncore,
        active_indices=active_indices,
        one_body=np.asarray(one_body, dtype=float),
        two_body_chemist=np.asarray(two_body, dtype=float),
        core_energy=float(core_energy),
        hartree_fock_energy=hartree_fock_energy,
        orbital_energies=np.asarray(mean_field.mo_energy),
        nuclear_repulsion=float(molecule.energy_nuc()),
        diagnostics=diagnostics,
        integral_backend=f"pyscf-{_pyscf_version()}",
        reference_energy=float(cas.kernel(mo_coeff)[0]),
        reference_solver="pyscf.mcscf.CASCI",
    )


def _pyscf_version() -> str:
    try:
        import pyscf

        return str(pyscf.__version__)
    except Exception:  # pragma: no cover - only for the metadata string
        return "unknown"


def _cas_data_native(spec: MoleculeSpec) -> _CasData:
    """Build the same quantities with the built-in STO-3G engine.

    This is the path that makes `prepare` work on Windows, where PySCF cannot be
    installed. `integrals.py` documents the scope: STO-3G, closed shell, s and p
    functions -- enough for LiH and not intended as a general chemistry code.
    """
    import integrals

    mean_field = integrals.run_rhf(
        spec.atoms, spec.basis, charge=spec.charge, spin=spec.spin
    )
    ncore = _validate_spec(spec, mean_field.n_electrons, mean_field.n_orbitals)
    active_indices = list(range(ncore, ncore + spec.n_active_orbitals))

    diagnostics = _mulliken_orbital_diagnostics(
        mean_field.overlap,
        integrals.ao_slices_by_atom(spec.atoms, spec.basis),
        [symbol.capitalize() for symbol, _ in spec.atoms],
        mean_field.mo_coeff,
        active_indices,
        spec.diagnostic_atom_indices,
    )

    one_body, two_body, core_energy = integrals.cas_integrals(
        mean_field, ncore, spec.n_active_orbitals
    )
    return _CasData(
        n_electrons=mean_field.n_electrons,
        n_orbitals=mean_field.n_orbitals,
        n_core=ncore,
        active_indices=active_indices,
        one_body=one_body,
        two_body_chemist=two_body,
        core_energy=core_energy,
        hartree_fock_energy=mean_field.e_hf,
        orbital_energies=mean_field.mo_energy,
        nuclear_repulsion=mean_field.e_nuc,
        diagnostics=diagnostics,
        integral_backend="native-sto3g",
        reference_energy=None,
        reference_solver="exact diagonalization in the N-electron sector",
    )


def _lowest_energy_in_number_sector(
    hamiltonian: Any, n_qubits: int, n_electrons: int
) -> float:
    """CASCI energy: the lowest eigenvalue among N-electron states.

    Equivalent to a CAS-CI solve, because the active-space Hamiltonian conserves
    particle number and the restricted block is exactly the CI matrix. It does
    not additionally project onto a spin sector, so a system whose active-space
    ground state is a triplet would return that instead; `validate` measures
    <S^2> of this eigenvector and fails if it is not the expected singlet.
    """
    from openfermion.linalg import get_sparse_operator, jw_number_restrict_operator

    sparse = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    restricted = jw_number_restrict_operator(sparse, n_electrons, n_qubits=n_qubits)
    dimension = restricted.shape[0]
    if dimension <= 512:
        dense = np.asarray(restricted.todense())
        return float(np.min(np.linalg.eigvalsh(dense)).real)
    from scipy.sparse.linalg import eigsh

    return float(eigsh(restricted, k=1, which="SA", return_eigenvectors=False)[0].real)


def build_qubit_hamiltonian(
    spec: MoleculeSpec,
    compression_tolerance: float = DEFAULT_HAMILTONIAN_CUTOFF,
    allow_delocalized_active_space: bool = False,
    integral_backend: str = "auto",
) -> ChemistryResult:
    """Construct a CASCI Hamiltonian in canonical HF frontier orbitals.

    `integral_backend` is "auto" (PySCF when importable, else the built-in
    STO-3G engine), "pyscf", or "native". The choice affects only how the
    molecular integrals are produced; everything downstream is shared, so the
    two backends are directly comparable and the one actually used is recorded
    in the cache metadata.
    """
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial

    data = _select_cas_data(spec, integral_backend)
    delocalized = _check_localization(data.diagnostics, allow_delocalized_active_space)

    two_body_openfermion = np.asarray(
        data.two_body_chemist.transpose(0, 2, 3, 1), dtype=float, order="C"
    )
    one_spin, two_spin = spinorb_from_spatial(
        np.asarray(data.one_body, dtype=float), two_body_openfermion
    )
    interaction = InteractionOperator(data.core_energy, one_spin, 0.5 * two_spin)
    hamiltonian = jordan_wigner(get_fermion_operator(interaction))
    hamiltonian.compress(abs_tol=COMPRESSION_TOLERANCE)

    n_qubits = 2 * spec.n_active_orbitals
    reference_energy = data.reference_energy
    if reference_energy is None:
        # Solve before truncation, so the reference is a property of the
        # molecule rather than of the Pauli cutoff it is later compared against.
        reference_energy = _lowest_energy_in_number_sector(
            hamiltonian, n_qubits, spec.n_active_electrons
        )

    terms_before = len(hamiltonian.terms)
    hamiltonian, discarded, discarded_l1 = truncate_qubit_operator(
        hamiltonian, compression_tolerance
    )

    core_energy = data.core_energy
    hartree_fock_energy = data.hartree_fock_energy
    orbital_energies = data.orbital_energies
    ncore = data.n_core
    active_indices = data.active_indices
    diagnostics = data.diagnostics
    n_mos = data.n_orbitals
    molecule_nelectron = data.n_electrons
    metadata = {
        "molecule": spec.name,
        "basis": spec.basis,
        "charge": spec.charge,
        "spin_2s": spec.spin,
        "geometry_source": spec.geometry_source,
        "model_note": spec.model_note,
        "orbital_selection": spec.orbital_selection,
        "spec_sha256": spec_fingerprint(spec),
        "workflow_sha256": workflow_fingerprint(),
        "atoms": _spec_payload(spec)["atoms"],
        "integral_backend": data.integral_backend,
        "total_electrons": molecule_nelectron,
        "total_spatial_orbitals": n_mos,
        "active_spatial_orbitals": spec.n_active_orbitals,
        "active_electrons": spec.n_active_electrons,
        "active_mo_indices_zero_based": active_indices,
        "active_mo_energies_hartree": [
            float(orbital_energies[index]) for index in active_indices
        ],
        "active_orbital_diagnostics": diagnostics,
        "min_active_orbital_localization": MIN_ACTIVE_ORBITAL_LOCALIZATION,
        "delocalized_active_orbitals_overridden": bool(delocalized),
        "frozen_core_orbitals": ncore,
        "qubits": 2 * spec.n_active_orbitals,
        "nuclear_repulsion_hartree": float(data.nuclear_repulsion),
        "core_energy_hartree": float(core_energy),
        "hartree_fock_energy_hartree": hartree_fock_energy,
        "reference_energy_hartree": reference_energy,
        "reference_method": "CASCI",
        "reference_solver": data.reference_solver,
        "pauli_terms": len(hamiltonian.terms),
        "pauli_terms_before_truncation": terms_before,
        "discarded_pauli_terms": discarded,
        "hamiltonian_cutoff_hartree": compression_tolerance,
        "truncation_l1_bound_hartree": discarded_l1,
    }
    return ChemistryResult(
        hamiltonian=hamiltonian,
        n_qubits=2 * spec.n_active_orbitals,
        n_active_electrons=spec.n_active_electrons,
        hartree_fock_energy=hartree_fock_energy,
        reference_energy=reference_energy,
        reference_method="CASCI",
        core_energy=float(core_energy),
        metadata=metadata,
    )


def save_chemistry(result: ChemistryResult, path: Path) -> None:
    terms = []
    for term, coefficient in sorted(result.hamiltonian.terms.items()):
        terms.append(
            {
                "pauli": [[int(qubit), pauli] for qubit, pauli in term],
                "real": float(np.real(coefficient)),
                "imag": float(np.imag(coefficient)),
            }
        )
    payload = {
        "schema": CACHE_SCHEMA,
        "n_qubits": result.n_qubits,
        "n_active_electrons": result.n_active_electrons,
        "hartree_fock_energy": result.hartree_fock_energy,
        "reference_energy": result.reference_energy,
        "reference_method": result.reference_method,
        "core_energy": result.core_energy,
        "metadata": result.metadata,
        "terms": terms,
    }
    write_json_atomic(path, payload)


def load_chemistry(path: Path, spec: MoleculeSpec | None = None) -> ChemistryResult:
    from openfermion import QubitOperator

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CACHE_SCHEMA:
        raise RuntimeError(f"Unsupported Hamiltonian cache schema in {path}")
    if spec is not None:
        expected = spec_fingerprint(spec)
        actual = payload.get("metadata", {}).get("spec_sha256")
        if actual != expected:
            raise RuntimeError(
                f"Stale Hamiltonian cache {path}: spec hash {actual}, expected {expected}"
            )
    operator = QubitOperator()
    for item in payload["terms"]:
        term = tuple((int(q), str(p)) for q, p in item["pauli"])
        operator.terms[term] = complex(float(item["real"]), float(item["imag"]))
    operator.compress(abs_tol=COMPRESSION_TOLERANCE)
    return ChemistryResult(
        hamiltonian=operator,
        n_qubits=int(payload["n_qubits"]),
        n_active_electrons=int(payload["n_active_electrons"]),
        hartree_fock_energy=float(payload["hartree_fock_energy"]),
        reference_energy=float(payload["reference_energy"]),
        reference_method=str(payload["reference_method"]),
        core_energy=float(payload["core_energy"]),
        metadata=dict(payload["metadata"]),
    )


def hartree_fock_occupation(n_active_electrons: int) -> list[int]:
    return list(range(n_active_electrons))


def electron_number_operator(n_qubits: int) -> Any:
    from openfermion import QubitOperator

    result = QubitOperator()
    for qubit in range(n_qubits):
        result += QubitOperator((), 0.5)
        result += QubitOperator(f"Z{qubit}", -0.5)
    result.compress(abs_tol=COMPRESSION_TOLERANCE)
    return result


def number_deviation_operator(n_qubits: int, target: int) -> Any:
    from openfermion import QubitOperator

    shifted = electron_number_operator(n_qubits) - QubitOperator((), float(target))
    result = shifted * shifted
    result.compress(abs_tol=COMPRESSION_TOLERANCE)
    return result


def spin_squared_qubit_operator(n_qubits: int) -> Any:
    """Return S^2 in the interleaved alpha/beta spin-orbital convention."""
    if n_qubits % 2:
        raise ValueError("An electronic spin-orbital register must have even size")
    try:
        from openfermion import jordan_wigner, s_squared_operator
    except ImportError:  # compatibility with older OpenFermion exports
        from openfermion import jordan_wigner
        from openfermion.hamiltonians import s_squared_operator

    result = jordan_wigner(s_squared_operator(n_qubits // 2))
    result.compress(abs_tol=COMPRESSION_TOLERANCE)
    return result


def add_electron_number_penalty(
    hamiltonian: Any, n_qubits: int, target: int, coefficient: float
) -> Any:
    result = hamiltonian + coefficient * number_deviation_operator(
        n_qubits, target
    )
    result.compress(abs_tol=COMPRESSION_TOLERANCE)
    return result


def conjugate_with_x(operator: Any, flip_qubits: Iterable[int]) -> Any:
    from openfermion import QubitOperator

    flip = set(int(qubit) for qubit in flip_qubits)
    result = QubitOperator()
    for term, coefficient in operator.terms.items():
        sign = -1 if sum(q in flip and p in ("Y", "Z") for q, p in term) % 2 else 1
        result.terms[term] = sign * coefficient
    result.compress(abs_tol=COMPRESSION_TOLERANCE)
    return result


def to_hartree_fock_frame(operator: Any, n_active_electrons: int) -> Any:
    return conjugate_with_x(operator, hartree_fock_occupation(n_active_electrons))


def _select_cas_data(spec: MoleculeSpec, integral_backend: str) -> _CasData:
    """Pick an integral engine, preferring PySCF when it is available.

    PySCF remains the default where it can be installed: it is the more general
    and far more heavily exercised code. The native engine exists so that
    Windows -- where PySCF ships no wheel and the sdist build is unsupported --
    is not locked out of `prepare` entirely.
    """
    import importlib.util

    choice = integral_backend.lower()
    if choice not in {"auto", "pyscf", "native"}:
        raise ValueError(
            f"Unknown integral backend {integral_backend!r}; "
            "expected 'auto', 'pyscf' or 'native'"
        )
    have_pyscf = importlib.util.find_spec("pyscf") is not None
    if choice == "pyscf":
        if not have_pyscf:
            raise SystemExit(
                "--integral-backend pyscf was requested but PySCF is not "
                "installed. PySCF publishes no Windows wheel; use "
                "--integral-backend native, or build the cache on "
                "Linux/macOS/WSL/Colab and copy it here."
            )
        return _cas_data_pyscf(spec)
    if choice == "native":
        return _cas_data_native(spec)
    return _cas_data_pyscf(spec) if have_pyscf else _cas_data_native(spec)


def prepare_main(spec: MoleculeSpec, default_cache: str) -> int:
    parser = argparse.ArgumentParser(description="Build and cache a CASCI qubit Hamiltonian")
    parser.add_argument("--output", type=Path, default=Path(default_cache))
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_HAMILTONIAN_CUTOFF,
        help="Drop Pauli terms below this magnitude. Runtime is proportional to "
        "term count; `python run.py validate` rejects a discarded-L1 bound above "
        "0.1 mHa, so the saving is audited.",
    )
    parser.add_argument(
        "--integral-backend",
        choices=("auto", "pyscf", "native"),
        default="auto",
        help="Molecular-integral engine. 'auto' uses PySCF when it is "
        "importable and the built-in STO-3G engine otherwise, which is what "
        "lets this command run on Windows. The engine used is recorded in the "
        "cache as integral_backend.",
    )
    parser.add_argument(
        "--allow-delocalized-active-space",
        action="store_true",
        help="Build the Hamiltonian even if the active orbitals carry little "
        "weight on the diagnostic atoms. Use only deliberately.",
    )
    args = parser.parse_args()
    if args.cutoff < 0:
        parser.error("--cutoff must be non-negative")
    result = build_qubit_hamiltonian(
        spec,
        args.cutoff,
        args.allow_delocalized_active_space,
        integral_backend=args.integral_backend,
    )
    save_chemistry(result, args.output)
    print(json.dumps(result.metadata, indent=2))
    print(f"Hamiltonian cache written: {args.output}")
    return 0
