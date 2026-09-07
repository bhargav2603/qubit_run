#!/usr/bin/env python3
"""Molecular specification, CASCI construction, operators, and cache I/O.

Backend-independent: this module knows nothing about Qiskit. The expensive
PySCF work runs once and is cached as JSON, and the local Qiskit Aer run loads
that verified qubit Hamiltonian. That split matters more here than it did on the
cluster, because **PySCF publishes no Windows wheel**: `prepare` is run once on
Linux, macOS, WSL or Colab, and only the cache is copied to the Windows machine.

Imports of PySCF and OpenFermion remain lazy so CLI help, the unit tests and the
cache reader all work in an environment that has neither.
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
ORBITAL_SELECTIONS = ("canonical_hf_frontier", "mp2_natural")
# Schema 2 adds solvation, orbital selection, natural occupations, and the
# reference-determinant energy. With MP2 natural orbitals the lowest active
# determinant is no longer the SCF determinant, so its energy has to be recorded
# separately instead of being assumed equal to E_RHF.
CACHE_SCHEMA = 2
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
    # None = vacuum. The CovAngelo protocol uses C-PCM with epsilon = 4.
    solvent_epsilon: float | None = None
    diagnostic_atom_indices: Sequence[int] = field(default_factory=tuple)


@dataclass
class ChemistryResult:
    hamiltonian: Any
    n_qubits: int
    n_active_electrons: int
    # The SCF energy of the whole molecule. Equal to the reference-determinant
    # energy only for canonical orbitals.
    hartree_fock_energy: float
    reference_energy: float
    reference_method: str
    core_energy: float
    metadata: dict[str, Any]
    # <det|H_active|det> for the determinant that fills the lowest active
    # orbitals -- exactly the state the quantum circuit prepares at theta = 0.
    # This, not hartree_fock_energy, is what the mapping checks must reproduce.
    reference_determinant_energy: float = 0.0


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
    for name in ("chemistry.py", "validation.py", "qiskit_runtime.py", "system.py", "run.py"):
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
    if spec.orbital_selection not in ORBITAL_SELECTIONS:
        raise ValueError(
            f"Unsupported orbital selection: {spec.orbital_selection}. "
            f"Choose from {ORBITAL_SELECTIONS}."
        )
    return ncore


def run_mean_field(spec: MoleculeSpec) -> tuple[Any, Any, float]:
    """Converge RHF, with C-PCM solvation when the spec asks for it.

    Returns (molecule, converged mean-field object, SCF energy). Shared by the
    Hamiltonian builder and the classical ladder so both describe the same
    electronic structure.
    """
    from pyscf import scf

    molecule = build_molecule(spec)
    mean_field = scf.RHF(molecule)
    if spec.solvent_epsilon is not None:
        # The paper uses a polarizable continuum with epsilon = 4 to mimic the
        # protein interior while keeping the system small enough to simulate.
        try:
            mean_field = mean_field.PCM()
        except AttributeError:  # older PySCF exposes it only as a function
            from pyscf import solvent

            mean_field = solvent.PCM(mean_field)
        mean_field.with_solvent.method = "C-PCM"
        mean_field.with_solvent.eps = float(spec.solvent_epsilon)
    mean_field.conv_tol = 1.0e-10
    mean_field.max_cycle = 100
    energy = float(mean_field.kernel())
    if not mean_field.converged:
        newton = mean_field.newton()
        newton.conv_tol = 1.0e-10
        newton.max_cycle = 50
        energy = float(newton.kernel(mean_field.mo_coeff, mean_field.mo_occ))
        mean_field = newton
    if not mean_field.converged:
        raise RuntimeError(f"PySCF RHF failed to converge for {spec.name}")
    return molecule, mean_field, energy


def select_orbitals(
    spec: MoleculeSpec, mean_field: Any, ncore: int
) -> tuple[np.ndarray, list[float] | None]:
    """Return the orbital coefficients defining the active space.

    `canonical_hf_frontier` keeps the SCF orbitals. `mp2_natural` replaces them
    with MP2 natural orbitals ordered by occupation, which is what the CovAngelo
    protocol uses. That ordering matters for an anion in a diffuse basis: the
    canonical frontier orbitals often describe the diffuse tail rather than the
    C-S reaction region, whereas natural occupations rank orbitals by how
    correlated they actually are.
    """
    if spec.orbital_selection == "canonical_hf_frontier":
        return np.asarray(mean_field.mo_coeff), None

    from pyscf import mp
    from pyscf.mcscf import addons

    perturbation = mp.MP2(mean_field)
    perturbation.kernel()
    occupations, natural_orbitals = addons.make_natural_orbitals(perturbation)
    occupations = np.asarray(occupations, dtype=float)
    order = np.argsort(-occupations)  # descending; most-occupied first
    return (
        np.asarray(natural_orbitals)[:, order],
        [float(value) for value in occupations[order]],
    )


def closed_shell_determinant_energy(
    one_body: np.ndarray, two_body_chemist: np.ndarray, core_energy: float, n_electrons: int
) -> float:
    """Energy of the determinant filling the lowest active orbitals.

    This is the state the circuit prepares before any rotation. For canonical
    orbitals it equals the SCF energy; for natural orbitals it does not, which
    is precisely why it has to be computed rather than assumed.
    """
    if n_electrons % 2:
        raise ValueError("closed-shell determinant requires an even electron count")
    occupied = range(n_electrons // 2)
    energy = float(core_energy)
    for i in occupied:
        energy += 2.0 * float(one_body[i, i])
    for i in occupied:
        for j in occupied:
            # (ii|jj) Coulomb, (ij|ji) exchange, chemists' notation.
            energy += 2.0 * float(two_body_chemist[i, i, j, j])
            energy -= float(two_body_chemist[i, j, j, i])
    return energy


def _mulliken_orbital_diagnostics(
    molecule: Any,
    mo_coeff: np.ndarray,
    active_indices: Sequence[int],
    atom_indices: Sequence[int],
) -> list[dict[str, Any]]:
    if not atom_indices:
        return []
    invalid = [index for index in atom_indices if index < 0 or index >= molecule.natm]
    if invalid:
        raise ValueError(f"Invalid diagnostic atom indices: {invalid}")
    overlap = molecule.intor_symmetric("int1e_ovlp")
    ao_slices = molecule.aoslice_by_atom()
    diagnostics: list[dict[str, Any]] = []
    for mo_index in active_indices:
        vector = np.asarray(mo_coeff[:, mo_index])
        s_vector = overlap @ vector
        populations: dict[str, float] = {}
        target_total = 0.0
        for atom_index in atom_indices:
            start, stop = (int(value) for value in ao_slices[atom_index, 2:4])
            value = float(np.real(np.dot(vector[start:stop], s_vector[start:stop])))
            label = f"{atom_index}:{molecule.atom_symbol(atom_index)}"
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


def build_qubit_hamiltonian(
    spec: MoleculeSpec,
    compression_tolerance: float = DEFAULT_HAMILTONIAN_CUTOFF,
    allow_delocalized_active_space: bool = False,
) -> ChemistryResult:
    """Construct a CASCI Hamiltonian in canonical HF frontier orbitals."""
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from pyscf import ao2mo, mcscf

    molecule, mean_field, hartree_fock_energy = run_mean_field(spec)

    n_mos = int(np.asarray(mean_field.mo_coeff).shape[1])
    ncore = _validate_spec(spec, int(molecule.nelectron), n_mos)
    mo_coeff, natural_occupations = select_orbitals(spec, mean_field, ncore)
    mo_coeff = np.asarray(mo_coeff)
    active_indices = list(range(ncore, ncore + spec.n_active_orbitals))

    diagnostics = _mulliken_orbital_diagnostics(
        molecule, mo_coeff, active_indices, spec.diagnostic_atom_indices
    )
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

    cas = mcscf.CASCI(
        mean_field, spec.n_active_orbitals, spec.n_active_electrons
    )
    cas.verbose = 0
    # get_h1eff wants the FULL MO coefficient matrix; get_h2eff wants only the
    # active columns and re-slices internally when given the full matrix. Passing
    # mo_coeff to both is correct but relies on that re-slice.
    one_body, core_energy = cas.get_h1eff(mo_coeff)
    two_body = ao2mo.restore(
        1, cas.get_h2eff(mo_coeff), spec.n_active_orbitals
    )
    one_body = np.asarray(one_body, dtype=float)
    two_body = np.asarray(two_body, dtype=float)
    reference_determinant_energy = closed_shell_determinant_energy(
        one_body, two_body, float(core_energy), spec.n_active_electrons
    )
    two_body_openfermion = np.asarray(
        two_body.transpose(0, 2, 3, 1), dtype=float, order="C"
    )
    one_spin, two_spin = spinorb_from_spatial(one_body, two_body_openfermion)
    interaction = InteractionOperator(
        float(core_energy), one_spin, 0.5 * two_spin
    )
    hamiltonian = jordan_wigner(get_fermion_operator(interaction))
    hamiltonian.compress(abs_tol=COMPRESSION_TOLERANCE)
    terms_before = len(hamiltonian.terms)
    hamiltonian, discarded, discarded_l1 = truncate_qubit_operator(
        hamiltonian, compression_tolerance
    )

    reference_energy = float(cas.kernel(mo_coeff)[0])
    orbital_energies = np.asarray(mean_field.mo_energy)
    metadata = {
        "molecule": spec.name,
        "basis": spec.basis,
        "charge": spec.charge,
        "spin_2s": spec.spin,
        "geometry_source": spec.geometry_source,
        "model_note": spec.model_note,
        "orbital_selection": spec.orbital_selection,
        "solvent_epsilon": spec.solvent_epsilon,
        "solvation_model": "C-PCM" if spec.solvent_epsilon is not None else "vacuum",
        "active_natural_occupations": (
            None
            if natural_occupations is None
            else [natural_occupations[index] for index in active_indices]
        ),
        "reference_determinant_energy_hartree": reference_determinant_energy,
        "spec_sha256": spec_fingerprint(spec),
        "workflow_sha256": workflow_fingerprint(),
        "atoms": _spec_payload(spec)["atoms"],
        "total_electrons": int(molecule.nelectron),
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
        "nuclear_repulsion_hartree": float(molecule.energy_nuc()),
        "core_energy_hartree": float(core_energy),
        "hartree_fock_energy_hartree": hartree_fock_energy,
        "reference_energy_hartree": reference_energy,
        "reference_method": "CASCI",
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
        reference_determinant_energy=reference_determinant_energy,
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
        "reference_determinant_energy": result.reference_determinant_energy,
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
        reference_determinant_energy=float(
            payload.get("reference_determinant_energy", payload["hartree_fock_energy"])
        ),
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


def require_pyscf() -> None:
    """Fail with instructions rather than a bare ModuleNotFoundError.

    `prepare` is the only command that needs PySCF, and PySCF ships no Windows
    wheel -- the sdist build wants nmake and a C/C++ toolchain, and the project
    does not support Windows even when one is present. Everything downstream
    consumes the JSON cache, so the fix is to build it elsewhere once, not to
    fight the installer.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("pyscf") is not None:
        return
    if sys.platform == "win32":
        raise SystemExit(
            "PySCF is not installed, and it publishes no Windows wheel, so "
            "`prepare` cannot run natively on Windows.\n\n"
            "Build the Hamiltonian once somewhere PySCF works, then copy two "
            "files back into this folder:\n"
            "  1. On WSL / Linux / macOS / Google Colab, from a copy of this "
            "folder:\n"
            "       pip install pyscf openfermion numpy scipy\n"
            "       python run.py prepare\n"
            "       python run.py validate\n"
            "  2. Copy acrylamide_thiolate_hamiltonian.json and\n"
            "     acrylamide_thiolate_hamiltonian.json.validated.json here.\n"
            "  3. Back on Windows:  python run.py vqe\n\n"
            "The cache is plain JSON and carries its own specification hash, so "
            "a stale or mismatched file is a hard error, not a wrong answer.\n"
            "Meanwhile `python run.py selftest` verifies the entire local "
            "Qiskit stack without PySCF."
        )
    raise SystemExit(
        "PySCF is not installed. Install it with `pip install pyscf` and re-run "
        "`python run.py prepare`."
    )


def prepare_main(spec: MoleculeSpec, default_cache: str) -> int:
    require_pyscf()
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
        "--allow-delocalized-active-space",
        action="store_true",
        help="Build the Hamiltonian even if the active orbitals carry little "
        "weight on the diagnostic atoms. Use only deliberately.",
    )
    args = parser.parse_args()
    if args.cutoff < 0:
        parser.error("--cutoff must be non-negative")
    result = build_qubit_hamiltonian(
        spec, args.cutoff, args.allow_delocalized_active_space
    )
    save_chemistry(result, args.output)
    print(json.dumps(result.metadata, indent=2))
    print(f"Hamiltonian cache written: {args.output}")
    return 0
