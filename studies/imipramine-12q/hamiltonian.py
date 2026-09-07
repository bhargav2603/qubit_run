#!/usr/bin/env python3
"""From a molecule specification to a cached qubit Hamiltonian.

CASCI construction, Jordan-Wigner mapping, the symmetry operators used as
penalties, the Hartree-Fock frame rotation, and cache I/O.

Backend-independent: this module knows nothing about Qiskit. Together with
`molecule.py` and `validation.py` it forms the physics layer that
`physics_fingerprint` hashes and the validation receipt is bound to, so the
execution layer can be edited without being able to alter -- or invalidate the
proof of -- a Hamiltonian that was built and validated elsewhere.

The expensive PySCF work runs once and is cached as JSON; everything after that
loads the verified qubit Hamiltonian. That split matters on Windows in
particular, because **PySCF publishes no Windows wheel**: `prepare` is run once
on Linux, macOS, WSL or Colab, and only the cache is copied to the Windows
machine. Imports of PySCF and OpenFermion stay lazy so CLI help, the unit tests
and the cache reader all work in an environment that has neither.
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
# Mulliken analysis of *virtual* orbitals is far less reliable than of occupied
# ones: virtuals have long basis-set tails, individual atomic contributions can
# even be negative, and the partitioning gets more arbitrary the higher up the
# virtual manifold you go. A threshold tight enough to be meaningful on an
# occupied pi orbital will reject a perfectly good pi* orbital, so the two are
# scored separately rather than being held to one number.
MIN_ACTIVE_VIRTUAL_LOCALIZATION = 0.20
CACHE_SCHEMA = 1
RECEIPT_SCHEMA = 2
COMPRESSION_TOLERANCE = 1.0e-12
DEFAULT_HAMILTONIAN_CUTOFF = 1.0e-6
# No state inside the target particle-number sector can lie below that sector's
# exact ground state, which validation proves equals the cached CASCI energy.
# An energy above it is therefore free evidence that the state did not leak
# anywhere that matters, usable when no observable evaluator is available.
SECTOR_ENERGY_TOLERANCE_HA = 1.0e-8

# Only these files can change the physics of a cached Hamiltonian. The
# validation receipt is bound to them alone, so that editing the simulator layer
# -- on a Windows machine where PySCF is not installed and the cache therefore
# cannot be rebuilt or re-validated -- does not invalidate a receipt it cannot
# possibly have affected. The full-folder workflow fingerprint is still recorded
# in every result for provenance.
PHYSICS_MODULES = ("molecule.py", "hamiltonian.py", "validation.py")


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
    # How much of each active orbital must sit on the diagnostic atoms. The
    # right value is a per-molecule judgement: excluding diffuse artifacts on a
    # vacuum anion needs only the default, while confirming that a frontier
    # orbital really is the aromatic pi system warrants a much stricter bar.
    # Occupied and virtual orbitals are scored separately -- see
    # MIN_ACTIVE_VIRTUAL_LOCALIZATION for why one number cannot serve both.
    min_active_orbital_localization: float | None = None
    min_active_virtual_localization: float | None = None
    # Published numbers for this exact active space, printed by `prepare` as a
    # sanity anchor. Deliberately excluded from the fingerprint: a literature
    # annotation must not invalidate a cache or a validation receipt.
    published_reference: dict[str, Any] | None = None


@dataclass
class CachedHamiltonian:
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
    # Literature annotation, not physics. Excluded so that citing a paper never
    # invalidates a cache.
    payload.pop("published_reference", None)
    return payload


def spec_fingerprint(spec: MoleculeSpec) -> str:
    encoded = json.dumps(
        _spec_payload(spec), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    try:
        for path in paths:
            digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    except OSError:
        return "unknown"
    return digest.hexdigest()


def workflow_fingerprint() -> str:
    """Hash every Python module in this folder, for provenance.

    Globbing rather than listing filenames means a module added later cannot
    slip into a run unhashed, so a result always identifies the exact code that
    produced it. This is recorded, never used as an execution gate -- see
    physics_fingerprint for that.
    """
    root = Path(__file__).resolve().parent
    return _digest_files(sorted(root.glob("*.py"), key=lambda item: item.name))


def physics_fingerprint() -> str:
    """Hash only the modules that can change a cached Hamiltonian.

    This is what the validation receipt is bound to. Narrowing it from the whole
    folder to PHYSICS_MODULES is what keeps the simulator layer editable on a
    machine that cannot re-validate: patching qiskit_runtime.py cannot alter a
    Hamiltonian that was built and validated elsewhere, so it must not be able
    to invalidate the proof that the Hamiltonian is correct.
    """
    root = Path(__file__).resolve().parent
    return _digest_files(root / name for name in PHYSICS_MODULES)


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_receipt_path(cache: Path) -> Path:
    """hamiltonian_cas6e6o.json -> hamiltonian_cas6e6o.validated.json"""
    return cache.with_suffix(".validated.json")


def penalty_key(number_penalty: float, spin_penalty: float) -> str:
    """Stable dictionary key for one validated penalty setting."""
    return f"number={float(number_penalty):.12g},spin={float(spin_penalty):.12g}"


def load_validation_receipt(cache: Path) -> dict[str, Any]:
    receipt_path = validation_receipt_path(cache)
    if not receipt_path.is_file():
        return {}
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return receipt if isinstance(receipt, dict) else {}


def receipt_is_current(receipt: dict[str, Any], cache: Path, spec: MoleculeSpec) -> bool:
    """Does this receipt describe the Hamiltonian and physics code we have now?"""
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return False
    return (
        receipt.get("cache_sha256") == file_fingerprint(cache)
        and receipt.get("spec_sha256") == spec_fingerprint(spec)
        and receipt.get("physics_sha256") == physics_fingerprint()
    )


def require_validation_receipt(
    cache: Path, spec: MoleculeSpec, number_penalty: float, spin_penalty: float
) -> dict[str, Any]:
    """Refuse to run VQE unless this exact Hamiltonian was proved correct.

    Bound to the cache, the molecule specification and PHYSICS_MODULES only. A
    receipt can carry several validated penalty settings, so the remedies the
    README recommends -- dropping the number penalty to recover <H> directly, or
    raising it to force the correct sector -- are already authorised when you
    discover from inside an allocation that you need one.
    """
    receipt_path = validation_receipt_path(cache)
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Missing validation receipt {receipt_path}. Run `python run.py validate` "
            "with the same penalty settings before VQE."
        )
    receipt = load_validation_receipt(cache)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError(
            f"Validation receipt {receipt_path} uses schema {receipt.get('schema')!r}, "
            f"expected {RECEIPT_SCHEMA}. Run `python run.py validate` again."
        )
    for key, value in (
        ("cache_sha256", file_fingerprint(cache)),
        ("spec_sha256", spec_fingerprint(spec)),
        ("physics_sha256", physics_fingerprint()),
    ):
        if receipt.get(key) != value:
            raise RuntimeError(
                f"Stale validation receipt {receipt_path}: {key} does not match. "
                "The cached Hamiltonian, the molecule or "
                f"{'/'.join(PHYSICS_MODULES)} changed since validation. "
                "Run `python run.py validate` again."
            )
    validated = receipt.get("validated_penalties", {})
    key = penalty_key(number_penalty, spin_penalty)
    if key not in validated:
        available = ", ".join(sorted(validated)) or "none"
        raise RuntimeError(
            f"No validation for {key}. Validated settings: {available}. "
            "Re-run `python run.py validate --number-penalty ... --spin-penalty ...` "
            "on the login node; it accepts several values at once and keeps them all."
        )
    entry = dict(validated[key])
    entry["cache_sha256"] = receipt["cache_sha256"]
    entry["spec_sha256"] = receipt["spec_sha256"]
    entry["physics_sha256"] = receipt["physics_sha256"]
    entry["validated_at_workflow_sha256"] = receipt.get("workflow_sha256")
    entry["penalty_key"] = key
    return entry


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
    molecule: Any,
    mo_coeff: np.ndarray,
    active_indices: Sequence[int],
    atom_indices: Sequence[int],
    n_occupied_mos: int | None = None,
    mo_energies: Sequence[float] | None = None,
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
        entry: dict[str, Any] = {
            "mo_index_zero_based": int(mo_index),
            "target_atom_mulliken_population": target_total,
            "per_atom_mulliken_population": populations,
        }
        if n_occupied_mos is not None:
            entry["occupancy"] = "occupied" if mo_index < n_occupied_mos else "virtual"
            offset = int(mo_index) - (n_occupied_mos - 1)
            entry["frontier_label"] = (
                f"HOMO{offset:+d}" if offset <= 0 else f"LUMO{offset - 1:+d}"
            ).replace("HOMO+0", "HOMO").replace("LUMO+0", "LUMO")
        if mo_energies is not None:
            entry["orbital_energy_hartree"] = float(mo_energies[mo_index])
        diagnostics.append(entry)
    return diagnostics


def localization_thresholds(spec: MoleculeSpec) -> tuple[float, float]:
    """(occupied, virtual) Mulliken thresholds for the active-space guard."""
    occupied = (
        MIN_ACTIVE_ORBITAL_LOCALIZATION
        if spec.min_active_orbital_localization is None
        else float(spec.min_active_orbital_localization)
    )
    virtual = (
        MIN_ACTIVE_VIRTUAL_LOCALIZATION
        if spec.min_active_virtual_localization is None
        else float(spec.min_active_virtual_localization)
    )
    return occupied, virtual


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


def converged_mean_field(spec: MoleculeSpec) -> tuple[Any, Any, float]:
    """Converged RHF for a spec, returning (molecule, mean_field, energy).

    On the rare occasion that plain DIIS stalls, second-order SCF finishes the
    job -- but the converged orbitals are copied back onto the ORIGINAL RHF
    object rather than handing the SOSCF wrapper downstream. mcscf.CASCI's
    treatment of a SOSCF object has changed across PySCF releases, and a cold
    fallback path that only executes on the day something has already gone wrong
    is the worst possible place to depend on version-specific behaviour.
    """
    from pyscf import scf

    molecule = build_molecule(spec)
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-10
    mean_field.max_cycle = 100
    hartree_fock_energy = float(mean_field.kernel())
    if not mean_field.converged:
        newton = mean_field.newton()
        newton.conv_tol = 1.0e-10
        newton.max_cycle = 50
        hartree_fock_energy = float(
            newton.kernel(mean_field.mo_coeff, mean_field.mo_occ)
        )
        if not newton.converged:
            raise RuntimeError(f"PySCF RHF failed to converge for {spec.name}")
        mean_field.mo_coeff = np.asarray(newton.mo_coeff)
        mean_field.mo_occ = np.asarray(newton.mo_occ)
        mean_field.mo_energy = np.asarray(newton.mo_energy)
        mean_field.e_tot = hartree_fock_energy
        mean_field.converged = True
    if not mean_field.converged:
        raise RuntimeError(f"PySCF RHF failed to converge for {spec.name}")
    return molecule, mean_field, hartree_fock_energy


def build_qubit_hamiltonian(
    spec: MoleculeSpec,
    compression_tolerance: float = DEFAULT_HAMILTONIAN_CUTOFF,
    allow_delocalized_active_space: bool = False,
) -> CachedHamiltonian:
    """Construct a CASCI Hamiltonian in canonical HF frontier orbitals."""
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from pyscf import ao2mo, mcscf

    molecule, mean_field, hartree_fock_energy = converged_mean_field(spec)

    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_mos = int(mo_coeff.shape[1])
    ncore = _validate_spec(spec, int(molecule.nelectron), n_mos)
    active_indices = list(range(ncore, ncore + spec.n_active_orbitals))
    n_occupied_mos = int(molecule.nelectron) // 2

    diagnostics = _mulliken_orbital_diagnostics(
        molecule,
        mo_coeff,
        active_indices,
        spec.diagnostic_atom_indices,
        n_occupied_mos=n_occupied_mos,
        mo_energies=np.asarray(mean_field.mo_energy),
    )
    occupied_threshold, virtual_threshold = localization_thresholds(spec)
    delocalized = [
        entry
        for entry in diagnostics
        if entry["target_atom_mulliken_population"]
        < (occupied_threshold if entry["occupancy"] == "occupied" else virtual_threshold)
    ]
    if delocalized and not allow_delocalized_active_space:
        offenders = ", ".join(
            f"MO {entry['mo_index_zero_based']} ({entry.get('frontier_label', '?')}, "
            f"{entry['occupancy']}, {entry['target_atom_mulliken_population']:.3f} vs "
            f"{occupied_threshold if entry['occupancy'] == 'occupied' else virtual_threshold})"
            for entry in delocalized
        )
        raise RuntimeError(
            "Canonical frontier orbitals carry too little weight on the target atoms: "
            f"{offenders}. The active space is therefore not describing the chemistry "
            "it was chosen for -- for a vacuum anion the diffuse set usually defines "
            "the frontier, and for a conjugated molecule a saturated side chain can. "
            "Run `python run.py prepare --diagnose` to see the whole frontier window "
            "with populations and orbital energies, then shift the active space or "
            "add polarization functions. Lowering the threshold is not a fix; "
            "--allow-delocalized-active-space overrides it deliberately."
        )

    cas = mcscf.CASCI(
        mean_field, spec.n_active_orbitals, spec.n_active_electrons
    )
    cas.verbose = 0
    # get_h1eff takes the FULL MO coefficient matrix and slices it itself.
    # get_h2eff takes ONLY the active columns: older PySCF re-slices a full
    # matrix internally, newer PySCF does not, so passing the full matrix here
    # is a silently-wrong-Hamiltonian risk across versions. Slice explicitly.
    active_mo_coeff = np.asarray(mo_coeff[:, active_indices], order="C")
    one_body, core_energy = cas.get_h1eff(mo_coeff)
    two_body = ao2mo.restore(
        1, cas.get_h2eff(active_mo_coeff), spec.n_active_orbitals
    )
    two_body_openfermion = np.asarray(
        np.asarray(two_body).transpose(0, 2, 3, 1), dtype=float, order="C"
    )
    one_spin, two_spin = spinorb_from_spatial(
        np.asarray(one_body, dtype=float), two_body_openfermion
    )
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
        "min_active_orbital_localization": occupied_threshold,
        "min_active_virtual_localization": virtual_threshold,
        "delocalized_active_orbitals_overridden": bool(delocalized),
        "frozen_core_orbitals": ncore,
        "qubits": 2 * spec.n_active_orbitals,
        "nuclear_repulsion_hartree": float(molecule.energy_nuc()),
        "core_energy_hartree": float(core_energy),
        "hartree_fock_energy_hartree": hartree_fock_energy,
        "reference_energy_hartree": reference_energy,
        "reference_method": "CASCI",
        # The quantity to compare against the literature. Total energies move by
        # tens of mHa between conformers; the correlation energy recovered inside
        # a given active space does not, so this is what shows whether the same
        # six orbitals were selected.
        "active_space_correlation_energy_hartree": reference_energy - hartree_fock_energy,
        "pauli_terms": len(hamiltonian.terms),
        "pauli_terms_before_truncation": terms_before,
        "discarded_pauli_terms": discarded,
        "hamiltonian_cutoff_hartree": compression_tolerance,
        "truncation_l1_bound_hartree": discarded_l1,
    }
    return CachedHamiltonian(
        hamiltonian=hamiltonian,
        n_qubits=2 * spec.n_active_orbitals,
        n_active_electrons=spec.n_active_electrons,
        hartree_fock_energy=hartree_fock_energy,
        reference_energy=reference_energy,
        reference_method="CASCI",
        core_energy=float(core_energy),
        metadata=metadata,
    )


def save_cache(result: CachedHamiltonian, path: Path) -> None:
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


def load_cache(path: Path, spec: MoleculeSpec | None = None) -> CachedHamiltonian:
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
    return CachedHamiltonian(
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


def diagnose_frontier_window(spec: MoleculeSpec, window: int) -> int:
    """Report the frontier orbital window without building anything.

    One RHF run answers the question the active-space guard would otherwise only
    ever answer by refusing to build: which orbitals actually carry the target
    chemistry, and how far the requested window is from them. Always exits 0 --
    this is a measurement, not a gate.
    """
    molecule, mean_field, hartree_fock_energy = converged_mean_field(spec)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    mo_energy = np.asarray(mean_field.mo_energy)
    n_mos = int(mo_coeff.shape[1])
    n_occupied = int(molecule.nelectron) // 2
    ncore = _validate_spec(spec, int(molecule.nelectron), n_mos)
    requested = list(range(ncore, ncore + spec.n_active_orbitals))

    low = max(0, n_occupied - 1 - window)
    high = min(n_mos - 1, n_occupied + window)
    indices = list(range(low, high + 1))
    diagnostics = _mulliken_orbital_diagnostics(
        molecule,
        mo_coeff,
        indices,
        spec.diagnostic_atom_indices,
        n_occupied_mos=n_occupied,
        mo_energies=mo_energy,
    )
    occupied_threshold, virtual_threshold = localization_thresholds(spec)

    print(f"System                : {spec.name}")
    print(f"Basis / MOs           : {spec.basis} / {n_mos}")
    print(f"Electrons / doubly occ: {molecule.nelectron} / {n_occupied}")
    print(f"RHF energy            : {hartree_fock_energy:.12f} Ha")
    print(f"Requested active MOs  : {requested}  (CAS({spec.n_active_electrons}e,{spec.n_active_orbitals}o))")
    print(f"Thresholds occ / virt : {occupied_threshold:.2f} / {virtual_threshold:.2f}")
    print()
    print("  MO  label    occ/virt   energy (Ha)   pop on target atoms   verdict  in window")
    print("  " + "-" * 84)
    for entry in diagnostics:
        index = entry["mo_index_zero_based"]
        threshold = (
            occupied_threshold if entry["occupancy"] == "occupied" else virtual_threshold
        )
        population = entry["target_atom_mulliken_population"]
        verdict = "ok  " if population >= threshold else "LOW "
        print(
            f"  {index:3d}  {entry['frontier_label']:<8s} {entry['occupancy']:<9s}"
            f" {entry['orbital_energy_hartree']:+13.6f}"
            f" {population:21.3f}   {verdict}"
            f"     {'<==' if index in requested else ''}"
        )
    print()
    print(
        "A contiguous run of high-population orbitals straddling the HOMO/LUMO gap is\n"
        "the active space you want. If the requested window (<==) misses it, shift the\n"
        "active space rather than lowering a threshold; if nothing scores high, the\n"
        "canonical orbitals are too delocalized for this test and localized orbitals\n"
        "or a larger basis are needed."
    )
    return 0


def _report_published_reference(spec: MoleculeSpec, result: CachedHamiltonian) -> None:
    reference = spec.published_reference
    if not reference:
        return
    correlation = result.reference_energy - result.hartree_fock_energy
    print()
    print("Published reference ({}):".format(reference.get("source", "literature")))
    print(f"  active space           : {reference.get('active_space', '?')}")
    published_total = reference.get("casci_energy_hartree")
    if published_total is not None:
        delta = result.reference_energy - float(published_total)
        print(f"  published CASCI total  : {float(published_total):.6f} Ha")
        print(f"  this build CASCI total : {result.reference_energy:.6f} Ha  ({delta:+.6f} Ha)")
    published_correlation = reference.get("correlation_energy_hartree")
    print(f"  this build E_CASCI-E_HF: {1000.0 * correlation:+.3f} mHa")
    if published_correlation is not None:
        gap = correlation - float(published_correlation)
        print(
            f"  published E_CASCI-E_HF : {1000.0 * float(published_correlation):+.3f} mHa"
            f"  ({1000.0 * gap:+.3f} mHa)"
        )
    note = reference.get("note")
    if note:
        print(f"  note                   : {note}")


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
        "--allow-delocalized-active-space",
        action="store_true",
        help="Build the Hamiltonian even if the active orbitals carry little "
        "weight on the diagnostic atoms. Use only deliberately.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run RHF and report the frontier orbital window -- populations, "
        "energies and which orbitals the active space would take -- then exit "
        "without building. Answers in one run what a failed guard answers in many.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="How many orbitals either side of the HOMO/LUMO gap --diagnose reports.",
    )
    args = parser.parse_args()
    if args.cutoff < 0:
        parser.error("--cutoff must be non-negative")
    if args.window < 1:
        parser.error("--window must be positive")
    if args.diagnose:
        return diagnose_frontier_window(spec, args.window)
    result = build_qubit_hamiltonian(
        spec, args.cutoff, args.allow_delocalized_active_space
    )
    save_cache(result, args.output)
    print(json.dumps(result.metadata, indent=2))
    _report_published_reference(spec, result)
    print()
    print(f"Hamiltonian cache written: {args.output}")
    return 0
