#!/usr/bin/env python3
"""Prove a cached Hamiltonian correct before anything is allowed to run on it.

Every check here runs from the JSON cache alone -- no PySCF, no network, no
Qiskit -- so a Hamiltonian built once on Colab can be re-proved on the machine
that consumes it. Passing writes a hash-bound receipt, and `run.py hivqe`
refuses to start without one.

The check that matters most is the **projection check**. The subspace
Hamiltonian is built by a string-driven contraction for speed, and an earlier
version of that contraction restricted the two-electron intermediate state to
the selected subspace. That is not P H P -- it is strictly smaller -- and it
produced subspace energies *below* the exact CAS ground state, which is
impossible. Comparing the contraction against Slater-Condon matrix elements
computed one at a time is what caught it, so that comparison is a permanent
check and not a one-off debugging session.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from determinants import (
    ActiveSpace,
    Subspace,
    SubspaceHamiltonian,
    hartree_fock_determinant,
    hartree_fock_subspace,
    make_strings,
    matrix_element,
    solve_subspace,
    spin_square,
)
from hamiltonian import (
    RECEIPT_SCHEMA,
    MoleculeSpec,
    file_fingerprint,
    load_cache,
    physics_fingerprint,
    spec_fingerprint,
    validation_receipt_path,
    workflow_fingerprint,
    write_json_atomic,
)


HARTREE_FOCK_TOLERANCE = 1.0e-8
SYMMETRY_TOLERANCE = 1.0e-10
CONTRACTION_TOLERANCE = 1.0e-9
SPIN_TOLERANCE = 1.0e-9
# A projected Hamiltonian cannot have an eigenvalue below the exact CAS ground
# state. The allowance is numerical, not physical.
VARIATIONAL_TOLERANCE = 1.0e-9


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = True
        self.detail = ""
        self.value: Any = None

    def record(self, passed: bool, detail: str, value: Any = None) -> "Check":
        self.passed = bool(passed)
        self.detail = detail
        self.value = value
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
        }


def check_integral_symmetry(space: ActiveSpace) -> list[Check]:
    checks = []
    residual = float(np.abs(space.h1e - space.h1e.T).max())
    checks.append(
        Check("h1e is symmetric").record(
            residual <= SYMMETRY_TOLERANCE, f"max |h - h^T| = {residual:.3e}", residual
        )
    )
    eri = space.eri
    for label, transposed in (
        ("(pq|rs) = (qp|rs)", eri.transpose(1, 0, 2, 3)),
        ("(pq|rs) = (pq|sr)", eri.transpose(0, 1, 3, 2)),
        ("(pq|rs) = (rs|pq)", eri.transpose(2, 3, 0, 1)),
    ):
        residual = float(np.abs(eri - transposed).max())
        checks.append(
            Check(f"eri symmetry {label}").record(
                residual <= SYMMETRY_TOLERANCE,
                f"max residual = {residual:.3e}",
                residual,
            )
        )
    return checks


def check_hartree_fock(space: ActiveSpace, metadata: dict[str, Any]) -> list[Check]:
    reference = hartree_fock_determinant(space.n_alpha, space.n_beta)
    energy = matrix_element(space, reference, reference)
    expected = float(metadata["energies"]["rhf_total"])
    difference = abs(energy - expected)
    checks = [
        Check("Hartree-Fock determinant reproduces the SCF energy").record(
            difference <= HARTREE_FOCK_TOLERANCE,
            f"Slater-Condon {energy:.12f} vs PySCF RHF {expected:.12f}, "
            f"difference {difference:.3e} Ha",
            difference,
        )
    ]
    spin = spin_square(
        space, hartree_fock_subspace(space.n_alpha, space.n_beta), np.ones((1, 1))
    )
    checks.append(
        Check("Hartree-Fock determinant is a singlet").record(
            abs(spin) <= SPIN_TOLERANCE, f"<S^2> = {spin:.3e}", float(spin)
        )
    )
    return checks


def check_contraction(space: ActiveSpace, seed: int = 5, n_strings: int = 8) -> Check:
    """The string contraction must equal Slater-Condon, element by element.

    A random subspace rather than a physically motivated one: the contraction
    must be exact for every projection, and a subspace chosen to look like a
    wavefunction would sample only the easy part of the index space.
    """
    rng = np.random.default_rng(seed)
    all_a = make_strings(space.n_orbitals, space.n_alpha)
    all_b = make_strings(space.n_orbitals, space.n_beta)
    picked_a = all_a[np.sort(rng.choice(len(all_a), size=min(n_strings, len(all_a)), replace=False))]
    picked_b = all_b[np.sort(rng.choice(len(all_b), size=min(n_strings, len(all_b)), replace=False))]
    subspace = Subspace(picked_a, picked_b)

    operator = SubspaceHamiltonian(space, subspace)
    contracted = operator.dense()

    dimension = subspace.dimension
    explicit = np.zeros((dimension, dimension))
    determinants = [
        (int(a), int(b)) for a in subspace.strings_a for b in subspace.strings_b
    ]
    for row, bra in enumerate(determinants):
        for column, ket in enumerate(determinants):
            explicit[row, column] = matrix_element(space, bra, ket)

    residual = float(np.abs(contracted - explicit).max())
    return Check("projected Hamiltonian equals Slater-Condon").record(
        residual <= CONTRACTION_TOLERANCE,
        f"{dimension} determinants, max |contraction - Slater-Condon| = "
        f"{residual:.3e} Ha",
        residual,
    )


def check_diagonal(space: ActiveSpace, seed: int = 6, n_strings: int = 10) -> Check:
    rng = np.random.default_rng(seed)
    all_a = make_strings(space.n_orbitals, space.n_alpha)
    all_b = make_strings(space.n_orbitals, space.n_beta)
    picked_a = all_a[np.sort(rng.choice(len(all_a), size=min(n_strings, len(all_a)), replace=False))]
    picked_b = all_b[np.sort(rng.choice(len(all_b), size=min(n_strings, len(all_b)), replace=False))]
    subspace = Subspace(picked_a, picked_b)
    operator = SubspaceHamiltonian(space, subspace)
    diagonal = operator.diagonal()
    worst = 0.0
    for row, a in enumerate(subspace.strings_a):
        for column, b in enumerate(subspace.strings_b):
            exact = matrix_element(space, (int(a), int(b)), (int(a), int(b)))
            worst = max(worst, abs(diagonal[row, column] - exact))
    return Check("Davidson preconditioner diagonal is exact").record(
        worst <= CONTRACTION_TOLERANCE,
        f"max |closed form - Slater-Condon| = {worst:.3e} Ha",
        worst,
    )


def check_variational_bound(
    space: ActiveSpace, metadata: dict[str, Any], seeds: tuple[int, ...] = (1, 2, 3)
) -> list[Check]:
    """No projected subspace may sit below the exact CAS ground state.

    This is the cheapest possible test of the whole classical engine, and it is
    a physical statement rather than a numerical one: P H P has its lowest
    eigenvalue above the true ground energy for every P, so a single violation
    condemns the contraction, the integrals or the cached reference.
    """
    reference = float(metadata["energies"]["casci_total"])
    all_a = make_strings(space.n_orbitals, space.n_alpha)
    all_b = make_strings(space.n_orbitals, space.n_beta)
    checks = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        size = min(24, len(all_a))
        picked_a = all_a[np.sort(rng.choice(len(all_a), size=size, replace=False))]
        picked_b = all_b[np.sort(rng.choice(len(all_b), size=size, replace=False))]
        result = solve_subspace(space, Subspace(picked_a, picked_b))
        margin = result.energy - reference
        checks.append(
            Check(f"random {result.dimension}-determinant subspace is above CASCI").record(
                margin >= -VARIATIONAL_TOLERANCE,
                f"E = {result.energy:.9f} Ha, {margin * 1000:+.3f} mHa above the "
                f"cached {metadata['energies']['reference_method']} reference",
                margin,
            )
        )
    return checks


def check_reference_reachable(
    space: ActiveSpace, metadata: dict[str, Any], max_determinants: int
) -> Check:
    """A selected-CI run must approach the cached CASCI energy from above.

    Together with the bound above this brackets the cached reference: nothing
    may go below it, and a serious subspace must get close to it. If PySCF's
    CASCI and this engine disagreed about the Hamiltonian, one of the two would
    fail.
    """
    from hivqe import HiVqeSettings, run_hivqe

    reference = float(metadata["energies"]["casci_total"])
    settings = HiVqeSettings(
        simulator="none",
        max_determinants=max_determinants,
        expansion=max(64, max_determinants // 40),
        max_iterations=12,
        energy_tolerance=1.0e-7,
        patience=2,
        optimizer="none",
    )
    result = run_hivqe(
        space,
        reference,
        float(metadata["energies"]["rhf_total"]),
        settings,
        progress=lambda _line: None,
    )
    gap = result.energy - reference
    return Check("selected CI converges toward the cached reference").record(
        gap >= -VARIATIONAL_TOLERANCE,
        f"{result.dimension:,} of {space.full_dimension:,} determinants reaches "
        f"{result.energy:.9f} Ha, {gap * 1000:+.3f} mHa above the reference",
        gap,
    )


def validate_cache(
    cache: Path, spec: MoleculeSpec, max_determinants: int, deep: bool
) -> tuple[list[Check], dict[str, Any]]:
    cached = load_cache(cache)
    space, metadata = cached.space, cached.metadata

    checks: list[Check] = []
    checks.append(
        Check("cache matches the molecule specification").record(
            metadata.get("spec_sha256") == spec_fingerprint(spec),
            f"cache {str(metadata.get('spec_sha256'))[:12]}... vs current "
            f"{spec_fingerprint(spec)[:12]}...",
        )
    )
    checks.append(
        Check("active space is the published one").record(
            metadata["n_active_electrons"] == spec.n_active_electrons
            and metadata["n_active_orbitals"] == spec.n_active_orbitals
            and metadata["n_qubits"] == spec.n_qubits,
            f"CAS({metadata['n_active_electrons']}e,{metadata['n_active_orbitals']}o)"
            f" -> {metadata['n_qubits']} qubits, "
            f"{metadata['full_cas_determinants']:,} determinants",
            metadata["full_cas_determinants"],
        )
    )
    checks.append(
        Check("frozen core is the sulfur shell").record(
            bool(metadata.get("core_ok", False)),
            f"worst population on {spec.central_atom} "
            f"{min(entry['population_on_central_atom'] for entry in metadata['core_diagnostics']):.3f}"
            f", core-active gap {metadata['core_active_gap_hartree']:.3f} Ha",
        )
    )
    checks += check_integral_symmetry(space)
    checks += check_hartree_fock(space, metadata)
    checks.append(check_contraction(space))
    checks.append(check_diagonal(space))
    checks += check_variational_bound(space, metadata)
    if deep:
        checks.append(check_reference_reachable(space, metadata, max_determinants))

    summary = {
        "bond_angstrom": cached.bond_angstrom,
        "full_dimension": space.full_dimension,
        "reference_energy": cached.reference_energy,
        "hartree_fock_energy": cached.hartree_fock_energy,
    }
    return checks, summary


def write_receipt(
    cache: Path, spec: MoleculeSpec, checks: list[Check], summary: dict[str, Any]
) -> Path:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "validated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "cache": Path(cache).name,
        "cache_sha256": file_fingerprint(cache),
        "spec_sha256": spec_fingerprint(spec),
        "physics_sha256": physics_fingerprint(),
        "workflow_sha256": workflow_fingerprint(),
        "summary": summary,
        "checks": [check.as_dict() for check in checks],
        "all_passed": all(check.passed for check in checks),
    }
    target = validation_receipt_path(cache)
    write_json_atomic(target, receipt)
    return target


def require_validation_receipt(cache: Path, spec: MoleculeSpec) -> dict[str, Any]:
    """Refuse to run unless this exact Hamiltonian was proved correct.

    Bound to the cache, the molecule specification and the physics modules only.
    Editing `visualize.py` or `ansatz.py` cannot invalidate a proof about a
    Hamiltonian they never touch -- which matters on a machine that cannot
    re-run `prepare` to earn a new receipt.
    """
    cache = Path(cache)
    receipt_path = validation_receipt_path(cache)
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Missing validation receipt {receipt_path.name}. Run "
            f"`python run.py validate --distance {cache.stem.split('_r')[-1]}` first."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError(
            f"{receipt_path.name} uses receipt schema {receipt.get('schema')!r}, "
            f"expected {RECEIPT_SCHEMA}. Re-run `validate`."
        )
    if not receipt.get("all_passed"):
        failed = [c["name"] for c in receipt.get("checks", []) if not c["passed"]]
        raise RuntimeError(
            f"{receipt_path.name} records failed checks: {', '.join(failed)}."
        )
    for key, value in (
        ("cache_sha256", file_fingerprint(cache)),
        ("spec_sha256", spec_fingerprint(spec)),
        ("physics_sha256", physics_fingerprint()),
    ):
        if receipt.get(key) != value:
            raise RuntimeError(
                f"Stale receipt {receipt_path.name}: {key} does not match. The "
                "cached Hamiltonian, the molecule or the physics modules changed "
                "since validation. Re-run `validate`."
            )
    return receipt


def validate_main(spec: MoleculeSpec, cache_root: str) -> int:
    parser = argparse.ArgumentParser(
        description="Check a cached Li2S Hamiltonian against exact references."
    )
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument(
        "--all", action="store_true", help="validate every cached distance"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also run a selected-CI to confirm it approaches the cached CASCI energy",
    )
    parser.add_argument("--max-determinants", type=int, default=8000)
    arguments = parser.parse_args()

    from hamiltonian import cache_path_for

    if arguments.all:
        caches = sorted(
            path
            for path in Path(cache_root).glob("li2s_r*.json")
            if not path.name.endswith(".validated.json")
        )
    else:
        distance = (
            spec.equilibrium_bond_angstrom
            if arguments.distance is None
            else arguments.distance
        )
        caches = [cache_path_for(cache_root, distance)]

    if not caches:
        print(f"No cached Hamiltonians in {cache_root}. Run `prepare` first.")
        return 1

    overall = True
    for cache in caches:
        if not cache.is_file():
            print(f"MISSING  {cache}")
            overall = False
            continue
        checks, summary = validate_cache(
            cache, spec, arguments.max_determinants, arguments.deep
        )
        passed = all(check.passed for check in checks)
        overall = overall and passed
        print(f"\n{cache.name}  (r = {summary['bond_angstrom']:.3f} A)")
        for check in checks:
            mark = "ok  " if check.passed else "FAIL"
            print(f"  [{mark}] {check.name}\n         {check.detail}")
        receipt = write_receipt(cache, spec, checks, summary)
        print(f"  receipt -> {receipt.name}  ({'PASSED' if passed else 'FAILED'})")

    print("\n" + ("ALL CHECKS PASSED" if overall else "SOME CHECKS FAILED"))
    return 0 if overall else 1
