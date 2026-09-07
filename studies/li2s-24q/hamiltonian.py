#!/usr/bin/env python3
"""From a Li2S geometry to a cached CAS(12e,12o) active-space Hamiltonian.

This is the only module that imports PySCF, and it is imported only by
`run.py prepare`. Everything downstream -- validation, HI-VQE, the classical
baselines, the plots -- reads a JSON cache and never needs a quantum-chemistry
package again. That split is what lets the expensive part run once on Colab or
Linux while the rest runs anywhere, including Windows, where PySCF publishes no
wheel.

The cache stores the active-space integrals themselves (h1e, eri, E_core), not
a qubit operator. HI-VQE diagonalises the Hamiltonian in a determinant basis, so
a Pauli decomposition would be a detour -- and an expensive one: the same
24-qubit Li2S Hamiltonian is 15,697 Pauli words, every one of which a
conventional VQE would have to measure. `run.py paulis` counts them, precisely
so that number can be checked rather than quoted.

Floating-point arrays are stored as base64 of their little-endian float64 bytes,
so a cache round-trips bit-exactly; a JSON list of decimal literals would not.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CHEMICAL_ACCURACY_HA = 1.6e-3
CACHE_SCHEMA = 1
RECEIPT_SCHEMA = 1

# Only these files can change the physics of a cached Hamiltonian. The validation
# receipt is bound to them alone, so that editing the simulator or plotting layer
# on a machine where PySCF is not installed -- and where the cache therefore
# cannot be rebuilt -- does not invalidate a receipt it cannot have affected.
PHYSICS_MODULES = ("molecule.py", "hamiltonian.py", "determinants.py", "validation.py")


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeSpec:
    """A fixed active-space specification, parameterised by one bond length."""

    name: str
    basis: str
    charge: int
    spin: int
    n_active_orbitals: int
    n_active_electrons: int
    equilibrium_bond_angstrom: float
    geometry_source: str
    model_note: str
    # Atoms are built by `geometry()`; this records the template for hashing.
    geometry_kind: str = "linear_symmetric_triatomic"
    central_atom: str = "S"
    terminal_atom: str = "Li"
    core_atom_index: int = 0
    min_core_localization: float = 0.90
    min_core_gap_hartree: float = 1.0
    published_reference: dict[str, Any] | None = None

    def geometry(
        self, bond_angstrom: float | None = None, fixed_angstrom: float | None = None
    ) -> list[tuple[str, tuple[float, float, float]]]:
        """Linear X-A-X with the *second* terminal atom at `bond_angstrom`.

        The dissociation coordinate of this study is "remove one lithium", so
        the fixed bond stays at equilibrium while the scanned one is stretched.
        At the equilibrium point the two are equal and the molecule is the
        symmetric D-infinity-h structure.
        """
        fixed = self.equilibrium_bond_angstrom if fixed_angstrom is None else fixed_angstrom
        scanned = fixed if bond_angstrom is None else bond_angstrom
        return [
            (self.central_atom, (0.0, 0.0, 0.0)),
            (self.terminal_atom, (0.0, 0.0, -float(fixed))),
            (self.terminal_atom, (0.0, 0.0, float(scanned))),
        ]

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_active_orbitals


def _spec_payload(spec: MoleculeSpec) -> dict[str, Any]:
    payload = asdict(spec)
    # Literature annotation, not physics: citing a paper must never invalidate
    # a cache or a validation receipt.
    payload.pop("published_reference", None)
    return payload


def spec_fingerprint(spec: MoleculeSpec) -> str:
    encoded = json.dumps(
        _spec_payload(spec), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------
# Fingerprints, receipts, atomic writes
# --------------------------------------------------------------------------


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
    slip into a run unhashed. Recorded in every result; never an execution gate.
    """
    root = Path(__file__).resolve().parent
    return _digest_files(sorted(root.glob("*.py"), key=lambda item: item.name))


def physics_fingerprint() -> str:
    """Hash only the modules that can change a cached Hamiltonian."""
    root = Path(__file__).resolve().parent
    return _digest_files(root / name for name in PHYSICS_MODULES)


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_receipt_path(cache: Path) -> Path:
    """foo.json -> foo.validated.json"""
    return Path(cache).with_suffix(".validated.json")


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


def encode_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    return {
        "shape": list(contiguous.shape),
        "dtype": "<f8",
        "base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("dtype") != "<f8":
        raise ValueError(f"unsupported array dtype {payload.get('dtype')!r}")
    raw = base64.b64decode(payload["base64"])
    return np.frombuffer(raw, dtype="<f8").reshape(tuple(payload["shape"])).copy()


# --------------------------------------------------------------------------
# Cache reading
# --------------------------------------------------------------------------


@dataclass
class CachedHamiltonian:
    """Everything downstream needs, with no quantum-chemistry package present."""

    space: Any  # determinants.ActiveSpace, imported lazily to keep layering clean
    metadata: dict[str, Any]

    @property
    def bond_angstrom(self) -> float:
        return float(self.metadata["bond_angstrom"])

    @property
    def hartree_fock_energy(self) -> float:
        return float(self.metadata["energies"]["rhf_total"])

    @property
    def reference_energy(self) -> float:
        return float(self.metadata["energies"]["casci_total"])

    @property
    def reference_method(self) -> str:
        return str(self.metadata["energies"]["reference_method"])


def load_cache(path: str | Path) -> CachedHamiltonian:
    from determinants import ActiveSpace

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CACHE_SCHEMA:
        raise RuntimeError(
            f"{path} uses cache schema {payload.get('schema')!r}, expected "
            f"{CACHE_SCHEMA}. Rebuild it with `python run.py prepare`."
        )
    integrals = payload["integrals"]
    space = ActiveSpace(
        n_orbitals=int(payload["n_active_orbitals"]),
        n_alpha=int(payload["n_alpha"]),
        n_beta=int(payload["n_beta"]),
        h1e=decode_array(integrals["h1e"]),
        eri=decode_array(integrals["eri"]),
        core_energy=float(integrals["core_energy"]),
    )
    metadata = {key: value for key, value in payload.items() if key != "integrals"}
    return CachedHamiltonian(space=space, metadata=metadata)


def cache_path_for(root: str | Path, bond_angstrom: float) -> Path:
    return Path(root) / f"li2s_r{float(bond_angstrom):.3f}.json"


# --------------------------------------------------------------------------
# Building (PySCF only)
# --------------------------------------------------------------------------


def build_molecule(spec: MoleculeSpec, bond_angstrom: float | None = None) -> Any:
    from pyscf import gto

    return gto.M(
        atom=[(symbol, coords) for symbol, coords in spec.geometry(bond_angstrom)],
        basis=spec.basis,
        unit="Angstrom",
        charge=spec.charge,
        spin=spec.spin,
        verbose=0,
    )


def _core_diagnostics(
    molecule: Any, mo_coeff: np.ndarray, mo_energy: np.ndarray, n_core: int, atom: int
) -> list[dict[str, Any]]:
    """Mulliken population of each frozen orbital on the central atom.

    CAS(12e,12o) on a 22-electron molecule freezes the five lowest orbitals, and
    for Li2S those should be the sulfur 1s, 2s and 2p shell -- the only orbitals
    deep enough to be genuinely inert. If a lithium 1s or a valence orbital ever
    drifted below them the active space would be describing something else while
    still converging cleanly, so this is measured rather than assumed.
    """
    overlap = molecule.intor_symmetric("int1e_ovlp")
    slices = molecule.aoslice_by_atom()
    start, stop = int(slices[atom, 2]), int(slices[atom, 3])
    diagnostics = []
    for index in range(n_core):
        vector = np.asarray(mo_coeff[:, index])
        gross = vector * (overlap @ vector)
        diagnostics.append(
            {
                "orbital": index,
                "energy_hartree": float(mo_energy[index]),
                "population_on_central_atom": float(gross[start:stop].sum()),
                "total_population": float(gross.sum()),
            }
        )
    return diagnostics


def _use_full_sz_solver(solver: Any, molecule: Any, fci: Any) -> None:
    """Make the reference the lowest state of the whole S_z = 0 sector.

    HI-VQE searches every determinant with n_alpha up and n_beta down and takes
    the lowest eigenvalue it can reach. For its error to be an error rather than
    an artefact, the reference has to be the lowest eigenvalue of that same
    sector -- no more constrained, no less.

    PySCF's default for a closed-shell RHF reference is `direct_spin0`, which
    exploits the alpha<->beta symmetry of the CI matrix. That is a real speedup
    and it is silently wrong here: a CI vector constrained to be symmetric under
    that exchange spans only the even-S states, so the solver cannot return a
    triplet even when a triplet is the ground state.

    Near equilibrium this is invisible, because the singlet is lower anyway. At
    a dissociated Li-S bond it is not: the fragments are Li(2S) and LiS(2Pi),
    two doublets whose singlet and triplet couplings are nearly degenerate, and
    whichever is lower is the physical answer. Scored against a singlet-only
    reference, a correct HI-VQE run that finds the triplet comes back with a
    *negative* error and is reported INCONSISTENT -- which reads as a bug and is
    not one.

    `direct_spin1` imposes only S_z, which is exactly the constraint HI-VQE
    itself is under.
    """
    solver.fcisolver = fci.direct_spin1.FCISolver(molecule)
    # The comparison downstream is at the tenth of a millihartree, and the
    # near-degeneracy that motivates this function is also what makes Davidson
    # lazy about the last few digits.
    solver.fcisolver.conv_tol = 1e-12


def build_active_space(
    spec: MoleculeSpec,
    bond_angstrom: float | None = None,
    orbitals: str = "hf",
    guess_mo: np.ndarray | None = None,
    allow_bad_core: bool = False,
    max_scf_cycles: int = 200,
) -> dict[str, Any]:
    """Run SCF, select the active space, and return everything worth caching."""
    from pyscf import ao2mo, fci, mcscf, scf

    molecule = build_molecule(spec, bond_angstrom)
    n_electrons = int(molecule.nelectron)
    n_core = (n_electrons - spec.n_active_electrons) // 2
    if 2 * n_core + spec.n_active_electrons != n_electrons:
        raise ValueError(
            f"{spec.n_active_electrons} active electrons leaves an odd inactive "
            f"count for {n_electrons} electrons"
        )

    mean_field = scf.RHF(molecule)
    mean_field.max_cycle = max_scf_cycles
    mean_field.conv_tol = 1e-11
    if guess_mo is not None:
        # Orbital continuation along a dissociation scan. RHF on a stretched
        # bond has several solutions and a fresh atomic guess routinely walks
        # into a different one, which puts a discontinuity in the curve that no
        # amount of correlation treatment can repair.
        occupation = np.zeros(guess_mo.shape[1])
        occupation[: n_electrons // 2] = 2
        density = mean_field.make_rdm1(np.asarray(guess_mo), occupation)
        mean_field.kernel(dm0=density)
    else:
        mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(
            f"RHF did not converge at bond length {bond_angstrom} A; try "
            "--continue-from on a shorter distance"
        )

    n_mos = mean_field.mo_coeff.shape[1]
    if n_core + spec.n_active_orbitals > n_mos:
        raise ValueError(
            f"{n_core} core + {spec.n_active_orbitals} active orbitals exceeds "
            f"the {n_mos} molecular orbitals in basis {spec.basis}"
        )

    if orbitals == "hf":
        solver = mcscf.CASCI(mean_field, spec.n_active_orbitals, spec.n_active_electrons)
        _use_full_sz_solver(solver, molecule, fci)
        solver.kernel()
        mo_coeff = np.asarray(mean_field.mo_coeff)
        mo_energy = np.asarray(mean_field.mo_energy)
        reference_method = "CASCI"
    elif orbitals == "casscf":
        solver = mcscf.CASSCF(
            mean_field, spec.n_active_orbitals, spec.n_active_electrons
        )
        _use_full_sz_solver(solver, molecule, fci)
        solver.max_cycle_macro = 100
        solver.kernel()
        mo_coeff = np.asarray(solver.mo_coeff)
        mo_energy = np.asarray(mean_field.mo_energy)
        reference_method = "CASSCF"
    else:
        raise ValueError(f"unknown orbital choice {orbitals!r}; use hf or casscf")

    # What spin state did the reference actually land on? Recorded rather than
    # assumed, because at long bond length the answer changes and every error
    # downstream is measured against this number.
    reference_spin_squared, reference_multiplicity = solver.fcisolver.spin_square(
        solver.ci, solver.ncas, solver.nelecas
    )

    core = _core_diagnostics(molecule, mo_coeff, mo_energy, n_core, spec.core_atom_index)
    gap = float(mo_energy[n_core] - mo_energy[n_core - 1]) if n_core else float("inf")
    worst = min((entry["population_on_central_atom"] for entry in core), default=1.0)
    core_ok = worst >= spec.min_core_localization and gap >= spec.min_core_gap_hartree
    if not core_ok and not allow_bad_core:
        raise RuntimeError(
            "the frozen core does not look like the sulfur 1s/2s/2p shell: worst "
            f"population on {spec.central_atom} is {worst:.3f} (need "
            f"{spec.min_core_localization}), core-active gap is {gap:.3f} Ha (need "
            f"{spec.min_core_gap_hartree}). Re-run with --allow-bad-core only if "
            "you have looked at `prepare --diagnose` and understand why."
        )

    h1e, core_energy = solver.get_h1eff()
    eri = ao2mo.restore(1, solver.get_h2eff(), spec.n_active_orbitals)

    n_alpha = (spec.n_active_electrons + spec.spin) // 2
    n_beta = (spec.n_active_electrons - spec.spin) // 2

    return {
        "molecule": molecule,
        "mean_field": mean_field,
        "solver": solver,
        "mo_coeff": mo_coeff,
        "mo_energy": mo_energy,
        "n_core": n_core,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "h1e": np.asarray(h1e),
        "eri": np.asarray(eri),
        "core_energy": float(core_energy),
        "reference_method": reference_method,
        "reference_energy": float(solver.e_tot),
        "reference_spin_squared": float(reference_spin_squared),
        "reference_multiplicity": float(reference_multiplicity),
        "rhf_energy": float(mean_field.e_tot),
        "nuclear_repulsion": float(molecule.energy_nuc()),
        "core_diagnostics": core,
        "core_active_gap_hartree": gap,
        "core_ok": bool(core_ok),
    }


def cache_payload(
    spec: MoleculeSpec,
    built: dict[str, Any],
    bond_angstrom: float,
    orbitals: str,
) -> dict[str, Any]:
    from determinants import (
        ActiveSpace,
        hartree_fock_determinant,
        matrix_element,
        n_determinants,
    )

    space = ActiveSpace(
        n_orbitals=spec.n_active_orbitals,
        n_alpha=built["n_alpha"],
        n_beta=built["n_beta"],
        h1e=built["h1e"],
        eri=built["eri"],
        core_energy=built["core_energy"],
    )
    reference = hartree_fock_determinant(built["n_alpha"], built["n_beta"])
    # The energy of the Hartree-Fock determinant evaluated through *our* Slater-
    # Condon code must equal PySCF's RHF total energy exactly. It is the single
    # cheapest end-to-end check that the integrals, the core energy, the orbital
    # ordering and the determinant convention all agree, and it is recorded in
    # the cache so `validate` can re-check it without PySCF.
    determinant_energy = matrix_element(space, reference, reference)

    return {
        "schema": CACHE_SCHEMA,
        "molecule": spec.name,
        "spec_sha256": spec_fingerprint(spec),
        "physics_sha256": physics_fingerprint(),
        "workflow_sha256": workflow_fingerprint(),
        "bond_angstrom": float(bond_angstrom),
        "fixed_bond_angstrom": float(spec.equilibrium_bond_angstrom),
        "geometry": [
            [symbol, list(coords)] for symbol, coords in spec.geometry(bond_angstrom)
        ],
        "basis": spec.basis,
        "charge": spec.charge,
        "spin": spec.spin,
        "orbitals": orbitals,
        "n_active_orbitals": spec.n_active_orbitals,
        "n_active_electrons": spec.n_active_electrons,
        "n_alpha": built["n_alpha"],
        "n_beta": built["n_beta"],
        "n_core_orbitals": built["n_core"],
        "n_molecular_orbitals": int(built["mo_coeff"].shape[1]),
        "n_qubits": spec.n_qubits,
        "full_cas_determinants": n_determinants(
            spec.n_active_orbitals, built["n_alpha"], built["n_beta"]
        ),
        "energies": {
            "rhf_total": built["rhf_energy"],
            "casci_total": built["reference_energy"],
            "reference_method": built["reference_method"],
            # <S^2> of the reference state. 0 is a singlet, 2 a triplet. It is
            # not fixed along the dissociation coordinate, and the point where
            # it changes is physics, not a glitch.
            "reference_spin_squared": built["reference_spin_squared"],
            "reference_multiplicity": built["reference_multiplicity"],
            "nuclear_repulsion": built["nuclear_repulsion"],
            "core_energy": built["core_energy"],
            "hartree_fock_determinant": determinant_energy,
            "hartree_fock_determinant_error": determinant_energy - built["rhf_energy"],
            "active_correlation": built["reference_energy"] - built["rhf_energy"],
        },
        "core_diagnostics": built["core_diagnostics"],
        "core_active_gap_hartree": built["core_active_gap_hartree"],
        "core_ok": built["core_ok"],
        "mo_energy": encode_array(built["mo_energy"]),
        "mo_coeff": encode_array(built["mo_coeff"]),
        "integrals": {
            "h1e": encode_array(built["h1e"]),
            "eri": encode_array(built["eri"]),
            "core_energy": built["core_energy"],
            "notation": "chemist (pq|rs); h1e and eri are active-space only",
        },
    }


def stored_mo_coeff(cache: str | Path) -> np.ndarray:
    payload = json.loads(Path(cache).read_text(encoding="utf-8"))
    return decode_array(payload["mo_coeff"])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _describe(payload: dict[str, Any], spec: MoleculeSpec) -> str:
    energies = payload["energies"]
    lines = [
        f"  geometry              linear {spec.terminal_atom}-{spec.central_atom}-"
        f"{spec.terminal_atom}, fixed {payload['fixed_bond_angstrom']:.3f} A, "
        f"scanned {payload['bond_angstrom']:.3f} A",
        f"  basis / orbitals      {payload['basis']} / {payload['orbitals']}",
        f"  active space          CAS({payload['n_active_electrons']}e,"
        f"{payload['n_active_orbitals']}o) -> {payload['n_qubits']} qubits",
        f"  frozen core           {payload['n_core_orbitals']} orbitals, "
        f"core-active gap {payload['core_active_gap_hartree']:.3f} Ha",
        f"  full CAS dimension    {payload['full_cas_determinants']:,} determinants",
        f"  E(RHF)                {energies['rhf_total']:.9f} Ha",
        f"  E({energies['reference_method']})              "
        f"{energies['casci_total']:.9f} Ha",
        f"  correlation in CAS    {energies['active_correlation']:.9f} Ha",
        f"  HF determinant check  {energies['hartree_fock_determinant_error']:+.2e} Ha "
        "(our Slater-Condon vs PySCF RHF)",
    ]
    return "\n".join(lines)


def prepare_main(spec: MoleculeSpec, cache_root: str) -> int:
    parser = argparse.ArgumentParser(
        description="Build and cache the Li2S CAS(12e,12o) active-space Hamiltonian."
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="stretched Li-S bond in angstrom (default: the equilibrium value)",
    )
    parser.add_argument(
        "--orbitals",
        choices=("hf", "casscf"),
        default="hf",
        help="hf reproduces the paper's CASCI reference; casscf relaxes the orbitals",
    )
    parser.add_argument(
        "--continue-from",
        type=float,
        default=None,
        help="reuse the converged orbitals cached at this distance as the SCF guess",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="report the orbital window and build nothing",
    )
    parser.add_argument("--allow-bad-core", action="store_true")
    arguments = parser.parse_args()

    distance = (
        spec.equilibrium_bond_angstrom if arguments.distance is None else arguments.distance
    )
    guess = None
    if arguments.continue_from is not None:
        previous = cache_path_for(cache_root, arguments.continue_from)
        if not previous.is_file():
            print(f"No cache at {previous} to continue from.")
            return 1
        guess = stored_mo_coeff(previous)

    if arguments.diagnose:
        return _diagnose(spec, distance)

    built = build_active_space(
        spec,
        distance,
        orbitals=arguments.orbitals,
        guess_mo=guess,
        allow_bad_core=arguments.allow_bad_core,
    )
    payload = cache_payload(spec, built, distance, arguments.orbitals)
    target = cache_path_for(cache_root, distance)
    write_json_atomic(target, payload)

    print(f"{spec.name}: cached to {target}")
    print(_describe(payload, spec))
    if spec.published_reference:
        print("\n  Published reference for this active space:")
        for key, value in spec.published_reference.items():
            print(f"    {key:22s} {value}")
    print("\nNext: python run.py validate --distance %.3f" % distance)
    return 0


def _diagnose(spec: MoleculeSpec, distance: float) -> int:
    from pyscf import scf

    molecule = build_molecule(spec, distance)
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-11
    mean_field.kernel()
    n_electrons = int(molecule.nelectron)
    n_core = (n_electrons - spec.n_active_electrons) // 2
    n_occupied = n_electrons // 2
    active = range(n_core, n_core + spec.n_active_orbitals)

    overlap = molecule.intor_symmetric("int1e_ovlp")
    slices = molecule.aoslice_by_atom()
    print(f"{spec.name} at {distance:.3f} A, basis {spec.basis}")
    print(f"  {n_electrons} electrons, {mean_field.mo_coeff.shape[1]} orbitals, "
          f"{n_occupied} occupied, frozen core = {n_core}")
    print(f"  E(RHF) = {mean_field.e_tot:.9f} Ha, converged = {mean_field.converged}")
    print()
    print(f"  {'MO':>4} {'occ':>5} {'energy (Ha)':>14}  " + "  ".join(
        f"{molecule.atom_symbol(a)}{a}".rjust(7) for a in range(molecule.natm)
    ) + "   window")
    for index in range(min(mean_field.mo_coeff.shape[1], n_core + spec.n_active_orbitals + 3)):
        vector = np.asarray(mean_field.mo_coeff[:, index])
        gross = vector * (overlap @ vector)
        populations = [
            gross[int(slices[a, 2]) : int(slices[a, 3])].sum()
            for a in range(molecule.natm)
        ]
        marker = "<== active" if index in active else (
            "    core" if index < n_core else ""
        )
        print(
            f"  {index:>4} {2 if index < n_occupied else 0:>5} "
            f"{mean_field.mo_energy[index]:>14.6f}  "
            + "  ".join(f"{value:>7.3f}" for value in populations)
            + f"   {marker}"
        )
    print("\nNothing was built. Drop --diagnose to cache this active space.")
    return 0
