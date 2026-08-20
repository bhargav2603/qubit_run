#!/usr/bin/env python3
"""Command drivers: turn a validated cache plus CLI flags into a result file.

`hivqe.py` is the algorithm and knows nothing about files or arguments; this
module is the part that reads a cache, enforces the validation receipt, runs the
thing and writes JSON. Keeping them apart is what lets `selftest.py` exercise the
algorithm with no cache present at all.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hamiltonian import (
    MoleculeSpec,
    cache_path_for,
    load_cache,
    workflow_fingerprint,
    write_json_atomic,
)


# --------------------------------------------------------------------------
# hivqe
# --------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--simulator",
        choices=("sector", "aer", "none"),
        default="sector",
        help="sector: exact, inside the particle-number sector, ~4x faster than "
        "Aer at 24 qubits. aer: Qiskit Aer, the reference path and the only one "
        "that can carry a noise model. none: no quantum layer, the classical "
        "selected-CI control",
    )
    parser.add_argument(
        "--max-determinants",
        type=int,
        default=20000,
        help="largest subspace ever diagonalised; the accuracy/cost dial",
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=400,
        help="candidate configurations the classical single/double step offers "
        "each iteration",
    )
    parser.add_argument(
        "--expansion-references",
        type=int,
        default=8,
        help="leading configurations the excitations are generated from; "
        "`--expansion-references 1 --ranking coupling` reproduces the paper's "
        "selection rule exactly",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=3.0,
        help="amplitude screening prunes to max-determinants/this, leaving the "
        "expansion room to grow back to the cap",
    )
    parser.add_argument(
        "--ranking",
        choices=("pt2", "coupling"),
        default="pt2",
        help="how expansion candidates are scored. pt2: the Epstein-Nesbet "
        "energy gain |<D|H|Psi>|^2 / (E - <D|H|D>), the CIPSI criterion. "
        "coupling: the bare numerator, which is the rule the paper states -- "
        "use it with --expansion-references 1 to reproduce that rule exactly",
    )
    parser.add_argument(
        "--spin-complete",
        action="store_true",
        help="force the alpha and beta string sets to be the same set, so the "
        "subspace is closed under the spin flip. Cuts spin contamination by "
        "about a third at the price of ~10 mHa of variational energy, because "
        "one shared set spends dimension: turn it on when a run comes back "
        "SPIN CONTAMINATED, not by default",
    )
    parser.add_argument(
        "--no-pt2",
        action="store_true",
        help="skip the Epstein-Nesbet correction reported beside the "
        "variational energy (one extra sigma product on an enlarged space)",
    )
    parser.add_argument(
        "--pt2-factor",
        type=float,
        default=4.0,
        help="how many times the working dimension the first-order space may "
        "reach. Larger captures more of the correction and costs more memory",
    )
    parser.add_argument("--amplitude-threshold", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--energy-tolerance", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--init-scale", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimizer", choices=("spsa", "none"), default="spsa")
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument(
        "--optimizer-every",
        type=int,
        default=1,
        help="run SPSA only every Nth iteration. Sampling dominates the wall "
        "time and SPSA triples it, so 3 is a good speed/quality trade",
    )
    parser.add_argument(
        "--givens-depth",
        type=int,
        default=None,
        help="Givens brickwork depth per network (default: the orbital count, "
        "which is what spans the whole active space). Halving it roughly halves "
        "the sampling cost, at the price of proposals that cannot reach distant "
        "orbitals",
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="ablation: no classical single/double expansion, quantum proposals only",
    )
    parser.add_argument("--readout-error", type=float, default=0.0)
    parser.add_argument("--depolarizing-error", type=float, default=0.0)
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="discard out-of-sector samples instead of repairing them",
    )
    parser.add_argument("--tag", default="", help="suffix for the result filename")


def _settings_from(arguments: argparse.Namespace):
    from hivqe import HiVqeSettings

    return HiVqeSettings(
        max_determinants=arguments.max_determinants,
        expansion=arguments.expansion,
        growth_factor=arguments.growth_factor,
        expansion_references=arguments.expansion_references,
        amplitude_threshold=arguments.amplitude_threshold,
        max_iterations=arguments.max_iterations,
        energy_tolerance=arguments.energy_tolerance,
        patience=arguments.patience,
        simulator=arguments.simulator,
        reps=arguments.reps,
        shots=arguments.shots,
        init_scale=arguments.init_scale,
        seed=arguments.seed,
        readout_error=arguments.readout_error,
        depolarizing_error=arguments.depolarizing_error,
        recover=not arguments.no_recovery,
        optimizer=arguments.optimizer,
        optimizer_steps=arguments.optimizer_steps,
        optimizer_every=arguments.optimizer_every,
        givens_depth=arguments.givens_depth,
        use_expansion=not arguments.no_expansion,
        ranking=arguments.ranking,
        spin_complete=arguments.spin_complete,
        pt2=not arguments.no_pt2,
        pt2_factor=arguments.pt2_factor,
    )


def _result_name(bond: float, arguments: argparse.Namespace) -> str:
    parts = [f"hivqe_r{bond:.3f}", arguments.simulator]
    if arguments.no_expansion:
        parts.append("noexp")
    if arguments.readout_error or arguments.depolarizing_error:
        parts.append("noisy")
    if arguments.tag:
        parts.append(arguments.tag)
    return "_".join(parts) + ".json"


def run_one(
    spec: MoleculeSpec,
    cache_root: str,
    results_root: str,
    bond: float,
    arguments: argparse.Namespace,
    quiet: bool = False,
) -> dict[str, Any]:
    from hivqe import run_hivqe
    from validation import require_validation_receipt

    cache = cache_path_for(cache_root, bond)
    if not cache.is_file():
        raise FileNotFoundError(
            f"No cached Hamiltonian at {cache}. Run "
            f"`python run.py prepare --distance {bond:.3f}` where PySCF is installed."
        )
    receipt = require_validation_receipt(cache, spec)
    cached = load_cache(cache)

    progress = (lambda _line: None) if quiet else print
    if not quiet:
        print(
            f"{spec.name} r = {cached.bond_angstrom:.3f} A | "
            f"CAS({cached.metadata['n_active_electrons']}e,"
            f"{cached.metadata['n_active_orbitals']}o) | "
            f"{cached.metadata['n_qubits']} qubits | "
            f"{cached.space.full_dimension:,} determinants"
        )
        print(
            f"  E(RHF) = {cached.hartree_fock_energy:.9f}   "
            f"E({cached.reference_method}) = {cached.reference_energy:.9f}   "
            f"correlation = {cached.reference_energy - cached.hartree_fock_energy:.6f} Ha"
        )

    settings = _settings_from(arguments)
    result = run_hivqe(
        cached.space,
        cached.reference_energy,
        cached.hartree_fock_energy,
        settings,
        progress=progress,
    )

    payload = result.as_dict()
    payload.update(
        {
            "molecule": spec.name,
            "bond_angstrom": cached.bond_angstrom,
            "basis": cached.metadata["basis"],
            "orbitals": cached.metadata["orbitals"],
            "n_qubits": cached.metadata["n_qubits"],
            "n_active_electrons": cached.metadata["n_active_electrons"],
            "n_active_orbitals": cached.metadata["n_active_orbitals"],
            "reference_method": cached.reference_method,
            # Which spin state the reference landed on. Along a dissociation
            # coordinate this is not constant, and an error is only meaningful
            # once you know the two numbers came from the same sector.
            "reference_spin_squared": cached.metadata["energies"].get(
                "reference_spin_squared"
            ),
            "cache": cache.name,
            "cache_sha256": receipt["cache_sha256"],
            "workflow_sha256": workflow_fingerprint(),
            "published_reference": spec.published_reference,
        }
    )
    target = Path(results_root) / _result_name(cached.bond_angstrom, arguments)
    write_json_atomic(target, payload)

    if not quiet:
        _print_verdict(result, target)
    return payload


def _print_verdict(result, target: Path) -> None:
    print()
    print(f"  verdict               {result.verdict}")
    print(f"  energy                {result.energy:.9f} Ha")
    print(
        f"  error vs reference    {result.error_millihartree:+.4f} mHa   "
        f"(chemical accuracy is 1.6 mHa)"
    )
    if result.pt2_determinants:
        print(
            f"  + PT2 correction      {result.pt2_correction * 1000:+.4f} mHa "
            f"over {result.pt2_determinants:,} determinants outside the subspace"
        )
        print(
            f"  energy + PT2          {result.energy_pt2:.9f} Ha   "
            f"error {result.error_pt2_hartree * 1000:+.4f} mHa   "
            f"({'not variational' if result.pt2_reliable else 'NOT USABLE -- see above'})"
        )
    print(
        f"  determinants used     {result.dimension:,} of "
        f"{result.full_dimension:,}  ({100 * result.subspace_fraction:.3f}%)"
    )
    print(f"  correlation recovered {100 * result.correlation_recovered:.3f}%")
    print(
        f"  <S^2>                 {result.spin_squared:.6f}   "
        f"(off the nearest S(S+1) by {result.spin_contamination:.2e})"
    )
    print(f"  iterations            {result.iterations}  [{result.stop_reason}]")
    print(f"  wall time             {result.seconds:.1f} s")
    if result.circuit_metrics:
        metrics = result.circuit_metrics
        print(
            f"  circuit               {metrics.get('qubits')} qubits, depth "
            f"{metrics.get('depth')}, {metrics.get('two_qubit_gates')} two-qubit "
            f"gates, {metrics.get('parameters')} parameters"
        )
        if "transpiled_two_qubit_gates" in metrics:
            print(
                f"  transpiled            depth {metrics['transpiled_depth']}, "
                f"{metrics['transpiled_two_qubit_gates']} CZ on a linear chain"
            )
    print(f"\n  written to {target}")


def hivqe_main(spec: MoleculeSpec, cache_root: str, results_root: str) -> int:
    parser = argparse.ArgumentParser(
        description="Run HI-VQE on a validated Li2S Hamiltonian."
    )
    parser.add_argument("--distance", type=float, default=None)
    _add_common_arguments(parser)
    arguments = parser.parse_args()
    bond = (
        spec.equilibrium_bond_angstrom
        if arguments.distance is None
        else arguments.distance
    )
    try:
        run_one(spec, cache_root, results_root, bond, arguments)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"\n{error}")
        return 1
    return 0


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------


def scan_main(
    spec: MoleculeSpec, cache_root: str, results_root: str, distances: tuple[float, ...]
) -> int:
    parser = argparse.ArgumentParser(
        description="Run HI-VQE along the Li-S dissociation coordinate."
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=list(distances),
        help="bond lengths in angstrom",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="run this many geometries at once. Each worker is a separate "
        "process holding its own copy of the Hamiltonian (a few hundred MB at "
        "24 qubits), so 2-3 is right for a free Colab runtime",
    )
    _add_common_arguments(parser)
    arguments = parser.parse_args()

    rows: list[dict[str, Any]] = []
    started = time.time()
    header = (
        f"{'r (A)':>7} {'E(HI-VQE)':>17} {'E(ref)':>17} {'err (mHa)':>11} "
        f"{'dets':>9} {'%space':>8} {'S2':>6} {'refS2':>6} {'verdict':>26}"
    )

    def show(bond: float, payload: dict[str, Any]) -> None:
        reference_spin = payload.get("reference_spin_squared")
        reference_text = "  --  " if reference_spin is None else f"{reference_spin:>6.3f}"
        print(
            f"{bond:>7.3f} {payload['energy']:>17.9f} "
            f"{payload['reference_energy']:>17.9f} "
            f"{payload['error_millihartree']:>11.4f} "
            f"{payload['dimension']:>9,} "
            f"{100 * payload['subspace_fraction']:>7.3f}% "
            f"{payload['spin_squared']:>6.3f} {reference_text} "
            f"{payload['verdict']:>26}"
        )

    workers = max(1, int(arguments.workers))
    print(header)
    print("-" * len(header))

    if workers > 1:
        # Geometries are completely independent -- separate caches, separate
        # results files, no shared state -- so the scan is the one place in this
        # workflow with free parallelism. Processes rather than threads because
        # the expensive parts are Python-level loops in the sampler and the
        # expansion, which a thread pool could not overlap.
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_one, spec, cache_root, results_root, bond, arguments, True
                ): bond
                for bond in arguments.distances
            }
            for future, bond in sorted(futures.items(), key=lambda item: item[1]):
                try:
                    payload = future.result()
                except (FileNotFoundError, RuntimeError) as error:
                    print(f"{bond:>7.3f}  skipped: {error}")
                    continue
                rows.append(payload)
                show(bond, payload)
        rows.sort(key=lambda row: row["bond_angstrom"])
    else:
        for bond in arguments.distances:
            try:
                payload = run_one(
                    spec, cache_root, results_root, bond, arguments, quiet=True
                )
            except (FileNotFoundError, RuntimeError) as error:
                print(f"{bond:>7.3f}  skipped: {error}")
                continue
            rows.append(payload)
            show(bond, payload)

    if not rows:
        print("\nNothing ran. Build and validate the caches first.")
        return 1

    errors = np.array([abs(row["error_millihartree"]) for row in rows])
    curve = {
        "molecule": spec.name,
        "distances": [row["bond_angstrom"] for row in rows],
        "hivqe": [row["energy"] for row in rows],
        "reference": [row["reference_energy"] for row in rows],
        "hartree_fock": [row["hartree_fock_energy"] for row in rows],
        "error_millihartree": [row["error_millihartree"] for row in rows],
        # Carried alongside, never instead of, the variational curve.
        "hivqe_pt2": [row.get("energy_pt2", row["energy"]) for row in rows],
        "error_pt2_millihartree": [
            row.get("error_pt2_millihartree", row["error_millihartree"])
            for row in rows
        ],
        "spin_squared": [row["spin_squared"] for row in rows],
        "reference_spin_squared": [
            row.get("reference_spin_squared") for row in rows
        ],
        "dimension": [row["dimension"] for row in rows],
        "subspace_fraction": [row["subspace_fraction"] for row in rows],
        "verdict": [row["verdict"] for row in rows],
        "settings": rows[0]["settings"],
        "n_qubits": rows[0]["n_qubits"],
        "reference_method": rows[0]["reference_method"],
        "full_dimension": rows[0]["full_dimension"],
        "workflow_sha256": workflow_fingerprint(),
        "seconds": time.time() - started,
    }
    target = Path(results_root) / (
        f"scan_{arguments.simulator}{'_' + arguments.tag if arguments.tag else ''}.json"
    )
    write_json_atomic(target, curve)

    print()
    print(f"  points inside chemical accuracy   {int((errors <= 1.6).sum())} / {len(errors)}")
    print(f"  worst error                       {errors.max():.4f} mHa")
    print(f"  mean error                        {errors.mean():.4f} mHa")
    print(f"  total wall time                   {curve['seconds'] / 60:.1f} min")
    print(f"\n  written to {target}")
    return 0


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def geometry_main(spec: MoleculeSpec, cache_root: str) -> int:
    """Locate the equilibrium bond length on this folder's own CASCI surface."""
    parser = argparse.ArgumentParser(
        description="Symmetric-stretch scan to find the Li2S equilibrium bond length."
    )
    parser.add_argument("--start", type=float, default=1.90)
    parser.add_argument("--stop", type=float, default=2.40)
    parser.add_argument("--step", type=float, default=0.05)
    arguments = parser.parse_args()

    from hamiltonian import build_active_space

    distances = np.arange(
        arguments.start, arguments.stop + 0.5 * arguments.step, arguments.step
    )
    print(f"{'r (A)':>7} {'E(RHF)':>17} {'E(CASCI)':>17}")
    print("-" * 43)
    energies = []
    guess = None
    for bond in distances:
        # Both bonds move together, so the *fixed* bond has to move too --
        # which means replacing it in the spec, not just passing a new scanned
        # distance. Getting this wrong would scan a bond-stretch coordinate and
        # report it as an equilibrium geometry.
        built = build_active_space(
            _symmetric_spec(spec, float(bond)),
            float(bond),
            orbitals="hf",
            guess_mo=guess,
            allow_bad_core=True,
        )
        guess = built["mo_coeff"]
        energies.append((float(bond), built["rhf_energy"], built["reference_energy"]))
        print(f"{bond:>7.3f} {built['rhf_energy']:>17.9f} {built['reference_energy']:>17.9f}")

    values = np.array([entry[2] for entry in energies])
    best = int(np.argmin(values))
    print()
    print(f"  minimum on this grid: r = {energies[best][0]:.3f} A, "
          f"E(CASCI) = {values[best]:.9f} Ha")
    if 0 < best < len(values) - 1:
        # Parabolic refinement through the three points around the minimum.
        left, middle, right = values[best - 1], values[best], values[best + 1]
        step = energies[1][0] - energies[0][0]
        shift = 0.5 * step * (left - right) / (left - 2 * middle + right)
        print(f"  parabolic minimum:    r = {energies[best][0] + shift:.4f} A")
    print(
        f"\n  molecule.py currently uses "
        f"EQUILIBRIUM_BOND_ANGSTROM = {spec.equilibrium_bond_angstrom}"
    )
    print("  Edit it if this scan disagrees, then rebuild the caches.")
    return 0


def _symmetric_spec(spec: MoleculeSpec, bond: float) -> MoleculeSpec:
    from dataclasses import replace

    return replace(spec, equilibrium_bond_angstrom=float(bond))


# --------------------------------------------------------------------------
# paulis
# --------------------------------------------------------------------------


def paulis_main(spec: MoleculeSpec, cache_root: str) -> int:
    """Count the Jordan-Wigner Pauli words a conventional VQE would measure.

    The HI-VQE paper's headline cost comparison for this exact system is that
    Li2S at 24 qubits needs 15,697 Pauli-word measurements in a conventional
    VQE, against one measurement per iteration for HI-VQE. That number is
    checkable, so this checks it.
    """
    parser = argparse.ArgumentParser(description="Count JW Pauli terms for the cache.")
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    arguments = parser.parse_args()

    bond = (
        spec.equilibrium_bond_angstrom
        if arguments.distance is None
        else arguments.distance
    )
    cache = cache_path_for(cache_root, bond)
    if not cache.is_file():
        print(f"No cached Hamiltonian at {cache}.")
        return 1
    space = load_cache(cache).space

    try:
        from openfermion import InteractionOperator, jordan_wigner
    except ImportError:
        print("This command needs OpenFermion:  pip install openfermion")
        return 1

    m = space.n_orbitals
    n_spin_orbitals = 2 * m
    one = np.zeros((n_spin_orbitals, n_spin_orbitals))
    two = np.zeros((n_spin_orbitals,) * 4)
    for p in range(m):
        for q in range(m):
            for spin in range(2):
                one[p + spin * m, q + spin * m] = space.h1e[p, q]
    for p in range(m):
        for q in range(m):
            for r in range(m):
                for s in range(m):
                    value = 0.5 * space.eri[p, q, r, s]
                    if value == 0.0:
                        continue
                    for sp in range(2):
                        for sq in range(2):
                            two[
                                p + sp * m, r + sq * m, s + sq * m, q + sp * m
                            ] += value

    print(f"Jordan-Wigner decomposition of Li2S CAS({space.n_electrons}e,{m}o) "
          f"at r = {bond:.3f} A ...")
    operator = jordan_wigner(InteractionOperator(space.core_energy, one, two))
    terms = {
        key: value
        for key, value in operator.terms.items()
        if abs(value) > arguments.cutoff
    }
    identity = sum(1 for key in terms if len(key) == 0)
    weights: dict[int, int] = {}
    for key in terms:
        weights[len(key)] = weights.get(len(key), 0) + 1

    published = (spec.published_reference or {}).get("pauli_words_for_conventional_vqe")
    print(f"  qubits                      {n_spin_orbitals}")
    print(f"  Pauli words (|coef| > {arguments.cutoff:g})   {len(terms):,}")
    print(f"  of which identity           {identity}")
    print(f"  non-identity words          {len(terms) - identity:,}")
    for weight in sorted(weights):
        print(f"    weight {weight:>2}                 {weights[weight]:,}")
    nonzero = int(np.count_nonzero(np.abs(space.eri) > arguments.cutoff))
    print(
        f"  non-zero two-electron integrals  {nonzero:,} of {space.eri.size:,} "
        f"({100 * nonzero / space.eri.size:.1f}%)"
    )
    if published:
        print(f"\n  arXiv:2503.06292 reports    {published:,} Pauli words for Li2S")
        print(f"  difference                  {len(terms) - identity - published:+,}")
        print(
            "\n  What drives that difference is the line above it. The Pauli-word\n"
            "  count is set almost entirely by how many two-electron integrals are\n"
            "  exactly zero, and that is decided by the molecular point group -- a\n"
            "  linear Li-S-Li has a great many. A count taken on integrals with no\n"
            "  symmetry zeros is an upper bound, not a discrepancy. Two knobs move\n"
            "  it: --cutoff (screening) and the orbital set the cache was built in."
        )
    print(
        "\n  A conventional VQE measures every one of these, every iteration. "
        "HI-VQE measures the circuit once and diagonalises classically."
    )
    return 0
