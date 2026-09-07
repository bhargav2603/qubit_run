#!/usr/bin/env python3
"""Classical wavefunction baselines in exactly the same active space.

A quantum result without a classical scale is a number with no meaning. These
methods are frozen to the identical CAS(12e,12o) window, so the comparison is
between methods and not between models:

* **MP2** -- second-order perturbation theory. Fine near equilibrium, and it is
  supposed to fail as the bond stretches; watching it fail is the point.
* **CCSD / CCSD(T)** -- the classical workhorses. CCSD(T) is "gold standard"
  only while the reference determinant dominates. On a breaking Li-S bond it
  stops being, and the triples correction is where that first becomes visible.
* **CASCI** -- exact in this active space, and therefore the reference every
  error in this folder is measured against.

Needs PySCF, so it runs wherever `prepare` runs. Everything it produces is
written to JSON and read back by `summary`, `plot` and `report` without it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hamiltonian import (
    MoleculeSpec,
    build_active_space,
    cache_path_for,
    load_cache,
    stored_mo_coeff,
    write_json_atomic,
)


def active_space_baselines(
    spec: MoleculeSpec,
    bond_angstrom: float,
    guess_mo: np.ndarray | None = None,
) -> dict[str, Any]:
    """MP2, CCSD and CCSD(T) frozen to the CAS(12e,12o) window."""
    from pyscf import cc, mp

    built = build_active_space(spec, bond_angstrom, orbitals="hf", guess_mo=guess_mo)
    mean_field = built["mean_field"]
    n_core = built["n_core"]
    n_active = spec.n_active_orbitals
    n_mos = int(built["mo_coeff"].shape[1])
    frozen = list(range(n_core)) + list(range(n_core + n_active, n_mos))

    results: dict[str, Any] = {
        "bond_angstrom": float(bond_angstrom),
        "frozen_orbitals": frozen,
        "rhf_total": built["rhf_energy"],
        "casci_total": built["reference_energy"],
    }

    try:
        second_order = mp.MP2(mean_field, frozen=frozen).run()
        results["mp2_total"] = float(second_order.e_tot)
        results["mp2_correlation"] = float(second_order.e_corr)
    except Exception as error:  # pragma: no cover - MP2 rarely fails
        results["mp2_error"] = str(error)

    try:
        coupled = cc.CCSD(mean_field, frozen=frozen)
        coupled.max_cycle = 200
        coupled.conv_tol = 1e-9
        coupled.run()
        results["ccsd_total"] = float(coupled.e_tot)
        results["ccsd_correlation"] = float(coupled.e_corr)
        results["ccsd_converged"] = bool(coupled.converged)
        # The Lee-Taylor T1 diagnostic: ||t1|| / sqrt(N) with the norm taken over
        # SPIN orbitals and N the number of correlated electrons. For a
        # closed-shell RCCSD, PySCF's t1 is the spatial-orbital array of shape
        # (nocc, nvir), whose spin-orbital norm is sqrt(2) times larger, and
        # N = 2 * nocc -- so the two factors of sqrt(2) cancel and the
        # diagnostic is simply ||t1_spatial|| / sqrt(nocc). Above about 0.02 the
        # reference determinant is no longer dominant and the CCSD(T) number
        # below should not be read as a gold standard.
        t1 = np.asarray(coupled.t1)
        results["t1_diagnostic"] = float(
            np.linalg.norm(t1) / np.sqrt(max(t1.shape[0], 1))
        )
        results["max_t1_amplitude"] = float(np.abs(t1).max()) if t1.size else 0.0
        triples = coupled.ccsd_t()
        results["ccsd_t_total"] = float(coupled.e_tot + triples)
        results["triples_correction"] = float(triples)
    except Exception as error:
        results["ccsd_error"] = str(error)

    return results


def classical_main(spec: MoleculeSpec, cache_root: str, results_root: str) -> int:
    parser = argparse.ArgumentParser(
        description="MP2, CCSD and CCSD(T) in the same active space as the quantum run."
    )
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every distance that already has a cached Hamiltonian",
    )
    arguments = parser.parse_args()

    distances: list[float]
    if arguments.all:
        distances = sorted(
            float(load_cache(path).bond_angstrom)
            for path in Path(cache_root).glob("li2s_r*.json")
            if not path.name.endswith(".validated.json")
        )
        if not distances:
            print(f"No cached Hamiltonians in {cache_root}. Run `prepare` first.")
            return 1
    else:
        distances = [
            spec.equilibrium_bond_angstrom
            if arguments.distance is None
            else arguments.distance
        ]

    target = Path(results_root) / "classical.json"
    existing: dict[str, Any] = {}
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))

    print(
        f"{'r (A)':>7} {'RHF':>15} {'MP2':>15} {'CCSD':>15} "
        f"{'CCSD(T)':>15} {'CASCI':>15} {'T1':>7}"
    )
    guess = None
    for distance in distances:
        cached = cache_path_for(cache_root, distance)
        if cached.is_file():
            guess = stored_mo_coeff(cached)
        entry = active_space_baselines(spec, distance, guess_mo=guess)
        existing[f"{distance:.3f}"] = entry
        print(
            f"{distance:>7.3f} {entry['rhf_total']:>15.8f} "
            f"{entry.get('mp2_total', float('nan')):>15.8f} "
            f"{entry.get('ccsd_total', float('nan')):>15.8f} "
            f"{entry.get('ccsd_t_total', float('nan')):>15.8f} "
            f"{entry['casci_total']:>15.8f} "
            f"{entry.get('t1_diagnostic', float('nan')):>7.4f}"
        )

    write_json_atomic(target, existing)
    print(f"\nWritten to {target}")
    print(
        "\nT1 above ~0.02 means the reference determinant is no longer dominant "
        "and CCSD(T) is no longer a gold standard -- which is exactly the regime "
        "this dissociation enters, and why CASCI is the reference here."
    )
    return 0
