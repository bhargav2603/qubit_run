#!/usr/bin/env python3
"""The classical benchmark ladder, and the TS/FAR energy difference.

Two things the VQE result is meaningless without.

**The ladder.** A single CASCI number has no context. Running HF, MP2, CCSD,
CCSD(T) and CASCI on the identical geometry and basis shows how far the chosen
active space is from converged chemistry. CCSD(T) over the full orbital space is
the practical gold standard here; CASCI(ne,no) is exact *within its active
space* and is what the VQE is benchmarked against. The gap between those two is
the honest measure of how much the active-space truncation costs -- and it is
almost always far larger than the VQE's error against CASCI.

**The difference.** Absolute energies of a 15-atom anion are not interesting and
not comparable to anything. The transition-state-like geometry minus the
separated-reactant geometry is. Systematic errors cancel in the difference,
which is why the paper reports barriers rather than single points.

Caveat stated plainly: the two geometries here are rigid, so this is a
rigid-scan energy difference along an approach coordinate, not a relaxed
reaction barrier. It is a benchmark quantity, not a kinetic prediction.

This module needs PySCF and therefore does not run on Windows. See the README.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from chemistry import (
    MoleculeSpec,
    require_pyscf,
    run_mean_field,
    spec_fingerprint,
    workflow_fingerprint,
    write_json_atomic,
)

HARTREE_TO_KCAL = 627.5094740631


def classical_ladder(
    spec: MoleculeSpec, methods: tuple[str, ...] = ("HF", "MP2", "CCSD", "CCSD(T)", "CASCI")
) -> dict[str, Any]:
    """Run the requested classical methods on one geometry.

    Every method reuses the same converged mean field, so differences between
    them are correlation treatment alone, not a different reference state.
    """
    require_pyscf()
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}

    started = time.perf_counter()
    molecule, mean_field, hf_energy = run_mean_field(spec)
    timings["HF"] = time.perf_counter() - started
    results["HF"] = hf_energy

    if "MP2" in methods:
        from pyscf import mp

        started = time.perf_counter()
        results["MP2"] = float(mp.MP2(mean_field).run().e_tot)
        timings["MP2"] = time.perf_counter() - started

    if "CCSD" in methods or "CCSD(T)" in methods:
        from pyscf import cc

        started = time.perf_counter()
        coupled_cluster = cc.CCSD(mean_field)
        coupled_cluster.conv_tol = 1.0e-9
        coupled_cluster.max_cycle = 200
        coupled_cluster.kernel()
        timings["CCSD"] = time.perf_counter() - started
        if not coupled_cluster.converged:
            results["CCSD"] = None
            results["CCSD_note"] = "CCSD did not converge"
        else:
            if "CCSD" in methods:
                results["CCSD"] = float(coupled_cluster.e_tot)
            if "CCSD(T)" in methods:
                started = time.perf_counter()
                results["CCSD(T)"] = float(
                    coupled_cluster.e_tot + coupled_cluster.ccsd_t()
                )
                timings["CCSD(T)"] = time.perf_counter() - started

    if "CASCI" in methods:
        from pyscf import mcscf

        from chemistry import _validate_spec, select_orbitals

        started = time.perf_counter()
        n_mos = int(np.asarray(mean_field.mo_coeff).shape[1])
        ncore = _validate_spec(spec, int(molecule.nelectron), n_mos)
        mo_coeff, _ = select_orbitals(spec, mean_field, ncore)
        cas = mcscf.CASCI(mean_field, spec.n_active_orbitals, spec.n_active_electrons)
        cas.verbose = 0
        results["CASCI"] = float(cas.kernel(np.asarray(mo_coeff))[0])
        timings["CASCI"] = time.perf_counter() - started

    return {
        "spec_name": spec.name,
        "spec_sha256": spec_fingerprint(spec),
        "basis": spec.basis,
        "charge": spec.charge,
        "solvent_epsilon": spec.solvent_epsilon,
        "orbital_selection": spec.orbital_selection,
        "active_space": f"({spec.n_active_electrons}e,{spec.n_active_orbitals}o)",
        "qubits": 2 * spec.n_active_orbitals,
        "total_electrons": int(molecule.nelectron),
        "total_spatial_orbitals": int(np.asarray(mean_field.mo_coeff).shape[1]),
        "energies_hartree": results,
        "runtime_seconds": timings,
    }


def _difference_table(ts: dict[str, Any], far: dict[str, Any]) -> dict[str, Any]:
    """TS minus FAR for every method both geometries share."""
    rows = {}
    for method in ("HF", "MP2", "CCSD", "CCSD(T)", "CASCI"):
        a = ts["energies_hartree"].get(method)
        b = far["energies_hartree"].get(method)
        if a is None or b is None:
            continue
        delta = float(a) - float(b)
        rows[method] = {
            "ts_hartree": float(a),
            "far_hartree": float(b),
            "delta_hartree": delta,
            "delta_kcal_per_mol": delta * HARTREE_TO_KCAL,
        }
    return rows


def ladder_main(make_spec: Any, default_result: str) -> int:
    parser = argparse.ArgumentParser(
        description="Classical benchmark ladder and TS-FAR energy difference"
    )
    parser.add_argument("--active-space", default="8q", help="8q, 12q or 16q")
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument(
        "--orbital-selection", default="mp2_natural",
        choices=("mp2_natural", "canonical_hf_frontier"),
    )
    parser.add_argument(
        "--solvent-epsilon", type=float, default=4.0,
        help="C-PCM dielectric constant. Pass a negative value for vacuum.",
    )
    parser.add_argument(
        "--methods", default="HF,MP2,CCSD,CCSD(T),CASCI",
        help="Comma-separated subset. CCSD(T) on aug-cc-pVDZ is the expensive one.",
    )
    parser.add_argument(
        "--geometries", default="ts,far",
        help="Comma-separated. Both are needed for the energy difference.",
    )
    parser.add_argument("--result", type=Path, default=Path(default_result))
    args = parser.parse_args()

    # Fail fast and with instructions, before printing a header that suggests
    # anything is about to run.
    require_pyscf()

    epsilon = None if args.solvent_epsilon < 0 else args.solvent_epsilon
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]

    payload: dict[str, Any] = {
        "workflow_sha256": workflow_fingerprint(),
        "methods": list(methods),
        "geometries": {},
    }
    print("=" * 78)
    print("Classical benchmark ladder")
    print("=" * 78)
    for geometry in geometries:
        spec = make_spec(
            geometry=geometry,
            active_space=args.active_space,
            basis=args.basis,
            orbital_selection=args.orbital_selection,
            solvent_epsilon=epsilon,
        )
        print(f"\n[{geometry.upper()}] {spec.name}", flush=True)
        outcome = classical_ladder(spec, methods)
        payload["geometries"][geometry] = outcome
        for method, energy in outcome["energies_hartree"].items():
            if isinstance(energy, float):
                seconds = outcome["runtime_seconds"].get(method)
                suffix = f"   [{seconds:.1f} s]" if seconds else ""
                print(f"  {method:<9} {energy:>20.10f} Ha{suffix}", flush=True)

    if "ts" in payload["geometries"] and "far" in payload["geometries"]:
        differences = _difference_table(
            payload["geometries"]["ts"], payload["geometries"]["far"]
        )
        payload["ts_minus_far"] = differences
        print("\n" + "-" * 78)
        print("TS - FAR (rigid-scan energy difference, NOT a relaxed barrier)")
        print("-" * 78)
        print(f"  {'method':<9} {'delta / Ha':>16} {'delta / kcal/mol':>20}")
        for method, row in differences.items():
            print(
                f"  {method:<9} {row['delta_hartree']:>16.10f}"
                f" {row['delta_kcal_per_mol']:>20.4f}"
            )
        gold = differences.get("CCSD(T)")
        cas = differences.get("CASCI")
        if gold and cas:
            gap = (cas["delta_kcal_per_mol"] - gold["delta_kcal_per_mol"])
            payload["active_space_truncation_error_kcal_per_mol"] = gap
            print(
                f"\n  CASCI - CCSD(T) on the difference: {gap:+.4f} kcal/mol"
                "\n  ^ this is the cost of the active-space truncation. The VQE's"
                "\n    error against CASCI must be read against this number, not"
                "\n    against zero."
            )

    write_json_atomic(args.result, payload)
    print(f"\nWritten: {args.result}")
    return 0
