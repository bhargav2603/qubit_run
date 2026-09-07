#!/usr/bin/env python3
"""Tabulate every result in results/ -- single points, scans and baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CHEMICAL_ACCURACY_MHA = 1.6


def load_results(root: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Return (single-point runs, scans, classical baselines)."""
    root = Path(root)
    singles: list[dict[str, Any]] = []
    scans: list[dict[str, Any]] = []
    classical: dict[str, Any] = {}
    if not root.is_dir():
        return singles, scans, classical
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["_file"] = path.name
        if path.name == "classical.json":
            classical = payload
        elif path.name.startswith("scan"):
            scans.append(payload)
        elif path.name.startswith("hivqe"):
            singles.append(payload)
    return singles, scans, classical


def _pt2_column(run: dict[str, Any]) -> str:
    """The perturbatively corrected error, or a dash when it was not computed.

    Shown beside the variational error, never instead of it: PT2 is not
    variational, so the upper-bound guarantee applies to the other column.
    """
    if not run.get("pt2_determinants"):
        return "--"
    value = f"{run.get('error_pt2_millihartree', float('nan')):.4f}"
    # A correction bigger than the correlation the subspace captured is outside
    # the domain of perturbation theory. Marked, not hidden.
    return value if run.get("pt2_reliable", True) else value + "!"


def _label(run: dict[str, Any]) -> str:
    settings = run.get("settings", {})
    parts = [settings.get("simulator", "?")]
    if not settings.get("use_expansion", True):
        parts.append("no-expansion")
    if settings.get("readout_error") or settings.get("depolarizing_error"):
        parts.append("noisy")
    if settings.get("optimizer") == "none":
        parts.append("no-opt")
    return "+".join(parts)


def summary_main(results_root: str) -> int:
    parser = argparse.ArgumentParser(description="Tabulate the results directory.")
    parser.add_argument(
        "--sort",
        choices=("file", "error", "distance", "determinants"),
        default="distance",
    )
    arguments = parser.parse_args()

    singles, scans, classical = load_results(results_root)
    if not singles and not scans and not classical:
        print(f"No results in {results_root}. Run `python run.py hivqe` first.")
        return 1

    if singles:
        keys = {
            "file": lambda run: run["_file"],
            "error": lambda run: abs(run.get("error_millihartree", 1e9)),
            "distance": lambda run: run.get("bond_angstrom", 0.0),
            "determinants": lambda run: run.get("dimension", 0),
        }
        singles.sort(key=keys[arguments.sort])
        print("Single-point HI-VQE runs")
        header = (
            f"{'r (A)':>7} {'method':>22} {'energy (Ha)':>17} {'err (mHa)':>10} "
            f"{'+PT2 (mHa)':>11} {'dets':>9} {'%space':>8} {'corr %':>8} "
            f"{'iters':>6} {'s':>7}  verdict"
        )
        print(header)
        print("-" * len(header))
        for run in singles:
            print(
                f"{run.get('bond_angstrom', float('nan')):>7.3f} "
                f"{_label(run):>22} "
                f"{run.get('energy', float('nan')):>17.9f} "
                f"{run.get('error_millihartree', float('nan')):>10.4f} "
                f"{_pt2_column(run):>11} "
                f"{run.get('dimension', 0):>9,} "
                f"{100 * run.get('subspace_fraction', 0):>7.3f}% "
                f"{100 * run.get('correlation_recovered', 0):>7.2f}% "
                f"{run.get('iterations', 0):>6} "
                f"{run.get('seconds', 0):>7.1f}  {run.get('verdict', '')}"
            )
        inside = sum(
            1
            for run in singles
            if abs(run.get("error_millihartree", 1e9)) <= CHEMICAL_ACCURACY_MHA
        )
        print(
            f"\n  {inside} of {len(singles)} runs inside chemical accuracy "
            f"({CHEMICAL_ACCURACY_MHA} mHa)"
        )

    for scan in scans:
        print(f"\nDissociation scan  [{scan['_file']}]")
        header = (
            f"{'r (A)':>7} {'E(HF)':>17} {'E(HI-VQE)':>17} "
            f"{'E(' + scan.get('reference_method', 'ref') + ')':>17} "
            f"{'err (mHa)':>10} {'dets':>9}"
        )
        print(header)
        print("-" * len(header))
        for index, distance in enumerate(scan["distances"]):
            print(
                f"{distance:>7.3f} {scan['hartree_fock'][index]:>17.9f} "
                f"{scan['hivqe'][index]:>17.9f} {scan['reference'][index]:>17.9f} "
                f"{scan['error_millihartree'][index]:>10.4f} "
                f"{scan['dimension'][index]:>9,}"
            )
        errors = [abs(value) for value in scan["error_millihartree"]]
        hartree_fock_errors = [
            abs(scan["hartree_fock"][i] - scan["reference"][i]) * 1000
            for i in range(len(scan["distances"]))
        ]
        print(
            f"\n  HI-VQE worst {max(errors):.4f} mHa, mean {sum(errors) / len(errors):.4f} mHa"
        )
        print(
            f"  Hartree-Fock worst {max(hartree_fock_errors):.1f} mHa, "
            f"mean {sum(hartree_fock_errors) / len(hartree_fock_errors):.1f} mHa"
        )
        print(
            f"  inside chemical accuracy: "
            f"{sum(1 for e in errors if e <= CHEMICAL_ACCURACY_MHA)} / {len(errors)}"
        )

    if classical:
        rows = sorted(
            (key, value)
            for key, value in classical.items()
            if not key.startswith("_")
        )
        print("\nClassical baselines in the same active space")
        header = (
            f"{'r (A)':>7} {'RHF':>17} {'MP2':>17} {'CCSD':>17} "
            f"{'CCSD(T)':>17} {'CASCI':>17} {'T1':>7}"
        )
        print(header)
        print("-" * len(header))
        for key, entry in rows:
            print(
                f"{float(key):>7.3f} {entry.get('rhf_total', float('nan')):>17.9f} "
                f"{entry.get('mp2_total', float('nan')):>17.9f} "
                f"{entry.get('ccsd_total', float('nan')):>17.9f} "
                f"{entry.get('ccsd_t_total', float('nan')):>17.9f} "
                f"{entry.get('casci_total', float('nan')):>17.9f} "
                f"{entry.get('t1_diagnostic', float('nan')):>7.4f}"
            )
    return 0
