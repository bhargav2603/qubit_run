#!/usr/bin/env python3
"""Tabulate every result, so a layer scan and an ADAPT run are read side by side.

Two methods write into `results/` and they are not the same shape: the
hardware-efficient VQE has a layer count, a simulator method and an MPS
truncation error, while ADAPT-VQE has an operator pool, a gate count and no
simulator at all. Rather than two tables, both are folded onto the columns they
genuinely share -- what the ansatz was, how many parameters it had, the energy,
the error against CASCI, the symmetry contamination, and *one independent
re-evaluation of the same state* -- because that last column is the one that
decides whether a row can be believed, and each method has exactly one of them:

* the hardware-efficient rows re-evaluate the converged MPS state with an exact
  statevector, so the column is MPS truncation error;
* the ADAPT rows re-evaluate the emitted Qiskit circuit against the sparse
  algebra that optimized it, so the column is circuit synthesis error.

Both are "exact recomputation minus reported energy", both must sit inside the
same 0.16 mHa budget, and both are zero when nothing was approximated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERDICT = {
    "verified_chemical_accuracy": "VERIFIED",
    "outside_chemical_accuracy": "outside",
    "symmetry_broken": "SYMMETRY BROKEN",
    "spin_contaminated": "SPIN CONTAMINATED",
    "above_hartree_fock": "ABOVE HF",
    "mps_truncation_too_coarse": "MPS TOO COARSE",
    "circuit_disagrees": "CIRCUIT DISAGREES",
    "inconsistent": "INCONSISTENT",
}
METHOD = {"matrix_product_state": "mps", "statevector": "sv"}


def _ansatz_label(ansatz: dict[str, Any], simulator: dict[str, Any]) -> str:
    """One column that says what was actually run, for either method."""
    if ansatz.get("pool"):
        return f"adapt/{ansatz['pool']}"
    method = METHOD.get(simulator.get("method", ""), simulator.get("method", ""))
    layers = ansatz.get("layers")
    depth = "?" if layers is None else str(layers)
    return f"hea-L{depth}" + (f"/{method}" if method else "")


def _row(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        vqe = payload["vqe"]
        ansatz = payload["ansatz"]
    except (OSError, ValueError, KeyError):
        return None
    simulator = payload.get("simulator") or {}
    optimizer = payload.get("optimizer") or {}
    circuit = payload.get("circuit") or {}
    contamination = vqe.get("contamination_energy_hartree")
    # The independent re-evaluation of the converged state -- MPS truncation for
    # the hardware-efficient rows, circuit synthesis for the ADAPT rows.
    crosscheck = simulator.get("mps_vs_statevector_hartree")
    if crosscheck is None:
        crosscheck = circuit.get("energy_error_hartree")
    return {
        "file": path.name,
        "ansatz": _ansatz_label(ansatz, simulator),
        "parameters": ansatz.get("parameters"),
        "energy": vqe.get("physical_energy_hartree"),
        "error_mha": vqe.get("error_millihartree"),
        "contamination_mha": None if contamination is None else 1000.0 * contamination,
        "crosscheck_mha": None if crosscheck is None else 1000.0 * crosscheck,
        "two_qubit_gates": circuit.get("two_qubit_gates"),
        "iterations": optimizer.get("iterations"),
        "runtime_s": payload.get("runtime_seconds"),
        "status": VERDICT.get(payload.get("status", ""), payload.get("status", "?")),
    }


def _cell(value: Any, spec: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def summary_main(default_directory: str) -> int:
    parser = argparse.ArgumentParser(description="Summarize VQE and ADAPT-VQE results")
    parser.add_argument("--results", type=Path, default=Path(default_directory))
    parser.add_argument(
        "--sort",
        choices=("error", "runtime", "parameters", "file"),
        default="error",
        help="Order rows by absolute CASCI error (default), runtime, ansatz size, "
        "or filename.",
    )
    args = parser.parse_args()

    if not args.results.is_dir():
        print(f"No results directory yet: {args.results}")
        return 0
    rows = [row for row in (_row(path) for path in sorted(args.results.glob("*.json"))) if row]
    if not rows:
        print(f"No readable results in {args.results}")
        return 0

    if args.sort == "error":
        rows.sort(key=lambda row: (row["error_mha"] is None, abs(row["error_mha"] or 0.0)))
    elif args.sort == "runtime":
        rows.sort(key=lambda row: (row["runtime_s"] is None, row["runtime_s"] or 0.0))
    elif args.sort == "parameters":
        rows.sort(key=lambda row: (row["parameters"] is None, row["parameters"] or 0))

    header = (
        f"{'result file':<24} {'ansatz':<14} {'par':>4} {'energy (Ha)':>17}"
        f" {'err (mHa)':>11} {'contam':>9} {'xcheck':>10} {'2q':>5} {'iter':>5}"
        f" {'runtime':>9}  status"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['file']:<24} {row['ansatz']:<14} {_cell(row['parameters'], 'd'):>4} "
            f"{_cell(row['energy'], '.9f'):>17} {_cell(row['error_mha'], '+.3f'):>11} "
            f"{_cell(row['contamination_mha'], '.4f'):>9} "
            f"{_cell(row['crosscheck_mha'], '+.4f'):>10} "
            f"{_cell(row['two_qubit_gates'], 'd'):>5} {_cell(row['iterations'], 'd'):>5} "
            f"{_cell(row['runtime_s'], '.1f'):>9}  {row['status']}"
        )
    print()
    print("ansatz  hea-L<n>/<sim> = hardware-efficient VQE; adapt/<pool> = ADAPT-VQE.")
    print("par     ansatz parameters (ADAPT: operators selected). iter = optimizer")
    print("        iterations for hea rows, operators added for adapt rows.")
    print("err     physical energy minus CASCI; VERIFIED needs |err| <= 1.6 mHa.")
    print("contam  symmetry breaking priced at 1 Ha per unit, in mHa; budget 0.1600.")
    print("        adapt rows carry no particle-number leak at all -- the pool cannot")
    print("        leave the six-electron sector -- so the column is pure <S^2>.")
    print("xcheck  independent re-evaluation of the same converged state minus the")
    print("        reported energy, in mHa: MPS-vs-statevector for hea rows, Qiskit-")
    print("        circuit-vs-sparse-algebra for adapt rows. 'sv' hea rows are zero")
    print("        by construction; adapt rows are zero unless synthesis broke.")
    print("2q      two-qubit gates after transpilation (adapt rows only).")
    return 0
