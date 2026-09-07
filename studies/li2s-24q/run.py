#!/usr/bin/env python3
"""Single command-line entry point for the Li2S 24-qubit HI-VQE workflow."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE = str(ROOT / "cache")
RESULTS = str(ROOT / "results")


def _usage() -> str:
    name = Path(__file__).name
    return f"""Li2S CAS(12e,12o), 24 qubits -- HI-VQE (arXiv:2503.06292) on Qiskit

Usage: python {name} COMMAND [options]

Commands:
  selftest   Prove the whole stack against exact answers, with no PySCF and no
             cache. Start here -- it needs nothing but numpy, scipy and qiskit.
  backend    Report the Qiskit/Aer environment and cross-check the sector
             simulator against Aer amplitude by amplitude.
  geometry   Symmetric-stretch scan for the equilibrium bond length (PySCF).
  prepare    Build and cache the CAS(12e,12o) Hamiltonian at one bond length
             (PySCF; --diagnose reports the orbital window and builds nothing).
  validate   Prove a cached Hamiltonian correct and write its receipt. No PySCF.
  paulis     Count the Pauli words a conventional VQE would have to measure.
  classical  MP2, CCSD and CCSD(T) frozen to the same active space (PySCF).
  hivqe      Run HI-VQE at one bond length.
  adapt      Run ADAPT-VQE at one bond length -- the conventional-VQE
             comparison, given its best case: an ansatz it builds itself,
             exact expectation values, infinite shots and no measurement cost.
  scan       Run HI-VQE along the whole Li-S dissociation coordinate.
  summary    Tabulate every result in results/.
  plot       Chart every result in results/ and save the PNGs.
  report     Write the full structured report (Markdown + HTML) into results/.

Use `python {name} COMMAND --help` for that command's options.

Typical first run, all on one Colab CPU:

  python {name} selftest
  python {name} prepare --distance 2.10
  python {name} validate --distance 2.10
  python {name} hivqe   --distance 2.10
  python {name} report
"""


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    command = sys.argv[1].lower()
    sys.argv = [f"{Path(__file__).name} {command}", *sys.argv[2:]]

    # Imported per command so that `prepare` is the only path that touches
    # PySCF and `validate` is the only path that must not need Qiskit. That is
    # what lets the two halves of this workflow live on different machines.
    from molecule import DISSOCIATION_DISTANCES, LI2S

    if command == "selftest":
        from selftest import selftest_main

        return selftest_main()
    if command == "backend":
        from selftest import backend_main

        return backend_main()
    if command == "geometry":
        from driver import geometry_main

        return geometry_main(LI2S, CACHE)
    if command == "prepare":
        from hamiltonian import prepare_main

        return prepare_main(LI2S, CACHE)
    if command == "validate":
        from validation import validate_main

        return validate_main(LI2S, CACHE)
    if command == "paulis":
        from driver import paulis_main

        return paulis_main(LI2S, CACHE)
    if command == "classical":
        from classical import classical_main

        return classical_main(LI2S, CACHE, RESULTS)
    if command == "hivqe":
        from driver import hivqe_main

        return hivqe_main(LI2S, CACHE, RESULTS)
    if command == "adapt":
        from driver import adapt_main

        return adapt_main(LI2S, CACHE, RESULTS)
    if command == "scan":
        from driver import scan_main

        return scan_main(LI2S, CACHE, RESULTS, DISSOCIATION_DISTANCES)
    if command == "summary":
        from summary import summary_main

        return summary_main(RESULTS)
    if command == "plot":
        from visualize import plot_main

        return plot_main(RESULTS)
    if command == "report":
        from report import report_main

        return report_main(LI2S, CACHE, RESULTS)

    print(f"Unknown command: {command}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
