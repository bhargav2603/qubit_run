#!/usr/bin/env python3
"""Single command-line entry point for the local Qiskit Aer imipramine workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from molecule import IMIPRAMINE


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
# The folder name already says which molecule and active space this is, so the
# artifacts inside it are named for what they are. Identical to the QARP
# folder's names, so a cache built there is usable here unchanged.
CACHE = str(ROOT / "hamiltonian_cas6e6o.json")
LABEL = "Imipramine CASCI(6e,6o)"
QUBITS = 2 * IMIPRAMINE.n_active_orbitals


def _usage() -> str:
    name = Path(__file__).name
    return f"""Usage: python {name} COMMAND [options]

Commands:
  selftest   Verify the whole local stack with no PySCF and no cache  <- start here
  backend    Inspect Aer, cross-check MPS against statevector, check gradients
  prepare    Build and cache the CASCI(6e,6o) Hamiltonian with PySCF
             (PySCF has no Windows wheel: run this on Linux/macOS/WSL/Colab)
             (--diagnose reports the frontier orbital window and builds nothing)
  validate   Compare the cached 12-qubit Hamiltonian with exact references
  vqe        Run the Qiskit Aer hardware-efficient VQE on the validated cache
  adapt      Run ADAPT-VQE -- the published reference method. Grows a
             particle-number-conserving ansatz one operator at a time until the
             pool has nothing left worth adding, then emits and verifies the
             Qiskit circuit. Needs no penalty and no layer scan.
  classical  MP2, CCSD and CCSD(T) frozen to the same active space, so the
             quantum error has a scale. Needs PySCF -- run it where `prepare`
             runs. Writes results/classical.json.
  summary    Tabulate every result in results/
  plot       Chart every result in results/ and save the PNGs
             (in Colab, `import visualize; visualize.show()` renders inline)
  deck       The five presentation figures -- error, gates, accuracy, cost and
             the pipeline diagram -- into results/deck/
             (in Colab, `visualize.show_deck()` renders them inline)

Use `python {name} COMMAND --help` for command options.
"""


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    command = sys.argv[1].lower()
    sys.argv = [f"{Path(__file__).name} {command}", *sys.argv[2:]]

    # Imported per command so that `prepare` never touches Qiskit and every
    # other command never touches PySCF -- which is what lets the two halves of
    # this workflow live on different machines.
    if command == "selftest":
        from selftest import selftest_main

        return selftest_main(IMIPRAMINE)
    if command == "summary":
        from summary import summary_main

        return summary_main(str(RESULTS))
    if command == "plot":
        from visualize import plot_main

        return plot_main(str(RESULTS))
    if command == "deck":
        from visualize import deck_main

        return deck_main(str(RESULTS))
    if command == "classical":
        from classical import classical_main

        return classical_main(IMIPRAMINE, CACHE, str(RESULTS))
    if command == "prepare":
        from hamiltonian import prepare_main

        return prepare_main(IMIPRAMINE, CACHE)
    if command == "validate":
        from validation import validate_main

        return validate_main(IMIPRAMINE, CACHE)
    if command == "vqe":
        from qiskit_runtime import vqe_main

        return vqe_main(
            IMIPRAMINE,
            CACHE,
            str(RESULTS),
            f"Qiskit Aer {LABEL} VQE",
            default_layers=4,
        )
    if command == "adapt":
        from adapt_runtime import adapt_main

        return adapt_main(
            IMIPRAMINE,
            CACHE,
            str(RESULTS),
            f"ADAPT-VQE {LABEL}",
        )
    if command == "backend":
        from qiskit_runtime import backend_main

        return backend_main(QUBITS, LABEL)

    print(f"Unknown command: {command}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
