#!/usr/bin/env python3
"""Single command-line entry point for the local Qiskit Aer workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from chemistry import prepare_main
from classical import ladder_main
from qiskit_runtime import backend_main, vqe_main
from selftest import selftest_main
from system import ACRYLAMIDE_THIOLATE, make_spec
from validation import validate_main


ROOT = Path(__file__).resolve().parent
CACHE = str(ROOT / "acrylamide_thiolate_hamiltonian.json")
RESULT = str(ROOT / "acrylamide_thiolate_vqe_result.json")
LADDER = str(ROOT / "acrylamide_thiolate_classical_ladder.json")
LABEL = "Acrylamide-methanethiolate CAS(4e,4o)"


def _usage() -> str:
    return f"""Usage: python {Path(__file__).name} COMMAND [options]

Local, needs only Qiskit + Aer + OpenFermion:
  selftest   Verify the whole local stack with no PySCF and no cache  <- start here
  backend    Inspect Aer, cross-check MPS vs statevector, time a VQE iteration
  validate   Compare a cached Hamiltonian against exact references
  vqe        Run the VQE (UCCSD by default) on the validated cache

Needs PySCF -- run on Linux/macOS/WSL/Colab, not Windows:
  prepare    Build and cache the active-space Hamiltonian
  ladder     Classical benchmark ladder (HF/MP2/CCSD/CCSD(T)/CASCI) + TS-FAR

Use `python {Path(__file__).name} COMMAND --help` for command options.
"""


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    command = sys.argv[1].lower()
    sys.argv = [f"{Path(__file__).name} {command}", *sys.argv[2:]]
    if command == "prepare":
        return prepare_main(ACRYLAMIDE_THIOLATE, CACHE)
    if command == "validate":
        return validate_main(ACRYLAMIDE_THIOLATE, CACHE)
    if command == "vqe":
        return vqe_main(
            ACRYLAMIDE_THIOLATE, CACHE, RESULT,
            "Qiskit Aer Acrylamide-Methanethiolate VQE (UCCSD / CovAngelo protocol)",
        )
    if command == "ladder":
        return ladder_main(make_spec, LADDER)
    if command == "backend":
        return backend_main(8, LABEL)
    if command == "selftest":
        return selftest_main()

    print(f"Unknown command: {command}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
