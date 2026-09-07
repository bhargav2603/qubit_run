#!/usr/bin/env python3
"""Single command-line entry point for the local Qiskit Aer LiH workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from chemistry import prepare_main
from qiskit_runtime import backend_main, vqe_main
from selftest import selftest_main
from system import DEFAULT_SPACE, SPACES
from validation import validate_main


ROOT = Path(__file__).resolve().parent


def _usage() -> str:
    name = Path(__file__).name
    return f"""Usage: python {name} COMMAND [--space {{{"|".join(SPACES)}}}] [options]

Commands:
  selftest   Verify the whole local stack with no PySCF and no cache  <- start here
  backend    Inspect Aer, cross-check MPS against statevector, check gradients
  prepare    Build and cache the LiH Hamiltonian with PySCF
             (PySCF has no Windows wheel: run this on Linux/macOS/WSL/Colab)
  validate   Compare the cached Hamiltonian with exact references
  vqe        Run the Qiskit Aer VQE on the validated cache

Active spaces (--space, default {DEFAULT_SPACE}):
  frozen-core  CASCI(2e,5o), Li 1s frozen, 10 qubits, reference is CASCI
  full         FCI(4e,6o), whole STO-3G space, 12 qubits, reference is FCI

Each space keeps its own Hamiltonian cache and result file, so switching does
not silently reuse the other one's numbers.

Use `python {name} COMMAND --help` for command options.
"""


def _extract_space(argv: list[str]) -> tuple[str, list[str]]:
    """Pull `--space X` / `--space=X` out of argv before the subcommand parses it."""
    space = DEFAULT_SPACE
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--space":
            if index + 1 >= len(argv):
                raise SystemExit("--space requires a value")
            space = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--space="):
            space = argument.split("=", 1)[1]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    if space not in SPACES:
        raise SystemExit(
            f"Unknown --space {space!r}; choose from {', '.join(SPACES)}"
        )
    return space, remaining


def main() -> int:
    # Extract --space from the whole line, not just the tail, so it may sit on
    # either side of the command.
    space, argv = _extract_space(sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0

    command = argv[0].lower()
    rest = argv[1:]
    spec = SPACES[space]
    cache = str(ROOT / f"lih_{space.replace('-', '_')}_hamiltonian.json")
    result = str(ROOT / f"lih_{space.replace('-', '_')}_vqe_result.json")
    n_qubits = 2 * spec.n_active_orbitals

    sys.argv = [f"{Path(__file__).name} {command} --space {space}", *rest]
    if command == "prepare":
        return prepare_main(spec, cache)
    if command == "validate":
        return validate_main(spec, cache)
    if command == "vqe":
        return vqe_main(
            spec,
            cache,
            result,
            f"Qiskit Aer {spec.name} VQE",
            default_layers=4,
        )
    if command == "backend":
        return backend_main(n_qubits, spec.name)
    if command == "selftest":
        return selftest_main(spec)

    print(f"Unknown command: {command}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
