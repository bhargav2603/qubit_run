#!/usr/bin/env python3
"""Build (or validate) one point of the study: a geometry x active-space pair.

`run.py prepare` handles the single default specification. This handles any
point of the TS/FAR x 8q/12q/16q grid, which is what `colab_prepare.py` drives.

Needs PySCF for `--build`; `--validate-only` needs only OpenFermion and SciPy
and therefore runs on Windows too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chemistry import (
    DEFAULT_HAMILTONIAN_CUTOFF,
    build_qubit_hamiltonian,
    require_pyscf,
    save_chemistry,
)
from system import make_spec
from validation import validate_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="ts", help="ts or far")
    parser.add_argument("--active-space", default="8q", help="8q, 12q or 16q")
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument(
        "--orbital-selection", default="mp2_natural",
        choices=("mp2_natural", "canonical_hf_frontier"),
    )
    parser.add_argument(
        "--solvent-epsilon", type=float, default=4.0,
        help="C-PCM dielectric. Pass a negative value for vacuum.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_HAMILTONIAN_CUTOFF)
    parser.add_argument("--allow-delocalized-active-space", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip the PySCF build and validate an existing cache instead.",
    )
    parser.add_argument(
        "--number-penalty", type=float, default=0.0,
        help="Must match the VQE. UCCSD needs none, so this defaults to 0.",
    )
    parser.add_argument("--spin-penalty", type=float, default=0.0)
    args = parser.parse_args()

    spec = make_spec(
        geometry=args.geometry,
        active_space=args.active_space,
        basis=args.basis,
        orbital_selection=args.orbital_selection,
        solvent_epsilon=None if args.solvent_epsilon < 0 else args.solvent_epsilon,
    )

    if args.validate_only:
        # validate_main parses its own arguments; hand it the equivalent line.
        sys.argv = [
            "prepare_point.py validate",
            "--cache", str(args.output),
            "--number-penalty", str(args.number_penalty),
            "--spin-penalty", str(args.spin_penalty),
        ]
        return validate_main(spec, str(args.output))

    require_pyscf()
    print(f"Building: {spec.name}", flush=True)
    result = build_qubit_hamiltonian(
        spec, args.cutoff, args.allow_delocalized_active_space
    )
    save_chemistry(result, args.output)
    print(json.dumps(result.metadata, indent=2))
    print(f"\nQubits / Pauli terms : {result.n_qubits} / {len(result.hamiltonian.terms)}")
    print(f"SCF energy           : {result.hartree_fock_energy:.10f} Ha")
    print(f"Reference determinant: {result.reference_determinant_energy:.10f} Ha")
    print(f"CASCI energy         : {result.reference_energy:.10f} Ha")
    print(f"Cache written        : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
