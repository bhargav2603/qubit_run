#!/usr/bin/env python3
"""Build every Hamiltonian and the classical ladder where PySCF works.

PySCF publishes no Windows wheel, so this is the one step that has to run
somewhere else -- Google Colab, WSL, Linux or macOS. It is also the *only* step
that has to: everything downstream reads the JSON cache and runs natively on
Windows.

Google Colab, in a fresh notebook:

    !pip install -q pyscf openfermion "qiskit==2.5.*" "qiskit-aer==0.17.*" \
                    "qiskit-nature==0.8.*"
    # upload this folder, or clone it, then:
    %cd acrylamide_thiolate_qiskit
    !python colab_prepare.py --active-spaces 8q --geometries ts,far
    # then download the .zip it prints

Colab notes, stated plainly: sessions die on ~90 minutes idle and are wiped when
they end, so download the archive before you close the tab. Free-tier Colab has
about two vCPUs, so CCSD(T) on aug-cc-pVDZ for this 15-atom anion is the slow
part -- use `--skip-ladder` first if you only want the Hamiltonians.

Every artifact carries a specification hash and a workflow hash, so a cache that
does not match the code you run it against is a hard error rather than a wrong
answer. Copy the cache *and* its `.validated.json` receipt.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(arguments: list[str]) -> bool:
    print("\n$ " + " ".join(arguments), flush=True)
    started = time.perf_counter()
    outcome = subprocess.run([sys.executable, *arguments], cwd=ROOT)
    print(f"  -> exit {outcome.returncode} in {time.perf_counter() - started:.1f} s", flush=True)
    return outcome.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", default="ts,far")
    parser.add_argument(
        "--active-spaces", default="8q",
        help="Comma-separated: 8q, 12q, 16q. Start with 8q; 16q is much slower.",
    )
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument("--orbital-selection", default="mp2_natural")
    parser.add_argument("--solvent-epsilon", type=float, default=4.0)
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument(
        "--ladder-methods", default="HF,MP2,CCSD,CCSD(T),CASCI",
        help="Drop CCSD(T) if the session is short.",
    )
    parser.add_argument("--archive", default="acrylamide_artifacts")
    args = parser.parse_args()

    try:
        import pyscf  # noqa: F401
    except ImportError:
        print(
            "PySCF is not installed. This script must run where PySCF works:\n"
            "  pip install pyscf openfermion 'qiskit==2.5.*' 'qiskit-aer==0.17.*' "
            "'qiskit-nature==0.8.*'",
            file=sys.stderr,
        )
        return 1

    geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]
    active_spaces = [a.strip() for a in args.active_spaces.split(",") if a.strip()]
    common = [
        "--basis", args.basis,
        "--orbital-selection", args.orbital_selection,
        "--solvent-epsilon", str(args.solvent_epsilon),
    ]

    failures: list[str] = []
    produced: list[Path] = []
    for active_space in active_spaces:
        for geometry in geometries:
            tag = f"{geometry}_{active_space}"
            cache = ROOT / f"hamiltonian_{tag}.json"
            if not _run(["prepare_point.py", "--geometry", geometry,
                         "--active-space", active_space, "--output", str(cache), *common]):
                failures.append(f"prepare {tag}")
                continue
            if not _run(["prepare_point.py", "--geometry", geometry,
                         "--active-space", active_space, "--output", str(cache),
                         "--validate-only", *common]):
                failures.append(f"validate {tag}")
                continue
            produced.extend([cache, cache.with_name(f"{cache.name}.validated.json")])

    if not args.skip_ladder:
        for active_space in active_spaces:
            ladder = ROOT / f"classical_ladder_{active_space}.json"
            if _run(["run.py", "ladder", "--active-space", active_space,
                     "--methods", args.ladder_methods,
                     "--geometries", args.geometries,
                     "--basis", args.basis,
                     "--orbital-selection", args.orbital_selection,
                     "--solvent-epsilon", str(args.solvent_epsilon),
                     "--result", str(ladder)]):
                produced.append(ladder)
            else:
                failures.append(f"ladder {active_space}")

    existing = [path for path in produced if path.is_file()]
    if existing:
        staging = ROOT / "_artifacts"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for path in existing:
            shutil.copy2(path, staging / path.name)
        archive = shutil.make_archive(str(ROOT / args.archive), "zip", staging)
        shutil.rmtree(staging, ignore_errors=True)
        print("\n" + "=" * 74)
        print(f"Archive written: {archive}")
        for path in existing:
            print(f"  {path.name}")
        print("=" * 74)
        print(
            "Download this archive, unzip it into acrylamide_thiolate_qiskit/ on\n"
            "your Windows machine, then run:\n"
            "    python run.py validate --cache hamiltonian_ts_8q.json\n"
            "    python run.py vqe --cache hamiltonian_ts_8q.json"
        )

    if failures:
        print("\nFAILED steps: " + ", ".join(failures), file=sys.stderr)
        print(
            "\nIf `prepare` failed on the active-space guard, the anion's canonical\n"
            "frontier orbitals are describing the diffuse tail rather than the C-S\n"
            "region. MP2 natural orbitals usually fix this; if not, restrict diffuse\n"
            "functions to the reacting atoms (see the README).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
