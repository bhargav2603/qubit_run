#!/usr/bin/env python3
"""Classical correlated baselines in the *same* active space, for comparison.

The workflow already has two classical numbers -- Hartree-Fock, which is the
floor, and CASCI, which is exact and is the target. Those bracket the answer but
they do not answer the question a chemist asks first:

    is the quantum method better than a cheap classical correlated method?

CASCI being exact makes it the reference, not a competitor. MP2, CCSD and
CCSD(T) are the competitors, and without them "ADAPT reached 0.18 mHa" has no
scale -- 0.18 mHa is excellent if CCSD(T) is at 2 mHa and irrelevant if CCSD(T)
is at 0.02 mHa. This module computes them so the comparison is on the slide
rather than in the room.

**The active space must match, or the comparison is meaningless.** Running CCSD
on all 152 electrons of imipramine would correlate 73 core orbitals that CASCI
never touched and produce a number tens of hartree away for reasons that have
nothing to do with the ansatz. Everything here is therefore frozen down to the
identical CAS(6e,6o) window `hamiltonian.py` builds: orbitals 0..ncore-1 and
ncore+ncas..nmo-1 are frozen, leaving exactly the six active electrons in the
six active orbitals.

With six electrons in six orbitals, singles and doubles are not the whole story
-- FCI in that space reaches hextuple excitations -- so CCSD is a genuine
approximation here and CCSD(T) is a genuine test, not a formality. Which way it
lands is not predictable from the size alone: a pi-conjugated active space
carries static correlation that coupled cluster handles poorly. That is the
point of measuring it.

This needs PySCF, so it runs where `prepare` runs -- Linux, macOS, WSL or Colab.
It repeats the 45-atom RHF (the expensive part, minutes); the correlated solves
in six orbitals are instant.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from hamiltonian import MoleculeSpec, converged_mean_field


def fci_dimension(n_active_orbitals: int, n_active_electrons: int, spin: int = 0) -> int:
    """Determinant count for the active space -- the axis on which quantum matters.

    This is the honest "why quantum eventually" number. It is what the classical
    exact solve must diagonalize, and it is what the quantum register replaces
    with 2 * n_active_orbitals qubits. At CAS(6e,6o) it is 400, which a laptop
    diagonalizes in milliseconds; the crossover is several active spaces away and
    a chart that shows where is more honest than one that implies it is here.
    """
    n_beta = (n_active_electrons - spin) // 2
    n_alpha = n_active_electrons - n_beta
    return math.comb(n_active_orbitals, n_alpha) * math.comb(n_active_orbitals, n_beta)


def classical_baselines(spec: MoleculeSpec) -> dict[str, Any]:
    """MP2, CCSD and CCSD(T) frozen to the identical active space, plus timings.

    The timings matter as much as the energies. A cost chart that reports only
    the quantum runtime invites the assumption that the classical side was
    expensive too, and here it emphatically is not -- these solves are a few
    milliseconds each once the mean field exists. Reporting that is what makes
    the accuracy comparison credible rather than promotional.

    The shared RHF is timed separately and excluded from the per-method numbers,
    because every method here needs it, including the quantum one: `prepare`
    cannot build the Hamiltonian without it. Charging it to the classical
    correlated methods alone would understate the quantum cost, not overstate it.
    """
    from pyscf import cc, mp

    started = time.perf_counter()
    molecule, mean_field, hartree_fock_energy = converged_mean_field(spec)
    mean_field_seconds = time.perf_counter() - started

    n_mos = int(mean_field.mo_coeff.shape[1])
    n_core = (int(molecule.nelectron) - spec.n_active_electrons) // 2
    active = range(n_core, n_core + spec.n_active_orbitals)
    # PySCF freezes by MO index. Everything below the window and everything above
    # it: what remains correlated is exactly the CASCI active space.
    frozen = list(range(n_core)) + list(range(n_core + spec.n_active_orbitals, n_mos))

    methods: dict[str, Any] = {}

    started = time.perf_counter()
    mp2 = mp.MP2(mean_field, frozen=frozen).run()
    methods["MP2"] = {
        "energy_hartree": float(mp2.e_tot),
        "seconds": time.perf_counter() - started,
        "note": "second-order perturbation theory, same active space",
    }

    started = time.perf_counter()
    coupled_cluster = cc.CCSD(mean_field, frozen=frozen).run()
    ccsd_seconds = time.perf_counter() - started
    methods["CCSD"] = {
        "energy_hartree": float(coupled_cluster.e_tot),
        "seconds": ccsd_seconds,
        "converged": bool(coupled_cluster.converged),
        "note": "singles and doubles; not exact here -- CAS(6e,6o) reaches hextuples",
    }

    started = time.perf_counter()
    triples = float(coupled_cluster.ccsd_t())
    methods["CCSD(T)"] = {
        "energy_hartree": float(coupled_cluster.e_tot) + triples,
        "seconds": ccsd_seconds + (time.perf_counter() - started),
        "triples_correction_hartree": triples,
        "note": "perturbative triples on top of CCSD",
    }

    return {
        "molecule": spec.name,
        "basis": spec.basis,
        "active_space": f"CAS({spec.n_active_electrons}e,{spec.n_active_orbitals}o)",
        "active_spatial_orbitals": spec.n_active_orbitals,
        "active_electrons": spec.n_active_electrons,
        "qubits": 2 * spec.n_active_orbitals,
        "frozen_core_orbitals": n_core,
        "active_mo_indices_zero_based": list(active),
        "total_spatial_orbitals": n_mos,
        "fci_dimension": fci_dimension(spec.n_active_orbitals, spec.n_active_electrons),
        "hartree_fock_energy_hartree": float(hartree_fock_energy),
        "mean_field_seconds": mean_field_seconds,
        "methods": methods,
    }


def _reference_energy(cache: str) -> float | None:
    """The CASCI energy from the cache, so errors can be reported here too.

    Optional on purpose: the baselines are worth computing and storing even on a
    machine where the cache has not been built yet, and `visualize` can supply
    the reference from any result JSON instead.
    """
    try:
        payload = json.loads(Path(cache).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("reference_energy_hartree", "casci_energy_hartree"):
        if key in payload:
            return float(payload[key])
    return None


def classical_main(spec: MoleculeSpec, cache: str, results: str) -> int:
    """`run.py classical` -- compute the baselines and write results/classical.json."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classical correlated baselines in the CASCI active space"
    )
    parser.add_argument("--cache", type=Path, default=Path(cache))
    parser.add_argument("--result", type=Path, default=None,
                        help="Output JSON. Defaults to <results>/classical.json.")
    args = parser.parse_args()

    print(f"Classical baselines: {spec.name} "
          f"CAS({spec.n_active_electrons}e,{spec.n_active_orbitals}o)/{spec.basis}", flush=True)
    print("Repeating the 45-atom RHF -- this is the slow part.", flush=True)

    payload = classical_baselines(spec)
    reference = _reference_energy(str(args.cache))
    payload["reference_energy_hartree"] = reference
    payload["reference_method"] = "CASCI" if reference is not None else None

    print(f"\nFCI dimension              : {payload['fci_dimension']:,} determinants")
    print(f"Hartree-Fock               : {payload['hartree_fock_energy_hartree']:.12f} Ha "
          f"({payload['mean_field_seconds']:.1f} s)")
    if reference is not None:
        print(f"CASCI (exact, reference)   : {reference:.12f} Ha")
    print()
    print(f"{'method':<10} {'energy (Ha)':>18} {'error (mHa)':>13} {'seconds':>9}")
    for name, entry in payload["methods"].items():
        energy = entry["energy_hartree"]
        error = "" if reference is None else f"{1000.0 * (energy - reference):+13.4f}"
        print(f"{name:<10} {energy:>18.12f} {error:>13} {entry['seconds']:>9.3f}")

    if reference is not None:
        best = min(payload["methods"].items(),
                   key=lambda item: abs(item[1]["energy_hartree"] - reference))
        print(f"\nBest classical correlated  : {best[0]} at "
              f"{1000.0 * abs(best[1]['energy_hartree'] - reference):.4f} mHa")
        print("Compare this against the quantum `Error vs CASCI`. If the classical "
              "method wins,\nthe honest framing is verification, not advantage.")

    destination = args.result or (Path(results) / "classical.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {destination}")
    return 0
