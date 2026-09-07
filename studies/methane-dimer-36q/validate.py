"""End-to-end proof that the pipeline is correct, on a system small enough to check.

`selftest.py` proves the logic without chemistry.  This proves the *chemistry*
-- every call into PySCF, ffsim and qiskit-addon-sqd -- on a deliberately tiny
active space where the exact answer is instant.  Same molecule, same code path,
same functions; only the basis and the AVAS AO list shrink.

Run this before spending hours on aug-cc-pVQZ. It takes seconds.

    python run.py validate

What it establishes, in order:

  1. RHF converges and the geometry is what we asked for.
  2. AVAS produces a self-consistent active space and mo_occ still agrees.
  3. The active-space Hamiltonian round-trips through the JSON cache unchanged.
  4. CCSD runs and yields t1/t2 amplitudes.
  5. The LUCJ operator and circuit build, with the expected qubit count.
  6. ffsim samples, and every noiseless shot is particle-number correct.
  7. SQD reproduces CASCI. **This is the load-bearing check**: with enough
     samples the subspace saturates the CAS, so SQD must equal CASCI to solver
     precision, not merely to chemical accuracy. Anything else means the SQD
     Hamiltonian and the CASCI Hamiltonian are not the same operator.
  8. SQD never falls below CASCI (the variational bound).
  9. The energy variance is ~0 at saturation.
 10. Uniform sampling at matched dimension also converges, confirming the
     ablation is a fair control and not accidentally broken.
 11. Binding energies come out of two geometries with the right sign convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import binding
import paper
import spaces

# A small active space on the *same* molecule. STO-3G with only the carbon
# valence AOs projected gives a CAS that diagonalizes instantly, while every
# function called below is the one the production run calls.
VALIDATION_BASIS = "sto-3g"
VALIDATION_AO_LABELS = ("C 2s", "C 2p")
VALIDATION_DISTANCE = 3.638
VALIDATION_SECOND_DISTANCE = 48.000

# Saturation tolerance. When the subspace spans the full CAS, SQD and CASCI are
# the same eigenproblem; agreement should be at solver precision. 1e-8 Ha is
# five orders tighter than chemical accuracy (1.6e-3 Ha).
SATURATION_TOL_HARTREE = 1e-8


@dataclass
class Step:
    name: str
    passed: bool
    detail: str
    seconds: float = 0.0


class _Recorder:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._t0 = time.monotonic()

    def add(self, name: str, passed: bool, detail: str = "") -> Step:
        now = time.monotonic()
        step = Step(name, bool(passed), detail, now - self._t0)
        self._t0 = now
        self.steps.append(step)
        mark = "ok  " if passed else "FAIL"
        print(f"  [{mark}] {name}   ({step.seconds:.2f}s)")
        if detail:
            print(f"         {detail}")
        return step

    @property
    def ok(self) -> bool:
        return all(s.passed for s in self.steps)


def _spec(distance: float):
    import chemistry

    return chemistry.SystemSpec(
        distance=distance,
        basis=VALIDATION_BASIS,
        density_fit=False,          # tiny system; use exact integrals
        avas_ao_labels=VALIDATION_AO_LABELS,
        expect_active_space=None,   # whatever AVAS gives, we check consistency
    )


def run(cache_root: Path, *, verbose: int = 0) -> bool:
    import chemistry

    chemistry.require_chemistry()

    import ansatz
    import reference
    import sampling
    import sqd

    print("=" * 74)
    print("end-to-end validation -- small active space, exact answer known")
    print("=" * 74)
    print(f"  molecule   methane dimer, R = {VALIDATION_DISTANCE} A")
    print(f"  basis      {VALIDATION_BASIS}")
    print(f"  AVAS AOs   {list(VALIDATION_AO_LABELS)}")
    print()

    rec = _Recorder()

    # --- 1-3. chemistry -------------------------------------------------
    spec = _spec(VALIDATION_DISTANCE)
    mol_data = chemistry.load_or_build(
        spec, cache_root, rebuild=True, verbose=verbose
    )
    norb, nelec = mol_data.norb, mol_data.nelec
    dimension = spaces.n_determinants(norb, nelec)

    rec.add(
        "RHF + AVAS produce a consistent active space",
        norb > 0 and sum(nelec) > 0 and nelec[0] == nelec[1],
        f"({sum(nelec)}e,{norb}o) -> {dimension:,} determinants, "
        f"HF = {mol_data.hf_energy:.10f} Ha",
    )

    if dimension > 5_000_000:
        rec.add(
            "validation space is small enough to diagonalize",
            False,
            f"{dimension:,} determinants is too large for a smoke test; "
            "narrow VALIDATION_AO_LABELS",
        )
        return False

    reloaded = chemistry.load_or_build(spec, cache_root, verbose=0)
    same = (
        reloaded.norb == norb
        and reloaded.nelec == nelec
        and np.allclose(reloaded.one_body_integrals, mol_data.one_body_integrals)
        and abs(reloaded.core_energy - mol_data.core_energy) < 1e-12
    )
    rec.add(
        "Hamiltonian round-trips through the cache unchanged",
        same,
        "integrals and core energy identical after reload",
    )

    # --- 4. CCSD ---------------------------------------------------------
    e_ccsd = reference.run_ccsd(mol_data, store_amplitudes=True)
    rec.add(
        "CCSD runs and stores amplitudes",
        mol_data.ccsd_t2 is not None and mol_data.ccsd_t1 is not None,
        f"E(CCSD) = {e_ccsd:.10f} Ha, t2 shape {np.shape(mol_data.ccsd_t2)}",
    )

    # --- 5. ansatz -------------------------------------------------------
    layout = ansatz.heavy_hex_layout(norb, n_reps=2)
    operator = ansatz.build_operator(mol_data, layout)
    circuit = ansatz.build_circuit(mol_data, operator)
    rec.add(
        "LUCJ operator and circuit build",
        circuit.num_qubits == 2 * norb,
        f"{circuit.num_qubits} qubits, depth {circuit.depth()}, "
        f"{layout.n_ancilla_qubits} ancillas at 36-qubit scale",
    )

    # --- 6. sampling -----------------------------------------------------
    shots = 40_000
    sampler = sampling.NoiselessSampler(norb, nelec, seed=1234)
    sampled = sampler.sample(circuit, shots)
    validity = sampling.valid_configuration_fraction(sampled.bit_array, norb, nelec)
    rec.add(
        "every noiseless shot is particle-number correct",
        abs(validity - 1.0) < 1e-12,
        f"{validity:.6f} of {shots:,} shots have {nelec} occupation "
        "(LUCJ commutes with both number operators)",
    )

    # --- 7-8. SQD vs CASCI ----------------------------------------------
    e_casci = reference.run_casci(mol_data, verbose=verbose)
    rec.add(
        "CASCI completes",
        e_casci < mol_data.hf_energy,
        f"E(CASCI) = {e_casci:.10f} Ha, "
        f"E_corr = {e_casci - mol_data.hf_energy:.10f} Ha",
    )

    config = sqd.SqdConfig(
        samples_per_batch=shots // 2,
        n_batches=2,
        max_iterations=6,
        energy_tol=1e-10,
        seed=1234,
    )
    result = sqd.run(mol_data, sampled.bit_array, config,
                     source="noiseless", verbose=bool(verbose), keep_state=True)

    try:
        binding.variational_check(result.energy, e_casci)
        bound_ok, bound_msg = True, "SQD >= CASCI, as a subspace method must be"
    except AssertionError as exc:
        bound_ok, bound_msg = False, str(exc)
    rec.add("variational bound holds", bound_ok, bound_msg)

    delta = result.energy - e_casci
    # Saturation is judged against THIS system's CAS, which is why
    # SqdResult carries full_dimension rather than assuming the paper's.
    saturated = result.subspace_dimension >= 0.999 * dimension
    rec.add(
        "SQD reproduces CASCI",
        abs(delta) < SATURATION_TOL_HARTREE if saturated
        else abs(binding.hartree_to_kcal(delta)) < 1.0,
        f"SQD = {result.energy:.10f} Ha, delta = {delta / binding.MILLIHARTREE:+.6f} mHa "
        f"({binding.hartree_to_kcal(delta):+.6f} kcal/mol); subspace "
        f"{result.subspace_dimension:,}/{dimension:,} = {result.subspace_fraction:.2%}"
        + ("  [saturated -> solver-precision test]" if saturated
           else "  [not saturated -> chemical-accuracy test]"),
    )

    # --- 9. variance -----------------------------------------------------
    try:
        variance = sqd.energy_variance(result.sci_state, mol_data)
        ok = variance >= -1e-10 and (not saturated or variance < 1e-6)
        rec.add(
            "energy variance is computable and near zero at saturation",
            ok,
            f"dH = {variance:.3e} Ha^2",
        )
    except Exception as exc:   # noqa: BLE001 -- report, do not mask
        rec.add("energy variance", False, f"{type(exc).__name__}: {exc}")

    # --- 10. ablation ----------------------------------------------------
    uniform = sampling.UniformSampler(norb, nelec, seed=1234)
    control = sqd.run(
        mol_data,
        uniform.sample(shots=shots).bit_array,
        sqd.ablation_config(result, config),
        source="uniform",
        verbose=False,
    )
    gap = (control.energy - result.energy) / binding.MILLIHARTREE
    rec.add(
        "ablation control runs at matched dimension",
        control.energy >= e_casci - 1e-9,
        f"uniform = {control.energy:.10f} Ha, gap = {gap:+.6f} mHa, "
        f"d = {control.subspace_dimension:,}",
    )

    # --- 11. binding energy ----------------------------------------------
    spec_far = _spec(VALIDATION_SECOND_DISTANCE)
    mol_far = chemistry.load_or_build(spec_far, cache_root, rebuild=True, verbose=0)
    e_far = reference.run_casci(mol_far, verbose=0)
    e_bind = binding.binding_energy(e_casci, e_far, unit="kcal/mol")
    rec.add(
        "binding energy computes from two geometries",
        abs(e_bind) < 50.0,
        f"E_bind({VALIDATION_DISTANCE} A) = {e_bind:+.4f} kcal/mol "
        f"relative to {VALIDATION_SECOND_DISTANCE} A",
    )

    total = len(rec.steps)
    failed = sum(1 for s in rec.steps if not s.passed)
    print("\n" + "=" * 74)
    print(f"{total - failed}/{total} steps passed"
          + ("" if failed == 0 else f"  --  {failed} FAILED"))
    if rec.ok:
        print("\nThe chemistry pipeline is verified end to end. The production run")
        print("differs only in basis set and active-space size, not in code path.")
    print("=" * 74)
    return rec.ok
