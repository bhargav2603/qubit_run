# LiH on Qiskit Aer

A local, Windows-native VQE for LiH/STO-3G on Qiskit 2.5 + qiskit-aer, built to
be comparable with `../lih` (the QARP/Qulacs workflow that ran on a Fujitsu
FX700) and with `../acrylamide_thiolate_qiskit` (the same Aer stack pointed at a
different molecule).

The whole pipeline — integrals, validation, VQE — runs on this machine with no
PySCF and no cloud step.

## Environment

| Component | Version | Why |
| --- | --- | --- |
| Python | 3.12 | The scientific wheels this stack needs are still catching up on 3.13. |
| Qiskit | 2.5.x | Current release line; the local-simulation speedups are the ones this workflow leans on. |
| qiskit-aer | 0.17.x | The simulator. `matrix_product_state` and `statevector`. |
| qiskit-nature | 0.8.x | UCCSD only. No PySCF driver is used. |
| OpenFermion | 1.8.x | Fermion-to-qubit mapping and the exact classical references. |
| PySCF | 2.5+ | **Optional.** Preferred for `prepare` where it installs; not required. |

The repository's `..\.venv-qiskit` already satisfies this (Python 3.12.13,
Qiskit 2.5.2, Aer 0.17.2, qiskit-nature 0.8.0, OpenFermion 1.8.1). To build a
fresh one:

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Results

LiH/STO-3G at r = 1.595 Å, computed here, noiseless `statevector`:

| Quantity | Frozen core (10q) | Full space (12q) |
| --- | --- | --- |
| E(RHF) | −7.862023874015 | −7.862023874015 |
| E(reference) | −7.882174519890 (CASCI) | −7.882401946215 (FCI) |
| Correlation available | 20.151 mHa | 20.378 mHa |

VQE against those references:

| Ansatz | Space | Parameters | Error vs reference | Correlation | Circuits | Runtime | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UCCSD | frozen core (10q) | 24 | **+0.000000 mHa** | 100% | 641 | 48 s | VERIFIED |
| UCCSD | full (12q) | 92 | **+0.011001 mHa** | 99.95% | 4,259 | 2532 s | VERIFIED |
| `real_amplitudes`, 4 layers | frozen core | 50 | +20.150496 mHa | 0.001% | ~41,000 | 409 s | outside chemical accuracy |
| `real_amplitudes`, 8 layers | frozen core | 90 | +3.407633 mHa | 83% | 95,391 | 1473 s | **SYMMETRY BROKEN** |

Chemical accuracy is 1.6 mHa. UCCSD clears it in both active spaces — by six
orders of magnitude on the 10-qubit problem and by a factor of 145 on the
12-qubit one, where finite-difference gradient noise rather than the ansatz sets
the floor. Particle-number leakage is 2.3e-15 and 1.4e-07 respectively: the
state stays where it belongs without any penalty doing work.

The 8-layer row is the one worth reading carefully. Doubling the depth *does*
pull the energy down — 20.15 mHa to 3.41 mHa — but it buys that by leaving the
physical sector: particle-number leakage 1.4e-04 and <S^2> = 2.1e-03, which
price out at 2.19 mHa against a 0.16 mHa budget. The symmetry it broke is worth
more than the error it has left, so the number is not a LiH energy at all. It
cost 95,391 circuits and 25 minutes to get there.

That is the failure mode the accounting exists to catch. Reported without the
symmetry audit, "3.4 mHa from CASCI" would look like respectable progress
towards chemical accuracy rather than a state that is no longer a two-electron
singlet.

## Two integral backends

`prepare` needs molecular integrals. PySCF is the obvious source and is used
automatically when it is importable, but **it publishes no Windows wheel** and
its source build is unsupported there, which used to block this command
entirely.

`integrals.py` is the fallback: a self-contained McMurchie–Davidson
implementation of STO-3G one- and two-electron integrals plus closed-shell RHF,
in NumPy only. At six basis functions it is a smaller and far more auditable
dependency than a Fortran-backed chemistry package.

```powershell
python run.py prepare                            # auto: PySCF if present, else native
python run.py prepare --integral-backend native  # force the built-in engine
python run.py prepare --integral-backend pyscf   # force PySCF, error if absent
```

The engine used is written into the cache as `integral_backend`, so a result
always records which one produced it.

**Scope, stated plainly:** the native engine does STO-3G, closed shell, s and p
functions. That covers LiH and nothing beyond row 2. It refuses other bases and
unknown elements rather than guessing. For anything larger, install PySCF.

It is pinned by `tests/test_integrals.py` against the H2/STO-3G textbook energy
(Szabo & Ostlund), the published LiH/STO-3G RHF and FCI energies, and — the
strongest check available without a second chemistry code — the requirement that
the Hartree-Fock determinant of the *second-quantized* Hamiltonian equal the SCF
energy to 10 decimal places in both active spaces. That ties the CAS
transformation, the two-electron transformation and the frozen-core energy
together; an error in any one of them breaks it.

## Active spaces

LiH/STO-3G has 6 spatial orbitals holding 4 electrons. Two choices, selected
with `--space`, each with its own cache and result file:

| `--space` | Active space | Qubits | Reference energy |
| --- | --- | --- | --- |
| `frozen-core` *(default)* | CASCI(2e,5o), Li 1s frozen | 10 | CASCI in that space |
| `full` | FCI(4e,6o), whole STO-3G space | 12 | FCI in STO-3G |

Li's 1s pair is chemically inert: freezing it costs 0.23 mHa of the 20.378 mHa
of correlation, which is why the smaller problem is the better-posed one.
**The reference follows the space**: with `frozen-core` the target is
CASCI(2e,5o), not full-space FCI, and an energy *below* it is proof the state
left the two-electron sector — not proof of better chemistry.

## Two ansätze

| `--ansatz` | Circuit | Conserves N, S | `--layers` | Gradient |
| --- | --- | --- | --- | --- |
| `real-amplitudes` *(default)* | RY + linear CX ladder | no — needs a penalty | yes | exact parameter-shift |
| `uccsd` | singles + doubles evolutions | **yes, by construction** | ignored | finite difference |

`real-amplitudes` is the hardware-efficient baseline and the one that matches
`../lih`. `uccsd` is the chemistry-inspired alternative the benchmark
literature uses, and on this molecule it is the one that works.

Because UCCSD conserves particle number exactly, its number penalty contributes
identically zero — measured leakage at the optimum is −1.3e−15. The penalty
machinery is still applied and still audited; it simply has nothing to correct.

UCCSD parameters are shared across several Pauli terms per excitation, so the
parameter-shift precondition fails and the runtime falls back to central finite
differences. That is detected, announced, and exact enough on a noiseless
simulator — `selftest` measures the two against each other at 3e−08.

## Commands

```powershell
python run.py selftest                    # start here; no cache needed
python run.py backend                     # Aer inspection, MPS vs statevector, gradients
python run.py prepare                     # builds the Hamiltonian, runs locally
python run.py validate                    # exact checks; writes the validation receipt
python run.py vqe --ansatz uccsd --method statevector --start-noise 0
```

`--space full` works on any of them, and goes before or after the command's own
flags:

```powershell
python run.py vqe --space full --ansatz uccsd --method statevector --start-noise 0
```

`vqe` refuses to start without a validation receipt that matches the cache, the
molecule specification, the workflow source and the penalty settings. Editing
any workflow file invalidates it — re-run `validate`.

### `--start-noise 0` for UCCSD

The kick exists for the hardware-efficient circuit, which has *no* live
parameters at θ = 0. UCCSD has 15 of 24 live in the frozen-core space and 63 of
92 in the full space (singles are dead by Brillouin's theorem; doubles are not),
so it does not need one, and starting exactly at Hartree-Fock is both cleaner
and closer to the literature.

### The penalty has to match

`validate` is what tells you whether the number penalty is large enough: it
diagonalizes the *penalised* Hamiltonian exactly and fails if the ground state
does not sit in the target electron sector. If it does fail, raise the penalty
in **both** commands, because the receipt records the value and `vqe` rejects a
mismatch:

```powershell
python run.py validate --number-penalty 4
python run.py vqe      --number-penalty 4
```

On real LiH the default 1.0 Ha is sufficient in both spaces. On the self-test's
synthetic system it is not, and 4.0 Ha is; `selftest --check-penalty-sector`
runs the same search and reports the smallest adequate value.

## What the numbers mean

* **Chemical accuracy** here means within 1.6 mHa of the reference *for the
  identical cached Hamiltonian*. STO-3G itself sits roughly 1 kcal/mol from the
  basis-set limit, so this is a statement about the solver, not about experiment.
* `verified_chemical_accuracy` additionally requires that symmetry breaking and
  MPS truncation each cost under 0.16 mHa. A hardware-efficient ansatz does not
  conserve particle number, so the leak is measured and priced rather than
  assumed away.
* MPS is approximate whenever truncation bites, so the converged energy is
  always re-evaluated with `statevector` and the gap is reported.

## A note on simulator method and speed

`matrix_product_state` is the default for `vqe` because it is the method that
scales to the larger active spaces this workflow is a warm-up for, and because
`vqe` re-checks its converged energy against `statevector` and refuses to claim
accuracy if the two disagree.

At *this* size it is not the fast choice. Measured on this machine, 10 qubits
and an ~880-term objective, one energy-plus-gradient batch (101 circuits, 50
parameters):

| Method | Per L-BFGS-B iteration |
| --- | --- |
| `matrix_product_state` | ~13 s |
| `statevector` | ~2.4 s |

MPS pays a per-Pauli-term contraction cost that a 10-qubit statevector simply
does not have, and UCCSD's much deeper circuits widen the gap further. Use
`--method statevector` for LiH; use MPS when you are measuring MPS.

## Reading a failed run

| Symptom | Meaning |
| --- | --- |
| Energy **above** Hartree-Fock | Optimizer failure, not expressivity. Do not add layers. |
| Energy **below** the reference | The state left the electron sector. Raise `--number-penalty`. |
| `OUTSIDE CHEMICAL ACCURACY` | Ansatz-limited. On LiH, switch to `--ansatz uccsd`; adding layers buys less than it appears to (see Results). |
| `SYMMETRY BROKEN` | The state left the electron-number or spin sector. The energy is not comparable to the reference at all. Raise `--number-penalty`, or use an ansatz that conserves both. |
| `MPS TRUNCATION TOO COARSE` | Raise `--max-bond-dimension` or lower `--truncation-threshold`. |
| 0 optimizer iterations | The start is frozen — see below. |
| `Stale validation receipt` | A workflow file changed. Re-run `validate`. |

### The frozen start

At θ = 0 the ansatz *is* the Hartree-Fock determinant, which is the right
reference and, for the hardware-efficient circuit, a bad starting point. The
Hamiltonian conserves particle number, so a lone RY excitation out of the
reference has exactly zero gradient there. For **both** LiH spaces the whole
`real_amplitudes` circuit freezes:

```
frozen-core (10q, 2e)   Live parameters at theta=0 : 0/50 (|grad| = 3.553e-15)
full        (12q, 4e)   Live parameters at theta=0 : 0/60 (|grad| = 0.000e+00)
```

With `--start-noise 0`, L-BFGS-B therefore returns after zero iterations,
reporting clean convergence at E_HF. `real-amplitudes` defaults to a
`--start-noise 0.05` kick to break out of it.

A bigger kick is **not** a better one. Measured on the self-test's synthetic
2e/10-qubit system, 4 layers, L-BFGS-B to convergence:

| Start | Result |
| --- | --- |
| θ = 0 exactly | 0 iterations, stuck at E_HF |
| θ = 0 + N(0, 0.05) | converges, 0.0% of correlation recovered |
| θ = 0 + N(0, 0.2) | converges, 0.0% recovered |
| θ = 0 + N(0, 0.5) | converges **2.20 Ha above** E_HF |
| uniform [−π, π] | fails to converge, **1.29 Ha above** E_HF |

Near θ = 0 the landscape is flat; further out it is full of high local minima.
On real LiH the kick does let the optimizer move, and it still recovers
essentially none of the correlation — the limitation is the ansatz, not the
start.

UCCSD has no such problem: its doubles amplitudes have a finite gradient at the
Hartree-Fock reference, which is the same quantity that makes MP2 non-zero.

## What `selftest` proves — and what it doesn't

It proves the stack: operator conversion (spectrally *and* determinant by
determinant, which is what pins down qubit ordering), the Hartree-Fock frame
rotation, MPS against exact statevector, parameter-shift against finite
differences, and that the VQE respects the variational bound, terminates
cleanly, and never returns a point worse than its own start.

It does **not** assert that the VQE reaches the exact answer or beats
Hartree-Fock. Its Hamiltonian is random, and per the table above the
hardware-efficient ansatz genuinely cannot solve it at this filling. A self-test
that demanded otherwise would be asserting expressivity it cannot promise.
Correlation recovery is printed as a diagnostic, not scored.

For the claim that the *chemistry* is right, the evidence is
`tests/test_integrals.py` and `python run.py validate`, not `selftest`.

## Files

| File | Role |
| --- | --- |
| `system.py` | The two LiH specifications. The only file that knows this is LiH. |
| `integrals.py` | Built-in STO-3G integrals + RHF, so `prepare` runs without PySCF. |
| `chemistry.py` | CASCI construction, operators, cache I/O. No Qiskit imports. |
| `validation.py` | Exact classical validation; writes the receipt `vqe` demands. |
| `qiskit_runtime.py` | Aer evaluation, ansätze, batched gradients, the VQE. |
| `selftest.py` | End-to-end proof of the stack with no cache. |
| `run.py` | The single entry point. |
| `tests/test_core.py` | Specification, fingerprints, operator conversion, UCCSD ordering. |
| `tests/test_integrals.py` | The integral engine against published energies. |

Run them with `python -m unittest discover -s tests`.

`tests/test_core.py` parses `../lih/chemistry.py` and asserts the geometry and
basis match, so the Aer and QARP workflows cannot silently drift onto different
molecules.
