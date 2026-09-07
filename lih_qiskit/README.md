# LiH on Qiskit Aer — VQE against exact CASCI/FCI

A local VQE for **LiH / STO-3G** on Qiskit 2.5 + qiskit-aer. Integrals,
validation and the VQE all run on one machine, Windows included: the STO-3G
integrals and RHF are implemented in NumPy here, so **PySCF is optional**.

| | |
| --- | --- |
| System | LiH, r = 1.595 Å (experimental r<sub>e</sub>, NIST CCCBDB), STO-3G, vacuum |
| Active spaces | CASCI(2e,5o) → 10 qubits · FCI(4e,6o) → 12 qubits |
| Ansätze | `uccsd` · `real-amplitudes` (hardware-efficient baseline) |
| Reference | Exact diagonalisation in the N-electron sector |
| Target | 1.6 mHa vs. that reference, symmetry breaking priced under 0.16 mHa |

## Install

```powershell
uv venv --python 3.12 .venv && .venv\Scripts\activate
uv pip install -r requirements.txt
```

Verified: Python 3.12.13 · qiskit 2.5.2 · qiskit-aer 0.17.2 · qiskit-nature 0.8.0 ·
openfermion 1.8.1 · numpy 2.5.2 · scipy 1.18.0.

## Run

```powershell
python run.py selftest                    # prove the stack — no cache needed
python run.py backend                     # Aer, MPS vs statevector, gradients
python run.py prepare                     # build + cache the Hamiltonian
python run.py validate                    # exact checks; writes the receipt
python run.py vqe --ansatz uccsd --method statevector --start-noise 0
python -m unittest discover -s tests
```

`--space full` runs the 12-qubit space and may go on either side of the command:

```powershell
python run.py vqe --space full --ansatz uccsd --method statevector --start-noise 0
```

Each space keeps its own cache and result file, so switching cannot silently
reuse the other one's numbers. `vqe` refuses to start without a validation
receipt matching the cache, the specification, the workflow source and the
penalty settings — edit any workflow file and you must re-run `validate`.

### Options that matter

| Flag | Default | Notes |
| --- | --- | --- |
| `--space {frozen-core,full}` | `frozen-core` | The reference follows the space: CASCI(2e,5o) or FCI. |
| `--ansatz {real-amplitudes,uccsd}` | `real-amplitudes` | Use `uccsd` for LiH. |
| `--layers` | `4` | `real-amplitudes` depth. Ignored by UCCSD. |
| `--method {matrix_product_state,statevector}` | MPS | `statevector` is ~5× faster at this size. |
| `--start-noise` | `0.05` | Use `0` with UCCSD. See below. |
| `--number-penalty` | `1.0` | Must match the receipt. `validate --check-penalty-sector` finds the smallest adequate value. |
| `--integral-backend {auto,pyscf,native}` | `auto` | `prepare` only. |

**`--start-noise`**: at θ = 0 the ansatz *is* the Hartree–Fock determinant, and
`real_amplitudes` has **0 live parameters** there — the Hamiltonian conserves N,
so a lone RY excitation out of the reference has exactly zero gradient. Without
a kick, L-BFGS-B returns after zero iterations. A bigger kick is not a better
one: measured on the self-test system, σ = 0.5 converges 2.20 Ha *above* E_HF.
UCCSD has no such problem — 15 of 24 parameters are live at θ = 0.

## Results

LiH / STO-3G at r = 1.595 Å, noiseless `statevector`:

| | Frozen core (10q) | Full space (12q) |
| --- | --- | --- |
| E(RHF) | −7.862023874015 Ha | −7.862023874015 Ha |
| E(reference) | −7.882174519890 (CASCI) | −7.882401946215 (FCI) |
| Correlation available | 20.151 mHa | 20.378 mHa |
| Pauli terms | 276 | 631 |

| Ansatz | Space | Params | Error vs. reference | Circuits | Runtime | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| UCCSD | 10q | 24 | **+1.6 × 10⁻⁷ mHa** | 641 | 48 s | verified |
| UCCSD | 12q | 92 | **+0.011001 mHa** | 4,259 | 2,532 s | verified |
| `real_amplitudes` ×4 | 10q | 50 | +20.150496 mHa | ~41,000 | 409 s | outside accuracy |
| `real_amplitudes` ×8 | 10q | 90 | +3.407633 mHa | 95,391 | 1,473 s | **symmetry broken** |

UCCSD clears 1.6 mHa in both spaces, with particle-number leakage of 2.3 × 10⁻¹⁵
and 1.4 × 10⁻⁷ — the penalty machinery has nothing to correct.

**The 8-layer row is the one to read carefully.** Doubling the depth pulls the
energy from 20.15 to 3.41 mHa, but buys that by leaving the physical sector:
leakage 1.4 × 10⁻⁴ and ⟨S²⟩ = 2.1 × 10⁻³, which price out at 2.19 mHa against a
0.16 mHa budget. The symmetry it broke is worth more than the error it has left,
so the number is not a LiH energy at all. Without the symmetry audit it would
read as respectable progress toward chemical accuracy.

Chemical accuracy here means 1.6 mHa from the reference **for the identical
cached Hamiltonian**. STO-3G itself sits roughly 1 kcal/mol from the basis-set
limit, so this is a statement about the solver, not about experiment.

## Reading a run

| Symptom | Meaning |
| --- | --- |
| Energy **above** Hartree–Fock | Optimizer failure, not expressivity. Do not add layers. |
| Energy **below** the reference | The state left the electron sector. Raise `--number-penalty`. |
| `OUTSIDE CHEMICAL ACCURACY` | Ansatz-limited. Switch to `--ansatz uccsd`. |
| `SYMMETRY BROKEN` | Not comparable to the reference at all. |
| `MPS TRUNCATION TOO COARSE` | Raise `--max-bond-dimension` or lower `--truncation-threshold`. |
| 0 optimizer iterations | Frozen start — see `--start-noise` above. |
| `Stale validation receipt` | A workflow file changed. Re-run `validate`. |

## The native integral engine

PySCF publishes no Windows wheel, so `integrals.py` provides a self-contained
McMurchie–Davidson STO-3G implementation plus closed-shell RHF, in NumPy only.
`prepare` uses PySCF when importable and falls back to this otherwise; the engine
used is recorded in the cache as `integral_backend`.

Scope: STO-3G, closed shell, s and p functions — LiH and nothing beyond row 2.
It refuses other bases rather than guessing. `tests/test_integrals.py` pins it
against the H₂/STO-3G textbook energy (Szabo & Ostlund), the published LiH RHF
and FCI energies, and the requirement that the Hartree–Fock determinant of the
*second-quantized* Hamiltonian equal the SCF energy to 10 decimals in both active
spaces — which ties the CAS transformation, the two-electron transformation and
the frozen-core energy together.

## Report

[`report/`](report/) holds a standalone research report on this study —
*Ansatz Expressivity and Symmetry Accounting in a 10- and 12-Qubit LiH
Benchmark*, 11 pages. Build with `pdflatex main.tex` (three passes).

## Files

| File | Role |
| --- | --- |
| `system.py` | The two LiH specifications. The only file that knows this is LiH. |
| `integrals.py` | Built-in STO-3G integrals + RHF. |
| `chemistry.py` | CASCI construction, operators, cache I/O. No Qiskit imports. |
| `validation.py` | Exact validation; writes the receipt `vqe` demands. |
| `qiskit_runtime.py` | Aer evaluation, ansätze, batched gradients, the VQE. |
| `selftest.py` | End-to-end proof of the stack with no cache. |
| `run.py` | The single entry point. |
| `tests/` | Specification, fingerprints, operator conversion, integral engine. |

`selftest` proves the *stack* — operator conversion determinant by determinant
(which is what pins qubit ordering), MPS against statevector, parameter-shift
against finite differences, the variational bound. It does **not** assert the VQE
reaches the exact answer: its Hamiltonian is random. For the claim that the
chemistry is right, the evidence is `tests/test_integrals.py` and `validate`.

`tests/test_core.py` parses `../lih/chemistry.py` and asserts the geometry and
basis match, so this workflow and the QARP/Qulacs one cannot silently drift onto
different molecules.
