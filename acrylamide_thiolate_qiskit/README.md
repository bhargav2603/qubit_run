# Acrylamide + methanethiolate — local Qiskit VQE benchmarked against exact FCI

A VQE-UCCSD study of the C–S bond-forming region of a thio-Michael addition,
following the **simulator protocol of CovAngelo** ([arXiv:2604.10487](https://arxiv.org/abs/2604.10487)),
built to run on a laptop with Qiskit 2.5 and Qiskit Aer.

**What this delivers:** the VQE energy against the exact classical answer for the
identical Hamiltonian, across a ladder of active spaces, at two geometries, with
a full classical benchmark ladder for context.

**What it is not:** a reproduction of CovAngelo's numbers. That paper's ECC-DMET
embedding with quantum-information-optimized bath orbitals is its actual
novelty, is not available in PySCF or Qiskit, and rebuilding it is a research
project. Absolute energies here are **not comparable to its Table 2**. What
carries over is the *methodology* and the VQE-vs-FCI agreement.

---

## Read this first: what the paper actually found

| | CovAngelo | Here |
|---|---|---|
| Ansatz | **UCCSD** (they tested hardware-efficient, symmetry-preserving and ADAPT-VQE, and reported UCCSD "most reliable") | UCCSD ✓ |
| Orbitals | **MP2 natural orbitals** for active-space selection | MP2 natural ✓ |
| Solvation | PCM, ε = 4 | C-PCM, ε = 4 ✓ |
| Geometries | TS and separated reactants → a difference | TS and FAR ✓ |
| Basis | 6-31++G\*\*, aug-cc-pVDZ | aug-cc-pVDZ (configurable) ✓ |
| Reference | FCI | CASCI = FCI in the active space ✓ |
| Hamiltonian | **ECC-DMET, 2 fragment orbitals + optimized bath** | CAS(ne,no) — **different** ✗ |
| Hardware run | IQM Garnet, 8 qubits | not attempted — see below |

Their hardware result missed FCI by **2.058–2.943 Ha** (≈1,300–1,850 kcal/mol);
the paper attributes this to device error rates. Their *simulator* results are
the reproducible ones, and they used Qiskit AerSimulator for them. This workflow
targets that regime.

Their own numbers also show orbital choice moving the VQE energy by **0.887 Ha** —
far more than any ansatz or optimizer detail. That is why `mp2_natural` is the
default here.

---

## The one blocker: PySCF has no Windows wheel

Confirmed on this machine — `uv pip install pyscf` fails at
`CMake Error: CMAKE_C_COMPILER not set`. There are macOS and Linux wheels only.

This affects exactly two commands, and the workflow is built to survive it:

| Command | Needs | Runs where |
|---|---|---|
| `prepare`, `ladder` | PySCF | Colab / WSL / Linux / macOS — **once** |
| `validate` | OpenFermion + SciPy | anywhere, including Windows |
| `selftest`, `backend`, `vqe` | Qiskit + Aer | **Windows, natively** |

The cache is plain JSON carrying its own specification hash, so a mismatched
file is a hard error, not a wrong answer.

---

## Step 0 — Environment (Windows, once)

```powershell
cd acrylamide_thiolate_qiskit
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Verified on Windows 11 / Python 3.12.13: **qiskit 2.5.2 · qiskit-aer 0.17.2 ·
qiskit-nature 0.8.0 · openfermion 1.8.1 · numpy 2.5.2 · scipy 1.18.0**.

qiskit-nature 0.8.0 predates Qiskit 2.x, so the pairing was *tested*, not
assumed: UCCSD builds, simulates on Aer, and places θ = 0 exactly on the
Hartree–Fock determinant.

## Step 1 — Prove the local stack, before any chemistry

```powershell
python run.py selftest        # ~65 s, 16 checks, no PySCF and no cache needed
python run.py backend
python -m unittest discover -s tests -v
```

`selftest` generates a random but *structurally real* CAS(4e,4o) Hamiltonian
through the same OpenFermion path the real one uses, then checks every component
against exact classical answers. Actual output:

```
[PASS] determinant energy formula is correct        classical vs qubit: -2.487e-14 Ha
[PASS] operator conversion: spectra agree           max |dE| = 7.105e-14 Ha
[PASS] interleaved ordering: determinants agree     max |dE| = 7.105e-15 Ha over 8 determinants
[PASS] blocked ordering: determinants agree         max |dE| = 7.105e-15 Ha over 8 determinants
[PASS] theta=0 is the reference determinant         error = -1.425e-12 Ha
[PASS] UCCSD conserves particle number exactly      <N> = 4.0000000000, leak = 6.800e-16
[PASS] parameter-shift correctly rejected for UCCSD parameters drive multiple rotations
[PASS] VQE respects the variational bound           E_vqe - E_exact = +0.9782 mHa
[PASS] VQE improves on the reference determinant    recovered 99.74 % of the correlation energy
16/16 checks passed
```

### The check that matters most: spin-orbital ordering

OpenFermion emits spin orbitals **interleaved** (2p = α_p, 2p+1 = β_p).
qiskit-nature's `JordanWignerMapper` expects them **blocked** (all α, then all β).
The permutation is `[0, 4, 1, 5, 2, 6, 3, 7]` at 8 qubits.

Get this wrong and you obtain a Hamiltonian with an **identical spectrum** and
completely wrong energies for every determinant — a bit permutation is a
similarity transform, so no eigenvalue check can detect it. It is therefore
verified determinant by determinant, in both conventions.

## Step 2 — Build the Hamiltonians where PySCF works

Upload this folder to Colab (or use WSL/Linux/macOS):

```bash
pip install -q pyscf openfermion "qiskit==2.5.*" "qiskit-aer==0.17.*" "qiskit-nature==0.8.*"
cd acrylamide_thiolate_qiskit
python colab_prepare.py --active-spaces 8q --geometries ts,far
```

This builds and validates every point, runs the classical ladder, and writes a
`.zip` to download. Unzip it here, then:

```powershell
python run.py validate --cache hamiltonian_ts_8q.json --number-penalty 0
python run.py vqe      --cache hamiltonian_ts_8q.json
```

Colab caveats, plainly: sessions die after ~90 min idle and are wiped on close —
download the archive before you close the tab. Free tier is ~2 vCPU, so CCSD(T)
on aug-cc-pVDZ is the slow part; use `--skip-ladder` for a first pass.

### If `prepare` refuses to build

A vacuum anion in a fully augmented basis often has a **positive HOMO**, so the
frontier orbitals describe the diffuse tail rather than the C–S region. The
active-space guard refuses to build in that case. MP2 natural orbitals usually
fix it — that is one reason the paper uses them. If it still fires, keep diffuse
functions only on the reacting atoms via a per-element basis dict, or override
with `--allow-delocalized-active-space` (and do not report the result as
describing the reaction region).

## Step 3 — The deliverable

```powershell
python run.py vqe --cache hamiltonian_ts_8q.json
python run.py vqe --cache hamiltonian_far_8q.json
```

Each run reports the VQE energy, the CASCI reference, the error in mHa, and the
percentage of correlation energy recovered. The **TS − FAR difference** of those
is the physically meaningful number; the classical ladder gives it context.

The ansatz is chosen automatically by register size — **UCCSD up to 10 qubits,
k-UpCCGSD above** — because UCCSD's parameter count and depth both grow as
O(N⁴) and it stops being affordable past ~12 qubits. Override at will:

```powershell
python run.py vqe --ansatz uccsd          # most accurate to ~12q
python run.py vqe --ansatz kupccgsd -k 3  # higher k = more accurate, proportionally slower
python run.py vqe --ansatz adapt          # grows the ansatz operator by operator
python run.py vqe --ansatz hea            # the ansatz the paper rejected, for comparison
python run.py vqe --method matrix_product_state
python run.py vqe --optimizer cobyla      # what the paper used
```

---

## Measurements that changed the defaults

Everything below was measured on this machine (Windows 11, 8 logical CPUs,
8 qubits, UCCSD depth 1646 / 1376 CX, 361 Pauli terms). Each one overturned an
assumption.

**1. `parameter_binds` is worth 31.5×.** Binding parameters with
`assign_parameters` costs **419 ms per circuit**, because each of the 26 UCCSD
parameters appears in ~8 gates as a symbolic expression evaluated in Python.
`backend.run` itself costs only 66 ms. Handing Aer the unbound circuit plus a
bind table moves it all into C++:

| 53-vector gradient batch | build | run | total |
|---|---|---|---|
| `assign_parameters` | 33.55 s | 9.18 s | **42.73 s** |
| `parameter_binds` | 0.04 s | 1.32 s | **1.36 s** |

Results identical to 3.9e-14. End to end this took the self-test VQE from
**407 s to 47 s**.

**2. Parallelism is method-dependent, and the obvious answer is wrong.**

| | threads=1, exp=8 | threads=8, exp=1 |
|---|---|---|
| statevector | 70.8 ms/circuit | **36.0 ms** |
| MPS | **83.9 ms** | 153.9 ms |

A *shallow* circuit is too small to thread, so circuit-level parallelism wins; a
*deep* one threads well. UCCSD is deep. Defaults are now set per method.

**3. Statevector beats MPS by 2.3× here — as predicted.** UCCSD's 1376 entangling
gates saturate the bond dimension (χ ≤ 2^(n/2) = 16 at 8 qubits), so MPS
compresses nothing and pays pure overhead. **The MPS memory knob is ansatz
depth, not qubit count.** MPS remains available and is the method that scales for
*weakly* entangled circuits at larger n; the converged energy is cross-checked
against exact statevector either way.

**4. Parameter-shift is invalid for UCCSD.** Each parameter drives ~8 rotations,
so the two-term shift rule silently returns a wrong gradient.
`parameter_shift_is_exact()` detects and refuses it. And qiskit-algorithms'
adjoint `ReverseEstimatorGradient` — the textbook "fast" answer — is numerically
correct (agrees to 1.7e-11) but took **619 s for one 26-component gradient**
versus 1.36 s for batched finite differences. It is not used.

**5. Aer hard-crashes on deep circuits on Windows — fixed here.** A 12-qubit
UCCSD circuit (depth 11,127) kills the process with `0xC00000FD`
(STACK_OVERFLOW), on both statevector and MPS, with no Python traceback:

| circuit | gates | depth | result |
|---|---|---|---|
| UCCSD 8q | 1,950 | 1,646 | fine |
| UCCSD 10q | 4,969 | 4,286 | fine |
| UCCSD 12q | 12,657 | **11,127** | **hard crash** |
| random 12q | 12,000 | 3,139 | fine |
| random 16q | 12,000 | 2,390 | fine |

It tracks circuit **depth**, not qubit or gate count — Aer recurses
proportionally to depth and a Windows thread gets a 1 MB stack against Linux's
8 MB. `run_with_deep_stack()` runs every Aer call on a 64 MB thread, after which
12q completes in 0.71 s. Note 256 MB and above fail to allocate the thread on
Windows, so 64 MB is not a knob to raise casually.

**6. Measured UCCSD scaling on this machine** (8 cores, statevector, molecular-
scale observable):

| | parameters | depth | 1 energy | 1 gradient | 50 iterations |
|---|---|---|---|---|---|
| 8q (4e,4o) | 26 | 1,646 | 0.77 s | 0.7 min | **0.6 h** |
| 12q (6e,6o) | 117 | 11,127 | 0.54 s | 2.1 min | **1.8 h** |
| 14q (7e,7o) | 204 | 22,163 | 2.38 s | 16.2 min | **13.5 h** |
| 16q (8e,8o) | — | — | not reached in ~20 min | — | — |

Depth roughly doubles per two qubits and parameter count grows as O(N⁴), so the
gradient cost compounds both ways. 8q and 12q are comfortable; 14q is an
overnight run; 16q was not measured to completion here.

**7. Ansatz comparison, measured** (8q synthetic system, correlation energy
374.58 mHa, exact sector FCI as reference):

| ansatz | params | depth | error vs FCI | correlation recovered | time |
|---|---|---|---|---|---|
| UCCSD | 26 | 1,646 | +0.978 mHa | 99.739 % | **8.7 s** |
| k-UpCCGSD k=1 | 14 | 544 | +7.939 mHa | 97.881 % | 6.1 s |
| k-UpCCGSD k=2 | 28 | 1,086 | **+0.342 mHa** | **99.909 %** | 95.3 s |
| k-UpCCGSD k=3 | 42 | 1,628 | **+0.136 mHa** | **99.964 %** | 228.4 s |
| ADAPT-VQE | 26 | 1,590 | +0.983 mHa | 99.738 % | 63.9 s |

Two things to read off this. **k-UpCCGSD at k≥2 is more accurate than UCCSD**,
because its generalized excitations are not restricted to occupied→virtual.
And **ADAPT selected all 26 of 26 pool operators — it rediscovered UCCSD
exactly**, for the same energy at 7.3x the cost. ADAPT earns its keep when the
pool is far larger than the number of operators actually needed; at 8 qubits it
is not, and it is pure overhead. It is implemented and available, but `auto`
does not select it.

**8. Scaling of k-UpCCGSD (k=2), measured** on 8 cores with an O(N⁴) Pauli
count, assuming a 50-iteration VQE:

| qubits | params | depth | 1 energy | 1 gradient | 50 iterations |
|---|---|---|---|---|---|
| 12q | 66 | 3,226 | 0.23 s | 31 s | **0.43 h** |
| 14q | 90 | 4,908 | 0.69 s | 124 s | **1.7 h** |
| 16q | 120 | 7,040 | 1.91 s | 459 s | **6.4 h** |
| 18q | 152 | 9,780 | 9.33 s | 2,846 s | **40 h** |
| 20q | 190 | 13,054 | 55.8 s | 21,271 s | **295 h** |

The wall is not memory — a 20-qubit statevector is 16 MB. It is that cost grows
as (parameters × depth × 2ⁿ × terms), and every factor grows at once.
**16 qubits is the practical ceiling on a laptop; 20 is out of reach.**

**9. UCCSD deletes the entire penalty apparatus.** It conserves particle number
exactly (leak = 3e-16, no penalty term), so there is no penalty to tune, no
sector to validate, no contamination budget, and no `SYMMETRY BROKEN` mode. The
hardware-efficient path needed all of it — and on the self-test system it left
256 mHa of error where UCCSD leaves **0.98 mHa**.

Note: UCCSD conserves N and S_z but **not S²** at arbitrary amplitudes — its
α→α and β→β amplitudes are independent. It becomes a spin eigenstate at the
variational minimum, which is where ⟨S²⟩ is checked.

---

## Reading the result

| Verdict | Meaning |
|---|---|
| `VERIFIED` | Within 1.6 mHa of CASCI, symmetry breaking and MPS truncation inside budget. |
| `MPS TRUNCATION TOO COARSE` | Raise `--max-bond-dimension` or lower `--truncation-threshold`. |
| `FAILURE: above the reference determinant` | Optimizer failure, not an ansatz limit. |
| `SYMMETRY BROKEN` | Only reachable with `--ansatz hea`. Raise `--number-penalty`, or switch to UCCSD. |
| `OUTSIDE CHEMICAL ACCURACY` | Ansatz-limited. Raise `--maxiter` first. |

## Scope and honesty

- The two geometries are **rigid**, so TS − FAR is a rigid-scan energy
  difference along an approach coordinate, **not a relaxed reaction barrier**.
- Chemical accuracy here means agreement with CASCI **for this exact active-space
  Hamiltonian**. The `ladder` command exists to show how far that active space
  is from converged chemistry — the CASCI−CCSD(T) gap is almost always far
  larger than the VQE's error against CASCI, and should be reported alongside it.
- This is a noiseless simulation. It says nothing about hardware feasibility.
- Everything is caching-safe: caches carry a specification hash, results carry a
  workflow hash covering `qiskit_runtime.py`, and JSON is written atomically.

## Files

| File | Purpose |
|---|---|
| `system.py` | Geometries (TS/FAR), active-space ladder, `make_spec()`. |
| `chemistry.py` | PySCF/CASCI construction, PCM, MP2 natural orbitals, cache I/O, fingerprints. |
| `classical.py` | HF/MP2/CCSD/CCSD(T)/CASCI ladder and the TS−FAR table. |
| `validation.py` | Exact sparse-matrix checks and the hash-bound receipt. |
| `qiskit_runtime.py` | Operator conversion, ordering permutation, UCCSD/k-UpCCGSD, batched VQE. |
| `adapt.py` | ADAPT-VQE with sparse-matrix operator screening. |
| `selftest.py` | Full-stack proof with no PySCF and no cache. |
| `prepare_point.py` | Build/validate one geometry × active-space point. |
| `colab_prepare.py` | Drive the whole PySCF stage on Colab and package the results. |
| `run.py` | CLI: `selftest`, `backend`, `validate`, `vqe`, `prepare`, `ladder`. |
