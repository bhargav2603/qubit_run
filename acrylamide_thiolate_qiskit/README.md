# Acrylamide + methanethiolate — Qiskit VQE benchmarked against exact FCI

A VQE study of the **C–S bond-forming region of a thio-Michael addition**,
following the simulator protocol of *CovAngelo*
([arXiv:2604.10487](https://arxiv.org/abs/2604.10487)), on a laptop with
Qiskit 2.5 and Qiskit Aer.

**Delivers:** the VQE energy against the exact classical answer for the
*identical* Hamiltonian, at two geometries, with a classical ladder
(HF / MP2 / CCSD / CCSD(T) / CASCI) for context.

**Does not deliver:** a reproduction of CovAngelo's numbers. Their ECC-DMET
embedding with quantum-information-optimized bath orbitals is the paper's actual
novelty and is not available in PySCF or Qiskit. Absolute energies here are
**not comparable to its Table 2**; the methodology and the VQE-vs-FCI agreement
are what carry over.

| | |
| --- | --- |
| System | Acrylamide + CH₃S⁻, geometries `ts` and `far` (rigid) |
| Basis | aug-cc-pVDZ (configurable), C-PCM at ε = 4, MP2 natural orbitals |
| Active space | CAS(ne,no) ladder; 8 qubits is the working default |
| Ansätze | `uccsd` · `kupccgsd` · `adapt` · `hea`; `auto` picks by register size |
| Reference | CASCI, which equals FCI inside the active space |
| Target | 1.6 mHa vs. that reference |

## Install

```powershell
uv venv --python 3.12 .venv && .venv\Scripts\activate
uv pip install -r requirements.txt
```

Verified: Windows 11 / Python 3.12.13 · qiskit 2.5.2 · qiskit-aer 0.17.2 ·
qiskit-nature 0.8.0 · openfermion 1.8.1 · numpy 2.5.2 · scipy 1.18.0.

**PySCF publishes no Windows wheel.** It carries a `sys_platform != "win32"`
marker so `pip install -r` still succeeds there. This affects two commands only:

| Command | Needs | Runs where |
| --- | --- | --- |
| `prepare`, `ladder` | PySCF | Colab / WSL / Linux / macOS — **once** |
| `validate` | OpenFermion + SciPy | anywhere, including Windows |
| `selftest`, `backend`, `vqe` | Qiskit + Aer | **Windows, natively** |

The cache is plain JSON carrying its own specification hash, so a mismatched
file is a hard error, not a wrong answer.

## Run

**1. Prove the stack** (Windows, no chemistry package, no cache):

```powershell
python run.py selftest          # ~65 s, 16 checks
python run.py backend
python -m unittest discover -s tests -v
```

**2. Build the Hamiltonians** where PySCF works — Colab, WSL, Linux, macOS:

```bash
pip install -q pyscf openfermion "qiskit==2.5.*" "qiskit-aer==0.17.*" "qiskit-nature==0.8.*"
python colab_prepare.py --active-spaces 8q --geometries ts,far
```

This builds and validates every point, runs the classical ladder, and writes a
`.zip` to download. Colab sessions are wiped on close — download before closing.
CCSD(T) on aug-cc-pVDZ is the slow part; use `--skip-ladder` for a first pass.

**3. Unzip here and run the VQE** (anywhere):

```powershell
python run.py validate --cache hamiltonian_ts_8q.json --number-penalty 0
python run.py vqe      --cache hamiltonian_ts_8q.json
python run.py vqe      --cache hamiltonian_far_8q.json
```

Each run reports the VQE energy, the CASCI reference, the error in mHa and the
correlation recovered. The **TS − FAR difference** is the physically meaningful
number; `python run.py ladder` gives it classical context.

### Commands

| Command | Purpose | PySCF |
| --- | --- | --- |
| `selftest` | Full-stack proof on a synthetic but structurally real CAS(4e,4o). Start here. | no |
| `backend` | Aer inspection, MPS vs. statevector, VQE iteration timing. | no |
| `validate` | Exact sparse-matrix checks; writes the receipt. | no |
| `vqe` | Runs the VQE, writes the result JSON. | no |
| `prepare` | Builds and caches one geometry × active-space point. | **yes** |
| `ladder` | Classical benchmark ladder and the TS − FAR table. | **yes** |

### Options that matter

| Flag | Default | Notes |
| --- | --- | --- |
| `--ansatz {auto,uccsd,kupccgsd,adapt,hea}` | `auto` | UCCSD ≤ 10 qubits, k-UpCCGSD above. |
| `--kupccgsd-k` | `2` | Higher k = more accurate, proportionally slower. |
| `--method {statevector,matrix_product_state}` | `statevector` | See below. |
| `--optimizer` | `l-bfgs-b` | `cobyla` is what the paper used. |
| `--geometry {ts,far}` · `--active-space` | — | `prepare` only. |
| `--orbital-selection {mp2_natural,canonical}` | `mp2_natural` | The paper's choice; orbital choice moved *their* VQE energy by 0.887 Ha. |
| `--solvent-epsilon` | `4` | C-PCM dielectric. |
| `--number-penalty` | must match the receipt | UCCSD conserves N exactly, so use `0`. |

```powershell
python run.py vqe --ansatz kupccgsd -k 3
python run.py vqe --ansatz hea            # the ansatz the paper rejected, for comparison
```

## Results

Measured on this machine (Windows 11, 8 logical CPUs) on the self-test's
synthetic 8-qubit active space, whose exact sector FCI energy is known and whose
correlation energy is 374.58 mHa:

| Ansatz | Params | Depth | Error vs. FCI | Correlation | Time |
| --- | --- | --- | --- | --- | --- |
| UCCSD | 26 | 1,646 | +0.978 mHa | 99.739 % | **8.7 s** |
| k-UpCCGSD, k=1 | 14 | 544 | +7.939 mHa | 97.881 % | 6.1 s |
| k-UpCCGSD, k=2 | 28 | 1,086 | **+0.342 mHa** | 99.909 % | 95.3 s |
| k-UpCCGSD, k=3 | 42 | 1,628 | **+0.136 mHa** | 99.964 % | 228.4 s |
| ADAPT-VQE | 26 | 1,590 | +0.983 mHa | 99.738 % | 63.9 s |

k-UpCCGSD at k ≥ 2 beats UCCSD because its generalized excitations are not
restricted to occupied → virtual. **ADAPT selected all 26 of 26 pool operators —
it rediscovered UCCSD exactly**, for the same energy at 7.3× the cost; at 8
qubits the pool is not larger than the number of operators actually needed, so
`auto` does not select it.

Measured UCCSD cost on 8 cores, statevector, 50 VQE iterations:

| | Params | Depth | 1 gradient | 50 iterations |
| --- | --- | --- | --- | --- |
| 8q (4e,4o) | 26 | 1,646 | 0.7 min | 0.6 h |
| 12q (6e,6o) | 117 | 11,127 | 2.1 min | 1.8 h |
| 14q (7e,7o) | 204 | 22,163 | 16.2 min | 13.5 h |

With k-UpCCGSD (k=2), 16 qubits costs ~6.4 h and 20 qubits ~295 h. The wall is
not memory — a 20-qubit statevector is 16 MB — it is that cost grows as
(parameters × depth × 2ⁿ × terms) and every factor grows at once.
**16 qubits is the practical ceiling on a laptop.**

## Implementation notes worth knowing

**Spin-orbital ordering is the check that matters most.** OpenFermion emits spin
orbitals *interleaved* (2p = α_p, 2p+1 = β_p); qiskit-nature's
`JordanWignerMapper` expects them *blocked*. Get this wrong and you obtain a
Hamiltonian with an **identical spectrum** and completely wrong energies for
every determinant — a bit permutation is a similarity transform, so no
eigenvalue check can detect it. `selftest` verifies it determinant by
determinant, in both conventions.

**Statevector, not MPS, at this size.** UCCSD's 1,376 entangling gates saturate
the bond dimension (χ ≤ 2^(n/2) = 16 at 8 qubits), so MPS compresses nothing and
pays pure overhead — statevector is 2.3× faster. The MPS memory knob is *ansatz
depth*, not qubit count. MPS remains available and is the method that scales for
weakly entangled circuits at larger n.

**Parameter-shift is invalid for UCCSD.** Each parameter drives ~8 rotations, so
the two-term shift rule silently returns a wrong gradient;
`parameter_shift_is_exact()` detects and refuses it. Batched finite differences
are used instead, via Aer's `parameter_binds` — that alone is worth 31.5×
(42.7 s → 1.4 s for a 53-vector gradient batch, identical to 3.9 × 10⁻¹⁴).

**Aer hard-crashes on deep circuits on Windows.** A 12-qubit UCCSD circuit
(depth 11,127) kills the process with `0xC00000FD` (STACK_OVERFLOW), with no
traceback. It tracks circuit *depth*, not qubit or gate count: Aer recurses
proportionally to depth and a Windows thread gets a 1 MB stack against Linux's
8 MB. `run_with_deep_stack()` runs every Aer call on a 64 MB thread, after which
12q completes in 0.71 s. 256 MB and above fail to allocate on Windows, so 64 MB
is not a knob to raise casually.

**UCCSD deletes the penalty apparatus.** It conserves particle number exactly
(leak 3 × 10⁻¹⁶), so there is no penalty to tune and no `SYMMETRY BROKEN` mode.
It conserves N and S_z but **not S²** at arbitrary amplitudes — it becomes a spin
eigenstate at the variational minimum, which is where ⟨S²⟩ is checked.

## Reading the result

| Verdict | Meaning |
| --- | --- |
| `VERIFIED` | Within 1.6 mHa of CASCI, symmetry breaking and MPS truncation inside budget. |
| `MPS TRUNCATION TOO COARSE` | Raise `--max-bond-dimension` or lower `--truncation-threshold`. |
| `FAILURE: above the reference determinant` | Optimizer failure, not an ansatz limit. |
| `SYMMETRY BROKEN` | Only reachable with `--ansatz hea`. Raise `--number-penalty`, or use UCCSD. |
| `OUTSIDE CHEMICAL ACCURACY` | Ansatz-limited. Raise `--maxiter` first. |

**If `prepare` refuses to build:** a vacuum anion in a fully augmented basis
often has a positive HOMO, so the frontier orbitals describe the diffuse tail
rather than the C–S region, and the active-space guard refuses. MP2 natural
orbitals usually fix it. Otherwise keep diffuse functions only on the reacting
atoms via a per-element basis dict, or override with
`--allow-delocalized-active-space` — and do not then report the result as
describing the reaction region.

## Scope

- The two geometries are **rigid**, so TS − FAR is a rigid-scan energy difference
  along an approach coordinate, **not a relaxed reaction barrier**.
- Chemical accuracy means agreement with CASCI **for this exact active-space
  Hamiltonian**. The CASCI − CCSD(T) gap from `ladder` is almost always far
  larger than the VQE's error against CASCI, and should be reported alongside it.
- This is a noiseless simulation. It says nothing about hardware feasibility.
  (CovAngelo's own IQM Garnet run missed FCI by 2.058–2.943 Ha.)

## Files

| File | Purpose |
| --- | --- |
| `system.py` | Geometries (TS/FAR), active-space ladder, `make_spec()`. |
| `chemistry.py` | PySCF/CASCI construction, PCM, MP2 natural orbitals, cache I/O. |
| `classical.py` | The HF/MP2/CCSD/CCSD(T)/CASCI ladder and the TS−FAR table. |
| `validation.py` | Exact sparse-matrix checks and the hash-bound receipt. |
| `qiskit_runtime.py` | Operator conversion, ordering permutation, ansätze, batched VQE. |
| `adapt.py` | ADAPT-VQE with sparse-matrix operator screening. |
| `selftest.py` | Full-stack proof with no PySCF and no cache. |
| `prepare_point.py` / `colab_prepare.py` | One point / the whole PySCF stage. |
| `run.py` | The single entry point. |
