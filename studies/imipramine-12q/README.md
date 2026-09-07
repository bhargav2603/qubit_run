# Imipramine CASCI(6e,6o), 12 qubits — ADAPT-VQE on Qiskit Aer

A 12-qubit ADAPT-VQE study that runs locally on Qiskit and Qiskit Aer, with no
cluster and no vendor toolchain.

Imipramine is a tricyclic antidepressant and the closest thing at 12 qubits to a
drug molecule with published quantum work at the same active space. RHF/6-31G
canonical orbitals, CASCI(6e,6o), following the Hamiltonian setup reported in
[Koziell-Pipe et al. (2026)](https://arxiv.org/abs/2607.22468). It does **not**
claim to reproduce their ADAPT-GQE circuits or their private conformer dataset.

The chemistry and the simulator are kept strictly apart. `molecule.py`,
`hamiltonian.py` and `validation.py` import no Qiskit, and the validation
receipt is bound to a `physics_fingerprint` over exactly those three files — so
the execution layer can be edited on a machine that cannot re-validate, without
being able to alter, or invalidate the proof of, a Hamiltonian built
elsewhere.

| | |
| --- | --- |
| System | Imipramine, 45 atoms, PubChem CID 3696, 6-31G, 237 orbitals / 152 electrons |
| Active space | CASCI(6e,6o), canonical RHF frontier, 73 frozen core orbitals |
| Size | 12 qubits · 1,819 Pauli terms · FCI dimension 400 |
| Reference | CASCI, −842.0180419414 Ha (E_HF = −841.9973061558 Ha) |
| Correlation available | 20.736 mHa |
| Target | 1.6 mHa vs. CASCI, symmetry breaking priced under 0.16 mHa |

## Install

```powershell
uv venv --python 3.12 .venv && .venv\Scripts\activate
uv pip install -r requirements.txt
```

Verified: Windows 11 / Python 3.12.13 · qiskit 2.5.2 · qiskit-aer 0.17.2 ·
openfermion 1.8.1 · numpy 2.5.2 · scipy 1.18.0.

**PySCF publishes no Windows wheel** (it carries a `sys_platform != "win32"`
marker so `pip install -r` still succeeds). That affects `prepare` and
`classical` only — the expensive 45-atom RHF/CASCI work runs **once** and is
cached as JSON:

| Step | Needs | Where |
| --- | --- | --- |
| `prepare`, `classical` | PySCF | WSL / Linux / macOS / Colab — **once** |
| `validate` | OpenFermion + SciPy | anywhere, including Windows |
| `selftest`, `backend`, `vqe`, `adapt`, `summary`, `plot` | Qiskit + Aer | **Windows, natively** |

[`colab.ipynb`](colab.ipynb) does `prepare`, `validate` and `adapt` in one free
Colab session and hands back the cache, the receipt and the result.

## Run

**1. Prove the stack** (Windows, no cache):

```powershell
python run.py selftest                    # 67 checks, ~8 min
python run.py backend
python -m unittest discover -s tests -v   # 38 tests, ~9 s
```

**2. Build the Hamiltonian once**, where PySCF works:

```bash
python run.py prepare
python run.py classical      # MP2/CCSD/CCSD(T) in the same active space
```

**3. Copy `hamiltonian_cas6e6o.json` back, then anywhere:**

```powershell
python run.py validate --number-penalty 0 1 4
python run.py adapt
python run.py summary
python run.py plot
```

Pass `0 1 4` to `validate` so both `adapt` (which needs the unconstrained entry)
and `vqe` (which needs 1) have a receipt. Editing any workflow file invalidates
the receipt — re-run `validate`.

### Commands

| Command | Purpose | PySCF |
| --- | --- | --- |
| `selftest` | Three-stage proof of the whole local stack. Start here. | no |
| `backend` | Aer environment, MPS vs. statevector, gradient checks. | no |
| `prepare` | Builds and caches the Hamiltonian. `--diagnose` reports the orbital window and builds nothing. | **yes** |
| `classical` | MP2, CCSD, CCSD(T) in the same active space. | **yes** |
| `validate` | Exact checks; writes the hash-bound receipt. | no |
| `adapt` | **ADAPT-VQE — the published method.** Run this for the benchmark. | no |
| `vqe` | Hardware-efficient `real_amplitudes` baseline on Aer. | no |
| `summary` / `plot` / `deck` | Table, charts, presentation figures. | no |

### Which ansatz

| | `run.py adapt` | `run.py vqe` |
| --- | --- | --- |
| Ansatz | grown by gradient | `real_amplitudes`, fixed depth |
| Symmetry | N exact by construction; ⟨S²⟩ measured and priced | N broken, penalized, priced |
| Gradients | adjoint, O(k) | parameter-shift, O(k²) |
| Depth | an output | a knob you scan |
| Engine | SciPy sparse algebra, circuit verified at the end | Qiskit Aer circuits |
| Matches the paper | **yes** | no |

`adapt` is the one to run. `vqe` exists to measure the gap a shallow
hardware-efficient ansatz leaves, not to beat it.

### Options that matter

| Flag | Default | Notes |
| --- | --- | --- |
| `--pool {pair,uccsd,uccgsd}` | `uccgsd` | `adapt`. 435 / 63 / 15 operators. `uccgsd` is the paper's choice. |
| `--gradient-tolerance` | `1e-3` | `adapt`. Stop when no pool operator beats this. |
| `--max-operators` | `40` | `adapt`. Slack, not a target — published circuits used 8–19. |
| `--layers` | `4` | `vqe`. Each run writes `results/vqe_L<n>_<method>.json`, so a scan accumulates. |
| `--method {matrix_product_state,statevector}` | MPS | `vqe`. `statevector` is faster at 12 qubits. |
| `--number-penalty` | `1` (`vqe`), `0` (`adapt`) | Must match a receipt entry. |

## Results

The Hamiltonian was built on Colab (PySCF); both runs and the validation
receipt were regenerated locally on Windows 11 / Python 3.13.7 / qiskit 2.3.1.
Results live in `results/`.

| Run | Pool | Operators | Error vs. CASCI | ⟨S²⟩ | 2q gates | Depth | Time | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `adapt_uccgsd` | 435 ops | 58 | **+0.017252 mHa** | 1.1 × 10⁻⁶ | 4,241 | 6,977 | 412 s | verified |
| `adapt_uccsd` | 63 ops | 31 (budget exhausted) | +0.891663 mHa | 2.8 × 10⁻³ | 2,053 | 3,368 | 130 s | **spin contaminated** |

The `uccgsd` run stopped on its own gradient threshold (largest remaining pool
gradient 8.62 × 10⁻⁴ < 10⁻³) and its emitted Qiskit circuit reproduces the
optimized energy to 1.5 × 10⁻¹² Ha with exact (non-Trotter) synthesis.

Both energies are reproducible to the digits shown: the same run on Colab under
qiskit 1.0.2 gave +0.0173 and +0.8917 mHa. Only the transpiled gate counts and
wall times differ between the two stacks — the physics does not.
Particle-number leakage is **exactly zero** — the pool operators commute with N
by construction.

The `uccsd` run is the instructive failure. It hit `--max-operators 31` with a
pool gradient of 2.0 × 10⁻², i.e. still descending, and its ⟨S²⟩ prices out at
2.76 mHa against a 0.16 mHa budget. Its energy is 0.89 mHa from CASCI, which
*looks* like chemical accuracy — the spin audit is what stops it being reported
as such. Raise `--max-operators` or lower `--gradient-tolerance` to finish it.

**Classical baselines in the same active space:**

| Method | Energy (Ha) | Error vs. CASCI | Wall time |
| --- | --- | --- | --- |
| RHF | −841.9973061558 | +20.736 mHa | 92 s |
| MP2 | −842.0102524898 | +7.789 mHa | 5.7 s |
| CCSD | −842.0180213368 | +0.021 mHa | 253 s |
| CCSD(T) | −842.0180339214 | +0.008 mHa | 502 s |

Read the two tables together. CCSD(T) is 0.008 mHa from CASCI at 502 s;
ADAPT-VQE is 0.017 mHa at 133 s. CAS(6e,6o) is small enough that coupled cluster
is already excellent, so **this is a validation benchmark, not a demonstration of
quantum advantage** — which is exactly what the classical ladder exists to make
checkable rather than assumed.

**Published comparison** ([arXiv:2607.22468](https://arxiv.org/abs/2607.22468),
Table 1): statevector ADAPT-GQE reaches 1.15 mHa on CAS(6e,6o)/6-31G, and their
12-qubit datasets were trained to only 5–10 mHa. Their conformer differs from
this PubChem one, so compare **E_CASCI − E_HF**, not total energies.

## Why ADAPT is the right method here

**Particle number is structural, not penalized.** Every pool operator is a
fermionic excitation with balanced creation and annihilation indices, so it
commutes with N and S_z exactly. The state cannot leave the six-electron sector
— which is why `adapt` runs at `--number-penalty 0` and why the whole
penalty/sector-selection apparatus the hardware-efficient path needs does not
apply.

**S² is a weaker claim, and is not overstated.** Spin-complementing makes the
*singles* proper singlet operators and the `pair` pool is seniority-zero, but a
general spin-complemented **double does not** commute with S²:

| Pool | Singles breaking S² | Doubles breaking S² |
| --- | --- | --- |
| `pair` | — | 0 / 15 |
| `uccsd` | 0 / 9 | 27 / 54 |
| `uccgsd` | 0 / 15 | 315 / 420 |

In practice ⟨S²⟩ comes back at ~10⁻¹⁹: the Hamiltonian is spin-free and the
reference is a closed-shell singlet, so the variational minimum *is* the singlet
ground state. But that is a result of the optimization, not a property of the
ansatz — so ⟨S²⟩ is measured at the optimum and priced into the contamination
budget.

**Gradients are adjoint, and the optimization is sparse algebra.** Parameter
shift needs 2k+1 evaluations of k operator exponentials each — O(k²); the
reverse-mode sweep reuses every intermediate state and costs 2k exponentials
total. At 12 qubits the state is 4,096 amplitudes, so exp(θA)|ψ⟩ is one
`expm_multiply`, about a millisecond. The Qiskit circuit is built *at the end*,
transpiled, and its energy checked against the algebra.

## Reading the result

| Verdict | Meaning |
| --- | --- |
| `VERIFIED` | Within 1.6 mHa of CASCI, symmetry breaking and cross-checks inside budget, above the exact sector ground state. |
| `INCONSISTENT` | Below the exact six-electron ground state by more than its measured leak — impossible. Re-run `validate` and `selftest`; do not report the number. |
| `SPIN CONTAMINATED` | (`adapt`) ⟨N⟩ exact but ⟨S²⟩ is not zero — the run did not converge. Lower `--gradient-tolerance`, raise `--max-operators`, or use `--pool pair`. |
| `CIRCUIT DISAGREES` | (`adapt`) The emitted circuit does not reproduce the optimized energy. The energy is still correct; the circuit is not. |
| `MPS TOO COARSE` | (`vqe`) Raise `--max-bond-dimension` or lower `--truncation-threshold`. |
| `SYMMETRY BROKEN` | (`vqe`) The state left the six-electron or singlet sector. Raise `--number-penalty`. |
| `ABOVE HF` | Optimizer failure, not an ansatz limit. **Do not add layers.** |

The accuracy test is *energetic*, not a zero-leakage demand: a hardware-efficient
ansatz always leaks a little, so each broken symmetry is priced at 1 Ha per unit
and the total must stay under **0.16 mHa** — one tenth of chemical accuracy, small
enough that contamination cannot be what produced the reported error.

## Report

[`report/`](report/) holds a standalone research report on this study —
*ADAPT-VQE on Imipramine, CAS(6e,6o), 12 Qubits*, 13 pages. Build with
`pdflatex main.tex` (three passes).

## Files

| File | Purpose |
| --- | --- |
| `run.py` | The one interface. |
| `molecule.py` | Geometry, basis, active space, π-core diagnostics. The only imipramine-specific file. |
| `hamiltonian.py` | Molecule → cached qubit Hamiltonian. CASCI, Jordan–Wigner, cache I/O. Backend-independent. |
| `validation.py` | Exact classical checks and the hash-bound receipt. |
| `qiskit_runtime.py` | OpenFermion→Qiskit conversion, Aer MPS evaluation, parameter-shift VQE. |
| `adapt_runtime.py` | ADAPT-VQE: pools, adjoint-gradient optimization, circuit emission and verification. |
| `selftest.py` | Structure checks plus a full stack proof, no PySCF and no cache. |
| `summary.py` / `visualize.py` | Results table and charts. |
| `colab.ipynb` | One-click Colab: build, validate, run ADAPT, download. |
| `tests/` | Specification, fingerprint, receipt, operator-conversion and ADAPT-pool tests. |

Generated, not source: `hamiltonian_cas6e6o.json`,
`hamiltonian_cas6e6o.validated.json`, `results/`.
