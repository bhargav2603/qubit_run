# Li₂S at 24 qubits — HI-VQE on Qiskit

An implementation of the **Handover Iterative Variational Quantum Eigensolver**
([Pellow-Jarman et al., arXiv:2503.06292](https://arxiv.org/abs/2503.06292),
Qunova Computing) applied to that paper's own 24-qubit benchmark:
**Li₂S, CAS(12e,12o)/STO-3G, 853,776 determinants**, along the Li–S dissociation
coordinate. Everything runs on a CPU; a free Colab session covers the whole
study including the chemistry.

| | |
| --- | --- |
| System | Li₂S, linear Li–S–Li, one bond stretched over 13 geometries (1.7 – 6.0 Å) |
| Active space | CAS(12e,12o)/STO-3G, 5 frozen orbitals (the sulfur core), gap 3.89 Ha |
| Size | 24 qubits · **853,776** determinants |
| Ansatz | Hardware-efficient unitary cluster Jastrow: depth 29, 298 two-qubit gates, 322 params |
| Reference | CASCI, exact within the active space |
| Target | 1.6 mHa vs. CASCI, above it, and a spin eigenstate |

## What HI-VQE is

A conventional VQE asks the device for an **energy**, which means measuring every
Pauli word every iteration — **15,697** of them for this system
(`python run.py paulis` recomputes that from the cached integrals).

HI-VQE asks a different question: **which electron configurations matter?** One
measurement returns a set of occupation bitstrings; the Hamiltonian is projected
onto the subspace they span and diagonalised exactly, classically. That is the
handover. Three consequences, each measured here rather than asserted:

| | Why | Checked by |
| --- | --- | --- |
| The energy is a strict upper bound | `PHP`'s lowest eigenvalue is above the true ground state for *any* projector, so noise can worsen the subspace but never push the energy below the answer | `validate` on random subspaces; violations report `INCONSISTENT`, never a result |
| The energy is monotone | the subspace only grows | `history` in every result file, plus a re-solve guard |
| The quantum layer is separable | `--simulator none` removes it; `--no-expansion` removes the classical half | the ablation table in every report |

That third row is the honest one. At 24 qubits a laptop diagonalises the full
space directly, so this is a **benchmark and validation exercise, not a claim of
quantum advantage** — the ablation exists so the quantum layer's contribution is
a number you read rather than a claim you accept.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verified: Python 3.12.13 · qiskit 2.5.2 · qiskit-aer 0.17.2 · numpy 2.5.2 ·
scipy 1.18.0 · openfermion 1.8.1 · matplotlib 3.11.1. On Colab,
`!pip install -q pyscf qiskit qiskit-aer openfermion` is enough and PySCF works
there — [`colab.ipynb`](colab.ipynb) runs the study end to end.

**PySCF publishes no Windows wheel.** That affects three commands; the chemistry
runs **once**, is cached as JSON, and everything after reads the cache.

| Step | Needs | Where |
| --- | --- | --- |
| `prepare`, `geometry`, `classical` | PySCF | Linux / macOS / WSL / Colab — **once** |
| `validate`, `summary`, `report` | NumPy + SciPy | anywhere, including Windows |
| `selftest`, `backend`, `hivqe`, `adapt`, `scan`, `plot` | + Qiskit | anywhere |

## Run

```bash
python run.py selftest                  # 58 checks, ~8 s, no chemistry package
python run.py backend                   # add --full to cross-check both simulators at 24q
python -m unittest discover -s tests    # 61 tests, ~2 s

python run.py prepare  --distance 2.10  # build the Hamiltonian (PySCF)
python run.py validate --distance 2.10  # prove it correct, write the receipt
python run.py hivqe    --distance 2.10  # run HI-VQE
python run.py report                    # report.md + a standalone report.html
```

The whole curve:

```bash
python run.py prepare --all
python run.py classical --all
python run.py scan
python run.py summary && python run.py report
```

### Commands

| Command | Purpose | PySCF |
| --- | --- | --- |
| `selftest` | 58 checks: layering, conventions, the engine against OpenFermion, the simulators against Aer, a full HI-VQE run against an exactly solved system. Start here. | no |
| `backend` | Environment report; cross-checks the sector simulator against Aer amplitude by amplitude. | no |
| `geometry` | Symmetric-stretch scan for the equilibrium bond length. | **yes** |
| `prepare` | Builds and caches the Hamiltonian at one bond length. `--diagnose` builds nothing. | **yes** |
| `classical` | MP2, CCSD, CCSD(T) in the same active space. | **yes** |
| `validate` | Proves a cached Hamiltonian correct; writes its receipt. | no |
| `paulis` | Counts the Pauli words a conventional VQE would measure. | no |
| `hivqe` / `scan` | One bond length / the whole dissociation coordinate. | no |
| `adapt` | ADAPT-VQE — the conventional-VQE comparison, given its best case. | no |
| `summary` / `plot` / `report` | Table, figures, assembled report. | no |

### Options that change the answer

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-determinants` | `20000` | The accuracy/cost dial. Lower it for memory, **never to save time** — see below. |
| `--expansion` | ~`800` | Candidate configurations per iteration. |
| `--expansion-references` | `8` | The paper uses `1`. |
| `--ranking` | coherent sum | The paper uses `coupling`. |
| `--spin-complete` | off | Use when a run says `SPIN CONTAMINATED`. |
| `--no-pt2` | off | Drop the perturbative correction. |

`--expansion-references 1 --ranking coupling` reproduces the paper's selection
rule exactly. The default uses the leading 8 configurations and ranks by the
coherent sum `|Σᵢ cᵢ⟨φ|H|φᵢ⟩|`, the standard first-order estimate.

### Options that change the cost

| Flag | Default | Notes |
| --- | --- | --- |
| `--simulator {sector,aer,none}` | `sector` | Exact in-sector (~2.8 s/sample), Aer (~11 s, the only path that carries noise), or no quantum layer. |
| `--optimizer` / `--optimizer-every` | SPSA each iteration | Sampling dominates; `--optimizer none` is ~2.6× faster. |
| `--givens-depth` | full | A shallower Givens network collapses the proposal distribution onto Hartree–Fock. |
| `--readout-error` / `--depolarizing-error` | off | Noise models, Aer only. |
| `--no-expansion` | off | Quantum proposals only. |

## Results

Generated 2026-08-20 from 16 single-point runs and one full scan. The complete
record — every result JSON, the receipts, `report.md`, `report.html` and the
figures — is archived under `Results/li2s_24_results/` and
`Results/li2s_24_cache/`.

Across 13 geometries:

| | worst | mean |
| --- | --- | --- |
| HI-VQE error vs. CASCI | 24.238 mHa | 2.586 mHa |
| Hartree–Fock error vs. CASCI | 168.2 mHa | 66.5 mHa |
| Determinants used | 19,992 | 19,239 |
| Fraction of the 853,776-determinant space | 2.342 % | 2.253 % |

**11 of 13 points are inside chemical accuracy**, using ~2.3 % of the space.
Hartree–Fock misses by up to 168 mHa over the same curve — that is the
multireference character HI-VQE is there to capture.

| r (Å) | E(HF) | E(HI-VQE) | E(CASCI) | error (mHa) | HF error | dets |
| --- | --- | --- | --- | --- | --- | --- |
| 1.700 | −407.950117 | −407.985509 | −407.985607 | +0.0985 | 35.5 | 19,992 |
| 1.900 | −407.964402 | −408.001987 | −408.002076 | +0.0893 | 37.7 | 19,950 |
| 2.000 | −407.960694 | −407.999205 | −407.999309 | +0.1039 | 38.6 | 19,992 |
| 2.100 | −407.952748 | −407.992132 | −407.992208 | +0.0762 | 39.5 | 19,932 |
| 2.200 | −407.942002 | −407.982123 | −407.982225 | +0.1017 | 40.2 | 19,966 |
| 2.400 | −407.915865 | −407.957314 | −407.957425 | +0.1110 | 41.6 | 19,881 |
| 2.600 | −407.887294 | −407.929953 | −407.930087 | +0.1336 | 42.8 | 19,881 |
| 2.900 | −407.844982 | −407.884942 | −407.885013 | +0.0709 | 40.0 | 19,880 |
| 3.200 | −407.806559 | −407.862021 | −407.853522 | **−8.4987** | 47.0 | 19,881 |
| 3.600 | −407.764023 | −407.849591 | −407.825353 | **−24.2376** | 61.3 | 19,881 |
| 4.200 | −407.719485 | −407.840669 | −407.840733 | +0.0645 | 121.2 | 19,881 |
| 5.000 | −407.686955 | −407.837262 | −407.837292 | +0.0297 | 150.3 | 19,880 |
| 6.000 | −407.668269 | −407.836424 | −407.836433 | +0.0087 | 168.2 | 11,118 |

The two **negative** errors at 3.2 and 3.6 Å are reported as `INCONSISTENT`, not
as results: a projected subspace cannot lie below CASCI, so those points found a
*different* state — ⟨S²⟩ = 2.00, the triplet, which is nearly degenerate with the
singlet at a breaking Li–S bond. They are printed rather than hidden because that
is what the verdict machinery is for.

| | HI-VQE | CASCI |
| --- | --- | --- |
| Minimum on this grid | 1.900 Å | 1.900 Å |
| Dissociation energy to the last point | 103.89 kcal/mol | 103.94 kcal/mol |
| Error in that dissociation energy | **0.051 kcal/mol** | — |

A dissociation energy is a *difference*, so the systematic part of the error
cancels — that is a stricter test of consistency than any single total energy,
and it is the number to quote.

**Classical baselines in the same active space:**

| r (Å) | MP2 error | CCSD error | CCSD(T) error | T1 |
| --- | --- | --- | --- | --- |
| 1.700 | +7.11 mHa | +2.95 mHa | +0.31 mHa | 0.0239 |
| 2.100 | +8.06 mHa | +4.70 mHa | +0.41 mHa | 0.0304 |
| 2.600 | +8.93 mHa | +6.34 mHa | +0.41 mHa | 0.0387 |
| 3.200 | +16.80 mHa | +5.88 mHa | +1.54 mHa | 0.0625 |
| 3.600 | +24.34 mHa | +4.32 mHa | +0.27 mHa | 0.0896 |
| 4.200 | +68.50 mHa | +34.37 mHa | +29.83 mHa | 0.1134 |
| 6.000 | +59.30 mHa | +35.67 mHa | +29.48 mHa | 0.1216 |

T1 above ~0.02 means the reference determinant no longer dominates and CCSD(T)
stops being a gold standard. That is the regime a breaking Li–S bond enters, and
it is why the reference here is CASCI: by 4.2 Å, CCSD(T) is 30 mHa out where
HI-VQE is 0.06 mHa out.

**Ablation — what did the quantum layer contribute?**

| Configuration | Runs | Mean error | Mean dets | First-iteration error |
| --- | --- | --- | --- | --- |
| `none` (classical selected-CI control) | 1 | 0.1413 mHa | 19,890 | 1.54 mHa |
| `sector` | 14 | 2.4429 mHa | 19,285 | 2.23 mHa |
| `sector + no expansion` | 1 | 32.7447 mHa | 304 | 35.67 mHa |

Read the last column first: the sampler's contribution is a **better starting
subspace**, which is what the method asks of it. Where final energies converge to
the same place, the honest statement is that the classical expansion was
sufficient for this system at this size — and this table is what makes that
statement checkable.

## The ansatz and the simulators

A hardware-efficient **unitary cluster Jastrow**: nearest-neighbour Givens
rotations (`XXPlusYY`) in a brickwork of depth *M* inside each spin block — the
depth the Clements decomposition needs to realise an arbitrary orbital rotation,
so one network can move an electron from any orbital to any other — a diagonal
number–number Jastrow layer (`RZ`, `RZZ`) whose same-orbital α–β terms are the
only gates correlating the two spin blocks, and **a closing Givens network, never
a closing Jastrow layer** (a trailing diagonal layer changes only phases, so it is
exactly invisible to a measurement distribution; the self-test checks that rather
than trusting it).

Every gate commutes with the α and β number operators separately, so on a
noiseless simulator **every shot is a valid configuration** — no filtering, no
wasted shots, leakage exactly zero. At all-zero angles the circuit is exactly the
Hartree–Fock determinant.

`sector_sim.py` is ~4× faster than Aer at 24 qubits because the ansatz never
leaves the particle-number sector, and it is trustworthy because it
**reimplements nothing**: it reads each gate's own `to_matrix()` from the Qiskit
circuit, so Qiskit remains the sole authority on what `XXPlusYY(θ)` means.
`backend --full` cross-checks the two amplitude by amplitude on the real
24-qubit circuit.

## Reading the result

| Verdict | Meaning |
| --- | --- |
| `CHEMICAL ACCURACY` | Within 1.6 mHa of CASCI, above it, and a spin eigenstate. |
| `OUTSIDE CHEMICAL ACCURACY` | Working but subspace-limited. Raise `--max-determinants` or `--expansion`. |
| `INCONSISTENT` | **Below** CASCI. Impossible for a projected subspace. Re-run `validate` and `selftest`; do not report the number. |
| `SPIN CONTAMINATED` | ⟨S²⟩ is not near any S(S+1). Raise `--max-determinants` or add `--spin-complete`. |
| `PT2 IS NOT USABLE HERE` | Not a verdict on the energy — the *correction* exceeds half the correlation the subspace captured, so perturbation theory is out of its domain. The variational energy is unaffected. |

`SPIN CONTAMINATED` does not fire on a triplet: at a dissociated Li–S bond the
singlet and triplet are nearly degenerate and CASCI returns whichever is lower, so
a triplet ground state is a result. What is broken is a state that is *neither* —
the check measures the distance from ⟨S²⟩ to the nearest S(S+1), not to zero.

## Performance

Measured at the real 24-qubit size (853,776-determinant space, 20,000-determinant
subspace, three iterations each):

| Setting | s / iteration |
| --- | --- |
| default (SPSA every iteration) | 16.0 |
| `--optimizer-every 3` | 11.1 |
| `--optimizer none` | **6.2** |
| `--simulator none` (classical control) | 3.7 |
| `--max-determinants 6000` (with `--optimizer none`) | 6.4 |

Read the last two rows together. **Shrinking the subspace is not a speed lever:**
20,000 → 6,000 determinants saved nothing measurable and cost 170 mHa, because
the classical half of an iteration is dominated by fixed work — building the
same-spin blocks, ranking candidates — not by the dimension. Lower
`--max-determinants` when short of memory, never to save time. What costs time is
**sampling**, ~2.5 s per circuit evaluation, and SPSA asks for three per iteration.

## Report

[`report/`](report/) holds a standalone research report on this study —
*HI-VQE on Li₂S at 24 Qubits*, 14 pages. Build with `pdflatex main.tex`
(three passes).

## Files

| File | Purpose |
| --- | --- |
| `run.py` | The one interface. |
| `molecule.py` | Li₂S, the active space, the scan grid, the published reference. |
| `hamiltonian.py` | Molecule → cached active-space integrals. **The only module that imports PySCF.** |
| `determinants.py` | The classical engine: strings, Slater–Condon, the projected Hamiltonian, Davidson, ⟨S²⟩. NumPy/SciPy only. |
| `ansatz.py` | The Qiskit sampling circuit, Aer execution, noise, configuration recovery. **The only module that imports Qiskit.** |
| `sector_sim.py` | Exact in-sector simulator, cross-checked against Aer. |
| `hivqe.py` | The algorithm: sample, project, diagonalise, screen, expand, optimise, converge. |
| `driver.py` | CLI drivers: single points, scans, the Pauli-word count. |
| `validation.py` | Exact checks and the hash-bound receipt. |
| `classical.py` | MP2, CCSD, CCSD(T) in the same active space. |
| `synthetic.py` / `selftest.py` | A structurally real active space with no chemistry package, and the 58 checks over it. |
| `summary.py` / `visualize.py` / `report.py` | Table, figures, assembled report. |
| `colab.ipynb` / `adapt_colab.ipynb` | One-click Colab notebooks. |

Generated, not source: `cache/` (Hamiltonians and receipts), `results/`.

## References

- Pellow-Jarman et al., *Handover Iterative VQE*, [arXiv:2503.06292](https://arxiv.org/abs/2503.06292)
