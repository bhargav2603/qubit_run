# Li₂S at 24 qubits — HI-VQE on Qiskit

A complete, self-contained implementation of the **Handover Iterative Variational
Quantum Eigensolver** ([Pellow-Jarman et al., arXiv:2503.06292](https://arxiv.org/abs/2503.06292),
Qunova Computing) applied to that paper's own 24-qubit benchmark system:
**Li₂S, CAS(12e,12o)/STO-3G, 853,776 determinants**, along the Li–S bond
dissociation coordinate.

Everything runs on a CPU. A free Google Colab session is enough for the whole
study, including the quantum chemistry.

```bash
python run.py selftest                  # prove the stack — needs no chemistry package
python run.py prepare  --distance 2.10  # build the Hamiltonian (PySCF)
python run.py validate --distance 2.10  # prove it correct, write the receipt
python run.py hivqe    --distance 2.10  # run HI-VQE
python run.py report                    # Markdown + standalone HTML with figures
```

---

## What HI-VQE actually is, and why it is worth implementing

A conventional VQE asks the quantum device for an **energy**. That means
measuring every Pauli word in the Hamiltonian, every iteration. For this exact
system the paper counts **15,697 Pauli words** — 15,697 circuit executions per
optimizer step, and `python run.py paulis` recomputes that number from the cached
integrals so you can check it rather than quote it.

HI-VQE asks the device a different question: **which electron configurations
matter?** One measurement returns a set of occupation bitstrings. The
Hamiltonian is then projected onto the subspace those configurations span and
diagonalised *exactly, classically*. That is the handover.

Three consequences, and this folder measures all three rather than asserting them:

| | Why it follows | Where it is checked |
|---|---|---|
| **The energy is a strict upper bound** | `P H P` has its lowest eigenvalue above the true ground state for *any* projector `P`, so device noise can make the subspace worse but can never push the energy below the answer | `validate` on random subspaces; any violation is reported as `INCONSISTENT`, never as a result |
| **The energy is monotone** | the subspace only grows | `history` in every result file, and a re-solve guard in the loop |
| **The quantum layer is separable** | `--simulator none` removes it entirely; `--no-expansion` removes the classical half instead | the ablation table in every report |

That third row is the honest one. At 24 qubits the full CAS space is 853,776
determinants, which a laptop diagonalises directly — so this is a **benchmark and
validation exercise, not a claim of quantum advantage**, and the ablation is there
precisely so the quantum layer's contribution is a number you read rather than a
claim you accept.

---

## The files

| File | Purpose |
|---|---|
| `run.py` | The one interface. Every operation is `python run.py <command>`. |
| `molecule.py` | The system: Li₂S, the active space, the scan grid, the published reference. The only file specific to this molecule. |
| `hamiltonian.py` | Molecule → cached active-space integrals. RHF, CASCI, orbital diagnostics, cache I/O, fingerprints. **The only module that imports PySCF.** |
| `determinants.py` | The classical engine: strings, Slater–Condon rules, the projected Hamiltonian, Davidson, ⟨S²⟩, occupancies. Plain NumPy and SciPy. |
| `ansatz.py` | The Qiskit sampling circuit, Aer execution, noise models, configuration recovery. **The only module that imports Qiskit.** |
| `sector_sim.py` | An exact simulator that replays the same Qiskit circuit inside the particle-number sector — ~4× faster than Aer at 24 qubits, and cross-checked against it. |
| `hivqe.py` | The algorithm: sample, project, diagonalise, screen, expand, optimise, converge. |
| `driver.py` | CLI drivers: single points, the dissociation scan, the geometry scan, the Pauli-word count. |
| `validation.py` | Exact checks on a cached Hamiltonian, and the hash-bound validation receipt. |
| `classical.py` | MP2, CCSD and CCSD(T) frozen to the same active space, so the quantum error has a scale. |
| `synthetic.py` | A structurally real active space with no chemistry package, for the self-test. |
| `selftest.py` | 58 checks: layering, conventions, the engine against OpenFermion, the simulators against Aer, and a full HI-VQE run against an exactly solved system. |
| `summary.py` / `visualize.py` / `report.py` | The table, the figures, and the assembled report. |
| `colab.ipynb` | One-click Colab notebook: install, build, validate, run, scan, report, download. |
| `tests/` | 61 unit tests, ~2 s. |

Generated, not source: `cache/` (Hamiltonians and receipts), `results/` (runs,
figures, reports).

---

## Step 0 — Environment

```bash
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate
pip install -r requirements.txt
```

Verified on Python 3.12.13 · qiskit 2.5.2 · qiskit-aer 0.17.2 · numpy 2.5.2 ·
scipy 1.18.0 · openfermion 1.8.1 · matplotlib 3.11.1.

**PySCF publishes no Windows wheel.** That affects exactly three commands —
`prepare`, `geometry` and `classical` — and the workflow is built to survive it:
the expensive chemistry runs **once**, is cached as JSON, and everything after
that reads the cache.

| Step | Needs | Where it runs |
|---|---|---|
| `prepare`, `geometry`, `classical` | PySCF | Linux, macOS, WSL or Colab — **once** |
| `validate`, `summary`, `report` | NumPy + SciPy | anywhere, including Windows |
| `selftest`, `backend`, `hivqe`, `scan`, `plot` | + Qiskit | anywhere, including Windows |

The cache is plain JSON carrying its own specification hash, so a stale or
mismatched file is a hard error rather than a wrong answer. Copying it between
machines is safe by construction.

## Step 1 — Prove the stack before any chemistry

```bash
python run.py selftest
python run.py backend           # add --full to cross-check both simulators at 24 qubits
python -m unittest discover -s tests
```

Measured on this machine: **58 checks in ~8 s**, plus 61 unit tests in ~2 s.
Neither needs PySCF or a cache.

The self-test's job is to make the conventions falsifiable. Two of its checks
exist because they caught real bugs during development, and they say so:

- **Slater–Condon against OpenFermion.** The double-excitation matrix element had
  the wrong overall sign — the rule is stated for `a†_p a†_r a_s a_q` and the
  phase was being accumulated for the opposite creation order. Only an
  independent Jordan–Wigner implementation finds that: it is a global sign on one
  of five cases, and the spectrum of a small system still looks plausible.
- **The projection check.** An earlier contraction restricted the two-electron
  *intermediate* state to the subspace, computing `P E P E P` instead of
  `P (E E) P`. It produced subspace energies **below** the exact ground state — a
  variational impossibility. `P H P` is now compared against the exact sub-block
  of `H`, element by element.

A third bug survived until the algorithm stage: **Davidson converged cleanly to
the wrong eigenpair.** Started from the Hartree–Fock determinant on a system
whose sector ground state is a triplet, the Krylov space has exactly zero overlap
with the answer, so it returns a real eigenvector, a residual of 1e-10, and an
energy 40 mHa too high. The fix is in `davidson`: the starting basis always
includes the lowest-diagonal determinant and one deterministic pseudo-random
vector alongside the warm start, and the HI-VQE loop re-solves cold if an energy
ever goes *up*.

## Step 2 — Build the Hamiltonian, where PySCF works

```bash
python run.py prepare --diagnose            # look at the orbital window first
python run.py prepare --distance 2.10
python run.py validate --distance 2.10
```

### The active space, and why it is what it is

Li₂S has **22 electrons** and **19 STO-3G orbitals** (Li: 1s, 2s, 2p; S: adds 3s,
3p). CAS(12e,12o) therefore fixes everything else by arithmetic:

- 22 − 12 = 10 inactive electrons ⇒ **5 frozen orbitals**;
- 12 active orbitals from the remaining 14 ⇒ **2 highest virtuals dropped**;
- 12 spatial orbitals ⇒ **24 qubits** under Jordan–Wigner;
- C(12,6)² = **853,776 determinants**, matching the paper's Table 1 exactly.

Those 5 frozen orbitals *should* be the sulfur 1s, 2s and 2p shell, which sit near
−91, −9 and −6.7 Ha while nothing else in the molecule is below −3 Ha. `prepare`
**measures** that — the Mulliken population of each frozen orbital on sulfur, and
the core/active energy gap — and refuses to build if it is not true. Use
`--diagnose` to see the whole window before spending an SCF on a guess.

### The dissociation coordinate

The paper's Figure 6 dissociates Li₂S "by removing a lithium atom". Here that is
made explicit: linear Li–S–Li, one bond held at equilibrium, the other stretched.
The equilibrium bond length is **not** taken from experiment — it is the minimum
of this folder's own STO-3G CASCI symmetric stretch, which `python run.py
geometry` recomputes and prints. An STO-3G study should be measured against an
STO-3G minimum.

At long bond length the fragments are Li(²S) and LiS(²Π): two open shells, no
single determinant, and RHF fails by hundreds of millihartree. That is the whole
point of the benchmark. `prepare --continue-from` reuses the previous geometry's
converged orbitals as the SCF guess, because a fresh atomic guess on a stretched
bond routinely lands on a different SCF solution and puts a discontinuity in the
curve that no correlation treatment can repair.

### The validation receipt

`validate` writes a hash-bound receipt and `hivqe` refuses to run without one. It
is bound to **the physics only** — the cache, the molecule specification, and
`molecule.py` / `hamiltonian.py` / `determinants.py` / `validation.py`. Editing
`ansatz.py` or `visualize.py` does not invalidate it, which is the point on a
machine that cannot reinstall PySCF to earn a new one. The full-folder
`workflow_sha256` is still recorded in every result for provenance.

## Step 3 — Run it

```bash
python run.py hivqe --distance 2.10
python run.py scan                        # the whole dissociation curve
python run.py summary
```

### The ansatz

A **hardware-efficient unitary cluster Jastrow**, built as a real Qiskit circuit:

- nearest-neighbour Givens rotations (`XXPlusYY`) in a brickwork of depth *M*
  inside each spin block — depth *M* is what the Clements decomposition needs to
  realise an arbitrary orbital rotation, so one network can move an electron from
  any orbital to any other. A shallower network cannot, and the proposal
  distribution collapses onto Hartree–Fock and its immediate neighbours;
- a diagonal number–number Jastrow layer (`RZ`, `RZZ`), including the
  same-orbital α–β terms that are the only gates correlating the two spin blocks;
- **a closing Givens network, never a closing Jastrow layer.** A trailing
  diagonal layer changes only phases, so it is exactly invisible to a measurement
  distribution — it would cost gates and shots while doing nothing. The self-test
  checks that claim rather than trusting it.

Every gate commutes with the α and β number operators separately, so on a
noiseless simulator **every shot is a valid configuration**: no electron-count
filtering, no wasted shots, leakage exactly zero rather than merely small. At all-zero
angles the circuit is exactly the Hartree–Fock determinant.

### The two simulators

| | `--simulator sector` (default) | `--simulator aer` | `--simulator none` |
|---|---|---|---|
| What it is | exact statevector inside the 853,776-dimensional particle-number sector | Qiskit Aer over all 16,777,216 amplitudes | no quantum layer at all |
| Cost per sample, 24 qubits | **~2.8 s** | ~11 s | 0 |
| Noise models | no — noiseless by construction | **yes** | — |
| Role | the working default | the reference path, and the only one that can carry noise | the classical selected-CI control |

`sector_sim.py` is fast because the ansatz never leaves the sector, and it is
trustworthy because it **reimplements nothing**: it reads each gate's own
`to_matrix()` from the Qiskit circuit and applies that, so Qiskit remains the sole
authority on what `XXPlusYY(θ)` means. `run.py backend --full` cross-checks the
two amplitude by amplitude on the real 24-qubit circuit; the self-test does the
same on a small one.

### The knobs that matter

```bash
python run.py hivqe --ranking coupling           # the paper's selection rule
python run.py hivqe --spin-complete              # when a run says SPIN CONTAMINATED
python run.py hivqe --no-pt2                     # skip the perturbative correction
python run.py hivqe --max-determinants 40000     # bigger subspace, tighter energy
python run.py hivqe --expansion 800              # more candidates per iteration
python run.py hivqe --expansion-references 1     # exactly the paper's selection rule
python run.py hivqe --simulator aer --readout-error 0.02   # noise + recovery
python run.py hivqe --simulator none             # the classical control
python run.py hivqe --no-expansion               # quantum proposals only
```

`--max-determinants` is the accuracy/cost dial. `--expansion-references` is worth
understanding: the paper generates single and double excitations from the *one*
configuration with the largest amplitude and ranks them by `|⟨φ_ref|H|φ⟩|`. The
default here uses the leading 8 and ranks by the coherent sum
`|Σᵢ cᵢ ⟨φ|H|φᵢ⟩|`, which is the standard first-order estimate and strictly better
informed. **`--expansion-references 1 --ranking coupling` reproduces the paper's
rule exactly.**

### Selection, spin, and the perturbative correction

Three settings that change the *answer* rather than the cost. All three were
decided by measurement over 18 runs on six synthetic active spaces where the
exact energy is known, not by argument — those systems are far more strongly
correlated than Li₂S, so read the direction rather than the magnitude.

**`--ranking`** — how expansion candidates are scored.

| | Score | Median error over 18 runs |
|---|---|---|
| `coupling` | `|Σᵢ cᵢ⟨φ\|H\|φᵢ⟩|`, the paper's rule | 216.6 mHa |
| `pt2` *(default)* | that numerator squared over `E − ⟨φ\|H\|φ⟩` | **202.9 mHa** |

The denominator is what separates a large coupling to a high-lying determinant
from a modest coupling to a near-degenerate one — the distinction that decides a
stretched bond. It is the CIPSI criterion (Huron, Malrieu, Rancurel 1973) and the
acquisition function of Active-Sampling SQD ([arXiv:2603.13536](https://arxiv.org/abs/2603.13536)).
It won 13 of 18 runs variationally and 15 of 18 once corrected.

**`--spin-complete`** — force the α and β string sets to be the same set, so the
subspace is closed under exchanging the two blocks. A determinant `(Ia, Ib)` and
its partner `(Ib, Ia)` are degenerate for a spin-free Hamiltonian at Sz = 0, so a
space holding one but not the other cannot represent a spin eigenstate. **Off by
default**, because the measurement says it is a trade rather than a free win: it
cut the distance from ⟨S²⟩ to the nearest S(S+1) by about a third (median
3.3e-2 → 2.2e-2) but made the variational energy *worse* in 16 of 18 runs (median
216.6 → 227.0 mHa), since one shared set spends dimension the two blocks were
using separately. **Turn it on when a run comes back `SPIN CONTAMINATED`** — that
is the case it is for. It is a necessary condition, not full spin adaptation: the
space is closed under the flip, not projected onto an S² eigenvector.

**The Epstein–Nesbet correction** (on by default; `--no-pt2` to skip) is the
largest accuracy gain here, and the only one that is not a trade:

```
E_PT2 = Σ_{D ∉ S} |⟨D|H|Ψ⟩|² / (E − ⟨D|H|D⟩)
```

It improved **18 of 18** runs, median 216.6 → 87.5 mHa. It is nearly free because
both pieces already exist: embed the converged CI vector into a space that also
holds the candidates, apply the same `sigma` the eigensolver uses, and every
numerator falls out at once; the denominators are the diagonal Davidson already
builds as its preconditioner. `pt2_determinants` in each result records how many
determinants the sum actually ran over.

**It is reported beside the variational energy and never folded into it.** PT2 is
not variational, so the strict upper bound — the property this whole workflow is
built to preserve — applies to `energy`, not to `energy_pt2`. And perturbation
theory has a domain: from a nearly-Hartree–Fock subspace on a strongly correlated
system it can overshoot straight past the exact answer, which a 1-determinant
subspace on a test system duly did, returning −7.6 Ha against a variational error
of +4.1 Ha. When the correction exceeds half the correlation energy the subspace
already captured, the run says `PT2 IS NOT USABLE HERE` and the summary table
marks the number with `!`. Grow the subspace rather than trusting it.

### Noise, and configuration recovery

`--simulator aer --readout-error 0.02` flips measured bits, which breaks the
electron count on a large fraction of shots. `--no-recovery` discards those shots;
by default they are **repaired** the way SQD does it — when a string has too many
electrons, empty the orbitals the previous iteration's wavefunction says are least
likely to be occupied, and vice versa. The self-test verifies that noise really
does break the counts and that recovery really does restore all of them.

## Step 4 — Reading the result

| Printed verdict | Meaning |
|---|---|
| `CHEMICAL ACCURACY` | within 1.6 mHa of CASCI, above it, and a spin eigenstate |
| `OUTSIDE CHEMICAL ACCURACY` | working but subspace-limited. Raise `--max-determinants` or `--expansion` |
| `INCONSISTENT` | **below** CASCI. Impossible for a projected subspace. Re-run `validate` and `selftest`; do not report the number |
| `SPIN CONTAMINATED` | ⟨S²⟩ is not near any S(S+1), so the subspace is not spin-complete. Raise `--max-determinants`, or add `--spin-complete` |
| `PT2 IS NOT USABLE HERE` | not a verdict on the energy — the *correction* is larger than half the correlation the subspace captured, so perturbation theory is out of its domain. The variational energy above it is unaffected |

Note what `SPIN CONTAMINATED` is *not*. It does not fire on a triplet: at a
dissociated Li–S bond the singlet and triplet are nearly degenerate and CASCI
returns whichever is lower, so a triplet ground state is a result. What is a
broken calculation is a state that is neither — and the check measures the
distance from ⟨S²⟩ to the nearest S(S+1) rather than to zero.

### Figures and reports

```bash
python run.py plot                # PNGs into results/figures/
python run.py report              # report.md + a standalone report.html
```

`report.html` embeds its figures as data URIs, so the single file is the whole
report. In Colab, `import visualize; visualize.show()` renders the same figures
inline.

| Figure | Question it answers |
|---|---|
| Dissociation curve | Does HI-VQE track CASCI *everywhere*, not just near the minimum? Hartree–Fock's failure at long range is drawn on the same axes. |
| Convergence | Did it converge, and how many determinants did each iteration hold? |
| Energy | Where the energy actually went, in hartree, between the Hartree–Fock and CASCI lines — the correlation recovered as a distance rather than a number. One geometry only. |
| Compression | How much accuracy did each determinant buy, against the full 853,776? |
| Ablation | What did the quantum sampler contribute, against the classical-only control? |
| Occupancies | Where the electrons sit, and how far from 2 and 0 — the multireference character, made visible. |
| Cost | 15,697 measurement settings per iteration versus 1, drawn to scale, beside the circuit's own gate counts. |

The report also computes the **dissociation energy** from the curve and its error
against CASCI. That is the number to quote: an energy *difference* cancels most of
the systematic error that an absolute total energy carries, so it is a much
stricter test of consistency.

---

## Performance

Measured on this machine (Windows 11, Python 3.12, 12 orbitals / 24 qubits,
853,776-determinant full space):

| Operation | Cost |
|---|---|
| One 24-qubit circuit sample, `--simulator sector` | ~2.8 s |
| One 24-qubit circuit sample, `--simulator aer` | ~11 s |
| Build the subspace Hamiltonian, ~20,000 determinants | ~1 s |
| One Davidson matrix–vector product at that size | ~0.06 s |
| One diagonalisation, warm-started | ~2 s |
| A 10-iteration, 30,000-determinant classical run | ~48 s |
| Full self-test | ~8 s |

### How long a whole study takes, and how to shorten it

Measured at the real 24-qubit size (853,776-determinant space, a 20,000-determinant
subspace, three iterations each):

| Setting | s / iteration | Energy |
|---|---|---|
| default (SPSA every iteration) | 16.0 | -392.374183 |
| `--optimizer-every 3` | 11.1 | -392.374183 |
| `--optimizer none` | **6.2** | -392.374183 |
| `--optimizer none --givens-depth 6` | 4.7 | -392.402745 |
| `--simulator none` (classical control, no sampling at all) | 3.7 | -- |
| `--max-determinants 6000` (with `--optimizer none`) | 6.4 | -392.204802 |

Read the last two rows together, because they are the counter-intuitive part.
**Shrinking the subspace is not a speed lever.** Going from 20,000 determinants
to 6,000 saved nothing measurable and cost 170 mHa: the classical half of an
iteration is dominated by fixed work -- building the same-spin blocks, ranking
expansion candidates -- and not by the dimension. Lower `--max-determinants` when
you are short of memory, never to save time.

What costs time is **sampling**, at roughly 2.5 s per circuit evaluation, and
SPSA asks for three of those per iteration instead of one.

### The state is built once per θ, not once per iteration

Forming the 853,776-amplitude sector state is the single largest cost in a run,
and θ does not move at all under `--optimizer none` — so the same state was being
rebuilt from scratch on every iteration. It is now cached on the angle vector.
Measured on the real 24-qubit circuit (298 two-qubit gates, 322 parameters):

| | Time |
|---|---|
| First sample, builds the state | **3.60 s** |
| Second sample, same θ | **0.036 s** |
| A 12-iteration run, `--optimizer none` | 43.2 s → **4.0 s** of sampling |

That is ~39 s per geometry, or roughly **8 minutes off a 13-point scan**. SPSA
gains little from it by construction — each step moves θ somewhere new — which is
one more reason `--optimizer none` is the right default for this benchmark.

`circuit_runs` in every result counts evaluations that actually ran, so a cache
hit shows up as a zero rather than being quietly counted as a measurement.

Two things that looked like optimizations and were **measured and rejected**:
rebuilding `single_excitation_maps` as vectorised NumPy, and generating the
same-spin connections instead of scanning all string pairs. Both are faster only
above ~250 strings per block; the default subspace is 141 × 141, where they ran
1.3–1.4× *slower*. The determinant count is 20,000 but the string count is its
square root, and that is the number those loops actually see. The chunked `sigma`
path did keep its change — it was rebuilding 144 sliced sparse matrices inside
every matrix–vector product, and hoisting them out is 1.68× on that path (it is
the path taken at large subspaces or on a low-memory machine; the default
unchunked path was already clean, and measures 1.02×).

| Want | Use | Effect |
|---|---|---|
| The same answer, much faster | `--optimizer none` | ~2.6x. On this benchmark the optimizer moved the energy by nothing at all |
| To keep the optimizer but pay less | `--optimizer-every 3` | ~1.4x, with the SPSA schedule spread over more iterations rather than truncated |
| Faster still, slightly worse proposals | `--givens-depth 6` | a further ~1.3x; halves the circuit, at the price of proposals that cannot reach distant orbitals |
| A whole scan | `--workers 2` | geometries are independent, so they parallelise cleanly: ~1.5-2x on two cores |

Putting the safe ones together:

```bash
python run.py scan --optimizer none --workers 2
```

| Job | Default | With the above |
|---|---|---|
| One HI-VQE run, 12 iterations | ~3.2 min | ~1.2 min |
| A single validated point, start to finish | ~10 min | ~6 min |
| The full 13-geometry scan | 50-90 min | **20-30 min** |
| `selftest` + unit tests | ~11 s | -- |

`prepare` is the one step never timed on this machine -- PySCF has no Windows
wheel -- so its CASCI cost enters the end-to-end figures as an estimate.

Two changes inside the code took about 10-15% off everything above, and both came
from profiling rather than from guessing. The opposite-spin excitation maps are
now transposed once per operator instead of being re-sliced on every
matrix-vector product; SciPy was rebuilding 240,000 sparse matrices in a
three-iteration run. And Davidson's residual tolerance was loosened from 1e-9 to
1e-7, which is still far inside chemical accuracy because the energy error is
second order in the residual.

The cost driver is neither the qubit count nor, at these sizes, the subspace
dimension: it is the number of circuit evaluations. Aer's 24-qubit statevector is
268 MB; the sector representation of the same state is 13 MB, and it is the
*only* part of the Hilbert space the ansatz can ever reach — which is why
sampling it costs 2.5 s instead of 11 s. The determinant side is the cheap half,
and that is the structural point of the method.

Two design decisions carry most of the speed:

**The Hamiltonian is split by spin.** `H = E_core + H_α ⊗ 1 + 1 ⊗ H_β + V_αβ`.
The same-spin blocks act on one string block each, so they are built explicitly
once per subspace and reused by every Davidson iteration. The opposite-spin term
factorises, so it needs no intermediate state outside the subspace. Because a
tensor-product projector factorises as `P = P_a ⊗ P_b`, every piece projects
independently and `P H P` comes out exact — the earlier projection bug cannot
recur in this formulation.

**Davidson, warm-started.** The subspace grows monotonically, so the previous
iteration's ground state — embedded into the enlarged space — is already an
excellent approximation and converges in a handful of matrix–vector products.
ARPACK accepts a starting vector too but still builds a fresh Krylov space, which
at these dimensions dominates the whole run.

---

## Scientific boundary — what is comparable with the paper, and what is not

| | arXiv:2503.06292 | Here |
|---|---|---|
| Active space | CAS(12e,12o)/STO-3G, 24 qubits, 853,776 determinants | **identical** |
| Reference method | CASCI | **identical** |
| Algorithm | sample → project → diagonalise → screen → expand → optimise | **identical** |
| Dissociation coordinate | "removing a lithium atom" | identical in kind |
| Pauli-word count for a conventional VQE | 15,697 | recomputed by `run.py paulis` |
| Geometry | **not published** | linear Li–S–Li, this folder's own CASCI equilibrium |
| Per-point energies | **not published** | reported in full |
| Circuit / ansatz | **not specified** | hardware-efficient unitary cluster Jastrow, fully described above |
| Optimizer, shot counts, convergence thresholds | **not specified** | SPSA, all thresholds exposed as flags |
| Hardware | NISQ framing; results are simulated | CPU simulation only |

**Total energies are therefore not reproducible from that paper, and this folder
does not claim to reproduce them.** The paper publishes no geometry, no per-point
energies, no orbital specification beyond the active-space size, and no circuit.

What *is* comparable, and what this workflow actually measures:

1. **Error against your own CASCI at every geometry.** The paper's claim is that
   HI-VQE stays inside chemical accuracy across the whole dissociation curve while
   Hartree–Fock is off by hundreds of millihartree. Both halves of that are
   directly testable and are what `scan` reports.
2. **The fraction of the determinant space needed.** The paper's compression claim
   for its other systems is three orders of magnitude fewer states than the full
   space, and two fewer than SHCI. `subspace_fraction` is in every result file.
3. **The shape of the curve**, and the dissociation energy derived from it.

One more boundary worth stating plainly: at 24 qubits the exact answer is
classically cheap. CAS(12e,12o) is 853,776 determinants, and the reference CASCI
in this folder is computed directly. **Nothing here is evidence of quantum
advantage, and the ablation table is included so that the quantum layer's actual
contribution — a better starting subspace — is visible rather than implied.** The
value of the exercise is a validated, reproducible pipeline at a size where the
answer is known, which is the only regime in which a method can be shown to be
right.

---

## References

The method is one paper's; almost every component of a working implementation is
someone else's. Attribution, and where each piece shows up in this folder:

| Work | What this folder took from it |
|---|---|
| Pellow-Jarman, McFarthing, Kang, Yoo, Elala, Pellow-Jarman, Nakliang, Kim, Rhee, *HIVQE: Handover Iterative Variational Quantum Eigensolver for Efficient Quantum Chemistry Calculations*, [arXiv:2503.06292](https://arxiv.org/abs/2503.06292) | **The method**, and the Li₂S CAS(12e,12o)/24-qubit benchmark, the 853,776-determinant space, the 15,697-Pauli-word cost comparison, and the tensor-product treatment of α and β configurations. `hivqe.py` follows its Figure 3 step for step. |
| Robledo-Moreno *et al.*, *Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer*, [arXiv:2405.05068](https://arxiv.org/abs/2405.05068); *Sci. Adv.* **11**, eadu9991 (2025) | **Configuration recovery** — repairing a sample whose electron count noise has broken, using the previous iteration's orbital occupancies as the prior. `ansatz.py::_recover_string`. The subspace-projection-and-diagonalise structure of SQD is HI-VQE's nearest relative. |
| Motta *et al.*, *Bridging physical intuition and hardware efficiency for correlated electronic states: the local unitary cluster Jastrow ansatz* (2023) | **The ansatz family.** `ansatz.py` is a hardware-efficient unitary cluster Jastrow: orbital-rotation-like Givens networks alternating with diagonal number–number layers. |
| Clements, Humphreys, Metcalf, Kolthammer, Walmsley, *Optimal design for universal multiport interferometers*, *Optica* **3**, 1460 (2016) | **Why the Givens brickwork is depth *M*.** That is what a nearest-neighbour network needs to realise an arbitrary orbital rotation — and why a shallower one collapses the proposal distribution. |
| Huron, Malrieu, Rancurel, *J. Chem. Phys.* **58**, 5745 (1973) — CIPSI | **The selection rule.** Ranking candidate configurations by their first-order coupling to the current wavefunction. `--expansion-references 1` reduces it to the rule the HI-VQE paper states. |
| Kanno *et al.*, *Active Sampling Sample-based Quantum Diagonalization from Finite-Shot Measurements*, [arXiv:2603.13536](https://arxiv.org/abs/2603.13536) | **The Epstein–Nesbet acquisition function**: ranking candidate basis states by their expected second-order energy improvement rather than by their bare coupling. `--ranking pt2`, which is the default. |
| Holmes, Tubman, Umrigar, *J. Chem. Theory Comput.* **12**, 3674 (2016) — heat-bath CI / SHCI | The selected-CI lineage the HI-VQE paper benchmarks its compression against. |
| Knowles, Handy, *Chem. Phys. Lett.* **111**, 315 (1984) | **The string-driven CI contraction**, and the α/β decomposition `H = E_core + H_α ⊗ 1 + 1 ⊗ H_β + V_αβ` that `determinants.py` is built on. |
| Davidson, *J. Comput. Phys.* **17**, 87 (1975) | **The eigensolver**, and its diagonal preconditioner. |
| Lee, Taylor, *Int. J. Quantum Chem.* **36**, 199 (1989) | **The T1 diagnostic** reported by `classical.py`, and the threshold above which CCSD(T) stops being a gold standard. |
| Spall, *IEEE Trans. Autom. Control* **37**, 332 (1992) | **SPSA**, the two-evaluation optimizer used on the circuit parameters. |
| McClean, Rubin, Sung *et al.*, *OpenFermion: the electronic structure package for quantum computers*, *Quantum Sci. Technol.* **5**, 034014 (2020) | The **independent** Jordan–Wigner implementation the self-test checks our determinant conventions against. It shares no code with this folder, which is the entire point. |
| Sun *et al.*, *PySCF*, *WIREs Comput. Mol. Sci.* **8**, e1340 (2018); *J. Chem. Phys.* **153**, 024109 (2020) | RHF, CASCI, MP2, CCSD(T) and the active-space integrals, used by `prepare` and `classical` only. |
