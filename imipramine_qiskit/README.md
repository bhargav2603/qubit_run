# Imipramine CASCI(6e,6o), 12 qubits — local Qiskit Aer

The same 12-qubit study as [`../imipramine_cas6e6o/`](../imipramine_cas6e6o/),
rebuilt to run on a local machine with Qiskit and Qiskit Aer instead of Fujitsu
QARP, MPI and Qulacs on an A64FX cluster.

Imipramine is a tricyclic antidepressant and the closest thing at 12 qubits to a
drug molecule with published quantum work at the same active space. RHF/6-31G
canonical orbitals, CASCI(6e,6o), following the Hamiltonian setup reported in
[Koziell-Pipe et al. (2026)](https://arxiv.org/abs/2607.22468). It does **not**
claim to reproduce their ADAPT-GQE circuits or their private conformer dataset.

The chemistry is deliberately unchanged. `molecule.py` and `validation.py` are
**byte-identical** to the QARP folder's and `hamiltonian.py` differs only in its
docstring and two comments — `tests/test_core.py` fails if that ever stops being
true — so a disagreement between the two backends is a backend result, not a
different molecule.

The folder name carries the active space because imipramine appears three times
in that paper — CAS(6e,6o), CAS(6e,7o) and CAS(8e,8o) at 12, 14 and 16 qubits.
Those are separate calculations and get separate folders.

| File | Purpose |
|---|---|
| `run.py` | The one command interface. Every operation is `python run.py <command>`. |
| `molecule.py` | The molecule: geometry, basis, active space, π-core diagnostics. The only file specific to imipramine. |
| `hamiltonian.py` | Molecule → cached qubit Hamiltonian. CASCI, Jordan–Wigner, symmetry operators, Hartree–Fock frame, cache I/O. Backend-independent. |
| `validation.py` | Exact classical checks on the cached Hamiltonian, and the hash-bound validation receipt. |
| `qiskit_runtime.py` | **New.** OpenFermion→Qiskit conversion, Aer MPS evaluation, parameter-shift VQE. Replaces `qarp_runtime.py`. |
| `adapt_runtime.py` | **New.** ADAPT-VQE — the published reference method. Operator pools, sparse adjoint-gradient optimization, Qiskit circuit emission and verification. |
| `selftest.py` | **New.** Structure checks plus a full local stack proof against exact answers, with no PySCF and no cache. |
| `summary.py` | The results table, covering both methods. |
| `visualize.py` | **New.** The charts — convergence, energy, stopping criterion, operator choice, and a cross-run leaderboard. `run.py plot` saves PNGs; `visualize.show()` renders inline in Colab. |
| `colab.ipynb` | **New.** One-click Colab notebook: build the Hamiltonian, validate it, run ADAPT, download the artifacts. |
| `requirements.txt` | Pinned local environment. |
| `tests/` | Specification, fingerprint, receipt, artifact-safety, operator-conversion and ADAPT-pool tests. |

Generated, not source: `hamiltonian_cas6e6o.json` (the cache),
`hamiltonian_cas6e6o.validated.json` (the receipt), `results/` (VQE output).

### Two ansätze, and which one to use

The 1.6 mHa test is algorithmic accuracy against CASCI for exactly this cached
Hamiltonian. It is not experimental or basis-set-limit accuracy.

| | `run.py vqe` | `run.py adapt` |
|---|---|---|
| Ansatz | `real_amplitudes`, fixed depth | ADAPT-VQE, grown by gradient |
| Symmetry | N broken, penalized, priced | N exact by construction; ⟨S²⟩ measured and priced |
| Gradients | parameter-shift, O(k²) | adjoint, O(k) |
| Depth | a knob you scan | an output |
| Engine | Qiskit Aer circuits | SciPy sparse algebra, circuit verified at the end |
| Matches the paper | no | **yes** |

**`adapt` is the one to run for the benchmark.** The paper reaches 1.15 mHa with
statevector ADAPT-GQE circuits and trained its 12-qubit datasets to only 5–10 mHa,
so 1.6 mHa from a *shallow hardware-efficient* ansatz is a deliberately harder
target than anything published here — `vqe` exists to measure that gap, not to
beat it. See [ADAPT-VQE](#step-4--adapt-vqe-the-published-method) below.

---

## The one thing to know before you start

**PySCF publishes no Windows wheel.** There are macOS and Linux wheels only; the
source build wants `nmake` and a C/C++ toolchain, and the project does not
support Windows even when one is present.

That affects exactly one command, `prepare`, and the workflow was already built
to survive it: the expensive 45-atom RHF/CASCI work is done **once** and cached
as JSON, and everything after that reads the cache.

| Step | Needs | Where it runs |
|---|---|---|
| `prepare` | PySCF | WSL, Linux, macOS or Google Colab — **once** |
| `validate` | OpenFermion + SciPy | Anywhere, including Windows |
| `selftest`, `backend`, `vqe`, `adapt`, `summary` | Qiskit + Aer | **Windows, natively** |

If you have no Linux box, [`colab.ipynb`](colab.ipynb) does the whole thing —
`prepare`, `validate` and `adapt` — in one free Colab session and hands back the
cache, the receipt and the result.

The cache is plain JSON carrying its own specification hash, so a stale or
mismatched file is a hard error rather than a wrong answer. Copying it between
machines is safe by construction.

## Step 0 — Environment (Windows, once)

Python 3.12 and the pinned stack:

```powershell
cd imipramine_qiskit
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Verified working on Windows 11:

```
Python 3.12.13 · qiskit 2.5.2 · qiskit-aer 0.17.2 · openfermion 1.8.1 · numpy 2.5.2 · scipy 1.18.0
```

Python 3.13 is avoided deliberately — not every wheel in this stack ships for it
yet, and the ones that do are newer than the ones that are known-good here. The
repository's `..\.venv-qiskit` already satisfies `requirements.txt`.

## Step 1 — Prove the local stack works, before any chemistry

```powershell
python run.py selftest
python run.py backend
python -m unittest discover -s tests -v
```

`selftest` needs no PySCF and no cache, and runs in three stages. Measured on
this machine: **67 checks, all passing, in ~8 minutes**; the unit tests are 38
tests in ~9 s.

**Structure** (instant, NumPy only — `--structure-only` stops here, which is what
to run on the Linux box that only builds the Hamiltonian). It re-derives the
aromatic core from the stored geometry rather than trusting it: aromatic CH
carbons carry exactly one hydrogen and ring-junction carbons none, while every
sp³ carbon in the ethano bridge, the propyl tail and the N-methyls carries two or
three, so the twelve π carbons and the azepine nitrogen fall out of the geometry
alone. It also checks the module layering both ways, the fingerprints, the
artifact names and the accuracy-scoring arithmetic.

**Simulation**. It generates a random but *structurally real* CAS(6e,6o)
Hamiltonian — through the identical OpenFermion path the real one uses, so it is
Hermitian, particle-number conserving and has a genuine six-electron ground
state — and checks every local component against exact classical answers, ending
in a full VQE checked against exact diagonalization in the six-electron sector.

That VQE runs on `statevector` rather than MPS, which costs nothing in coverage
and four-fifths of the runtime: MPS and statevector are cross-checked against
each other twice anyway, once at random angles and once at the converged state.

**ADAPT**, on the same synthetic Hamiltonian. ADAPT makes a *structural* claim
the hardware-efficient path only makes statistically — that the state can never
leave the six-electron sector — and that claim is what removes the penalty term
from its accuracy story. A structural claim is worth nothing unless the structure
is checked, so this stage builds all three pools and verifies, **from the
commutators rather than from a docstring**, exactly which symmetries hold: that
[N, A] = 0 for every operator, that [S², A] = 0 for every single, and that S²
breaking is confined to the general doubles. It then checks that no pool repeats
an operator (on the operators themselves, not on the key that deduplicated
them), that the adjoint gradient equals finite differences, that the *selection*
gradient does too, that the ADAPT optimum is a singlet within the contamination
budget, and that the Qiskit circuit reproduces the sparse algebra that chose it.
`--skip-adapt` omits it; nothing else depends on it.

That S² check is not decoration. An earlier version of this folder asserted the
pool conserved S² by construction; this stage is what proved it wrong, and the
claim was corrected rather than the check relaxed.

The synthetic Hamiltonian is far more strongly correlated than any molecule, so
a short ADAPT run on it is deliberately *not* expected to reach chemical
accuracy. The checks are on monotonicity, symmetry and synthesis, which are the
properties that must hold whatever the Hamiltonian is.

The check that matters most is **qubit ordering**, and it is worth being precise
about what the risk actually is. `to_sparse_pauli_op` maps OpenFermion's qubit
*k* to Qiskit's qubit *k*; what the two libraries disagree about is how a basis
state is serialized into a vector index — OpenFermion puts qubit 0 in the most
significant bit, Qiskit in the least. So the two dense matrices of the same
correct operator differ by reversing the bits of both indices, and a genuinely
reversed register would produce a matrix with an *identical spectrum* and wrong
energies for every determinant. Comparing spectra therefore proves nothing here.
The self-test instead undoes that permutation and demands exact equality of
every matrix element (measured: 3.6e-14 Ha over 421,583 nonzeros), then checks
determinants a second time through the whole Aer path — frame rotation,
transpilation, layout relabelling, `save_expectation_value` — because none of
those steps is exercised by comparing two matrices.

## Step 2 — Build the Hamiltonian once, where PySCF works

Copy this folder to WSL/Linux/macOS, or upload it to Colab, then:

```bash
pip install pyscf openfermion numpy scipy
python run.py prepare --diagnose     # look before you build
python run.py prepare
python run.py validate
```

Do not continue unless validation prints `ALL CHECKS PASSED`. Copy **two** files
back into this folder on Windows:

```
hamiltonian_cas6e6o.json
hamiltonian_cas6e6o.validated.json
```

`prepare` prints the paper's published numbers beside its own, and so does
`vqe`. Do **not** expect the total energies to agree: this folder uses one
PubChem conformer and the paper used unpublished MD/NEB snapshots, and tens of
mHa of conformational difference is real physics, not error. The comparable
quantity is the active-space correlation energy `E_CASCI − E_HF`, which is
conformer-insensitive and is what shows whether the same six frontier orbitals
were selected.

### The active-space guard

CAS(6e,6o) on canonical orbitals takes HOMO−2…LUMO+2. For imipramine those
*should* be the π system of the dibenzazepine ring pair. They could instead be
the side-chain dimethylamino lone pair, the sp³ ethano bridge, or the propyl
tail — all irrelevant to what makes imipramine a tricyclic antidepressant, and
all of which would still converge cleanly and still pass every mapping check.

`prepare` therefore measures the Mulliken population of each active orbital on
the twelve aromatic carbons plus the bridging azepine nitrogen, and **refuses to
build** if one falls short. Occupied and virtual orbitals are held to different
bars — **0.80** and **0.65** — and that difference is deliberate, not a
convenience: Mulliken populations of virtual orbitals carry long basis-set tails
and individual atomic contributions can even be negative, so a threshold tight
enough to be meaningful on an occupied π orbital will reject a perfectly good π*.

**Look at the window before building.** `prepare --diagnose` runs RHF once and
prints every orbital either side of the gap with its energy, its population on
the π core, and whether the requested active space would take it:

```
  MO  label    occ/virt   energy (Ha)   pop on target atoms   verdict  in window
   73  HOMO-2   occupied     -0.351042                 0.913   ok       <==
```

One run tells you where the π system actually is. Without it, a misplaced window
costs one failed 45-atom RHF per guess.

If the guard fires, change the active space — shift the window, or add
polarization with `basis="6-31g*"` — not the threshold.
`--allow-delocalized-active-space` overrides it, but a result obtained that way
is not describing the pharmacophore and must not be reported as if it were.

### The validation receipt

Validation writes a hash-bound receipt, and `vqe` refuses to run without one.
Two details matter in practice:

**It is bound to the physics only** — the cache, the molecule specification, and
`molecule.py` / `hamiltonian.py` / `validation.py`. Editing `qiskit_runtime.py`
does **not** invalidate it, and that is the point on Windows: PySCF cannot be
installed there, so a receipt that the simulator layer could revoke would be a
receipt you could not get back. A patch to the Aer layer cannot change a
Hamiltonian that was built and proved elsewhere. The full-folder
`workflow_sha256` is still recorded in every result for provenance; it is simply
not an execution gate. `tests/test_core.py` asserts both halves of that.

**It holds several penalty settings at once.** `validate` checks `0`, `1` and `4`
by default and keeps them all, because you find out which one you need later —
on the machine that has no PySCF:

```bash
python run.py validate --number-penalty 0 1 4 16
```

Validation prints, per penalty setting, whether it selects the six-electron
sector and by what energy margin. If the 1 Ha penalty is rejected, run the VQE
with a larger one that was accepted:

```powershell
python run.py vqe --number-penalty 4
```

`prepare` also drops Pauli terms below `1e-6` Ha. Runtime is proportional to
term count, so this is a direct speedup; validation refuses the cache if the
discarded terms sum to more than 0.1 mHa, so the saving is audited rather than
assumed. `--cutoff 1e-12` disables it.

## Step 3 — Run the VQE on Windows

```powershell
python run.py vqe
```

Useful options:

```powershell
python run.py vqe --layers 8                  # deeper ansatz
python run.py vqe --method statevector        # faster at 12 qubits
python run.py vqe --max-bond-dimension 16     # cap MPS memory
python run.py vqe --start-noise 0             # start at exactly theta=0
python run.py vqe --optimizer cobyla          # gradient-free
```

Start at four layers. If the energy is below Hartree–Fock but outside 1.6 mHa,
the ansatz is the limit — add depth. Each run writes its own
`results/vqe_L<layers>_<method>.json`, so a scan accumulates instead of
overwriting itself:

```powershell
foreach ($n in 4,6,8,10,12) { python run.py vqe --layers $n }
python run.py summary --sort error
```

The real cost driver is `parameters × optimizer iterations × Pauli terms`. One
L-BFGS-B iteration costs 1 + 2N circuits at N = 12·(layers + 1) parameters — 121
circuits at four layers, 313 at twelve — and they all go to Aer in a single
batched `run` call.

## Step 4 — ADAPT-VQE, the published method

```powershell
python run.py adapt                      # uccgsd pool, the paper's choice
python run.py adapt --pool uccsd         # 7x smaller, usually enough at 12 qubits
python run.py adapt --gradient-tolerance 1e-4 --max-operators 60
python run.py summary
```

This is the method the published work actually uses. Instead of fixing a circuit
and optimizing its angles, ADAPT **grows** the ansatz: at each step it computes
dE/dθ at θ = 0 for every operator in a pool, appends the one with the largest
gradient, re-optimizes *every* parameter, and stops when no operator is worth
adding. Depth is an output, not a knob.

Three things follow, and each one removes a caveat rather than adding one.

**Particle number is structural, not penalized.** Every pool operator is a
fermionic excitation with balanced creation and annihilation indices, so it
commutes with N and S<sub>z</sub> exactly. The state cannot leave the
six-electron sector, which is why `adapt` runs at `--number-penalty 0`, why its
leakage is exactly zero rather than merely small, and why the whole penalty /
sector-selection apparatus the hardware-efficient path needs simply does not
apply. It needs the *unconstrained* receipt entry, which `validate` always
writes:

```bash
python run.py validate --number-penalty 0 1 4
```

**S² is a weaker claim, and is deliberately not overstated.** Spin-complementing
each excitation with its partner makes the *singles* proper singlet operators,
and the `pair` pool is seniority-zero, so both commute with S² exactly. A general
spin-complemented **double does not** — complementing buys invariance under
flipping S<sub>z</sub>, which is strictly weaker than commuting with S². Measured
on the 12-qubit pools:

| Pool | Singles breaking S² | Doubles breaking S² |
|---|---|---|
| `pair` | — | 0 / 15 |
| `uccsd` | 0 / 9 | 27 / 54 |
| `uccgsd` | 0 / 15 | 315 / 420 |

In practice ⟨S²⟩ comes back at ~1e-19 anyway: the Hamiltonian is spin-free and
the reference is a closed-shell singlet, so the variational minimum *is* the
singlet ground state. But that is a result of the optimization, not a property of
the ansatz, so ⟨S²⟩ is measured at the optimum and priced into the same 0.16 mHa
contamination budget the hardware-efficient path uses. If a run ever comes back
`SPIN CONTAMINATED`, it did not converge — lower `--gradient-tolerance`, raise
`--max-operators`, or use `--pool pair`. The self-test verifies all of this from
the commutators directly.

**Gradients are adjoint, not parameter-shift.** Parameter shift needs 2k+1 full
evaluations for k parameters, and each of those is itself k operator
exponentials — O(k²). The reverse-mode sweep here reuses every intermediate
state and costs 2k exponentials total. The self-test checks it against finite
differences.

**The optimization is sparse algebra, not circuit simulation.** At 12 qubits the
state is 4096 amplitudes, so exp(θA)|ψ⟩ is one `expm_multiply` — about a
millisecond. The Qiskit circuit is built *at the end*, transpiled for a gate
count, and its energy checked against the algebra. Every Pauli string inside one
excitation generator commutes with the others, so `PauliEvolutionGate` synthesis
is exact rather than a Trotter step — measured, not assumed. If the two disagree
by more than 1e-9 Ha the run is reported as `CIRCUIT DISAGREES`: the energy is
still right, the circuit is not.

| Pool | Operators | Singles / doubles | What it is |
|---|---|---|---|
| `pair` | 15 | 0 / 15 | k-UpCCGSD paired doubles; cheapest |
| `uccsd` | 63 | 9 / 54 | occupied → virtual only, relative to HF |
| `uccgsd` | 435 | 15 / 420 | **generalized** — every excitation between orbital pairs. The paper's choice at 12 qubits. |

The published circuits used a median of 8–19 operators at twelve qubits, so the
`--max-operators 40` default is slack, not a target; ADAPT normally stops on the
gradient threshold well before it. If it stops on the cap instead, the run says
so in `stop_reason` and the final pool gradient is recorded either way.

## Seeing the result

```powershell
python run.py plot                    # PNGs into results/
python run.py plot --output figures   # somewhere else
```

In Colab, `import visualize; visualize.show()` renders the same charts inline
along with a stat row. Four panels per ADAPT run, each answering one question:

| Panel | Question | Why it earns the space |
|---|---|---|
| Convergence to chemical accuracy | Did it converge, and to what? | Error vs CASCI on a log axis with the 1.6 mHa pass mark and the published 1.15 mHa drawn in. The chart the workflow exists to produce. |
| Where the energy went | What total energy came out? | Absolute hartree between the HF and CASCI lines, so recovered correlation is a visible distance rather than a number to be trusted. |
| Why it stopped | Is this a result or an unfinished run? | Largest remaining pool gradient against the stopping threshold. A capped run and a converged one look nothing alike — and that matters more than the final energy. |
| What ADAPT chose | Which excitations carry the correlation? | Selected operators by rotation angle, split into singles and doubles. |

Plus one leaderboard across every run in `results/`, so pools and ansätze are
compared on the axis that matters.

`matplotlib` is the only dependency, it is imported lazily, and nothing in the
physics or simulator path touches it — a machine without it loses the pictures
and nothing else.

## Reading the result

| Printed verdict | Meaning |
|---|---|
| `VERIFIED` | Within 1.6 mHa of CASCI, with symmetry breaking and the method's own cross-check each inside budget, and above the exact sector ground state. |
| `INCONSISTENT` | The energy is below the exact six-electron ground state by more than its own measured leak — impossible for a state in that sector. Re-run `validate` and `selftest`; do not report the number. |
| `MPS TOO COARSE` | (`vqe`) The approximate and exact energies disagree. Raise `--max-bond-dimension` or lower `--truncation-threshold`. |
| `CIRCUIT DISAGREES` | (`adapt`) The emitted circuit does not reproduce the optimized energy, so the exponentials did not synthesize exactly. The energy is still correct; the circuit is not. |
| `ABOVE HF` | Optimizer failure, not an ansatz limit. **Do not add layers** — check the live-parameter line and `--start-noise`. |
| `SYMMETRY BROKEN` | (`vqe`) The state left the six-electron or singlet sector. Raise `--number-penalty` to a validated value. For `adapt`, particle-number leakage here is impossible and means the pool, the reference determinant or the cache is wrong. |
| `SPIN CONTAMINATED` | (`adapt`) ⟨N⟩ is exact but ⟨S²⟩ is not zero, so the run did not reach the variational minimum. Lower `--gradient-tolerance`, raise `--max-operators`, or use `--pool pair`. |
| `outside` | Working, ansatz-limited. Add layers (`vqe`) or lower `--gradient-tolerance` (`adapt`). |

### Why the accuracy test is what it is

Subtracting the penalty recovers ⟨H⟩ exactly by linearity — but ⟨H⟩ is only a
variational bound *within* the six-electron sector, so a state that has leaked
out of it can sit below CASCI while meaning nothing. A hardware-efficient ansatz
is not particle-number conserving and always leaks a little, so demanding zero
leakage would fail every physically correct run.

The test is therefore energetic: each broken symmetry is priced at 1 Ha per unit
and the total must stay under **0.16 mHa**, one tenth of chemical accuracy. That
is small enough that the contamination cannot be what produced the reported
error. `contamination_energy_hartree` is printed and stored beside the energy.

The S² penalty is **off** by default: it adds a few hundred Pauli terms to every
energy evaluation to guard a secondary risk, given a spin-free Hamiltonian, a
real-amplitude ansatz and a closed-shell start. ⟨S²⟩ is still measured at the
optimum and still priced into the budget — at the full 1 Ha rate — so switching
the penalty off cannot buy a weaker claim. Re-enable with `--spin-penalty 1` if
⟨S²⟩ comes back bad, and validate with the same setting first.

---

## What changed from the QARP version, and why

**θ = 0 is guaranteed to be Hartree–Fock, not probed for.** The Hamiltonian is
conjugated by X on the occupied spin orbitals, and `real_amplitudes` at θ = 0 is
the identity, so the optimizer provably starts at E_HF with both penalties at
exactly zero. QARP ships as pre-compiled bytecode with no public API, so that
folder needs a runtime probe and a fail-closed preflight to get the same
guarantee; here it is a property of the circuit. `vqe` still checks it
numerically and aborts if it does not hold, because that check also catches a
reversed qubit register.

**Gradients are exact and batched.** Every ansatz parameter is a single RY angle,
so the parameter-shift rule is exact rather than a finite-difference
approximation. All 2N shifted circuits go to Aer in one `run` call.
`parameter_shift_is_exact()` verifies the preconditions rather than assuming
them, so swapping the ansatz degrades to finite differences instead of to
silently wrong gradients.

**Observables at the optimum are always available.** The QARP workflow carries an
entire second-class result — `VERIFIED (upper bound)`, with the penalty left in
the reported energy — for the case where the installed API exposes no
fixed-parameter observable evaluation. That failure mode cannot arise here, so
⟨N⟩, ⟨S²⟩ and the contamination budget are always measured.

**The sector energy bound is kept anyway.** It is no longer the fallback, but
validation still proves the exact ground energy of the six-electron sector, and
`vqe` still checks the converged energy against it. A state below that bound by
more than its own measured leak is a contradiction, not a better number, and is
reported as `INCONSISTENT`.

**MPS is cross-checked against statevector.** MPS is exact only while nothing is
truncated. `vqe` re-evaluates the converged state with `statevector`, prints the
difference, and withdraws the accuracy claim if it exceeds 0.16 mHa — the same
budget already applied to symmetry breaking.

**No MPI, no Slurm, no `job.sh`/`sim.job`.** A layer scan is a `foreach` loop and
the thread scan is gone: 12 qubits is 4096 amplitudes, 64 KiB, entirely in cache,
so extra Aer threads add fork/join cost without adding work.

### A finding that applies to the QARP workflow too

**θ = 0 is the right reference but a poor starting point.** On a
number-conserving Hamiltonian, a single RY rotation deep in the circuit can only
reach states whose electron count differs from the reference, so its gradient at
θ = 0 is exactly zero. Much of the ansatz is therefore frozen at an exact zero
start, and L-BFGS-B stops early reporting clean convergence.

`--start-noise` therefore defaults to a N(0, 0.05) kick. It moves the starting
energy by well under a millihartree — the penalty stays effectively zero and the
start is still Hartree–Fock for every practical purpose — while making every
parameter live. `vqe` prints the live-parameter count every run, and warns if you
ask for an exact zero start. `--start-noise 0` reproduces the frozen-start
behaviour on purpose, which is what the QARP workflow does today;
`--random-start` reproduces the original vacuum-start failure.

---

## Performance notes

Measured on this machine (Windows 11, Python 3.12.13, 12 qubits, a 1819-term
Hamiltonian — the same order as the real cache after the `1e-6` cutoff):

| | per circuit |
|---|---|
| `statevector` | ~60–100 ms |
| `matrix_product_state`, product state (θ = 0) | ~140 ms |
| `matrix_product_state`, converged HEA state | ~370 ms |

At 12 qubits **MPS is the slower method**, and that is expected. A state vector
is 4096 amplitudes — 64 KiB, sitting in cache — so there is nothing to compress,
and MPS only pays for its bookkeeping; its cost then grows with the entanglement
the optimizer builds, which is why the converged state is nearly 3× the product
state. MPS is here because it is the method that scales to the CAS(6e,7o) and
CAS(8e,8o) spaces this system is a warm-up for. `--method statevector` is the
right choice for this molecule if you only care about wall time.

Cost is set by `parameters × iterations × Pauli terms`, not by the state vector.
One L-BFGS-B iteration is 1 + 2N circuits: at four layers (N = 60) that is 121
circuits, roughly 12 s on statevector and 45 s on MPS. A four-layer run that
converges in ~20 iterations is therefore minutes on statevector and tens of
minutes on MPS; budget accordingly before starting a layer scan.

Aer threads are left on `auto`, for the same reason the QARP workflow pinned one
Qulacs thread: at this size extra threads mostly add fork/join cost. `--threads 1`
if you want it pinned.

### ADAPT is on a different cost curve entirely

`adapt` does not simulate circuits during optimization. Measured on the same
machine and the same 12-qubit, 1819-term Hamiltonian:

| | cost |
|---|---|
| Enumerate a pool | ~1 s, all three pools together |
| Build the sparse engine, `uccsd` (63 operators) | ~8–11 s, once |
| Build the sparse engine, `uccgsd` (435 operators) | **~130 s and ~570 MB, once** |
| One full pool-gradient sweep over 435 candidates | ~48 ms |
| One `expm_multiply` | ~1 ms |
| 6-operator ADAPT run, 656 exponentials | ~1 s |
| 30-operator `uccsd` run, start to finish | ~140 s |
| Emit + verify the circuit (`optimization_level=1`) | ~16–25 s |

The `uccgsd` engine build is the one place `adapt` sits still for a couple of
minutes — it is 435 `get_sparse_operator` calls, done once at startup, and it is
why operator *selection* afterwards costs 48 ms for the whole pool. Nothing about
it needs more than a free Colab runtime.

For contrast, one L-BFGS-B iteration of the four-layer hardware-efficient VQE is
121 Aer circuits at ~109 ms each — the whole self-test VQE was 2791 circuits and
228 s to reach an energy ADAPT beat in one second with six operators.

Two things drive that. The adjoint gradient is O(k) rather than O(k²) — 2k
exponentials for k parameters, against 2k+1 full evaluations each costing k. And
a 4096-amplitude state times a sparse operator is simply much cheaper than
transpiling, binding and simulating a circuit that prepares the same state. The
circuit is still built and checked at the end, so nothing is bought on credit;
that check is now the most expensive step of the whole command.

This is why the answer to "can I benchmark this on Colab" is yes without
qualification: the expensive part is `prepare`, which is classical quantum
chemistry and runs once.

---

## Scientific boundary, and what is comparable with the paper

Chemical accuracy here means agreement with CASCI for this exact active-space
Hamiltonian, not with experiment and not at the basis-set limit. The published
numbers, for CAS(6e,6o)/6-31G at 12 qubits
([Koziell-Pipe et al. 2026](https://arxiv.org/abs/2607.22468), Table 1):

| Quantity | Value |
|---|---|
| CASCI reference | −841.953249 Ha |
| Statevector VQE (ADAPT-GQE) | −841.952101 Ha (1.15 mHa) |
| Helios-1 hardware | −841.752841 Ha (200.4 mHa) |
| Two-qubit gates | 1440 → 244 after pytket optimization |

### Point by point

| | Paper | Here |
|---|---|---|
| Active space, basis, mapping | CAS(6e,6o)/6-31G, JW, 12 qubits | identical |
| Reference method | ADAPT-VQE, UCCGSD pool | identical (`run.py adapt`) |
| Reference state | HF determinant | identical |
| Convergence | pool gradient threshold | identical |
| Amortization | transformer (GQE) trained over ~15,000 geometries to emit circuits without re-optimizing | **not reproduced** — and not needed: for a single molecule the reference method *is* the target |
| Hardware | Helios-1 trapped ion, PMSV symmetry verification | **not attempted** — CPU/Colab only |
| Geometry | unpublished MD/NEB conformer ensemble | one PubChem CID 3696 conformer |

**What that last row costs you.** Total energies are not comparable across
conformers — tens of mHa of conformational difference is real physics, not
error — and the paper released no geometries, no code and no data. So the
absolute −841.952101 Ha is *not* reproducible here, and no amount of numerical
care changes that.

**What is comparable, and is the actual benchmark:**

1. **Error against your own CASCI.** Both sides compute E<sub>CASCI</sub> for
   their own Hamiltonian and report E<sub>VQE</sub> − E<sub>CASCI</sub>. Theirs
   is 1.15 mHa. This is conformer-independent and is what `adapt` prints as
   `Error vs CASCI`.
2. **Correlation energy recovered**, E<sub>CASCI</sub> − E<sub>HF</sub> and the
   fraction of it the ansatz captures — also conformer-insensitive, and the check
   that the same six frontier orbitals were selected.
3. **Two-qubit gate count** at the same accuracy, against their 244.

They trained to 5–10 mHa for their 12-qubit datasets and did not demonstrate a
universal 1.6 mHa guarantee. This workflow applies the stricter test on purpose
and reports failure if it is not met.
