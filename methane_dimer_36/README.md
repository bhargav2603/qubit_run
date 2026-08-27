# Methane dimer at 36 qubits — SQD on Qiskit

A reproduction of the 36-qubit sample-based quantum diagonalization experiment in
**[Kaliakin, Shajan, Robledo Moreno, Li, Mitra, Motta, Johnson, Saki, Das, Sitdikov,
Mezzacapo & Merz, *Accurate quantum-centric simulations of supramolecular
interactions*, arXiv:2410.09209](https://arxiv.org/abs/2410.09209)**
(*Communications Physics*, 2025) — the methane dimer potential energy surface at
**CAS(16e,16o)/aug-cc-pVQZ**, **165,636,900 determinants**, benchmarked against
CASCI in the same active space.

```bash
python run.py selftest                     # prove the logic — needs no chemistry package
python run.py plan                         # what this costs, before you spend it
python run.py validate                     # prove the CHEMISTRY, end to end, in seconds
python run.py geometry --distance 3.638    # build and inspect the structure

python run.py prepare   --distance 3.638   # RHF + AVAS -> cached Hamiltonian
python run.py reference --distance 3.638   # CASCI / CCSD / CCSD(T)
python run.py sqd       --distance 3.638   # the 36-qubit run
python run.py ablation  --distance 3.638   # the control that makes it mean something
python run.py verify                       # the paper's headline claim, pass/fail
python run.py report                       # Markdown + standalone HTML
```

## Read this before running anything expensive

**`python run.py validate` is not optional.** It exercises the entire chemistry
pipeline — RHF, AVAS, cache round-trip, CCSD, LUCJ, ffsim sampling, SQD,
variance, ablation, binding energy — on a tiny STO-3G active space where the
exact answer is instant. Same molecule, same functions, only the basis shrinks.

Its load-bearing assertion: **when the subspace saturates the CAS, SQD must equal
CASCI to 1×10⁻⁸ Ha** — solver precision, not chemical accuracy. If the SQD
Hamiltonian and the CASCI Hamiltonian are ever not the same operator, that check
fails immediately instead of producing a plausible-looking wrong number hours later.

### What "chemical accuracy" can and cannot mean here

| Claim | Achievable? |
|---|---|
| SQD agrees with **your own CASCI** to < 1 kcal/mol | **Yes** — this is the paper's claim, and `run.py verify` tests it against their 0.010 kcal/mol figure at 3.638 Å |
| SQD never falls below CASCI (variational bound) | **Yes** — guaranteed by construction, and asserted |
| Binding-energy *curve shape* matches the paper | **Yes**, up to the orientation choice |
| **Absolute total energies match the paper** | **No** — and no amount of correct code changes this |

The last row is not a limitation of this implementation. The paper never
publishes the monomer orientation, and D3d/D3h/D2d differ by more than the effect
being measured. Since SQD and CASCI are computed at the *same* geometry, the
agreement under test is unaffected; the absolute numbers are not.

---

## Why this paper, at this size

Three SQD results sit near 30 qubits. They are not equally reproducible, and the
differences are not cosmetic:

| | methylamine **30q** | LiPF₆ salts **32q** | **methane dimer 36q** |
|---|---|---|---|
| Venue | J. Phys. Chem. B | preprint only | **Communications Physics** |
| Geometry published | procedure only | ✗ Gaussian, not deposited | **✓ explicit distance grid** |
| Active space specified | ✓ AVAS + AO list | ✗ "HOMO–LUMO frontier" | **✓ AVAS + AO list** |
| SQD hyperparameters | partial | ✗ none given | **✓ Table II** |
| Code you must reinvent | **IEF-PCM patch** | none | **none** |
| Reference | CASCI | CASCI | **CASCI + CCSD(T)** |

The methylamine paper states plainly that "the SQD IEF-PCM method is enabled
through *modification* of the Qiskit addon: SQD and PySCF codes," and publishes
no code-availability statement. This paper's only code citation is stock
[`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd), and its SI
prints the distance grid. That is the whole reason to prefer it.

## What runs where

`pyscf` and `ffsim` publish **no Windows wheels** — manylinux and macOS only,
which I verified against PyPI rather than assumed:

```
ffsim  0.0.84  -> manylinux_2_28_x86_64, manylinux_2_28_aarch64, macosx_11_0_arm64
pyscf  2.14.0  -> manylinux_2_17_x86_64, manylinux2014_aarch64, macosx_*
```

So the chemistry runs on **Colab or WSL2**, not on Windows. Everything that
decides whether a run is *correct* was deliberately kept free of that
dependency, and runs anywhere:

```bash
python run.py selftest      # 29 checks, no chemistry packages
python -m pytest tests/ -q  # 83 tests, no chemistry packages
python run.py plan          # the full cost model
python run.py geometry      # structures, validated and exportable to XYZ
```

## The numbers this is anchored to

```
active space           CAS(16e,16o) / aug-cc-pVQZ
qubits                 32 occupation + 4 ancilla = 36
alpha strings          C(16,8) = 12,870
determinants           165,636,900
sector statevector     2.47 GiB
dense 2^32 vector      64.00 GiB  (26x larger)

paper subspace d       126,000,000  (76.1% of the full CAS)
  one CI vector        961 MiB
  10 batches parallel  112.65 GiB   <- what the paper ran, on 10 CPUs
  10 batches sequential 11.27 GiB   <- the affordable schedule
```

**Where the 36 comes from.** 16 spatial orbitals occupy 32 qubits under
Jordan–Wigner. The LUCJ ansatz also needs α–β density-density interactions,
which heavy-hex routes through ancillas at every fourth orbital:
`len(range(0, 16, 4)) = 4`. That is 32 + 4 = 36, matching the paper's Figure 2B.
`ansatz.validate_layout` asserts it rather than trusting it.

## Three things worth knowing before you start

**1. The classical reference costs as much as the quantum run.** CASCI(16e,16o)
is a 165,636,900-dimensional eigenproblem — a single CI vector is 961 MiB and
Davidson holds a dozen. Budget ~15 GiB. Free Colab will not do it; high-RAM
Colab will. `run.py plan` prints this before you commit.

**2. The paper's converged point is not reachable in parallel on one machine.**
`|χ̃_b| = 20×10³` gives `d = 1.26×10⁸`, and ten batches resident at once is
113 GiB. Run them sequentially (11 GiB peak), or use the paper's own escape
hatch: SI §I extrapolates from `|χ̃_b| = 9, 11, 14 ×10³` to the converged
answer. Those three rungs are in `run.py --rung`, and `binding.variance_extrapolate`
implements the extrapolation.

**3. "Percent of shots valid" is a weak diagnostic *for this system*.** The
familiar framing — 2% valid on hardware versus ~10⁻⁶ at random — comes from the
52-qubit N₂ case, which is far from half filling. The methane dimer is (8,8) in
16 orbitals, exactly half filled, and

```
C(16,8)^2 / 2^32 = 3.86%
```

of *random* 32-bit strings are already valid configurations. A few percent
validity on hardware would demonstrate nothing here. The self-test asserts this
so it cannot be quietly misread later. **The discriminating measurement is the
ablation**, not the validity fraction — which is why `run.py ablation` exists and
why the report leads with it.

## The files

| File | Purpose |
|---|---|
| `run.py` | The one interface. Every operation is `python run.py <command>`. |
| `paper.py` | Every value taken from arXiv:2410.09209, and an explicit record of what it does **not** publish. |
| `geometry.py` | Methane dimer construction. The three candidate orientations, and the invariant checks. |
| `spaces.py` | Determinant counting, memory models, the cost ladder. No chemistry imports. |
| `binding.py` | Binding energies (paper Eq. 2), variance extrapolation, the variational bound check. |
| `chemistry.py` | Geometry → RHF → AVAS → cached Hamiltonian. **The only module that imports PySCF.** |
| `reference.py` | CASCI / CCSD / CCSD(T) in the active space. |
| `ansatz.py` | LUCJ construction and the heavy-hex layout. **The only module that imports ffsim's variational API.** |
| `sampling.py` | Three sample sources — noiseless, hardware, uniform — behind one interface. |
| `sqd.py` | Instrumented wrapper over `diagonalize_fermionic_hamiltonian`, plus the energy variance. |
| `validation.py` | Fingerprints, receipts, invariants. |
| `selftest.py` | 29 checks that run with no chemistry packages. |
| `validate.py` | End-to-end chemistry proof on a tiny active space. Run before anything expensive. |
| `report.py` | Markdown + standalone HTML, always carrying its own caveats. |

The design rule: **PySCF is imported in exactly one module and ffsim's
variational API in exactly one other.** Everything that decides correctness sits
outside both, which is why 80 tests run on a machine that cannot install either.

## What is and is not comparable with the paper

| | arXiv:2410.09209 | Here |
|---|---|---|
| Active space | CAS(16e,16o)/aug-cc-pVQZ, 165,636,900 dets | **identical** |
| Qubits | 36 (32 + 4 ancilla) | **identical, and asserted** |
| Distance grid | 14 PES points + 3.638 Å + 48.000 Å | **identical** |
| AVAS AO list | C[2s,2p], H[1s] | **identical** |
| Binding energy | `E(R) − E(48 Å)`, Eq. (2) | **identical, and enforced in code** |
| Reference method | CASCI (16e,16o) | **identical** |
| SQD parameters | \|χ̃\|=200k, K=10, \|χ̃_b\|=20k, 10 steps | **identical** (Table II) |
| Mitigation | gate twirling + DD, no measurement twirling | identical, when run on hardware |
| **Monomer orientation** | **not published** | **ours — D3d by default, a required flag** |
| **C–H bond length** | **not published** | **ours — 1.0870 Å, a flag** |
| Convergence tolerances, `n_reps`, seed | **not published** | ours, all exposed as flags |
| Hardware | ibm_cleveland | noiseless by default; any Heron if you have access |

**Absolute total energies are therefore not reproducible from this paper, and
this folder does not claim to reproduce them.** The paper publishes a distance
grid but no coordinates, and never names the relative orientation of the two
monomers — which for a methane dimer is not a detail. D3d, D3h and D2d are
separated by a few tenths of a kcal/mol, comparable to the binding energy itself
and far larger than the 0.010 kcal/mol agreement being tested.

What *is* comparable, and what this workflow measures:

1. **SQD against your own CASCI, at the same geometry.** Both sides move
   together when the orientation changes, so their agreement — the paper's
   actual claim — is invariant to the one thing that is missing. This is the
   pass/fail in `run.py verify`, and the target is 0.010 kcal/mol at 3.638 Å.
2. **The subspace fraction.** `subspace_fraction` is in every result file.
3. **The shape of the binding curve**, and the well depth it implies.
4. **The ablation gap** — SQD versus uniform sampling at matched subspace
   dimension.

## Scope, stated plainly

At 36 qubits the full CAS is 165,636,900 determinants and is exactly
diagonalizable on a large-memory machine, and the paper's own converged subspace
covers **76%** of it. **Nothing here is evidence of quantum advantage.** Of the
three systems in that paper, only the 54-qubit (16e,24o) run compresses anything
— its subspace is 0.046% of the CAS. This one is a validation of the
sampling-plus-recovery pipeline at a size where the answer is known, which is the
only regime in which a method can be shown to be *right*.

The ablation table is included so the quantum layer's contribution is a number
you read rather than a claim you accept.

## References

| Work | What this folder took from it |
|---|---|
| Kaliakin *et al.*, *Accurate quantum-centric simulations of supramolecular interactions*, [arXiv:2410.09209](https://arxiv.org/abs/2410.09209); *Commun. Phys.* (2025) | **The experiment**: the (16e,16o) active space, the AVAS AO list, the distance grid, Table II's SQD parameters, and Eq. (2)'s binding-energy definition. |
| Robledo-Moreno *et al.*, *Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer*, [arXiv:2405.05068](https://arxiv.org/abs/2405.05068); *Sci. Adv.* **11**, eadu9991 (2025) | **SQD itself** — subspace projection and self-consistent configuration recovery. |
| Motta *et al.*, *Bridging physical intuition and hardware efficiency for correlated electronic states: the local unitary cluster Jastrow ansatz* (2023) | **The ansatz.** |
| Sayfutyarova *et al.*, *J. Chem. Theory Comput.* **13**, 4063 (2017) | **AVAS**, the active-space construction. |
| [`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd) 0.13.1 | Configuration recovery, batching, the selected-CI solver. |
| [`ffsim`](https://github.com/qiskit-community/ffsim) 0.0.84 | LUCJ construction, the heavy-hex pass manager, and exact sampling inside the particle-number sector. |
| [PySCF](https://github.com/pyscf/pyscf) 2.14.0 | SCF, AVAS, integrals, CASCI, CCSD(T). |
