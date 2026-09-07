# qubit_run — quantum chemistry benchmarks on Qiskit

Five self-contained quantum-chemistry studies, each reproducing a published
benchmark and each scored against an **exact classical answer for the identical
Hamiltonian**. Every study is one folder, one entry point (`python run.py`), and
one JSON result carrying the SHA-256 of the code that produced it.

The common thesis across all five: an energy is only a result if the state that
produced it is in the symmetry sector the reference describes. Each study
measures and *prices* the symmetry it breaks, and reports the number as unusable
when the price exceeds the error. Three of the five found a headline number that
this test invalidated.

---

## The studies

| Study | System | Qubits | Method | Status |
| --- | --- | --- | --- | --- |
| [studies/lih-10q](studies/lih-10q/) | LiH / STO-3G | 10 · 12 | UCCSD vs. hardware-efficient | complete |
| [studies/acrylamide-thiolate-8q](studies/acrylamide-thiolate-8q/) | Thio-Michael addition, C–S bond | 8 | UCCSD · k-UpCCGSD · ADAPT | **solver verified, chemistry pending** |
| [studies/imipramine-12q](studies/imipramine-12q/) | Imipramine, CAS(6e,6o)/6-31G | 12 | ADAPT-VQE | complete |
| [studies/li2s-24q](studies/li2s-24q/) | Li₂S dissociation, CAS(12e,12o) | 24 | HI-VQE | complete |
| [studies/methane-dimer-36q](studies/methane-dimer-36q/) | Methane dimer, CAS(16e,16o) | 36 | SQD + LUCJ | complete |

Each folder has its own `README.md` (install, commands, results, how to read a
verdict) and a `report/` holding a standalone LaTeX research report.

## Headline results

| Study | Reference | Best result | Verdict |
| --- | --- | --- | --- |
| lih-10q | CASCI(2e,5o), exact | UCCSD, **1.6 × 10⁻⁷ mHa** | verified |
| lih-10q | — | `real_amplitudes` ×8, 3.41 mHa | **symmetry broken** — contamination prices at 2.19 mHa |
| imipramine-12q | CASCI, exact | ADAPT `uccgsd`, **+0.017 mHa** | verified |
| imipramine-12q | — | ADAPT `uccsd`, +0.892 mHa | **spin contaminated** — and stopped on its operator budget |
| li2s-24q | CASCI, exact | 11 of 13 geometries inside 1.6 mHa | 2 points: **the reference is invalid**, not the method |
| acrylamide-8q | exact sector FCI | UCCSD, +0.978 mHa on the verification system | no molecular Hamiltonian built yet |

## Repository layout

```
qubit_run/
├── studies/                    one folder per benchmark, each self-contained
│   ├── acrylamide-thiolate-8q/
│   ├── imipramine-12q/
│   ├── li2s-24q/
│   ├── lih-10q/
│   └── methane-dimer-36q/
├── docs/
│   ├── combined-report/        the 12+24 qubit report (imipramine + Li₂S)
│   └── sqd-skqd-review.md      literature review of SQD/SKQD — the background
│                               for studies/methane-dimer-36q
└── .gitignore                  shared Python / LaTeX / venv rules
```

Inside a study:

```
<study>/
├── run.py                the single entry point — `python run.py` lists commands
├── README.md             install, commands, results, troubleshooting
├── requirements.txt      pinned stack
├── report/               main.tex + figures/ + main.pdf
├── tests/                `python -m unittest discover -s tests`
└── *.py                  flat modules, imported flat by run.py
```

Modules sit flat inside each study on purpose: `workflow_fingerprint()` hashes
them **by filename**, and every cache, validation receipt and result is bound to
that hash. Moving a module invalidates every artifact that depends on it.

## Getting started

Pick a study, then:

```bash
cd studies/lih-10q
uv venv --python 3.12 .venv && .venv/Scripts/activate   # or: source .venv/bin/activate
uv pip install -r requirements.txt

python run.py selftest     # prove the stack against exact answers — start here
python run.py              # list every command
```

`selftest` needs no chemistry package and no cached data. It is the right first
command in every study.

## The PySCF constraint

**PySCF publishes no Windows wheel**, and its source build is unsupported there.
Every study is built to survive that: the expensive chemistry runs **once**,
caches to JSON carrying a specification hash, and everything downstream reads the
cache and runs natively on Windows.

| Needs PySCF | Runs anywhere |
| --- | --- |
| `prepare`, `classical`, `ladder`, `geometry`, `reference` | `selftest`, `backend`, `validate`, `vqe`, `adapt`, `hivqe`, `sqd`, `scan`, `summary`, `plot`, `report` |

`requirements.txt` marks PySCF `sys_platform != "win32"` so `pip install -r`
succeeds on Windows. Colab notebooks are provided where the split matters
(`imipramine-12q`, `li2s-24q`, `methane-dimer-36q`).

`studies/lih-10q` is the exception: it ships a self-contained McMurchie–Davidson
STO-3G integral engine in NumPy, so it needs no chemistry package at all.

## Provenance

Every result JSON records the SHA-256 of the cache it read, the molecular
specification, and the source that produced it. A mismatch is a hard error, not
a warning — `vqe` refuses to start against a stale receipt, so editing a
workflow file invalidates every downstream result rather than silently producing
a mismatched one.

Caches and results are gitignored in most studies because they are large and
reproducible. To archive a specific run deliberately:

```bash
git add -f <cache>.json <cache>.validated.json
```

Keep the pair together — the receipt is bound to the cache by hash and is
meaningless without it.
