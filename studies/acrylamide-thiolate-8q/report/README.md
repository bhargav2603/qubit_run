# Report source — acrylamide + methanethiolate

`main.tex` — **Verification of a Local VQE Stack for a Thio-Michael Addition**.

This is a **verification report, not a chemistry report.** No Hamiltonian has
been built for either target geometry, because PySCF publishes no Windows wheel
and no WSL distribution is installed on the machine used. §7 states what remains
— three commands, with the projected cost.

What the report does establish: 20 exact checks on the solver, an ansatz
comparison, five implementation findings, and the measured qubit count at which
this workflow stops fitting on a laptop.

## Compile

```bash
pdflatex main.tex
pdflatex main.tex      # second pass: table of contents + cross-references
pdflatex main.tex      # third pass: settles the last references
```

No `bibtex` pass — the bibliography is inline.

**Verified:** compiles with MiKTeX 25.12 / pdfTeX. 11 pages, 0 errors,
0 undefined references, 0 overfull boxes, 0 hyperref warnings.

Packages used, all standard: `geometry, amsmath, amssymb, graphicx, booktabs,
array, caption, enumitem, fancyhdr, microtype, xcolor, tcolorbox, hyperref`.

## Figures

Generated from the measurement tables in the folder README; there is no result
JSON to plot from yet.

| File | Figure | Content |
|---|---|---|
| `acr-ansatz.png` | 1 | Five ansätze against exact sector FCI |
| `acr-cost.png` | 2 | Accuracy against wall time, marker area ∝ circuit depth |
| `acr-scaling.png` | 3 | Projected 50-iteration cost against register size |

## Data provenance

- **Table 3** (the 20 checks): a `python run.py selftest` run executed for this
  report on Python 3.13.7 / qiskit 2.3.1 / qiskit-aer 0.17.2 / qiskit-nature
  0.7.2, 20/20 passing in 168.5 s. Both environments are tabulated in
  Appendix A.
- **Tables 4, 5, 8–12**: measurements recorded in the folder README, taken on
  Python 3.12.13 / qiskit 2.5.2 / qiskit-aer 0.17.2, 8 logical cores.
- **Tables 6 and 7** beyond 14 qubits are projections from measured
  per-gradient costs, not completed runs. This is stated in the Limitations
  section rather than left for the reader to infer.

## What is missing

`acrylamide_thiolate_hamiltonian.json`, `..._vqe_result.json` and
`..._classical_ladder.json` do not exist. Producing them:

```bash
# once, on Colab / Linux / macOS / WSL
python colab_prepare.py --active-spaces 8q --geometries ts,far

# then anywhere, including Windows
python run.py validate --cache hamiltonian_ts_8q.json --number-penalty 0
python run.py vqe      --cache hamiltonian_ts_8q.json
python run.py vqe      --cache hamiltonian_far_8q.json
python run.py ladder
```

Projected at ~0.6 h per geometry for the VQE at 8 qubits with UCCSD, plus the
PySCF stage, whose dominant cost is CCSD(T) on aug-cc-pVDZ. Once those files
exist, §7 is replaced by a results section reporting the TS−FAR difference
beside the CASCI−CCSD(T) gap, and the report's title changes.
