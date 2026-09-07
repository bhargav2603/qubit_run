# Report source — LiH on Qiskit Aer

`main.tex` — **Ansatz Expressivity and Symmetry Accounting in a 10- and
12-Qubit LiH Benchmark**.

The argument is that the eight-layer `real_amplitudes` run, which improves on
the four-layer run by a factor of six, is not a usable result: its symmetry
contamination prices at 2.19 mHa against a 0.16 mHa budget, which is more than
the error it claims to have left. §5, *Claims not supported by these data*,
lists the five claims declined and the experiment that would decide each. No new
calculations are assumed.

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

Generated from the four committed result JSONs. This workflow has no plotting
command of its own, so the figures are produced by the snippet recorded in the
project notes rather than by `run.py`.

| File | Figure | Content |
|---|---|---|
| `lih-error.png` | 1 | Error against exact diagonalisation, all four runs |
| `lih-contamination.png` | 2 | Error beside the priced symmetry contamination |
| `lih-cost.png` | 3 | Circuits executed against error, marker area ∝ wall time |

## Data provenance

Every quantity comes from files committed beside `run.py`:

- `lih_frozen_core_vqe_result.json` — UCCSD, 10 qubits
- `lih_full_vqe_uccsd.json` — UCCSD, 12 qubits
- `lih_frozen_core_vqe_hea4.json` — `real_amplitudes`, 4 layers
- `lih_frozen_core_vqe_hea8.json` — `real_amplitudes`, 8 layers
- `lih_frozen_core_hamiltonian.json`, `lih_full_hamiltonian.json` and their
  `.validated.json` receipts

Two exceptions, both stated in the report: the initialisation study (Table 6)
and the MPS-versus-statevector timings come from `selftest` and `backend`
measurements recorded in the folder README, not from a result file.
