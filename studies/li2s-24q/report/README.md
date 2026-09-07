# Report source — Li₂S at 24 qubits

`main.tex` — **HI-VQE on Li₂S at 24 Qubits: a variationally bounded method used
as a certificate on its own reference**.

The argument is that the two geometries where HI-VQE falls *below* CASCI are not
method failures. A projected subspace energy cannot lie below a correct
reference, so the violation is information about the reference — confirmed
independently by the non-monotonicity it induces in the reference curve, and by
both points returning ⟨S²⟩ = 2.00. §5, *Claims not supported by these data*,
lists the six claims declined, including the ablation's negative reading, which
is reported rather than buried.

## Compile

```bash
pdflatex main.tex
pdflatex main.tex      # second pass: table of contents + cross-references
pdflatex main.tex      # third pass: settles the last references
```

No `bibtex` pass — the bibliography is inline.

**Verified:** compiles with MiKTeX 25.12 / pdfTeX. 14 pages, 0 errors,
0 undefined references, 0 overfull boxes, 0 hyperref warnings.

Packages used, all standard: `geometry, amsmath, amssymb, graphicx, booktabs,
array, caption, enumitem, fancyhdr, microtype, xcolor, tcolorbox, hyperref`.

## Figures

Composite PNGs from `run.py plot`, split into individual panels and captioned
separately. Filenames use hyphens, not underscores.

| File | Figure | Content |
|---|---|---|
| `b-cost-measurement.png` | 1 | 15,697 measurement settings against 1 |
| `b-cost-circuit.png` | 2 | Sampling-circuit resources |
| `b-curve.png` | 3 | Dissociation curve |
| `b-curve-error.png` | 4 | Error against CASCI, log scale |
| `b-conv-error.png` | 5 | Error by iteration |
| `b-conv-dets.png` | 6 | Subspace size by iteration |
| `li2s-occupancies.png` | 7 | Natural occupancies at both ends |
| `li2s-ablation.png` | 8 | Ablation — reproduced with its caveat, see §3.6 |

`li2s-dissociation.png`, `li2s-convergence.png` and `li2s-cost.png` are the
source composites and are retained. `li2s-compression.png` and
`li2s-energy.png` are unused.

## Data provenance

Every quantity comes from `../cache/` and `../results/`, workflow digest
`172e18c71130`:

| Path | Contents |
|---|---|
| `cache/li2s_r*.json` | 13 cached Hamiltonians |
| `cache/li2s_r*.validated.json` | their receipts |
| `results/hivqe_r*_sector.json` | 13 single-point runs |
| `results/hivqe_r2.100_none.json` | classical selected-CI control |
| `results/hivqe_r2.100_sector_noexp.json` | sampler-only control |
| `results/hivqe_r2.100_sector_paper-rule.json` | the paper's own selection rule |
| `results/scan_sector.json` | the assembled dissociation scan |
| `results/classical.json` | MP2 / CCSD / CCSD(T) with T1 diagnostics |
| `results/report.md`, `report.html`, `report_data.json` | the workflow's own report |

Sixteen single-point runs and one scan.

## Superseded data

In the external archive, the `li2s_24_cache/` and `li2s_24_results/` sets
*without* a `(1)` suffix precede the `direct_spin0` → `direct_spin1` reference
correction and differ by 10.1 mHa in the CASCI energy at 6.00 Å. Only the `(1)`
sets were copied into this folder, and no quantity in the report uses the
superseded ones.

## A note on the two invalid points

The report diagnoses the reference at 3.20 and 3.60 Å as wrong, but does not
repair it. Closing the curve means recomputing CASCI at those two geometries in
the triplet sector and re-scoring. Until that is done, the HI-VQE energies at
those points are not claimed as correct either — they are lower numbers than a
reference that is itself unreliable.
