# Report source

`main.tex` — **Benchmarking ADAPT-VQE and HI-VQE at 12 and 24 Qubits**:
combined research report on Case A (imipramine, 12 qubits, ADAPT-VQE) and
Case B (Li₂S, 24 qubits, HI-VQE).

The argument is that three of the study's comparisons were invalid for a common
reason — states compared across different spin sectors, or across different
operator budgets — and that each was identified by recording one additional
expectation value. §6, *Claims not supported by these data*, lists the six
claims declined and the experiment that would decide each. Every claim is scoped
to what the existing data supports; no new calculations are assumed.

## Compile

```bash
pdflatex main.tex
pdflatex main.tex      # second pass: table of contents + cross-references
```

No `bibtex` pass — the bibliography is inline.

**Verified:** compiles with MiKTeX 25.12 / pdfTeX. 27 pages, **0 errors,
0 undefined references, 0 overfull boxes, 0 hyperref warnings**.

Packages used, all standard: `geometry, amsmath, amssymb, graphicx, booktabs,
array, caption, enumitem, fancyhdr, microtype, xcolor, tcolorbox, hyperref`.

## Figures

The composite multi-panel PNGs emitted by the workflow have been split into
individual panels, each placed and captioned separately. Filenames use hyphens,
not underscores.

| File | Figure | Content |
|---|---|---|
| `a-uccgsd-error.png` | 1 | ADAPT convergence, UCCGSD pool |
| `a-uccgsd-energy.png` | 2 | Same trajectory, absolute energy |
| `a-uccgsd-gradient.png` | 3 | Pool gradient — the termination criterion |
| `a-uccgsd-operators.png` | 4 | Largest rotation angles in the ansatz |
| `a-uccsd-gradient.png` | 5 | Truncated run halting on its budget |
| `a-uccsd-error.png` | 6 | Truncated run error trajectory |
| `li2s-occupancies.png` | 7 | Natural occupancies at both ends |
| `b-cost-measurement.png` | 8 | Measurement settings per iteration |
| `b-cost-circuit.png` | 9 | Sampling-circuit resources |
| `b-curve.png` | 10 | Dissociation curve |
| `b-curve-error.png` | 11 | Error against CASCI, log scale |
| `b-conv-error.png` | 12 | Error by iteration |
| `b-conv-dets.png` | 13 | Subspace size by iteration |
| `li2s-ablation.png` | 14 | Ablation — **reproduced to be withdrawn**, see §4.7 |

The original composites (`imipramine-uccgsd.png`, `imipramine-uccsd.png`,
`li2s-dissociation.png`, `li2s-convergence.png`, `li2s-cost.png`) are retained
as the sources of the crops. `imipramine-comparison.png`,
`li2s-compression.png` and `li2s-energy.png` are unused.

## Data provenance

Every quantity comes from these result sets only:

- `Results/imipramine_results/` (physics `c6cd75127aac`, matching the current
  `imipramine_qiskit/` source under line-ending normalisation)
- `Results/li2s_24_cache (1)/` and `Results/li2s_24_results (1)/`
  (physics `f6989a3a6000`, matching the current `li2s_24/` source)

The `li2s_24_cache/` and `li2s_24_results/` sets **without** the `(1)` suffix
are superseded: they precede the `direct_spin0` → `direct_spin1` reference
correction and differ by 10.1 mHa in the CASCI energy at 6.00 Å. They are
described in Appendix E and are not used for any quantity in the report.
