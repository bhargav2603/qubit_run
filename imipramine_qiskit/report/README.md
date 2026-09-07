# Report source — imipramine CAS(6e,6o), 12 qubits

`main.tex` — **ADAPT-VQE on Imipramine, CAS(6e,6o), 12 Qubits**.

The argument is that the study's second run does not support the pool
comparison it appears to make. It halted on an exhausted operator budget with a
residual pool gradient twenty times its own threshold, and its energy is
separately invalidated by a spin contamination that prices at seventeen times
the budget. Both were identified by quantities the algorithm already computes.
§6, *Claims not supported by these data*, lists the six claims declined and the
experiment that would decide each.

## Compile

```bash
pdflatex main.tex
pdflatex main.tex      # second pass: table of contents + cross-references
pdflatex main.tex      # third pass: settles the last references
```

No `bibtex` pass — the bibliography is inline.

**Verified:** compiles with MiKTeX 25.12 / pdfTeX. 13 pages, 0 errors,
0 undefined references, 0 overfull boxes, 0 hyperref warnings.

Packages used, all standard: `geometry, amsmath, amssymb, graphicx, booktabs,
array, caption, enumitem, fancyhdr, microtype, xcolor, tcolorbox, hyperref`.

## Figures

The composite four-panel PNGs emitted by `run.py plot` have been split into
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

`imipramine-uccgsd.png` and `imipramine-uccsd.png` are the source composites and
are retained. `imipramine-comparison.png`, `a-uccsd-energy.png` and
`a-uccsd-operators.png` are unused.

Figures 3 and 5 are the pair that carries the report's main argument: one run
reaches its stopping threshold and the other does not.

## Data provenance

Every quantity comes from `../results/`, physics digest `c6cd75127aac`:

| File | Contents |
|---|---|
| `adapt_uccgsd.json` | the verified run — 58 operators, +0.0173 mHa |
| `adapt_uccsd.json` | the truncated, spin-contaminated run |
| `classical.json` | MP2 / CCSD / CCSD(T) in the same active space |
| `../hamiltonian_cas6e6o.json` | the 12-qubit cache, 1,819 Pauli terms |
| `../hamiltonian_cas6e6o.validated.json` | its receipt, three penalty keys |

Both runs carry identical cache, specification and physics digests, so they
solved the same operator — which is the precondition for comparing them at all.

Results were produced on Colab (Linux 6.6.122, Python 3.12.13, qiskit 1.0.2,
qiskit-aer 0.14.2, pyscf 2.14.0). The Windows stack the workflow also targets is
recorded alongside in every result file and in Appendix C.
