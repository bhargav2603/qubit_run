"""Turn the result files into a report that states its own limits.

The report always carries the two things a reader needs in order to judge the
numbers: which geometry choice produced them (the paper does not publish one),
and what fraction of the CAS the subspace covered (which is what separates a
compression claim from a validation exercise).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import binding
import geometry
import paper
import spaces
import validation


def _rows(results: list[dict[str, Any]], kind: str) -> dict[float, dict[str, Any]]:
    return {r["spec"]["distance"]: r for r in results if r.get("kind") == kind}


def _markdown(results: list[dict[str, Any]]) -> str:
    references = _rows(results, "reference")
    sqd_runs = _rows(results, "sqd")
    ablations = _rows(results, "ablation")

    lines: list[str] = []
    add = lines.append

    add("# Methane dimer at 36 qubits — SQD reproduction")
    add("")
    add(f"Reproducing **{paper.TITLE}**  ")
    add(f"[arXiv:{paper.ARXIV_ID}{paper.ARXIV_VERSION}]({paper.URL}) — {paper.JOURNAL}")
    add("")
    add(f"- Active space: CAS({paper.N_ELECTRONS}e,{paper.N_ORBITALS}o) / {paper.BASIS}")
    add(f"- Qubits: {paper.N_QUBITS_OCCUPATION} occupation + "
        f"{paper.N_QUBITS_ANCILLA} ancilla = **{paper.N_QUBITS_TOTAL}**")
    add(f"- Full CAS: **{spaces.full_cas_dimension():,}** determinants")
    add(f"- Binding energy: `{paper.BINDING_ENERGY_DEFINITION}`")
    add("")

    orientations = {r["spec"]["orientation"] for r in results if "spec" in r}
    if orientations:
        add("## Geometry — an unpublished choice")
        add("")
        add(f"Orientation(s) used: **{', '.join(sorted(orientations))}**. "
            "The paper does not publish coordinates or the relative orientation "
            "of the monomers; see `paper.GEOMETRY_GAP`. Because SQD and CASCI are "
            "computed at the *same* geometry, their agreement — the claim under "
            "test — is unaffected. Absolute energies and the well depth are not.")
        add("")

    if references:
        add("## Classical references (active space)")
        add("")
        add("| R (Å) | RHF | CCSD | CCSD(T) | CASCI |")
        add("|---:|---:|---:|---:|---:|")
        for distance, row in sorted(references.items()):
            def fmt(key: str) -> str:
                value = row.get(key)
                return f"{value:.8f}" if value is not None else "—"
            add(f"| {distance:.3f} | {fmt('hf')} | {fmt('ccsd')} "
                f"| {fmt('ccsd_t')} | {fmt('casci')} |")
        add("")

    if sqd_runs:
        add("## SQD")
        add("")
        add("| R (Å) | rung | SQD (Ha) | CASCI (Ha) | Δ (mHa) | Δ (kcal/mol) "
            "| d | % of CAS |")
        add("|---:|:--|---:|---:|---:|---:|---:|---:|")
        for distance, row in sorted(sqd_runs.items()):
            casci = references.get(distance, {}).get("casci")
            if casci is not None:
                agreement = binding.Agreement(row["energy"], casci)
                delta_mha = f"{agreement.delta_mha:+.4f}"
                delta_kcal = f"{agreement.delta_kcal:+.4f}"
                casci_text = f"{casci:.8f}"
            else:
                delta_mha = delta_kcal = casci_text = "—"
            add(f"| {distance:.3f} | {row.get('rung', '—')} | {row['energy']:.8f} "
                f"| {casci_text} | {delta_mha} | {delta_kcal} "
                f"| {row['subspace_dimension']:,} "
                f"| {row['subspace_fraction']:.2%} |")
        add("")

    if ablations:
        add("## Ablation — did the quantum samples matter?")
        add("")
        add("Uniform random configurations at matched subspace dimension. "
            "The gap is the quantum layer's actual contribution.")
        add("")
        add("| R (Å) | SQD (Ha) | uniform (Ha) | gap (mHa) | d (SQD) | d (uniform) |")
        add("|---:|---:|---:|---:|---:|---:|")
        for distance, row in sorted(ablations.items()):
            quantum = sqd_runs.get(distance)
            if quantum is None:
                continue
            gap = (row["energy"] - quantum["energy"]) / binding.MILLIHARTREE
            add(f"| {distance:.3f} | {quantum['energy']:.8f} | {row['energy']:.8f} "
                f"| {gap:+.4f} | {quantum['subspace_dimension']:,} "
                f"| {row['subspace_dimension']:,} |")
        add("")

    # Binding curves, when the unbound reference exists.
    for label, rows in (("CASCI", references), ("SQD", sqd_runs)):
        key = "casci" if label == "CASCI" else "energy"
        energies = {
            d: r[key] for d, r in rows.items() if r.get(key) is not None
        }
        if paper.UNBOUND_DISTANCE in energies and len(energies) > 1:
            curve = binding.binding_curve(
                energies, unbound_distance=paper.UNBOUND_DISTANCE
            )
            add(f"## Binding energy — {label}")
            add("")
            add("| R (Å) | E_bind (kcal/mol) |")
            add("|---:|---:|")
            for distance, value in sorted(curve.items()):
                add(f"| {distance:.3f} | {value:+.4f} |")
            add("")

    add("## Scope")
    add("")
    add(f"At 36 qubits the full CAS is {spaces.full_cas_dimension():,} determinants "
        "and is exactly diagonalizable, and the paper's own converged subspace "
        f"covers {spaces.subspace_fraction(paper.SUBSPACE_DIMENSION):.0%} of it. "
        "**Nothing here is evidence of quantum advantage.** This is a validation "
        "of the sampling-plus-recovery pipeline at a size where the answer is "
        "known, which is the only regime in which a method can be shown correct. "
        "The ablation table exists so the quantum layer's contribution is a "
        "number you read rather than a claim you accept.")
    add("")
    add(f"Physics fingerprint `{validation.physics_fingerprint()}` · "
        f"workflow `{validation.workflow_fingerprint()}`")
    return "\n".join(lines)


def _html(markdown_text: str) -> str:
    body = html.escape(markdown_text)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Methane dimer @ 36 qubits</title>
<style>
 body {{ max-width: 60rem; margin: 3rem auto; padding: 0 1.5rem;
        font: 15px/1.65 -apple-system, "Segoe UI", system-ui, sans-serif;
        color: #1a1a1a; background: #fff; }}
 pre  {{ white-space: pre-wrap; font: 13px/1.6 ui-monospace, Menlo, Consolas, monospace; }}
 @media (prefers-color-scheme: dark) {{
   body {{ color: #e8e8e8; background: #16181c; }}
 }}
</style>
<pre>{body}</pre>
"""


def generate(results_dir: Path, markdown_path: Path, html_path: Path) -> int:
    results = validation.load_all(results_dir)
    if not results:
        print(f"no results in {results_dir}. Run `python run.py sqd` first.")
        return 1

    stale = sum(1 for r in results if r.get("_stale"))
    text = _markdown(results)
    markdown_path.write_text(text, encoding="utf-8")
    html_path.write_text(_html(text), encoding="utf-8")

    print(f"wrote {markdown_path}")
    print(f"wrote {html_path}")
    print(f"{len(results)} result file(s)"
          + (f", {stale} stale (physics modules changed since)" if stale else ""))
    return 0
