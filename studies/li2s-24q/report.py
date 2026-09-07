#!/usr/bin/env python3
"""Assemble everything in results/ into one report: Markdown and standalone HTML.

The HTML embeds its figures as base64 data URIs, so the single file is the whole
report and can be mailed, uploaded or opened from a Colab download without
carrying a directory of PNGs with it.

The report states what was measured, against what reference, with what settings
and what code fingerprint. Where a comparison with arXiv:2503.06292 is not
possible -- and the total energies are not, because the paper publishes no
geometry -- it says so in the report rather than in a footnote nobody reads.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _datetime
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hamiltonian import MoleculeSpec, load_cache, workflow_fingerprint
from summary import load_results


CHEMICAL_ACCURACY_MHA = 1.6


def _fmt(value: Any, spec: str = ".6f", missing: str = "--") -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return missing
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def _environment() -> dict[str, str]:
    versions = {}
    for module in ("numpy", "scipy", "qiskit", "qiskit_aer", "pyscf", "openfermion"):
        try:
            imported = __import__(module)
            versions[module] = getattr(imported, "__version__", "unknown")
        except ImportError:
            versions[module] = "not installed"
    versions["python"] = sys.version.split()[0]
    versions["platform"] = platform.platform()
    return versions


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def section_header(spec: MoleculeSpec, singles, scans) -> list[str]:
    stamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    published = spec.published_reference or {}
    return [
        f"# {spec.name} at 24 qubits -- HI-VQE",
        "",
        f"*Generated {stamp} -- workflow `{workflow_fingerprint()[:12]}`*",
        "",
        "Handover Iterative Variational Quantum Eigensolver "
        "([arXiv:2503.06292](https://arxiv.org/abs/2503.06292), Qunova Computing) "
        f"applied to {spec.name} in the active space that paper specifies: "
        f"**CAS({spec.n_active_electrons}e,{spec.n_active_orbitals}o) / "
        f"{spec.basis.upper()}, {spec.n_qubits} qubits, "
        f"{published.get('full_cas_determinants', 0):,} determinants**. "
        "Circuits are Qiskit; the classical eigensolver is exact.",
        "",
        f"- **{len(singles)}** single-point runs, **{len(scans)}** dissociation scans",
        "- Reference: CASCI, exact within this active space",
        f"- Target: **{CHEMICAL_ACCURACY_MHA} mHa** (chemical accuracy) against that reference",
        "",
    ]


def section_headline(singles, scans) -> list[str]:
    lines = ["## Headline result", ""]
    if scans:
        scan = scans[0]
        errors = np.abs(np.array(scan["error_millihartree"]))
        reference = np.array(scan["reference"])
        hartree_fock = np.abs(np.array(scan["hartree_fock"]) - reference) * 1000
        fractions = np.array(scan["subspace_fraction"]) * 100
        inside = int((errors <= CHEMICAL_ACCURACY_MHA).sum())
        lines += [
            f"Across **{len(errors)} geometries** of the Li-S dissociation coordinate:",
            "",
            "| | worst | mean |",
            "|---|---|---|",
            f"| HI-VQE error vs CASCI | {errors.max():.3f} mHa | {errors.mean():.3f} mHa |",
            f"| Hartree-Fock error vs CASCI | {hartree_fock.max():.1f} mHa | "
            f"{hartree_fock.mean():.1f} mHa |",
            f"| Determinants used | {max(scan['dimension']):,} | "
            f"{int(np.mean(scan['dimension'])):,} |",
            f"| Fraction of the CAS space | {fractions.max():.3f}% | {fractions.mean():.3f}% |",
            "",
            f"**{inside} of {len(errors)}** points are inside chemical accuracy. "
            f"Hartree-Fock misses by up to {hartree_fock.max():.0f} mHa over the same "
            "curve, which is the multireference character HI-VQE is there to capture.",
            "",
        ]
    elif singles:
        best = min(singles, key=lambda run: abs(run.get("error_millihartree", 1e9)))
        lines += [
            f"Best single point: **r = {best.get('bond_angstrom', 0):.3f} A**, error "
            f"**{best.get('error_millihartree', float('nan')):+.4f} mHa** using "
            f"**{best.get('dimension', 0):,}** of {best.get('full_dimension', 0):,} "
            f"determinants ({100 * best.get('subspace_fraction', 0):.3f}% of the space).",
            "",
        ]
    else:
        lines += ["No runs found. Nothing to report.", ""]
    return lines


def section_single_points(singles) -> list[str]:
    if not singles:
        return []
    lines = [
        "## Single-point runs",
        "",
        "| r (A) | configuration | E(HI-VQE) (Ha) | error (mHa) | "
        "error +PT2 (mHa) | determinants | "
        "% of space | correlation | <S^2> | iters | s | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in sorted(singles, key=lambda r: (r.get("bond_angstrom", 0), r["_file"])):
        settings = run.get("settings", {})
        configuration = settings.get("simulator", "?")
        if not settings.get("use_expansion", True):
            configuration += " + no expansion"
        if settings.get("readout_error") or settings.get("depolarizing_error"):
            configuration += " + noise"
        lines.append(
            f"| {_fmt(run.get('bond_angstrom'), '.3f')} | {configuration} | "
            f"{_fmt(run.get('energy'), '.9f')} | "
            f"{_fmt(run.get('error_millihartree'), '+.4f')} | "
            f"{_fmt(run.get('error_pt2_millihartree'), '+.4f') if run.get('pt2_determinants') else '--'} | "
            f"{run.get('dimension', 0):,} | "
            f"{_fmt(100 * run.get('subspace_fraction', 0), '.3f')}% | "
            f"{_fmt(100 * run.get('correlation_recovered', 0), '.2f')}% | "
            f"{_fmt(run.get('spin_squared'), '.2e')} | {run.get('iterations', 0)} | "
            f"{_fmt(run.get('seconds'), '.0f')} | {run.get('verdict', '')} |"
        )
    lines.append("")
    return lines


def section_scan(scans, classical) -> list[str]:
    if not scans:
        return []
    lines = []
    for scan in scans:
        method = scan.get("reference_method", "CASCI")
        lines += [
            "## Dissociation curve",
            "",
            f"One Li-S bond stretched, the other held at equilibrium. "
            f"{len(scan['distances'])} geometries.",
            "",
            f"| r (A) | E(HF) | E(HI-VQE) | E({method}) | error (mHa) | "
            "HF error (mHa) | determinants | % of space |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for index, distance in enumerate(scan["distances"]):
            hartree_fock_error = (
                abs(scan["hartree_fock"][index] - scan["reference"][index]) * 1000
            )
            lines.append(
                f"| {distance:.3f} | {scan['hartree_fock'][index]:.6f} | "
                f"{scan['hivqe'][index]:.6f} | {scan['reference'][index]:.6f} | "
                f"{scan['error_millihartree'][index]:+.4f} | {hartree_fock_error:.1f} | "
                f"{scan['dimension'][index]:,} | "
                f"{100 * scan['subspace_fraction'][index]:.3f}% |"
            )
        lines.append("")

        distances = np.array(scan["distances"])
        reference = np.array(scan["reference"])
        hivqe = np.array(scan["hivqe"])
        minimum = int(np.argmin(reference))
        lines += [
            "### Derived quantities",
            "",
            "| Quantity | HI-VQE | " + method + " |",
            "|---|---|---|",
            f"| Minimum on this grid | {distances[int(np.argmin(hivqe))]:.3f} A | "
            f"{distances[minimum]:.3f} A |",
        ]
        if minimum == len(reference) - 1:
            # The curve is still falling at the last point, so there is no well
            # to measure a dissociation energy out of. Reporting one anyway
            # would print a confident 0.00 kcal/mol, which is the most
            # misleading thing this section could do.
            lines += [
                "",
                f"**No dissociation energy is reported.** The {method} minimum is at "
                f"{distances[minimum]:.3f} A, the last point on this grid, so the "
                "curve has no bracketed well. Extend the scan to shorter and longer "
                "bond lengths before reading a binding energy off it.",
                "",
            ]
        else:
            dissociation = (reference[-1] - reference[minimum]) * 627.509474
            hivqe_dissociation = (hivqe[-1] - hivqe[minimum]) * 627.509474
            lines += [
                f"| Dissociation energy to the last point | "
                f"{hivqe_dissociation:.2f} kcal/mol | {dissociation:.2f} kcal/mol |",
                f"| Error in that dissociation energy | "
                f"{abs(hivqe_dissociation - dissociation):.3f} kcal/mol | -- |",
                "",
                "A dissociation energy is a *difference*, so the systematic part of "
                "the error cancels: the number above is a stricter test of "
                "consistency than any single total energy.",
                "",
            ]
    if classical:
        lines += section_classical(classical)
    return lines


def section_classical(classical) -> list[str]:
    rows = sorted(
        (float(key), value)
        for key, value in classical.items()
        if not key.startswith("_")
    )
    if not rows:
        return []
    lines = [
        "### Classical baselines in the same active space",
        "",
        "| r (A) | MP2 error | CCSD error | CCSD(T) error | T1 diagnostic |",
        "|---|---|---|---|---|",
    ]
    for distance, entry in rows:
        exact = entry.get("casci_total")
        def error(key):
            if exact is None or key not in entry:
                return "--"
            return f"{(entry[key] - exact) * 1000:+.2f} mHa"

        lines.append(
            f"| {distance:.3f} | {error('mp2_total')} | {error('ccsd_total')} | "
            f"{error('ccsd_t_total')} | {_fmt(entry.get('t1_diagnostic'), '.4f')} |"
        )
    lines += [
        "",
        "T1 above roughly 0.02 means the reference determinant no longer dominates, "
        "so CCSD(T) stops being a gold standard there. That is the regime a breaking "
        "Li-S bond enters, and it is why the reference in this study is CASCI.",
        "",
    ]
    return lines


def section_ablation(singles) -> list[str]:
    if not singles:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in singles:
        settings = run.get("settings", {})
        key = settings.get("simulator", "?")
        if not settings.get("use_expansion", True):
            key += " + no expansion"
        grouped.setdefault(key, []).append(run)
    if len(grouped) < 2:
        return []
    lines = [
        "## Ablation: what did the quantum layer contribute?",
        "",
        "The same loop with one half removed at a time. `none` is the classical "
        "selected-CI control -- no circuit at all -- and `no expansion` removes the "
        "classical single/double step so only the sampler proposes configurations.",
        "",
        "| configuration | runs | mean error (mHa) | mean determinants | "
        "first-iteration error (mHa) |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(grouped):
        runs = grouped[key]
        errors = [abs(run.get("error_millihartree", np.nan)) for run in runs]
        first = [
            abs(run["history"][0]["error_hartree"]) * 1000
            for run in runs
            if run.get("history")
        ]
        lines.append(
            f"| {key} | {len(runs)} | {_fmt(np.nanmean(errors), '.4f')} | "
            f"{int(np.mean([run.get('dimension', 0) for run in runs])):,} | "
            f"{_fmt(np.mean(first) if first else np.nan, '.2f')} |"
        )
    lines += [
        "",
        "Read the last column first. The quantum sampler's contribution is a *better "
        "starting subspace*, which is exactly what the method asks of it: the circuit "
        "proposes configurations and the classical eigensolver prices them. Where the "
        "final energies converge to the same place, the honest statement is that the "
        "classical expansion was sufficient for this system at this size -- and this "
        "table is what makes that statement checkable rather than assumed.",
        "",
    ]
    return lines


def section_method(spec: MoleculeSpec, singles, cache_root: str) -> list[str]:
    lines = [
        "## Method and settings",
        "",
        "### The active space",
        "",
        f"{spec.name} has 22 electrons in {spec.basis.upper()}. "
        f"CAS({spec.n_active_electrons}e,{spec.n_active_orbitals}o) therefore freezes "
        "the 5 lowest orbitals -- the sulfur 1s, 2s and 2p shell, which sit near -91, "
        "-9 and -6.7 Ha while nothing else in the molecule is below -3 Ha -- and drops "
        f"the 2 highest virtuals. `prepare` measures that rather than assuming it and "
        "refuses to build if the frozen orbitals are not the sulfur core.",
        "",
    ]
    caches = sorted(Path(cache_root).glob("li2s_r*.json"))
    caches = [path for path in caches if not path.name.endswith(".validated.json")]
    if caches:
        try:
            cached = load_cache(caches[0])
            lines += [
                "| | |",
                "|---|---|",
                f"| Geometry | linear {spec.terminal_atom}-{spec.central_atom}-"
                f"{spec.terminal_atom} |",
                f"| Basis | {cached.metadata['basis']} |",
                f"| Orbitals | {cached.metadata['orbitals']} |",
                f"| Frozen core | {cached.metadata['n_core_orbitals']} orbitals, "
                f"core-active gap {cached.metadata['core_active_gap_hartree']:.2f} Ha |",
                f"| Qubits | {cached.metadata['n_qubits']} |",
                f"| Full CAS dimension | {cached.metadata['full_cas_determinants']:,} |",
                "",
            ]
        except Exception:
            pass

    if singles:
        settings = singles[0].get("settings", {})
        metrics = singles[0].get("circuit_metrics", {})
        lines += [
            "### The sampling circuit",
            "",
            "A hardware-efficient unitary cluster Jastrow: nearest-neighbour Givens "
            "rotations (`XXPlusYY`) inside each spin block, a diagonal number-number "
            "Jastrow layer (`RZ`, `RZZ`) that correlates the two blocks, and a closing "
            "Givens network. Every gate commutes with both spin-resolved number "
            "operators, so on a noiseless simulator every shot is a valid "
            "configuration and the leakage is exactly zero rather than merely small.",
            "",
            "| | |",
            "|---|---|",
        ]
        for key, label in (
            ("qubits", "Qubits"),
            ("depth", "Circuit depth"),
            ("two_qubit_gates", "Two-qubit gates"),
            ("parameters", "Parameters"),
            ("transpiled_depth", "Depth on a linear chain"),
            ("transpiled_two_qubit_gates", "CZ on a linear chain"),
        ):
            if key in metrics:
                lines.append(f"| {label} | {metrics[key]:,} |")
        lines += [
            "",
            "### HI-VQE settings",
            "",
            "| Setting | Value | What it controls |",
            "|---|---|---|",
            f"| `--simulator` | {settings.get('simulator')} | which simulator samples "
            "the circuit |",
            f"| `--max-determinants` | {settings.get('max_determinants'):,} | cap on "
            "the subspace after amplitude screening |",
            f"| `--expansion` | {settings.get('expansion')} | candidate configurations "
            "offered by the classical single/double step each iteration |",
            f"| `--expansion-references` | {settings.get('expansion_references')} | "
            "leading configurations the excitations are generated from (1 reproduces "
            "the paper's rule exactly) |",
            f"| `--amplitude-threshold` | {settings.get('amplitude_threshold'):g} | "
            "below this marginal weight a configuration is dropped |",
            f"| `--shots` | {settings.get('shots'):,} | measurements per circuit "
            "evaluation |",
            f"| `--reps` | {settings.get('reps')} | cluster-Jastrow layers |",
            f"| `--optimizer` | {settings.get('optimizer')} | how theta is moved |",
            "",
        ]
    return lines


def section_provenance(spec: MoleculeSpec, singles) -> list[str]:
    versions = _environment()
    lines = [
        "## Provenance and honest boundaries",
        "",
        "### What is comparable with arXiv:2503.06292, and what is not",
        "",
        "| | Paper | Here |",
        "|---|---|---|",
        f"| Active space | CAS(12e,12o)/STO-3G, 24 qubits, 853,776 determinants | "
        f"identical |",
        "| Reference method | CASCI | identical |",
        "| Algorithm | HI-VQE: sample, project, diagonalise, expand, optimise | "
        "identical |",
        "| Dissociation coordinate | \"removing a lithium atom\" | identical in kind |",
        "| Geometry | **not published** | linear Li-S-Li, this folder's own CASCI "
        "equilibrium |",
        "| Per-point energies | **not published** | reported in full above |",
        "| Circuit / ansatz | **not specified** | hardware-efficient unitary cluster "
        "Jastrow, described above |",
        "| Hardware | NISQ device in the framing; simulation in the results | "
        "CPU simulation only |",
        "",
        "**Total energies are therefore not reproducible from that paper and are not "
        "claimed to be.** What is comparable, and what this report actually measures, "
        "is the error against our own CASCI at each geometry, the fraction of the "
        "determinant space needed to reach it, and the shape of the dissociation "
        "curve. Those are the quantities the paper's own figures report.",
        "",
        "### The variational guarantee",
        "",
        "Every energy here is the lowest eigenvalue of `P H P` for some projector `P`, "
        "which is an upper bound on the exact active-space ground state for any `P` "
        "whatsoever. A run reporting an energy *below* CASCI is not a better answer, "
        "it is a broken one, and it is reported as `INCONSISTENT` rather than as a "
        "result. Noise on the sampler can only make the proposed subspace worse -- it "
        "cannot bias the energy downward -- which is the property that makes the "
        "handover worth doing.",
        "",
        "### Environment",
        "",
        "| | |",
        "|---|---|",
    ]
    for key, value in versions.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"Workflow fingerprint: `{workflow_fingerprint()}`", ""]
    if singles:
        lines += [
            f"Hamiltonian cache fingerprint: "
            f"`{singles[0].get('cache_sha256', 'unknown')}`",
            "",
        ]
    return lines


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_markdown(spec: MoleculeSpec, cache_root: str, results_root: str) -> str:
    singles, scans, classical = load_results(results_root)
    lines: list[str] = []
    lines += section_header(spec, singles, scans)
    lines += section_headline(singles, scans)
    lines += section_scan(scans, classical)
    lines += section_single_points(singles)
    lines += section_ablation(singles)
    lines += section_method(spec, singles, cache_root)
    lines += section_provenance(spec, singles)
    return "\n".join(lines)


def _figures_as_data_uris(results_root: str) -> dict[str, str]:
    try:
        from visualize import build_figures
    except ImportError:
        return {}
    try:
        figures = build_figures(results_root)
    except ImportError:
        return {}
    encoded = {}
    for name, figure in figures.items():
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        encoded[name] = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
    return encoded


FIGURE_CAPTIONS = {
    "dissociation": "The dissociation curve, and the error against CASCI on a log "
    "axis. Hartree-Fock's failure at long bond length is the physics the method is "
    "for.",
    "convergence": "Error against CASCI per iteration, and how many determinants "
    "each iteration held.",
    "compression": "Accuracy bought per determinant. The vertical line is the full "
    "CAS space.",
    "ablation": "Mean error with the quantum layer (blue) and without it (orange).",
    "occupancies": "Active-orbital occupancies. Departure from 2 and 0 is the "
    "multireference character.",
    "cost": "Measurement settings per iteration, and the sampling circuit.",
}

HTML_TEMPLATE = """<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --ink: #1a1a1a; --muted: #5a6270; --line: #e3e6ea; --bg: #ffffff;
    --accent: #1f6feb; --soft: #f6f8fa;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ink: #e8eaed; --muted: #9aa4b2; --line: #2c313a; --bg: #14171c;
      --accent: #6ea8fe; --soft: #1b1f26;
    }}
  }}
  html {{ background: var(--bg); }}
  body {{
    margin: 0 auto; padding: 2.5rem 1.5rem 5rem; max-width: 60rem;
    background: var(--bg); color: var(--ink);
    font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 1.9rem; line-height: 1.2; margin: 0 0 .4rem; }}
  h2 {{ font-size: 1.3rem; margin: 2.6rem 0 .8rem; padding-top: .8rem;
        border-top: 1px solid var(--line); }}
  h3 {{ font-size: 1.05rem; margin: 1.8rem 0 .5rem; color: var(--muted); }}
  em {{ color: var(--muted); font-style: normal; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 13px; }}
  th, td {{ padding: .42rem .6rem; border-bottom: 1px solid var(--line);
            text-align: left; white-space: nowrap; }}
  th {{ background: var(--soft); font-weight: 600; }}
  td:first-child, th:first-child {{ white-space: normal; }}
  .scroll {{ overflow-x: auto; }}
  code {{ background: var(--soft); padding: .1rem .32rem; border-radius: 4px;
          font-size: .88em; }}
  figure {{ margin: 1.6rem 0; }}
  figure img {{ max-width: 100%; border: 1px solid var(--line); border-radius: 8px;
                background: #fff; }}
  figcaption {{ color: var(--muted); font-size: 13px; margin-top: .5rem; }}
  a {{ color: var(--accent); }}
  strong {{ font-weight: 650; }}
</style>
{body}
"""


def markdown_to_html(markdown: str, figures: dict[str, str]) -> str:
    """A deliberately small Markdown renderer for exactly what this file emits.

    Pulling in a Markdown library for six constructs would add a dependency to a
    workflow whose whole point is that it runs on a bare Colab CPU.
    """
    import html as html_module
    import re

    def inline(text: str) -> str:
        text = html_module.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text
        )
        return text

    out: list[str] = []
    lines = markdown.split("\n")
    index = 0
    inserted: set[str] = set()

    while index < len(lines):
        line = lines[index]
        if line.startswith("|") and index + 1 < len(lines) and set(
            lines[index + 1].replace("|", "").strip()
        ) <= set("-: "):
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            index += 2
            body = []
            while index < len(lines) and lines[index].startswith("|"):
                body.append(
                    [cell.strip() for cell in lines[index].strip("|").split("|")]
                )
                index += 1
            out.append('<div class="scroll"><table><thead><tr>')
            out += [f"<th>{inline(cell)}</th>" for cell in headers]
            out.append("</tr></thead><tbody>")
            for row in body:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table></div>")
            continue

        stripped = line.strip()
        if stripped.startswith("### "):
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            title = stripped[3:]
            out.append(f"<h2>{inline(title)}</h2>")
            for key, uri in figures.items():
                if key in inserted:
                    continue
                if _figure_belongs(key, title):
                    out.append(
                        f'<figure><img alt="{key}" src="{uri}">'
                        f"<figcaption>{inline(FIGURE_CAPTIONS.get(key, key))}"
                        "</figcaption></figure>"
                    )
                    inserted.add(key)
        elif stripped.startswith("# "):
            out.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif stripped.startswith("- "):
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(f"<li>{inline(lines[index].strip()[2:])}</li>")
                index += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif stripped:
            out.append(f"<p>{inline(stripped)}</p>")
        index += 1

    leftover = [key for key in figures if key not in inserted]
    if leftover:
        out.append("<h2>Further figures</h2>")
        for key in leftover:
            out.append(
                f'<figure><img alt="{key}" src="{figures[key]}">'
                f"<figcaption>{FIGURE_CAPTIONS.get(key, key)}</figcaption></figure>"
            )
    return HTML_TEMPLATE.format(title="Li2S at 24 qubits -- HI-VQE", body="\n".join(out))


def _figure_belongs(key: str, heading: str) -> bool:
    heading = heading.lower()
    if key.startswith("dissociation"):
        return "dissociation" in heading
    if key == "convergence":
        return "single-point" in heading
    if key == "compression":
        return "single-point" in heading
    if key == "ablation":
        return "ablation" in heading
    if key == "occupancies":
        return "single-point" in heading
    if key == "cost":
        return "method" in heading
    return False


def report_main(spec: MoleculeSpec, cache_root: str, results_root: str) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble results/ into a Markdown and a standalone HTML report."
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--no-figures", action="store_true", help="skip matplotlib entirely"
    )
    arguments = parser.parse_args()

    singles, scans, _ = load_results(results_root)
    if not singles and not scans:
        print(f"No results in {results_root}. Run `python run.py hivqe` first.")
        return 1

    target = Path(arguments.output or results_root)
    target.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(spec, cache_root, results_root)
    markdown_path = target / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"  wrote {markdown_path}")

    figures = {} if arguments.no_figures else _figures_as_data_uris(results_root)
    if not figures and not arguments.no_figures:
        print("  (matplotlib unavailable -- the HTML report will have no figures)")
    html_path = target / "report.html"
    html_path.write_text(markdown_to_html(markdown, figures), encoding="utf-8")
    print(f"  wrote {html_path}  ({html_path.stat().st_size / 1024:.0f} KB, "
          f"{len(figures)} figures embedded)")

    json_path = target / "report_data.json"
    json_path.write_text(
        json.dumps(
            {
                "generated": _datetime.datetime.now(
                    _datetime.timezone.utc
                ).isoformat(),
                "workflow_sha256": workflow_fingerprint(),
                "environment": _environment(),
                "single_points": singles,
                "scans": scans,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {json_path}")
    print("\nOpen report.html in a browser -- it is self-contained.")
    return 0
