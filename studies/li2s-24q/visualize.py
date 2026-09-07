#!/usr/bin/env python3
"""Charts. Every panel answers one question; none of them are decoration.

matplotlib is imported lazily and nothing in the physics or simulator path
touches it, so a machine without it loses the pictures and nothing else.

In Colab, `import visualize; visualize.show()` renders the same figures inline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from summary import load_results


CHEMICAL_ACCURACY_MHA = 1.6

# One palette, used consistently: the reference is always black, Hartree-Fock is
# always grey, HI-VQE is always the strong blue, and classical post-HF methods
# share the warm end. A reader who learns the colours on one figure keeps them.
COLOURS = {
    "reference": "#111111",
    "hivqe": "#1f6feb",
    "hartree_fock": "#8b949e",
    "mp2": "#d29922",
    "ccsd": "#db6d28",
    "ccsd_t": "#a371f7",
    "accuracy": "#2da44e",
    "quantum": "#1f6feb",
    "classical": "#bf5b04",
}


def _style() -> Any:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _accuracy_band(axis, label: bool = True) -> None:
    axis.axhline(
        CHEMICAL_ACCURACY_MHA,
        color=COLOURS["accuracy"],
        linestyle="--",
        linewidth=1.1,
        label="chemical accuracy (1.6 mHa)" if label else None,
    )


def _label(run: dict[str, Any]) -> str:
    settings = run.get("settings", {})
    parts = [settings.get("simulator", "?")]
    if not settings.get("use_expansion", True):
        parts.append("no expansion")
    if settings.get("readout_error") or settings.get("depolarizing_error"):
        parts.append("noisy")
    return " + ".join(parts)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def figure_convergence(runs: list[dict[str, Any]], plt):
    """Did it converge, and to what? The chart the workflow exists to produce."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for run in runs:
        history = run.get("history", [])
        if not history:
            continue
        iterations = [record["iteration"] for record in history]
        errors = [abs(record["error_hartree"]) * 1000 for record in history]
        name = f"r = {run.get('bond_angstrom', 0):.2f} A, {_label(run)}"
        axes[0].plot(iterations, errors, marker="o", markersize=3.5, label=name)
        axes[1].plot(
            iterations,
            [record["dimension"] for record in history],
            marker="o",
            markersize=3.5,
            label=name,
        )
    _accuracy_band(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("HI-VQE iteration")
    axes[0].set_ylabel("|E - E(CASCI)|  (mHa)")
    axes[0].set_title("Convergence to the exact active-space energy")
    axes[0].legend(fontsize=7)

    if runs:
        full = runs[0].get("full_dimension")
        if full:
            axes[1].axhline(
                full,
                color=COLOURS["reference"],
                linestyle=":",
                linewidth=1.1,
                label=f"full CAS ({full:,})",
            )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("HI-VQE iteration")
    axes[1].set_ylabel("determinants in the subspace")
    axes[1].set_title("How much of the space it actually needed")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    return figure


def figure_dissociation(scan: dict[str, Any], classical: dict[str, Any], plt):
    """The result: does the curve track CASCI everywhere, not just near the minimum?"""
    figure, axes = plt.subplots(
        2, 1, figsize=(7.6, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    distances = np.array(scan["distances"])
    reference = np.array(scan["reference"])
    method = scan.get("reference_method", "CASCI")

    axes[0].plot(
        distances,
        scan["hartree_fock"],
        color=COLOURS["hartree_fock"],
        marker="s",
        markersize=3.5,
        linewidth=1.2,
        label="Hartree-Fock",
    )
    for key, colour, label in (
        ("mp2_total", COLOURS["mp2"], "MP2"),
        ("ccsd_total", COLOURS["ccsd"], "CCSD"),
        ("ccsd_t_total", COLOURS["ccsd_t"], "CCSD(T)"),
    ):
        values = _classical_series(classical, distances, key)
        if values is not None:
            axes[0].plot(
                distances,
                values,
                color=colour,
                marker="^",
                markersize=3.0,
                linewidth=1.1,
                alpha=0.9,
                label=label,
            )
    axes[0].plot(
        distances,
        reference,
        color=COLOURS["reference"],
        linewidth=2.0,
        label=f"{method} (exact in this active space)",
    )
    axes[0].plot(
        distances,
        scan["hivqe"],
        color=COLOURS["hivqe"],
        marker="o",
        markersize=4.5,
        linestyle="none",
        label="HI-VQE",
    )
    qubits = int(scan.get("n_qubits", 24))
    axes[0].set_ylabel("total energy (Ha)")
    axes[0].set_title(
        f"Li-S dissociation, {qubits} qubits, "
        f"{scan.get('full_dimension', 0):,} determinants in the full CAS"
    )
    axes[0].legend(fontsize=8)

    errors = np.abs(np.array(scan["error_millihartree"]))
    hartree_fock_errors = (
        np.abs(np.array(scan["hartree_fock"]) - reference) * 1000
    )
    axes[1].plot(
        distances,
        hartree_fock_errors,
        color=COLOURS["hartree_fock"],
        marker="s",
        markersize=3.5,
        label="Hartree-Fock",
    )
    for key, colour, label in (
        ("mp2_total", COLOURS["mp2"], "MP2"),
        ("ccsd_total", COLOURS["ccsd"], "CCSD"),
        ("ccsd_t_total", COLOURS["ccsd_t"], "CCSD(T)"),
    ):
        values = _classical_series(classical, distances, key)
        if values is not None:
            axes[1].plot(
                distances,
                np.abs(np.array(values) - reference) * 1000,
                color=colour,
                marker="^",
                markersize=3.0,
                alpha=0.9,
                label=label,
            )
    axes[1].plot(
        distances,
        np.maximum(errors, 1e-4),
        color=COLOURS["hivqe"],
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label="HI-VQE",
    )
    _accuracy_band(axes[1])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("stretched Li-S distance (A)")
    axes[1].set_ylabel(f"|E - E({method})|  (mHa)")
    axes[1].set_title("Error against the exact active-space energy")
    axes[1].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return figure


def _classical_series(
    classical: dict[str, Any], distances: np.ndarray, key: str
) -> list[float] | None:
    if not classical:
        return None
    values = []
    for distance in distances:
        entry = classical.get(f"{distance:.3f}")
        if not entry or key not in entry:
            return None
        values.append(float(entry[key]))
    return values


def figure_energy(runs: list[dict[str, Any]], plt):
    """Where did the energy actually go, in hartree rather than in error?

    One geometry only. Total energies at different bond lengths differ by far
    more than the correlation energy being recovered, so drawing several on one
    axis flattens the thing the figure exists to show.
    """
    by_distance: dict[float, list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("history"):
            by_distance.setdefault(round(float(run.get("bond_angstrom", 0.0)), 4), []).append(run)
    if not by_distance:
        return None
    bond = max(sorted(by_distance), key=lambda key: len(by_distance[key]))
    chosen = by_distance[bond]

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    hartree_fock = float(chosen[0]["hartree_fock_energy"])
    reference = float(chosen[0]["reference_energy"])
    method = chosen[0].get("reference_method", "CASCI")

    axis.axhline(
        hartree_fock,
        color=COLOURS["hartree_fock"],
        linestyle="--",
        linewidth=1.2,
        label=f"Hartree-Fock  {hartree_fock:.6f} Ha",
    )
    axis.axhline(
        reference,
        color=COLOURS["reference"],
        linewidth=1.6,
        label=f"{method} (exact here)  {reference:.6f} Ha",
    )
    for run in chosen:
        history = run["history"]
        axis.plot(
            [record["iteration"] for record in history],
            [record["energy"] for record in history],
            marker="o",
            markersize=4.0,
            label=_label(run),
        )

    # The correlation energy is the gap between the two reference lines, so
    # annotating it makes "how much was recovered" a distance on the page.
    correlation = reference - hartree_fock
    axis.annotate(
        f"correlation energy\n{abs(correlation) * 1000:.1f} mHa",
        xy=(0.02, 0.5),
        xycoords="axes fraction",
        fontsize=8,
        color=COLOURS["hartree_fock"],
        va="center",
    )
    span = abs(correlation)
    axis.set_ylim(reference - 0.08 * span, hartree_fock + 0.08 * span)
    axis.set_xlabel("HI-VQE iteration")
    axis.set_ylabel("total energy (Ha)")
    axis.set_title(f"Where the energy went, r = {bond:.2f} A")
    axis.legend(fontsize=7)
    figure.tight_layout()
    return figure


def figure_compression(runs: list[dict[str, Any]], plt):
    """How few determinants bought chemical accuracy?"""
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    grouped: dict[str, list[tuple[int, float]]] = {}
    for run in runs:
        for record in run.get("history", []):
            grouped.setdefault(_label(run), []).append(
                (record["dimension"], abs(record["error_hartree"]) * 1000)
            )
    for label, points in grouped.items():
        points.sort()
        axis.plot(
            [p[0] for p in points],
            [max(p[1], 1e-4) for p in points],
            marker="o",
            markersize=3.5,
            linestyle="none",
            alpha=0.75,
            label=label,
        )
    full = next((run.get("full_dimension") for run in runs if run.get("full_dimension")), None)
    if full:
        axis.axvline(
            full,
            color=COLOURS["reference"],
            linestyle=":",
            linewidth=1.2,
            label=f"full CAS ({full:,})",
        )
    _accuracy_band(axis)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("determinants in the subspace")
    axis.set_ylabel("|E - E(CASCI)|  (mHa)")
    axis.set_title("Subspace compression: accuracy bought per determinant")
    axis.legend(fontsize=7)
    figure.tight_layout()
    return figure


def figure_ablation(runs: list[dict[str, Any]], plt):
    """What did the quantum sampler actually contribute?"""
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    by_label: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_label.setdefault(_label(run), []).append(run)
    labels = sorted(by_label)
    if not labels:
        return figure
    positions = np.arange(len(labels))
    errors = [
        float(np.mean([abs(r.get("error_millihartree", np.nan)) for r in by_label[k]]))
        for k in labels
    ]
    colours = [
        COLOURS["classical"] if key.startswith("none") else COLOURS["quantum"]
        for key in labels
    ]
    axis.barh(positions, np.maximum(errors, 1e-4), color=colours, height=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.axvline(
        CHEMICAL_ACCURACY_MHA,
        color=COLOURS["accuracy"],
        linestyle="--",
        linewidth=1.1,
        label="chemical accuracy",
    )
    axis.set_xscale("log")
    axis.set_xlabel("mean |E - E(CASCI)|  (mHa)")
    axis.set_title("Ablation: quantum proposals (blue) vs classical only (orange)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def figure_occupancies(runs: list[dict[str, Any]], plt):
    """Where the electrons actually sit, and how that changes on dissociation."""
    usable = [run for run in runs if run.get("occupancies_alpha")]
    if not usable:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    usable = sorted(usable, key=lambda run: run.get("bond_angstrom", 0.0))
    # One run per bond length. Several ablations at the same geometry would
    # otherwise be drawn as separate bars that are, by construction, the same
    # physical state -- which reads as a disagreement that is not there.
    by_distance: dict[float, dict[str, Any]] = {}
    for run in usable:
        by_distance.setdefault(round(float(run.get("bond_angstrom", 0.0)), 4), run)
    distinct = [by_distance[key] for key in sorted(by_distance)]
    picked = [distinct[0]] if len(distinct) == 1 else [distinct[0], distinct[-1]]
    width = 0.8 / len(picked)
    for index, run in enumerate(picked):
        total = np.array(run["occupancies_alpha"]) + np.array(run["occupancies_beta"])
        positions = np.arange(len(total)) + index * width
        axis.bar(
            positions,
            total,
            width=width,
            label=f"r = {run.get('bond_angstrom', 0):.2f} A",
            alpha=0.9,
        )
    axis.axhline(2.0, color=COLOURS["reference"], linewidth=0.8, linestyle=":")
    axis.set_xlabel("active orbital")
    axis.set_ylabel("occupancy")
    axis.set_title(
        "Active-orbital occupancies. Deviation from 2 and 0 is the multireference "
        "character"
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def figure_cost(runs: list[dict[str, Any]], plt):
    """The measurement argument, drawn to scale."""
    run = next((r for r in runs if r.get("circuit_metrics")), None)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    published = (run or {}).get("published_reference") or {}
    pauli = published.get("pauli_words_for_conventional_vqe", 15697)
    iterations = (run or {}).get("iterations", 1) or 1
    circuits = sum(record.get("circuit_runs", 0) for record in (run or {}).get("history", []))
    circuits = max(circuits, 1)

    axes[0].bar(
        ["conventional VQE\n(per iteration)", "HI-VQE\n(per iteration)"],
        [pauli, 1],
        color=[COLOURS["hartree_fock"], COLOURS["hivqe"]],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("distinct measurement settings")
    axes[0].set_title(f"Measurements per iteration ({pauli:,}x)")
    for index, value in enumerate([pauli, 1]):
        axes[0].text(index, value * 1.3, f"{value:,}", ha="center", fontsize=9)

    metrics = (run or {}).get("circuit_metrics", {})
    if metrics:
        names = ["qubits", "depth", "two_qubit_gates", "parameters"]
        labels = ["qubits", "depth", "2-qubit gates", "parameters"]
        values = [metrics.get(key, 0) for key in names]
        if "transpiled_two_qubit_gates" in metrics:
            labels.append("CZ (linear chain)")
            values.append(metrics["transpiled_two_qubit_gates"])
        axes[1].barh(labels, values, color=COLOURS["quantum"], height=0.6)
        axes[1].set_xscale("log")
        axes[1].set_title("Sampling circuit")
        for index, value in enumerate(values):
            axes[1].text(value * 1.1, index, f"{value:,}", va="center", fontsize=8)
    axes[1].set_xlabel("count")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def build_figures(results_root: str) -> dict[str, Any]:
    plt = _style()
    singles, scans, classical = load_results(results_root)
    figures: dict[str, Any] = {}
    if singles:
        figures["convergence"] = figure_convergence(singles, plt)
        energy = figure_energy(singles, plt)
        if energy is not None:
            figures["energy"] = energy
        figures["compression"] = figure_compression(singles, plt)
        figures["ablation"] = figure_ablation(singles, plt)
        occupancies = figure_occupancies(singles, plt)
        if occupancies is not None:
            figures["occupancies"] = occupancies
        figures["cost"] = figure_cost(singles, plt)
    for index, scan in enumerate(scans):
        key = "dissociation" if index == 0 else f"dissociation_{index}"
        figures[key] = figure_dissociation(scan, classical, plt)
    return figures


def plot_main(results_root: str) -> int:
    parser = argparse.ArgumentParser(description="Chart everything in results/.")
    parser.add_argument("--output", default=None, help="directory for the PNGs")
    arguments = parser.parse_args()

    try:
        figures = build_figures(results_root)
    except ImportError:
        print("matplotlib is not installed:  pip install matplotlib")
        return 1
    if not figures:
        print(f"No results in {results_root}. Run `python run.py hivqe` first.")
        return 1

    target = Path(arguments.output or (Path(results_root) / "figures"))
    target.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = target / f"{name}.png"
        figure.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    print(f"\n{len(figures)} figures in {target}")
    return 0


def show(results_root: str | None = None) -> None:
    """Render every figure inline. For Colab."""
    import matplotlib.pyplot as plt

    root = results_root or str(Path(__file__).resolve().parent / "results")
    figures = build_figures(root)
    if not figures:
        print(f"No results in {root}.")
        return
    for name in figures:
        print(f"--- {name} ---")
    plt.show()
