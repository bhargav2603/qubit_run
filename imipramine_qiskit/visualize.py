#!/usr/bin/env python3
"""Charts for ADAPT and VQE results, for Colab or for saving to disk.

Four questions worth a picture, and nothing else:

1. **Did it converge, and to what?** Error against CASCI per operator added, on a
   log axis, with the 1.6 mHa pass mark and the published 1.15 mHa drawn in.
   This is the chart the whole workflow exists to produce.
2. **Where did the energy actually go?** The same run in absolute hartree,
   between the Hartree-Fock and CASCI reference lines, so the correlation energy
   being recovered is visible as a distance rather than inferred from a number.
3. **Why did it stop?** The largest remaining pool gradient per iteration against
   the threshold that ends the loop. A run that stopped on the operator cap looks
   completely different here from one that converged, and the distinction matters
   more than the final energy.
4. **What did it choose?** The selected excitations by rotation angle, split into
   singles and doubles.

Design notes, since charts are easy to get subtly wrong:

* Two categorical colors are used anywhere in this file (singles vs doubles), and
  that pair was checked rather than eyeballed -- worst-case colour-vision-
  deficient separation dE 24.7, normal-vision 33.6, both far above the floors.
* Reference and threshold lines are dashed; data is solid. Gridlines are solid
  hairlines one step off the surface. Nothing is dual-axis.
* Values are labelled selectively -- the endpoint and the extreme -- never on
  every point.
* The figures paint their own light surface, so they stay legible whichever
  Colab theme is active.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Chart surface and ink. Light in both Colab themes on purpose: a PNG cannot
# adapt to the page, so it commits to one legible surface instead of inheriting
# a background it cannot see.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots 1 and 2. Validated as a pair for colour-vision deficiency.
SERIES_SINGLE = "#2a78d6"
SERIES_DOUBLE = "#eb6834"
# Status colours are reserved for state and never used as a series.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

CHEMICAL_ACCURACY_MHA = 1.6


def _style() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.grid": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_MUTED,
            "ytick.labelcolor": INK_MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )
    return plt


def _frame(ax: Any, title: str, subtitle: str = "", grid_axis: str = "y") -> None:
    """Recessive chrome: no box, hairline grid on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    # 22pt clears a 9.5pt subtitle sitting 2% of the axes height above the frame.
    # At 16 the two collide on short figures, which is most of the deck.
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=9.5, color=INK_SECONDARY, va="bottom",
        )


def _pad_category_axis(ax: Any, count: int, minimum: int) -> None:
    """Hold a minimum number of category slots in a fixed-height panel.

    Bar thickness encodes nothing -- only length does -- so it must not vary with
    how many rows happen to exist. Three operators in a panel sized for sixteen
    would otherwise render as three enormous slabs. The padding goes *below* the
    bars so the data still starts at the top edge, where reading starts.

    Only needed where panel height is fixed by a grid. A standalone figure sizes
    itself by row count instead, which is better: no dead space at all.
    """
    slots = max(count, minimum)
    top = count - 1 + 0.7
    ax.set_ylim(top - slots, top)


def _marker(ax: Any, x: float, y: float, color: str, size: float = 6.5) -> None:
    """A dot with a surface ring, so it stays readable where it crosses a line."""
    ax.plot(
        [x], [y], marker="o", markersize=size, color=color,
        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=6, linestyle="none",
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


COMPARISON_MIN_SLOTS = 6


def _comparison_figsize(rows: int) -> tuple[float, float]:
    """Height from the slot count, so bar pitch stays constant as runs accumulate.

    Paired with `_pad_category_axis` at the same minimum: the figure grows one
    row at a time, and below the minimum it holds its height and leaves the
    empty slots visible. A leaderboard with room to fill is honest; two bars
    stretched to fill the panel are not.
    """
    return (9.0, 0.42 * max(rows, COMPARISON_MIN_SLOTS) + 2.0)


def load_runs(results: str | Path) -> list[dict[str, Any]]:
    """Every readable result JSON, newest schema only, ADAPT runs first."""
    directory = Path(results)
    if not directory.is_dir():
        return []
    runs = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["vqe"], payload["ansatz"]
        except (OSError, ValueError, KeyError):
            continue
        payload["_name"] = path.name
        payload["_is_adapt"] = bool(payload.get("adapt"))
        runs.append(payload)
    runs.sort(key=lambda run: (not run["_is_adapt"], run["_name"]))
    return runs


def _trajectory(run: dict[str, Any]) -> tuple[list[int], list[float], list[float]]:
    """(operators added, energy, largest remaining pool gradient) including the start.

    Iteration 0 is the bare Hartree-Fock determinant, which is where every ADAPT
    run begins and is what makes the first step's size meaningful.
    """
    history = run["adapt"]["history"]
    counts = [0] + [entry["iteration"] for entry in history]
    energies = [run["vqe"]["hartree_fock_energy_hartree"]] + [
        entry["energy_hartree"] for entry in history
    ]
    gradients = [entry["pool_gradient"] for entry in history]
    return counts, energies, gradients


# --------------------------------------------------------------------------
# The four charts
# --------------------------------------------------------------------------


def plot_convergence(run: dict[str, Any], ax: Any) -> None:
    """Error against CASCI per operator added, log axis. The headline chart."""
    counts, energies, _ = _trajectory(run)
    reference = run["vqe"]["reference_energy_hartree"]
    # Absolute value: a converged run sits above CASCI by the variational
    # principle, but the last step can land a hair below it numerically, and a
    # log axis has no opinion about sign.
    errors = [max(abs(1000.0 * (energy - reference)), 1e-6) for energy in energies]

    published = (run.get("published_reference") or {}).get("statevector_error_millihartree")

    # Bound the axis to the story: the run's own range, the pass mark, and the
    # published anchor. Anchoring the band at zero instead would open three empty
    # decades below the data and squash every real difference into the top inch.
    interesting = [value for value in errors if value > 1e-9]
    floor = min([*interesting, CHEMICAL_ACCURACY_MHA, float(published or CHEMICAL_ACCURACY_MHA)])
    ceiling = max(errors)
    ax.set_yscale("log")
    ax.set_ylim(floor / 3.0, ceiling * 3.0)

    ax.axhspan(floor / 3.0, CHEMICAL_ACCURACY_MHA, color=STATUS_GOOD, alpha=0.07, zorder=0)
    ax.axhline(CHEMICAL_ACCURACY_MHA, color=STATUS_GOOD, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(
        counts[-1], CHEMICAL_ACCURACY_MHA * 1.15, "chemical accuracy  1.6 mHa",
        color=STATUS_GOOD, fontsize=9, ha="right", va="bottom", zorder=7,
    )
    if published:
        ax.axhline(float(published), color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)
        ax.text(
            counts[0], float(published) * 0.87, f"published  {float(published):.2f} mHa",
            color=INK_SECONDARY, fontsize=9, ha="left", va="top", zorder=7,
        )

    ax.plot(counts, errors, color=SERIES_SINGLE, linewidth=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)
    _marker(ax, counts[-1], errors[-1], SERIES_SINGLE)

    final = errors[-1]
    verified = final <= CHEMICAL_ACCURACY_MHA
    ax.annotate(
        f"{final:,.3f} mHa",
        xy=(counts[-1], final), xytext=(-8, 12), textcoords="offset points",
        fontsize=10, fontweight="bold",
        color=STATUS_GOOD if verified else INK_PRIMARY,
        ha="right", zorder=8,
    )

    ax.set_xlabel("operators in the ansatz")
    ax.set_ylabel("| energy − CASCI |   (mHa)")
    _frame(
        ax,
        "Convergence to chemical accuracy",
        f"{run['ansatz'].get('pool', '?')} pool · "
        f"{'inside' if verified else 'outside'} the 1.6 mHa band",
    )


def plot_energy(run: dict[str, Any], ax: Any) -> None:
    """The same run in absolute hartree, between the two reference energies."""
    counts, energies, _ = _trajectory(run)
    hartree_fock = run["vqe"]["hartree_fock_energy_hartree"]
    reference = run["vqe"]["reference_energy_hartree"]

    ax.axhline(hartree_fock, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(reference, color=STATUS_GOOD, linewidth=1.2, linestyle="--", zorder=2)
    ax.fill_between(counts, energies, reference, color=SERIES_SINGLE, alpha=0.10, zorder=1)
    ax.plot(counts, energies, color=SERIES_SINGLE, linewidth=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)
    _marker(ax, counts[-1], energies[-1], SERIES_SINGLE)

    # Both energies are negative and CASCI is the lower one, so the correlation
    # energy is HF - CASCI > 0 and the fraction recovered is how far the run
    # descended from HF as a share of that. Writing it the other way round is how
    # this line first came out at -96%.
    span = hartree_fock - reference
    # Offset in points, not data units: at data coordinates the reference line
    # runs straight through its own label.
    # Hartree-Fock is the top of this axis, so its label goes *below* the line and
    # at the right edge, where the curve has descended away from it. Above the
    # line is the subtitle's space.
    ax.annotate("Hartree-Fock", xy=(counts[-1], hartree_fock), xytext=(-4, -6),
                textcoords="offset points", color=INK_SECONDARY, fontsize=9,
                va="top", ha="right", zorder=7)
    ax.annotate("CASCI (exact in this active space)", xy=(counts[0], reference),
                xytext=(4, -6), textcoords="offset points", color=STATUS_GOOD,
                fontsize=9, va="top", ha="left", zorder=7)

    recovered = 100.0 * (hartree_fock - energies[-1]) / span if span else 0.0
    _frame(
        ax,
        "Where the energy went",
        f"{recovered:.2f}% of the {1000.0 * span:,.1f} mHa correlation energy recovered",
    )
    ax.set_xlabel("operators in the ansatz")
    ax.set_ylabel("energy  (Ha)")


def plot_gradients(run: dict[str, Any], ax: Any) -> None:
    """Largest remaining pool gradient against the threshold that ends the loop."""
    counts, _, gradients = _trajectory(run)
    tolerance = run["adapt"]["gradient_tolerance"]
    converged = run["adapt"]["converged"]
    iterations = counts[1:]

    ax.axhline(tolerance, color=STATUS_GOOD if converged else STATUS_CRITICAL,
               linewidth=1.2, linestyle="--", zorder=2)
    ax.text(
        iterations[-1] if iterations else 0, tolerance * 1.3,
        f"stop below {tolerance:g}", fontsize=9, ha="right", va="bottom",
        color=STATUS_GOOD if converged else STATUS_CRITICAL, zorder=7,
    )
    ax.plot(iterations, gradients, color=SERIES_SINGLE, linewidth=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)
    if iterations:
        _marker(ax, iterations[-1], gradients[-1], SERIES_SINGLE)

    final = run["adapt"]["final_pool_gradient_max"]
    _frame(
        ax,
        "Why it stopped",
        (f"converged — nothing left in the pool worth adding"
         if converged else
         f"hit the operator cap with {final:.1e} still on the table"),
    )
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("largest pool gradient")


def plot_operators(run: dict[str, Any], ax: Any, limit: int = 16) -> None:
    """Selected excitations by rotation angle, split into singles and doubles."""
    labels = run["ansatz"]["operators"]
    angles = run["ansatz"]["angles"]
    ranks = [entry["rank"] for entry in run["adapt"]["history"]]

    rows = sorted(
        zip(labels, angles, ranks), key=lambda row: abs(row[1]), reverse=True
    )[:limit]
    rows.reverse()
    names = [row[0] for row in rows]
    sizes = [abs(row[1]) for row in rows]
    colors = [SERIES_SINGLE if row[2] == 1 else SERIES_DOUBLE for row in rows]

    positions = range(len(rows))
    # height < 1 leaves the surface gap that separates neighbouring bars; no
    # stroke is drawn around them.
    ax.barh(list(positions), sizes, height=0.62, color=colors, zorder=3)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(names, fontsize=8.5)
    # Keep a floor on the number of slots so a short run does not inflate its
    # bars to fill the panel. Bar thickness should say nothing; only length does.
    _pad_category_axis(ax, len(rows), minimum=8)

    # Label the extreme only -- a number beside every bar goes unread.
    if rows:
        ax.text(sizes[-1], len(rows) - 1, f"  {sizes[-1]:.3f}", va="center",
                fontsize=9, color=INK_SECONDARY, zorder=7)

    from matplotlib.patches import Patch

    n_singles = sum(1 for _, _, rank in rows if rank == 1)
    # Only the ranks actually present. A `pair` pool has no singles at all, and a
    # legend entry for a colour that appears nowhere invites a hunt for it.
    handles = [
        Patch(facecolor=color, label=label)
        for rank, color, label in (
            (1, SERIES_SINGLE, "single excitation"),
            (2, SERIES_DOUBLE, "double excitation"),
        )
        if any(row[2] == rank for row in rows)
    ]
    if len(handles) > 1:
        ax.legend(handles=handles, loc="lower right", fontsize=9,
                  labelcolor=INK_SECONDARY)
    _frame(
        ax,
        "What ADAPT chose",
        f"top {len(rows)} of {len(labels)} operators by |angle| · "
        f"{n_singles} singles, {len(rows) - n_singles} doubles",
        grid_axis="x",
    )
    ax.set_xlabel("|rotation angle|  (radians)")


def plot_comparison(runs: list[dict[str, Any]], ax: Any) -> None:
    """Every run in results/, by distance from CASCI. One measure, one colour."""
    rows = []
    for run in runs:
        error = run["vqe"].get("error_millihartree")
        if error is None:
            continue
        ansatz = run["ansatz"]
        name = (
            f"adapt/{ansatz['pool']}" if ansatz.get("pool")
            else f"hea-L{ansatz.get('layers', '?')}"
        )
        rows.append((f"{name}  ({ansatz.get('parameters', '?')}p)", abs(error)))
    rows.sort(key=lambda row: row[1], reverse=True)
    if not rows:
        return

    names = [row[0] for row in rows]
    values = [max(row[1], 1e-6) for row in rows]
    positions = range(len(rows))
    ax.barh(list(positions), values, height=0.55, color=SERIES_SINGLE, zorder=3)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(names, fontsize=9.5)
    _pad_category_axis(ax, len(rows), minimum=COMPARISON_MIN_SLOTS)

    ax.axvline(CHEMICAL_ACCURACY_MHA, color=STATUS_GOOD, linewidth=1.2,
               linestyle="--", zorder=4)
    # In the empty slots below the bars: above the top bar is where the title
    # lives, and on top of a bar the green is unreadable against the fill.
    ax.annotate(" chemical accuracy 1.6 mHa", xy=(CHEMICAL_ACCURACY_MHA, ax.get_ylim()[0]),
                xytext=(3, 6), textcoords="offset points", color=STATUS_GOOD,
                fontsize=9, va="bottom", ha="left", zorder=7)

    best = min(range(len(values)), key=lambda index: values[index])
    ax.text(values[best], best, f"  {values[best]:,.3f}", va="center",
            fontsize=9.5, fontweight="bold", color=INK_PRIMARY, zorder=7)

    ax.set_xscale("log")
    ax.set_xlabel("| energy − CASCI |   (mHa, log scale)")
    _frame(ax, "Every run so far", f"{len(rows)} results · lower is better", grid_axis="x")


# --------------------------------------------------------------------------
# The presentation deck: five charts that argue rather than report
#
# The four charts above answer "did this run work". These five answer "should
# anyone care", which is a different question and needs different pictures.
# Together they defend one sentence:
#
#   ADAPT reproduces exact classical chemistry on a real tricyclic antidepressant
#   at 12 qubits, for N times the published gate cost, while classical still wins
#   outright at this size -- and here is where that flips.
#
# Every chart below defends one clause. A chart that defends none was cut.
# --------------------------------------------------------------------------


PUBLISHED_GATES_RAW = 1440
PUBLISHED_GATES_OPTIMIZED = 244
PUBLISHED_ERROR_MHA = 1.15

# Colour discipline carries over: classical and quantum are the two categorical
# slots, and they are the same validated pair used for singles and doubles.
CLASSICAL = SERIES_SINGLE
QUANTUM = SERIES_DOUBLE


def load_classical(results: str | Path) -> dict[str, Any] | None:
    """`results/classical.json` if `run.py classical` has been run, else None.

    Every chart that uses this degrades to quantum-only rather than failing.
    Classical baselines need PySCF, so a Windows machine legitimately will not
    have them, and a deck that refuses to render there would be useless exactly
    where it is most often assembled.
    """
    path = Path(results) / "classical.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if payload.get("methods") else None


def _error_mha(run: dict[str, Any]) -> float:
    return abs(float(run["vqe"].get("error_millihartree", float("nan"))))


def _run_label(run: dict[str, Any]) -> str:
    ansatz = run["ansatz"]
    if ansatz.get("pool"):
        return f"ADAPT/{ansatz['pool']}"
    return f"HEA L{ansatz.get('layers', '?')}"


def gate_trajectory(
    run: dict[str, Any], points: int = 9, optimization_level: int = 3
) -> list[tuple[int, int, float]]:
    """(operators, two-qubit gates, error mHa) sampled along a finished ADAPT run.

    This is the only expensive thing in this file -- one transpile per point, a
    second or two each -- so it samples rather than walking all forty. Nine
    points is enough to draw a curve and cheap enough to sit in a notebook cell.

    Sampling is log-spaced. The interesting structure is at the short end, where
    a handful of operators buys most of the correlation energy, and a linear
    sample would spend most of its budget on the flat tail.
    """
    from adapt_runtime import circuit_cost, circuit_from_result

    history = (run.get("adapt") or {}).get("history") or []
    if not history:
        return []
    reference = float(run["vqe"]["reference_energy_hartree"])
    total = len(history)

    indices = sorted({
        max(1, min(total, int(round(total ** (step / (points - 1))))))
        for step in range(points)
    }) if points > 1 else [total]
    if total not in indices:
        indices.append(total)

    trajectory = []
    for count in indices:
        cost = circuit_cost(circuit_from_result(run, count), optimization_level)
        error = abs(1000.0 * (history[count - 1]["energy_hartree"] - reference))
        trajectory.append((count, cost["two_qubit_gates"], error))
    return trajectory


def plot_pareto(
    runs: list[dict[str, Any]], ax: Any, trajectories: dict[str, list] | None = None
) -> None:
    """Accuracy against circuit cost, log-log. The chart that makes the argument.

    Error alone flatters the method and gate count alone damns it; the pair is
    the only honest statement, and it is the standard figure in the field for
    exactly that reason. The published work is one point on the same axes, so
    "competitive" or "not" is read off rather than asserted.

    Down-and-left is better. The elbow is the operator count worth shipping:
    past it the curve runs flat and every additional operator buys gates without
    buying accuracy, which is precisely the trade a reader cannot see from a
    convergence plot.
    """
    trajectories = trajectories or {}
    plotted = False

    for run in runs:
        trajectory = trajectories.get(run["_name"])
        if not trajectory:
            continue
        counts = [point[0] for point in trajectory]
        gates = [point[1] for point in trajectory]
        errors = [max(point[2], 1e-4) for point in trajectory]
        ax.plot(gates, errors, color=QUANTUM, linewidth=1.6, zorder=4,
                marker="o", markersize=3.5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, label=f"{_run_label(run)} (grown)")
        plotted = True
        # Label only the ends: where it started and where it stopped.
        for index in (0, len(trajectory) - 1):
            count = counts[index]
            ax.annotate(f"{count} op" + ("s" if count != 1 else ""),
                        xy=(gates[index], errors[index]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8.5, color=INK_SECONDARY, zorder=7)

    for run in runs:
        # A run with a measured trajectory is already on the chart at its own
        # endpoint. Plotting the stored count too would put the same run at two
        # x positions whenever the result predates a synthesis change -- which
        # looks like an inconsistency rather than the improvement it is. That
        # comparison belongs on the gate chart, where it is labelled.
        if trajectories.get(run["_name"]):
            continue
        circuit = run.get("circuit") or {}
        gates = circuit.get("two_qubit_gates")
        if not gates:
            continue
        _marker(ax, gates, max(_error_mha(run), 1e-4), QUANTUM, size=8.0)
        ax.annotate(f"  {_run_label(run)}", xy=(gates, max(_error_mha(run), 1e-4)),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                    color=INK_SECONDARY, va="center", zorder=7)
        plotted = True

    _marker(ax, PUBLISHED_GATES_OPTIMIZED, PUBLISHED_ERROR_MHA, CLASSICAL, size=9.0)
    ax.annotate(
        "  published\n  (Helios-1 paper, pytket)",
        xy=(PUBLISHED_GATES_OPTIMIZED, PUBLISHED_ERROR_MHA),
        xytext=(8, -2), textcoords="offset points",
        fontsize=9, color=CLASSICAL, va="top", zorder=7,
    )

    ax.axhline(CHEMICAL_ACCURACY_MHA, color=STATUS_GOOD, linewidth=1.2,
               linestyle="--", zorder=3)
    ax.annotate(" chemical accuracy 1.6 mHa", xy=(0.0, CHEMICAL_ACCURACY_MHA),
                xycoords=("axes fraction", "data"), xytext=(2, 4),
                textcoords="offset points", color=STATUS_GOOD, fontsize=9, zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("two-qubit gates after transpilation   (log scale)")
    ax.set_ylabel("| energy − CASCI |   (mHa, log)")
    _frame(ax, "Accuracy costs gates",
           "down and left is better · the elbow is the ansatz worth shipping")
    if plotted:
        ax.legend(loc="upper right", fontsize=9)


def plot_method_ladder(
    runs: list[dict[str, Any]], classical: dict[str, Any] | None, ax: Any
) -> None:
    """Every method's distance from CASCI on one log axis, classical vs quantum.

    CASCI is the reference and therefore not a competitor -- it is the zero of
    this axis, not a bar on it. The competitors are MP2, CCSD and CCSD(T), and
    without them a quantum error has no scale.

    This chart is allowed to say the quantum method lost. If CCSD(T) lands left
    of ADAPT the honest reading is that this is a verification result at a size
    where classical is still better, which is a true and defensible thing to
    present -- and far better than a deck that quietly omitted the baseline.
    """
    rows: list[tuple[str, float, str]] = []

    if classical:
        reference = classical.get("reference_energy_hartree")
        if reference is None:
            reference = next(
                (run["vqe"]["reference_energy_hartree"] for run in runs), None
            )
        if reference is not None:
            rows.append(("Hartree–Fock", abs(
                1000.0 * (classical["hartree_fock_energy_hartree"] - reference)
            ), CLASSICAL))
            for name, entry in classical["methods"].items():
                rows.append((name, abs(
                    1000.0 * (entry["energy_hartree"] - reference)
                ), CLASSICAL))
    elif runs:
        vqe = runs[0]["vqe"]
        rows.append(("Hartree–Fock", abs(
            1000.0 * (vqe["hartree_fock_energy_hartree"] - vqe["reference_energy_hartree"])
        ), CLASSICAL))

    rows.append(("published ADAPT-GQE", PUBLISHED_ERROR_MHA, QUANTUM))
    for run in runs:
        error = _error_mha(run)
        if error == error:  # not NaN
            rows.append((f"{_run_label(run)}  ({run['ansatz'].get('parameters', '?')}p)",
                         error, QUANTUM))

    rows = [row for row in rows if row[1] > 0]
    rows.sort(key=lambda row: row[1], reverse=True)
    if not rows:
        return

    positions = list(range(len(rows)))
    values = [max(row[1], 1e-4) for row in rows]
    floor = min(values) / 4.0
    for position, (name, value, color) in zip(positions, rows):
        ax.plot([floor, value], [position, position], color=color,
                linewidth=1.4, alpha=0.35, zorder=3, solid_capstyle="butt")
        _marker(ax, value, position, color, size=7.5)
        ax.text(value, position, f"  {value:,.2f}", va="center", fontsize=9,
                color=INK_PRIMARY, zorder=7)

    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axvline(CHEMICAL_ACCURACY_MHA, color=STATUS_GOOD, linewidth=1.2,
               linestyle="--", zorder=4)
    ax.annotate(" 1.6 mHa", xy=(CHEMICAL_ACCURACY_MHA, -0.65),
                xytext=(3, 0), textcoords="offset points", color=STATUS_GOOD,
                fontsize=9, va="bottom", zorder=7)
    ax.set_xscale("log")
    ax.set_xlim(left=floor)
    ax.set_xlabel("| energy − CASCI |   (mHa, log scale) · lower is better")

    handles = [
        plt_line(CLASSICAL, "classical (same active space)"),
        plt_line(QUANTUM, "quantum ansatz"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    _frame(ax, "Classical and quantum, same Hamiltonian",
           "CASCI is the reference, so it is the zero of this axis", grid_axis="x")


def plt_line(color: str, label: str) -> Any:
    """A legend proxy. Defined here so the deck charts stay import-light."""
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, marker="o", linestyle="-",
                  markersize=6.5, linewidth=1.4, label=label)


def plot_cost(
    runs: list[dict[str, Any]], classical: dict[str, Any] | None, ax: Any
) -> None:
    """Wall-clock per method, log scale. The chart that admits classical wins.

    At CAS(6e,6o) the exact classical solve is a 400-determinant diagonalization
    -- milliseconds -- against minutes for the quantum simulation. Stating that
    yourself is what makes the accuracy comparison credible; leaving it for the
    audience to compute is not.

    The mean field is excluded, deliberately. Every method here needs it,
    including the quantum one -- `prepare` cannot build a Hamiltonian without it
    -- so charging it to the classical side alone would flatter the quantum
    numbers rather than test them.
    """
    rows: list[tuple[str, float, str]] = []
    if classical:
        for name, entry in classical["methods"].items():
            rows.append((name, max(float(entry["seconds"]), 1e-4), CLASSICAL))
    for run in runs:
        seconds = float(run.get("runtime_seconds") or 0.0)
        if seconds > 0:
            rows.append((_run_label(run), seconds, QUANTUM))
    if not rows:
        return

    rows.sort(key=lambda row: row[1])
    positions = list(range(len(rows)))
    ax.barh(positions, [row[1] for row in rows], height=0.55,
            color=[row[2] for row in rows], zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for position, (_, value, _) in zip(positions, rows):
        # Seconds span five orders here. `%g` would render 1065 as 1.07e+03,
        # which is unreadable next to a bar labelled 0.004.
        text = f"{value:,.0f}" if value >= 10 else f"{value:.3g}"
        ax.text(value, position, f"  {text}", va="center", fontsize=9,
                color=INK_PRIMARY, zorder=7)

    ax.set_xscale("log")
    ax.set_xlabel("wall-clock seconds (log scale) · mean field excluded, all methods need it")
    subtitle = "classical is exact here and orders of magnitude cheaper"
    if classical:
        subtitle = (f"exact solve is {classical['fci_dimension']:,} determinants · "
                    "classical wins outright at this size")
    _frame(ax, "What it cost", subtitle, grid_axis="x")


def plot_scaling(classical: dict[str, Any] | None, ax: Any, max_orbitals: int = 16) -> None:
    """FCI dimension against active space -- where the crossover actually lives.

    The only honest "why quantum" chart in the deck. Classical exact cost grows
    combinatorially in the active space while the qubit count grows linearly, so
    this is the axis on which the argument is eventually won. It is emphatically
    not won at CAS(6e,6o), and the chart says so by putting the current point
    near the bottom of a curve that keeps going.
    """
    from math import comb

    orbitals = list(range(4, max_orbitals + 1))
    dimensions = [comb(n, n // 2) ** 2 for n in orbitals]
    qubits = [2 * n for n in orbitals]

    ax.plot(qubits, dimensions, color=CLASSICAL, linewidth=1.8, zorder=4,
            marker="o", markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_yscale("log")

    current_orbitals = (classical or {}).get("active_spatial_orbitals", 6)
    current_qubits = 2 * current_orbitals
    current_dimension = comb(current_orbitals, current_orbitals // 2) ** 2
    _marker(ax, current_qubits, current_dimension, QUANTUM, size=9.0)
    ax.annotate(
        f"  you are here\n  CAS({current_orbitals}e,{current_orbitals}o) · "
        f"{current_dimension:,} determinants",
        xy=(current_qubits, current_dimension), xytext=(6, -4),
        textcoords="offset points", fontsize=9, color=QUANTUM, va="top", zorder=7,
    )

    for label, level in (("a laptop", 1e6), ("a cluster", 1e10)):
        ax.axhline(level, color=BASELINE, linewidth=1.0, linestyle="--", zorder=2)
        ax.annotate(f" exact solve stops being free somewhere past {label}",
                    xy=(0.0, level), xycoords=("axes fraction", "data"),
                    xytext=(2, 4), textcoords="offset points",
                    fontsize=8.5, color=INK_MUTED, zorder=6)

    ax.set_xlabel("qubits   (2 × active spatial orbitals)")
    ax.set_ylabel("FCI determinants (log)")
    ax.set_xticks(qubits[::2])
    _frame(ax, "Where quantum starts to matter",
           "classical exact cost is combinatorial; the register is linear")


def plot_gate_ladder(runs: list[dict[str, Any]], ax: Any) -> None:
    """Two-qubit gate count against the published baseline, with the work shown.

    The weakest number in the study, so it gets its own chart rather than a
    footnote. It also carries the one piece of engineering that moved it: the
    synthesis change, measured rather than claimed.
    """
    rows: list[tuple[str, float, str]] = [
        ("published, raw", PUBLISHED_GATES_RAW, CLASSICAL),
        ("published, pytket", PUBLISHED_GATES_OPTIMIZED, CLASSICAL),
    ]
    for run in runs:
        gates = (run.get("circuit") or {}).get("two_qubit_gates")
        if gates:
            rows.append((
                f"{_run_label(run)}  ({run['ansatz'].get('parameters', '?')} ops)",
                float(gates), QUANTUM,
            ))
    if len(rows) < 3:
        return

    rows.sort(key=lambda row: row[1])
    positions = list(range(len(rows)))
    ax.barh(positions, [row[1] for row in rows], height=0.55,
            color=[row[2] for row in rows], zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    best_quantum = min((row[1] for row in rows if row[2] == QUANTUM), default=None)
    for position, (_, value, color) in zip(positions, rows):
        weight = "bold" if color == QUANTUM and value == best_quantum else "normal"
        ax.text(value, position, f"  {value:,.0f}", va="center", fontsize=9,
                fontweight=weight, color=INK_PRIMARY, zorder=7)

    subtitle = "lower is better · the gap that is still open"
    if best_quantum:
        subtitle = (f"best here is {best_quantum / PUBLISHED_GATES_OPTIMIZED:.0f}× the "
                    "published pytket-optimized count")
    ax.set_xlabel("two-qubit gates after transpilation")
    _frame(ax, "Circuit cost against the published baseline", subtitle, grid_axis="x")


def plot_pipeline(ax: Any, run: dict[str, Any] | None = None) -> None:
    """How the code actually works: modules, artifacts, and the gates that refuse.

    A treemap of this repository would size boxes by lines of code, which encodes
    nothing anyone needs -- the interesting structure is the dataflow and, more
    than that, the five places the workflow is allowed to stop and say no. Those
    refusals are the credibility argument, and they are why a run comes back
    `spin_contaminated` instead of quietly reporting a number that looks fine.

    Timings are drawn from the run when one is supplied, so the architecture
    diagram doubles as a profile.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    adapt = (run or {}).get("adapt") or {}
    total = float((run or {}).get("runtime_seconds") or 0.0)
    pool_seconds = float(adapt.get("pool_build_seconds") or 0.0)
    opt_seconds = float(adapt.get("optimization_seconds") or 0.0)
    circuit_seconds = max(total - pool_seconds - opt_seconds, 0.0)

    # Three rows of boxes with an eight-unit gap between them. Every gate label
    # is placed into one of those gaps rather than beside its diamond, because a
    # label beside a diamond is a label on top of a box.
    ROW = {"top": 50.0, "middle": 31.0, "bottom": 12.0}
    HEIGHT = 11.0

    def box(x, row, w, label, sub, face, edge):
        y = ROW[row]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, HEIGHT, boxstyle="round,pad=0.6,rounding_size=1.6",
            linewidth=1.1, edgecolor=edge, facecolor=face, zorder=3,
        ))
        ax.text(x + w / 2, y + HEIGHT / 2 + (1.7 if sub else 0), label,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK_PRIMARY, zorder=4)
        if sub:
            ax.text(x + w / 2, y + HEIGHT / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=8, color=INK_SECONDARY, zorder=4)

    def arrow(x1, y1, x2, y2, head=True):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>" if head else "-",
            mutation_scale=11, linewidth=1.0, color=INK_MUTED,
            zorder=2, shrinkA=2, shrinkB=2,
        ))

    def gate(x, y, text, label_x=None, label_y=None, ha="center", va="top"):
        """A refusal point. Red because it is allowed to stop the workflow."""
        ax.plot([x], [y], marker="D", markersize=8, color=STATUS_CRITICAL,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6)
        ax.text(label_x if label_x is not None else x,
                label_y if label_y is not None else y - 4.2,
                text, ha=ha, va=va, fontsize=8, color=STATUS_CRITICAL,
                zorder=6, linespacing=1.4)

    module_face, module_edge = "#eef3fb", CLASSICAL
    artifact_face, artifact_edge = "#fdf0e9", QUANTUM
    mid_top, mid_middle, mid_bottom = 55.5, 36.5, 17.5

    box(2, "top", 20, "molecule.py", "geometry · active space", module_face, module_edge)
    box(27, "top", 22, "hamiltonian.py", "RHF → CASCI → JW", module_face, module_edge)
    box(54, "top", 20, "cache JSON", "1819 Pauli terms", artifact_face, artifact_edge)
    box(79, "top", 19, "validation.py", "exact checks", module_face, module_edge)
    arrow(22, mid_top, 27, mid_top)
    arrow(49, mid_top, 54, mid_top)
    arrow(74, mid_top, 79, mid_top)
    gate(51.5, mid_top, "① π-core localization\n≥0.80 occ / ≥0.65 virt Mulliken",
         label_y=47.5)

    pool_label = f"pool build{f'  {pool_seconds:.0f}s' if pool_seconds else ''}"
    opt_label = f"sparse ADAPT{f'  {opt_seconds:.0f}s' if opt_seconds else ''}"
    box(20, "middle", 22, "qiskit_runtime.py", "Aer HEA VQE", module_face, module_edge)
    box(48, "middle", 22, "adapt_runtime.py", f"{pool_label} · {opt_label}",
        module_face, module_edge)
    box(79, "middle", 19, "receipt JSON", "hash-bound", artifact_face, artifact_edge)
    arrow(88.5, ROW["top"], 88.5, ROW["middle"] + HEIGHT)
    arrow(79, mid_middle, 70, mid_middle)
    arrow(48, mid_middle, 42, mid_middle)
    gate(74.5, mid_middle, "② receipt required\nspec + cache + physics sha256",
         label_y=44.0, va="bottom")

    circuit_sub = "PauliEvolutionGate · fountain"
    if circuit_seconds:
        circuit_sub += f"  {circuit_seconds:.0f}s"
    box(2, "bottom", 22, "visualize.py", "these charts", module_face, module_edge)
    box(48, "bottom", 22, "circuit emission", circuit_sub, module_face, module_edge)
    box(76, "bottom", 22, "results/*.json", "energy · gates · ⟨S²⟩",
        artifact_face, artifact_edge)
    arrow(59, ROW["middle"], 59, ROW["bottom"] + HEIGHT)
    gate(59, 27.0, "③ circuit vs algebra  ≤ 1e-9 Ha",
         label_x=54.5, label_y=27.0, ha="right", va="center")
    arrow(70, mid_bottom, 76, mid_bottom)
    gate(73, mid_bottom, "④ sector bound\n⑤ contamination ≤ 0.16 mHa",
         label_y=25.0, va="bottom")

    # results -> visualize routed under everything: a straight line would cross
    # the circuit-emission box and read as a dependency that does not exist.
    arrow(87, ROW["bottom"], 87, 6.0, head=False)
    arrow(87, 6.0, 13, 6.0, head=False)
    arrow(13, 6.0, 13, ROW["bottom"])

    ax.text(2, 68, "How the workflow runs", fontsize=12.5, fontweight="bold",
            color=INK_PRIMARY, va="top")
    ax.text(2, 64.2,
            "blue = module   orange = artifact on disk   "
            "◆ = a checkpoint that can refuse and stop the run",
            fontsize=9, color=INK_SECONDARY, va="top")


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def summary_html(run: dict[str, Any]) -> str:
    """The headline numbers as a stat row. A number is not a bar chart."""
    vqe = run["vqe"]
    circuit = run.get("circuit") or {}
    hartree_fock = vqe["hartree_fock_energy_hartree"]
    span = vqe["reference_energy_hartree"] - hartree_fock
    recovered = 100.0 * (vqe["physical_energy_hartree"] - hartree_fock) / span if span else 0.0
    error = vqe.get("error_millihartree", float("nan"))
    verified = abs(error) <= CHEMICAL_ACCURACY_MHA

    tiles = [
        ("Error vs CASCI", f"{error:+,.3f}", "mHa", STATUS_GOOD if verified else INK_PRIMARY),
        ("Total energy", f"{vqe['physical_energy_hartree']:,.6f}", "Ha", INK_PRIMARY),
        ("Correlation recovered", f"{recovered:.2f}", "%", INK_PRIMARY),
        ("Operators", f"{run['ansatz']['parameters']:d}", "", INK_PRIMARY),
        ("Two-qubit gates", f"{circuit.get('two_qubit_gates', '—')}", "", INK_PRIMARY),
        ("Electrons ⟨N⟩", f"{vqe['electron_number']:.6f}", "", INK_PRIMARY),
        ("Spin ⟨S²⟩", f"{vqe['spin_squared']:.2e}", "", INK_PRIMARY),
        ("Runtime", f"{run.get('runtime_seconds', 0):,.0f}", "s", INK_PRIMARY),
    ]
    cells = "".join(
        f'<div style="flex:1 1 150px;min-width:150px;padding:14px 16px;'
        f'background:{SURFACE};border:1px solid rgba(11,11,11,0.10);border-radius:10px">'
        f'<div style="font-size:11.5px;color:{INK_SECONDARY};margin-bottom:6px">{label}</div>'
        f'<div style="font-size:22px;font-weight:600;color:{color};line-height:1.15">{value}'
        f'<span style="font-size:12.5px;color:{INK_MUTED};font-weight:400"> {unit}</span></div>'
        f"</div>"
        for label, value, unit, color in tiles
    )
    verdict = run.get("status", "?").replace("_", " ")
    return (
        f'<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif">'
        f'<div style="font-size:15px;font-weight:600;color:{INK_PRIMARY};margin-bottom:2px">'
        f'{run["_name"]} — {verdict}</div>'
        f'<div style="font-size:12px;color:{INK_SECONDARY};margin-bottom:12px">'
        f'{run["chemistry"].get("molecule", "")} · {run["ansatz"]["name"]}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px">{cells}</div></div>'
    )


def figure_for(run: dict[str, Any]) -> Any:
    """The four ADAPT charts as one small-multiple grid."""
    plt = _style()
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_convergence(run, axes[0][0])
    plot_energy(run, axes[0][1])
    plot_gradients(run, axes[1][0])
    plot_operators(run, axes[1][1])
    figure.tight_layout(pad=2.6, h_pad=4.0, w_pad=3.5)
    return figure


def show(results: str | Path = "results") -> None:
    """Render everything in a notebook: a stat row and charts per ADAPT run."""
    plt = _style()
    runs = load_runs(results)
    if not runs:
        print(f"No readable results in {results}. Run `python run.py adapt` first.")
        return

    try:
        from IPython.display import HTML, display
    except ImportError:
        display = HTML = None

    adapt_runs = [run for run in runs if run["_is_adapt"]]
    for run in adapt_runs:
        if display is not None:
            display(HTML(summary_html(run)))
        else:
            print(f"\n{run['_name']}: {run['vqe'].get('error_millihartree'):+.3f} mHa")
        figure_for(run)
        plt.show()

    if len(runs) > 1:
        figure, ax = plt.subplots(figsize=_comparison_figsize(len(runs)))
        plot_comparison(runs, ax)
        figure.tight_layout(pad=2.2)
        plt.show()

    if not adapt_runs:
        print("No ADAPT runs found -- the per-run charts need `python run.py adapt`.")


DECK_ORDER = ("error", "gates", "accuracy", "cost", "pipeline")


def deck_figures(
    runs: list[dict[str, Any]],
    classical: dict[str, Any] | None = None,
    trajectories: dict[str, list] | None = None,
) -> list[tuple[str, Any]]:
    """The five presentation figures, in the order they should be presented.

    One figure per claim, named for the claim rather than the chart type, so a
    deck can be assembled by filename without opening the PNGs:

    1. `error`    -- where the quantum number sits among classical methods
    2. `gates`    -- what the circuit costs against the published baseline
    3. `accuracy` -- the two traded against each other; the chart that argues
    4. `cost`     -- wall clock now, and where the classical side stops being free
    5. `pipeline` -- how the code runs and where it is allowed to refuse

    `cost` is the only one with two panels, because "what it cost" and "when does
    that change" are one argument split across two time horizons and separating
    them into different slides loses the point.
    """
    plt = _style()
    adapt_runs = [run for run in runs if run["_is_adapt"]] or runs
    figures: list[tuple[str, Any]] = []

    figure, ax = plt.subplots(figsize=(9.5, 0.46 * (len(runs) + 5) + 2.6))
    plot_method_ladder(runs, classical, ax)
    figure.tight_layout(pad=2.2)
    figures.append(("error", figure))

    figure, ax = plt.subplots(figsize=(9.5, 0.46 * (len(runs) + 2) + 2.4))
    plot_gate_ladder(runs, ax)
    figure.tight_layout(pad=2.2)
    figures.append(("gates", figure))

    figure, ax = plt.subplots(figsize=(9.5, 6.2))
    plot_pareto(runs, ax, trajectories)
    figure.tight_layout(pad=2.4)
    figures.append(("accuracy", figure))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    plot_cost(runs, classical, axes[0])
    plot_scaling(classical, axes[1])
    figure.tight_layout(pad=2.6, w_pad=4.0)
    figures.append(("cost", figure))

    figure, ax = plt.subplots(figsize=(13.5, 8.0))
    plot_pipeline(ax, adapt_runs[0] if adapt_runs else None)
    figure.tight_layout(pad=1.4)
    figures.append(("pipeline", figure))

    return figures


def show_deck(results: str | Path = "results", pareto_points: int = 9) -> None:
    """Render the five presentation figures inline in a notebook.

    `pareto_points` costs one transpile each and is the only slow thing here.
    Set it to 0 to skip the grown-ansatz curve; the chart still draws the
    finished runs and the published point, which is most of the argument.
    """
    plt = _style()
    runs = load_runs(results)
    if not runs:
        print(f"No readable results in {results}. Run `python run.py adapt` first.")
        return
    classical = load_classical(results)
    if classical is None:
        print("No results/classical.json -- run `python run.py classical` (needs "
              "PySCF) to put MP2/CCSD/CCSD(T) on the error and cost charts.\n")

    trajectories: dict[str, list] = {}
    if pareto_points:
        for run in runs:
            if not run["_is_adapt"]:
                continue
            print(f"measuring circuit cost along {run['_name']} "
                  f"({pareto_points} transpiles)...", flush=True)
            trajectories[run["_name"]] = gate_trajectory(run, pareto_points)

    for _, figure in deck_figures(runs, classical, trajectories):
        plt.show()


def deck_main(default_results: str) -> int:
    """`run.py deck` -- the five presentation figures, saved as PNG."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Presentation figures: error, gates, accuracy, cost, pipeline"
    )
    parser.add_argument("--results", type=Path, default=Path(default_results))
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory for the PNGs. Defaults to <results>/deck.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pareto-points", type=int, default=9,
                        help="Circuit-cost samples along each ADAPT run. One "
                             "transpile each; 0 skips the grown-ansatz curve.")
    args = parser.parse_args()

    plt = _style()
    runs = load_runs(args.results)
    if not runs:
        print(f"No readable results in {args.results}. Run `python run.py adapt` first.")
        return 1
    classical = load_classical(args.results)
    if classical is None:
        print("note: no results/classical.json -- the error and cost charts will "
              "omit MP2/CCSD/CCSD(T). Run `python run.py classical` on a machine "
              "with PySCF to add them.")

    trajectories: dict[str, list] = {}
    if args.pareto_points:
        for run in runs:
            if not run["_is_adapt"]:
                continue
            print(f"measuring circuit cost along {run['_name']}...", flush=True)
            trajectories[run["_name"]] = gate_trajectory(run, args.pareto_points)

    output = args.output or (args.results / "deck")
    output.mkdir(parents=True, exist_ok=True)
    for index, (name, figure) in enumerate(deck_figures(runs, classical, trajectories), 1):
        path = output / f"{index}_{name}.png"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {path}")
    return 0


def plot_main(default_results: str) -> int:
    """`run.py plot` -- same charts, saved as PNG instead of shown inline."""
    import argparse

    parser = argparse.ArgumentParser(description="Chart ADAPT and VQE results")
    parser.add_argument("--results", type=Path, default=Path(default_results))
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory for the PNGs. Defaults to the results directory.")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    plt = _style()
    runs = load_runs(args.results)
    if not runs:
        print(f"No readable results in {args.results}. Run `python run.py adapt` first.")
        return 1
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)

    written = []
    for run in runs:
        if not run["_is_adapt"]:
            continue
        figure = figure_for(run)
        path = output / f"{Path(run['_name']).stem}.png"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    if len(runs) > 1:
        figure, ax = plt.subplots(figsize=_comparison_figsize(len(runs)))
        plot_comparison(runs, ax)
        figure.tight_layout(pad=2.2)
        path = output / "comparison.png"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    for path in written:
        print(f"wrote {path}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(plot_main("results"))
