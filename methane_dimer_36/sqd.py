"""Sample-based quantum diagonalization.

A thin, instrumented wrapper over
``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``.  The addon owns
self-consistent configuration recovery, batching and the selected-CI solve;
this module supplies the paper's parameters, records what happened at every
iteration, and computes the energy variance that the paper's extrapolation
requires but the addon does not return.

Parameter provenance
--------------------
Table II fixes |chi| = 200e3, K = 10, |chi_b| = 20e3, d = 12.6e7 and 10
recovery steps.  It does not fix the convergence tolerances, the carryover
threshold or the seed.  Those are this folder's, are exposed as flags, and are
written into every result file -- see ``paper.SQD_HYPERPARAMETER_GAP``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

import paper
import spaces


@dataclass(frozen=True)
class SqdConfig:
    """SQD parameters for one run."""

    samples_per_batch: int = paper.SAMPLES_PER_BATCH
    n_batches: int = paper.N_BATCHES
    max_iterations: int = paper.RECOVERY_STEPS
    energy_tol: float = 1e-6
    occupancies_tol: float = 1e-4
    carryover_threshold: float = 1e-4
    symmetrize_spin: bool = True
    max_dim: int | None = None
    seed: int | None = 12345

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SqdIteration:
    """One step of self-consistent configuration recovery."""

    index: int
    best_energy: float
    batch_energies: list[float]
    batch_dimensions: list[int]
    elapsed_seconds: float

    @property
    def best_dimension(self) -> int:
        return self.batch_dimensions[int(np.argmin(self.batch_energies))]


@dataclass
class SqdResult:
    """Outcome of one SQD run."""

    energy: float
    subspace_dimension: int
    config: dict[str, Any]
    history: list[SqdIteration] = field(default_factory=list)
    occupancies: list[list[float]] | None = None
    spin_square: float | None = None
    variance: float | None = None
    source: str = "unknown"
    elapsed_seconds: float = 0.0
    sci_state: object | None = None   # excluded from to_dict(); see run(keep_state=)
    full_dimension: int | None = None  # CAS size of THIS system, not the paper's

    @property
    def subspace_fraction(self) -> float:
        """Fraction of *this* system's CAS that the subspace spans.

        Must not be hardcoded to the paper's 165,636,900: `run.py validate`
        exercises the identical code path on a much smaller active space, and a
        fixed denominator would report a meaningless fraction there and silently
        defeat the saturation check that makes validation meaningful.
        """
        total = self.full_dimension or spaces.full_cas_dimension()
        return self.subspace_dimension / total

    @property
    def converged(self) -> bool:
        if len(self.history) < 2:
            return False
        return abs(self.history[-1].best_energy - self.history[-2].best_energy) < self.config[
            "energy_tol"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": self.energy,
            "subspace_dimension": self.subspace_dimension,
            "subspace_fraction": self.subspace_fraction,
            "full_dimension": self.full_dimension,
            "config": self.config,
            "occupancies": self.occupancies,
            "spin_square": self.spin_square,
            "variance": self.variance,
            "source": self.source,
            "elapsed_seconds": self.elapsed_seconds,
            "converged": self.converged,
            "history": [
                {
                    "index": step.index,
                    "best_energy": step.best_energy,
                    "best_dimension": step.best_dimension,
                    "batch_energies": step.batch_energies,
                    "batch_dimensions": step.batch_dimensions,
                    "elapsed_seconds": step.elapsed_seconds,
                }
                for step in self.history
            ],
        }


def _dimension(sci_state) -> int:
    return int(len(sci_state.ci_strs_a) * len(sci_state.ci_strs_b))


def run(
    mol_data,
    bit_array,
    config: SqdConfig,
    *,
    source: str = "unknown",
    verbose: bool = True,
    keep_state: bool = False,
) -> SqdResult:
    """Run SQD to convergence and return an instrumented result.

    Args:
        keep_state: attach the final ``SCIState`` to the result. Needed for
            ``energy_variance``. Off by default because at (16e,16o) that
            object is large and would dominate a result file; the attribute is
            excluded from ``to_dict`` either way.
    """
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    hamiltonian = mol_data.hamiltonian
    norb, nelec = mol_data.norb, mol_data.nelec
    full_dimension = spaces.n_determinants(norb, nelec)

    history: list[SqdIteration] = []
    started = time.monotonic()
    last_tick = started

    def callback(results) -> None:
        nonlocal last_tick
        now = time.monotonic()
        energies = [float(r.energy) + mol_data.core_energy for r in results]
        dimensions = [_dimension(r.sci_state) for r in results]
        step = SqdIteration(
            index=len(history),
            best_energy=min(energies),
            batch_energies=energies,
            batch_dimensions=dimensions,
            elapsed_seconds=now - last_tick,
        )
        history.append(step)
        last_tick = now
        if verbose:
            print(
                f"    iter {step.index:2d}  E = {step.best_energy:.10f} Ha  "
                f"d = {step.best_dimension:,}  "
                f"({step.best_dimension / full_dimension:.2%} of CAS)  "
                f"{step.elapsed_seconds:.1f}s"
            )

    result = diagonalize_fermionic_hamiltonian(
        hamiltonian.one_body_tensor,
        hamiltonian.two_body_tensor,
        bit_array,
        samples_per_batch=config.samples_per_batch,
        norb=norb,
        nelec=nelec,
        num_batches=config.n_batches,
        energy_tol=config.energy_tol,
        occupancies_tol=config.occupancies_tol,
        max_iterations=config.max_iterations,
        symmetrize_spin=config.symmetrize_spin,
        max_dim=config.max_dim,
        carryover_threshold=config.carryover_threshold,
        callback=callback,
        seed=config.seed,
    )

    return SqdResult(
        # solve_sci returns the active-space electronic energy only (it builds
        # the energy from RDMs contracted with the one/two-body tensors), so the
        # core energy is added here exactly once.
        energy=float(result.energy) + mol_data.core_energy,
        subspace_dimension=_dimension(result.sci_state),
        config=config.to_dict(),
        history=history,
        occupancies=[list(map(float, occ)) for occ in result.orbital_occupancies],
        spin_square=float(result.sci_state.spin_square()),
        source=source,
        elapsed_seconds=time.monotonic() - started,
        sci_state=result.sci_state if keep_state else None,
        full_dimension=full_dimension,
    )


# --------------------------------------------------------------------------
# Energy variance
# --------------------------------------------------------------------------

def energy_variance(sci_state, mol_data, *, verbose: bool = False) -> float:
    """Compute dH = <psi|H^2|psi> - <psi|H|psi>^2 for an SCI state.

    The paper extrapolates total energy linearly to dH -> 0 (SI Figure 5), so
    this is required for the extrapolation rungs. It is not cheap: H|psi>
    leaves the sparse subspace, so the state is embedded into the full CAS
    vector and the Hamiltonian applied there.

    At (16e,16o) that means two dense 165,636,900-element vectors, about
    2.6 GB. `run.py plan` reports this before you commit to it.
    """
    import pyscf.fci

    norb, nelec = sci_state.norb, sci_state.nelec
    hamiltonian = mol_data.hamiltonian

    dimension = spaces.n_determinants(norb, nelec)
    if verbose:
        print(
            f"    variance: embedding into {dimension:,} determinants "
            f"(~{spaces.humanize_bytes(dimension * 8 * 2)} for two vectors)"
        )

    # Embed the sparse SCI amplitudes into the dense CAS vector.
    strings_a = pyscf.fci.cistring.make_strings(range(norb), nelec[0])
    strings_b = pyscf.fci.cistring.make_strings(range(norb), nelec[1])
    index_a = {int(s): i for i, s in enumerate(strings_a)}
    index_b = {int(s): i for i, s in enumerate(strings_b)}

    vector = np.zeros((len(strings_a), len(strings_b)))
    rows = [index_a[int(s)] for s in sci_state.ci_strs_a]
    cols = [index_b[int(s)] for s in sci_state.ci_strs_b]
    vector[np.ix_(rows, cols)] = np.asarray(sci_state.amplitudes, dtype=float)

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("SCI state has zero norm")
    vector /= norm

    h2e = pyscf.fci.direct_spin1.absorb_h1e(
        hamiltonian.one_body_tensor, hamiltonian.two_body_tensor, norb, nelec, 0.5
    )
    h_psi = pyscf.fci.direct_spin1.contract_2e(h2e, vector, norb, nelec)

    energy = float(np.dot(vector.ravel(), h_psi.ravel()))
    h_squared = float(np.dot(h_psi.ravel(), h_psi.ravel()))
    return h_squared - energy**2


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------

def ablation_config(reference: SqdResult, config: SqdConfig) -> SqdConfig:
    """Config for a uniform-sampling control matched to a real run.

    Matching the subspace dimension is the whole point: a control at a
    different dimension compares two things at once and settles nothing.
    """
    from dataclasses import replace

    return replace(config, max_dim=reference.subspace_dimension)
