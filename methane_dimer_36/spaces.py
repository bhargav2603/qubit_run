"""Determinant counting, memory models, and the cost ladder.

Nothing here needs a chemistry package.  That is deliberate: the numbers that
decide whether a run is affordable should be knowable *before* installing a
2 GB toolchain, and checkable on a laptop that cannot run the study at all.

The one number every part of this folder is anchored to:

    C(16, 8)^2 = 12870^2 = 165,636,900

which is the full CAS(16e,16o) dimension and matches Table 1 of the HiVQE paper
for the same active-space size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import paper

COMPLEX128_BYTES = 16
FLOAT64_BYTES = 8


def n_determinants(norb: int, nelec: tuple[int, int]) -> int:
    """Dimension of the (n_alpha, n_beta) particle-number sector."""
    n_a, n_b = nelec
    return math.comb(norb, n_a) * math.comb(norb, n_b)


def n_strings(norb: int, n_electrons: int) -> int:
    """Number of occupation strings for one spin species."""
    return math.comb(norb, n_electrons)


def full_cas_dimension() -> int:
    """The paper's (16e,16o) space: 165,636,900 determinants."""
    return n_determinants(paper.N_ORBITALS, (paper.N_ALPHA, paper.N_BETA))


def statevector_bytes(norb: int, nelec: tuple[int, int]) -> int:
    """Memory for a *sector* statevector -- what ffsim actually allocates.

    ffsim never materialises the 2^(2*norb) qubit statevector.  The LUCJ ansatz
    conserves N_alpha and N_beta separately, so the state lives in the sector.
    At (16e,16o) that is the difference between 2.65 GB and 1.2 exabytes.
    """
    return n_determinants(norb, nelec) * COMPLEX128_BYTES


def qubit_statevector_bytes(norb: int) -> int:
    """Memory a dense 2^(2*norb) simulator would need.  For contrast only."""
    return (1 << (2 * norb)) * COMPLEX128_BYTES


def sector_compression_factor(norb: int, nelec: tuple[int, int]) -> float:
    """How much the particle-number sector buys over a dense simulator."""
    return qubit_statevector_bytes(norb) / statevector_bytes(norb, nelec)


@dataclass(frozen=True)
class SubspaceCost:
    """Classical cost of one SQD batch at subspace dimension ``d``."""

    dimension: int
    n_batches: int
    davidson_vectors: int

    @property
    def ci_vector_bytes(self) -> int:
        return self.dimension * FLOAT64_BYTES

    @property
    def batch_peak_bytes(self) -> int:
        """One batch: the Davidson subspace plus the trial and sigma vectors."""
        return self.ci_vector_bytes * (self.davidson_vectors + 2)

    @property
    def parallel_peak_bytes(self) -> int:
        """All batches resident at once, which is what Ray across K CPUs does."""
        return self.batch_peak_bytes * self.n_batches

    @property
    def sequential_peak_bytes(self) -> int:
        """One batch at a time -- the affordable schedule."""
        return self.batch_peak_bytes

    def fits_in(self, ram_bytes: int, *, parallel: bool) -> bool:
        peak = self.parallel_peak_bytes if parallel else self.sequential_peak_bytes
        return peak <= ram_bytes


def subspace_fraction(dimension: int) -> float:
    """Fraction of the full CAS spanned by a subspace of this dimension."""
    return dimension / full_cas_dimension()


def paper_subspace_cost(davidson_vectors: int = 10) -> SubspaceCost:
    """Cost of reproducing the Table II row verbatim (d = 1.26e8)."""
    return SubspaceCost(
        dimension=paper.SUBSPACE_DIMENSION,
        n_batches=paper.N_BATCHES,
        davidson_vectors=davidson_vectors,
    )


# --------------------------------------------------------------------------
# The cost ladder
# --------------------------------------------------------------------------
# The paper's headline configuration is not reachable on a single commodity
# machine, and pretending otherwise wastes a day discovering it.  These rungs
# are the honest options, cheapest first.  `run.py plan` prints them.

@dataclass(frozen=True)
class Rung:
    name: str
    samples_per_batch: int
    n_batches: int
    max_dim: int | None
    published: bool
    note: str


COST_LADDER: tuple[Rung, ...] = (
    Rung(
        name="smoke",
        samples_per_batch=300,
        n_batches=3,
        max_dim=20_000,
        published=False,
        note="Pipeline shakedown. Not a physics result; energies will be poor.",
    ),
    Rung(
        name="extrapolation-low",
        samples_per_batch=9_000,
        n_batches=paper.N_BATCHES,
        max_dim=None,
        published=True,
        note="Lowest of the paper's three extrapolation points (SI section I).",
    ),
    Rung(
        name="extrapolation-mid",
        samples_per_batch=11_000,
        n_batches=paper.N_BATCHES,
        max_dim=None,
        published=True,
        note="Middle extrapolation point.",
    ),
    Rung(
        name="extrapolation-high",
        samples_per_batch=14_000,
        n_batches=paper.N_BATCHES,
        max_dim=None,
        published=True,
        note="Highest extrapolation point. With the two below, extrapolates to |chi_b|=20e3.",
    ),
    Rung(
        name="converged",
        samples_per_batch=paper.SAMPLES_PER_BATCH,
        n_batches=paper.N_BATCHES,
        max_dim=None,
        published=True,
        note="Table II verbatim: d = 1.26e8. Needs a large-memory machine.",
    ),
)


def rung(name: str) -> Rung:
    for r in COST_LADDER:
        if r.name == name:
            return r
    raise KeyError(f"unknown rung {name!r}; choose from {[r.name for r in COST_LADDER]}")


def humanize_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < step:
            return f"{value:.2f} {unit}"
        value /= step
    return f"{value:.2f} EiB"


def describe_scale() -> str:
    """A human-readable statement of what this system costs, with no chemistry."""
    norb = paper.N_ORBITALS
    nelec = (paper.N_ALPHA, paper.N_BETA)
    dets = n_determinants(norb, nelec)
    lines = [
        f"active space           CAS({paper.N_ELECTRONS}e,{paper.N_ORBITALS}o) / {paper.BASIS}",
        f"qubits                 {paper.N_QUBITS_OCCUPATION} occupation "
        f"+ {paper.N_QUBITS_ANCILLA} ancilla = {paper.N_QUBITS_TOTAL}",
        f"alpha strings          C({norb},{nelec[0]}) = {n_strings(norb, nelec[0]):,}",
        f"determinants           {dets:,}",
        f"sector statevector     {humanize_bytes(statevector_bytes(norb, nelec))}",
        f"dense 2^{2 * norb} vector    {humanize_bytes(qubit_statevector_bytes(norb))} "
        f"({sector_compression_factor(norb, nelec):,.0f}x larger)",
    ]
    cost = paper_subspace_cost()
    lines += [
        "",
        f"paper subspace d       {cost.dimension:,} "
        f"({subspace_fraction(cost.dimension):.1%} of the full CAS)",
        f"  one CI vector        {humanize_bytes(cost.ci_vector_bytes)}",
        f"  one batch, peak      {humanize_bytes(cost.batch_peak_bytes)}",
        f"  {cost.n_batches} batches in parallel  {humanize_bytes(cost.parallel_peak_bytes)}"
        "   <- what the paper ran, on 10 CPUs",
        f"  {cost.n_batches} batches sequential   {humanize_bytes(cost.sequential_peak_bytes)} peak"
        "   <- the affordable schedule",
    ]
    return "\n".join(lines)
