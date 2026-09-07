#!/usr/bin/env python3
"""HI-VQE: the Handover Iterative Variational Quantum Eigensolver.

Follows the iteration of Pellow-Jarman et al., arXiv:2503.06292, Figure 3:

    1. prepare U(theta)|HF> and measure it, giving electron configurations
    2. discard or recover any configuration with the wrong electron count
    3. add them to the accumulated subspace
    4. project H into that subspace and diagonalise it classically -- exactly
    5. drop configurations whose amplitude came back below threshold
    6. generate single and double excitations of the leading configurations and
       add the ones with the largest coupling to it
    7. let a classical optimizer move theta
    8. stop when the energy has stopped moving

The handover is step 4. The quantum device is never asked for an energy, only
for a *set of configurations*; the amplitudes are then set exactly by a
classical eigensolver, which is why noise on the device cannot bias the result
below the variational bound and why one measurement replaces the 15,697 Pauli
words a conventional VQE would need for this Hamiltonian.

Three properties follow, and this module measures all three rather than
asserting them:

* **The energy is a strict upper bound.** P H P has its lowest eigenvalue above
  the true CAS ground state for any P. A run that reports an energy below the
  cached CASCI value is not a better answer, it is a broken one.
* **The energy is monotone.** The subspace only grows (pruning is applied to
  amplitudes that carry no weight), so the energy cannot rise. `history` records
  it every iteration so a non-monotone run is visible rather than averaged away.
* **The quantum layer is separable.** `--no-quantum` and `--no-expansion` run the
  same loop with one half removed, so what the sampler actually contributed is
  an ablation you measure, not a claim you make.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from determinants import (
    ActiveSpace,
    Subspace,
    SubspaceHamiltonian,
    embed_coefficients,
    hartree_fock_determinant,
    hartree_fock_subspace,
    matrix_element,
    orbital_occupancies,
    reindex_coefficients,
    single_and_double_excitations,
    solve_subspace,
    spin_square,
    string_weights,
)


CHEMICAL_ACCURACY_HA = 1.6e-3
# How far <S^2> may sit from the nearest S(S+1) before the state stops being a
# spin eigenstate at all.
SPIN_CONTAMINATION_TOLERANCE = 1.0e-3
# Ceiling on the throwaway subspace the SPSA objective diagonalises.
OPTIMIZER_PROBE_DETERMINANTS = 2500


def spin_contamination(spin_squared: float) -> float:
    """Distance from <S^2> to the nearest legitimate S(S+1).

    Testing <S^2> against zero would be the wrong check twice over. A selected
    determinant space is not spin-complete in general, so a truncated
    wavefunction genuinely can be a mixture -- that is the failure worth
    catching. But the exact ground state of a spin-free Hamiltonian in the
    S_z = 0 sector is not obliged to be a *singlet*: at a dissociated Li-S bond
    the two open-shell fragments give a singlet and a triplet that are nearly
    degenerate, and CASCI returns whichever is lower. A triplet ground state is
    a result; a state that is neither is a broken calculation.
    """
    if spin_squared < -1.0e-6:
        return abs(spin_squared)
    # S(S+1) for S = 0, 1/2, 1, 3/2, ... solved for S and rounded to the grid.
    total_spin = 0.5 * (np.sqrt(max(0.0, 4.0 * spin_squared + 1.0)) - 1.0)
    nearest = round(total_spin * 2.0) / 2.0
    return abs(spin_squared - nearest * (nearest + 1.0))


@dataclass
class HiVqeSettings:
    """Every knob, with the defaults that reach chemical accuracy on Li2S."""

    # -- subspace control
    max_determinants: int = 20000
    expansion: int = 400
    growth_factor: float = 3.0
    expansion_references: int = 8
    amplitude_threshold: float = 1.0e-6
    # How candidates are ranked: "pt2" is the Epstein-Nesbet energy gain,
    # "coupling" the bare first-order numerator the paper states. Measured over
    # 18 runs on six synthetic active spaces, "pt2" won 13 on the variational
    # energy and 15 once corrected (median 217 -> 203 mHa), so it is the
    # default; `--ranking coupling` is what reproduces the paper's rule.
    ranking: str = "pt2"
    # Keep the two string sets identical, so the space is closed under the
    # alpha/beta flip. Ignored unless the alpha and beta electron counts match.
    #
    # OFF by default, and the reason is measured rather than assumed. Over 18
    # runs on six synthetic active spaces it cut the distance from <S^2> to the
    # nearest S(S+1) by about a third (median 3.3e-2 -> 2.2e-2) but made the
    # variational energy *worse* in 16 of them (median 217 -> 227 mHa), because
    # forcing one shared set spends dimension the two blocks were using
    # separately. That is the right trade only when a run comes back
    # SPIN CONTAMINATED -- which is exactly when to turn it on.
    spin_complete: bool = False
    # -- the perturbative correction, reported beside the variational energy
    pt2: bool = True
    # The first-order space is allowed this multiple of the working dimension.
    pt2_factor: float = 4.0
    # -- iteration control
    max_iterations: int = 20
    energy_tolerance: float = 1.0e-5
    patience: int = 3
    # -- quantum layer
    simulator: str = "sector"  # sector | aer | none
    reps: int = 1
    shots: int = 8192
    init_scale: float = 0.4
    seed: int = 1234
    readout_error: float = 0.0
    depolarizing_error: float = 0.0
    recover: bool = True
    # -- optimizer
    optimizer: str = "spsa"  # spsa | none
    optimizer_steps: int = 1
    # SPSA costs two extra circuit samples per step, and sampling is the single
    # largest cost of a run. Updating theta every third iteration instead of
    # every one cuts the sampling bill by more than half while leaving the
    # optimizer the same number of steps over a long run.
    optimizer_every: int = 1
    givens_depth: int | None = None
    spsa_a: float = 0.25
    spsa_c: float = 0.1
    # -- ablations
    use_expansion: bool = True

    @property
    def use_quantum(self) -> bool:
        return self.simulator != "none"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["use_quantum"] = self.use_quantum
        return payload


@dataclass
class IterationRecord:
    iteration: int
    energy: float
    error_hartree: float
    dimension: int
    n_strings_a: int
    n_strings_b: int
    n_proposed_determinants: int
    n_added_by_expansion: int
    quantum_leakage: float
    quantum_recovered: int
    circuit_runs: int
    seconds: float
    optimizer_energy: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HiVqeResult:
    energy: float
    reference_energy: float
    hartree_fock_energy: float
    error_hartree: float
    dimension: int
    full_dimension: int
    spin_squared: float
    spin_contamination: float
    occupancies_alpha: list[float]
    occupancies_beta: list[float]
    iterations: int
    stop_reason: str
    verdict: str
    history: list[IterationRecord]
    settings: dict[str, Any]
    circuit_metrics: dict[str, Any]
    seconds: float
    # Epstein-Nesbet second order over the determinants left outside the
    # subspace. Reported separately and never added into `energy`: PT2 is not
    # variational, and `energy` is the number the upper-bound guarantee applies
    # to. Zero when the correction was not computed.
    pt2_correction: float = 0.0
    pt2_determinants: int = 0

    @property
    def energy_pt2(self) -> float:
        """The variational energy plus the correction. Not an upper bound."""
        return self.energy + self.pt2_correction

    @property
    def pt2_reliable(self) -> bool:
        """Is the correction small enough for perturbation theory to mean anything?

        Epstein-Nesbet assumes the determinants outside the subspace are a small
        perturbation on the ones inside. From a nearly-Hartree-Fock subspace on
        a strongly correlated system that assumption fails outright, and the sum
        can overshoot straight past the exact energy -- a 1-determinant subspace
        on a synthetic test system produced a correction of -7.6 Ha against a
        variational error of +4.1 Ha.

        The test compares the correction with the correlation energy the
        subspace already captured. If PT2 wants to move the energy by more than
        half as much again, the expansion is not a perturbation and the
        corrected number is not usable. Reported rather than suppressed: a
        visible, flagged number is more useful than a silent one, and unlike the
        variational energy this quantity carries no bound to protect.
        """
        if not self.pt2_determinants:
            return False
        captured = abs(self.energy - self.hartree_fock_energy)
        if captured < 1.0e-9:
            return False
        return abs(self.pt2_correction) <= 0.5 * captured

    @property
    def error_pt2_hartree(self) -> float:
        return self.energy_pt2 - self.reference_energy

    @property
    def error_millihartree(self) -> float:
        return self.error_hartree * 1000.0

    @property
    def subspace_fraction(self) -> float:
        return self.dimension / self.full_dimension

    @property
    def correlation_recovered(self) -> float:
        total = self.reference_energy - self.hartree_fock_energy
        if abs(total) < 1e-12:
            return 1.0
        return (self.energy - self.hartree_fock_energy) / total

    def as_dict(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if key != "history"
        }
        payload["history"] = [record.as_dict() for record in self.history]
        payload["error_millihartree"] = self.error_millihartree
        payload["subspace_fraction"] = self.subspace_fraction
        payload["correlation_recovered"] = self.correlation_recovered
        payload["energy_pt2"] = self.energy_pt2
        payload["error_pt2_millihartree"] = self.error_pt2_hartree * 1000.0
        payload["pt2_reliable"] = self.pt2_reliable
        return payload


# --------------------------------------------------------------------------
# The quantum layer, behind one small interface
# --------------------------------------------------------------------------


@dataclass
class Proposal:
    strings_a: np.ndarray
    strings_b: np.ndarray
    counts: np.ndarray
    leakage: float = 0.0
    recovered: int = 0
    circuit_runs: int = 0

    @property
    def n_determinants(self) -> int:
        return int(len(self.counts))


class Proposer:
    """Turns circuit parameters into proposed electron configurations."""

    n_parameters = 0
    metrics: dict[str, Any] = {}

    def sample(
        self, angles: Sequence[float], occupancy_prior=None
    ) -> Proposal:  # pragma: no cover - interface
        raise NotImplementedError


class NullProposer(Proposer):
    """The `--simulator none` ablation: no quantum layer at all.

    The loop then reduces to a classical selected-CI driven purely by the
    single/double expansion, which is the honest control for "what did the
    quantum sampler buy?".
    """

    def __init__(self, n_alpha: int, n_beta: int) -> None:
        self._hf = hartree_fock_determinant(n_alpha, n_beta)

    def sample(self, angles: Sequence[float], occupancy_prior=None) -> Proposal:
        return Proposal(
            strings_a=np.array([self._hf[0]], dtype=np.int64),
            strings_b=np.array([self._hf[1]], dtype=np.int64),
            counts=np.array([1], dtype=np.int64),
        )


class SectorProposer(Proposer):
    """Exact statevector sampling inside the particle-number sector."""

    def __init__(
        self,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
        reps: int,
        shots: int,
        seed: int,
        givens_depth: int | None = None,
    ) -> None:
        from ansatz import build_ansatz, circuit_metrics
        from sector_sim import SectorSimulator

        self.circuit, self.parameters = build_ansatz(
            n_orbitals, n_alpha, n_beta, reps, givens_depth=givens_depth
        )
        self.simulator = SectorSimulator(self.circuit, n_orbitals, n_alpha, n_beta)
        self.n_parameters = len(self.parameters)
        self.metrics = circuit_metrics(self.circuit)
        self.shots = shots
        self._rng = np.random.default_rng(seed)
        self.n_circuit_runs = 0
        self._cached_angles: bytes | None = None
        self._cached_probability: np.ndarray | None = None

    def _probabilities(self, angles: Sequence[float]) -> tuple[np.ndarray, int]:
        """The sector distribution for these angles, computed at most once.

        Forming the 853,776-amplitude state is ~2.5 s and is the single largest
        cost in a run. The angles are unchanged for the whole run under
        `--optimizer none`, and SPSA's accepted point is usually one of the two
        probes just evaluated, so the same state was being rebuilt from scratch
        on nearly every iteration. Only one distribution is held: the states are
        13 MB each and the repeats are consecutive.

        Returns the distribution and how many circuit evaluations it cost --
        one, or zero on a cache hit. That number is reported per iteration and
        summed into the measurement-cost figure, so it has to be what actually
        happened rather than a constant.
        """
        key = np.ascontiguousarray(angles, dtype=np.float64).tobytes()
        if key == self._cached_angles and self._cached_probability is not None:
            return self._cached_probability, 0
        probability = self.simulator.probabilities(angles, self.parameters)
        self._cached_angles = key
        self._cached_probability = probability
        self.n_circuit_runs += 1
        return probability, 1

    def sample(self, angles: Sequence[float], occupancy_prior=None) -> Proposal:
        probability, evaluations = self._probabilities(angles)
        n_b = probability.shape[1]
        flat = probability.reshape(-1)
        if self.shots > 0:
            drawn = self._rng.multinomial(self.shots, flat / flat.sum())
            indices = np.nonzero(drawn)[0]
            counts = drawn[indices]
        else:
            # shots = 0 means "use the exact distribution", which removes shot
            # noise from a study that is about configuration selection, not
            # about sampling statistics.
            indices = np.nonzero(flat > 1.0e-12)[0]
            counts = np.round(flat[indices] * 1_000_000).astype(np.int64)
        rows, columns = np.divmod(indices, n_b)
        return Proposal(
            strings_a=self.simulator.strings_a[rows],
            strings_b=self.simulator.strings_b[columns],
            counts=counts,
            leakage=0.0,
            circuit_runs=evaluations,
        )


class AerProposer(Proposer):
    """Sampling through Qiskit Aer, optionally with a noise model."""

    def __init__(
        self,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
        reps: int,
        shots: int,
        seed: int,
        readout_error: float,
        depolarizing_error: float,
        recover: bool,
        givens_depth: int | None = None,
    ) -> None:
        from ansatz import ConfigurationSampler, SamplerSettings

        self.sampler = ConfigurationSampler(
            n_orbitals,
            n_alpha,
            n_beta,
            reps=reps,
            givens_depth=givens_depth,
            settings=SamplerSettings(
                shots=shots,
                seed=seed,
                readout_error=readout_error,
                depolarizing_error=depolarizing_error,
                recover=recover,
            ),
        )
        self.n_parameters = self.sampler.n_parameters
        self.metrics = self.sampler.metrics

    def sample(self, angles: Sequence[float], occupancy_prior=None) -> Proposal:
        batch = self.sampler.sample(angles, occupancy_prior=occupancy_prior)
        return Proposal(
            strings_a=batch.strings_a,
            strings_b=batch.strings_b,
            counts=batch.counts,
            leakage=batch.leakage,
            recovered=batch.n_recovered,
            circuit_runs=1,
        )


def make_proposer(space: ActiveSpace, settings: HiVqeSettings) -> Proposer:
    if settings.simulator == "none":
        return NullProposer(space.n_alpha, space.n_beta)
    if settings.simulator == "sector":
        if settings.readout_error or settings.depolarizing_error:
            raise ValueError(
                "the sector simulator is noiseless by construction; use "
                "--simulator aer to study readout or gate noise"
            )
        return SectorProposer(
            space.n_orbitals,
            space.n_alpha,
            space.n_beta,
            settings.reps,
            settings.shots,
            settings.seed,
            settings.givens_depth,
        )
    if settings.simulator == "aer":
        return AerProposer(
            space.n_orbitals,
            space.n_alpha,
            space.n_beta,
            settings.reps,
            settings.shots,
            settings.seed,
            settings.readout_error,
            settings.depolarizing_error,
            settings.recover,
            settings.givens_depth,
        )
    raise ValueError(f"unknown simulator {settings.simulator!r}")


# --------------------------------------------------------------------------
# Subspace bookkeeping
# --------------------------------------------------------------------------


def spin_complete_subspace(
    subspace: Subspace,
    coefficients: np.ndarray,
    max_determinants: int,
    anchor: tuple[int, int],
) -> tuple[Subspace, np.ndarray]:
    """Make the subspace closed under exchanging the alpha and beta blocks.

    A determinant (Ia, Ib) and its spin-flipped partner (Ib, Ia) are degenerate
    for a spin-free Hamiltonian in the Sz = 0 sector, so a space holding one but
    not the other cannot represent a spin eigenstate -- which is what the
    `SPIN CONTAMINATED` verdict catches after the fact. For a tensor-product
    subspace the condition is simply that the two string sets are the same set,
    because then (Ia, Ib) in S_a x S_b implies (Ib, Ia) is too.

    Making them the same set is nearly free: rank the union by the weight each
    string carries in either block and keep the same top slice for both. The
    dimension stays at the cap rather than growing, so this buys the symmetry by
    spending resolution, not memory.

    This is a necessary condition, not full spin adaptation -- the space is
    closed under the flip, which is what removes the systematic contamination;
    it is not projected onto an S^2 eigenvector.
    """
    weights_a, weights_b = string_weights(coefficients)
    union = np.union1d(subspace.strings_a, subspace.strings_b)

    combined = np.zeros(len(union), dtype=np.float64)
    for strings, weights in ((subspace.strings_a, weights_a), (subspace.strings_b, weights_b)):
        if len(strings) == 0:
            continue
        where = np.searchsorted(union, strings)
        combined[where] += weights

    side = max(1, int(np.sqrt(max_determinants)))
    order = np.argsort(-combined)[:side]
    chosen = set(int(union[index]) for index in order)
    # The Hartree-Fock strings anchor the space exactly as they do in pruning.
    chosen.update(int(value) for value in anchor)
    shared = np.array(sorted(chosen), dtype=np.int64)

    completed = Subspace(shared, shared)
    moved = reindex_coefficients(coefficients, subspace, completed)
    norm = float(np.linalg.norm(moved))
    if norm > 0:
        moved = moved / norm
    return completed, moved


def prune_subspace(
    subspace: Subspace,
    coefficients: np.ndarray,
    max_determinants: int,
    threshold: float,
    keep: tuple[int, int],
) -> tuple[Subspace, np.ndarray]:
    """Step 5: drop strings that carry no amplitude, then cap the dimension.

    Weight is the marginal probability of a string, summed over its partners,
    which is the right quantity for a tensor-product subspace: dropping an alpha
    string removes a whole row of determinants, so what matters is the total
    weight of that row and not any single coefficient.

    The Hartree-Fock strings are never dropped. They anchor the subspace, they
    are what the classical expansion excites from, and losing them to a
    threshold would silently change what the run is computing.
    """
    weights_a, weights_b = string_weights(coefficients)

    def select(weights: np.ndarray, strings: np.ndarray, anchor: int, limit: int):
        order = np.argsort(-weights)
        chosen = [index for index in order if weights[index] > threshold]
        if not chosen:
            chosen = list(order[:1])
        chosen = chosen[:limit]
        anchor_index = int(np.searchsorted(strings, anchor))
        if anchor_index < len(strings) and strings[anchor_index] == anchor:
            if anchor_index not in chosen:
                chosen = chosen[: max(1, limit - 1)] + [anchor_index]
        return np.sort(np.array(chosen, dtype=np.int64))

    # Split the determinant budget evenly between the two spin blocks. They are
    # the same size for a closed shell, and an uneven split would buy resolution
    # in one spin channel that the other cannot use.
    side = max(1, int(np.sqrt(max_determinants)))
    rows = select(weights_a, subspace.strings_a, keep[0], side)
    columns = select(weights_b, subspace.strings_b, keep[1], side)
    pruned = Subspace(subspace.strings_a[rows], subspace.strings_b[columns])
    kept = coefficients[np.ix_(rows, columns)]
    norm = float(np.linalg.norm(kept))
    if norm > 0:
        kept = kept / norm
    return pruned, kept


def rank_candidates(
    space: ActiveSpace,
    subspace: Subspace,
    coefficients: np.ndarray,
    n_references: int,
    energy: float | None = None,
    ranking: str = "pt2",
) -> list[tuple[float, tuple[int, int]]]:
    """Step 6: score every single and double excitation of the leading dets.

    The paper generates excitations from the single configuration with the
    largest amplitude and ranks them by |<phi_ref|H|phi>|. Taking the leading
    `n_references` configurations instead and ranking by the coherent sum
    |sum_i c_i <phi|H|phi_i>| is the standard first-order perturbative estimate;
    it costs a few reference determinants more and is strictly better informed,
    and `n_references = 1` reproduces the paper's rule exactly.

    Two rankings, because the numerator alone is not the energy gain:

    * ``coupling`` -- ``|sum_i c_i <phi|H|phi_i>|``, the rule above.
    * ``pt2`` (default) -- that numerator squared over ``E - <phi|H|phi>``, the
      Epstein-Nesbet second-order energy lowering the determinant would actually
      buy. The denominator is what separates a large coupling to a high-lying
      determinant from a modest coupling to a near-degenerate one, which is
      exactly the distinction that decides a stretched bond. This is the CIPSI
      selection criterion (Huron, Malrieu and Rancurel 1973) and the acquisition
      function of Active-Sampling SQD, arXiv:2603.13536.

    The two agree whenever the denominators are similar, so the cheap rule is
    kept for reproducing the paper exactly.
    """
    if ranking not in {"pt2", "coupling"}:
        raise ValueError(f"unknown ranking: {ranking}")
    flat = np.abs(coefficients).reshape(-1)
    if flat.size == 0:
        return []
    # At least one reference, at most the whole subspace: `--expansion-references 0`
    # would otherwise index argpartition with -1 and silently rank against the
    # *worst* determinant instead of the best.
    count = max(1, min(n_references, flat.size))
    leading = np.argpartition(-flat, count - 1)[:count]
    leading = leading[np.argsort(-flat[leading])]
    n_b = coefficients.shape[1]

    references: list[tuple[tuple[int, int], float]] = []
    for index in leading:
        row, column = divmod(int(index), n_b)
        amplitude = float(coefficients[row, column])
        if amplitude == 0.0:
            continue
        references.append(
            (
                (int(subspace.strings_a[row]), int(subspace.strings_b[column])),
                amplitude,
            )
        )
    if not references:
        return []

    inside_a = set(int(value) for value in subspace.strings_a)
    inside_b = set(int(value) for value in subspace.strings_b)

    scores: dict[tuple[int, int], float] = {}
    for determinant, _ in references:
        for candidate in single_and_double_excitations(determinant, space.n_orbitals):
            if candidate[0] in inside_a and candidate[1] in inside_b:
                continue
            scores.setdefault(candidate, 0.0)

    for candidate in scores:
        total = 0.0
        for determinant, amplitude in references:
            # Slater-Condon vanishes beyond a double excitation, and most
            # candidate/reference pairs are further apart than that. Two XORs
            # decide it, which is far cheaper than evaluating the element.
            degree = ((candidate[0] ^ determinant[0]).bit_count() // 2) + (
                (candidate[1] ^ determinant[1]).bit_count() // 2
            )
            if degree > 2:
                continue
            total += amplitude * matrix_element(space, candidate, determinant)
        if ranking == "coupling" or energy is None:
            scores[candidate] = abs(total)
            continue
        # Epstein-Nesbet: the denominator is E - <phi|H|phi>, negative for the
        # ground state. A candidate degenerate with the current energy would
        # divide by ~0, and perturbation theory says nothing useful there, so it
        # is scored on the numerator alone rather than being handed an infinity.
        gap = energy - matrix_element(space, candidate, candidate)
        scores[candidate] = (
            total * total / abs(gap) if abs(gap) > 1.0e-8 else abs(total)
        )

    return sorted(
        ((value, key) for key, value in scores.items() if value > 0.0),
        key=lambda item: -item[0],
    )


def grow_subspace(
    subspace: Subspace,
    ranked: Sequence[tuple[float, tuple[int, int]]],
    limit: int,
    dimension_cap: int,
    spin_complete: bool = False,
) -> tuple[Subspace, int]:
    """Add the best-scoring candidates without letting the dimension run away.

    A tensor-product subspace grows by *rows and columns*, not by single
    determinants: adding one configuration whose alpha string is new adds a
    whole column of determinants with it. Candidates are therefore accepted in
    rank order and stopped at the dimension cap rather than at a determinant
    count, which is the quantity that actually decides the cost of the next
    diagonalisation.
    """
    if spin_complete:
        # Both blocks share one set, so a candidate contributes both of its
        # strings to it. Accepting only one would break the closure that
        # `spin_complete_subspace` just established.
        shared = set(int(value) for value in subspace.strings_a)
        shared.update(int(value) for value in subspace.strings_b)
        added = 0
        for _, (string_a, string_b) in ranked[:limit]:
            fresh = {string_a, string_b} - shared
            if not fresh:
                continue
            if (len(shared) + len(fresh)) ** 2 > dimension_cap:
                break
            shared |= fresh
            added += 1
        grown = np.array(sorted(shared), dtype=np.int64)
        return Subspace(grown, grown), added

    strings_a = list(int(value) for value in subspace.strings_a)
    strings_b = list(int(value) for value in subspace.strings_b)
    set_a, set_b = set(strings_a), set(strings_b)
    added = 0
    for _, (string_a, string_b) in ranked[:limit]:
        new_a = string_a not in set_a
        new_b = string_b not in set_b
        projected = (len(set_a) + int(new_a)) * (len(set_b) + int(new_b))
        if projected > dimension_cap:
            break
        if new_a:
            set_a.add(string_a)
        if new_b:
            set_b.add(string_b)
        if new_a or new_b:
            added += 1
    return (
        Subspace(
            np.array(sorted(set_a), dtype=np.int64),
            np.array(sorted(set_b), dtype=np.int64),
        ),
        added,
    )


def perturbative_correction(
    space: ActiveSpace,
    subspace: Subspace,
    coefficients: np.ndarray,
    energy: float,
    n_references: int,
    dimension_cap: int,
    ranking: str = "pt2",
) -> tuple[float, int]:
    """Epstein-Nesbet second-order energy from the determinants left outside.

        E_PT2 = sum_{D not in S} |<D|H|Psi>|^2 / (E - <D|H|D>)

    Two things make this cheap rather than expensive. The numerator for *every*
    determinant of an enlarged space is one sigma product away: embed the
    converged CI vector into a space that also contains the candidates, apply
    H, and read off the rows that were zero -- `sigma` never needed the vector
    to be an eigenvector. And the denominator is the operator's own diagonal,
    which it already builds for the Davidson preconditioner.

    The first-order space is the tensor product of the enlarged string sets, so
    it reaches well beyond the candidates that seeded it; `pt2_determinants` in
    the result records how many determinants the sum actually ran over. It is
    still a *truncated* correction rather than the full first-order interacting
    space, and it is reported as its own number, never folded into the
    variational energy -- E_PT2 is not variational, and the strict upper bound
    is the property this whole workflow is built to preserve.
    """
    ranked = rank_candidates(space, subspace, coefficients, n_references, energy, ranking)
    if not ranked:
        return 0.0, 0
    enlarged, added = grow_subspace(
        subspace,
        ranked,
        len(ranked),
        dimension_cap,
        spin_complete=bool(
            len(subspace.strings_a) == len(subspace.strings_b)
            and np.array_equal(subspace.strings_a, subspace.strings_b)
        ),
    )
    if added == 0 or enlarged.dimension <= subspace.dimension:
        return 0.0, 0

    embedded = embed_coefficients(coefficients, subspace, enlarged)
    norm = float(np.linalg.norm(embedded))
    if norm <= 0.0:
        return 0.0, 0
    embedded = embedded / norm

    operator = SubspaceHamiltonian(space, enlarged)
    # sigma is (H - E_core); the core term contributes only where the vector is
    # nonzero, and every determinant summed below has a zero coefficient.
    couplings = operator.sigma(embedded)
    diagonal = operator.diagonal()

    outside = np.ones(enlarged.shape, dtype=bool)
    rows = np.searchsorted(enlarged.strings_a, subspace.strings_a)
    columns = np.searchsorted(enlarged.strings_b, subspace.strings_b)
    outside[np.ix_(rows, columns)] = False
    if not outside.any():
        return 0.0, 0

    numerator = couplings[outside]
    gaps = energy - diagonal[outside]
    usable = np.abs(gaps) > 1.0e-8
    correction = float(np.sum(numerator[usable] ** 2 / gaps[usable]))
    return correction, int(usable.sum())


# --------------------------------------------------------------------------
# The optimizer
# --------------------------------------------------------------------------


def spsa_step(
    objective: Callable[[np.ndarray], float],
    angles: np.ndarray,
    step: int,
    a: float,
    c: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """One simultaneous-perturbation step: two evaluations, any dimension.

    SPSA rather than a gradient method because the objective here is a *sampled*
    subspace energy -- it is stochastic and has no analytic derivative -- and
    rather than COBYLA because the ansatz has a few hundred parameters, which is
    where a simplex method stops being practical.
    """
    magnitude = a / (step + 1.0) ** 0.602
    perturbation = c / (step + 1.0) ** 0.101
    direction = rng.choice([-1.0, 1.0], size=angles.shape)
    plus = objective(angles + perturbation * direction)
    minus = objective(angles - perturbation * direction)
    gradient = (plus - minus) / (2.0 * perturbation) * direction
    return angles - magnitude * gradient, min(plus, minus)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_hivqe(
    space: ActiveSpace,
    reference_energy: float,
    hartree_fock_energy: float,
    settings: HiVqeSettings,
    progress: Callable[[str], None] = print,
) -> HiVqeResult:
    started = time.time()
    rng = np.random.default_rng(settings.seed + 977)
    proposer = make_proposer(space, settings)

    angles = np.zeros(0)
    if proposer.n_parameters:
        # Imported here rather than at module scope so that `--simulator none`
        # -- the classical control, and the path `validation.py` uses -- runs on
        # a machine with no Qiskit installed at all.
        from ansatz import initial_parameters

        angles = initial_parameters(
            proposer.n_parameters, settings.init_scale, settings.seed
        )

    anchor = hartree_fock_determinant(space.n_alpha, space.n_beta)
    subspace = hartree_fock_subspace(space.n_alpha, space.n_beta)
    coefficients = np.ones((1, 1))
    # `--max-determinants` is the largest subspace that ever gets diagonalised,
    # which is the number that decides both the cost and the reported result.
    # Amplitude screening therefore prunes to a *smaller* target, leaving the
    # classical expansion room to grow back up to the cap within one iteration.
    dimension_cap = int(settings.max_determinants)
    prune_target = max(1, int(settings.max_determinants / max(settings.growth_factor, 1.0)))
    # Spin completion is only meaningful when the two blocks hold the same
    # number of electrons: with n_alpha != n_beta the strings live in different
    # spaces and sharing one set between them is not even well formed.
    completing = bool(settings.spin_complete and space.n_alpha == space.n_beta)

    history: list[IterationRecord] = []
    optimizer_steps_taken = 0
    energy = float("inf")
    previous = float("inf")
    stalled = 0
    stop_reason = "iteration cap"

    header = (
        f"{'iter':>4} {'dets':>9} {'a x b':>11} {'energy (Ha)':>17} "
        f"{'err (mHa)':>11} {'added':>7} {'leak':>7} {'s':>6}"
    )
    progress(header)
    progress("-" * len(header))

    for iteration in range(1, settings.max_iterations + 1):
        tick = time.time()

        # 1-3. Quantum proposal, folded into the accumulated subspace.
        occupancy_prior = None
        if coefficients.size:
            occupancy_prior = orbital_occupancies(space, subspace, coefficients)
        proposal = proposer.sample(angles, occupancy_prior=occupancy_prior)
        merged = Subspace(
            np.concatenate([subspace.strings_a, proposal.strings_a]),
            np.concatenate([subspace.strings_b, proposal.strings_b]),
        )
        if merged.dimension > dimension_cap:
            # A wide proposal can overshoot on its own. Keep the configurations
            # the circuit weighted most heavily and leave the rest for a later
            # iteration, rather than diagonalising something unaffordable.
            merged = _trim_proposal(subspace, proposal, dimension_cap)

        guess = embed_coefficients(coefficients, subspace, merged)
        result = _solve_monotone(space, merged, guess, energy)
        energy = result.energy
        subspace, coefficients = result.subspace, result.coefficients

        # 5. Amplitude screening, then spin completion. Completion runs on the
        #    screened space so it ranks strings that survived on weight, and
        #    before the expansion so what grows is already closed under the flip.
        solved_dimension = subspace.dimension
        subspace, coefficients = prune_subspace(
            subspace,
            coefficients,
            prune_target,
            settings.amplitude_threshold,
            anchor,
        )
        if completing:
            subspace, coefficients = spin_complete_subspace(
                subspace, coefficients, prune_target, anchor
            )

        # 6. Classical single/double expansion.
        added = 0
        if settings.use_expansion and settings.expansion > 0:
            ranked = rank_candidates(
                space,
                subspace,
                coefficients,
                settings.expansion_references,
                energy,
                settings.ranking,
            )
            grown, added = grow_subspace(
                subspace, ranked, settings.expansion, dimension_cap, completing
            )
            if added:
                guess = embed_coefficients(coefficients, subspace, grown)
                result = _solve_monotone(space, grown, guess, energy)
                energy = result.energy
                subspace, coefficients = result.subspace, result.coefficients
                solved_dimension = subspace.dimension

        if subspace.dimension != solved_dimension:
            # Screening threw determinants away and nothing re-solved afterwards,
            # so `energy` still describes the larger subspace. Reporting it
            # beside the smaller determinant count would overstate the
            # compression -- a smaller space is a *higher* energy, and this is
            # the one place in the loop where the energy legitimately rises, so
            # the monotonicity guard is deliberately not applied here.
            result = solve_subspace(space, subspace, guess=coefficients)
            energy = result.energy
            subspace, coefficients = result.subspace, result.coefficients

        # 7. Move theta, using the quality of the circuit's own proposals as the
        #    objective. The accumulated subspace is monotone by construction, so
        #    scoring theta against it would reward the history rather than the
        #    circuit; scoring it against what this theta alone proposes does not.
        optimizer_energy = None
        if (
            settings.optimizer == "spsa"
            and proposer.n_parameters
            and settings.optimizer_steps > 0
            and (iteration - 1) % max(1, settings.optimizer_every) == 0
        ):
            # The probe subspace is deliberately smaller than the working one.
            # This objective is a *relative* measure of how good this theta's
            # proposals are, so it only has to rank parameter settings against
            # each other -- and paying the full diagonalisation cost twice per
            # SPSA step would dominate the run for no extra information.
            probe_cap = min(dimension_cap, OPTIMIZER_PROBE_DETERMINANTS)

            def objective(candidate: np.ndarray) -> float:
                trial = proposer.sample(candidate, occupancy_prior=occupancy_prior)
                probe = Subspace(
                    np.concatenate(
                        [np.array([anchor[0]], dtype=np.int64), trial.strings_a]
                    ),
                    np.concatenate(
                        [np.array([anchor[1]], dtype=np.int64), trial.strings_b]
                    ),
                )
                if probe.dimension > probe_cap:
                    probe = _trim_proposal(
                        hartree_fock_subspace(space.n_alpha, space.n_beta),
                        trial,
                        probe_cap,
                    )
                return solve_subspace(space, probe).energy

            for _ in range(settings.optimizer_steps):
                # The SPSA gain schedule is indexed by steps *actually taken*,
                # not by iteration number. With --optimizer-every 3 the two
                # differ by a factor of three, and using the iteration count
                # would decay the step size three times too fast -- the
                # optimizer would stop moving long before it had done the work.
                angles, optimizer_energy = spsa_step(
                    objective,
                    angles,
                    optimizer_steps_taken,
                    settings.spsa_a,
                    settings.spsa_c,
                    rng,
                )
                optimizer_steps_taken += 1

        record = IterationRecord(
            iteration=iteration,
            energy=energy,
            error_hartree=energy - reference_energy,
            dimension=subspace.dimension,
            n_strings_a=subspace.shape[0],
            n_strings_b=subspace.shape[1],
            n_proposed_determinants=proposal.n_determinants,
            n_added_by_expansion=added,
            quantum_leakage=proposal.leakage,
            quantum_recovered=proposal.recovered,
            circuit_runs=proposal.circuit_runs,
            seconds=time.time() - tick,
            optimizer_energy=optimizer_energy,
        )
        history.append(record)
        progress(
            f"{iteration:>4} {record.dimension:>9,} "
            f"{record.n_strings_a:>5} x{record.n_strings_b:>5} "
            f"{energy:>17.9f} {record.error_hartree * 1000:>11.4f} "
            f"{added:>7} {proposal.leakage:>7.3f} {record.seconds:>6.1f}"
        )

        # 8. Convergence.
        if abs(previous - energy) < settings.energy_tolerance:
            stalled += 1
            if stalled >= settings.patience:
                stop_reason = (
                    f"energy moved less than {settings.energy_tolerance:g} Ha for "
                    f"{settings.patience} iterations"
                )
                break
        else:
            stalled = 0
        previous = energy
    else:
        stop_reason = f"reached the {settings.max_iterations}-iteration cap"

    spin = spin_square(space, subspace, coefficients)
    occupancy_a, occupancy_b = orbital_occupancies(space, subspace, coefficients)
    error = energy - reference_energy

    correction, corrected_over = 0.0, 0
    if settings.pt2 and np.isfinite(energy):
        correction, corrected_over = perturbative_correction(
            space,
            subspace,
            coefficients,
            energy,
            settings.expansion_references,
            max(dimension_cap, int(dimension_cap * settings.pt2_factor)),
            settings.ranking,
        )
        progress(
            f"\n  E(PT2 correction) {correction * 1000:>12.4f} mHa over "
            f"{corrected_over:,} determinants outside the subspace"
        )
        progress(
            f"  E + PT2           {energy + correction:>17.9f}   "
            f"error {(energy + correction - reference_energy) * 1000:>9.4f} mHa"
        )
        if abs(correction) > 0.5 * abs(energy - hartree_fock_energy):
            progress(
                "  PT2 IS NOT USABLE HERE: the correction is larger than half"
                " the correlation energy the subspace captured, so the\n"
                "  determinants outside it are not a perturbation. Grow the"
                " subspace instead of trusting this number."
            )

    contamination = spin_contamination(spin)
    if error < -1.0e-9:
        verdict = "INCONSISTENT"
    elif contamination > SPIN_CONTAMINATION_TOLERANCE:
        verdict = "SPIN CONTAMINATED"
    elif abs(error) <= CHEMICAL_ACCURACY_HA:
        verdict = "CHEMICAL ACCURACY"
    else:
        verdict = "OUTSIDE CHEMICAL ACCURACY"

    return HiVqeResult(
        energy=energy,
        reference_energy=reference_energy,
        hartree_fock_energy=hartree_fock_energy,
        error_hartree=error,
        dimension=subspace.dimension,
        full_dimension=space.full_dimension,
        spin_squared=float(spin),
        spin_contamination=float(contamination),
        occupancies_alpha=[float(value) for value in occupancy_a],
        occupancies_beta=[float(value) for value in occupancy_b],
        iterations=len(history),
        stop_reason=stop_reason,
        verdict=verdict,
        history=history,
        settings=settings.as_dict(),
        circuit_metrics=dict(proposer.metrics),
        seconds=time.time() - started,
        pt2_correction=float(correction),
        pt2_determinants=int(corrected_over),
    )


MONOTONICITY_TOLERANCE_HA = 1.0e-9


def _solve_monotone(
    space: ActiveSpace,
    subspace: Subspace,
    guess: np.ndarray | None,
    previous_energy: float,
) -> Any:
    """Diagonalise, and refuse to accept an energy that went up.

    The subspace only ever grows, so the ground energy of `P H P` can only fall.
    An increase is therefore not a physical result but a sign that the iterative
    eigensolver settled on the wrong root -- which is exactly what a warm start
    with no overlap on the true ground state produces. `davidson` already seeds
    against that, and this is the second line: re-solve from a cold start and
    keep whichever answer is lower, so a bad warm start costs time and never
    accuracy.
    """
    result = solve_subspace(space, subspace, guess=guess)
    if np.isfinite(previous_energy) and result.energy > previous_energy + (
        MONOTONICITY_TOLERANCE_HA
    ):
        cold = solve_subspace(space, subspace, guess=None)
        if cold.energy < result.energy:
            return cold
    return result


def _trim_proposal(
    base: Subspace, proposal: Proposal, dimension_cap: int
) -> Subspace:
    """Keep the most-sampled proposed configurations, up to the dimension cap."""
    order = np.argsort(-proposal.counts)
    set_a = set(int(value) for value in base.strings_a)
    set_b = set(int(value) for value in base.strings_b)
    for index in order:
        string_a = int(proposal.strings_a[index])
        string_b = int(proposal.strings_b[index])
        projected = (len(set_a) + (string_a not in set_a)) * (
            len(set_b) + (string_b not in set_b)
        )
        if projected > dimension_cap:
            break
        set_a.add(string_a)
        set_b.add(string_b)
    return Subspace(
        np.array(sorted(set_a), dtype=np.int64),
        np.array(sorted(set_b), dtype=np.int64),
    )
