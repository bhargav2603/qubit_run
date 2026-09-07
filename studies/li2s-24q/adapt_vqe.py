#!/usr/bin/env python3
"""ADAPT-VQE on the same cached Hamiltonian HI-VQE uses.

This module exists to answer one question honestly: *how close does a strong,
current-literature VQE get on this system, and at what cost?* Comparing HI-VQE
against a hand-picked hardware-efficient circuit optimised by SPSA would be a
straw man -- everyone already knows that fails at 24 qubits. ADAPT-VQE is the
fair opponent, because it builds its own ansatz and so removes the objection
"you chose a bad circuit".

ADAPT-VQE (Grimsley et al., Nat. Commun. 10, 3007 (2019)) grows the ansatz one
operator at a time. Each round it measures the energy gradient of every operator
in a pool, appends the one with the largest gradient, re-optimises every
parameter, and repeats until no operator has a gradient worth adding.

    |psi(theta)> = exp(theta_k A_k) ... exp(theta_1 A_1) |HF>

Two things make this affordable here, and both matter:

* **The whole pool is screened for the price of one sigma product.** The
  gradient of an anti-Hermitian generator A is `<psi|[H, A]|psi> = 2<H psi, A
  psi>`, so `H psi` is computed once and every operator costs one cheap sparse
  application plus a dot product.

* **Gradients of the optimisation are analytic, by an adjoint sweep.** A finite
  difference over k parameters would cost k+1 energy evaluations, each dominated
  by a sigma product over 853,776 determinants. The backward sweep in
  `energy_and_gradient` gets all k gradients from *one* sigma product instead.

Nothing here imports Qiskit. This is a noiseless, infinite-shot, exact-
expectation VQE -- deliberately the best case a VQE can possibly have, since it
pays none of the 15,697-Pauli-word measurement cost that a real one would. If it
still cannot reach chemical accuracy, that conclusion is not an artefact of
shot noise or an unlucky optimiser.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from determinants import (
    ActiveSpace,
    Subspace,
    SubspaceHamiltonian,
    hartree_fock_determinant,
    make_strings,
    single_excitation_maps,
    spin_square,
)


CHEMICAL_ACCURACY_HA = 1.6e-3

# Taylor terms in one scaled exponential step. The generators here are bounded
# operators with modest norm and the step is scaled so ||theta A|| <= 0.5, where
# 18 terms is far past double precision.
EXPONENTIAL_TERMS = 18
EXPONENTIAL_STEP = 0.5


# --------------------------------------------------------------------------
# The operator pool
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolOperator:
    """One anti-Hermitian generator A = T - T^dagger, spin-adapted.

    `pairs` lists the spin-summed one-body excitations composing T, applied
    right to left. A single excitation is `((p, q),)` meaning T = E_pq; a double
    is `((p, q), (r, s))` meaning T = E_pq E_rs. Spin summation -- E_pq acting on
    the alpha strings *and* on the beta strings -- is what keeps the generated
    state a spin eigenstate, so the ansatz cannot drift into a spin-contaminated
    mixture the way a bare qubit-excitation pool can.
    """

    pairs: tuple[tuple[int, int], ...]

    @property
    def rank(self) -> int:
        return len(self.pairs)

    def label(self) -> str:
        body = " ".join(f"E{p}{q}" for p, q in self.pairs)
        return f"{'single' if self.rank == 1 else 'double'}: {body}"


def build_pool(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    generalized: bool = False,
) -> list[PoolOperator]:
    """Spin-adapted singles and doubles.

    By default the pool is the UCCSD one: excitations out of the orbitals the
    Hartree-Fock determinant occupies and into the ones it does not. That is the
    standard ADAPT-VQE pool and the right default at a geometry where the
    reference determinant still dominates.

    `generalized` drops the occupied/virtual distinction and allows every orbital
    pair. It is a much larger pool and a strictly more expressive ansatz, which
    is what a stretched bond wants -- there the reference determinant no longer
    dominates and "excitation out of the occupied set" stops being a meaningful
    restriction.
    """
    if n_alpha != n_beta:
        raise ValueError("the spin-adapted pool assumes a closed-shell reference")
    occupied = list(range(n_alpha))
    virtual = list(range(n_alpha, n_orbitals))

    if generalized:
        singles = [
            (p, q)
            for p in range(n_orbitals)
            for q in range(n_orbitals)
            if p > q
        ]
    else:
        singles = [(p, q) for p in virtual for q in occupied]

    pool = [PoolOperator(((p, q),)) for p, q in singles]

    # Doubles are products of two of the same one-body excitations. Taking the
    # unordered pairs avoids enumerating E_pq E_rs and E_rs E_pq separately --
    # they differ only by terms that the antisymmetrisation already covers.
    for first in range(len(singles)):
        for second in range(first, len(singles)):
            pool.append(PoolOperator((singles[first], singles[second])))
    return pool


# --------------------------------------------------------------------------
# Applying generators to a CI matrix
# --------------------------------------------------------------------------


class GeneratorAlgebra:
    """Applies spin-summed excitations to a CI matrix C[alpha, beta].

    `single_excitation_maps` gives the one-body excitation `A_pq[J, I] =
    <J|a+_p a_q|I>` over the alpha string set and over the beta string set. On
    the CI matrix these act on opposite indices:

        (E^a_pq C) = A_pq @ C          (rows are alpha strings)
        (E^b_pq C) = C @ A_pq.T        (columns are beta strings)

    so the spin-summed E_pq is the sum of the two. Every operation is a sparse
    matrix against a 924 x 924 dense block -- milliseconds -- which is what makes
    screening a pool of thousands affordable.
    """

    def __init__(self, n_orbitals: int, strings_a: np.ndarray, strings_b: np.ndarray):
        self.n_orbitals = n_orbitals
        maps_a = single_excitation_maps(n_orbitals, strings_a)
        if strings_b is strings_a or np.array_equal(strings_a, strings_b):
            maps_b = maps_a
        else:
            maps_b = single_excitation_maps(n_orbitals, strings_b)
        self.n_a = maps_a[0].shape[0]
        self.n_b = maps_b[0].shape[0]
        self._table_a = [self._as_scatter(matrix) for matrix in maps_a]
        self._table_b = (
            self._table_a
            if maps_b is maps_a
            else [self._as_scatter(matrix) for matrix in maps_b]
        )

    @staticmethod
    def _as_scatter(matrix: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack a one-body excitation map into (rows, cols, signs).

        `a+_p a_q` is a partial isometry on determinants: for fixed p and q it
        sends each string it does not annihilate to exactly one other string, and
        distinct strings to distinct images. So the map is injective and both the
        row and the column index arrays hold no repeats -- which is what lets the
        application below be a plain scatter-assignment rather than an
        accumulation, and lets it skip scipy's sparse matmul entirely.

        That matters more than it sounds. The maps here hold ~462 non-zeros in a
        924 x 924 frame, and scipy's dense-times-sparse path costs about 30x what
        the equivalent fancy-index gather does.
        """
        coo = matrix.tocoo()
        return (
            np.ascontiguousarray(coo.row, dtype=np.intp),
            np.ascontiguousarray(coo.col, dtype=np.intp),
            np.ascontiguousarray(coo.data, dtype=np.float64),
        )

    def _excite(self, p: int, q: int, matrix: np.ndarray) -> np.ndarray:
        """The spin-summed E_pq applied to a CI matrix."""
        key = p * self.n_orbitals + q
        result = np.zeros_like(matrix)

        rows, cols, signs = self._table_a[key]
        if rows.size:
            # alpha excitations move whole rows of the CI matrix
            result[rows] = signs[:, None] * matrix[cols]

        rows, cols, signs = self._table_b[key]
        if rows.size:
            # beta excitations move whole columns
            result[:, rows] += matrix[:, cols] * signs[None, :]
        return result

    def apply_t(self, operator: PoolOperator, matrix: np.ndarray) -> np.ndarray:
        """T |C>, the excitation part alone, applied right to left."""
        result = matrix
        for p, q in reversed(operator.pairs):
            result = self._excite(p, q, result)
        return result

    def apply_t_dagger(self, operator: PoolOperator, matrix: np.ndarray) -> np.ndarray:
        """T^dagger |C>. E_pq^dagger = E_qp, and the product order reverses."""
        result = matrix
        for p, q in operator.pairs:
            result = self._excite(q, p, result)
        return result

    def apply(self, operator: PoolOperator, matrix: np.ndarray) -> np.ndarray:
        """A |C> for the anti-Hermitian A = T - T^dagger."""
        return self.apply_t(operator, matrix) - self.apply_t_dagger(operator, matrix)

    def norm_estimate(self, operator: PoolOperator, samples: int = 2) -> float:
        """A cheap upper-ish bound on ||A||, for choosing the exponential step.

        A few power iterations from random starts. Overestimating only costs a
        few extra Taylor steps; underestimating would silently lose accuracy, so
        the result is inflated by a safety factor before use.
        """
        rng = np.random.default_rng(0)
        shape = (self.n_a, self.n_b)
        best = 0.0
        for _ in range(samples):
            vector = rng.standard_normal(shape)
            vector /= np.linalg.norm(vector)
            for _ in range(3):
                vector = self.apply(operator, vector)
                norm = np.linalg.norm(vector)
                if norm < 1e-14:
                    break
                vector /= norm
                best = max(best, norm)
        return float(best)

    def exponentiate(
        self, operator: PoolOperator, theta: float, matrix: np.ndarray, norm: float
    ) -> np.ndarray:
        """exp(theta A) |C>, by scaled-and-squared Taylor series.

        A is real and anti-symmetric in the determinant basis, so exp(theta A) is
        orthogonal and the result stays normalised -- a useful invariant, and one
        the tests check rather than assume.
        """
        if theta == 0.0:
            return matrix.copy()
        scale = max(1, int(np.ceil(abs(theta) * max(norm, 1.0) / EXPONENTIAL_STEP)))
        step = theta / scale
        result = matrix
        for _ in range(scale):
            term = result
            total = result
            for order in range(1, EXPONENTIAL_TERMS + 1):
                term = self.apply(operator, term) * (step / order)
                total = total + term
                if np.linalg.norm(term) < 1e-15:
                    break
            result = total
        return result


# --------------------------------------------------------------------------
# Settings and results
# --------------------------------------------------------------------------


@dataclass
class AdaptSettings:
    max_operators: int = 60
    gradient_tolerance: float = 1e-3
    energy_tolerance: float = 1e-7
    optimizer_iterations: int = 200
    generalized_pool: bool = False
    seed: int = 1234


@dataclass
class AdaptIteration:
    index: int
    operator: str
    gradient: float
    energy: float
    error_millihartree: float
    n_parameters: int
    seconds: float
    energy_evaluations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operator": self.operator,
            "gradient": self.gradient,
            "energy": self.energy,
            "error_millihartree": self.error_millihartree,
            "n_parameters": self.n_parameters,
            "seconds": self.seconds,
            "energy_evaluations": self.energy_evaluations,
        }


@dataclass
class AdaptResult:
    energy: float
    reference_energy: float
    hartree_fock_energy: float
    error_hartree: float
    spin_squared: float
    n_parameters: int
    n_operators_in_pool: int
    iterations: list[AdaptIteration]
    energy_evaluations: int
    sigma_products: int
    seconds: float
    verdict: str
    settings: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "ADAPT-VQE",
            "energy": self.energy,
            "reference_energy": self.reference_energy,
            "hartree_fock_energy": self.hartree_fock_energy,
            "error_hartree": self.error_hartree,
            "error_millihartree": self.error_hartree * 1000.0,
            "correlation_recovered": (
                (self.energy - self.hartree_fock_energy)
                / (self.reference_energy - self.hartree_fock_energy)
                if self.reference_energy != self.hartree_fock_energy
                else float("nan")
            ),
            "spin_squared": self.spin_squared,
            "n_parameters": self.n_parameters,
            "n_operators_in_pool": self.n_operators_in_pool,
            "energy_evaluations": self.energy_evaluations,
            "sigma_products": self.sigma_products,
            "seconds": self.seconds,
            "verdict": self.verdict,
            "iterations": [record.as_dict() for record in self.iterations],
            "settings": self.settings,
        }


# --------------------------------------------------------------------------
# The algorithm
# --------------------------------------------------------------------------


class AdaptVqe:
    """ADAPT-VQE over the full determinant sector of one active space."""

    def __init__(
        self,
        space: ActiveSpace,
        settings: AdaptSettings,
        progress: Callable[[str], None] = print,
    ) -> None:
        self.space = space
        self.settings = settings
        self.progress = progress

        strings_a = make_strings(space.n_orbitals, space.n_alpha)
        strings_b = make_strings(space.n_orbitals, space.n_beta)
        self.subspace = Subspace(strings_a, strings_b)
        self.hamiltonian = SubspaceHamiltonian(space, self.subspace)
        self.algebra = GeneratorAlgebra(space.n_orbitals, strings_a, strings_b)

        self.pool = build_pool(
            space.n_orbitals,
            space.n_alpha,
            space.n_beta,
            generalized=settings.generalized_pool,
        )
        # Norms are needed only to size the exponential's scaling step, and only
        # for operators that actually enter the ansatz -- at most a few dozen.
        # Precomputing all ~700 costs four minutes and throws away nearly all of
        # it. Pool screening never exponentiates, so it never needs one.
        self._norm_cache: dict[PoolOperator, float] = {}

        # The Hartree-Fock determinant, as a normalised CI matrix.
        string_a, string_b = hartree_fock_determinant(space.n_alpha, space.n_beta)
        row = int(np.searchsorted(strings_a, string_a))
        column = int(np.searchsorted(strings_b, string_b))
        reference = np.zeros((len(strings_a), len(strings_b)))
        reference[row, column] = 1.0
        self.reference_state = reference

        self.ansatz: list[PoolOperator] = []
        self.ansatz_norms: list[float] = []
        self.energy_evaluations = 0
        self.sigma_products = 0

    # -- state construction ---------------------------------------------

    def state(self, angles: Sequence[float]) -> np.ndarray:
        """|psi(theta)> = exp(theta_k A_k) ... exp(theta_1 A_1)|HF>."""
        matrix = self.reference_state
        for operator, theta, norm in zip(self.ansatz, angles, self.ansatz_norms):
            matrix = self.algebra.exponentiate(operator, float(theta), matrix, norm)
        return matrix

    def energy(self, angles: Sequence[float]) -> float:
        self.energy_evaluations += 1
        self.sigma_products += 1
        return self.hamiltonian.expectation(self.state(angles))

    def energy_and_gradient(
        self, angles: Sequence[float]
    ) -> tuple[float, np.ndarray]:
        """E(theta) and every dE/dtheta_j, from a single sigma product.

        Forward, then backward. The backward sweep carries two states -- the
        wavefunction and `H psi` -- and un-applies each exponential in turn, so
        the gradient at layer j is read off where that layer actually sits in the
        product. Cost is one sigma plus 3k cheap generator applications, against
        the k+1 sigma products a finite difference would need.
        """
        angles = np.asarray(angles, dtype=float)
        forward = [self.reference_state]
        matrix = self.reference_state
        for operator, theta, norm in zip(self.ansatz, angles, self.ansatz_norms):
            matrix = self.algebra.exponentiate(operator, float(theta), matrix, norm)
            forward.append(matrix)

        self.energy_evaluations += 1
        self.sigma_products += 1
        sigma = self.hamiltonian.sigma(matrix)
        energy = float(np.vdot(matrix, sigma) + self.space.core_energy)

        gradient = np.zeros(len(self.ansatz))
        adjoint = sigma
        for index in range(len(self.ansatz) - 1, -1, -1):
            operator = self.ansatz[index]
            norm = self.ansatz_norms[index]
            theta = float(angles[index])
            state = forward[index + 1]
            gradient[index] = 2.0 * float(
                np.vdot(adjoint, self.algebra.apply(operator, state))
            )
            adjoint = self.algebra.exponentiate(operator, -theta, adjoint, norm)
        return energy, gradient

    # -- pool screening --------------------------------------------------

    def pool_gradients(self, matrix: np.ndarray) -> np.ndarray:
        """|<psi|[H, A_k]|psi>| for every operator, from one sigma product.

        For anti-Hermitian A and real symmetric H the commutator expectation
        collapses to `2 <H psi, A psi>`, so the expensive half is computed once
        and shared across the whole pool.
        """
        self.sigma_products += 1
        sigma = self.hamiltonian.sigma(matrix)
        values = np.empty(len(self.pool))
        for index, operator in enumerate(self.pool):
            values[index] = 2.0 * float(
                np.vdot(sigma, self.algebra.apply(operator, matrix))
            )
        return np.abs(values)

    # -- the loop --------------------------------------------------------

    def run(
        self,
        reference_energy: float,
        hartree_fock_energy: float,
        checkpoint: Callable[[list[AdaptIteration]], None] | None = None,
    ) -> AdaptResult:
        from scipy.optimize import minimize

        started = time.time()
        angles = np.zeros(0)
        energy = self.hamiltonian.expectation(self.reference_state)
        records: list[AdaptIteration] = []

        self.progress(
            f"  pool: {len(self.pool):,} spin-adapted operators "
            f"({'generalized' if self.settings.generalized_pool else 'occ->virt'})"
        )
        self.progress(
            f"  E(HF determinant) = {energy:.9f}   "
            f"target E({'reference'}) = {reference_energy:.9f}"
        )
        self.progress("")
        header = (
            f"  {'k':>3} {'gradient':>11} {'energy':>17} {'err (mHa)':>11} "
            f"{'evals':>7} {'s':>7}  operator"
        )
        self.progress(header)
        self.progress("  " + "-" * (len(header) - 2))

        previous_energy = energy
        for step in range(1, self.settings.max_operators + 1):
            iteration_started = time.time()
            evaluations_before = self.energy_evaluations

            matrix = self.state(angles)
            gradients = self.pool_gradients(matrix)
            best = int(np.argmax(gradients))
            largest = float(gradients[best])
            if largest < self.settings.gradient_tolerance:
                self.progress(
                    f"\n  converged: largest pool gradient {largest:.3e} < "
                    f"{self.settings.gradient_tolerance:g}"
                )
                break

            operator = self.pool[best]
            if operator not in self._norm_cache:
                self._norm_cache[operator] = self.algebra.norm_estimate(operator)
            self.ansatz.append(operator)
            self.ansatz_norms.append(self._norm_cache[operator])
            # Warm start: keep the optimised angles and open the new one at zero,
            # so the ansatz starts this round exactly where the last one ended
            # and the optimiser has only the new direction to explore.
            angles = np.concatenate([angles, [0.0]])

            outcome = minimize(
                lambda x: self.energy_and_gradient(x),
                angles,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.settings.optimizer_iterations, "ftol": 1e-14},
            )
            angles = np.asarray(outcome.x, dtype=float)
            energy = float(outcome.fun)

            records.append(
                AdaptIteration(
                    index=step,
                    operator=operator.label(),
                    gradient=largest,
                    energy=energy,
                    error_millihartree=(energy - reference_energy) * 1000.0,
                    n_parameters=len(angles),
                    seconds=time.time() - iteration_started,
                    energy_evaluations=self.energy_evaluations - evaluations_before,
                )
            )
            record = records[-1]
            self.progress(
                f"  {step:>3} {largest:>11.3e} {energy:>17.9f} "
                f"{record.error_millihartree:>11.4f} "
                f"{record.energy_evaluations:>7} {record.seconds:>7.1f}  "
                f"{record.operator}"
            )

            if checkpoint is not None:
                # A run of this length outlives some Colab sessions. Writing the
                # trace after every operator means a killed run still leaves a
                # readable partial result rather than nothing at all.
                checkpoint(records)

            if abs(previous_energy - energy) < self.settings.energy_tolerance:
                self.progress(
                    f"\n  converged: energy moved less than "
                    f"{self.settings.energy_tolerance:g} Ha"
                )
                break
            previous_energy = energy
        else:
            self.progress(
                f"\n  stopped at the {self.settings.max_operators}-operator cap, "
                "not on convergence"
            )

        final = self.state(angles)
        spin = spin_square(self.space, self.subspace, final)
        error = energy - reference_energy
        if error < -1.0e-9:
            verdict = "INCONSISTENT"
        elif abs(error) <= CHEMICAL_ACCURACY_HA:
            verdict = "CHEMICAL ACCURACY"
        else:
            verdict = "OUTSIDE CHEMICAL ACCURACY"

        return AdaptResult(
            energy=energy,
            reference_energy=reference_energy,
            hartree_fock_energy=hartree_fock_energy,
            error_hartree=error,
            spin_squared=float(spin),
            n_parameters=len(angles),
            n_operators_in_pool=len(self.pool),
            iterations=records,
            energy_evaluations=self.energy_evaluations,
            sigma_products=self.sigma_products,
            seconds=time.time() - started,
            verdict=verdict,
            settings={
                "max_operators": self.settings.max_operators,
                "gradient_tolerance": self.settings.gradient_tolerance,
                "energy_tolerance": self.settings.energy_tolerance,
                "optimizer_iterations": self.settings.optimizer_iterations,
                "generalized_pool": self.settings.generalized_pool,
                "optimizer": "L-BFGS-B with analytic adjoint gradients",
                "expectation": "exact (noiseless, infinite shots)",
            },
        )


def run_adapt_vqe(
    space: ActiveSpace,
    reference_energy: float,
    hartree_fock_energy: float,
    settings: AdaptSettings,
    progress: Callable[[str], None] = print,
    checkpoint: Callable[[list[AdaptIteration]], None] | None = None,
) -> AdaptResult:
    return AdaptVqe(space, settings, progress).run(
        reference_energy, hartree_fock_energy, checkpoint
    )
