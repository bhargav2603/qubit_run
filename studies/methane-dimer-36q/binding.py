"""Binding energies and variance extrapolation.

Two things in this module are easy to get subtly wrong and expensive to notice
later, so both are enforced here rather than left to callers.

1. The binding energy is a difference of *dimer* energies at two separations,
   never ``E_AB - E_A - E_B``.  The paper's Eq. (2) uses the dimer at 48.000 A
   as the unbound reference precisely so the active space is the same object on
   both sides of the subtraction and its error cancels.  Passing monomer
   energies into ``binding_energy`` is a bug, and there is no overload that
   accepts them.

2. The paper's extrapolation is in the energy *variance*, not in the sample
   count.  SI Figure 5 defines

       dH = <psi|H^2|psi> - <psi|H|psi>^2

   and extrapolates total energy linearly to dH -> 0.  That works because a
   selected-CI energy approaches the exact eigenvalue linearly in the variance
   of the truncated state, which a fit against |chi_b| does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# CODATA 2018.
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
MILLIHARTREE = 1.0e-3

# 1 kcal/mol, the conventional chemical-accuracy threshold, in hartree.
CHEMICAL_ACCURACY_HARTREE = 1.0 / HARTREE_TO_KCAL_PER_MOL


def hartree_to_kcal(energy: float) -> float:
    return energy * HARTREE_TO_KCAL_PER_MOL


def kcal_to_hartree(energy: float) -> float:
    return energy / HARTREE_TO_KCAL_PER_MOL


def binding_energy(
    dimer_energy: float,
    unbound_energy: float,
    *,
    unit: str = "hartree",
) -> float:
    """Paper Eq. (2): ``E_bind(R) = E_dimer(R) - E_dimer(48.000 A)``.

    Both arguments must be **dimer** total energies computed in the *same*
    active space. A negative result means bound.
    """
    delta = dimer_energy - unbound_energy
    if unit == "hartree":
        return delta
    if unit in ("kcal", "kcal/mol"):
        return hartree_to_kcal(delta)
    raise ValueError(f"unknown unit {unit!r}; use 'hartree' or 'kcal/mol'")


def binding_curve(
    energies: dict[float, float],
    *,
    unbound_distance: float,
    unit: str = "kcal/mol",
) -> dict[float, float]:
    """Convert a {distance: total energy} map into a binding-energy curve.

    The unbound point is consumed as the reference and does not appear in the
    output. Its absence from ``energies`` is an error, not a default.
    """
    if unbound_distance not in energies:
        raise KeyError(
            f"no total energy at the unbound reference {unbound_distance} A; "
            "the binding energy is undefined without it (paper Eq. 2)"
        )
    reference = energies[unbound_distance]
    return {
        distance: binding_energy(energy, reference, unit=unit)
        for distance, energy in sorted(energies.items())
        if distance != unbound_distance
    }


# --------------------------------------------------------------------------
# Variance extrapolation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Extrapolation:
    """Linear extrapolation of energy to zero variance."""

    energy: float           # intercept: the dH -> 0 estimate
    stderr: float           # standard error on the intercept
    slope: float
    r_squared: float
    n_points: int

    @property
    def energy_error_mha(self) -> float:
        return self.stderr / MILLIHARTREE

    def __str__(self) -> str:
        return (
            f"E(dH->0) = {self.energy:.8f} +/- {self.stderr:.8f} Ha "
            f"({self.energy_error_mha:.2f} mHa), R^2 = {self.r_squared:.6f}, "
            f"n = {self.n_points}"
        )


def variance_extrapolate(
    energies: "np.ndarray | list[float]",
    variances: "np.ndarray | list[float]",
) -> Extrapolation:
    """Extrapolate total energy linearly to zero energy variance.

    Fits ``E = a + b * dH`` by ordinary least squares and returns the intercept
    ``a`` with its standard error, which is what the paper's error bars show.

    Args:
        energies:  total energies, in hartree.
        variances: matching energy variances dH, in hartree^2.

    Raises:
        ValueError: on fewer than three points, mismatched lengths, or
            negative variances -- each of which indicates a broken upstream
            calculation rather than a hard extrapolation.
    """
    e = np.asarray(energies, dtype=float)
    v = np.asarray(variances, dtype=float)

    if e.shape != v.shape:
        raise ValueError(f"shape mismatch: {e.shape} energies vs {v.shape} variances")
    if e.size < 3:
        raise ValueError(
            f"need at least 3 points to extrapolate with an error estimate, got {e.size}"
        )
    if np.any(v < 0.0):
        raise ValueError("negative energy variance: the eigensolver did not converge")
    if np.ptp(v) == 0.0:
        raise ValueError("all variances identical; the fit is degenerate")

    n = e.size
    design = np.vstack([np.ones(n), v]).T
    coefficients, residuals, *_ = np.linalg.lstsq(design, e, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])

    fitted = design @ coefficients
    ss_residual = float(np.sum((e - fitted) ** 2))
    ss_total = float(np.sum((e - e.mean()) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else 1.0

    # Standard error of the intercept for a straight-line fit.
    dof = n - 2
    if dof > 0 and ss_residual > 0.0:
        sigma_sq = ss_residual / dof
        v_mean = float(v.mean())
        s_vv = float(np.sum((v - v_mean) ** 2))
        stderr = math.sqrt(sigma_sq * (1.0 / n + v_mean**2 / s_vv))
    else:
        stderr = 0.0

    return Extrapolation(
        energy=intercept,
        stderr=stderr,
        slope=slope,
        r_squared=r_squared,
        n_points=n,
    )


# --------------------------------------------------------------------------
# Agreement reporting
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Agreement:
    """Signed deviation of a computed energy from a reference."""

    computed: float
    reference: float
    label: str = "SQD vs CASCI"

    @property
    def delta_hartree(self) -> float:
        return self.computed - self.reference

    @property
    def delta_mha(self) -> float:
        return self.delta_hartree / MILLIHARTREE

    @property
    def delta_kcal(self) -> float:
        return hartree_to_kcal(self.delta_hartree)

    @property
    def within_chemical_accuracy(self) -> bool:
        return abs(self.delta_kcal) <= 1.0

    def meets(self, threshold_kcal: float) -> bool:
        return abs(self.delta_kcal) <= threshold_kcal

    def __str__(self) -> str:
        verdict = "PASS" if self.within_chemical_accuracy else "OUTSIDE 1 kcal/mol"
        return (
            f"{self.label}: {self.delta_mha:+.4f} mHa "
            f"({self.delta_kcal:+.4f} kcal/mol)  [{verdict}]"
        )


def variational_check(computed: float, reference: float, *, tolerance: float = 1e-9) -> None:
    """A subspace energy can never fall below the exact energy in that space.

    ``P H P`` has its lowest eigenvalue at or above the lowest eigenvalue of
    ``H`` for any projector ``P``, so SQD is a strict upper bound on CASCI in
    the same active space. Device noise can only make the subspace worse. A
    violation therefore means a bug -- mismatched integrals, a different
    geometry, or a broken active space -- and never a good result.
    """
    if computed < reference - tolerance:
        raise AssertionError(
            f"variational bound violated: computed {computed:.10f} Ha lies "
            f"{(reference - computed) / MILLIHARTREE:.4f} mHa BELOW the reference "
            f"{reference:.10f} Ha. SQD cannot beat exact diagonalization in the "
            "same space; this indicates the two were not computed for the same "
            "Hamiltonian."
        )
