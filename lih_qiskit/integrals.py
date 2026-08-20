#!/usr/bin/env python3
"""Self-contained STO-3G integrals and closed-shell RHF, in NumPy only.

Why this file exists
--------------------
PySCF publishes no Windows wheel, so `prepare` -- the one command that needs
real molecular integrals -- could not run on this machine at all. Everything
downstream (validate, vqe) consumes a JSON cache, so the whole workflow was
blocked on a single dependency that cannot be installed here.

LiH/STO-3G is 6 basis functions. At that size the integrals are a few hundred
lines of McMurchie-Davidson recursion, which is a smaller and far more auditable
dependency than a Fortran-backed quantum chemistry package. This module is that
implementation. It is deliberately *not* a general chemistry code: it supports
contracted Cartesian Gaussians up to p functions and closed-shell RHF, because
that is exactly what LiH/STO-3G needs and nothing more.

`chemistry.py` prefers PySCF when it is importable and falls back to this, so a
cache built on Linux and a cache built here are produced by the same downstream
code path and are directly comparable. The integral backend is recorded in the
cache metadata as `integral_backend` so a result always says which engine made
it.

Conventions
-----------
* Geometry in Angstrom on the way in, Bohr everywhere internally.
* Two-electron integrals are returned in **chemist** notation ``(pq|rs)``,
  matching ``pyscf.ao2mo.restore(1, ...)``. `chemistry.py` applies the same
  ``transpose(0, 2, 3, 1)`` that openfermionpyscf applies, so the OpenFermion
  physicist-notation convention is reached by the identical route either way.

Accuracy is checked against published LiH/STO-3G and H2/STO-3G energies in
`tests/test_integrals.py`, not asserted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.special import gamma as _gamma, gammainc as _gammainc

BOHR_PER_ANGSTROM = 1.0 / 0.52917721092
"""CODATA 2010 Bohr radius, matching PySCF's default so geometries agree."""

ATOMIC_NUMBER = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8}

# STO-3G, verbatim from the Basis Set Exchange (https://www.basissetexchange.org,
# basis "STO-3G", elements H and Li). Each entry is (angular_momentum, [(exponent,
# contraction_coefficient), ...]). The Li L=0/L=1 pair at identical exponents is
# the SP shell, split here into two shells because this code contracts each
# angular momentum separately.
STO3G: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {
    "H": [
        (
            0,
            [
                (0.3425250914e01, 0.1543289673e00),
                (0.6239137298e00, 0.5353281423e00),
                (0.1688554040e00, 0.4446345422e00),
            ],
        )
    ],
    "Li": [
        (
            0,
            [
                (0.1611957475e02, 0.1543289673e00),
                (0.2936200663e01, 0.5353281423e00),
                (0.7946504870e00, 0.4446345422e00),
            ],
        ),
        (
            0,
            [
                (0.6362897469e00, -0.9996722919e-01),
                (0.1478600533e00, 0.3995128261e00),
                (0.4808867840e-01, 0.7001154689e00),
            ],
        ),
        (
            1,
            [
                (0.6362897469e00, 0.1559162750e00),
                (0.1478600533e00, 0.6076837186e00),
                (0.4808867840e-01, 0.3919573931e00),
            ],
        ),
    ],
}

# Cartesian components per angular momentum. Only s and p are needed for
# STO-3G rows 1-2; for l <= 1 Cartesian and spherical bases coincide, so no
# spherical transformation is required.
_CARTESIAN: dict[int, list[tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
}


@dataclass(frozen=True)
class ContractedGaussian:
    """One contracted Cartesian Gaussian basis function."""

    center: tuple[float, float, float]  # Bohr
    powers: tuple[int, int, int]
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]  # already include primitive normalization


@dataclass
class MeanField:
    """Converged closed-shell RHF solution plus the AO integrals behind it."""

    n_electrons: int
    n_orbitals: int
    mo_coeff: np.ndarray  # (nao, nmo)
    mo_energy: np.ndarray  # (nmo,)
    e_hf: float
    e_nuc: float
    hcore: np.ndarray
    eri: np.ndarray  # chemist notation (pq|rs), AO basis
    overlap: np.ndarray
    converged: bool
    n_iterations: int


# --------------------------------------------------------------------------
# Primitive integrals (McMurchie-Davidson)
# --------------------------------------------------------------------------


def _double_factorial(n: int) -> float:
    """(2k-1)!! with the standard (-1)!! == 1 convention."""
    if n <= 0:
        return 1.0
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def _primitive_norm(exponent: float, powers: tuple[int, int, int]) -> float:
    l, m, n = powers
    numerator = (2.0 * exponent / math.pi) ** 0.75 * (4.0 * exponent) ** (
        0.5 * (l + m + n)
    )
    denominator = math.sqrt(
        _double_factorial(2 * l - 1)
        * _double_factorial(2 * m - 1)
        * _double_factorial(2 * n - 1)
    )
    return numerator / denominator


def _boys(n: int, t: float) -> float:
    """F_n(t) = int_0^1 s^(2n) exp(-t s^2) ds.

    The incomplete-gamma form is stable at large t, where the confluent
    hypergeometric series suffers catastrophic cancellation; the two-term
    Taylor expansion covers the removable 0/0 at t -> 0.
    """
    if t < 1.0e-10:
        return 1.0 / (2 * n + 1) - t / (2 * n + 3)
    a = n + 0.5
    return _gammainc(a, t) * _gamma(a) / (2.0 * t**a)


@lru_cache(maxsize=None)
def _hermite(i: int, j: int, t: int, distance: float, a: float, b: float) -> float:
    """Expansion coefficient E_t^{ij} of a Gaussian product in Hermite functions."""
    if t < 0 or i < 0 or j < 0 or t > i + j:
        return 0.0
    p = a + b
    mu = a * b / p
    if i == 0 and j == 0:
        return math.exp(-mu * distance * distance)
    if j == 0:
        return (
            _hermite(i - 1, j, t - 1, distance, a, b) / (2.0 * p)
            - (mu * distance / a) * _hermite(i - 1, j, t, distance, a, b)
            + (t + 1) * _hermite(i - 1, j, t + 1, distance, a, b)
        )
    return (
        _hermite(i, j - 1, t - 1, distance, a, b) / (2.0 * p)
        + (mu * distance / b) * _hermite(i, j - 1, t, distance, a, b)
        + (t + 1) * _hermite(i, j - 1, t + 1, distance, a, b)
    )


@lru_cache(maxsize=None)
def _coulomb_hermite(
    t: int, u: int, v: int, n: int, p: float, pcx: float, pcy: float, pcz: float
) -> float:
    """Hermite Coulomb integral R^{tuv}_n."""
    if t == u == v == 0:
        distance_squared = pcx * pcx + pcy * pcy + pcz * pcz
        return (-2.0 * p) ** n * _boys(n, p * distance_squared)
    if t > 0:
        value = pcx * _coulomb_hermite(t - 1, u, v, n + 1, p, pcx, pcy, pcz)
        if t > 1:
            value += (t - 1) * _coulomb_hermite(t - 2, u, v, n + 1, p, pcx, pcy, pcz)
        return value
    if u > 0:
        value = pcy * _coulomb_hermite(t, u - 1, v, n + 1, p, pcx, pcy, pcz)
        if u > 1:
            value += (u - 1) * _coulomb_hermite(t, u - 2, v, n + 1, p, pcx, pcy, pcz)
        return value
    value = pcz * _coulomb_hermite(t, u, v - 1, n + 1, p, pcx, pcy, pcz)
    if v > 1:
        value += (v - 1) * _coulomb_hermite(t, u, v - 2, n + 1, p, pcx, pcy, pcz)
    return value


def _overlap_primitive(
    a: float,
    powers_a: tuple[int, int, int],
    center_a: Sequence[float],
    b: float,
    powers_b: tuple[int, int, int],
    center_b: Sequence[float],
) -> float:
    if any(power < 0 for power in powers_a) or any(power < 0 for power in powers_b):
        return 0.0
    product = 1.0
    for axis in range(3):
        product *= _hermite(
            powers_a[axis],
            powers_b[axis],
            0,
            center_a[axis] - center_b[axis],
            a,
            b,
        )
    return product * (math.pi / (a + b)) ** 1.5


def _kinetic_primitive(
    a: float,
    powers_a: tuple[int, int, int],
    center_a: Sequence[float],
    b: float,
    powers_b: tuple[int, int, int],
    center_b: Sequence[float],
) -> float:
    l, m, n = powers_b
    value = b * (2 * (l + m + n) + 3) * _overlap_primitive(
        a, powers_a, center_a, b, powers_b, center_b
    )
    for axis, power in enumerate(powers_b):
        raised = list(powers_b)
        raised[axis] += 2
        value -= (
            2.0
            * b
            * b
            * _overlap_primitive(a, powers_a, center_a, b, tuple(raised), center_b)
        )
        if power >= 2:
            lowered = list(powers_b)
            lowered[axis] -= 2
            value -= (
                0.5
                * power
                * (power - 1)
                * _overlap_primitive(
                    a, powers_a, center_a, b, tuple(lowered), center_b
                )
            )
    return value


def _nuclear_primitive(
    a: float,
    powers_a: tuple[int, int, int],
    center_a: np.ndarray,
    b: float,
    powers_b: tuple[int, int, int],
    center_b: np.ndarray,
    nucleus: np.ndarray,
) -> float:
    p = a + b
    product_center = (a * center_a + b * center_b) / p
    pc = product_center - nucleus
    value = 0.0
    for t in range(powers_a[0] + powers_b[0] + 1):
        ex = _hermite(powers_a[0], powers_b[0], t, center_a[0] - center_b[0], a, b)
        if ex == 0.0:
            continue
        for u in range(powers_a[1] + powers_b[1] + 1):
            ey = _hermite(powers_a[1], powers_b[1], u, center_a[1] - center_b[1], a, b)
            if ey == 0.0:
                continue
            for v in range(powers_a[2] + powers_b[2] + 1):
                ez = _hermite(
                    powers_a[2], powers_b[2], v, center_a[2] - center_b[2], a, b
                )
                if ez == 0.0:
                    continue
                value += (
                    ex
                    * ey
                    * ez
                    * _coulomb_hermite(t, u, v, 0, p, pc[0], pc[1], pc[2])
                )
    return 2.0 * math.pi / p * value


def _eri_primitive(
    a: float,
    powers_a: tuple[int, int, int],
    center_a: np.ndarray,
    b: float,
    powers_b: tuple[int, int, int],
    center_b: np.ndarray,
    c: float,
    powers_c: tuple[int, int, int],
    center_c: np.ndarray,
    d: float,
    powers_d: tuple[int, int, int],
    center_d: np.ndarray,
) -> float:
    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    center_p = (a * center_a + b * center_b) / p
    center_q = (c * center_c + d * center_d) / q
    pq = center_p - center_q
    value = 0.0
    for t in range(powers_a[0] + powers_b[0] + 1):
        ex1 = _hermite(powers_a[0], powers_b[0], t, center_a[0] - center_b[0], a, b)
        if ex1 == 0.0:
            continue
        for u in range(powers_a[1] + powers_b[1] + 1):
            ey1 = _hermite(
                powers_a[1], powers_b[1], u, center_a[1] - center_b[1], a, b
            )
            if ey1 == 0.0:
                continue
            for v in range(powers_a[2] + powers_b[2] + 1):
                ez1 = _hermite(
                    powers_a[2], powers_b[2], v, center_a[2] - center_b[2], a, b
                )
                if ez1 == 0.0:
                    continue
                head = ex1 * ey1 * ez1
                for tau in range(powers_c[0] + powers_d[0] + 1):
                    ex2 = _hermite(
                        powers_c[0], powers_d[0], tau, center_c[0] - center_d[0], c, d
                    )
                    if ex2 == 0.0:
                        continue
                    for nu in range(powers_c[1] + powers_d[1] + 1):
                        ey2 = _hermite(
                            powers_c[1],
                            powers_d[1],
                            nu,
                            center_c[1] - center_d[1],
                            c,
                            d,
                        )
                        if ey2 == 0.0:
                            continue
                        for phi in range(powers_c[2] + powers_d[2] + 1):
                            ez2 = _hermite(
                                powers_c[2],
                                powers_d[2],
                                phi,
                                center_c[2] - center_d[2],
                                c,
                                d,
                            )
                            if ez2 == 0.0:
                                continue
                            sign = -1.0 if (tau + nu + phi) % 2 else 1.0
                            value += (
                                head
                                * sign
                                * ex2
                                * ey2
                                * ez2
                                * _coulomb_hermite(
                                    t + tau,
                                    u + nu,
                                    v + phi,
                                    0,
                                    alpha,
                                    pq[0],
                                    pq[1],
                                    pq[2],
                                )
                            )
    return 2.0 * math.pi**2.5 / (p * q * math.sqrt(p + q)) * value


# --------------------------------------------------------------------------
# Contracted integrals over a basis
# --------------------------------------------------------------------------


def build_basis(
    atoms: Sequence[tuple[str, tuple[float, float, float]]],
    basis_name: str | dict[str, str],
) -> list[ContractedGaussian]:
    """Expand a geometry in Angstrom into normalized contracted Gaussians."""
    if isinstance(basis_name, dict):
        names = {symbol.capitalize(): value.lower() for symbol, value in basis_name.items()}
    else:
        names = None
        if basis_name.lower().replace("_", "-") != "sto-3g":
            raise NotImplementedError(
                f"The built-in integral engine only implements STO-3G, not "
                f"{basis_name!r}. Install PySCF (Linux/macOS/WSL) for other bases."
            )

    functions: list[ContractedGaussian] = []
    for symbol, position in atoms:
        element = symbol.capitalize()
        if names is not None:
            requested = names.get(element, "sto-3g")
            if requested.replace("_", "-") != "sto-3g":
                raise NotImplementedError(
                    f"The built-in integral engine only implements STO-3G, not "
                    f"{requested!r} on {element}."
                )
        if element not in STO3G:
            raise NotImplementedError(
                f"No built-in STO-3G data for {element}. Add it from "
                "https://www.basissetexchange.org to integrals.STO3G, or install "
                "PySCF on a platform that supports it."
            )
        center = tuple(float(x) * BOHR_PER_ANGSTROM for x in position)
        for angular_momentum, primitives in STO3G[element]:
            exponents = tuple(float(exponent) for exponent, _ in primitives)
            raw = tuple(float(coefficient) for _, coefficient in primitives)
            for powers in _CARTESIAN[angular_momentum]:
                scaled = tuple(
                    coefficient * _primitive_norm(exponent, powers)
                    for exponent, coefficient in zip(exponents, raw)
                )
                function = ContractedGaussian(center, powers, exponents, scaled)
                # Renormalize the contraction so <phi|phi> = 1. BSE coefficients
                # normalize the primitives, not the contracted function.
                self_overlap = _contract(
                    function, function, _overlap_primitive
                )
                norm = 1.0 / math.sqrt(self_overlap)
                functions.append(
                    ContractedGaussian(
                        center,
                        powers,
                        exponents,
                        tuple(coefficient * norm for coefficient in scaled),
                    )
                )
    return functions


def ao_slices_by_atom(
    atoms: Sequence[tuple[str, tuple[float, float, float]]],
    basis_name: str | dict[str, str],
) -> list[tuple[int, int]]:
    """Half-open (start, stop) basis-function range for each atom.

    Mirrors PySCF's ``aoslice_by_atom`` columns 2:4, so the Mulliken
    localization diagnostic reads the same either way. `build_basis` appends
    functions atom by atom in input order, which is what makes these contiguous.
    """
    slices: list[tuple[int, int]] = []
    start = 0
    for symbol, position in atoms:
        count = len(build_basis([(symbol, position)], basis_name))
        slices.append((start, start + count))
        start += count
    return slices


def _contract(first: ContractedGaussian, second: ContractedGaussian, kernel) -> float:
    total = 0.0
    for a, ca in zip(first.exponents, first.coefficients):
        for b, cb in zip(second.exponents, second.coefficients):
            total += ca * cb * kernel(
                a, first.powers, first.center, b, second.powers, second.center
            )
    return total


def overlap_matrix(basis: Sequence[ContractedGaussian]) -> np.ndarray:
    n = len(basis)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            matrix[i, j] = matrix[j, i] = _contract(
                basis[i], basis[j], _overlap_primitive
            )
    return matrix


def kinetic_matrix(basis: Sequence[ContractedGaussian]) -> np.ndarray:
    n = len(basis)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            matrix[i, j] = matrix[j, i] = _contract(
                basis[i], basis[j], _kinetic_primitive
            )
    return matrix


def nuclear_matrix(
    basis: Sequence[ContractedGaussian],
    charges: Sequence[float],
    positions: np.ndarray,
) -> np.ndarray:
    n = len(basis)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            total = 0.0
            for charge, nucleus in zip(charges, positions):
                for a, ca in zip(basis[i].exponents, basis[i].coefficients):
                    for b, cb in zip(basis[j].exponents, basis[j].coefficients):
                        total -= (
                            charge
                            * ca
                            * cb
                            * _nuclear_primitive(
                                a,
                                basis[i].powers,
                                np.asarray(basis[i].center),
                                b,
                                basis[j].powers,
                                np.asarray(basis[j].center),
                                nucleus,
                            )
                        )
            matrix[i, j] = matrix[j, i] = total
    return matrix


def eri_tensor(basis: Sequence[ContractedGaussian]) -> np.ndarray:
    """Two-electron integrals (ij|kl) in chemist notation, 8-fold symmetry."""
    n = len(basis)
    tensor = np.zeros((n, n, n, n))
    centers = [np.asarray(function.center) for function in basis]
    for i in range(n):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k + 1):
                    if k * (k + 1) // 2 + l > ij:
                        continue
                    total = 0.0
                    for a, ca in zip(basis[i].exponents, basis[i].coefficients):
                        for b, cb in zip(basis[j].exponents, basis[j].coefficients):
                            for c, cc in zip(
                                basis[k].exponents, basis[k].coefficients
                            ):
                                for d, cd in zip(
                                    basis[l].exponents, basis[l].coefficients
                                ):
                                    total += (
                                        ca
                                        * cb
                                        * cc
                                        * cd
                                        * _eri_primitive(
                                            a,
                                            basis[i].powers,
                                            centers[i],
                                            b,
                                            basis[j].powers,
                                            centers[j],
                                            c,
                                            basis[k].powers,
                                            centers[k],
                                            d,
                                            basis[l].powers,
                                            centers[l],
                                        )
                                    )
                    for p, q, r, s in {
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    }:
                        tensor[p, q, r, s] = total
    return tensor


# --------------------------------------------------------------------------
# RHF
# --------------------------------------------------------------------------


def nuclear_repulsion(charges: Sequence[float], positions: np.ndarray) -> float:
    total = 0.0
    for i in range(len(charges)):
        for j in range(i):
            total += charges[i] * charges[j] / float(
                np.linalg.norm(positions[i] - positions[j])
            )
    return total


def _fock(hcore: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
    exchange = np.einsum("ls,mlsn->mn", density, eri, optimize=True)
    return hcore + coulomb - 0.5 * exchange


def run_rhf(
    atoms: Sequence[tuple[str, tuple[float, float, float]]],
    basis: str | dict[str, str],
    charge: int = 0,
    spin: int = 0,
    conv_tol: float = 1.0e-11,
    max_cycle: int = 200,
) -> MeanField:
    """Closed-shell restricted Hartree-Fock with DIIS acceleration."""
    if spin != 0:
        raise ValueError("The built-in integral engine supports singlets only")

    symbols = [symbol.capitalize() for symbol, _ in atoms]
    unknown = [symbol for symbol in symbols if symbol not in ATOMIC_NUMBER]
    if unknown:
        raise NotImplementedError(f"No atomic number on record for {unknown}")
    charges = [float(ATOMIC_NUMBER[symbol]) for symbol in symbols]
    positions = np.array(
        [[float(x) * BOHR_PER_ANGSTROM for x in coords] for _, coords in atoms]
    )
    n_electrons = int(sum(charges)) - int(charge)
    if n_electrons % 2:
        raise ValueError("An RHF reference needs an even electron count")
    n_occupied = n_electrons // 2

    basis_functions = build_basis(atoms, basis)
    overlap = overlap_matrix(basis_functions)
    hcore = kinetic_matrix(basis_functions) + nuclear_matrix(
        basis_functions, charges, positions
    )
    eri = eri_tensor(basis_functions)
    e_nuc = nuclear_repulsion(charges, positions)

    n_basis = len(basis_functions)
    if n_occupied > n_basis:
        raise ValueError("More occupied orbitals than basis functions")

    # Symmetric orthogonalization. STO-3G is far from linearly dependent at this
    # size, so no eigenvalue screening is needed.
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if eigenvalues.min() <= 0.0:
        raise RuntimeError("The overlap matrix is not positive definite")
    transform = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T

    def _density_from(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        energies, vectors = np.linalg.eigh(transform.T @ matrix @ transform)
        coefficients = transform @ vectors
        occupied = coefficients[:, :n_occupied]
        return 2.0 * occupied @ occupied.T, coefficients, energies

    # Core-Hamiltonian initial guess. Starting the iteration from a zero density
    # would push an identically zero DIIS error vector onto the stack, making the
    # DIIS system singular and freezing the Fock matrix at the guess -- which
    # looks exactly like instant convergence to a wrong energy.
    density, mo_coeff, orbital_energies = _density_from(hcore)

    converged = False
    energy = float("nan")
    error_norm = float("inf")
    delta_energy = float("inf")
    errors: list[np.ndarray] = []
    focks: list[np.ndarray] = []
    iteration = 0

    for iteration in range(1, max_cycle + 1):
        fock = _fock(hcore, eri, density)
        new_energy = 0.5 * float(np.sum(density * (hcore + fock))) + e_nuc

        # The commutator FDS - SDF vanishes exactly at a stationary point, so it
        # is the convergence measure; the energy difference alone can stall.
        error = (
            transform.T
            @ (fock @ density @ overlap - overlap @ density @ fock)
            @ transform
        )
        error_norm = float(np.max(np.abs(error)))
        delta_energy = abs(new_energy - energy)
        energy = new_energy
        if error_norm < 1.0e-9 and delta_energy < conv_tol:
            converged = True
            break

        errors.append(error)
        focks.append(fock)
        if len(errors) > 8:
            errors.pop(0)
            focks.pop(0)

        extrapolated = fock
        if len(errors) > 1:
            size = len(errors)
            system = np.zeros((size + 1, size + 1))
            system[:size, :size] = [
                [float(np.sum(a * b)) for b in errors] for a in errors
            ]
            system[size, :size] = -1.0
            system[:size, size] = -1.0
            target = np.zeros(size + 1)
            target[size] = -1.0
            try:
                weights = np.linalg.solve(system, target)[:size]
            except np.linalg.LinAlgError:
                pass  # Keep the plain Fock matrix and try again next cycle.
            else:
                if np.all(np.isfinite(weights)):
                    extrapolated = sum(
                        weight * f for weight, f in zip(weights, focks)
                    )

        density, mo_coeff, orbital_energies = _density_from(extrapolated)

    if not converged:
        raise RuntimeError(
            f"RHF did not converge in {max_cycle} cycles "
            f"(|FDS-SDF| = {error_norm:.3e}, dE = {delta_energy:.3e})"
        )

    # Report the orbitals that diagonalize the converged Fock matrix, so
    # mo_energy are true canonical orbital energies rather than the last
    # DIIS-extrapolated set.
    _, mo_coeff, orbital_energies = _density_from(_fock(hcore, eri, density))

    return MeanField(
        n_electrons=n_electrons,
        n_orbitals=n_basis,
        mo_coeff=mo_coeff,
        mo_energy=orbital_energies,
        e_hf=energy,
        e_nuc=e_nuc,
        hcore=hcore,
        eri=eri,
        overlap=overlap,
        converged=True,
        n_iterations=iteration,
    )


# --------------------------------------------------------------------------
# Active-space (CAS) integrals
# --------------------------------------------------------------------------


def cas_integrals(
    mean_field: MeanField, n_core: int, n_active: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Effective one-body integrals, active (pq|rs), and the frozen-core energy.

    Returns the same three quantities as PySCF's ``CASCI.get_h1eff`` and
    ``get_h2eff``: `h1eff` already contains the Coulomb and exchange field of the
    doubly occupied core, and `e_core` contains the nuclear repulsion plus the
    core's own energy, so
    ``E = e_core + <active part>`` reproduces the total energy.
    """
    if n_core < 0 or n_active < 1:
        raise ValueError("Invalid core/active partition")
    if n_core + n_active > mean_field.n_orbitals:
        raise ValueError("The core plus active space exceeds the orbital space")

    mo_coeff = mean_field.mo_coeff
    core = mo_coeff[:, :n_core]
    active = mo_coeff[:, n_core : n_core + n_active]

    core_density = 2.0 * core @ core.T if n_core else np.zeros_like(mean_field.hcore)
    coulomb = np.einsum("ls,mnls->mn", core_density, mean_field.eri, optimize=True)
    exchange = np.einsum("ls,mlsn->mn", core_density, mean_field.eri, optimize=True)
    core_field = coulomb - 0.5 * exchange

    h1eff = active.T @ (mean_field.hcore + core_field) @ active
    e_core = (
        mean_field.e_nuc
        + float(np.sum(core_density * mean_field.hcore))
        + 0.5 * float(np.sum(core_density * core_field))
    )

    eri_active = np.einsum(
        "mnls,mp,nq,lr,st->pqrt",
        mean_field.eri,
        active,
        active,
        active,
        active,
        optimize=True,
    )
    return h1eff, eri_active, e_core
