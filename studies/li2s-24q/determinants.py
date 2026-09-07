#!/usr/bin/env python3
"""The determinant-space engine: the classical half of HI-VQE.

HI-VQE never evaluates a Pauli expectation value. The quantum device proposes
*electron configurations*; the energy comes from projecting the electronic
Hamiltonian onto the subspace those configurations span and diagonalising it
exactly on a classical computer. Everything needed for that lives here.

Conventions, fixed once and verified against OpenFermion in `selftest.py`:

* An **alpha string** is an integer whose bit ``p`` is set when spatial orbital
  ``p`` holds an alpha electron. Same for beta. A **determinant** is a pair
  ``(string_a, string_b)``.
* The determinant is ``a+_{0a}^{n0} ... a+_{(M-1)a} a+_{0b} ... |vac>`` with the
  creation operators applied in **ascending index order**, alpha block first.
  This is exactly Jordan-Wigner with spin-orbital index ``p`` for alpha and
  ``M + p`` for beta, which is the qubit ordering `ansatz.py` samples in.
* One-body integrals are ``h1e[p, q]``; two-body integrals are in **chemist**
  notation ``eri[p, q, r, s] = (pq|rs)``, which is what PySCF's
  ``mcscf.CASCI.get_h2eff`` returns.

A subspace is stored as a **tensor product** of a selected set of alpha strings
with a selected set of beta strings, not as an arbitrary list of determinants.
That is not a simplification for convenience -- it is what the HI-VQE paper
specifies ("collate the alpha and beta configurations into a single set and take
the tensor product of that set with itself"), it is what makes the sigma
contraction below scale with the orbital count rather than with dim^2, and it is
what `pyscf.fci.selected_ci` and the SQD addon both do.

Nothing in this module imports PySCF, Qiskit or matplotlib. It is plain NumPy
and SciPy, so the whole classical half of the algorithm runs anywhere -- which
matters because PySCF has no Windows wheel and is needed only to *build* the
integrals, never to run on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator


# Davidson's convergence test is on the residual norm, and the energy error is
# second order in it: a residual of 1e-7 puts the eigenvalue within roughly
# 1e-13 Ha of exact for the gaps these Hamiltonians have. That is ten orders of
# magnitude inside chemical accuracy, so tightening it further buys nothing but
# matrix-vector products -- which the profiler shows are the second-largest cost
# of a run.
DAVIDSON_RESIDUAL_TOLERANCE = 1.0e-7


# Lookup table for vectorised population counts on arrays of strings. Python's
# int.bit_count() is fine for scalars; this is for the NumPy paths.
_POPCOUNT_TABLE = np.array(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


def popcount(values):
    """Number of set bits, for a scalar or an integer array."""
    if np.isscalar(values):
        return int(values).bit_count()
    array = np.ascontiguousarray(values, dtype=np.int64).view(np.uint64)
    view = array.view(np.uint8).reshape(*array.shape, 8)
    return _POPCOUNT_TABLE[view].sum(axis=-1).astype(np.int64)


def make_strings(n_orbitals: int, n_electrons: int) -> np.ndarray:
    """Every occupation string of `n_electrons` in `n_orbitals`, ascending.

    Sorted by integer value, which is a stable, implementation-independent
    ordering -- the same set of strings always gets the same indices, so a
    cached coefficient vector means the same thing on a different machine.
    """
    if not 0 <= n_electrons <= n_orbitals:
        raise ValueError(
            f"cannot place {n_electrons} electrons in {n_orbitals} orbitals"
        )
    strings = [
        sum(1 << orbital for orbital in occupied)
        for occupied in combinations(range(n_orbitals), n_electrons)
    ]
    return np.array(sorted(strings), dtype=np.int64)


def n_determinants(n_orbitals: int, n_alpha: int, n_beta: int) -> int:
    """Dimension of the full CAS space, as an exact integer."""
    return comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)


def occupied_orbitals(string: int, n_orbitals: int) -> tuple[int, ...]:
    return tuple(p for p in range(n_orbitals) if (string >> p) & 1)


def string_to_occupation(string: int, n_orbitals: int) -> np.ndarray:
    """Bit p of `string` -> element p of a 0/1 vector."""
    return np.array([(string >> p) & 1 for p in range(n_orbitals)], dtype=np.int8)


def apply_single_excitation(string: int, q: int, p: int) -> tuple[int, int]:
    """a+_p a_q |string>, returning (new string, sign). Sign 0 means it vanishes.

    Derived rather than tabulated: `a_q` on an ascending-ordered determinant
    contributes (-1) for every occupied orbital below q, and `a+_p` then
    contributes (-1) for every occupied orbital below p in what is left. Both
    operators sit in the same spin block and the pair is parity-even, so the
    other spin block never contributes a sign.
    """
    if not (string >> q) & 1:
        return 0, 0
    sign = -1 if (string & ((1 << q) - 1)).bit_count() & 1 else 1
    removed = string ^ (1 << q)
    if (removed >> p) & 1:
        return 0, 0
    if (removed & ((1 << p) - 1)).bit_count() & 1:
        sign = -sign
    return removed | (1 << p), sign


def excitation_degree(string_1: int, string_2: int) -> int:
    """Number of orbitals that differ, i.e. the excitation rank between them."""
    return (int(string_1) ^ int(string_2)).bit_count() // 2


def determinant_excitation_degree(
    det_1: tuple[int, int], det_2: tuple[int, int]
) -> int:
    return excitation_degree(det_1[0], det_2[0]) + excitation_degree(det_1[1], det_2[1])


# --------------------------------------------------------------------------
# Single-excitation maps
# --------------------------------------------------------------------------


def single_excitation_maps(
    n_orbitals: int,
    ket_strings: np.ndarray,
    bra_strings: np.ndarray | None = None,
) -> list[sp.csr_matrix]:
    """The matrices A_pq with A_pq[J, I] = <J| a+_p a_q |I>.

    Returned flattened as ``maps[p * n_orbitals + q]``. Rows index `bra_strings`
    (defaulting to `ket_strings`); columns index `ket_strings`. Restricting the
    rows to a *selected* bra set is exactly the projection P H P that defines
    the subspace Hamiltonian -- excitations that leave the selected space are
    dropped, which is the intended behaviour and not a truncation error.
    """
    if bra_strings is None:
        bra_strings = ket_strings
    bra_index = {int(value): row for row, value in enumerate(bra_strings)}
    n_ket = len(ket_strings)
    n_bra = len(bra_strings)
    n_pairs = n_orbitals * n_orbitals

    rows: list[list[int]] = [[] for _ in range(n_pairs)]
    cols: list[list[int]] = [[] for _ in range(n_pairs)]
    data: list[list[float]] = [[] for _ in range(n_pairs)]

    for column, raw in enumerate(ket_strings):
        string = int(raw)
        occupied = [p for p in range(n_orbitals) if (string >> p) & 1]
        for q in occupied:
            for p in range(n_orbitals):
                new_string, sign = apply_single_excitation(string, q, p)
                if sign == 0:
                    continue
                row = bra_index.get(new_string)
                if row is None:
                    continue
                key = p * n_orbitals + q
                rows[key].append(row)
                cols[key].append(column)
                data[key].append(float(sign))

    return [
        sp.csr_matrix(
            (data[key], (rows[key], cols[key])),
            shape=(n_bra, n_ket),
            dtype=np.float64,
        )
        for key in range(n_pairs)
    ]


# --------------------------------------------------------------------------
# The active space and the subspace
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveSpace:
    """The integrals and electron counts that define the active-space problem."""

    n_orbitals: int
    n_alpha: int
    n_beta: int
    h1e: np.ndarray
    eri: np.ndarray
    core_energy: float

    def __post_init__(self) -> None:
        m = self.n_orbitals
        if self.h1e.shape != (m, m):
            raise ValueError(f"h1e must be ({m}, {m}), got {self.h1e.shape}")
        if self.eri.shape != (m, m, m, m):
            raise ValueError(f"eri must be ({m},)*4, got {self.eri.shape}")
        if not 0 <= self.n_alpha <= m or not 0 <= self.n_beta <= m:
            raise ValueError("electron counts do not fit the orbital count")

    @property
    def n_electrons(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orbitals

    @property
    def full_dimension(self) -> int:
        return n_determinants(self.n_orbitals, self.n_alpha, self.n_beta)

    def folded_one_body(self) -> np.ndarray:
        """h'_pq = h_pq - 1/2 sum_r (pr|rq).

        Rewrites H = sum h_pq E_pq + 1/2 sum (pq|rs)(E_pq E_rs - d_qr E_ps)
        as    H = sum h'_pq E_pq + 1/2 sum (pq|rs) E_pq E_rs,
        which is the form the sigma contraction below evaluates.
        """
        return self.h1e - 0.5 * np.einsum("prrq->pq", self.eri)


@dataclass
class Subspace:
    """A selected set of alpha strings times a selected set of beta strings."""

    strings_a: np.ndarray
    strings_b: np.ndarray

    def __post_init__(self) -> None:
        self.strings_a = np.unique(np.asarray(self.strings_a, dtype=np.int64))
        self.strings_b = np.unique(np.asarray(self.strings_b, dtype=np.int64))

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.strings_a), len(self.strings_b)

    @property
    def dimension(self) -> int:
        return len(self.strings_a) * len(self.strings_b)

    def contains(self, determinant: tuple[int, int]) -> bool:
        found_a = np.searchsorted(self.strings_a, determinant[0])
        found_b = np.searchsorted(self.strings_b, determinant[1])
        in_a = found_a < len(self.strings_a) and self.strings_a[found_a] == determinant[0]
        in_b = found_b < len(self.strings_b) and self.strings_b[found_b] == determinant[1]
        return bool(in_a and in_b)

    def union(self, other: "Subspace") -> "Subspace":
        return Subspace(
            np.concatenate([self.strings_a, other.strings_a]),
            np.concatenate([self.strings_b, other.strings_b]),
        )

    def with_determinants(self, determinants: Iterable[tuple[int, int]]) -> "Subspace":
        extra_a = [int(det[0]) for det in determinants]
        extra_b = [int(det[1]) for det in determinants]
        if not extra_a:
            return Subspace(self.strings_a.copy(), self.strings_b.copy())
        return Subspace(
            np.concatenate([self.strings_a, np.array(extra_a, dtype=np.int64)]),
            np.concatenate([self.strings_b, np.array(extra_b, dtype=np.int64)]),
        )


def hartree_fock_determinant(n_alpha: int, n_beta: int) -> tuple[int, int]:
    """The aufbau determinant: the lowest `n` orbitals occupied in each spin."""
    return (1 << n_alpha) - 1, (1 << n_beta) - 1


def hartree_fock_subspace(n_alpha: int, n_beta: int) -> Subspace:
    string_a, string_b = hartree_fock_determinant(n_alpha, n_beta)
    return Subspace(np.array([string_a]), np.array([string_b]))


# --------------------------------------------------------------------------
# Projected Hamiltonian and its ground state
# --------------------------------------------------------------------------


def same_spin_hamiltonian(space: ActiveSpace, strings: np.ndarray) -> sp.csr_matrix:
    """The part of H acting on one spin block alone, as an explicit matrix.

    Splitting the Hamiltonian as

        H = E_core + H_alpha (x) 1 + 1 (x) H_beta + V_alpha-beta

    is what makes a tensor-product subspace cheap *and* exact. ``H_alpha`` is
    an ordinary electronic Hamiltonian for the alpha electrons only, so its
    matrix elements are plain Slater-Condon evaluations with no intermediate
    state to sum over -- which is precisely the step that has to run outside the
    subspace when the same-spin term is contracted through single excitations
    instead. Building it explicitly removes that trap by construction, and the
    matrix is tiny: only the strings of one spin block, not the determinants.

    It is also built once per subspace and reused by every Davidson iteration,
    where the contracted form would be recomputed on every matrix-vector
    product.
    """
    n_electrons = int(strings[0]).bit_count() if len(strings) else 0
    view = ActiveSpace(
        n_orbitals=space.n_orbitals,
        n_alpha=n_electrons,
        n_beta=0,
        h1e=space.h1e,
        eri=space.eri,
        core_energy=0.0,
    )
    count = len(strings)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for i in range(count):
        left = int(strings[i])
        for j in range(i, count):
            right = int(strings[j])
            if excitation_degree(left, right) > 2:
                continue
            value = matrix_element(view, (left, 0), (right, 0))
            if value == 0.0:
                continue
            rows.append(i)
            columns.append(j)
            data.append(value)
            if i != j:
                rows.append(j)
                columns.append(i)
                data.append(value)
    return sp.csr_matrix(
        (data, (rows, columns)), shape=(count, count), dtype=np.float64
    )


class SubspaceHamiltonian:
    """P H P for one subspace, as a matrix-free operator on the CI matrix.

    The Hamiltonian is split by spin:

        H = E_core + H_alpha (x) 1 + 1 (x) H_beta + sum_pqrs (pq|rs) E^a_pq E^b_rs

    The two same-spin pieces are built explicitly by `same_spin_hamiltonian` --
    they act on one string block each, so they are small. The opposite-spin
    piece factorises, ``<Ia Ib|E^a_pq E^b_rs|Ka Kb> = <Ia|E^a_pq|Ka><Ib|E^b_rs|Kb>``,
    so it needs no intermediate outside the subspace either. The two orderings of
    that term are equal once ``(pq|rs) = (rs|pq)`` is used, so one is evaluated
    at full weight rather than both at half.

    Because a tensor-product projector factorises as ``P = P_a (x) P_b``, every
    piece above projects independently and ``P H P`` comes out exact. Cost per
    matrix-vector product is set by the orbital count and the dimension, never
    by the dimension squared.
    """

    def __init__(
        self,
        space: ActiveSpace,
        subspace: Subspace,
        memory_budget_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.space = space
        self.subspace = subspace
        m = space.n_orbitals
        self.n_a, self.n_b = subspace.shape
        self.memory_budget_bytes = memory_budget_bytes

        self.maps_a = single_excitation_maps(m, subspace.strings_a)
        self.maps_b = single_excitation_maps(m, subspace.strings_b)
        # The opposite-spin term needs E^b_rs applied from the right, i.e. the
        # transpose of every beta map. Transposing (or slicing) them inside the
        # matrix-vector product means SciPy rebuilds 144 sparse matrices on every
        # Davidson iteration -- measured at a fifth of the total runtime of a
        # run. They are constant for the lifetime of this operator, so they are
        # built once here instead.
        self.maps_b_transposed = [matrix.T.tocsr() for matrix in self.maps_b]
        self.same_spin_a = same_spin_hamiltonian(space, subspace.strings_a)
        self.same_spin_b = same_spin_hamiltonian(space, subspace.strings_b)
        self._eri_matrix = space.eri.reshape(m * m, m * m)
        self._diagonal: np.ndarray | None = None
        # Davidson calls sigma tens of times on the same operator, and the
        # transient below is the largest allocation in the run -- 144 x n_a x
        # chunk doubles. Allocated once and reused rather than per matrix-vector
        # product, and the per-chunk column slices with it: rebuilding those
        # inside the loop is the SciPy churn the transposes above already exist
        # to avoid, and the chunked path was still paying it.
        self._transfer: np.ndarray | None = None
        self._chunked_maps: dict[tuple[int, int], list[sp.csr_matrix]] = {}

    @property
    def dimension(self) -> int:
        return self.n_a * self.n_b

    def memory_estimate_bytes(self) -> int:
        """Peak transient cost of one sigma, given the chunking below."""
        m = self.space.n_orbitals
        return int(2 * m * m * self.n_a * self._chunk() * 8)

    def _chunk(self) -> int:
        """How many columns of the CI matrix one opposite-spin pass may hold."""
        m = self.space.n_orbitals
        per_column = 2 * m * m * max(self.n_a, 1) * 8
        return max(1, min(self.n_b, self.memory_budget_bytes // max(per_column, 1)))

    def _chunk_maps(self, start: int, stop: int) -> list[sp.csr_matrix]:
        """The beta transposes restricted to one column block, built once."""
        key = (start, stop)
        cached = self._chunked_maps.get(key)
        if cached is None:
            cached = [
                matrix[:, start:stop].tocsr() for matrix in self.maps_b_transposed
            ]
            self._chunked_maps[key] = cached
        return cached

    def sigma(self, matrix: np.ndarray) -> np.ndarray:
        """(P H P - E_core) applied to a CI matrix of shape (n_a, n_b)."""
        m = self.space.n_orbitals
        n_pairs = m * m
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)

        result = self.same_spin_a @ matrix
        result += (self.same_spin_b @ matrix.T).T

        step = self._chunk()
        whole = step >= self.n_b
        if self._transfer is None or self._transfer.shape != (n_pairs, self.n_a, step):
            self._transfer = np.empty((n_pairs, self.n_a, step), dtype=np.float64)
        for start in range(0, self.n_b, step):
            stop = min(start + step, self.n_b)
            width = stop - start
            transferred = self._transfer[:, :, :width]
            maps = self.maps_b_transposed if whole else self._chunk_maps(start, stop)
            for index in range(n_pairs):
                # The unchunked path is the one that runs almost always, and it
                # uses the prebuilt transposes with no SciPy object churn at all.
                transferred[index] = matrix @ maps[index]
            gathered = (self._eri_matrix @ transferred.reshape(n_pairs, -1)).reshape(
                n_pairs, self.n_a, width
            )
            block = result[:, start:stop]
            for index in range(n_pairs):
                block += self.maps_a[index] @ gathered[index]
        return result

    def diagonal(self) -> np.ndarray:
        """<D|H|D> for every determinant, including E_core.

        Davidson needs the diagonal as a preconditioner, and evaluating it one
        Slater-Condon element at a time would cost more than several matrix-
        vector products. The closed form below follows from the diagonal
        Slater-Condon rule and is checked against `matrix_element` in the tests.
        """
        if self._diagonal is not None:
            return self._diagonal
        m = self.space.n_orbitals
        bits_a = np.array(
            [string_to_occupation(int(s), m) for s in self.subspace.strings_a],
            dtype=np.float64,
        )
        bits_b = np.array(
            [string_to_occupation(int(s), m) for s in self.subspace.strings_b],
            dtype=np.float64,
        )
        h_diagonal = np.diag(self.space.h1e)
        coulomb = np.einsum("ppqq->pq", self.space.eri)
        exchange = np.einsum("pqqp->pq", self.space.eri)

        one_a = bits_a @ h_diagonal
        one_b = bits_b @ h_diagonal
        j_a = 0.5 * np.einsum("ip,pq,iq->i", bits_a, coulomb, bits_a)
        j_b = 0.5 * np.einsum("ip,pq,iq->i", bits_b, coulomb, bits_b)
        j_cross = bits_a @ coulomb @ bits_b.T
        k_a = 0.5 * np.einsum("ip,pq,iq->i", bits_a, exchange, bits_a)
        k_b = 0.5 * np.einsum("ip,pq,iq->i", bits_b, exchange, bits_b)

        self._diagonal = (
            self.space.core_energy
            + (one_a + j_a - k_a)[:, None]
            + (one_b + j_b - k_b)[None, :]
            + j_cross
        )
        return self._diagonal

    def as_linear_operator(self) -> LinearOperator:
        """A SciPy operator over the flattened CI vector.

        Nothing in the workflow needs it -- `davidson` calls `sigma` directly --
        but it is what lets an independent eigensolver be pointed at exactly the
        same operator, which is how `selftest.py` cross-checks Davidson against
        ARPACK. That check is not decoration: Davidson converged cleanly to the
        wrong eigenpair once already.
        """
        shape = (self.dimension, self.dimension)

        def matvec(vector: np.ndarray) -> np.ndarray:
            matrix = np.asarray(vector, dtype=np.float64).reshape(self.n_a, self.n_b)
            return self.sigma(matrix).reshape(-1)

        return LinearOperator(shape, matvec=matvec, dtype=np.float64)

    def dense(self) -> np.ndarray:
        """The explicit matrix, core energy included. Small subspaces only."""
        dimension = self.dimension
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        basis = np.zeros((self.n_a, self.n_b), dtype=np.float64)
        for column in range(dimension):
            basis.reshape(-1)[column] = 1.0
            matrix[:, column] = self.sigma(basis).reshape(-1)
            basis.reshape(-1)[column] = 0.0
        matrix += self.space.core_energy * np.eye(dimension)
        return matrix

    def expectation(self, matrix: np.ndarray) -> float:
        """<C| H |C> for a normalised C, including the core energy."""
        return float(np.vdot(matrix, self.sigma(matrix)) + self.space.core_energy)


@dataclass
class SubspaceResult:
    energy: float
    coefficients: np.ndarray
    subspace: Subspace
    dimension: int
    converged: bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.subspace.shape


def davidson(
    operator: SubspaceHamiltonian,
    guess: np.ndarray | None = None,
    tolerance: float = DAVIDSON_RESIDUAL_TOLERANCE,
    max_iterations: int = 200,
    max_subspace: int = 24,
) -> tuple[float, np.ndarray, bool, int]:
    """Lowest eigenpair by Davidson with a diagonal preconditioner.

    Davidson rather than Lanczos for one specific reason: HI-VQE grows its
    subspace monotonically, so the previous iteration's ground state -- embedded
    into the enlarged space -- is already an excellent approximation, and
    Davidson can start from it and converge in a handful of matrix-vector
    products. ARPACK takes a starting vector too but still builds a fresh Krylov
    space, which at these dimensions is the dominant cost of the whole run.
    """
    dimension = operator.dimension
    diagonal = operator.diagonal().reshape(-1)

    # The starting basis is deliberately not just the warm-start vector.
    #
    # Davidson only ever explores the space its starting vectors reach, and a
    # Krylov space built on a vector with *zero overlap* with the ground state
    # converges cleanly to the wrong eigenpair -- a real eigenvector, a residual
    # at 1e-10, and an energy tens of millihartree too high. That is not
    # hypothetical: the Hartree-Fock determinant is a spin singlet, and where
    # the true ground state of the sector is a triplet its overlap with any
    # singlet start is exactly zero. Seeding with the lowest-diagonal
    # determinant and one deterministic pseudo-random vector costs two extra
    # matrix-vector products and removes the trap, while keeping every bit of
    # the warm start's speed when the warm start is good.
    seeds: list[np.ndarray] = []
    if guess is not None and np.any(guess):
        seeds.append(np.asarray(guess, dtype=np.float64).reshape(-1).copy())
    lowest = np.zeros(dimension)
    lowest[int(np.argmin(diagonal))] = 1.0
    seeds.append(lowest)
    seeds.append(np.random.default_rng(dimension).normal(size=dimension))

    basis: list[np.ndarray] = []
    for candidate in seeds:
        for existing in basis:
            candidate = candidate - (existing @ candidate) * existing
        norm = float(np.linalg.norm(candidate))
        if norm > 1.0e-10:
            basis.append(candidate / norm)
        if len(basis) >= min(3, dimension):
            break

    applied = [
        operator.sigma(vector.reshape(operator.n_a, operator.n_b)).reshape(-1)
        for vector in basis
    ]
    vector = basis[0]
    energy = float(vector @ applied[0]) + operator.space.core_energy
    products = len(basis)

    for _ in range(max_iterations):
        stacked = np.array(basis)
        projected = stacked @ np.array(applied).T
        projected = 0.5 * (projected + projected.T)
        values, vectors = np.linalg.eigh(projected)
        best = vectors[:, 0]
        theta = float(values[0])
        vector = stacked.T @ best
        residual = np.array(applied).T @ best - theta * vector
        energy = theta + operator.space.core_energy
        if np.linalg.norm(residual) < tolerance:
            return energy, vector.reshape(operator.n_a, operator.n_b), True, products

        # Davidson preconditioner. The shift can land on a denominator of zero
        # for the determinant the current vector is built on, so it is floored.
        denominator = theta + operator.space.core_energy - diagonal
        denominator = np.where(np.abs(denominator) < 1.0e-8, 1.0e-8, denominator)
        correction = residual / denominator

        if len(basis) >= max_subspace:
            basis = [vector / np.linalg.norm(vector)]
            applied = [
                operator.sigma(basis[0].reshape(operator.n_a, operator.n_b)).reshape(-1)
            ]
            products += 1

        # Two passes of Gram-Schmidt. The Rayleigh-Ritz step below assumes the
        # basis is orthonormal -- it solves eigh(V^T H V) rather than the
        # generalised problem -- so orthogonality lost to rounding would quietly
        # corrupt the small eigenproblem rather than merely slow it down.
        for _ in range(2):
            for existing in basis:
                correction -= (existing @ correction) * existing
        norm = float(np.linalg.norm(correction))
        if norm < 1.0e-10:
            # The Krylov space is exhausted. That is convergence only if the
            # residual already said so, which it did not, so report honestly.
            return energy, vector.reshape(operator.n_a, operator.n_b), False, products
        correction /= norm
        basis.append(correction)
        applied.append(
            operator.sigma(correction.reshape(operator.n_a, operator.n_b)).reshape(-1)
        )
        products += 1

    return energy, vector.reshape(operator.n_a, operator.n_b), False, products


def solve_subspace(
    space: ActiveSpace,
    subspace: Subspace,
    guess: np.ndarray | None = None,
    tolerance: float = DAVIDSON_RESIDUAL_TOLERANCE,
    max_iterations: int = 200,
    dense_threshold: int = 64,
    memory_budget_bytes: int = 256 * 1024 * 1024,
) -> SubspaceResult:
    """Lowest eigenpair of P H P.

    `guess` is the previous iteration's coefficient matrix, embedded into the
    new (larger) subspace by `embed_coefficients`.
    """
    operator = SubspaceHamiltonian(
        space, subspace, memory_budget_bytes=memory_budget_bytes
    )
    dimension = operator.dimension
    if dimension == 0:
        raise ValueError("cannot diagonalise an empty subspace")

    if dimension == 1:
        matrix = np.ones((1, 1), dtype=np.float64)
        return SubspaceResult(operator.expectation(matrix), matrix, subspace, 1, True)

    if dimension <= dense_threshold:
        dense = operator.dense()
        values, vectors = np.linalg.eigh(dense)
        coefficients = vectors[:, 0].reshape(operator.n_a, operator.n_b)
        return SubspaceResult(float(values[0]), coefficients, subspace, dimension, True)

    energy, coefficients, converged, _ = davidson(
        operator, guess=guess, tolerance=tolerance, max_iterations=max_iterations
    )
    coefficients = coefficients / np.linalg.norm(coefficients)
    return SubspaceResult(energy, coefficients, subspace, dimension, converged)


def reindex_coefficients(
    coefficients: np.ndarray, old: Subspace, new: Subspace
) -> np.ndarray:
    """Move a CI matrix onto any other subspace, keeping the shared part.

    `embed_coefficients` is the fast path for the usual case where the new
    subspace contains the old one. This is the general one: determinants present
    in both keep their coefficient, determinants only in `new` start at zero,
    and determinants only in `old` are dropped. Used by spin completion, which
    both adds and removes strings in the same step, so neither space contains
    the other.
    """
    result = np.zeros(new.shape, dtype=np.float64)
    if coefficients.size == 0 or result.size == 0:
        return result

    def locate(source: np.ndarray, target: np.ndarray):
        if len(target) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(len(source), dtype=bool)
        where = np.searchsorted(target, source)
        np.clip(where, 0, len(target) - 1, out=where)
        return where, target[where] == source

    rows, keep_rows = locate(old.strings_a, new.strings_a)
    columns, keep_columns = locate(old.strings_b, new.strings_b)
    if not keep_rows.any() or not keep_columns.any():
        return result
    result[np.ix_(rows[keep_rows], columns[keep_columns])] = coefficients[
        np.ix_(np.nonzero(keep_rows)[0], np.nonzero(keep_columns)[0])
    ]
    return result


def embed_coefficients(
    coefficients: np.ndarray, old: Subspace, new: Subspace
) -> np.ndarray:
    """Place an old CI matrix into a larger subspace, zero-padding the rest."""
    embedded = np.zeros(new.shape, dtype=np.float64)
    if coefficients.size == 0:
        return embedded
    rows = np.searchsorted(new.strings_a, old.strings_a)
    cols = np.searchsorted(new.strings_b, old.strings_b)
    if np.any(new.strings_a[rows] != old.strings_a) or np.any(
        new.strings_b[cols] != old.strings_b
    ):
        raise ValueError("the new subspace does not contain the old one")
    embedded[np.ix_(rows, cols)] = coefficients
    return embedded


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------


def string_weights(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal probability carried by each alpha string and each beta string."""
    squared = np.asarray(coefficients, dtype=np.float64) ** 2
    return squared.sum(axis=1), squared.sum(axis=0)


def orbital_occupancies(
    space: ActiveSpace, subspace: Subspace, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """<n_p,alpha> and <n_p,beta>: the diagonal of the spin-resolved 1-RDM.

    Computed from the string occupations directly, so it needs no excitation
    maps and is exact for any subspace. `ansatz.py` uses it as the occupancy
    prior for configuration recovery.
    """
    m = space.n_orbitals
    weights_a, weights_b = string_weights(coefficients)
    bits_a = np.array(
        [string_to_occupation(int(s), m) for s in subspace.strings_a], dtype=np.float64
    )
    bits_b = np.array(
        [string_to_occupation(int(s), m) for s in subspace.strings_b], dtype=np.float64
    )
    return weights_a @ bits_a, weights_b @ bits_b


def spin_square(
    space: ActiveSpace, subspace: Subspace, coefficients: np.ndarray
) -> float:
    """<S^2> for a state expanded in `subspace`, measured in the FULL CAS space.

    The excitation maps used here run from the selected strings to *all* strings
    of the sector, not back into the selected set. That matters: S^2 can take a
    determinant out of a subspace that is not spin-complete, and restricting the
    maps would silently discard exactly the terms that reveal contamination.

    Uses the standard identity
        <S^2> = (na - nb)/2 * ((na - nb)/2 + 1) + nb - sum_pq <E^a_pq E^b_qp>,
    which needs only single-excitation maps and one pass over p, q.
    """
    m = space.n_orbitals
    full_a = make_strings(m, space.n_alpha)
    full_b = make_strings(m, space.n_beta)
    maps_a = single_excitation_maps(m, subspace.strings_a, full_a)
    maps_b = single_excitation_maps(m, subspace.strings_b, full_b)
    embedded = embed_full(coefficients, subspace, full_a, full_b)

    total = 0.0
    for p in range(m):
        for q in range(m):
            left = maps_a[p * m + q] @ coefficients
            right = left @ maps_b[q * m + p].T
            total += float(np.einsum("ij,ij->", embedded, right))
    half = 0.5 * (space.n_alpha - space.n_beta)
    return half * (half + 1.0) + space.n_beta - total


def embed_full(
    coefficients: np.ndarray,
    subspace: Subspace,
    full_a: np.ndarray,
    full_b: np.ndarray,
) -> np.ndarray:
    """Embed a subspace CI matrix into the full CAS coefficient array."""
    embedded = np.zeros((len(full_a), len(full_b)), dtype=np.float64)
    rows = np.searchsorted(full_a, subspace.strings_a)
    cols = np.searchsorted(full_b, subspace.strings_b)
    embedded[np.ix_(rows, cols)] = coefficients
    return embedded


# --------------------------------------------------------------------------
# Slater-Condon matrix elements, for scoring candidate configurations
# --------------------------------------------------------------------------


def _spin_orbital_list(string_a: int, string_b: int, n_orbitals: int) -> list[int]:
    """Occupied spin-orbital indices, alpha block first, ascending."""
    indices = [p for p in range(n_orbitals) if (string_a >> p) & 1]
    indices += [n_orbitals + p for p in range(n_orbitals) if (string_b >> p) & 1]
    return indices


def _spin_orbital_string(string_a: int, string_b: int, n_orbitals: int) -> int:
    return int(string_a) | (int(string_b) << n_orbitals)


def _phase(occupied: int, index: int) -> int:
    return -1 if (occupied & ((1 << index) - 1)).bit_count() & 1 else 1


def matrix_element(
    space: ActiveSpace, bra: tuple[int, int], ket: tuple[int, int]
) -> float:
    """<bra| H |ket> by the Slater-Condon rules, core energy included.

    Written in the spin-orbital picture with the physicist-ordered antisymmetric
    integral <ij||kl>, which is the least error-prone way to get the phases
    right; the chemist-notation `eri` is converted inline. This is the scoring
    function HI-VQE's classical expansion step uses, so it runs over thousands
    of candidate determinants, not millions, and clarity beats micro-optimisation.
    """
    m = space.n_orbitals
    bra_string = _spin_orbital_string(bra[0], bra[1], m)
    ket_string = _spin_orbital_string(ket[0], ket[1], m)
    difference = bra_string ^ ket_string
    degree = difference.bit_count() // 2
    if degree > 2:
        return 0.0

    def spatial(index: int) -> int:
        return index % m

    def same_spin(index_1: int, index_2: int) -> bool:
        return (index_1 // m) == (index_2 // m)

    def one_body(index_1: int, index_2: int) -> float:
        if not same_spin(index_1, index_2):
            return 0.0
        return float(space.h1e[spatial(index_1), spatial(index_2)])

    def antisymmetric(i: int, j: int, k: int, l: int) -> float:
        """<ij||kl> = <ij|kl> - <ij|lk> in spin orbitals, from chemist eri."""
        value = 0.0
        if same_spin(i, k) and same_spin(j, l):
            value += float(
                space.eri[spatial(i), spatial(k), spatial(j), spatial(l)]
            )
        if same_spin(i, l) and same_spin(j, k):
            value -= float(
                space.eri[spatial(i), spatial(l), spatial(j), spatial(k)]
            )
        return value

    occupied_ket = _spin_orbital_list(ket[0], ket[1], m)

    if degree == 0:
        total = space.core_energy
        for index in occupied_ket:
            total += one_body(index, index)
        for position, i in enumerate(occupied_ket):
            for j in occupied_ket[position + 1 :]:
                total += antisymmetric(i, j, i, j)
        return float(total)

    removed = ket_string & difference
    added = bra_string & difference
    removed_indices = [i for i in range(2 * m) if (removed >> i) & 1]
    added_indices = [i for i in range(2 * m) if (added >> i) & 1]

    if degree == 1:
        (q,) = removed_indices
        (p,) = added_indices
        if not same_spin(p, q):
            return 0.0
        sign = _phase(ket_string, q) * _phase(ket_string ^ (1 << q), p)
        total = one_body(p, q)
        for index in occupied_ket:
            if index == q:
                continue
            total += antisymmetric(p, index, q, index)
        return float(sign * total)

    q, s = removed_indices
    p, r = added_indices
    # Phase of a+_r a+_p a_s a_q applied to |ket>, built one operator at a time.
    intermediate = ket_string ^ (1 << q)
    sign = _phase(ket_string, q) * _phase(intermediate, s)
    intermediate ^= 1 << s
    sign *= _phase(intermediate, p)
    intermediate |= 1 << p
    sign *= _phase(intermediate, r)
    # The Slater-Condon rule is stated for a+_p a+_r a_s a_q, which is the
    # opposite creation order, hence the extra minus. Verified against
    # OpenFermion in selftest.py rather than argued from the formula.
    return float(-sign * antisymmetric(p, r, q, s))


def single_and_double_excitations(
    determinant: tuple[int, int], n_orbitals: int
) -> list[tuple[int, int]]:
    """Every single and double excitation of one determinant, Sz preserved.

    Includes the same-spin (alpha-alpha, beta-beta) and opposite-spin
    (alpha-beta) doubles. The reference determinant itself is not returned.
    """
    string_a, string_b = int(determinant[0]), int(determinant[1])
    occupied_a = [p for p in range(n_orbitals) if (string_a >> p) & 1]
    virtual_a = [p for p in range(n_orbitals) if not (string_a >> p) & 1]
    occupied_b = [p for p in range(n_orbitals) if (string_b >> p) & 1]
    virtual_b = [p for p in range(n_orbitals) if not (string_b >> p) & 1]

    singles_a = [
        (string_a ^ (1 << i) | (1 << a)) for i in occupied_a for a in virtual_a
    ]
    singles_b = [
        (string_b ^ (1 << i) | (1 << a)) for i in occupied_b for a in virtual_b
    ]
    doubles_a = [
        (string_a ^ (1 << i) ^ (1 << j) | (1 << a) | (1 << b))
        for i, j in combinations(occupied_a, 2)
        for a, b in combinations(virtual_a, 2)
    ]
    doubles_b = [
        (string_b ^ (1 << i) ^ (1 << j) | (1 << a) | (1 << b))
        for i, j in combinations(occupied_b, 2)
        for a, b in combinations(virtual_b, 2)
    ]

    results: list[tuple[int, int]] = []
    results += [(new, string_b) for new in singles_a]
    results += [(string_a, new) for new in singles_b]
    results += [(new, string_b) for new in doubles_a]
    results += [(string_a, new) for new in doubles_b]
    results += [(new_a, new_b) for new_a in singles_a for new_b in singles_b]
    return results
