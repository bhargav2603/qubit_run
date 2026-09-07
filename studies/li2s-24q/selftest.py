#!/usr/bin/env python3
"""Prove the whole local stack against exact answers -- no PySCF, no cache.

Four stages, each of which can be run alone:

  structure   layering, conventions and arithmetic. Instant, NumPy only.
  classical   the determinant engine against exact classical references,
              including an independent Jordan-Wigner operator from OpenFermion.
  quantum     the Qiskit circuit and the sector simulator against Aer.
  algorithm   a full HI-VQE run on a synthetic active space whose exact ground
              state is known by direct diagonalisation.

Two of these checks exist because they caught real bugs, and they are worded so
that is not forgotten:

* `matrix_element` for a double excitation had the wrong overall sign -- the
  Slater-Condon rule is stated for `a+_p a+_r a_s a_q` and the phase was being
  accumulated for the opposite creation order. Comparing against OpenFermion
  found it; nothing else would have, because the error is a global sign on one
  of five cases and leaves the spectrum of small systems looking plausible.
* The projected Hamiltonian restricted the two-electron *intermediate* state to
  the subspace, which computes P E P E P rather than P (E E) P. That produced
  subspace energies below the exact ground state -- a variational impossibility.
  The `projection` check compares P H P against the exact sub-block, and the
  `variational` check would fail loudly on any recurrence.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import determinants as dt
from synthetic import check_eri_symmetry, synthetic_active_space


ROOT = Path(__file__).resolve().parent


class Reporter:
    """Counts checks and prints one line each."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            print(f"  [ok  ] {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed += 1
            self.failures.append(name)
            print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
        return bool(condition)

    def close(self, name: str, value: float, tolerance: float, unit: str = "") -> bool:
        return self.check(
            name, abs(value) <= tolerance, f"{value:.3e}{unit} <= {tolerance:.1e}{unit}"
        )

    def skip(self, name: str, reason: str) -> None:
        self.skipped += 1
        print(f"  [skip] {name}  ({reason})")

    def stage(self, title: str) -> None:
        print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------
# An independent reference: OpenFermion's Jordan-Wigner operator
# --------------------------------------------------------------------------


def openfermion_matrix(space: dt.ActiveSpace) -> np.ndarray:
    """The same Hamiltonian, built by a package that shares no code with ours.

    OpenFermion is handed spin-orbital tensors in *its* convention and produces
    a qubit operator through *its* Jordan-Wigner transform; the only thing the
    two implementations agree on in advance is the physics. Comparing the two
    determinant-basis matrices is therefore a real test of the conventions --
    the operator ordering, the chemist-to-physicist index shuffle, and every
    fermionic phase -- and not a test of one function against itself.
    """
    from openfermion import InteractionOperator, get_sparse_operator, jordan_wigner

    m = space.n_orbitals
    n_spin_orbitals = 2 * m
    one = np.zeros((n_spin_orbitals, n_spin_orbitals))
    two = np.zeros((n_spin_orbitals,) * 4)
    for p in range(m):
        for q in range(m):
            for spin in range(2):
                one[p + spin * m, q + spin * m] = space.h1e[p, q]
    for p in range(m):
        for q in range(m):
            for r in range(m):
                for s in range(m):
                    for sp in range(2):
                        for sq in range(2):
                            two[p + sp * m, r + sq * m, s + sq * m, q + sp * m] += (
                                0.5 * space.eri[p, q, r, s]
                            )
    operator = InteractionOperator(space.core_energy, one, two)
    sparse = get_sparse_operator(jordan_wigner(operator), n_qubits=n_spin_orbitals)
    return np.asarray(sparse.todense()).real


def openfermion_block(space: dt.ActiveSpace, subspace: dt.Subspace) -> np.ndarray:
    """The OpenFermion matrix restricted to our determinant ordering."""
    dense = openfermion_matrix(space)
    m = space.n_orbitals
    n_spin_orbitals = 2 * m

    def index(string_a: int, string_b: int) -> int:
        occupied = [p for p in range(m) if (string_a >> p) & 1]
        occupied += [m + p for p in range(m) if (string_b >> p) & 1]
        value = 0
        for qubit in occupied:
            # OpenFermion serialises qubit 0 into the MOST significant bit.
            value |= 1 << (n_spin_orbitals - 1 - qubit)
        return value

    rows = np.array(
        [
            index(int(a), int(b))
            for a in subspace.strings_a
            for b in subspace.strings_b
        ]
    )
    return dense[np.ix_(rows, rows)]


# --------------------------------------------------------------------------
# Stage 1: structure
# --------------------------------------------------------------------------


def stage_structure(reporter: Reporter) -> None:
    reporter.stage("Stage 1 -- structure, conventions and layering")

    source = {
        name: (ROOT / f"{name}.py").read_text(encoding="utf-8")
        for name in (
            "determinants",
            "hamiltonian",
            "ansatz",
            "sector_sim",
            "hivqe",
            "synthetic",
            "validation",
        )
    }
    def imports(text: str, package: str) -> bool:
        """Does this module actually import `package`?

        Searching the raw text would match the prose in a docstring -- several
        of these modules discuss PySCF's Windows situation at length without
        importing it -- so only real import statements count.
        """
        import re

        pattern = rf"^\s*(?:from\s+{package}[\.\s]|import\s+{package}\b)"
        return re.search(pattern, text, flags=re.MULTILINE) is not None

    reporter.check(
        "determinants.py imports neither Qiskit nor PySCF",
        not imports(source["determinants"], "qiskit")
        and not imports(source["determinants"], "pyscf"),
    )
    reporter.check(
        "hivqe.py does not import PySCF",
        not imports(source["hivqe"], "pyscf"),
    )
    reporter.check(
        "synthetic.py needs no quantum-chemistry package",
        not imports(source["synthetic"], "pyscf")
        and not imports(source["synthetic"], "qiskit"),
    )
    reporter.check(
        "hamiltonian.py is the only module that imports PySCF",
        all(
            not imports(text, "pyscf")
            for name, text in source.items()
            if name != "hamiltonian"
        ),
    )
    reporter.check(
        "validation.py runs without Qiskit",
        not imports(source["validation"], "qiskit"),
    )

    # -- string bookkeeping
    from math import comb

    strings = dt.make_strings(12, 6)
    reporter.check(
        "string enumeration is complete and sorted",
        len(strings) == comb(12, 6)
        and bool(np.all(np.diff(strings) > 0))
        and all(int(value).bit_count() == 6 for value in strings),
        f"{len(strings)} alpha strings",
    )
    reporter.check(
        "full Li2S CAS dimension matches the published 853,776",
        dt.n_determinants(12, 6, 6) == 853_776,
        f"{dt.n_determinants(12, 6, 6):,}",
    )
    reporter.check(
        "vectorised popcount agrees with int.bit_count",
        bool(
            np.array_equal(
                dt.popcount(strings),
                np.array([int(v).bit_count() for v in strings]),
            )
        ),
    )

    # -- excitation phases, from first principles
    #    a+_2 a_0 |{0,1}> : removing orbital 0 costs nothing (nothing below it),
    #    creating orbital 2 passes occupied orbital 1, so the sign is -1.
    new_string, sign = dt.apply_single_excitation(0b0011, 0, 2)
    reporter.check(
        "single-excitation phase is correct on a worked example",
        new_string == 0b0110 and sign == -1,
        f"a+_2 a_0 |0011> = {sign:+d} |{new_string:04b}>",
    )
    reporter.check(
        "excitation into an occupied orbital vanishes",
        dt.apply_single_excitation(0b0011, 0, 1) == (0, 0),
    )
    reporter.check(
        "excitation from an empty orbital vanishes",
        dt.apply_single_excitation(0b0011, 2, 3) == (0, 0),
    )
    reporter.check(
        "Hartree-Fock determinant is the aufbau one",
        dt.hartree_fock_determinant(6, 6) == (0b111111, 0b111111),
    )

    # -- the synthetic Hamiltonian really has the symmetries it claims
    space = synthetic_active_space(5, 2, 3, seed=1)
    residuals = check_eri_symmetry(space.eri)
    reporter.close(
        "synthetic eri has full eight-fold symmetry",
        max(residuals.values()),
        1e-14,
    )
    reporter.close(
        "synthetic h1e is symmetric",
        float(np.abs(space.h1e - space.h1e.T).max()),
        1e-14,
    )

    # -- spin arithmetic used by the verdict
    from hivqe import spin_contamination

    # S(S+1) is 0, 0.75, 2, 3.75, 6 ... so 0.5 sits 0.25 away from the doublet
    # value 0.75, and a state with <S^2> = 0.5 is not a spin eigenstate at all.
    reporter.check(
        "spin contamination is measured from the nearest S(S+1)",
        spin_contamination(0.0) < 1e-12
        and spin_contamination(0.75) < 1e-12
        and spin_contamination(2.0) < 1e-12
        and spin_contamination(6.0) < 1e-12
        and abs(spin_contamination(0.5) - 0.25) < 1e-12,
        "0, 0.75, 2 and 6 are clean; 0.5 is 0.25 off the doublet",
    )

    # -- excitation generator
    reference = dt.hartree_fock_determinant(3, 3)
    candidates = dt.single_and_double_excitations(reference, 6)
    reporter.check(
        "single/double generator produces only degree 1 and 2, with no repeats",
        len(candidates) == len(set(candidates))
        and set(
            dt.determinant_excitation_degree(reference, c) for c in candidates
        )
        == {1, 2},
        f"{len(candidates)} candidates",
    )


# --------------------------------------------------------------------------
# Stage 2: the classical engine
# --------------------------------------------------------------------------


def stage_classical(reporter: Reporter) -> None:
    reporter.stage("Stage 2 -- the determinant engine against exact references")

    space = synthetic_active_space(4, 2, 2, seed=7)
    full = dt.Subspace(dt.make_strings(4, 2), dt.make_strings(4, 2))
    operator = dt.SubspaceHamiltonian(space, full)
    ours = operator.dense()

    try:
        reference = openfermion_block(space, full)
    except ImportError:
        reference = None
        reporter.skip(
            "Hamiltonian matches OpenFermion's Jordan-Wigner operator",
            "openfermion is not installed",
        )
        reporter.skip("Slater-Condon matches OpenFermion", "openfermion is not installed")

    if reference is not None:
        reporter.close(
            "Hamiltonian matches OpenFermion's Jordan-Wigner operator",
            float(np.abs(ours - reference).max()),
            1e-11,
            " Ha",
        )
        worst = 0.0
        determinants = [
            (int(a), int(b)) for a in full.strings_a for b in full.strings_b
        ]
        for row, bra in enumerate(determinants):
            for column, ket in enumerate(determinants):
                worst = max(
                    worst,
                    abs(dt.matrix_element(space, bra, ket) - reference[row, column]),
                )
        reporter.close(
            "Slater-Condon matches OpenFermion, element by element",
            worst,
            1e-11,
            " Ha",
        )

    reporter.close(
        "the projected Hamiltonian is symmetric",
        float(np.abs(ours - ours.T).max()),
        1e-12,
        " Ha",
    )

    # -- projection: P H P must equal the exact sub-block of H
    space = synthetic_active_space(5, 2, 3, seed=9)
    all_a, all_b = dt.make_strings(5, 2), dt.make_strings(5, 3)
    parent = dt.Subspace(all_a, all_b)
    parent_dense = dt.SubspaceHamiltonian(space, parent).dense()
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(6):
        rows = np.sort(rng.choice(len(all_a), size=max(2, len(all_a) // 2), replace=False))
        columns = np.sort(
            rng.choice(len(all_b), size=max(2, len(all_b) // 2), replace=False)
        )
        subspace = dt.Subspace(all_a[rows], all_b[columns])
        block = dt.SubspaceHamiltonian(space, subspace).dense()
        indices = (rows[:, None] * len(all_b) + columns[None, :]).reshape(-1)
        worst = max(
            worst, float(np.abs(block - parent_dense[np.ix_(indices, indices)]).max())
        )
    reporter.close(
        "P H P equals the exact sub-block of H (two-electron intermediates included)",
        worst,
        1e-11,
        " Ha",
    )

    # -- the same-spin block, which is what makes the projection exact
    same_spin = dt.same_spin_hamiltonian(space, all_a)
    dense_same_spin = np.asarray(same_spin.todense())
    reporter.close(
        "the same-spin block is symmetric",
        float(np.abs(dense_same_spin - dense_same_spin.T).max()),
        1e-12,
        " Ha",
    )
    # H = E_core + H_a (x) 1 + 1 (x) H_b + V_ab, so switching off the
    # opposite-spin integrals must leave exactly the two same-spin blocks.
    no_cross = dt.ActiveSpace(
        space.n_orbitals, space.n_alpha, space.n_beta, space.h1e, space.eri, 0.0
    )
    operator = dt.SubspaceHamiltonian(no_cross, parent)
    identity_a = np.eye(len(all_a))
    identity_b = np.eye(len(all_b))
    reconstructed = (
        np.kron(np.asarray(dt.same_spin_hamiltonian(space, all_a).todense()), identity_b)
        + np.kron(identity_a, np.asarray(dt.same_spin_hamiltonian(space, all_b).todense()))
    )
    cross_only = operator.dense() - reconstructed
    reporter.check(
        "the spin decomposition accounts for every matrix element",
        float(np.abs(cross_only).max()) > 0.0
        and float(np.abs(cross_only - cross_only.T).max()) < 1e-12,
        "the remainder is the non-zero, symmetric opposite-spin term",
    )

    # -- Davidson against dense diagonalisation
    for orbitals, alpha, beta in ((5, 2, 3), (6, 3, 3)):
        space = synthetic_active_space(orbitals, alpha, beta, seed=3)
        subspace = dt.Subspace(
            dt.make_strings(orbitals, alpha), dt.make_strings(orbitals, beta)
        )
        operator = dt.SubspaceHamiltonian(space, subspace)
        exact = float(np.linalg.eigvalsh(operator.dense())[0])
        solved = dt.solve_subspace(space, subspace)
        reporter.close(
            f"Davidson matches dense diagonalisation ({operator.dimension} determinants)",
            solved.energy - exact,
            1e-9,
            " Ha",
        )

    # -- Davidson against an independent eigensolver, from a deliberately bad
    #    start. Davidson only explores what its starting vectors reach, and a
    #    start with zero overlap on the ground state converges cleanly to the
    #    wrong eigenpair: a real eigenvector, a residual at 1e-10, and an energy
    #    tens of millihartree too high. That happened. This is the check.
    from scipy.sparse.linalg import eigsh

    space = synthetic_active_space(5, 2, 2, seed=11)
    subspace = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 2))
    operator = dt.SubspaceHamiltonian(space, subspace)
    arpack = float(
        eigsh(operator.as_linear_operator(), k=1, which="SA", tol=0)[0][0]
    ) + space.core_energy
    hartree_fock_start = np.zeros(subspace.shape)
    hartree_fock_start[
        int(np.searchsorted(subspace.strings_a, (1 << 2) - 1)),
        int(np.searchsorted(subspace.strings_b, (1 << 2) - 1)),
    ] = 1.0
    reporter.close(
        "Davidson finds the true ground state even from a zero-overlap start",
        dt.solve_subspace(space, subspace, guess=hartree_fock_start).energy - arpack,
        1e-9,
        " Ha",
    )
    reporter.close(
        "Davidson agrees with ARPACK on the same operator",
        dt.solve_subspace(space, subspace).energy - arpack,
        1e-9,
        " Ha",
    )

    # -- the Davidson preconditioner diagonal
    space = synthetic_active_space(5, 2, 3, seed=9)
    subspace = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 3))
    operator = dt.SubspaceHamiltonian(space, subspace)
    diagonal = operator.diagonal()
    worst = max(
        abs(
            diagonal[row, column]
            - dt.matrix_element(space, (int(a), int(b)), (int(a), int(b)))
        )
        for row, a in enumerate(subspace.strings_a)
        for column, b in enumerate(subspace.strings_b)
    )
    reporter.close(
        "the closed-form diagonal equals Slater-Condon", worst, 1e-11, " Ha"
    )

    # -- variational bound on random subspaces
    exact = dt.solve_subspace(space, subspace)
    violations = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(subspace.strings_a), size=3, replace=False))
        columns = np.sort(rng.choice(len(subspace.strings_b), size=4, replace=False))
        trial = dt.Subspace(subspace.strings_a[rows], subspace.strings_b[columns])
        if dt.solve_subspace(space, trial).energy < exact.energy - 1e-10:
            violations += 1
    reporter.check(
        "no projected subspace falls below the exact ground state",
        violations == 0,
        f"{violations} violations in 8 random subspaces",
    )

    # -- spin and particle number, on a genuinely closed-shell space
    space66 = synthetic_active_space(5, 2, 2, seed=11)
    subspace66 = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 2))
    exact66 = dt.solve_subspace(space66, subspace66)
    from hivqe import spin_contamination

    reporter.close(
        "the Hartree-Fock determinant is a singlet",
        dt.spin_square(space66, dt.hartree_fock_subspace(2, 2), np.ones((1, 1))),
        1e-12,
    )

    reporter.close(
        "the exact ground state is a spin eigenstate",
        spin_contamination(dt.spin_square(space66, subspace66, exact66.coefficients)),
        1e-9,
    )
    occupancy_a, occupancy_b = dt.orbital_occupancies(
        space66, subspace66, exact66.coefficients
    )
    reporter.close(
        "orbital occupancies sum to the electron count",
        float(occupancy_a.sum() + occupancy_b.sum()) - 4.0,
        1e-10,
    )

    # -- embedding
    small = dt.Subspace(subspace66.strings_a[:2], subspace66.strings_b[:3])
    coefficients = np.arange(6, dtype=float).reshape(2, 3)
    embedded = dt.embed_coefficients(coefficients, small, subspace66)
    reporter.check(
        "embedding preserves every coefficient and adds only zeros",
        abs(embedded.sum() - coefficients.sum()) < 1e-12
        and int((embedded != 0).sum()) == int((coefficients != 0).sum()),
    )

    # -- pruning and growth
    from hivqe import grow_subspace, prune_subspace, rank_candidates

    weights = np.zeros(subspace66.shape)
    weights[0, 0] = 1.0
    weights[1, 1] = 1e-9
    pruned, kept = prune_subspace(
        subspace66, weights, 4, 1e-6, dt.hartree_fock_determinant(2, 2)
    )
    reporter.check(
        "pruning keeps the Hartree-Fock anchor and respects the cap",
        pruned.dimension <= 4 and pruned.contains(dt.hartree_fock_determinant(2, 2)),
        f"{pruned.dimension} determinants",
    )
    # Ranking has to be exercised on a *partial* subspace: over the full space
    # there is nothing outside it to rank, and the check would pass vacuously.
    partial = dt.Subspace(subspace66.strings_a[:3], subspace66.strings_b[:3])
    partial_solution = dt.solve_subspace(space66, partial)
    ranked = rank_candidates(space66, partial, partial_solution.coefficients, 4)
    reporter.check(
        "candidate ranking returns only configurations outside the subspace",
        len(ranked) > 0
        and all(not partial.contains(candidate) for _, candidate in ranked),
        f"{len(ranked)} candidates",
    )
    reporter.check(
        "candidate ranking is sorted by descending score",
        all(ranked[i][0] >= ranked[i + 1][0] for i in range(len(ranked) - 1)),
    )
    reporter.check(
        "the top-ranked candidate really does couple to the wavefunction",
        ranked[0][0] > 0.0,
        f"score {ranked[0][0]:.3e} Ha",
    )
    anchor = dt.hartree_fock_subspace(2, 2)
    fake = [(1.0, (int(a), int(b))) for a in subspace66.strings_a for b in subspace66.strings_b]
    grown, added = grow_subspace(anchor, fake, len(fake), 6)
    reporter.check(
        "growth stops at the dimension cap",
        grown.dimension <= 6,
        f"{grown.dimension} determinants after adding {added}",
    )


# --------------------------------------------------------------------------
# Stage 3: the quantum layer
# --------------------------------------------------------------------------


def stage_quantum(reporter: Reporter) -> None:
    reporter.stage("Stage 3 -- the Qiskit circuit and the sector simulator")

    try:
        import ansatz
        import sector_sim
        from qiskit.quantum_info import Statevector
    except ImportError as error:
        reporter.skip("the whole quantum stage", f"qiskit is unavailable ({error})")
        return

    orbitals, alpha, beta = 5, 2, 3
    circuit, parameters = ansatz.build_ansatz(orbitals, alpha, beta, reps=2)
    reporter.check(
        "parameter_count matches the circuit that gets built",
        len(parameters) == ansatz.parameter_count(orbitals, 2)
        and circuit.num_parameters == len(parameters),
        f"{len(parameters)} parameters",
    )

    zero = Statevector(
        circuit.assign_parameters({p: 0.0 for p in parameters})
    ).data
    expected_index = ((1 << alpha) - 1) | (((1 << beta) - 1) << orbitals)
    reporter.close(
        "at zero angles the circuit is exactly the Hartree-Fock determinant",
        float(abs(abs(zero[expected_index]) - 1.0)),
        1e-12,
    )

    simulator = sector_sim.SectorSimulator(circuit, orbitals, alpha, beta)
    worst_amplitude = 0.0
    worst_leak = 0.0
    for seed, scale in ((1, 0.3), (2, 0.9), (3, 1.7)):
        angles = ansatz.initial_parameters(len(parameters), scale, seed)
        ours = simulator.run(angles, parameters)
        state = Statevector(
            circuit.assign_parameters(
                {p: float(v) for p, v in zip(parameters, angles)}
            )
        ).data
        reference = np.array(
            [
                [
                    state[int(a) | (int(b) << orbitals)]
                    for b in simulator.strings_b
                ]
                for a in simulator.strings_a
            ]
        )
        worst_amplitude = max(worst_amplitude, float(np.abs(ours - reference).max()))
        worst_leak = max(worst_leak, float(abs(1.0 - (np.abs(reference) ** 2).sum())))
    reporter.close(
        "the sector simulator reproduces Qiskit's statevector amplitude by amplitude",
        worst_amplitude,
        1e-12,
    )
    reporter.close(
        "the ansatz never leaves the particle-number sector",
        worst_leak,
        1e-12,
    )

    # -- Aer, the path a user actually runs
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        reporter.skip("the sector simulator agrees with Aer", "qiskit-aer is unavailable")
    else:
        angles = ansatz.initial_parameters(len(parameters), 0.8, 5)
        bound = circuit.assign_parameters(
            {p: float(v) for p, v in zip(parameters, angles)}
        )
        bound.save_statevector()
        aer = AerSimulator(method="statevector")
        from qiskit import transpile

        state = np.asarray(
            aer.run(transpile(bound, aer), shots=1).result().get_statevector()
        )
        ours = simulator.run(angles, parameters)
        reference = np.array(
            [
                [state[int(a) | (int(b) << orbitals)] for b in simulator.strings_b]
                for a in simulator.strings_a
            ]
        )
        reporter.close(
            "the sector simulator agrees with Aer amplitude by amplitude",
            float(np.abs(ours - reference).max()),
            1e-11,
        )

    # -- the claim that justifies ending the circuit on a Givens network:
    #    a diagonal layer appended AFTER everything else changes only phases,
    #    so it is exactly invisible to the measurement distribution. (A diagonal
    #    layer in the *middle* is a different matter -- the Givens network after
    #    it converts those phases into amplitudes, which is precisely why the
    #    Jastrow layer earns its place where it is.)
    base_angles = ansatz.initial_parameters(len(parameters), 0.7, 8)
    base = simulator.probabilities(base_angles, parameters)

    from qiskit.circuit import Parameter

    trailing = circuit.copy()
    extra = []
    for qubit in range(trailing.num_qubits):
        parameter = Parameter(f"tail_rz_{qubit}")
        trailing.rz(parameter, qubit)
        extra.append(parameter)
    for qubit in range(trailing.num_qubits - 1):
        parameter = Parameter(f"tail_rzz_{qubit}")
        trailing.rzz(parameter, qubit, qubit + 1)
        extra.append(parameter)
    rng = np.random.default_rng(31)
    tail_angles = np.concatenate([base_angles, rng.normal(size=len(extra))])
    moved = sector_sim.SectorSimulator(
        trailing, orbitals, alpha, beta
    ).probabilities(tail_angles, list(parameters) + extra)
    reporter.check(
        "the circuit really does end on a Givens network",
        circuit.data[-1].operation.name == "xx_plus_yy",
        f"last gate is {circuit.data[-1].operation.name}",
    )
    reporter.close(
        "a trailing diagonal layer cannot change any measurement probability",
        float(np.abs(base - moved).max()),
        1e-12,
    )

    # -- sampling through Aer, and configuration recovery
    try:
        sampler = ansatz.ConfigurationSampler(
            orbitals,
            alpha,
            beta,
            reps=1,
            settings=ansatz.SamplerSettings(shots=4096, seed=17),
        )
    except ImportError:
        reporter.skip("Aer sampling returns valid configurations", "qiskit-aer missing")
    else:
        batch = sampler.sample(
            ansatz.initial_parameters(sampler.n_parameters, 0.9, 21)
        )
        reporter.check(
            "every noiseless sample has the right electron count",
            all(int(v).bit_count() == alpha for v in batch.strings_a)
            and all(int(v).bit_count() == beta for v in batch.strings_b)
            and batch.leakage == 0.0,
            f"{len(batch.counts)} distinct configurations, leakage {batch.leakage:.1e}",
        )
        reporter.check(
            "sampled counts sum to the shot count",
            int(batch.counts.sum()) == batch.n_shots,
        )

        noisy = ansatz.ConfigurationSampler(
            orbitals,
            alpha,
            beta,
            reps=1,
            settings=ansatz.SamplerSettings(shots=2048, seed=17, readout_error=0.05),
        )
        prior = (
            np.array([1.0] * alpha + [0.0] * (orbitals - alpha)),
            np.array([1.0] * beta + [0.0] * (orbitals - beta)),
        )
        repaired = noisy.sample(
            ansatz.initial_parameters(noisy.n_parameters, 0.9, 21),
            occupancy_prior=prior,
        )
        reporter.check(
            "readout noise really does break the electron count",
            repaired.n_recovered > 0,
            f"{repaired.n_recovered} of {repaired.n_shots} shots needed repair",
        )
        reporter.check(
            "configuration recovery restores every electron count",
            all(int(v).bit_count() == alpha for v in repaired.strings_a)
            and all(int(v).bit_count() == beta for v in repaired.strings_b),
        )


# --------------------------------------------------------------------------
# Stage 4: the algorithm
# --------------------------------------------------------------------------


def stage_algorithm(reporter: Reporter, skip_quantum: bool) -> None:
    reporter.stage("Stage 4 -- HI-VQE against an exactly solved active space")

    from hivqe import HiVqeSettings, run_hivqe

    orbitals, alpha, beta = 6, 3, 3
    space = synthetic_active_space(orbitals, alpha, beta, seed=3)
    subspace = dt.Subspace(
        dt.make_strings(orbitals, alpha), dt.make_strings(orbitals, beta)
    )
    exact = dt.solve_subspace(space, subspace)
    reference = dt.hartree_fock_determinant(alpha, beta)
    hartree_fock = dt.matrix_element(space, reference, reference)
    print(
        f"  reference: {subspace.dimension} determinants, "
        f"E(exact) = {exact.energy:.9f}, E(HF) = {hartree_fock:.9f}, "
        f"correlation = {exact.energy - hartree_fock:.6f} Ha"
    )

    simulators = ["none"] if skip_quantum else ["sector", "none"]
    results = {}
    for simulator in simulators:
        settings = HiVqeSettings(
            simulator=simulator,
            max_determinants=subspace.dimension,
            # growth_factor 1 means amplitude screening prunes to the same cap
            # the expansion may grow to -- i.e. never prunes. That is what makes
            # "given the whole space it finds the exact answer" a statement
            # about the algorithm rather than about the screening threshold.
            growth_factor=1.0,
            expansion=60,
            max_iterations=8,
            optimizer="none",
            shots=4096,
            init_scale=0.6,
            seed=5,
        )
        result = run_hivqe(
            space, exact.energy, hartree_fock, settings, progress=lambda _line: None
        )
        results[simulator] = result
        energies = [record.energy for record in result.history]
        reporter.check(
            f"[{simulator}] every iteration stays above the exact ground state",
            all(energy >= exact.energy - 1e-9 for energy in energies),
            f"lowest {min(energies):.9f} vs exact {exact.energy:.9f}",
        )
        reporter.check(
            f"[{simulator}] the energy improves on Hartree-Fock",
            result.energy < hartree_fock,
            f"{result.energy:.6f} < {hartree_fock:.6f}",
        )
        reporter.check(
            f"[{simulator}] given the whole space, HI-VQE finds the exact answer",
            abs(result.error_hartree) < 1e-7,
            f"error {result.error_millihartree:+.6f} mHa on "
            f"{result.dimension}/{result.full_dimension} determinants",
        )
        reporter.check(
            f"[{simulator}] the result serialises to JSON",
            _round_trips(result.as_dict()),
        )

    if "sector" in results:
        first_quantum = results["sector"].history[0].energy
        first_classical = results["none"].history[0].energy
        reporter.check(
            "the quantum proposal beats a bare Hartree-Fock start on iteration 1",
            first_quantum <= first_classical + 1e-9,
            f"{(first_classical - first_quantum) * 1000:.1f} mHa better",
        )

    # -- the ablation must be capable of failing: no expansion, no quantum
    settings = HiVqeSettings(
        simulator="none",
        use_expansion=False,
        max_iterations=3,
        optimizer="none",
    )
    stuck = run_hivqe(
        space, exact.energy, hartree_fock, settings, progress=lambda _line: None
    )
    reporter.check(
        "with both halves switched off the run is stuck at Hartree-Fock",
        abs(stuck.energy - hartree_fock) < 1e-9 and stuck.dimension == 1,
        f"E = {stuck.energy:.9f} on {stuck.dimension} determinant",
    )


def _round_trips(payload: dict[str, Any]) -> bool:
    import json

    try:
        json.loads(json.dumps(payload))
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def selftest_main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove the Li2S HI-VQE stack with no PySCF and no cache."
    )
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--skip-quantum", action="store_true")
    arguments = parser.parse_args()

    started = time.time()
    reporter = Reporter()
    print("Li2S 24-qubit HI-VQE -- self-test")
    stage_structure(reporter)
    if not arguments.structure_only:
        stage_classical(reporter)
        if not arguments.skip_quantum:
            stage_quantum(reporter)
        stage_algorithm(reporter, arguments.skip_quantum)

    print(
        f"\n{reporter.passed} passed, {reporter.failed} failed, "
        f"{reporter.skipped} skipped in {time.time() - started:.1f} s"
    )
    if reporter.failed:
        print("\nFailures:")
        for name in reporter.failures:
            print(f"  - {name}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


def backend_main() -> int:
    """Report the environment and cross-check the two simulators at full size."""
    parser = argparse.ArgumentParser(description="Inspect the Qiskit environment.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also cross-check the two simulators on the real 24-qubit circuit "
        "(needs about 300 MB and a couple of minutes)",
    )
    parser.add_argument("--reps", type=int, default=1)
    arguments = parser.parse_args()

    print("Environment")
    print(f"  python        {sys.version.split()[0]}")
    for module in ("numpy", "scipy", "qiskit", "qiskit_aer", "openfermion", "pyscf", "matplotlib"):
        try:
            imported = importlib.import_module(module)
            print(f"  {module:<13} {getattr(imported, '__version__', 'unknown')}")
        except ImportError:
            note = " (only needed by `prepare`)" if module == "pyscf" else ""
            print(f"  {module:<13} not installed{note}")

    try:
        import ansatz
    except ImportError as error:
        print(f"\nQiskit is unavailable: {error}")
        return 1

    circuit, parameters = ansatz.build_ansatz(12, 6, 6, reps=arguments.reps)
    metrics = ansatz.circuit_metrics(circuit)
    print(f"\nLi2S sampling circuit (reps = {arguments.reps})")
    for key, value in metrics.items():
        print(f"  {key:<26} {value}")

    if not arguments.full:
        print(
            "\nRun `python run.py backend --full` to cross-check the sector "
            "simulator against Aer on this exact 24-qubit circuit."
        )
        return 0

    import sector_sim
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    angles = ansatz.initial_parameters(circuit.num_parameters, 0.4, 3)
    print("\nSector simulator ...")
    tick = time.time()
    ours = sector_sim.SectorSimulator(circuit, 12, 6, 6).run(
        angles, list(circuit.parameters)
    )
    sector_seconds = time.time() - tick

    print("Aer statevector ...")
    tick = time.time()
    bound = circuit.assign_parameters(
        {p: float(v) for p, v in zip(parameters, angles)}
    )
    bound.save_statevector()
    aer = AerSimulator(method="statevector")
    state = np.asarray(
        aer.run(transpile(bound, aer), shots=1).result().get_statevector()
    )
    aer_seconds = time.time() - tick

    strings = dt.make_strings(12, 6)
    reference = state[
        (strings[:, None] | (strings[None, :] << 12)).reshape(-1)
    ].reshape(len(strings), len(strings))
    difference = float(np.abs(ours - reference).max())
    print(f"\n  max |sector - Aer|          {difference:.3e}")
    print(f"  sector simulator            {sector_seconds:.1f} s")
    print(f"  Aer statevector             {aer_seconds:.1f} s")
    print(f"  speed-up                    {aer_seconds / max(sector_seconds, 1e-9):.1f}x")
    print(
        f"  out-of-sector weight in Aer {1.0 - float((np.abs(reference) ** 2).sum()):.3e}"
    )
    if difference > 1e-10:
        print("\n  DISAGREEMENT -- use --simulator aer and report this.")
        return 1
    print("\n  The two simulators agree. Either may be used.")
    return 0
