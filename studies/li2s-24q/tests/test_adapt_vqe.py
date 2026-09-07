"""ADAPT-VQE, checked against things that are exactly known.

Every check here is against an independent computation rather than against a
previously recorded number: the exponential against `scipy.linalg.expm` on the
dense generator, the analytic gradients against central finite differences, and
the converged energy against a direct diagonalisation of the same Hamiltonian.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapt_vqe import (  # noqa: E402
    AdaptSettings,
    AdaptVqe,
    GeneratorAlgebra,
    PoolOperator,
    build_pool,
)
from determinants import (  # noqa: E402
    Subspace,
    make_strings,
    solve_subspace,
    spin_square,
)
from synthetic import synthetic_active_space  # noqa: E402


def small_space(n_orbitals=4, n_alpha=2, n_beta=2, seed=3):
    return synthetic_active_space(
        n_orbitals=n_orbitals, n_alpha=n_alpha, n_beta=n_beta, seed=seed
    )


def dense_generator(algebra, operator, shape):
    """A as an explicit matrix, by applying it to every basis vector."""
    size = shape[0] * shape[1]
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        columns.append(algebra.apply(operator, unit.reshape(shape)).reshape(-1))
    return np.column_stack(columns)


class TestGeneratorAlgebra(unittest.TestCase):
    def setUp(self):
        self.space = small_space()
        self.strings_a = make_strings(self.space.n_orbitals, self.space.n_alpha)
        self.strings_b = make_strings(self.space.n_orbitals, self.space.n_beta)
        self.algebra = GeneratorAlgebra(
            self.space.n_orbitals, self.strings_a, self.strings_b
        )
        self.shape = (len(self.strings_a), len(self.strings_b))

    def test_generator_is_antisymmetric(self):
        """A = T - T^dagger must be antisymmetric, or exp(A) is not orthogonal."""
        for operator in (PoolOperator(((2, 0),)), PoolOperator(((3, 1), (2, 0)))):
            dense = dense_generator(self.algebra, operator, self.shape)
            self.assertLess(
                np.abs(dense + dense.T).max(),
                1e-12,
                f"{operator.label()} is not antisymmetric",
            )

    def test_exponential_matches_scipy(self):
        """Scaled Taylor against an independent dense matrix exponential."""
        from scipy.linalg import expm

        rng = np.random.default_rng(0)
        for operator in (PoolOperator(((2, 0),)), PoolOperator(((3, 1), (2, 0)))):
            dense = dense_generator(self.algebra, operator, self.shape)
            norm = self.algebra.norm_estimate(operator)
            for theta in (-1.3, -0.2, 0.05, 0.7, 2.0):
                start = rng.standard_normal(self.shape)
                mine = self.algebra.exponentiate(operator, theta, start, norm)
                theirs = (expm(theta * dense) @ start.reshape(-1)).reshape(self.shape)
                self.assertLess(
                    np.abs(mine - theirs).max(),
                    1e-9,
                    f"{operator.label()} at theta={theta}",
                )

    def test_exponential_preserves_norm(self):
        """exp(theta A) is orthogonal, so it cannot change the norm."""
        operator = PoolOperator(((3, 1), (2, 0)))
        norm = self.algebra.norm_estimate(operator)
        start = np.zeros(self.shape)
        start[0, 0] = 1.0
        for theta in (0.3, -1.1, 2.5):
            moved = self.algebra.exponentiate(operator, theta, start, norm)
            self.assertAlmostEqual(np.linalg.norm(moved), 1.0, places=10)

    def test_spin_summed_excitation_preserves_spin(self):
        """A spin-adapted generator cannot contaminate a singlet reference."""
        subspace = Subspace(self.strings_a, self.strings_b)
        start = np.zeros(self.shape)
        start[0, 0] = 1.0
        self.assertAlmostEqual(
            spin_square(self.space, subspace, start), 0.0, places=10
        )
        operator = PoolOperator(((2, 0), (3, 1)))
        norm = self.algebra.norm_estimate(operator)
        moved = self.algebra.exponentiate(operator, 0.4, start, norm)
        self.assertLess(
            abs(spin_square(self.space, subspace, moved)),
            1e-8,
            "spin-summed generator produced a contaminated state",
        )


class TestPool(unittest.TestCase):
    def test_uccsd_pool_excites_out_of_the_occupied_set(self):
        pool = build_pool(4, 2, 2, generalized=False)
        singles = [op for op in pool if op.rank == 1]
        self.assertEqual(len(singles), 4)  # 2 virtual x 2 occupied
        for operator in singles:
            (p, q), = operator.pairs
            self.assertGreaterEqual(p, 2)
            self.assertLess(q, 2)

    def test_generalized_pool_is_larger(self):
        self.assertGreater(
            len(build_pool(4, 2, 2, generalized=True)),
            len(build_pool(4, 2, 2, generalized=False)),
        )

    def test_pool_rejects_open_shell(self):
        with self.assertRaises(ValueError):
            build_pool(4, 3, 1)


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.space = small_space()
        self.engine = AdaptVqe(self.space, AdaptSettings(), progress=lambda _: None)
        self.engine.ansatz = [
            PoolOperator(((2, 0),)),
            PoolOperator(((3, 1), (2, 0))),
            PoolOperator(((3, 0),)),
        ]
        self.engine.ansatz_norms = [
            self.engine.algebra.norm_estimate(op) for op in self.engine.ansatz
        ]

    def test_adjoint_gradient_matches_finite_difference(self):
        """The whole point of the adjoint sweep is that it is exact."""
        angles = np.array([0.21, -0.35, 0.08])
        energy, gradient = self.engine.energy_and_gradient(angles)
        self.assertAlmostEqual(energy, self.engine.energy(angles), places=10)

        delta = 1e-5
        for index in range(len(angles)):
            up = angles.copy()
            down = angles.copy()
            up[index] += delta
            down[index] -= delta
            numerical = (self.engine.energy(up) - self.engine.energy(down)) / (
                2 * delta
            )
            self.assertAlmostEqual(
                gradient[index],
                numerical,
                places=6,
                msg=f"analytic gradient {index} disagrees with finite difference",
            )

    def test_pool_gradient_matches_finite_difference(self):
        """Appending an operator at theta=0 must move the energy at that rate."""
        angles = np.array([0.15, -0.22, 0.31])
        matrix = self.engine.state(angles)
        screened = self.engine.pool_gradients(matrix)

        delta = 1e-5
        for index in (0, len(self.engine.pool) // 2, len(self.engine.pool) - 1):
            operator = self.engine.pool[index]
            norm = self.engine.algebra.norm_estimate(operator)
            up = self.engine.algebra.exponentiate(operator, delta, matrix, norm)
            down = self.engine.algebra.exponentiate(operator, -delta, matrix, norm)
            numerical = (
                self.engine.hamiltonian.expectation(up)
                - self.engine.hamiltonian.expectation(down)
            ) / (2 * delta)
            self.assertAlmostEqual(screened[index], abs(numerical), places=6)


class TestConvergence(unittest.TestCase):
    def test_reaches_the_exact_energy(self):
        """Given enough operators, ADAPT-VQE must find the exact ground state."""
        space = small_space()
        subspace = Subspace(
            make_strings(space.n_orbitals, space.n_alpha),
            make_strings(space.n_orbitals, space.n_beta),
        )
        exact = solve_subspace(space, subspace).energy

        settings = AdaptSettings(
            max_operators=25, gradient_tolerance=1e-6, energy_tolerance=1e-12
        )
        engine = AdaptVqe(space, settings, progress=lambda _: None)
        hartree_fock = engine.hamiltonian.expectation(engine.reference_state)
        result = engine.run(exact, hartree_fock)

        self.assertGreaterEqual(
            result.energy,
            exact - 1e-9,
            "ADAPT-VQE went below the exact ground state, which is impossible",
        )
        self.assertLess(
            abs(result.error_hartree),
            1.6e-3,
            f"did not reach chemical accuracy: {result.error_hartree * 1000:.4f} mHa",
        )
        self.assertLess(abs(result.spin_squared), 1e-6, "state is spin contaminated")

    def test_every_iteration_stays_above_the_exact_energy(self):
        """Variational means variational at every step, not only at the end."""
        space = small_space()
        subspace = Subspace(
            make_strings(space.n_orbitals, space.n_alpha),
            make_strings(space.n_orbitals, space.n_beta),
        )
        exact = solve_subspace(space, subspace).energy
        engine = AdaptVqe(
            space, AdaptSettings(max_operators=8), progress=lambda _: None
        )
        hartree_fock = engine.hamiltonian.expectation(engine.reference_state)
        result = engine.run(exact, hartree_fock)
        for record in result.iterations:
            self.assertGreaterEqual(
                record.energy,
                exact - 1e-9,
                f"iteration {record.index} fell below the exact ground state",
            )

    def test_energy_decreases_monotonically(self):
        """Each added operator can only help: the previous ansatz is nested."""
        space = small_space()
        subspace = Subspace(
            make_strings(space.n_orbitals, space.n_alpha),
            make_strings(space.n_orbitals, space.n_beta),
        )
        exact = solve_subspace(space, subspace).energy
        engine = AdaptVqe(
            space, AdaptSettings(max_operators=8), progress=lambda _: None
        )
        hartree_fock = engine.hamiltonian.expectation(engine.reference_state)
        result = engine.run(exact, hartree_fock)
        energies = [record.energy for record in result.iterations]
        for earlier, later in zip(energies, energies[1:]):
            self.assertLessEqual(later, earlier + 1e-9)

    def test_result_serialises(self):
        import json

        space = small_space()
        engine = AdaptVqe(
            space, AdaptSettings(max_operators=3), progress=lambda _: None
        )
        hartree_fock = engine.hamiltonian.expectation(engine.reference_state)
        result = engine.run(hartree_fock - 0.1, hartree_fock)
        json.dumps(result.as_dict())


if __name__ == "__main__":
    unittest.main()
