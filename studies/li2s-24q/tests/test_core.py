"""Unit tests for the Li2S 24-qubit HI-VQE workflow.

Run from the folder above:  python -m unittest discover -s tests -v

These are the fast, always-run checks. `run.py selftest` is the heavier proof
against exact answers and an independent Jordan-Wigner implementation; this file
covers conventions, invariants, cache integrity and the CLI surface, all in a
few seconds.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import determinants as dt  # noqa: E402
import hamiltonian as ham  # noqa: E402
from hivqe import (  # noqa: E402
    HiVqeSettings,
    grow_subspace,
    perturbative_correction,
    prune_subspace,
    rank_candidates,
    run_hivqe,
    spin_complete_subspace,
    spin_contamination,
)
from molecule import DISSOCIATION_DISTANCES, LI2S  # noqa: E402
from synthetic import check_eri_symmetry, synthetic_active_space  # noqa: E402


class TestDeterminantConventions(unittest.TestCase):
    def test_strings_are_complete_sorted_and_correct_weight(self):
        strings = dt.make_strings(6, 3)
        self.assertEqual(len(strings), 20)
        self.assertTrue(np.all(np.diff(strings) > 0))
        for value in strings:
            self.assertEqual(int(value).bit_count(), 3)

    def test_li2s_full_space_matches_the_published_dimension(self):
        self.assertEqual(dt.n_determinants(12, 6, 6), 853_776)

    def test_excitation_sign_from_first_principles(self):
        # a+_2 a_0 |{0,1}>: a_0 passes nothing, a+_2 passes occupied orbital 1.
        self.assertEqual(dt.apply_single_excitation(0b0011, 0, 2), (0b0110, -1))

        # a+_2 a_1 |{0,1}>: a_1 crosses orbital 0 and a+_2 crosses it again,
        # so the two signs cancel. Verified against OpenFermion in selftest.
        self.assertEqual(dt.apply_single_excitation(0b0011, 1, 2), (0b0101, 1))
        # No sign at all when nothing is crossed.
        self.assertEqual(dt.apply_single_excitation(0b0001, 0, 1), (0b0010, 1))

    def test_excitation_vanishes_when_it_should(self):
        self.assertEqual(dt.apply_single_excitation(0b0011, 0, 1), (0, 0))
        self.assertEqual(dt.apply_single_excitation(0b0011, 2, 3), (0, 0))

    def test_popcount_matches_python(self):
        strings = dt.make_strings(10, 5)
        np.testing.assert_array_equal(
            dt.popcount(strings), [int(v).bit_count() for v in strings]
        )

    def test_hartree_fock_determinant_is_aufbau(self):
        self.assertEqual(dt.hartree_fock_determinant(6, 6), (0b111111, 0b111111))

    def test_excitation_generator_is_degree_one_and_two_without_repeats(self):
        reference = dt.hartree_fock_determinant(3, 3)
        candidates = dt.single_and_double_excitations(reference, 6)
        self.assertEqual(len(candidates), len(set(candidates)))
        degrees = {dt.determinant_excitation_degree(reference, c) for c in candidates}
        self.assertEqual(degrees, {1, 2})


class TestSubspaceHamiltonian(unittest.TestCase):
    def setUp(self):
        self.space = synthetic_active_space(5, 2, 3, seed=9)
        self.all_a = dt.make_strings(5, 2)
        self.all_b = dt.make_strings(5, 3)
        self.parent = dt.Subspace(self.all_a, self.all_b)

    def test_synthetic_integrals_have_the_symmetries_of_real_ones(self):
        self.assertLess(max(check_eri_symmetry(self.space.eri).values()), 1e-14)
        self.assertLess(
            float(np.abs(self.space.h1e - self.space.h1e.T).max()), 1e-14
        )

    def test_contraction_equals_slater_condon(self):
        operator = dt.SubspaceHamiltonian(self.space, self.parent)
        contracted = operator.dense()
        determinants = [
            (int(a), int(b)) for a in self.all_a for b in self.all_b
        ]
        explicit = np.array(
            [
                [dt.matrix_element(self.space, bra, ket) for ket in determinants]
                for bra in determinants
            ]
        )
        np.testing.assert_allclose(contracted, explicit, atol=1e-11)

    def test_projection_equals_the_exact_sub_block(self):
        """P H P must be the sub-block of H, not a smaller operator.

        This is the regression test for the bug that produced subspace energies
        below the exact ground state.
        """
        parent_dense = dt.SubspaceHamiltonian(self.space, self.parent).dense()
        rng = np.random.default_rng(4)
        for _ in range(5):
            rows = np.sort(rng.choice(len(self.all_a), size=5, replace=False))
            columns = np.sort(rng.choice(len(self.all_b), size=5, replace=False))
            block = dt.SubspaceHamiltonian(
                self.space, dt.Subspace(self.all_a[rows], self.all_b[columns])
            ).dense()
            indices = (rows[:, None] * len(self.all_b) + columns[None, :]).reshape(-1)
            np.testing.assert_allclose(
                block, parent_dense[np.ix_(indices, indices)], atol=1e-11
            )

    def test_no_subspace_falls_below_the_exact_ground_state(self):
        exact = dt.solve_subspace(self.space, self.parent)
        rng = np.random.default_rng(0)
        for _ in range(6):
            rows = np.sort(rng.choice(len(self.all_a), size=3, replace=False))
            columns = np.sort(rng.choice(len(self.all_b), size=4, replace=False))
            trial = dt.solve_subspace(
                self.space, dt.Subspace(self.all_a[rows], self.all_b[columns])
            )
            self.assertGreaterEqual(trial.energy, exact.energy - 1e-10)

    def test_diagonal_matches_slater_condon(self):
        operator = dt.SubspaceHamiltonian(self.space, self.parent)
        diagonal = operator.diagonal()
        for row, a in enumerate(self.parent.strings_a):
            for column, b in enumerate(self.parent.strings_b):
                self.assertAlmostEqual(
                    diagonal[row, column],
                    dt.matrix_element(self.space, (int(a), int(b)), (int(a), int(b))),
                    places=9,
                )

    def test_davidson_matches_dense_diagonalisation(self):
        operator = dt.SubspaceHamiltonian(self.space, self.parent)
        exact = float(np.linalg.eigvalsh(operator.dense())[0])
        self.assertAlmostEqual(
            dt.solve_subspace(self.space, self.parent).energy, exact, places=9
        )

    def test_same_spin_block_is_symmetric(self):
        block = np.asarray(
            dt.same_spin_hamiltonian(self.space, self.all_a).todense()
        )
        np.testing.assert_allclose(block, block.T, atol=1e-12)

    def test_embedding_preserves_coefficients(self):
        small = dt.Subspace(self.all_a[:2], self.all_b[:3])
        coefficients = np.arange(6, dtype=float).reshape(2, 3)
        embedded = dt.embed_coefficients(coefficients, small, self.parent)
        self.assertAlmostEqual(embedded.sum(), coefficients.sum())
        self.assertEqual(int((embedded != 0).sum()), int((coefficients != 0).sum()))

    def test_embedding_refuses_a_subspace_that_does_not_contain_the_old_one(self):
        small = dt.Subspace(self.all_a[:2], self.all_b[:3])
        disjoint = dt.Subspace(self.all_a[3:5], self.all_b[3:5])
        with self.assertRaises(ValueError):
            dt.embed_coefficients(np.ones((2, 3)), small, disjoint)


class TestObservables(unittest.TestCase):
    def test_closed_shell_hartree_fock_is_a_singlet(self):
        space = synthetic_active_space(5, 2, 2, seed=11)
        spin = dt.spin_square(space, dt.hartree_fock_subspace(2, 2), np.ones((1, 1)))
        self.assertLess(abs(spin), 1e-12)

    def test_exact_ground_state_is_a_spin_eigenstate(self):
        space = synthetic_active_space(5, 2, 2, seed=11)
        subspace = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 2))
        result = dt.solve_subspace(space, subspace)
        spin = dt.spin_square(space, subspace, result.coefficients)
        self.assertLess(spin_contamination(spin), 1e-9)

    def test_occupancies_sum_to_the_electron_count(self):
        space = synthetic_active_space(5, 2, 2, seed=11)
        subspace = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 2))
        result = dt.solve_subspace(space, subspace)
        alpha, beta = dt.orbital_occupancies(space, subspace, result.coefficients)
        self.assertAlmostEqual(float(alpha.sum()), 2.0, places=9)
        self.assertAlmostEqual(float(beta.sum()), 2.0, places=9)

    def test_spin_contamination_measures_distance_to_the_nearest_multiplet(self):
        for clean in (0.0, 0.75, 2.0, 3.75, 6.0):
            self.assertLess(spin_contamination(clean), 1e-12)
        self.assertAlmostEqual(spin_contamination(0.5), 0.25, places=12)


class TestSubspaceBookkeeping(unittest.TestCase):
    def setUp(self):
        self.space = synthetic_active_space(5, 2, 2, seed=11)
        self.subspace = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 2))
        self.anchor = dt.hartree_fock_determinant(2, 2)

    def test_pruning_keeps_the_anchor_and_respects_the_cap(self):
        weights = np.zeros(self.subspace.shape)
        weights[0, 0] = 1.0
        pruned, kept = prune_subspace(self.subspace, weights, 4, 1e-6, self.anchor)
        self.assertLessEqual(pruned.dimension, 4)
        self.assertTrue(pruned.contains(self.anchor))
        self.assertEqual(kept.shape, pruned.shape)

    def test_pruning_renormalises_what_it_keeps(self):
        rng = np.random.default_rng(3)
        weights = rng.normal(size=self.subspace.shape)
        weights /= np.linalg.norm(weights)
        _, kept = prune_subspace(self.subspace, weights, 9, 1e-12, self.anchor)
        self.assertAlmostEqual(float(np.linalg.norm(kept)), 1.0, places=12)

    def test_growth_stops_at_the_dimension_cap(self):
        candidates = [
            (1.0, (int(a), int(b)))
            for a in self.subspace.strings_a
            for b in self.subspace.strings_b
        ]
        grown, added = grow_subspace(
            dt.hartree_fock_subspace(2, 2), candidates, len(candidates), 6
        )
        self.assertLessEqual(grown.dimension, 6)
        self.assertGreater(added, 0)

    def test_ranking_only_offers_configurations_outside_the_subspace(self):
        partial = dt.Subspace(self.subspace.strings_a[:3], self.subspace.strings_b[:3])
        solution = dt.solve_subspace(self.space, partial)
        ranked = rank_candidates(self.space, partial, solution.coefficients, 4)
        self.assertGreater(len(ranked), 0)
        for _, candidate in ranked:
            self.assertFalse(partial.contains(candidate))
        scores = [score for score, _ in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.space = synthetic_active_space(5, 2, 2, seed=11)
        self.subspace = dt.Subspace(dt.make_strings(5, 2), dt.make_strings(5, 2))
        self.exact = dt.solve_subspace(self.space, self.subspace)
        reference = dt.hartree_fock_determinant(2, 2)
        self.hartree_fock = dt.matrix_element(self.space, reference, reference)

    def _run(self, **overrides):
        settings = HiVqeSettings(
            simulator="none",
            max_determinants=self.subspace.dimension,
            # Never prune: the whole space is the cap and the expansion target.
            growth_factor=1.0,
            expansion=40,
            max_iterations=6,
            optimizer="none",
            **overrides,
        )
        return run_hivqe(
            self.space,
            self.exact.energy,
            self.hartree_fock,
            settings,
            progress=lambda _line: None,
        )

    def test_every_iteration_is_variational(self):
        result = self._run()
        for record in result.history:
            self.assertGreaterEqual(record.energy, self.exact.energy - 1e-9)

    def test_it_finds_the_exact_answer_given_the_whole_space(self):
        result = self._run()
        self.assertLess(abs(result.error_hartree), 1e-8)
        self.assertEqual(result.verdict, "CHEMICAL ACCURACY")

    def test_with_both_halves_off_it_is_stuck_at_hartree_fock(self):
        result = self._run(use_expansion=False)
        self.assertEqual(result.dimension, 1)
        self.assertAlmostEqual(result.energy, self.hartree_fock, places=9)

    def test_the_result_serialises(self):
        payload = self._run().as_dict()
        json.loads(json.dumps(payload))
        for key in (
            "energy",
            "error_millihartree",
            "dimension",
            "subspace_fraction",
            "verdict",
            "history",
            "settings",
        ):
            self.assertIn(key, payload)


class TestMoleculeSpecification(unittest.TestCase):
    def test_active_space_is_the_published_one(self):
        self.assertEqual(LI2S.n_active_electrons, 12)
        self.assertEqual(LI2S.n_active_orbitals, 12)
        self.assertEqual(LI2S.n_qubits, 24)
        self.assertEqual(
            LI2S.published_reference["full_cas_determinants"],
            dt.n_determinants(12, 6, 6),
        )

    def test_frozen_core_arithmetic_is_five_orbitals(self):
        # Li2S has 22 electrons; 12 active leaves 10, i.e. 5 doubly occupied.
        total_electrons = 3 + 3 + 16
        self.assertEqual((total_electrons - LI2S.n_active_electrons) % 2, 0)
        self.assertEqual((total_electrons - LI2S.n_active_electrons) // 2, 5)

    def test_geometry_is_linear_and_places_the_scanned_bond(self):
        atoms = LI2S.geometry(4.0)
        self.assertEqual([symbol for symbol, _ in atoms], ["S", "Li", "Li"])
        self.assertEqual(atoms[0][1], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(atoms[1][1][2], -LI2S.equilibrium_bond_angstrom)
        self.assertAlmostEqual(atoms[2][1][2], 4.0)
        for _, coordinates in atoms:
            self.assertEqual(coordinates[0], 0.0)
            self.assertEqual(coordinates[1], 0.0)

    def test_equilibrium_geometry_is_symmetric(self):
        atoms = LI2S.geometry()
        self.assertAlmostEqual(atoms[1][1][2], -atoms[2][1][2])

    def test_scan_distances_are_ascending_and_span_dissociation(self):
        self.assertEqual(list(DISSOCIATION_DISTANCES), sorted(DISSOCIATION_DISTANCES))
        self.assertLess(min(DISSOCIATION_DISTANCES), LI2S.equilibrium_bond_angstrom)
        self.assertGreater(max(DISSOCIATION_DISTANCES), 5.0)


class TestCacheAndReceipt(unittest.TestCase):
    def test_array_encoding_round_trips_bit_exactly(self):
        rng = np.random.default_rng(1)
        array = rng.normal(size=(4, 4, 4, 4))
        np.testing.assert_array_equal(
            ham.decode_array(ham.encode_array(array)), array
        )

    def test_spec_fingerprint_ignores_the_literature_citation(self):
        from dataclasses import replace

        changed = replace(LI2S, published_reference={"source": "something else"})
        self.assertEqual(ham.spec_fingerprint(LI2S), ham.spec_fingerprint(changed))

    def test_spec_fingerprint_tracks_the_active_space(self):
        from dataclasses import replace

        changed = replace(LI2S, n_active_orbitals=11)
        self.assertNotEqual(ham.spec_fingerprint(LI2S), ham.spec_fingerprint(changed))

    def test_receipt_path_derives_from_the_cache_name(self):
        self.assertEqual(
            ham.validation_receipt_path(Path("cache/li2s_r2.100.json")).name,
            "li2s_r2.100.validated.json",
        )

    def test_cache_filenames_are_stable_and_sortable(self):
        self.assertEqual(
            ham.cache_path_for("cache", 2.1).name, "li2s_r2.100.json"
        )
        self.assertEqual(
            ham.cache_path_for("cache", 10.0).name, "li2s_r10.000.json"
        )

    def test_atomic_write_leaves_no_temporary_behind(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "sub" / "thing.json"
            ham.write_json_atomic(target, {"a": 1})
            self.assertEqual(json.loads(target.read_text()), {"a": 1})
            self.assertEqual(
                [p.name for p in target.parent.iterdir()], ["thing.json"]
            )

    def test_physics_fingerprint_covers_the_physics_modules(self):
        for name in ham.PHYSICS_MODULES:
            self.assertTrue((ROOT / name).is_file(), name)
        self.assertIn("determinants.py", ham.PHYSICS_MODULES)
        self.assertNotIn("visualize.py", ham.PHYSICS_MODULES)
        self.assertNotIn("ansatz.py", ham.PHYSICS_MODULES)

    def test_workflow_fingerprint_is_stable_and_covers_every_module(self):
        self.assertEqual(ham.workflow_fingerprint(), ham.workflow_fingerprint())
        self.assertNotEqual(ham.workflow_fingerprint(), ham.physics_fingerprint())

    def test_missing_receipt_is_a_clear_error(self):
        from validation import require_validation_receipt

        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "li2s_r2.100.json"
            cache.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                require_validation_receipt(cache, LI2S)


class TestCommandSurface(unittest.TestCase):
    def test_every_advertised_command_is_implemented(self):
        import run as entry

        usage = entry._usage()
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        for command in (
            "selftest",
            "backend",
            "geometry",
            "prepare",
            "validate",
            "paulis",
            "classical",
            "hivqe",
            "scan",
            "summary",
            "plot",
            "report",
        ):
            self.assertIn(command, usage, f"{command} is not documented")
            self.assertIn(f'command == "{command}"', source, f"{command} is not wired")

    def test_unknown_commands_fail_loudly(self):
        import run as entry

        argv = sys.argv
        try:
            sys.argv = ["run.py", "definitely-not-a-command"]
            self.assertEqual(entry.main(), 2)
        finally:
            sys.argv = argv

    def test_module_layering_is_enforced(self):
        import re

        def imports(name: str, package: str) -> bool:
            text = (ROOT / f"{name}.py").read_text(encoding="utf-8")
            return bool(
                re.search(
                    rf"^\s*(?:from\s+{package}[\.\s]|import\s+{package}\b)",
                    text,
                    flags=re.MULTILINE,
                )
            )

        for module in ("determinants", "hivqe", "synthetic", "summary", "validation"):
            self.assertFalse(imports(module, "pyscf"), f"{module} imports pyscf")
        for module in ("determinants", "hivqe", "validation", "synthetic"):
            self.assertFalse(imports(module, "qiskit"), f"{module} imports qiskit")


class TestEngineOptimisations(unittest.TestCase):
    """The rewritten hot paths must be bit-for-bit what they replaced."""

    def setUp(self):
        self.space = synthetic_active_space(6, 3, 3, seed=19)
        self.strings = dt.make_strings(6, 3)

    def test_chunked_sigma_equals_the_whole_one(self):
        subspace = dt.Subspace(self.strings, self.strings)
        whole = dt.SubspaceHamiltonian(self.space, subspace)
        # A budget of one byte forces a single column at a time.
        chunked = dt.SubspaceHamiltonian(self.space, subspace, memory_budget_bytes=1)
        self.assertLess(chunked._chunk(), whole._chunk())
        rng = np.random.default_rng(4)
        for _ in range(3):
            vector = rng.normal(size=subspace.shape)
            np.testing.assert_allclose(
                whole.sigma(vector), chunked.sigma(vector), atol=1e-12
            )

    def test_the_reused_sigma_buffer_does_not_leak_between_calls(self):
        subspace = dt.Subspace(self.strings, self.strings)
        operator = dt.SubspaceHamiltonian(self.space, subspace)
        rng = np.random.default_rng(9)
        first = rng.normal(size=subspace.shape)
        second = rng.normal(size=subspace.shape)
        expected = operator.sigma(first).copy()
        operator.sigma(second)
        np.testing.assert_allclose(operator.sigma(first), expected, atol=1e-14)

    def test_reindexing_keeps_the_shared_determinants_and_zeroes_the_rest(self):
        old_space = dt.Subspace(
            np.array([0b000111, 0b001011], dtype=np.int64),
            np.array([0b000111, 0b001011], dtype=np.int64),
        )
        new_space = dt.Subspace(
            np.array([0b001011, 0b010011], dtype=np.int64),
            np.array([0b000111, 0b001011], dtype=np.int64),
        )
        coefficients = np.array([[1.0, 2.0], [3.0, 4.0]])
        moved = dt.reindex_coefficients(coefficients, old_space, new_space)
        # Row 0b001011 survives and keeps its values; 0b010011 is new, so zero.
        np.testing.assert_allclose(moved[0], [3.0, 4.0])
        np.testing.assert_allclose(moved[1], [0.0, 0.0])


class TestPerturbativeCorrection(unittest.TestCase):
    def setUp(self):
        self.space = synthetic_active_space(6, 3, 3, seed=23)
        self.strings = dt.make_strings(6, 3)
        self.full = dt.Subspace(self.strings, self.strings)
        self.exact = dt.solve_subspace(self.space, self.full).energy
        anchor = dt.hartree_fock_determinant(3, 3)
        self.hartree_fock = dt.matrix_element(self.space, anchor, anchor)

    def _converged(self, dimension, **overrides):
        settings = HiVqeSettings(
            max_determinants=dimension,
            expansion=40,
            max_iterations=6,
            simulator="none",
            optimizer="none",
            shots=0,
            **overrides,
        )
        return run_hivqe(
            self.space,
            self.exact,
            self.hartree_fock,
            settings,
            progress=lambda _line: None,
        )

    def test_the_correction_lowers_the_energy_and_is_not_added_in(self):
        result = self._converged(60)
        self.assertLessEqual(result.pt2_correction, 0.0)
        self.assertGreater(result.pt2_determinants, 0)
        # `energy` stays the variational number the upper bound applies to.
        self.assertGreaterEqual(result.energy, self.exact - 1e-9)
        self.assertAlmostEqual(
            result.energy_pt2, result.energy + result.pt2_correction, places=12
        )

    def test_the_correction_moves_the_energy_towards_the_exact_answer(self):
        result = self._converged(60)
        self.assertLess(abs(result.error_pt2_hartree), abs(result.error_hartree))

    def test_a_complete_space_has_nothing_left_to_correct(self):
        # Nothing lies outside the full space, so the sum is empty.
        correction, count = perturbative_correction(
            self.space,
            self.full,
            dt.solve_subspace(self.space, self.full).coefficients,
            self.exact,
            4,
            self.full.dimension,
        )
        self.assertEqual(count, 0)
        self.assertEqual(correction, 0.0)

    def test_it_can_be_switched_off(self):
        result = self._converged(60, pt2=False)
        self.assertEqual(result.pt2_correction, 0.0)
        self.assertEqual(result.energy_pt2, result.energy)

    def test_the_correction_is_serialised(self):
        payload = self._converged(60).as_dict()
        for key in ("pt2_correction", "pt2_determinants", "energy_pt2",
                    "error_pt2_millihartree"):
            self.assertIn(key, payload)


class TestSpinCompletion(unittest.TestCase):
    def test_completion_shares_one_string_set_between_the_blocks(self):
        subspace = dt.Subspace(
            np.array([0b000111, 0b001011, 0b010011], dtype=np.int64),
            np.array([0b000111, 0b001101], dtype=np.int64),
        )
        coefficients = np.array([[0.9, 0.2], [0.1, 0.05], [0.3, 0.02]])
        completed, moved = spin_complete_subspace(
            subspace, coefficients, 400, (0b000111, 0b000111)
        )
        np.testing.assert_array_equal(completed.strings_a, completed.strings_b)
        # Closed under the flip: every determinant's partner is present too.
        for string_a in completed.strings_a:
            for string_b in completed.strings_b:
                self.assertTrue(completed.contains((int(string_b), int(string_a))))
        self.assertAlmostEqual(float(np.linalg.norm(moved)), 1.0, places=12)

    def test_completion_respects_the_dimension_cap(self):
        strings = dt.make_strings(6, 3)
        subspace = dt.Subspace(strings, strings)
        coefficients = np.ones((len(strings), len(strings)))
        completed, _ = spin_complete_subspace(
            subspace, coefficients, 25, (0b000111, 0b000111)
        )
        self.assertLessEqual(completed.dimension, 36)

    def test_growth_keeps_the_shared_set_shared(self):
        shared = np.array([0b000111, 0b001011], dtype=np.int64)
        subspace = dt.Subspace(shared, shared)
        ranked = [(1.0, (0b010011, 0b100011)), (0.5, (0b001101, 0b000111))]
        grown, added = grow_subspace(subspace, ranked, 5, 400, spin_complete=True)
        np.testing.assert_array_equal(grown.strings_a, grown.strings_b)
        self.assertGreater(added, 0)

    def test_completion_is_skipped_for_an_open_shell(self):
        # n_alpha != n_beta: one shared set is not even well formed, so the
        # setting has to be ignored rather than applied.
        space = synthetic_active_space(6, 3, 2, seed=31)
        strings_a, strings_b = dt.make_strings(6, 3), dt.make_strings(6, 2)
        exact = dt.solve_subspace(space, dt.Subspace(strings_a, strings_b)).energy
        anchor = dt.hartree_fock_determinant(3, 2)
        settings = HiVqeSettings(
            max_determinants=40,
            expansion=20,
            max_iterations=3,
            simulator="none",
            optimizer="none",
            shots=0,
            spin_complete=True,
        )
        result = run_hivqe(
            space,
            exact,
            dt.matrix_element(space, anchor, anchor),
            settings,
            progress=lambda _line: None,
        )
        self.assertGreaterEqual(result.energy, exact - 1e-9)


class TestCandidateRanking(unittest.TestCase):
    def setUp(self):
        self.space = synthetic_active_space(6, 3, 3, seed=29)
        strings = dt.make_strings(6, 3)[:4]
        self.subspace = dt.Subspace(strings, strings)
        self.coefficients = np.eye(4) * 0.5

    def test_pt2_ranking_is_the_coupling_score_over_the_energy_gap(self):
        coupling = {
            key: value
            for value, key in rank_candidates(
                self.space, self.subspace, self.coefficients, 2, ranking="coupling"
            )
        }
        pt2 = {
            key: value
            for value, key in rank_candidates(
                self.space, self.subspace, self.coefficients, 2, -5.0, "pt2"
            )
        }
        self.assertTrue(pt2)
        for determinant, score in pt2.items():
            gap = -5.0 - dt.matrix_element(self.space, determinant, determinant)
            expected = coupling[determinant] ** 2 / abs(gap)
            self.assertAlmostEqual(score, expected, places=10)

    def test_an_unknown_ranking_is_refused(self):
        with self.assertRaises(ValueError):
            rank_candidates(
                self.space, self.subspace, self.coefficients, 2, -5.0, "nonsense"
            )

    def test_no_energy_falls_back_to_the_coupling_score(self):
        without = rank_candidates(self.space, self.subspace, self.coefficients, 2)
        coupling = rank_candidates(
            self.space, self.subspace, self.coefficients, 2, ranking="coupling"
        )
        self.assertEqual(without, coupling)


class TestSamplerCaching(unittest.TestCase):
    def test_an_unchanged_theta_costs_no_circuit_evaluation(self):
        from hivqe import SectorProposer

        proposer = SectorProposer(4, 2, 2, reps=1, shots=0, seed=3)
        angles = np.zeros(proposer.n_parameters)
        first = proposer.sample(angles)
        second = proposer.sample(angles)
        self.assertEqual(proposer.n_circuit_runs, 1)
        # The reported cost has to be what happened, not a constant.
        self.assertEqual(first.circuit_runs, 1)
        self.assertEqual(second.circuit_runs, 0)
        np.testing.assert_array_equal(first.strings_a, second.strings_a)

        proposer.sample(angles + 0.3)
        self.assertEqual(proposer.n_circuit_runs, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
