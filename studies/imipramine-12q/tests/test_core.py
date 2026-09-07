"""Dependency-light tests: specification, fingerprints, receipts, conversion.

Everything up to `OperatorConversionTests` needs nothing but the standard
library and NumPy, so it runs anywhere -- including the Linux box that only
builds the Hamiltonian. The conversion tests need Qiskit and OpenFermion but
*not* PySCF, which is the whole point: on Windows the operator-mapping code can
still be tested even though the molecule itself cannot be built there.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hamiltonian import (  # noqa: E402
    RECEIPT_SCHEMA,
    conjugate_with_x,
    electron_number_operator,
    file_fingerprint,
    hartree_fock_occupation,
    localization_thresholds,
    number_deviation_operator,
    penalty_key,
    physics_fingerprint,
    require_validation_receipt,
    spec_fingerprint,
    to_hartree_fock_frame,
    validation_receipt_path,
    workflow_fingerprint,
    write_json_atomic,
)
from molecule import AROMATIC_CORE_ATOMS, IMIPRAMINE, PUBLISHED_REFERENCE  # noqa: E402


def _installed(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


HAS_QISKIT = _installed("qiskit", "qiskit_aer")
HAS_OPENFERMION = _installed("openfermion")


class MoleculeSpecificationTests(unittest.TestCase):
    def test_active_space_invariants(self) -> None:
        self.assertEqual(IMIPRAMINE.n_active_electrons, 6)
        self.assertEqual(IMIPRAMINE.n_active_orbitals, 6)
        self.assertEqual(2 * IMIPRAMINE.n_active_orbitals, 12)
        self.assertEqual(IMIPRAMINE.charge, 0)
        self.assertEqual(IMIPRAMINE.spin, 0)
        self.assertEqual(IMIPRAMINE.basis, "6-31g")
        self.assertEqual(IMIPRAMINE.n_active_electrons % 2, 0)
        self.assertLessEqual(
            IMIPRAMINE.n_active_electrons, 2 * IMIPRAMINE.n_active_orbitals
        )

    def test_formula_is_imipramine(self) -> None:
        symbols = [symbol for symbol, _ in IMIPRAMINE.atoms]
        self.assertEqual(
            (symbols.count("C"), symbols.count("H"), symbols.count("N")), (19, 24, 2)
        )
        self.assertEqual(len(IMIPRAMINE.atoms), 45)
        self.assertEqual(set(symbols), {"C", "H", "N"})

    def test_geometry_is_physically_sane(self) -> None:
        """Every atom has a neighbour at a bond distance, and none overlap."""
        positions = [coords for _, coords in IMIPRAMINE.atoms]
        for index, position in enumerate(positions):
            others = [
                math.dist(position, other)
                for other_index, other in enumerate(positions)
                if other_index != index
            ]
            self.assertGreater(min(others), 0.85, f"atom {index} sits on another")
            self.assertLess(min(others), 1.85, f"atom {index} is unbonded")

    def test_pi_core_is_the_dibenzazepine_system(self) -> None:
        symbols = [symbol for symbol, _ in IMIPRAMINE.atoms]
        self.assertEqual(tuple(IMIPRAMINE.diagnostic_atom_indices), AROMATIC_CORE_ATOMS)
        self.assertEqual(len(AROMATIC_CORE_ATOMS), 13)
        self.assertEqual(sum(1 for i in AROMATIC_CORE_ATOMS if symbols[i] == "C"), 12)
        self.assertEqual(sum(1 for i in AROMATIC_CORE_ATOMS if symbols[i] == "N"), 1)
        self.assertTrue(all(0 <= i < len(symbols) for i in AROMATIC_CORE_ATOMS))

    def test_virtual_orbitals_have_their_own_looser_threshold(self) -> None:
        """One Mulliken threshold cannot serve occupied and virtual orbitals."""
        occupied, virtual = localization_thresholds(IMIPRAMINE)
        self.assertEqual(occupied, 0.80)
        self.assertEqual(virtual, 0.65)
        self.assertLess(virtual, occupied)

    def test_specification_hash_changes_with_scientific_input(self) -> None:
        original = spec_fingerprint(IMIPRAMINE)
        for changed in (
            replace(IMIPRAMINE, charge=1),
            replace(IMIPRAMINE, basis="6-31g*"),
            replace(IMIPRAMINE, n_active_orbitals=7),
            replace(IMIPRAMINE, min_active_orbital_localization=0.5),
        ):
            with self.subTest(field=changed.name):
                self.assertNotEqual(original, spec_fingerprint(changed))

    def test_literature_annotation_cannot_invalidate_a_cache(self) -> None:
        self.assertEqual(
            spec_fingerprint(IMIPRAMINE),
            spec_fingerprint(replace(IMIPRAMINE, published_reference=None)),
        )
        self.assertEqual(PUBLISHED_REFERENCE["casci_energy_hartree"], -841.953249)


class FingerprintTests(unittest.TestCase):
    def _restore(self, path: Path, content: bytes) -> None:
        path.write_bytes(content)

    def test_hashes_are_valid_sha256(self) -> None:
        for digest in (workflow_fingerprint(), physics_fingerprint()):
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        self.assertNotEqual(workflow_fingerprint(), physics_fingerprint())

    def test_chemistry_edits_invalidate_a_receipt(self) -> None:
        module = ROOT / "hamiltonian.py"
        original = module.read_bytes()
        self.addCleanup(self._restore, module, original)
        before = physics_fingerprint()
        module.write_bytes(original + b"\n# fingerprint probe\n")
        self.assertNotEqual(before, physics_fingerprint())

    def test_simulator_edits_do_not_invalidate_a_receipt(self) -> None:
        """The design point: qiskit_runtime.py cannot change a cached Hamiltonian.

        A Windows machine has no PySCF and therefore cannot re-validate. Binding
        the receipt to the physics modules alone is what lets the simulator layer
        be edited there without invalidating a proof it cannot have affected --
        while the full-folder workflow hash still records the change.
        """
        module = ROOT / "qiskit_runtime.py"
        original = module.read_bytes()
        self.addCleanup(self._restore, module, original)
        physics_before = physics_fingerprint()
        workflow_before = workflow_fingerprint()
        module.write_bytes(original + b"\n# fingerprint probe\n")
        self.assertEqual(physics_before, physics_fingerprint())
        self.assertNotEqual(workflow_before, workflow_fingerprint())

    def test_workflow_hash_covers_files_added_later(self) -> None:
        extra = ROOT / "_test_added_module.py"
        self.addCleanup(lambda: extra.unlink(missing_ok=True))
        before = workflow_fingerprint()
        extra.write_text("# not part of the original folder\n", encoding="utf-8")
        self.assertNotEqual(before, workflow_fingerprint())


class ReceiptTests(unittest.TestCase):
    """A VQE must not be able to run against an unproved Hamiltonian."""

    def _cleanup(self, *paths: Path) -> None:
        for path in paths:
            path.unlink(missing_ok=True)

    def _write_receipt(self, cache: Path, spec, penalties: dict) -> Path:
        receipt_path = validation_receipt_path(cache)
        write_json_atomic(
            receipt_path,
            {
                "schema": RECEIPT_SCHEMA,
                "cache_sha256": file_fingerprint(cache),
                "spec_sha256": spec_fingerprint(spec),
                "physics_sha256": physics_fingerprint(),
                "workflow_sha256": workflow_fingerprint(),
                "sector_ground_energy_hartree": -841.9,
                "validated_penalties": penalties,
            },
        )
        return receipt_path

    def _fixture(self, name: str) -> Path:
        cache = ROOT / name
        self.addCleanup(self._cleanup, cache, validation_receipt_path(cache))
        cache.write_text("stable-cache\n", encoding="utf-8")
        return cache

    def test_receipt_carries_several_penalty_settings(self) -> None:
        cache = self._fixture("_test_receipt_cache.json")
        self._write_receipt(
            cache,
            IMIPRAMINE,
            {
                penalty_key(1.0, 0.0): {"checks": "all_passed", "sector_ground_energy_hartree": -841.9},
                penalty_key(4.0, 0.0): {"checks": "all_passed", "sector_ground_energy_hartree": -841.9},
            },
        )
        for penalty in (1.0, 4.0):
            entry = require_validation_receipt(cache, IMIPRAMINE, penalty, 0.0)
            self.assertEqual(entry["checks"], "all_passed")
            self.assertEqual(entry["penalty_key"], penalty_key(penalty, 0.0))
            self.assertEqual(entry["physics_sha256"], physics_fingerprint())

    def test_unvalidated_penalty_is_refused(self) -> None:
        cache = self._fixture("_test_penalty_cache.json")
        self._write_receipt(
            cache, IMIPRAMINE, {penalty_key(1.0, 0.0): {"checks": "all_passed"}}
        )
        with self.assertRaisesRegex(RuntimeError, "No validation for"):
            require_validation_receipt(cache, IMIPRAMINE, 16.0, 0.0)
        with self.assertRaisesRegex(RuntimeError, "No validation for"):
            require_validation_receipt(cache, IMIPRAMINE, 1.0, 1.0)

    def test_mutated_cache_is_refused(self) -> None:
        cache = self._fixture("_test_mutated_cache.json")
        self._write_receipt(
            cache, IMIPRAMINE, {penalty_key(1.0, 0.0): {"checks": "all_passed"}}
        )
        cache.write_text("mutated-cache\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cache_sha256"):
            require_validation_receipt(cache, IMIPRAMINE, 1.0, 0.0)

    def test_receipt_for_another_active_space_is_refused(self) -> None:
        cache = self._fixture("_test_space_cache.json")
        self._write_receipt(
            cache,
            replace(IMIPRAMINE, n_active_orbitals=7),
            {penalty_key(1.0, 0.0): {"checks": "all_passed"}},
        )
        with self.assertRaisesRegex(RuntimeError, "spec_sha256"):
            require_validation_receipt(cache, IMIPRAMINE, 1.0, 0.0)

    def test_old_receipt_schema_is_refused(self) -> None:
        cache = self._fixture("_test_schema_cache.json")
        write_json_atomic(
            validation_receipt_path(cache),
            {
                "cache_sha256": file_fingerprint(cache),
                "spec_sha256": spec_fingerprint(IMIPRAMINE),
                "workflow_sha256": workflow_fingerprint(),
                "number_penalty_hartree": 1.0,
                "checks": "all_passed",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "schema"):
            require_validation_receipt(cache, IMIPRAMINE, 1.0, 0.0)

    def test_missing_receipt_is_refused(self) -> None:
        cache = self._fixture("_test_missing_receipt_cache.json")
        with self.assertRaises(FileNotFoundError):
            require_validation_receipt(cache, IMIPRAMINE, 1.0, 0.0)


class ArtifactSafetyTests(unittest.TestCase):
    def test_atomic_json_write_replaces_complete_document(self) -> None:
        path = ROOT / "_test_atomic_artifact.json"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        write_json_atomic(path, {"generation": 1})
        write_json_atomic(path, {"generation": 2, "complete": True})
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            {"generation": 2, "complete": True},
        )
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_receipt_name_has_one_json_suffix(self) -> None:
        import run

        cache = Path(run.CACHE)
        self.assertEqual(cache.name, "hamiltonian_cas6e6o.json")
        self.assertEqual(
            validation_receipt_path(cache).name, "hamiltonian_cas6e6o.validated.json"
        )

    @unittest.skipUnless(HAS_QISKIT, "needs qiskit")
    def test_scan_points_get_separate_result_files(self) -> None:
        import run
        from qiskit_runtime import _default_result_path

        names = {
            _default_result_path(Path(run.RESULTS), layers, method).name
            for layers in (4, 6, 8)
            for method in ("matrix_product_state", "statevector")
        }
        self.assertEqual(len(names), 6)
        self.assertIn("vqe_L4_mps.json", names)
        self.assertIn("vqe_L8_sv.json", names)

    def test_adapt_results_cannot_collide_with_vqe_results(self) -> None:
        from adapt_runtime import POOLS

        names = {f"adapt_{pool}.json" for pool in POOLS}
        self.assertEqual(len(names), len(POOLS))
        self.assertFalse(any(name.startswith("vqe_") for name in names))


@unittest.skipUnless(HAS_QISKIT and HAS_OPENFERMION, "needs qiskit and openfermion")
class OperatorConversionTests(unittest.TestCase):
    """Qubit ordering is the one place a silent, plausible-looking error hides."""

    def test_single_qubit_z_matches_openfermion_indexing(self) -> None:
        import numpy as np
        from openfermion import QubitOperator, get_sparse_operator
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import to_sparse_pauli_op

        try:
            from openfermion import jw_configuration_state
        except ImportError:
            from openfermion.linalg import jw_configuration_state

        for qubit in range(4):
            operator = QubitOperator(f"Z{qubit}")
            converted = to_sparse_pauli_op(operator, 4)
            for occupied in ([], [qubit], [(qubit + 1) % 4]):
                # OpenFermion's own reference value for this determinant.
                vector = jw_configuration_state(sorted(occupied), 4)
                expected = float(
                    np.real(
                        np.vdot(vector, get_sparse_operator(operator, n_qubits=4) @ vector)
                    )
                )
                # The same determinant in Qiskit, built by index, not by label.
                state = Statevector.from_int(0, dims=(2,) * 4)
                if occupied:
                    rotated = to_sparse_pauli_op(
                        conjugate_with_x(operator, occupied), 4
                    )
                else:
                    rotated = converted
                actual = float(np.real(state.expectation_value(rotated)))
                self.assertAlmostEqual(expected, actual, places=12)

    def test_hartree_fock_frame_maps_vacuum_to_the_reference_determinant(self) -> None:
        import numpy as np
        from openfermion import get_sparse_operator
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import to_sparse_pauli_op

        try:
            from openfermion import jw_configuration_state
        except ImportError:
            from openfermion.linalg import jw_configuration_state

        n_qubits = 2 * IMIPRAMINE.n_active_orbitals
        n_electrons = IMIPRAMINE.n_active_electrons
        number = electron_number_operator(n_qubits)
        deviation = number_deviation_operator(n_qubits, n_electrons)
        vacuum = Statevector.from_int(0, dims=(2,) * n_qubits)

        for operator, expected in ((number, float(n_electrons)), (deviation, 0.0)):
            framed = to_sparse_pauli_op(
                to_hartree_fock_frame(operator, n_electrons), n_qubits
            )
            self.assertAlmostEqual(
                float(np.real(vacuum.expectation_value(framed))), expected, places=12
            )

        # And the frame really is the Hartree-Fock determinant OpenFermion means.
        occupied = hartree_fock_occupation(n_electrons)
        vector = jw_configuration_state(occupied, n_qubits)
        reference = float(
            np.real(
                np.vdot(vector, get_sparse_operator(number, n_qubits=n_qubits) @ vector)
            )
        )
        self.assertAlmostEqual(reference, float(n_electrons), places=12)

    def test_non_hermitian_operator_is_rejected(self) -> None:
        from openfermion import QubitOperator

        from qiskit_runtime import to_sparse_pauli_op

        operator = QubitOperator("X0", 1.0j)
        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            to_sparse_pauli_op(operator, 2)

    def test_term_outside_the_register_is_rejected(self) -> None:
        from openfermion import QubitOperator

        from qiskit_runtime import to_sparse_pauli_op

        with self.assertRaisesRegex(ValueError, "outside"):
            to_sparse_pauli_op(QubitOperator("Z12"), 12)


@unittest.skipUnless(HAS_OPENFERMION, "needs openfermion")
class AdaptPoolTests(unittest.TestCase):
    """The ADAPT pool is where the symmetry claim is made or lost.

    Everything the ADAPT path says about accuracy rests on one structural
    property: no operator in the pool can move the state out of the six-electron
    sector. That is asserted by the pool's *construction*, which means
    construction is where it has to be tested -- by the time a wrong operator
    shows up in an energy it is indistinguishable from a good one.

    The corresponding S^2 claim does **not** hold for general doubles and is not
    tested here, because it is not true; see `selftest.py`, which measures the
    commutators. What is true, and is tested, is that every entry balances
    creation against annihilation indices and conserves Sz.
    """

    N_QUBITS = 12
    N_ELECTRONS = 6

    @classmethod
    def setUpClass(cls) -> None:
        from adapt_runtime import build_pool

        cls.pools = {
            kind: build_pool(cls.N_QUBITS, cls.N_ELECTRONS, kind)
            for kind in ("pair", "uccsd", "uccgsd")
        }

    def _split(self, kind: str) -> tuple[int, int]:
        singles = sum(1 for entry in self.pools[kind] if entry["rank"] == 1)
        return singles, len(self.pools[kind]) - singles

    def test_pool_sizes_are_what_the_definitions_imply(self) -> None:
        """Regression guard: an enumeration bug shows up here as a count, not a wrong energy.

        These are not arbitrary. `pair` is the 15 spatial-orbital pair moves of
        k-UpCCGSD with no singles at all. `uccsd` is occupied -> virtual only,
        relative to the Hartree-Fock determinant. `uccgsd` is generalized: every
        excitation between orbital pairs, which is why it is an order of
        magnitude larger and why it is the paper's choice at twelve qubits.
        """
        self.assertEqual(self._split("pair"), (0, 15))
        self.assertEqual(self._split("uccsd"), (9, 54))
        self.assertEqual(self._split("uccgsd"), (15, 420))

    def test_every_operator_conserves_particle_number_and_spin_projection(self) -> None:
        for kind, entries in self.pools.items():
            for entry in entries:
                with self.subTest(pool=kind, operator=entry["label"]):
                    creations = entry["creations"]
                    annihilations = entry["annihilations"]
                    self.assertEqual(len(creations), len(annihilations))
                    self.assertEqual(len(creations), entry["rank"])
                    self.assertEqual(
                        sum(index % 2 for index in creations),
                        sum(index % 2 for index in annihilations),
                    )
                    self.assertTrue(
                        all(0 <= index < self.N_QUBITS for index in creations + annihilations)
                    )

    def test_generators_are_anti_hermitian(self) -> None:
        """exp(theta A) is unitary only for anti-Hermitian A, and JW preserves that.

        After Jordan-Wigner every coefficient must be purely imaginary. A real
        component would mean the generator picked up a Hermitian part, and
        exp(theta A) would stop being norm preserving.
        """
        for kind, entries in self.pools.items():
            for entry in entries:
                with self.subTest(pool=kind, operator=entry["label"]):
                    self.assertTrue(entry["qubit"].terms)
                    for coefficient in entry["qubit"].terms.values():
                        self.assertLess(abs(complex(coefficient).real), 1e-12)

    def test_uccsd_moves_electrons_out_of_the_occupied_set(self) -> None:
        from hamiltonian import hartree_fock_occupation

        occupied = set(hartree_fock_occupation(self.N_ELECTRONS))
        for entry in self.pools["uccsd"]:
            with self.subTest(operator=entry["label"]):
                self.assertTrue(all(i in occupied for i in entry["annihilations"]))
                self.assertTrue(all(i not in occupied for i in entry["creations"]))

    def test_pair_pool_moves_both_electrons_of_a_spatial_orbital(self) -> None:
        for entry in self.pools["pair"]:
            with self.subTest(operator=entry["label"]):
                for indices in (entry["creations"], entry["annihilations"]):
                    self.assertEqual(len(indices), 2)
                    self.assertEqual(indices[0] // 2, indices[1] // 2)
                    self.assertNotEqual(indices[0] % 2, indices[1] % 2)

    def test_no_pool_repeats_an_operator(self) -> None:
        """T(a<-b) and T(b<-a) differ only by a sign the parameter absorbs.

        Keeping both would let ADAPT spend an iteration re-selecting an
        excitation it already has, so the pool builder deduplicates on a
        direction-independent key. This checks the operators themselves rather
        than that key, so a broken key cannot make the test pass.
        """
        for kind, entries in self.pools.items():
            keys = set()
            for entry in entries:
                terms = sorted(
                    (tuple(sorted(term)), complex(value).imag)
                    for term, value in entry["qubit"].terms.items()
                )
                sign = 1.0 if terms[0][1] >= 0 else -1.0
                keys.add(tuple((term, round(sign * value, 10)) for term, value in terms))
            with self.subTest(pool=kind):
                self.assertEqual(len(keys), len(entries))

    def test_unknown_pool_is_refused(self) -> None:
        from adapt_runtime import build_pool

        with self.assertRaisesRegex(ValueError, "Unknown pool"):
            build_pool(self.N_QUBITS, self.N_ELECTRONS, "uccsdt")


@unittest.skipUnless(HAS_QISKIT, "needs qiskit")
class AnsatzTests(unittest.TestCase):
    def test_theta_zero_is_the_computational_vacuum(self) -> None:
        import numpy as np
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import build_ansatz

        ansatz = build_ansatz(12, 3)
        state = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters)))
        self.assertAlmostEqual(abs(complex(state.data[0])), 1.0, places=12)

    def test_parameter_shift_preconditions_hold_for_the_default_ansatz(self) -> None:
        from qiskit_runtime import build_ansatz, parameter_shift_is_exact

        for layers in (4, 6):
            with self.subTest(layers=layers):
                ansatz = build_ansatz(12, layers)
                # real_amplitudes: one RY per qubit per layer, plus the final layer.
                self.assertEqual(ansatz.num_parameters, 12 * (layers + 1))
                self.assertTrue(parameter_shift_is_exact(ansatz))

    def test_layers_must_be_positive(self) -> None:
        from qiskit_runtime import build_ansatz

        with self.assertRaises(ValueError):
            build_ansatz(12, 0)


if __name__ == "__main__":
    unittest.main()
