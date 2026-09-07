"""Dependency-light tests: specification, fingerprints, artifacts, conversion.

The first classes need only NumPy. The conversion and ansatz tests need Qiskit,
qiskit-nature and OpenFermion but *not* PySCF -- which is the point: on Windows
the operator-mapping and ansatz code can still be tested even though the
molecule itself cannot be built there.
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

from chemistry import (  # noqa: E402
    closed_shell_determinant_energy,
    conjugate_with_x,
    electron_number_operator,
    file_fingerprint,
    hartree_fock_occupation,
    number_deviation_operator,
    require_validation_receipt,
    spec_fingerprint,
    to_hartree_fock_frame,
    validation_receipt_path,
    workflow_fingerprint,
    write_json_atomic,
)
from system import ACRYLAMIDE_THIOLATE, ACTIVE_SPACES, make_spec  # noqa: E402


def _installed(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


HAS_QISKIT = _installed("qiskit", "qiskit_aer")
HAS_NATURE = _installed("qiskit_nature")
HAS_OPENFERMION = _installed("openfermion")


class MoleculeSpecificationTests(unittest.TestCase):
    def test_register_and_geometry_invariants(self) -> None:
        spec = ACRYLAMIDE_THIOLATE
        self.assertEqual(2 * spec.n_active_orbitals, 8)
        self.assertEqual(spec.n_active_electrons, 4)
        self.assertEqual(spec.charge, -1)
        self.assertEqual(spec.spin, 0)
        self.assertEqual(len(spec.atoms), 15)
        beta_carbon = spec.atoms[4][1]
        sulfur = spec.atoms[10][1]
        self.assertAlmostEqual(math.dist(beta_carbon, sulfur), 2.35, places=3)

    def test_far_geometry_separates_only_the_thiolate(self) -> None:
        """The barrier is only meaningful if both fragments stay rigid."""
        ts = make_spec(geometry="ts")
        far = make_spec(geometry="far")
        self.assertEqual(len(ts.atoms), len(far.atoms))
        # Acrylamide (atoms 0-9) is untouched.
        for near_atom, far_atom in zip(ts.atoms[:10], far.atoms[:10]):
            self.assertEqual(near_atom, far_atom)
        # The thiolate has moved rigidly: every internal distance is preserved.
        for i in range(10, 15):
            for j in range(i + 1, 15):
                self.assertAlmostEqual(
                    math.dist(ts.atoms[i][1], ts.atoms[j][1]),
                    math.dist(far.atoms[i][1], far.atoms[j][1]),
                    places=9,
                )
        self.assertAlmostEqual(
            math.dist(far.atoms[4][1], far.atoms[10][1]), 8.00, places=3
        )

    def test_active_space_ladder_maps_to_expected_qubit_counts(self) -> None:
        for label, (electrons, orbitals) in ACTIVE_SPACES.items():
            spec = make_spec(active_space=label)
            self.assertEqual(2 * spec.n_active_orbitals, int(label.removesuffix("q")))
            self.assertEqual(spec.n_active_electrons, electrons)
            self.assertEqual(spec.n_active_orbitals, orbitals)

    def test_scientific_choices_change_the_specification_hash(self) -> None:
        """Solvation and orbital choice must invalidate a cache, not be silent."""
        base = make_spec()
        self.assertNotEqual(spec_fingerprint(base), spec_fingerprint(make_spec(geometry="far")))
        self.assertNotEqual(
            spec_fingerprint(base), spec_fingerprint(make_spec(solvent_epsilon=None))
        )
        self.assertNotEqual(
            spec_fingerprint(base),
            spec_fingerprint(make_spec(orbital_selection="canonical_hf_frontier")),
        )
        self.assertNotEqual(spec_fingerprint(base), spec_fingerprint(replace(base, charge=0)))

    def test_workflow_hash_covers_the_qiskit_runtime(self) -> None:
        runtime = ROOT / "qiskit_runtime.py"
        original = runtime.read_bytes()
        before = workflow_fingerprint()
        try:
            runtime.write_bytes(original + b"\n# fingerprint probe\n")
            self.assertNotEqual(before, workflow_fingerprint())
        finally:
            runtime.write_bytes(original)
        self.assertEqual(before, workflow_fingerprint())


class DeterminantEnergyTests(unittest.TestCase):
    def test_matches_a_hand_computed_two_orbital_case(self) -> None:
        import numpy as np

        one_body = np.array([[-1.5, 0.2], [0.2, -0.4]])
        two_body = np.zeros((2, 2, 2, 2))
        two_body[0, 0, 0, 0] = 0.7
        # Two electrons in orbital 0: 2*h00 + 2*(00|00) - (00|00) = 2*h00 + (00|00).
        self.assertAlmostEqual(
            closed_shell_determinant_energy(one_body, two_body, 1.25, 2),
            1.25 + 2 * (-1.5) + 0.7,
            places=12,
        )

    def test_rejects_an_odd_electron_count(self) -> None:
        import numpy as np

        with self.assertRaises(ValueError):
            closed_shell_determinant_energy(np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 0.0, 3)


class ArtifactSafetyTests(unittest.TestCase):
    def _cleanup(self, *paths: Path) -> None:
        for path in paths:
            path.unlink(missing_ok=True)

    def test_atomic_json_write_replaces_complete_document(self) -> None:
        path = ROOT / "_test_atomic_artifact.json"
        self.addCleanup(self._cleanup, path)
        write_json_atomic(path, {"generation": 1})
        write_json_atomic(path, {"generation": 2, "complete": True})
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")), {"generation": 2, "complete": True}
        )
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_validation_receipt_is_bound_to_cache_and_settings(self) -> None:
        cache = ROOT / "_test_receipt_cache.json"
        receipt_path = validation_receipt_path(cache)
        self.addCleanup(self._cleanup, cache, receipt_path)
        cache.write_text("stable-cache\n", encoding="utf-8")
        write_json_atomic(receipt_path, {
            "cache_sha256": file_fingerprint(cache),
            "spec_sha256": spec_fingerprint(ACRYLAMIDE_THIOLATE),
            "workflow_sha256": workflow_fingerprint(),
            "number_penalty_hartree": 0.0,
            "spin_penalty_hartree": 0.0,
            "checks": "all_passed",
        })
        loaded = require_validation_receipt(cache, ACRYLAMIDE_THIOLATE, 0.0, 0.0)
        self.assertEqual(loaded["checks"], "all_passed")

        cache.write_text("mutated-cache\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cache_sha256"):
            require_validation_receipt(cache, ACRYLAMIDE_THIOLATE, 0.0, 0.0)


@unittest.skipUnless(HAS_QISKIT and HAS_OPENFERMION, "needs qiskit and openfermion")
class OrderingTests(unittest.TestCase):
    """Spin-orbital ordering is where a silent, plausible-looking error hides."""

    def test_interleaved_to_blocked_is_a_permutation(self) -> None:
        from qiskit_runtime import interleaved_to_blocked

        for n_qubits in (2, 4, 8, 12, 16):
            mapping = interleaved_to_blocked(n_qubits)
            self.assertEqual(sorted(mapping), list(range(n_qubits)))
            half = n_qubits // 2
            for p in range(half):
                self.assertEqual(mapping[2 * p], p)            # alpha_p
                self.assertEqual(mapping[2 * p + 1], half + p)  # beta_p

    def test_blocked_map_sends_the_reference_determinant_where_nature_puts_it(self) -> None:
        """Interleaved {0,1,2,3} must land on the qubits HartreeFock excites."""
        from qiskit_runtime import interleaved_to_blocked

        mapping = interleaved_to_blocked(8)
        occupied = hartree_fock_occupation(4)
        self.assertEqual(sorted(mapping[q] for q in occupied), [0, 1, 4, 5])

    def test_rejects_a_non_permutation_map(self) -> None:
        from openfermion import QubitOperator

        from qiskit_runtime import to_sparse_pauli_op

        with self.assertRaisesRegex(ValueError, "permutation"):
            to_sparse_pauli_op(QubitOperator("Z0"), 4, [0, 0, 1, 2])

    def test_hartree_fock_frame_maps_vacuum_to_the_reference_determinant(self) -> None:
        import numpy as np
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import to_sparse_pauli_op

        n_qubits, n_electrons = 8, 4
        vacuum = Statevector.from_int(0, dims=(2,) * n_qubits)
        for operator, expected in (
            (electron_number_operator(n_qubits), float(n_electrons)),
            (number_deviation_operator(n_qubits, n_electrons), 0.0),
        ):
            framed = to_sparse_pauli_op(to_hartree_fock_frame(operator, n_electrons), n_qubits)
            self.assertAlmostEqual(
                float(np.real(vacuum.expectation_value(framed))), expected, places=12
            )

    def test_non_hermitian_operator_is_rejected(self) -> None:
        from openfermion import QubitOperator

        from qiskit_runtime import to_sparse_pauli_op

        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            to_sparse_pauli_op(QubitOperator("X0", 1.0j), 2)


@unittest.skipUnless(HAS_QISKIT and HAS_NATURE, "needs qiskit and qiskit-nature")
class AnsatzTests(unittest.TestCase):
    def test_uccsd_theta_zero_is_the_hartree_fock_determinant(self) -> None:
        import numpy as np
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import build_ansatz

        ansatz = build_ansatz(8, 4, "uccsd")
        state = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters)))
        amplitudes = np.abs(state.data)
        dominant = int(np.argmax(amplitudes))
        self.assertAlmostEqual(float(amplitudes[dominant]), 1.0, places=12)
        # Blocked ordering: alpha 0,1 -> qubits 0,1; beta 0,1 -> qubits 4,5.
        self.assertEqual(sorted(i for i in range(8) if dominant >> i & 1), [0, 1, 4, 5])

    def test_parameter_shift_is_rejected_for_uccsd_and_accepted_for_hea(self) -> None:
        from qiskit_runtime import build_ansatz, parameter_shift_is_exact

        self.assertFalse(parameter_shift_is_exact(build_ansatz(8, 4, "uccsd").decompose(reps=5)))
        hea = build_ansatz(8, 4, "hea", layers=4)
        self.assertEqual(hea.num_parameters, 40)
        self.assertTrue(parameter_shift_is_exact(hea))

    def test_uccsd_is_smaller_than_the_hardware_efficient_ansatz(self) -> None:
        from qiskit_runtime import build_ansatz

        self.assertEqual(build_ansatz(8, 4, "uccsd").num_parameters, 26)
        self.assertLess(
            build_ansatz(8, 4, "uccsd").num_parameters,
            build_ansatz(8, 4, "hea", layers=4).num_parameters,
        )


if __name__ == "__main__":
    unittest.main()
