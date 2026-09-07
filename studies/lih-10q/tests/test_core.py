"""Dependency-light tests: specification, fingerprints, artifacts, conversion.

The first four classes need nothing but NumPy, so they run anywhere. The
conversion tests need Qiskit and OpenFermion but *not* PySCF, which is the whole
point: on Windows the operator-mapping code can still be tested even though the
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
from system import DEFAULT_SPACE, LIH_FROZEN_CORE, LIH_FULL_SPACE, SPACES  # noqa: E402


def _installed(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


HAS_QISKIT = _installed("qiskit", "qiskit_aer")
HAS_OPENFERMION = _installed("openfermion")


class MoleculeSpecificationTests(unittest.TestCase):
    def test_active_space_invariants(self) -> None:
        # STO-3G LiH: Li 1s/2s/2p(3) + H 1s = 6 spatial orbitals, 4 electrons.
        self.assertEqual(2 * LIH_FULL_SPACE.n_active_orbitals, 12)
        self.assertEqual(LIH_FULL_SPACE.n_active_electrons, 4)
        # Freezing Li 1s removes one spatial orbital and one electron pair.
        self.assertEqual(2 * LIH_FROZEN_CORE.n_active_orbitals, 10)
        self.assertEqual(LIH_FROZEN_CORE.n_active_electrons, 2)
        for spec in (LIH_FROZEN_CORE, LIH_FULL_SPACE):
            with self.subTest(spec=spec.name):
                self.assertEqual(spec.charge, 0)
                self.assertEqual(spec.spin, 0)
                self.assertEqual(spec.basis, "sto-3g")
                self.assertEqual(len(spec.atoms), 2)
                self.assertEqual(spec.n_active_electrons % 2, 0)
                self.assertLessEqual(
                    spec.n_active_electrons, 2 * spec.n_active_orbitals
                )

    def test_bond_length_is_the_experimental_equilibrium(self) -> None:
        lithium, hydrogen = (coords for _, coords in LIH_FROZEN_CORE.atoms)
        self.assertAlmostEqual(math.dist(lithium, hydrogen), 1.595, places=6)

    def test_the_two_spaces_are_distinguishable(self) -> None:
        """A cache built for one space must never validate against the other."""
        self.assertNotEqual(
            spec_fingerprint(LIH_FROZEN_CORE), spec_fingerprint(LIH_FULL_SPACE)
        )

    def test_specification_hash_changes_with_scientific_input(self) -> None:
        original = spec_fingerprint(LIH_FROZEN_CORE)
        changed = spec_fingerprint(replace(LIH_FROZEN_CORE, charge=1))
        self.assertNotEqual(original, changed)

    def test_workflow_hash_is_valid_sha256(self) -> None:
        digest = workflow_fingerprint()
        self.assertEqual(len(digest), 64)
        int(digest, 16)

    def test_workflow_hash_covers_the_qiskit_runtime(self) -> None:
        """A change to the simulator layer must invalidate a validation receipt."""
        runtime = ROOT / "qiskit_runtime.py"
        original = runtime.read_bytes()
        before = workflow_fingerprint()
        try:
            runtime.write_bytes(original + b"\n# fingerprint probe\n")
            self.assertNotEqual(before, workflow_fingerprint())
        finally:
            runtime.write_bytes(original)
        self.assertEqual(before, workflow_fingerprint())


class SpaceSelectionTests(unittest.TestCase):
    """`--space` is stripped before the subcommand parses its own arguments."""

    def test_default_and_registry(self) -> None:
        from run import _extract_space

        self.assertIn(DEFAULT_SPACE, SPACES)
        self.assertEqual(_extract_space([]), (DEFAULT_SPACE, []))

    def test_both_spellings_are_removed_from_argv(self) -> None:
        from run import _extract_space

        for argv in (
            ["--space", "full", "--layers", "6"],
            ["--space=full", "--layers", "6"],
            ["--layers", "6", "--space", "full"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(_extract_space(argv), ("full", ["--layers", "6"]))

    def test_space_may_precede_or_follow_the_command(self) -> None:
        from run import _extract_space

        self.assertEqual(
            _extract_space(["--space", "full", "vqe", "--layers", "6"]),
            ("full", ["vqe", "--layers", "6"]),
        )
        self.assertEqual(
            _extract_space(["vqe", "--space", "full", "--layers", "6"]),
            ("full", ["vqe", "--layers", "6"]),
        )

    def test_unknown_space_is_rejected(self) -> None:
        from run import _extract_space

        with self.assertRaises(SystemExit):
            _extract_space(["--space", "cas22"])
        with self.assertRaises(SystemExit):
            _extract_space(["--space"])


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
            json.loads(path.read_text(encoding="utf-8")),
            {"generation": 2, "complete": True},
        )
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_validation_receipt_is_bound_to_cache_and_settings(self) -> None:
        cache = ROOT / "_test_receipt_cache.json"
        receipt_path = validation_receipt_path(cache)
        self.addCleanup(self._cleanup, cache, receipt_path)
        cache.write_text("stable-cache\n", encoding="utf-8")
        receipt = {
            "cache_sha256": file_fingerprint(cache),
            "spec_sha256": spec_fingerprint(LIH_FROZEN_CORE),
            "workflow_sha256": workflow_fingerprint(),
            "number_penalty_hartree": 1.0,
            "spin_penalty_hartree": 0.0,
            "checks": "all_passed",
        }
        write_json_atomic(receipt_path, receipt)
        loaded = require_validation_receipt(cache, LIH_FROZEN_CORE, 1.0, 0.0)
        self.assertEqual(loaded["checks"], "all_passed")

        cache.write_text("mutated-cache\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cache_sha256"):
            require_validation_receipt(cache, LIH_FROZEN_CORE, 1.0, 0.0)

    def test_receipt_from_the_other_space_is_rejected(self) -> None:
        cache = ROOT / "_test_space_cache.json"
        receipt_path = validation_receipt_path(cache)
        self.addCleanup(self._cleanup, cache, receipt_path)
        cache.write_text("stable-cache\n", encoding="utf-8")
        write_json_atomic(
            receipt_path,
            {
                "cache_sha256": file_fingerprint(cache),
                "spec_sha256": spec_fingerprint(LIH_FULL_SPACE),
                "workflow_sha256": workflow_fingerprint(),
                "number_penalty_hartree": 1.0,
                "spin_penalty_hartree": 0.0,
                "checks": "all_passed",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "spec_sha256"):
            require_validation_receipt(cache, LIH_FROZEN_CORE, 1.0, 0.0)


@unittest.skipUnless(HAS_QISKIT and HAS_OPENFERMION, "needs qiskit and openfermion")
class OperatorConversionTests(unittest.TestCase):
    """Qubit ordering is the one place a silent, plausible-looking error hides."""

    def test_single_qubit_z_matches_openfermion_indexing(self) -> None:
        import numpy as np
        from openfermion import QubitOperator, get_sparse_operator
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import to_sparse_pauli_op

        for qubit in range(4):
            operator = QubitOperator(f"Z{qubit}")
            converted = to_sparse_pauli_op(operator, 4)
            for occupied in ([], [qubit], [(qubit + 1) % 4]):
                # OpenFermion's own reference value for this determinant.
                try:
                    from openfermion import jw_configuration_state
                except ImportError:
                    from openfermion.linalg import jw_configuration_state
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

        for spec in (LIH_FROZEN_CORE, LIH_FULL_SPACE):
            n_qubits = 2 * spec.n_active_orbitals
            n_electrons = spec.n_active_electrons
            with self.subTest(qubits=n_qubits):
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
                        np.vdot(
                            vector, get_sparse_operator(number, n_qubits=n_qubits) @ vector
                        )
                    )
                )
                self.assertAlmostEqual(reference, float(n_electrons), places=12)

    def test_non_hermitian_operator_is_rejected(self) -> None:
        from openfermion import QubitOperator

        from qiskit_runtime import to_sparse_pauli_op

        operator = QubitOperator("X0", 1.0j)
        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            to_sparse_pauli_op(operator, 2)


@unittest.skipUnless(HAS_QISKIT, "needs qiskit")
class AnsatzTests(unittest.TestCase):
    def test_theta_zero_is_the_computational_vacuum(self) -> None:
        import numpy as np
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import build_ansatz

        ansatz = build_ansatz(10, 3)
        state = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters)))
        self.assertAlmostEqual(abs(complex(state.data[0])), 1.0, places=12)

    def test_parameter_shift_preconditions_hold_for_the_default_ansatz(self) -> None:
        from qiskit_runtime import build_ansatz, parameter_shift_is_exact

        for n_qubits in (10, 12):
            with self.subTest(qubits=n_qubits):
                ansatz = build_ansatz(n_qubits, 4)
                # real_amplitudes: one RY per qubit per layer, plus the final layer.
                self.assertEqual(ansatz.num_parameters, n_qubits * 5)
                self.assertTrue(parameter_shift_is_exact(ansatz))

    def test_unknown_ansatz_is_rejected(self) -> None:
        from qiskit_runtime import build_ansatz

        with self.assertRaisesRegex(ValueError, "Unsupported ansatz"):
            build_ansatz(10, 4, "hartree-fock")

    def test_uccsd_needs_an_electron_count(self) -> None:
        from qiskit_runtime import build_ansatz

        with self.assertRaisesRegex(ValueError, "active electron count"):
            build_ansatz(10, 4, "uccsd")


@unittest.skipUnless(HAS_QISKIT, "needs qiskit and qiskit-nature")
class UccsdOrderingTests(unittest.TestCase):
    """qiskit-nature blocks spin orbitals; OpenFermion interleaves them.

    Everything here guards that single permutation. It is the kind of mistake
    that produces a circuit which still looks right at theta = 0 and still
    conserves particle number, but explores the wrong excitations.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _installed("qiskit_nature"):
            raise unittest.SkipTest("needs qiskit-nature")

    def test_parameter_counts_match_the_active_spaces(self) -> None:
        from qiskit_runtime import build_ansatz

        # Singles and doubles for (2e,5o) and (4e,6o) respectively.
        self.assertEqual(build_ansatz(10, 1, "uccsd", 2).num_parameters, 24)
        self.assertEqual(build_ansatz(12, 1, "uccsd", 4).num_parameters, 92)

    def test_theta_zero_is_the_computational_vacuum(self) -> None:
        """The two X ladders must cancel, leaving the frame's reference state."""
        import numpy as np
        from qiskit.quantum_info import Statevector

        from qiskit_runtime import build_ansatz

        for n_qubits, n_electrons in ((10, 2), (12, 4)):
            with self.subTest(qubits=n_qubits):
                ansatz = build_ansatz(n_qubits, 1, "uccsd", n_electrons)
                state = Statevector(
                    ansatz.assign_parameters(np.zeros(ansatz.num_parameters))
                )
                self.assertAlmostEqual(abs(complex(state.data[0])), 1.0, places=12)

    def test_permutation_maps_the_blocked_reference_onto_the_interleaved_one(
        self,
    ) -> None:
        """qiskit-nature's own HF state, permuted, must be our occupation."""
        from qiskit.quantum_info import Statevector
        from qiskit_nature.second_q.circuit.library import HartreeFock
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        from qiskit import QuantumCircuit

        for n_qubits, n_electrons in ((10, 2), (12, 4)):
            with self.subTest(qubits=n_qubits):
                n_spatial = n_qubits // 2
                pairs = n_electrons // 2
                blocked = HartreeFock(n_spatial, (pairs, pairs), JordanWignerMapper())
                order = [
                    2 * (j % n_spatial) + (j // n_spatial) for j in range(n_qubits)
                ]
                permuted = QuantumCircuit(n_qubits)
                permuted.compose(blocked, qubits=order, inplace=True)

                amplitudes = Statevector(permuted).data
                occupied_index = int(max(range(len(amplitudes)), key=lambda i: abs(amplitudes[i])))
                # Qiskit statevector indices are little-endian: bit q of the
                # index is qubit q.
                occupied = sorted(
                    q for q in range(n_qubits) if occupied_index >> q & 1
                )
                self.assertEqual(occupied, hartree_fock_occupation(n_electrons))

    def test_uccsd_conserves_particle_number_at_random_angles(self) -> None:
        """Conserving N by construction is the reason to prefer this ansatz."""
        import numpy as np
        from qiskit.quantum_info import SparsePauliOp, Statevector

        from qiskit_runtime import build_ansatz, to_sparse_pauli_op

        n_qubits, n_electrons = 10, 2
        ansatz = build_ansatz(n_qubits, 1, "uccsd", n_electrons)
        number = to_sparse_pauli_op(
            to_hartree_fock_frame(
                electron_number_operator(n_qubits), n_electrons
            ),
            n_qubits,
        )
        rng = np.random.default_rng(11)
        angles = rng.uniform(-1.0, 1.0, ansatz.num_parameters)
        state = Statevector(ansatz.assign_parameters(angles))
        measured = float(np.real(state.expectation_value(SparsePauliOp(number))))
        self.assertAlmostEqual(measured, float(n_electrons), places=9)


if __name__ == "__main__":
    unittest.main()
