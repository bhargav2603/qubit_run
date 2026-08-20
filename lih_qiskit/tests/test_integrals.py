"""Tests for the built-in STO-3G integral engine.

The engine exists so `prepare` can run where PySCF cannot be installed, which
means it cannot be checked against PySCF on the machine that needs it most.
These tests pin it to published energies and to internal identities that a wrong
integral would break:

* H2/STO-3G at the Szabo & Ostlund geometry, where the RHF energy is a textbook
  number.
* LiH/STO-3G, which is the molecule this folder is about, and which exercises
  the Li 2p shell that H2 does not touch.
* The Hartree-Fock determinant of the *second-quantized* Hamiltonian must equal
  the RHF energy from the SCF, in both active spaces. That ties the CAS
  transformation, the frozen-core energy and the orbital coefficients together;
  an error in any one of them breaks it.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import integrals  # noqa: E402
from system import LIH_FROZEN_CORE, LIH_FULL_SPACE  # noqa: E402


HAS_OPENFERMION = importlib.util.find_spec("openfermion") is not None

# Szabo & Ostlund, "Modern Quantum Chemistry", H2/STO-3G at R = 1.4 bohr.
H2_GEOMETRY = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4 / integrals.BOHR_PER_ANGSTROM))]
H2_RHF_REFERENCE = -1.1167

LIH_GEOMETRY = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.595))]
LIH_RHF_REFERENCE = -7.8620
LIH_FCI_REFERENCE = -7.8824


class BasisTests(unittest.TestCase):
    def test_lih_has_six_contracted_functions(self):
        basis = integrals.build_basis(LIH_GEOMETRY, "sto-3g")
        # Li contributes 1s, 2s and three 2p; H contributes 1s.
        self.assertEqual(len(basis), 6)

    def test_every_contracted_function_is_normalized(self):
        basis = integrals.build_basis(LIH_GEOMETRY, "sto-3g")
        overlap = integrals.overlap_matrix(basis)
        np.testing.assert_allclose(np.diag(overlap), np.ones(6), atol=1e-12)

    def test_ao_slices_follow_input_order(self):
        slices = integrals.ao_slices_by_atom(LIH_GEOMETRY, "sto-3g")
        self.assertEqual(slices, [(0, 5), (5, 6)])

    def test_unsupported_basis_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            integrals.build_basis(LIH_GEOMETRY, "6-31g")

    def test_unsupported_element_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            integrals.build_basis([("C", (0.0, 0.0, 0.0))], "sto-3g")


class IntegralIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basis = integrals.build_basis(LIH_GEOMETRY, "sto-3g")
        cls.eri = integrals.eri_tensor(cls.basis)

    def test_overlap_is_symmetric_positive_definite(self):
        overlap = integrals.overlap_matrix(self.basis)
        np.testing.assert_allclose(overlap, overlap.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(overlap).min(), 0.0)

    def test_eri_has_eightfold_symmetry(self):
        eri = self.eri
        np.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-13)
        np.testing.assert_allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-13)
        np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-13)

    def test_coulomb_integrals_are_positive(self):
        # (ii|jj) is the repulsion of two non-negative densities.
        for i in range(len(self.basis)):
            for j in range(len(self.basis)):
                self.assertGreater(self.eri[i, i, j, j], 0.0)


class HartreeFockTests(unittest.TestCase):
    def test_h2_matches_the_textbook_energy(self):
        mean_field = integrals.run_rhf(H2_GEOMETRY, "sto-3g")
        self.assertAlmostEqual(mean_field.e_hf, H2_RHF_REFERENCE, places=4)

    def test_lih_matches_the_published_energy(self):
        mean_field = integrals.run_rhf(LIH_GEOMETRY, "sto-3g")
        self.assertAlmostEqual(mean_field.e_hf, LIH_RHF_REFERENCE, places=4)

    def test_lih_pi_orbitals_are_degenerate(self):
        # A linear molecule must give the 2p_x / 2p_y pair identical energies.
        mean_field = integrals.run_rhf(LIH_GEOMETRY, "sto-3g")
        energies = np.sort(mean_field.mo_energy)
        gaps = np.abs(np.diff(energies))
        self.assertLess(gaps.min(), 1e-10)

    def test_orbitals_are_orthonormal_in_the_overlap_metric(self):
        mean_field = integrals.run_rhf(LIH_GEOMETRY, "sto-3g")
        gram = mean_field.mo_coeff.T @ mean_field.overlap @ mean_field.mo_coeff
        np.testing.assert_allclose(gram, np.eye(mean_field.n_orbitals), atol=1e-11)

    def test_open_shell_is_refused(self):
        with self.assertRaises(ValueError):
            integrals.run_rhf(LIH_GEOMETRY, "sto-3g", spin=1)


class CasIntegralTests(unittest.TestCase):
    """The CAS partition must not move any energy around."""

    @classmethod
    def setUpClass(cls):
        cls.mean_field = integrals.run_rhf(LIH_GEOMETRY, "sto-3g")

    def test_full_space_core_energy_is_just_nuclear_repulsion(self):
        _, _, core_energy = integrals.cas_integrals(self.mean_field, 0, 6)
        self.assertAlmostEqual(core_energy, self.mean_field.e_nuc, places=12)

    def test_core_plus_active_exceeding_the_space_is_rejected(self):
        with self.assertRaises(ValueError):
            integrals.cas_integrals(self.mean_field, 2, 6)

    @unittest.skipUnless(HAS_OPENFERMION, "OpenFermion is not installed")
    def test_hartree_fock_determinant_reproduces_the_scf_energy(self):
        """The strongest available check without a second chemistry code.

        Runs the same route `chemistry.py` runs -- CAS integrals, chemist to
        physicist transpose, spinorb_from_spatial, Jordan-Wigner -- and reads the
        reference determinant straight off the diagonal. It can only match if the
        one-body effective integrals, the two-electron transformation and the
        frozen-core energy are each right.
        """
        from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
        from openfermion.chem.molecular_data import spinorb_from_spatial
        from openfermion.linalg import get_sparse_operator

        for spec, n_core in ((LIH_FULL_SPACE, 0), (LIH_FROZEN_CORE, 1)):
            with self.subTest(space=spec.name):
                one_body, two_body, core_energy = integrals.cas_integrals(
                    self.mean_field, n_core, spec.n_active_orbitals
                )
                physicist = np.asarray(
                    two_body.transpose(0, 2, 3, 1), dtype=float, order="C"
                )
                one_spin, two_spin = spinorb_from_spatial(one_body, physicist)
                hamiltonian = jordan_wigner(
                    get_fermion_operator(
                        InteractionOperator(core_energy, one_spin, 0.5 * two_spin)
                    )
                )
                n_qubits = 2 * spec.n_active_orbitals
                sparse = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
                index = 0
                for qubit in range(spec.n_active_electrons):
                    index |= 1 << (n_qubits - 1 - qubit)
                determinant_energy = float(np.real(sparse[index, index]))
                self.assertAlmostEqual(
                    determinant_energy, self.mean_field.e_hf, places=10
                )

    @unittest.skipUnless(HAS_OPENFERMION, "OpenFermion is not installed")
    def test_full_space_ground_state_matches_published_fci(self):
        from chemistry import build_qubit_hamiltonian

        result = build_qubit_hamiltonian(LIH_FULL_SPACE, integral_backend="native")
        self.assertAlmostEqual(result.reference_energy, LIH_FCI_REFERENCE, places=4)
        self.assertAlmostEqual(result.hartree_fock_energy, LIH_RHF_REFERENCE, places=4)

    @unittest.skipUnless(HAS_OPENFERMION, "OpenFermion is not installed")
    def test_freezing_the_core_costs_little_and_never_helps(self):
        """Freezing Li 1s can only raise the energy, and should barely do so."""
        from chemistry import build_qubit_hamiltonian

        full = build_qubit_hamiltonian(LIH_FULL_SPACE, integral_backend="native")
        frozen = build_qubit_hamiltonian(LIH_FROZEN_CORE, integral_backend="native")
        self.assertGreater(frozen.reference_energy, full.reference_energy)
        cost_mha = 1000.0 * (frozen.reference_energy - full.reference_energy)
        self.assertLess(cost_mha, 1.0)


if __name__ == "__main__":
    unittest.main()
