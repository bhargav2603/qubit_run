"""Unit tests for the parts that do not need a chemistry stack.

    pytest tests/ -q

``selftest.py`` is the narrative check that prints a report; this is the
adversarial one. Both run without PySCF or ffsim.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ansatz          # noqa: E402
import binding         # noqa: E402
import geometry        # noqa: E402
import paper           # noqa: E402
import spaces          # noqa: E402
import validation      # noqa: E402


# --------------------------------------------------------------------------
# Determinant counting
# --------------------------------------------------------------------------

def test_full_cas_dimension_matches_paper():
    assert spaces.full_cas_dimension() == 165_636_900
    assert spaces.full_cas_dimension() == math.comb(16, 8) ** 2


@pytest.mark.parametrize(
    "norb,nelec,expected",
    [
        (16, (8, 8), 165_636_900),    # this paper, methane dimer
        (12, (8, 8), 245_025),        # water dimer, 27 qubits, same paper
        (13, (7, 7), 2_944_656),      # methylamine, the 30-qubit alternative
        (12, (6, 6), 853_776),        # Li2S, HiVQE Table 1
        (15, (5, 5), 9_018_009),      # NH3, HiVQE Table 1
    ],
)
def test_dimensions_against_published_tables(norb, nelec, expected):
    """Cross-check the counter against numbers printed in the literature."""
    assert spaces.n_determinants(norb, nelec) == expected


def test_paper_subspace_is_a_large_fraction_of_cas():
    fraction = spaces.subspace_fraction(paper.SUBSPACE_DIMENSION)
    assert 0.75 < fraction < 0.77, "d = 12.6e7 should be ~76% of the CAS"


def test_batching_is_self_consistent():
    assert paper.SAMPLES_PER_BATCH * paper.N_BATCHES == paper.TOTAL_SAMPLES


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("orientation_name", sorted(geometry.ORIENTATIONS))
@pytest.mark.parametrize("distance", paper.FULL_TREATMENT_DISTANCES)
def test_geometry_invariants(orientation_name, distance):
    atoms = geometry.methane_dimer(distance, orientation_name=orientation_name)
    geometry.check_geometry(atoms, expected_distance=distance)
    assert len(atoms) == 10
    assert geometry.n_electrons(atoms) == 20


def test_monomers_stay_rigid_across_the_scan():
    """Changing R must not deform either monomer -- the paper holds them fixed."""
    reference = geometry.ch_bond_lengths(geometry.methane_dimer(2.500))
    for distance in paper.FULL_TREATMENT_DISTANCES:
        lengths = geometry.ch_bond_lengths(geometry.methane_dimer(distance))
        assert np.allclose(lengths, reference, atol=1e-12)


def test_tetrahedral_angles():
    atoms = geometry.methane_dimer(3.638)
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    vectors = coords[1:5] - coords[0]
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    for i in range(4):
        for j in range(i + 1, 4):
            angle = math.degrees(math.acos(float(np.clip(vectors[i] @ vectors[j], -1, 1))))
            assert angle == pytest.approx(math.degrees(math.acos(-1 / 3)), abs=1e-9)


def test_orientations_produce_different_structures():
    built = {
        name: np.array([[x, y, z] for _, x, y, z in geometry.methane_dimer(3.638, orientation_name=name)])
        for name in geometry.ORIENTATIONS
    }
    names = sorted(built)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(built[a] - built[b]).max() > 1e-6


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        geometry.methane_dimer(-1.0)
    with pytest.raises(ValueError):
        geometry.methane_dimer(3.638, r_ch=0.0)
    with pytest.raises(KeyError):
        geometry.methane_dimer(3.638, orientation_name="c2v")


def test_check_geometry_detects_wrong_distance():
    atoms = geometry.methane_dimer(3.638)
    with pytest.raises(AssertionError, match="C-C distance"):
        geometry.check_geometry(atoms, expected_distance=4.000)


def test_check_geometry_detects_deformed_monomer():
    atoms = geometry.methane_dimer(3.638)
    symbol, x, y, z = atoms[1]
    atoms[1] = (symbol, x, y, z + 0.05)          # stretch one C-H bond
    with pytest.raises(AssertionError, match="not rigid"):
        geometry.check_geometry(atoms, expected_distance=3.638)


# --------------------------------------------------------------------------
# Binding energies
# --------------------------------------------------------------------------

def test_binding_energy_sign_and_definition():
    assert binding.binding_energy(-80.5, -80.4) < 0        # bound
    assert binding.binding_energy(-80.4, -80.5) > 0        # repulsive
    assert binding.binding_energy(-80.5, -80.5) == 0.0


def test_binding_curve_requires_the_unbound_reference():
    with pytest.raises(KeyError, match="unbound"):
        binding.binding_curve({3.638: -80.5}, unbound_distance=48.0)


def test_binding_curve_drops_the_reference_point():
    energies = {3.638: -80.5, 4.0: -80.49, 48.0: -80.48}
    curve = binding.binding_curve(energies, unbound_distance=48.0)
    assert set(curve) == {3.638, 4.0}
    assert curve[3.638] == pytest.approx(binding.hartree_to_kcal(-0.02))


def test_variational_check():
    binding.variational_check(-80.4999, -80.5000)          # above: fine
    with pytest.raises(AssertionError, match="variational bound"):
        binding.variational_check(-80.5001, -80.5000)      # below: impossible


def test_agreement_thresholds():
    agreement = binding.Agreement(-80.500000, -80.499999)
    assert agreement.delta_mha == pytest.approx(-0.001, abs=1e-9)
    assert agreement.within_chemical_accuracy
    assert agreement.meets(paper.SQD_VS_CASCI_TARGET_KCAL)


# --------------------------------------------------------------------------
# Variance extrapolation
# --------------------------------------------------------------------------

def test_extrapolation_recovers_exact_intercept():
    variances = np.array([0.4, 0.3, 0.2, 0.1])
    fit = binding.variance_extrapolate(-80.5 + 0.01 * variances, variances)
    assert fit.energy == pytest.approx(-80.5, abs=1e-12)
    assert fit.slope == pytest.approx(0.01, abs=1e-12)
    assert fit.r_squared > 1 - 1e-12


def test_extrapolation_is_insensitive_to_point_order():
    # Tolerance is 1e-12 Ha, not machine epsilon: lstsq accumulates in a
    # different order when the rows are permuted, which moves the intercept by
    # a few 1e-14. That is nine orders of magnitude below chemical accuracy
    # (1.6e-3 Ha) and is arithmetic noise, not a property worth asserting away.
    variances = np.array([0.1, 0.4, 0.2])
    energies = -80.5 + 0.02 * variances
    order = [2, 0, 1]
    a = binding.variance_extrapolate(energies, variances)
    b = binding.variance_extrapolate(energies[order], variances[order])
    assert a.energy == pytest.approx(b.energy, abs=1e-12)
    assert a.slope == pytest.approx(b.slope, rel=1e-9)


@pytest.mark.parametrize(
    "energies,variances",
    [
        ([1.0, 2.0], [0.1, 0.2]),               # too few points
        ([1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),     # degenerate spread
        ([1.0, 2.0, 3.0], [0.1, -0.2, 0.3]),    # negative variance
        ([1.0, 2.0, 3.0], [0.1, 0.2]),          # mismatched lengths
    ],
)
def test_extrapolation_rejects_bad_input(energies, variances):
    with pytest.raises(ValueError):
        binding.variance_extrapolate(energies, variances)


# --------------------------------------------------------------------------
# LUCJ layout
# --------------------------------------------------------------------------

def test_heavy_hex_layout_reproduces_36_qubits():
    layout = ansatz.heavy_hex_layout(paper.N_ORBITALS)
    ansatz.validate_layout(layout)
    assert layout.n_occupation_qubits == 32
    assert layout.n_ancilla_qubits == 4
    assert layout.n_qubits_total == paper.N_QUBITS_TOTAL == 36


def test_layout_pair_structure():
    layout = ansatz.heavy_hex_layout(16)
    assert layout.alpha_alpha == [(p, p + 1) for p in range(15)]
    assert layout.alpha_beta == [(0, 0), (4, 4), (8, 8), (12, 12)]
    for a, b in layout.alpha_alpha + layout.alpha_beta:
        assert a <= b, "interaction pairs must be upper triangular"


def test_validate_layout_rejects_wrong_orbital_count():
    with pytest.raises(AssertionError):
        ansatz.validate_layout(ansatz.heavy_hex_layout(12))


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_fingerprints_are_stable_and_distinct():
    assert validation.physics_fingerprint() == validation.physics_fingerprint()
    assert len(validation.physics_fingerprint()) == 16
    assert validation.physics_fingerprint() != validation.workflow_fingerprint()


def test_all_invariants_pass():
    for name, passed, detail in validation.check_invariants():
        assert passed, f"{name}: {detail}"


def test_cost_ladder_is_ordered_and_named():
    names = [r.name for r in spaces.COST_LADDER]
    assert len(set(names)) == len(names)
    assert spaces.rung("converged").samples_per_batch == paper.SAMPLES_PER_BATCH
    with pytest.raises(KeyError):
        spaces.rung("nonexistent")


def test_paper_configuration_needs_more_ram_than_a_laptop():
    cost = spaces.paper_subspace_cost()
    assert not cost.fits_in(16 * 1024**3, parallel=True)
    assert cost.fits_in(16 * 1024**3, parallel=False)


# --------------------------------------------------------------------------
# Regression: subspace_fraction must be system-aware
# --------------------------------------------------------------------------

def _sqd_result(**kwargs):
    """Build an SqdResult without importing the chemistry stack."""
    import sqd
    defaults = dict(energy=-80.0, subspace_dimension=1000, config={"energy_tol": 1e-6})
    defaults.update(kwargs)
    return sqd.SqdResult(**defaults)


def test_subspace_fraction_uses_this_systems_cas():
    """Regression: the denominator was hardcoded to the paper's 165,636,900.

    That made the fraction meaningless for any other active space and silently
    defeated the saturation check in `run.py validate`, which is the step that
    proves SQD and CASCI are solving the same Hamiltonian.
    """
    small = _sqd_result(subspace_dimension=4900, full_dimension=4900)
    assert small.subspace_fraction == pytest.approx(1.0)

    half = _sqd_result(subspace_dimension=2450, full_dimension=4900)
    assert half.subspace_fraction == pytest.approx(0.5)


def test_subspace_fraction_falls_back_to_paper_cas():
    paper_run = _sqd_result(subspace_dimension=paper.SUBSPACE_DIMENSION)
    assert paper_run.subspace_fraction == pytest.approx(0.761, abs=1e-3)


def test_subspace_fraction_is_serialized():
    result = _sqd_result(subspace_dimension=4900, full_dimension=4900)
    payload = result.to_dict()
    assert payload["full_dimension"] == 4900
    assert payload["subspace_fraction"] == pytest.approx(1.0)
