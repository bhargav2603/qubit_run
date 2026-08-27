"""Prove the stack without a chemistry package.

Everything here runs on a bare NumPy/SciPy install -- which matters, because
the machine this folder is developed on cannot install PySCF or ffsim at all
(no Windows wheels). The logic that decides whether a run is correct should be
checkable on the machine you are sitting at, not only on the one that can
afford the calculation.

    python run.py selftest
"""

from __future__ import annotations

import math

import numpy as np

import binding
import geometry
import paper
import spaces
import validation


class SelfTestFailure(AssertionError):
    pass


def _check(name: str, condition: bool, detail: str = "") -> tuple[str, bool, str]:
    return (name, bool(condition), detail)


# --------------------------------------------------------------------------

def test_geometry() -> list[tuple[str, bool, str]]:
    out = []

    # Every published distance, every orientation, all invariants.
    failures = []
    for name in geometry.ORIENTATIONS:
        for distance in paper.FULL_TREATMENT_DISTANCES:
            atoms = geometry.methane_dimer(distance, orientation_name=name)
            try:
                geometry.check_geometry(atoms, expected_distance=distance)
            except AssertionError as exc:
                failures.append(f"{name}@{distance}: {exc}")
    out.append(_check(
        "geometry invariants hold on the full grid",
        not failures,
        f"{len(geometry.ORIENTATIONS)} orientations x "
        f"{len(paper.FULL_TREATMENT_DISTANCES)} distances"
        + (f"; failures: {failures[:2]}" if failures else ""),
    ))

    # Tetrahedral angles, to machine precision.
    atoms = geometry.methane_dimer(paper.EQUILIBRIUM_DISTANCE)
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    carbon, hydrogens = coords[0], coords[1:5]
    vectors = hydrogens - carbon
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    angles = [
        math.degrees(math.acos(float(np.clip(vectors[i] @ vectors[j], -1, 1))))
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    tetrahedral = math.degrees(math.acos(-1.0 / 3.0))
    out.append(_check(
        "monomer is exactly tetrahedral",
        max(abs(a - tetrahedral) for a in angles) < 1e-9,
        f"all 6 H-C-H angles = {tetrahedral:.6f} deg",
    ))

    # Monomers must be congruent: B is a mirror image of A, so their sorted
    # internal C-H sets are identical at every separation.
    lengths = geometry.ch_bond_lengths(atoms)
    out.append(_check(
        "monomers are rigid and congruent",
        float(lengths.max() - lengths.min()) < 1e-12,
        f"C-H = {lengths.mean():.6f} A, spread {lengths.max() - lengths.min():.2e}",
    ))

    # Orientation must actually change the structure, or the flag is a no-op.
    #
    # Note the metric: the *minimum* H...H contact cannot distinguish D3d from
    # D3h, because in both the apex hydrogens sit on the intermolecular axis and
    # the 60 deg twist only moves the tripods. Comparing full coordinate sets is
    # what actually detects the difference.
    structures = {
        name: np.array([
            [x, y, z]
            for _, x, y, z in geometry.methane_dimer(3.638, orientation_name=name)
        ])
        for name in geometry.ORIENTATIONS
    }
    names = sorted(structures)
    distinct = all(
        float(np.abs(structures[a] - structures[b]).max()) > 1e-6
        for i, a in enumerate(names)
        for b in names[i + 1:]
    )
    spreads = {
        f"{a}/{b}": float(np.abs(structures[a] - structures[b]).max())
        for i, a in enumerate(names)
        for b in names[i + 1:]
    }
    out.append(_check(
        "orientations are physically distinct",
        distinct,
        "max coordinate difference: "
        + ", ".join(f"{k}={v:.3f} A" for k, v in sorted(spreads.items())),
    ))

    # The unbound reference must genuinely be non-interacting.
    unbound = geometry.methane_dimer(paper.UNBOUND_DISTANCE)
    coords = np.array([[x, y, z] for _, x, y, z in unbound])
    cross = np.linalg.norm(coords[0:5][:, None] - coords[5:10][None], axis=2).min()
    out.append(_check(
        "48 A reference is non-interacting",
        cross > 40.0,
        f"closest atom pair across monomers is {cross:.2f} A",
    ))
    return out


def test_spaces() -> list[tuple[str, bool, str]]:
    out = []
    dimension = spaces.full_cas_dimension()
    out.append(_check(
        "CAS dimension = C(16,8)^2",
        dimension == math.comb(16, 8) ** 2 == 165_636_900,
        f"{dimension:,}",
    ))
    # Half filling is the *worst* case for sector compression -- C(16,8)^2 is the
    # largest sector there is. 26x is the true figure here, not the four or five
    # orders of magnitude seen at low filling (the 52-qubit N2 case).
    out.append(_check(
        "sector beats dense simulation",
        20.0 < spaces.sector_compression_factor(16, (8, 8)) < 30.0,
        f"{spaces.humanize_bytes(spaces.statevector_bytes(16, (8, 8)))} vs "
        f"{spaces.humanize_bytes(spaces.qubit_statevector_bytes(16))} "
        f"({spaces.sector_compression_factor(16, (8, 8)):,.1f}x) -- "
        "modest, because (8,8) is the largest sector at 16 orbitals",
    ))
    cost = spaces.paper_subspace_cost()
    out.append(_check(
        "paper configuration needs a large-memory machine",
        cost.parallel_peak_bytes > 64 * 1024**3,
        f"K=10 in parallel needs {spaces.humanize_bytes(cost.parallel_peak_bytes)}; "
        f"sequential {spaces.humanize_bytes(cost.sequential_peak_bytes)}",
    ))
    ladder = [r.name for r in spaces.COST_LADDER]
    out.append(_check(
        "cost ladder is monotonic in sample count",
        all(
            spaces.COST_LADDER[i].samples_per_batch
            <= spaces.COST_LADDER[i + 1].samples_per_batch
            for i in range(len(spaces.COST_LADDER) - 1)
        ),
        " -> ".join(ladder),
    ))
    return out


def test_binding() -> list[tuple[str, bool, str]]:
    out = []

    # Sign convention: bound states are negative.
    e_bind = binding.binding_energy(-80.5000, -80.4990)
    out.append(_check(
        "bound geometry gives negative binding energy",
        e_bind < 0,
        f"{binding.hartree_to_kcal(e_bind):.4f} kcal/mol",
    ))

    # The unbound reference must be consumed, not returned.
    energies = {3.638: -80.500000, 4.000: -80.499500, paper.UNBOUND_DISTANCE: -80.499000}
    curve = binding.binding_curve(energies, unbound_distance=paper.UNBOUND_DISTANCE)
    out.append(_check(
        "binding curve consumes the unbound point",
        paper.UNBOUND_DISTANCE not in curve and len(curve) == 2,
        f"{sorted(curve)} in kcal/mol",
    ))

    # Missing reference must be an error, never a silent default.
    try:
        binding.binding_curve({3.638: -80.5}, unbound_distance=paper.UNBOUND_DISTANCE)
        raised = False
    except KeyError:
        raised = True
    out.append(_check(
        "missing unbound reference raises",
        raised,
        "binding energy is undefined without E(48 A)",
    ))

    # Unit conversion round trip.
    out.append(_check(
        "hartree <-> kcal/mol round trip",
        abs(binding.kcal_to_hartree(binding.hartree_to_kcal(0.00123)) - 0.00123) < 1e-15,
        f"1 kcal/mol = {binding.CHEMICAL_ACCURACY_HARTREE:.8f} Ha",
    ))

    # Variational bound: SQD may not fall below CASCI.
    try:
        binding.variational_check(-80.50001, -80.50000)
        caught = False
    except AssertionError:
        caught = True
    out.append(_check(
        "variational violation is caught",
        caught,
        "P H P cannot lie below the exact eigenvalue",
    ))
    binding.variational_check(-80.49999, -80.50000)  # must not raise
    return out


def test_extrapolation() -> list[tuple[str, bool, str]]:
    out = []

    # Exact linear data must extrapolate to the exact intercept.
    variances = np.array([0.4, 0.3, 0.2, 0.1])
    exact = -80.500000
    slope = 0.01
    energies = exact + slope * variances
    fit = binding.variance_extrapolate(energies, variances)
    out.append(_check(
        "exact linear data recovers the intercept",
        abs(fit.energy - exact) < 1e-12 and fit.r_squared > 1 - 1e-12,
        str(fit),
    ))

    # Noisy data: the intercept should stay close and the error bar be finite.
    rng = np.random.default_rng(0)
    noisy = energies + rng.normal(0, 1e-6, size=energies.shape)
    fit = binding.variance_extrapolate(noisy, variances)
    out.append(_check(
        "noisy data gives a finite error bar",
        abs(fit.energy - exact) < 1e-4 and fit.stderr > 0,
        f"{fit.energy_error_mha:.4f} mHa error estimate",
    ))

    # Degenerate inputs must raise rather than return nonsense.
    failures = 0
    for bad_args in (
        ([1.0, 2.0], [0.1, 0.2]),                     # too few points
        ([1.0, 2.0, 3.0], [0.1, 0.1, 0.1]),           # no variance spread
        ([1.0, 2.0, 3.0], [0.1, -0.2, 0.3]),          # negative variance
    ):
        try:
            binding.variance_extrapolate(*bad_args)
        except ValueError:
            failures += 1
    out.append(_check(
        "degenerate extrapolations raise",
        failures == 3,
        "too-few points, zero spread, negative variance",
    ))

    # The paper uses exactly three sample sizes.
    out.append(_check(
        "paper's extrapolation uses 3 points",
        len(paper.EXTRAPOLATION_SAMPLE_SIZES) == 3,
        f"|chi_b| = {paper.EXTRAPOLATION_SAMPLE_SIZES}",
    ))
    return out


def test_sampling() -> list[tuple[str, bool, str]]:
    """Uniform sampler and validity statistics. Needs qiskit, not chemistry."""
    out = []
    try:
        import sampling
    except ImportError as exc:
        return [_check("sampling module imports", False, str(exc))]

    norb, nelec = paper.N_ORBITALS, (paper.N_ALPHA, paper.N_BETA)

    try:
        sampler = sampling.UniformSampler(norb, nelec, seed=7)
        result = sampler.sample(shots=512)
        fraction = sampling.valid_configuration_fraction(result.bit_array, norb, nelec)
        out.append(_check(
            "uniform sampler emits only valid configurations",
            abs(fraction - 1.0) < 1e-12,
            f"{fraction:.4f} of 512 shots have (8,8) occupation",
        ))
    except Exception as exc:  # qiskit missing or API drift
        out.append(_check("uniform sampler", False, f"{type(exc).__name__}: {exc}"))

    # The validity fraction is a strong diagnostic at low filling and a weak one
    # at half filling. This system is half filled, so the baseline is percent-
    # level, not the 1e-6 seen in the 52-qubit N2 tutorial. Asserting the true
    # value keeps that from being quietly misread as a quantum signal later.
    probability = sampling.random_validity_probability(norb, nelec)
    tutorial = sampling.random_validity_probability(26, (5, 5))
    out.append(_check(
        "random validity baseline is percent-level, not negligible",
        0.03 < probability < 0.05,
        f"random 32-bit string is a valid (8,8) configuration with p = {probability:.3%}; "
        f"compare (5,5)-in-26-orbitals at p = {tutorial:.2e}. Half filling makes "
        "valid strings common, so validity fraction is NOT the quantum signal here "
        "-- the ablation energy gap is.",
    ))
    return out


def test_paper_constants() -> list[tuple[str, bool, str]]:
    out = []
    out.append(_check(
        "Table II parameters present",
        (paper.TOTAL_SAMPLES, paper.N_BATCHES, paper.SAMPLES_PER_BATCH)
        == (200_000, 10, 20_000),
        f"|chi|={paper.TOTAL_SAMPLES:,}, K={paper.N_BATCHES}, "
        f"|chi_b|={paper.SAMPLES_PER_BATCH:,}, d={paper.SUBSPACE_DIMENSION:,}",
    ))
    out.append(_check(
        "batching is self-consistent",
        paper.SAMPLES_PER_BATCH * paper.N_BATCHES == paper.TOTAL_SAMPLES,
        f"{paper.SAMPLES_PER_BATCH:,} x {paper.N_BATCHES} = {paper.TOTAL_SAMPLES:,}",
    ))
    out.append(_check(
        "mitigation matches the paper's Methods",
        paper.GATE_TWIRLING and paper.DYNAMICAL_DECOUPLING
        and not paper.MEASUREMENT_TWIRLING,
        "gate twirling + DD, no measurement twirling",
    ))
    return out


# --------------------------------------------------------------------------

SUITES = {
    "paper constants": test_paper_constants,
    "geometry": test_geometry,
    "spaces": test_spaces,
    "binding energies": test_binding,
    "variance extrapolation": test_extrapolation,
    "sampling": test_sampling,
}


def run(verbose: bool = True) -> bool:
    """Run everything. Returns True if all checks pass."""
    total = failed = 0
    print("=" * 74)
    print("methane dimer @ 36 qubits -- self test (no chemistry packages needed)")
    print("=" * 74)

    for title, suite in SUITES.items():
        print(f"\n{title}")
        for name, passed, detail in suite():
            total += 1
            mark = "ok  " if passed else "FAIL"
            if not passed:
                failed += 1
            print(f"  [{mark}] {name}")
            if detail and (verbose or not passed):
                print(f"         {detail}")

    print("\ninvariants")
    for name, passed, detail in validation.check_invariants():
        total += 1
        mark = "ok  " if passed else "FAIL"
        if not passed:
            failed += 1
        print(f"  [{mark}] {name}")
        if detail and (verbose or not passed):
            print(f"         {detail}")

    print("\n" + "=" * 74)
    print(f"{total - failed}/{total} checks passed"
          + ("" if failed == 0 else f"  --  {failed} FAILED"))
    print(f"physics fingerprint  {validation.physics_fingerprint()}")
    print(f"workflow fingerprint {validation.workflow_fingerprint()}")
    print("=" * 74)
    return failed == 0
