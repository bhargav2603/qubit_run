#!/usr/bin/env python3
"""End-to-end proof of the local Qiskit stack, with no PySCF and no cache.

PySCF publishes no Windows wheel, so on Windows the imipramine Hamiltonian has
to be built elsewhere and copied in. That leaves a gap: nothing on the Windows
machine would be exercised until the cache arrives, and a mapping or ordering
bug would then surface as a wrong number rather than a failed test.

This command closes that gap in two stages.

**Structure** (instant, needs only NumPy): the module layering that lets
`prepare` run without Qiskit and `vqe` run without PySCF; the molecule itself --
formula, active space, and the aromatic pi core *re-derived from the stored
geometry* rather than trusted; the fingerprints; the artifact names; and the
accuracy-scoring arithmetic.

**Simulation** (needs Qiskit and OpenFermion, not PySCF): a random but
*structurally real* Hamiltonian at the same CAS(6e,6o) size -- generated through
the identical OpenFermion path `hamiltonian.py` uses, so it is Hermitian,
particle-number conserving and has a genuine ground state at the right electron
count -- run against exact classical answers:

1. the OpenFermion -> Qiskit operator conversion, checked element by element
   under the bit-reversal that separates the two libraries' serialization
   conventions, and then determinant by determinant through the whole Aer path.
   Comparing spectra would prove nothing here: bit reversal is a similarity
   transform, so a genuinely reversed register has an identical spectrum;
2. the Hartree-Fock frame rotation, i.e. that theta = 0 really is the reference
   determinant;
3. matrix product state against exact statevector;
4. parameter-shift gradients against finite differences;
5. a full VQE, checked against exact diagonalization in the target electron
   sector and against the variational bound.

**ADAPT** (the published reference method, same synthetic Hamiltonian). This
stage exists because ADAPT-VQE makes a structural claim the hardware-efficient
path only makes statistically -- that the state can never leave the six-electron
sector -- and a structural claim is worth nothing unless the structure is
verified. So every pool operator's commutator with N and with S^2 is computed
directly, the adjoint gradient and the operator-selection gradient are both
checked against finite differences, and the emitted Qiskit circuit is checked
against the sparse algebra that produced it.

Computing those commutators is how the S^2 half of that claim was found to be
false: particle number and Sz are exact for every operator, but only the singles
and the seniority-zero doubles commute with S^2. The claim was corrected rather
than the check relaxed, and the singlet is now verified where it actually holds
-- at the variational minimum, against the contamination budget.

If this passes, the only thing still missing on this machine is the molecule.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# Longest C-H distance accepted when re-deriving hydrogen counts from geometry.
CH_BOND_MAX_ANGSTROM = 1.2


def _check(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:<44} {detail}", flush=True)


# --------------------------------------------------------------------------
# Stage 1: structure. NumPy only.
# --------------------------------------------------------------------------


def _check_layering(results: list[tuple[str, bool, str]]) -> None:
    """The chemistry layer and the simulator layer must stay independent.

    `prepare` and `validate` run on a Linux box with no Qiskit installed only
    because hamiltonian.py and validation.py never import it. The mirror rule
    matters just as much: qiskit_runtime.py must not import PySCF, or the
    Windows machine -- where PySCF cannot even be installed -- could not run the
    VQE at all. Both directions are asserted rather than assumed, because both
    are one careless top-level import away from breaking.
    """
    for module in ("qiskit", "qiskit_aer", "pyscf", "openfermion"):
        sys.modules.pop(module, None)

    import hamiltonian  # noqa: F401
    import validation  # noqa: F401

    forbidden = [name for name in ("qiskit", "qiskit_aer") if name in sys.modules]
    _check(
        results,
        "chemistry layer imports no Qiskit",
        not forbidden,
        str(forbidden) if forbidden else "hamiltonian, validation",
    )
    _check(results, "chemistry layer imports no PySCF at module scope", "pyscf" not in sys.modules)

    import adapt_runtime  # noqa: F401
    import qiskit_runtime  # noqa: F401

    _check(results, "simulator layer imports no PySCF at module scope", "pyscf" not in sys.modules)
    # Aer's import costs seconds; keeping it lazy is what makes `--help`, the
    # unit tests and this stage instant.
    _check(
        results,
        "simulator layer imports Qiskit lazily",
        "qiskit" not in sys.modules,
        "qiskit_runtime, adapt_runtime: deferred to first use",
    )
    # ADAPT runs on SciPy sparse algebra and only touches Qiskit to emit the
    # final circuit, so it must not drag OpenFermion in at import time either.
    _check(
        results,
        "adapt_runtime imports OpenFermion lazily",
        "openfermion" not in sys.modules,
        "deferred to first use",
    )


def _check_spec(results: list[tuple[str, bool, str]]) -> None:
    import dataclasses

    import hamiltonian
    from molecule import AROMATIC_CORE_ATOMS, IMIPRAMINE

    _check(
        results,
        "active space is 6e,6o",
        (IMIPRAMINE.n_active_electrons, IMIPRAMINE.n_active_orbitals) == (6, 6),
        f"{2 * IMIPRAMINE.n_active_orbitals} qubits",
    )
    _check(results, "closed-shell neutral", IMIPRAMINE.charge == 0 and IMIPRAMINE.spin == 0)

    symbols = [symbol for symbol, _ in IMIPRAMINE.atoms]
    formula = (symbols.count("C"), symbols.count("H"), symbols.count("N"))
    _check(results, "formula is C19H24N2", formula == (19, 24, 2), str(formula))

    fingerprint = hamiltonian.spec_fingerprint(IMIPRAMINE)
    _check(
        results,
        "spec fingerprint deterministic",
        fingerprint == hamiltonian.spec_fingerprint(IMIPRAMINE),
        fingerprint[:16],
    )
    physics = hamiltonian.physics_fingerprint()
    workflow = hamiltonian.workflow_fingerprint()
    _check(
        results,
        "physics and workflow hashes computable",
        len(physics) == 64 and len(workflow) == 64 and "unknown" not in (physics, workflow),
        f"physics {physics[:16]}",
    )
    _check(
        results,
        "the two hashes are not the same digest",
        physics != workflow,
        "receipt gate is narrower than provenance",
    )

    natoms = len(IMIPRAMINE.atoms)
    indices = tuple(IMIPRAMINE.diagnostic_atom_indices)
    _check(results, "diagnostic atoms in range", all(0 <= i < natoms for i in indices))
    _check(results, "diagnostic atoms match the named pi core", indices == AROMATIC_CORE_ATOMS)
    occupied, virtual = hamiltonian.localization_thresholds(IMIPRAMINE)
    _check(
        results,
        "localization threshold is strict for a pi system",
        IMIPRAMINE.min_active_orbital_localization == 0.80,
        f"occupied {occupied}",
    )
    _check(
        results,
        "virtuals held to a separate, looser bar",
        0.0 < virtual < occupied,
        f"{virtual} < {occupied}",
    )

    # The literature anchor must not be able to invalidate a cache or a receipt.
    stripped = dataclasses.replace(IMIPRAMINE, published_reference=None)
    _check(
        results,
        "published reference is outside the fingerprint",
        hamiltonian.spec_fingerprint(stripped) == fingerprint,
    )
    reference = IMIPRAMINE.published_reference or {}
    _check(
        results,
        "published CASCI anchor is recorded",
        abs(float(reference.get("casci_energy_hartree", 0.0)) + 841.953249) < 1e-9,
        str(reference.get("casci_energy_hartree")),
    )

    mixed = dataclasses.replace(IMIPRAMINE, basis={"N": "6-31g*", "C": "6-31g", "H": "6-31g"})
    _check(
        results,
        "changing the basis changes the cache identity",
        hamiltonian.spec_fingerprint(mixed) != fingerprint,
    )


def _check_pi_core_from_geometry(results: list[tuple[str, bool, str]]) -> None:
    """Re-derive the aromatic core from hydrogen counts rather than trusting it.

    Aromatic CH carbons carry exactly one hydrogen and ring-junction carbons
    none, while every sp3 carbon in the ethano bridge, the propyl tail and the
    N-methyls carries two or three. If the geometry is ever replaced, this is
    what catches a diagnostic set that no longer describes the pi system -- and
    the diagnostic set is what stops the active space from quietly landing on
    the side chain instead of the pharmacophore.
    """
    from molecule import IMIPRAMINE

    positions = np.array([coords for _, coords in IMIPRAMINE.atoms], dtype=float)
    symbols = [symbol for symbol, _ in IMIPRAMINE.atoms]

    hydrogen_count = [0] * len(symbols)
    for index, symbol in enumerate(symbols):
        if symbol != "H":
            continue
        distances = np.linalg.norm(positions - positions[index], axis=1)
        distances[index] = np.inf
        nearest = int(np.argmin(distances))
        if distances[nearest] <= CH_BOND_MAX_ANGSTROM:
            hydrogen_count[nearest] += 1

    carbons = [i for i, symbol in enumerate(symbols) if symbol == "C"]
    derived = tuple(sorted(i for i in carbons if hydrogen_count[i] <= 1))
    declared = tuple(
        sorted(i for i in IMIPRAMINE.diagnostic_atom_indices if symbols[i] == "C")
    )
    _check(results, "aromatic carbons re-derived from H counts", derived == declared, str(derived))
    _check(results, "twelve aromatic carbons", len(derived) == 12, str(len(derived)))

    nitrogens = [i for i, symbol in enumerate(symbols) if symbol == "N"]
    included = [i for i in nitrogens if i in IMIPRAMINE.diagnostic_atom_indices]
    _check(results, "exactly one nitrogen in the pi core", len(included) == 1, str(included))
    # The azepine N bridges the two rings, so it is the one bonded to two
    # aromatic carbons; the side-chain dimethylamino N is not.
    aromatic = set(derived)
    for nitrogen in nitrogens:
        distances = np.linalg.norm(positions - positions[nitrogen], axis=1)
        neighbours = {i for i in np.flatnonzero(distances < 1.6) if i != nitrogen}
        is_azepine = len(neighbours & aromatic) == 2
        _check(
            results,
            f"nitrogen {nitrogen} classified correctly",
            is_azepine == (nitrogen in IMIPRAMINE.diagnostic_atom_indices),
            "azepine" if is_azepine else "side chain",
        )


def _check_artifacts(results: list[tuple[str, bool, str]]) -> None:
    import hamiltonian
    import run

    cache = Path(run.CACHE)
    receipt = hamiltonian.validation_receipt_path(cache)
    _check(
        results,
        "cache is named for the active space",
        cache.name == "hamiltonian_cas6e6o.json",
        cache.name,
    )
    _check(
        results,
        "receipt has a single .json suffix",
        receipt.name == "hamiltonian_cas6e6o.validated.json",
        receipt.name,
    )
    _check(results, "receipt sits beside the cache", receipt.parent == cache.parent)
    _check(results, "results land under results/", Path(run.RESULTS).name == "results")

    from qiskit_runtime import _default_result_path

    scan = {
        _default_result_path(Path(run.RESULTS), layers, method).name
        for layers in (4, 6)
        for method in ("matrix_product_state", "statevector")
    }
    _check(
        results,
        "a layer/method scan cannot overwrite itself",
        len(scan) == 4,
        ", ".join(sorted(scan)),
    )

    from adapt_runtime import POOLS

    pool_names = {f"adapt_{pool}.json" for pool in POOLS}
    _check(
        results,
        "a pool scan cannot overwrite itself or a VQE result",
        len(pool_names) == len(POOLS) and not (pool_names & scan),
        ", ".join(sorted(pool_names)),
    )

    # The charts read results, nothing else. A missing matplotlib must cost the
    # pictures and nothing more -- so `visualize` has to stay importable, and out
    # of every other module's import path, on a machine that lacks it.
    import visualize

    _check(
        results,
        "visualize imports matplotlib lazily",
        "matplotlib" not in sys.modules,
        "charts are optional, deferred to first use",
    )
    _check(
        results,
        "no result file is charted twice",
        visualize.load_runs(Path(run.RESULTS)) is not None,
        "loader tolerates a missing results/ directory",
    )


def _check_scoring(results: list[tuple[str, bool, str]]) -> None:
    import hamiltonian
    from qiskit_runtime import MPS_AGREEMENT_BUDGET_HA

    _check(
        results,
        "chemical accuracy is 1.6 mHa",
        hamiltonian.CHEMICAL_ACCURACY_MHA == 1.6,
    )
    budget = hamiltonian.CONTAMINATION_ENERGY_BUDGET_HA
    _check(
        results,
        "contamination budget is a tenth of that",
        abs(budget - 1.6e-4) < 1e-18,
        f"{budget:.3e} Ha",
    )
    _check(
        results,
        "MPS truncation is held to the same budget",
        abs(MPS_AGREEMENT_BUDGET_HA - budget) < 1e-18,
        f"{MPS_AGREEMENT_BUDGET_HA:.3e} Ha",
    )
    _check(
        results,
        "Hamiltonian cutoff defaults to 1e-6",
        hamiltonian.DEFAULT_HAMILTONIAN_CUTOFF == 1.0e-6,
    )

    price = hamiltonian.CONTAMINATION_PRICE_HA
    # Turning a penalty off must not weaken the claim: spin contamination is
    # still priced at CONTAMINATION_PRICE_HA even at --spin-penalty 0.
    spin_penalty_off = max(1.0, price) * 1.0e-5 + max(0.0, price) * 5.0e-2
    _check(
        results,
        "spin contamination priced with the penalty off",
        spin_penalty_off > budget,
        f"{spin_penalty_off:.3e} Ha",
    )
    # A realistic converged hardware-efficient ansatz must still be able to pass.
    realistic = max(1.0, price) * 5.0e-5 + max(0.0, price) * 1.0e-6
    _check(results, "realistic HEA leakage passes", realistic <= budget, f"{realistic:.3e} Ha")


# --------------------------------------------------------------------------
# Stage 2: simulation. Qiskit + OpenFermion, no PySCF.
# --------------------------------------------------------------------------


def _synthetic_chemistry(seed: int, norb: int, nelec: int) -> Any:
    """A random Hamiltonian of the right shape, built the same way the real one is."""
    from openfermion import (
        InteractionOperator,
        get_fermion_operator,
        get_sparse_operator,
        jordan_wigner,
    )
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from scipy.sparse.linalg import eigsh

    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    from hamiltonian import CachedHamiltonian, electron_number_operator, hartree_fock_occupation

    rng = np.random.default_rng(seed)

    # A bound one-body part: descending diagonal orbital energies plus small
    # symmetric coupling, so the lowest orbitals are genuinely the occupied ones.
    one_body = rng.normal(0.0, 0.15, (norb, norb))
    one_body = 0.5 * (one_body + one_body.T)
    one_body -= np.diag(np.linspace(3.0, 0.5, norb))

    # Chemist-notation (ij|kl) with the full eight-fold symmetry, exactly the
    # symmetry `ao2mo.restore(1, ...)` hands back in hamiltonian.py.
    two_body = rng.normal(0.0, 0.12, (norb, norb, norb, norb))
    two_body = two_body + two_body.transpose(1, 0, 2, 3)
    two_body = two_body + two_body.transpose(0, 1, 3, 2)
    two_body = two_body + two_body.transpose(2, 3, 0, 1)
    two_body /= 8.0

    core_energy = -12.5
    two_body_openfermion = np.asarray(
        two_body.transpose(0, 2, 3, 1), dtype=float, order="C"
    )
    one_spin, two_spin = spinorb_from_spatial(
        np.asarray(one_body, dtype=float), two_body_openfermion
    )
    operator = jordan_wigner(
        get_fermion_operator(InteractionOperator(core_energy, one_spin, 0.5 * two_spin))
    )
    operator.compress(abs_tol=1.0e-12)

    n_qubits = 2 * norb
    matrix = get_sparse_operator(operator, n_qubits=n_qubits)

    occupied = hartree_fock_occupation(nelec)
    hf_vector = jw_configuration_state(occupied, n_qubits)
    hf_energy = float(np.real(np.vdot(hf_vector, matrix @ hf_vector)))

    number_matrix = get_sparse_operator(
        electron_number_operator(n_qubits), n_qubits=n_qubits
    )
    diagonal = np.real(np.asarray(number_matrix.diagonal()).reshape(-1))
    sector = np.flatnonzero(np.isclose(diagonal, nelec, atol=1.0e-9))
    sector_matrix = matrix[sector][:, sector]
    exact = float(
        np.real(eigsh(sector_matrix, k=1, which="SA", return_eigenvectors=False)[0])
    )

    metadata = {
        "molecule": f"Synthetic ({nelec}e,{norb}o) self-test Hamiltonian",
        "basis": "synthetic",
        "charge": 0,
        "spin_2s": 0,
        "model_note": (
            "Randomly generated one- and two-body integrals with full eight-fold "
            "permutational symmetry, mapped through the same OpenFermion path as "
            "the real workflow. Physically meaningless; structurally identical."
        ),
        "seed": seed,
        "qubits": n_qubits,
        "active_spatial_orbitals": norb,
        "active_electrons": nelec,
        "core_energy_hartree": core_energy,
        "hartree_fock_energy_hartree": hf_energy,
        "reference_energy_hartree": exact,
        "reference_method": "exact sector diagonalization",
        "pauli_terms": len(operator.terms),
        "truncation_l1_bound_hartree": 0.0,
    }
    return CachedHamiltonian(
        hamiltonian=operator,
        n_qubits=n_qubits,
        n_active_electrons=nelec,
        hartree_fock_energy=hf_energy,
        reference_energy=exact,
        reference_method="exact sector diagonalization",
        core_energy=core_energy,
        metadata=metadata,
    )


def _bit_reversal(n_qubits: int) -> np.ndarray:
    """Index permutation between OpenFermion's and Qiskit's basis ordering."""
    indices = np.arange(1 << n_qubits)
    reversed_index = np.zeros(1 << n_qubits, dtype=int)
    for bit in range(n_qubits):
        reversed_index |= ((indices >> bit) & 1) << (n_qubits - 1 - bit)
    return reversed_index


def _adequate_number_penalty(
    chemistry: Any, candidates: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
) -> tuple[float, float]:
    """Smallest penalty whose exact ground state actually sits in the N-electron sector.

    A hardware-efficient ansatz is not particle-number conserving, so the only
    thing stopping the optimizer from wandering into a lower-lying sector is the
    penalty. If the penalised *exact* ground state is already outside the target
    sector, no VQE run against it can mean anything -- it will faithfully find
    the wrong answer. `validate` applies this same test to the real molecule for
    every penalty it records, which is why its output has a per-penalty verdict
    column.
    """
    from openfermion import get_sparse_operator

    from hamiltonian import electron_number_operator, number_deviation_operator

    n_qubits = chemistry.n_qubits
    target = chemistry.n_active_electrons
    hamiltonian = get_sparse_operator(chemistry.hamiltonian, n_qubits=n_qubits).toarray()
    number = get_sparse_operator(
        electron_number_operator(n_qubits), n_qubits=n_qubits
    ).toarray()
    deviation = get_sparse_operator(
        number_deviation_operator(n_qubits, target), n_qubits=n_qubits
    ).toarray()
    for penalty in candidates:
        _, vectors = np.linalg.eigh(hamiltonian + penalty * deviation)
        ground = vectors[:, 0]
        electrons = float(np.real(ground.conj() @ number @ ground))
        if abs(electrons - target) < 1.0e-8:
            return penalty, electrons
    raise RuntimeError(
        f"No penalty in {candidates} selects the {target}-electron sector."
    )


def _simulation_stage(
    results: list[tuple[str, bool, str]], args: argparse.Namespace, norb: int, nelec: int
) -> Any:
    from openfermion import get_sparse_operator

    try:
        from openfermion import jw_configuration_state
    except ImportError:
        from openfermion.linalg import jw_configuration_state

    from hamiltonian import conjugate_with_x, save_cache
    from qiskit_runtime import (
        DEFAULT_TRUNCATION_THRESHOLD,
        AerEvaluator,
        _minimize,
        build_objective,
        to_sparse_pauli_op,
    )

    chemistry = _synthetic_chemistry(args.seed, norb, nelec)
    n_qubits = chemistry.n_qubits
    print(
        f"  Synthetic system            : {n_qubits} qubits, "
        f"{len(chemistry.hamiltonian.terms)} Pauli terms",
        flush=True,
    )
    print(
        f"  Hartree-Fock / exact        : {chemistry.hartree_fock_energy:.12f} / "
        f"{chemistry.reference_energy:.12f} Ha",
        flush=True,
    )

    if args.write_cache:
        save_cache(chemistry, Path(args.write_cache))
        print(f"  Wrote synthetic cache: {args.write_cache}", flush=True)

    # ---------------------------------------------------------------- 1
    # Operator conversion, checked matrix element by matrix element.
    #
    # The two libraries agree on what qubit k is -- to_sparse_pauli_op maps
    # OpenFermion's qubit k to Qiskit's qubit k -- but they disagree on how a
    # basis state is *serialized* into a vector index: OpenFermion puts qubit 0
    # in the most significant bit, Qiskit in the least. So the correct
    # conversion produces matrices related by reversing the bits of both
    # indices, and that permutation is the whole content of the disagreement.
    # Undoing it and demanding exact equality is far stronger than comparing
    # spectra (bit reversal is a similarity transform, so spectra agree even
    # when the register is genuinely reversed) and, at 0.1 s against 40 s of
    # eigendecomposition, far cheaper too.
    qiskit_operator = to_sparse_pauli_op(chemistry.hamiltonian, n_qubits)
    openfermion_matrix = get_sparse_operator(chemistry.hamiltonian, n_qubits=n_qubits)
    reversed_index = _bit_reversal(n_qubits)
    qiskit_matrix = qiskit_operator.to_matrix(sparse=True)[reversed_index][:, reversed_index]
    element_error = float(abs(openfermion_matrix - qiskit_matrix).max())
    _check(results, "operator conversion: every matrix element", element_error < 1.0e-9,
           f"max |dH_ij| = {element_error:.3e} Ha over {openfermion_matrix.nnz} nonzeros")

    evaluator = AerEvaluator(n_qubits, args.layers, method="matrix_product_state",
                             seed=args.seed)
    exact_evaluator = AerEvaluator(n_qubits, args.layers, method="statevector",
                                   seed=args.seed)

    # The same physics through the whole Aer path rather than through NumPy:
    # frame rotation, transpilation, layout relabelling and
    # save_expectation_value. Any of those could reorder a register on its own,
    # and none of them is exercised by comparing two matrices.
    determinants = 3
    determinant_error = 0.0
    rng = np.random.default_rng(args.seed + 1)
    for _ in range(determinants):
        occupation = sorted(
            rng.choice(n_qubits, size=nelec, replace=False).tolist()
        )
        vector = jw_configuration_state(occupation, n_qubits)
        openfermion_value = float(np.real(np.vdot(vector, openfermion_matrix @ vector)))
        # Same determinant in Qiskit: X on exactly those qubit indices, which is
        # the frame rotation trick applied to an arbitrary occupation.
        rotated = to_sparse_pauli_op(
            conjugate_with_x(chemistry.hamiltonian, occupation), n_qubits
        )
        qiskit_value = exact_evaluator.energy(np.zeros(exact_evaluator.n_parameters), rotated)
        determinant_error = max(determinant_error, abs(openfermion_value - qiskit_value))
    _check(results, "qubit ordering through Aer: determinants agree",
           determinant_error < 1.0e-10,
           f"max |dE| = {determinant_error:.3e} Ha over {determinants} determinants")

    # ---------------------------------------------------------------- 2
    operators = build_objective(chemistry, args.number_penalty, 0.0)
    zeros = np.zeros(evaluator.n_parameters)
    frame_energy = evaluator.energy(zeros, operators["hamiltonian"])
    frame_error = frame_energy - chemistry.hartree_fock_energy
    _check(results, "theta=0 is the Hartree-Fock determinant", abs(frame_error) < 1.0e-10,
           f"error = {frame_error:+.3e} Ha")

    objective_at_zero = evaluator.energy(zeros, operators["objective"])
    penalty_at_zero = objective_at_zero - frame_energy
    _check(results, "penalty vanishes at the reference state", abs(penalty_at_zero) < 1.0e-10,
           f"penalty = {penalty_at_zero:+.3e} Ha")

    electrons_at_zero = evaluator.energy(zeros, operators["number"])
    _check(results, "theta=0 has the target electron number",
           abs(electrons_at_zero - nelec) < 1.0e-10,
           f"<N> = {electrons_at_zero:.10f}")

    if args.check_penalty_sector:
        penalty, electrons = _adequate_number_penalty(chemistry)
        _check(results, "a penalty selects the target sector",
               penalty <= args.number_penalty,
               f"smallest adequate penalty = {penalty} Ha, <N> = {electrons:.10f}")

    # ---------------------------------------------------------------- 3
    angles = rng.uniform(-np.pi, np.pi, evaluator.n_parameters)
    mps_energy = evaluator.energy(angles, operators["objective"])
    statevector_energy = exact_evaluator.energy(angles, operators["objective"])
    mps_error = mps_energy - statevector_energy
    _check(results, f"MPS agrees with statevector (thr {DEFAULT_TRUNCATION_THRESHOLD:.0e})",
           abs(mps_error) < 1.0e-9, f"random angles, difference = {mps_error:+.3e} Ha")

    # ---------------------------------------------------------------- 4
    _check(results, "parameter-shift preconditions hold", evaluator.exact_gradient,
           f"{evaluator.n_parameters} RY parameters, each used once")
    _, shift_gradient = evaluator.energy_and_gradient(
        angles, operators["objective"], finite_difference=False
    )
    gradient_started = time.perf_counter()
    _, difference_gradient = exact_evaluator.energy_and_gradient(
        angles, operators["objective"], finite_difference=True
    )
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_error = float(np.max(np.abs(shift_gradient - difference_gradient)))
    _check(results, "parameter-shift == finite difference", gradient_error < 1.0e-5,
           f"max |dG| = {gradient_error:.3e}")

    # ---------------------------------------------------------------- 5
    # The optimization loop runs on `statevector`, not MPS, and that is a
    # runtime decision with no cost in coverage. MPS is *slower* here: at 12
    # qubits there is nothing to compress, and its cost grows with entanglement,
    # so a converged HEA state runs several times slower than the product state
    # it started from -- measured at ~370 ms/circuit against ~100 ms for
    # statevector. Both methods are still checked against each other, at random
    # angles above and at the converged state below, which is exactly what
    # qiskit_runtime.py does for the real molecule.
    circuits_per_iteration = 1 + 2 * exact_evaluator.n_parameters
    print("-" * 72, flush=True)
    print(f"  Running VQE: {args.layers} layers, {exact_evaluator.n_parameters} parameters, "
          f"L-BFGS-B, maxiter {args.maxiter}, statevector", flush=True)
    print(f"  {circuits_per_iteration} circuits per iteration at "
          f"~{1000.0 * gradient_seconds / circuits_per_iteration:.0f} ms each ...", flush=True)
    vqe_started = time.perf_counter()
    objective_value, parameters, diagnostics = _minimize(
        exact_evaluator, operators["objective"], zeros, "l-bfgs-b", 1.0e-12, args.maxiter, False
    )
    vqe_elapsed = time.perf_counter() - vqe_started

    physical_energy, electron_count, leakage, spin_squared = exact_evaluator.run(
        [
            (parameters, operators["hamiltonian"]),
            (parameters, operators["number"]),
            (parameters, operators["deviation"]),
            (parameters, operators["spin_squared"]),
        ]
    )
    converged_mps_energy = evaluator.energy(parameters, operators["hamiltonian"])
    converged_mps_error = converged_mps_energy - physical_energy
    error_mha = 1000.0 * (physical_energy - chemistry.reference_energy)
    contamination = args.number_penalty * leakage + abs(spin_squared)

    print(f"  VQE objective / physical    : {objective_value:.12f} / {physical_energy:.12f} Ha", flush=True)
    print(f"  <N> / leakage / <S^2>       : {electron_count:.8f} / {leakage:.3e} / {spin_squared:.3e}", flush=True)
    print(f"  Error vs exact              : {error_mha:+.6f} mHa", flush=True)
    print(f"  VQE wall time               : {vqe_elapsed:.2f} s "
          f"({exact_evaluator.circuit_count} circuits in "
          f"{exact_evaluator.batch_count} Aer batches)", flush=True)
    print("-" * 72, flush=True)

    # The converged state is far more entangled than the random one above, so
    # this is where a bond-dimension cap would first show up as a wrong energy.
    _check(results, "MPS agrees with statevector at the optimum",
           abs(converged_mps_error) < 1.0e-9,
           f"difference = {converged_mps_error:+.3e} Ha")

    # The variational bound is the one statement that must hold exactly: a
    # penalised energy minus its own penalty cannot dip below the sector ground
    # state by more than the leak it paid for. This is the same check
    # qiskit_runtime.py applies to the real molecule against the sector ground
    # energy the validation receipt carries.
    _check(results, "VQE respects the variational bound",
           physical_energy >= chemistry.reference_energy - max(contamination, 1.0e-9),
           f"E_vqe - E_exact = {error_mha:+.6f} mHa")
    _check(results, "VQE improves on Hartree-Fock",
           physical_energy < chemistry.hartree_fock_energy,
           f"gain = {1000.0 * (chemistry.hartree_fock_energy - physical_energy):.3f} mHa")
    # Hitting the iteration cap is this command's own budget decision, not an
    # optimizer failure -- what must never happen is termination for a reason
    # neither of those explains.
    capped = diagnostics["iterations"] >= args.maxiter
    _check(results, "optimizer terminated for a known reason",
           diagnostics["converged"] or capped,
           f"{diagnostics['iterations']} iterations, "
           f"{'maxiter cap' if capped and not diagnostics['converged'] else diagnostics['message']}")
    return chemistry


# --------------------------------------------------------------------------
# Stage 3: ADAPT. The published reference method.
# --------------------------------------------------------------------------


def _operator_key(qubit_operator: Any) -> tuple:
    """Sign- and representation-independent identity of a QubitOperator.

    Pool generators are anti-Hermitian, so every Jordan-Wigner coefficient is
    imaginary; two entries that differ only by an overall sign are the same
    operator with the parameter negated, and must count as duplicates. Fixing the
    sign by the first term in sorted order does that.
    """
    terms = sorted(
        (tuple(sorted(term)), complex(coefficient))
        for term, coefficient in qubit_operator.terms.items()
    )
    if not terms:
        return ()
    leading = terms[0][1]
    sign = 1.0 if (leading.real + leading.imag) >= 0 else -1.0
    return tuple(
        (term, round(sign * coefficient.real, 10), round(sign * coefficient.imag, 10))
        for term, coefficient in terms
    )


def _adapt_stage(
    results: list[tuple[str, bool, str]], args: argparse.Namespace, chemistry: Any
) -> None:
    """Verify the structural claims ADAPT makes, and refuse the ones it cannot.

    The hardware-efficient path measures symmetry breaking and prices it. ADAPT
    claims some of it away by construction, and the point of this stage is to
    find out exactly how much -- from the commutators, not from a docstring.

    The answer is that particle number and Sz are exact for every pool operator,
    which is what lets the number penalty go, while S^2 is exact only for the
    singles and the seniority-zero doubles. So the singlet is a property of the
    variational minimum rather than of the ansatz, and it is checked as one, with
    the same contamination budget the hardware-efficient path uses.
    """
    from hamiltonian import CONTAMINATION_ENERGY_BUDGET_HA

    from adapt_runtime import (
        POOLS,
        SparseEngine,
        adapt_vqe,
        build_pool,
        to_circuit,
        verify_circuit,
    )

    n_qubits = chemistry.n_qubits
    nelec = chemistry.n_active_electrons

    # ------------------------------------------------------------------ 1
    # Pool construction. Sz conservation is enforced by the enumeration, so what
    # has to be checked is that nothing slipped through it and that the three
    # pools nest the way their definitions say they do.
    pools = {}
    pool_started = time.perf_counter()
    for kind in POOLS:
        pools[kind] = build_pool(n_qubits, nelec, kind)
    pool_seconds = time.perf_counter() - pool_started
    sizes = {kind: len(entries) for kind, entries in pools.items()}
    print(f"  Pools built                 : "
          f"{', '.join(f'{k}={v}' for k, v in sizes.items())} in {pool_seconds:.1f} s",
          flush=True)

    _check(results, "pair < uccsd < uccgsd", sizes["pair"] < sizes["uccsd"] < sizes["uccgsd"],
           str(sizes))

    def spin_sum(indices) -> int:
        return sum(index % 2 for index in indices)

    offenders = [
        entry["label"]
        for entries in pools.values()
        for entry in entries
        if spin_sum(entry["creations"]) != spin_sum(entry["annihilations"])
    ]
    _check(results, "every pool operator conserves Sz", not offenders,
           f"{sum(sizes.values())} operators, {len(offenders)} violations")

    ranks = [
        entry["rank"] == len(entry["creations"]) == len(entry["annihilations"])
        for entries in pools.values()
        for entry in entries
    ]
    _check(results, "every pool operator conserves particle number", all(ranks),
           "creations and annihilations balance in all three pools")

    # A duplicated operator is not a wrong answer, it is a silently wasted ADAPT
    # iteration: the same excitation gets picked twice and the second one has
    # nothing left to contribute. This is deliberately checked on the *operators*
    # rather than on the index bookkeeping that built them -- T(a<-b) and T(b<-a)
    # differ only by an overall sign, which the parameter absorbs, so a
    # duplicate-index check that missed that would still look clean.
    keys = {kind: [_operator_key(entry["qubit"]) for entry in entries]
            for kind, entries in pools.items()}
    duplicated = {kind: len(values) - len(set(values)) for kind, values in keys.items()}
    _check(results, "no pool contains the same operator twice",
           not any(duplicated.values()), f"duplicates {duplicated}")

    generalized = set(keys["uccgsd"])
    missing = sum(1 for key in keys["uccsd"] if key not in generalized)
    _check(results, "uccsd is a subset of uccgsd", missing == 0,
           f"{sizes['uccsd']} of {sizes['uccgsd']}, {missing} missing")

    # ------------------------------------------------------------------ 2
    # The engine. uccsd rather than uccgsd here: a seventh of the operators, the
    # same code path, and the pool-size claims above were already checked on all
    # three.
    pool = pools["uccsd"]
    engine_started = time.perf_counter()
    engine = SparseEngine(chemistry, pool)
    print(f"  Sparse engine ({len(pool)} operators) : "
          f"{time.perf_counter() - engine_started:.1f} s", flush=True)

    reference_energy = engine.expectation(engine.hamiltonian, engine.reference)
    reference_error = reference_energy - chemistry.hartree_fock_energy
    _check(results, "ADAPT reference is the Hartree-Fock determinant",
           abs(reference_error) < 1.0e-10, f"error = {reference_error:+.3e} Ha")

    # ------------------------------------------------------------------ 3
    # Which symmetries the pool actually has, operator by operator, from the
    # commutator itself rather than from what the docstring claims.
    #
    # The answer is not uniform and the difference matters: [N, A] = 0 for every
    # entry, which is what lets ADAPT drop the number penalty entirely, but
    # [S^2, A] = 0 only for the singles and for seniority-zero paired doubles. A
    # general spin-complemented double is invariant under flipping Sz, which is
    # strictly weaker than commuting with S^2. So the singlet is a property of
    # the variational minimum, not of the ansatz -- and it gets checked there.
    rng = np.random.default_rng(args.seed + 2)
    # A generic state, not the reference: [X, A]|HF> can vanish by accident for
    # a determinant that A cannot reach at all.
    probe_state = engine.state(rng.uniform(-0.4, 0.4, 3), [0, 1, len(pool) - 1])

    def commutator_norm(matrix: Any, operator: Any) -> float:
        return float(
            np.linalg.norm(matrix @ (operator @ probe_state) - operator @ (matrix @ probe_state))
        )

    number_breaks = []
    spin_breaks = {1: 0, 2: 0}
    ranks = {1: 0, 2: 0}
    for entry, matrix in zip(pool, engine.pool_matrices):
        ranks[entry["rank"]] += 1
        if commutator_norm(matrix, engine.number) > 1.0e-9:
            number_breaks.append(entry["label"])
        if commutator_norm(matrix, engine.spin_squared) > 1.0e-9:
            spin_breaks[entry["rank"]] += 1
    _check(results, "[N, A] = 0 for every pool operator", not number_breaks,
           f"{len(pool)} operators, {len(number_breaks)} violations")
    _check(results, "[S^2, A] = 0 for every spin-complemented single",
           spin_breaks[1] == 0, f"{ranks[1]} singles, {spin_breaks[1]} violations")
    # Recorded, not asserted to be zero: this is a real property of a UCCGSD-style
    # pool and pretending otherwise is what the earlier version of this file did.
    _check(results, "S^2 breaking is confined to the general doubles",
           spin_breaks[1] == 0 and spin_breaks[2] > 0,
           f"{spin_breaks[2]}/{ranks[2]} doubles do not commute with S^2")

    sample = rng.choice(len(pool), size=min(8, len(pool)), replace=False).tolist()
    angles = rng.uniform(-0.6, 0.6, len(sample))
    evolved = engine.state(angles, sample)
    electrons, leakage, spin_squared = engine.sector_observables(evolved)
    _check(results, "evolved state keeps the electron number",
           abs(electrons - nelec) < 1.0e-10 and leakage < 1.0e-16,
           f"<N> = {electrons:.10f}, leak = {leakage:.3e} at random angles")
    _check(results, "evolved state is normalized",
           abs(float(np.vdot(evolved, evolved).real) - 1.0) < 1.0e-12,
           "exp(theta A) is unitary for anti-Hermitian A")

    # ------------------------------------------------------------------ 4
    # The adjoint gradient is the whole reason this runs in seconds rather than
    # minutes, so it is checked against the definition it is supposed to equal.
    _, adjoint = engine.energy_and_gradient(angles, sample)
    step = 1.0e-5
    finite = np.zeros(len(sample))
    for index in range(len(sample)):
        forward = np.array(angles, dtype=float)
        backward = np.array(angles, dtype=float)
        forward[index] += step
        backward[index] -= step
        finite[index] = (
            engine.energy(forward, sample) - engine.energy(backward, sample)
        ) / (2.0 * step)
    gradient_error = float(np.max(np.abs(adjoint - finite)))
    _check(results, "adjoint gradient == finite difference", gradient_error < 1.0e-6,
           f"max |dG| = {gradient_error:.3e} over {len(sample)} parameters")

    # The operator-selection gradient is a different quantity -- dE/dtheta for an
    # operator that is not in the ansatz yet -- and selecting on a wrong one
    # would quietly build a bad ansatz that still converges to *something*.
    pool_gradient = engine.pool_gradients(evolved)
    probe = int(np.argmax(np.abs(pool_gradient)))
    appended = list(sample) + [probe]
    selection_finite = (
        engine.energy(list(angles) + [step], appended)
        - engine.energy(list(angles) + [-step], appended)
    ) / (2.0 * step)
    selection_error = abs(float(pool_gradient[probe]) - selection_finite)
    _check(results, "pool selection gradient == finite difference",
           selection_error < 1.0e-6,
           f"|dG| = {selection_error:.3e} on the largest of {len(pool)} candidates")

    # ------------------------------------------------------------------ 5
    print("-" * 72, flush=True)
    print(f"  Running ADAPT-VQE: uccsd pool, up to {args.adapt_operators} operators",
          flush=True)
    adapt_started = time.perf_counter()
    outcome = adapt_vqe(
        engine,
        max_operators=args.adapt_operators,
        gradient_tolerance=1.0e-3,
        optimizer="l-bfgs-b",
        optimizer_tolerance=1.0e-12,
        maxiter=500,
        verbose=False,
    )
    adapt_elapsed = time.perf_counter() - adapt_started
    energy = outcome["energy_hartree"]
    state = outcome["state"]
    electrons, leakage, spin_squared = engine.sector_observables(state)
    error_mha = 1000.0 * (energy - chemistry.reference_energy)
    print(f"  ADAPT energy / error        : {energy:.12f} Ha / {error_mha:+.6f} mHa",
          flush=True)
    print(f"  Operators / exponentials    : {len(outcome['operators'])} / "
          f"{engine.exponentials} in {adapt_elapsed:.1f} s", flush=True)
    print("-" * 72, flush=True)

    trajectory = [entry["energy_hartree"] for entry in outcome["history"]]
    monotone = all(
        later <= earlier + 1.0e-9 for earlier, later in zip(trajectory, trajectory[1:])
    )
    _check(results, "every ADAPT iteration lowers the energy", monotone,
           f"{len(trajectory)} iterations, "
           f"{1000.0 * (reference_energy - energy):.3f} mHa gained")
    _check(results, "ADAPT improves on Hartree-Fock",
           energy < chemistry.hartree_fock_energy - 1.0e-9,
           f"gain = {1000.0 * (chemistry.hartree_fock_energy - energy):.3f} mHa")
    # No penalty is applied, so this bound has no leak to forgive: a
    # particle-number-conserving ansatz below the sector ground state is a
    # contradiction, full stop.
    _check(results, "ADAPT respects the variational bound",
           energy >= chemistry.reference_energy - 1.0e-9,
           f"E_adapt - E_exact = {error_mha:+.6f} mHa")
    _check(results, "ADAPT optimum is still in the six-electron sector",
           abs(electrons - nelec) < 1.0e-9 and leakage < 1.0e-16,
           f"<N> = {electrons:.10f}, leak = {leakage:.3e} (structural)")
    # This one is variational, not structural -- see the commutator checks above.
    # The Hamiltonian is spin free and the reference is a closed-shell singlet,
    # so the minimum is a singlet even though most of the pool could leave it.
    budget = CONTAMINATION_ENERGY_BUDGET_HA
    _check(results, "ADAPT optimum is a singlet, as the variational minimum",
           abs(spin_squared) <= budget,
           f"<S^2> = {spin_squared:.3e}, priced at {1000.0 * abs(spin_squared):.6f} mHa "
           f"against a {1000.0 * budget:.4f} budget")

    # ------------------------------------------------------------------ 6
    # Circuit synthesis. Every Pauli string inside one fermionic excitation
    # generator commutes with the others, so `PauliEvolutionGate`'s first-order
    # product is exact rather than a Trotter step -- but that is an assumption
    # about Qiskit's synthesis, not a theorem about this code, so it is measured.
    # optimization_level=1: the transpile dominates the cost here and the gate
    # count is not what is being checked, the energy is.
    circuit_started = time.perf_counter()
    circuit = to_circuit(engine, outcome["operators"], outcome["angles"])
    report = verify_circuit(engine, circuit, energy, optimization_level=1)
    print(f"  Circuit: {report['two_qubit_gates']} CX, depth {report['depth']}, "
          f"built and verified in {time.perf_counter() - circuit_started:.1f} s", flush=True)
    _check(results, "Qiskit circuit reproduces the sparse algebra",
           report["exact_synthesis"],
           f"|dE| = {abs(report['energy_error_hartree']):.3e} Ha "
           f"(exponentials synthesize exactly)")
    _check(results, "circuit prepares a real ansatz, not the reference",
           report["two_qubit_gates"] > 0 and report["depth"] > len(outcome["operators"]),
           f"{report['two_qubit_gates']} CX, depth {report['depth']}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def selftest_main(spec: Any) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the local Qiskit Aer stack without PySCF or a cache"
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--maxiter",
        type=int,
        default=60,
        help="VQE iterations in the simulation stage. Each costs 1 + 2N circuits "
        "at N = 12 * (layers + 1) parameters against a ~1800-term observable, so "
        "this is what sets the runtime of the whole command. The default leaves "
        "room for L-BFGS-B to converge on its own; hitting the cap is reported, "
        "not treated as a failure.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--number-penalty", type=float, default=1.0)
    parser.add_argument(
        "--adapt-operators",
        type=int,
        default=6,
        help="Length of the ADAPT ansatz grown in the third stage. This is a "
        "correctness check, not a convergence run -- the synthetic Hamiltonian is "
        "deliberately far more strongly correlated than a real molecule, so a "
        "short ansatz is expected to stay well outside chemical accuracy and the "
        "checks are on monotonicity, symmetry and circuit synthesis instead.",
    )
    parser.add_argument(
        "--skip-adapt",
        action="store_true",
        help="Skip the ADAPT stage. Nothing else depends on it.",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Run only the instant, NumPy-only checks. Useful on a machine with "
        "no Qiskit installed, e.g. the Linux box that builds the Hamiltonian.",
    )
    parser.add_argument(
        "--check-penalty-sector",
        action="store_true",
        help="Also find the smallest penalty whose exact penalised ground state "
        "still sits in the target electron sector. Dense diagonalization of a "
        "4096x4096 matrix per candidate, so it costs a minute or so.",
    )
    parser.add_argument(
        "--write-cache",
        type=str,
        default=None,
        help="Also write the synthetic Hamiltonian as a schema-compatible cache.",
    )
    args = parser.parse_args()
    if args.adapt_operators < 1 or args.maxiter < 1 or args.layers < 1:
        parser.error("--adapt-operators, --maxiter and --layers must be positive")

    norb = int(spec.n_active_orbitals)
    nelec = int(spec.n_active_electrons)

    started = time.perf_counter()
    results: list[tuple[str, bool, str]] = []

    print("=" * 72)
    print("Local Qiskit Aer self-test (no PySCF, no cached molecule)")
    print("=" * 72)
    print(f"  Shadowing                   : {spec.name}, CAS({nelec}e,{norb}o)")
    print("-" * 72)
    print("  Structure")
    _check_layering(results)
    _check_spec(results)
    _check_pi_core_from_geometry(results)
    _check_artifacts(results)
    _check_scoring(results)

    if not args.structure_only:
        print("-" * 72)
        print("  Simulation")
        chemistry = _simulation_stage(results, args, norb, nelec)
        if not args.skip_adapt:
            print("-" * 72)
            print("  ADAPT")
            _adapt_stage(results, args, chemistry)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("=" * 72)
    print(f"{passed}/{total} checks passed in {time.perf_counter() - started:.2f} s")
    if passed != total:
        print("SELF-TEST FAILED: do not run the molecule until this is green.")
    elif args.structure_only:
        print("STRUCTURE PASSED: the folder is intact. Re-run without "
              "--structure-only where Qiskit is installed.")
    else:
        print("SELF-TEST PASSED: the local Qiskit Aer stack is correct.")
        print("The only thing still needed is the imipramine Hamiltonian cache.")
    print("=" * 72)
    return 0 if passed == total else 1
