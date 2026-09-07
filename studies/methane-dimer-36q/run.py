#!/usr/bin/env python
"""The one interface. Every operation is ``python run.py <command>``.

    python run.py selftest                     # no chemistry packages needed
    python run.py plan                         # cost model before you commit
    python run.py geometry --distance 3.638    # build and inspect structures

    python run.py prepare   --distance 3.638   # RHF + AVAS -> cached Hamiltonian
    python run.py reference --distance 3.638   # CASCI / CCSD / CCSD(T)
    python run.py sqd       --distance 3.638   # the 36-qubit run
    python run.py ablation  --distance 3.638   # uniform-sampling control
    python run.py verify                       # the headline claim, pass/fail

    python run.py scan --rung extrapolation-low
    python run.py report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import binding          # noqa: E402
import geometry         # noqa: E402
import paper            # noqa: E402
import spaces           # noqa: E402
import validation       # noqa: E402

DATA = HERE / "data"
CACHE = DATA / "cache"
RESULTS = DATA / "results"


# --------------------------------------------------------------------------
# Commands that need no chemistry stack
# --------------------------------------------------------------------------

def cmd_selftest(args) -> int:
    import selftest

    return 0 if selftest.run(verbose=not args.quiet) else 1


def cmd_plan(args) -> int:
    print(f"{paper.TITLE}")
    print(f"  arXiv:{paper.ARXIV_ID}{paper.ARXIV_VERSION}  --  {paper.JOURNAL}")
    print(f"  {paper.URL}\n")
    print(spaces.describe_scale())

    print("\ncost ladder")
    print(f"  {'rung':<20} {'|chi_b|':>8} {'K':>3}  {'max_dim':>12}  published")
    for r in spaces.COST_LADDER:
        cap = f"{r.max_dim:,}" if r.max_dim else "-"
        print(
            f"  {r.name:<20} {r.samples_per_batch:>8,} {r.n_batches:>3}  "
            f"{cap:>12}  {'yes' if r.published else 'no':<3}"
        )
        print(f"  {'':<20} {r.note}")

    print("\nclassical references")
    dimension = spaces.full_cas_dimension()
    casci = spaces.SubspaceCost(dimension=dimension, n_batches=1, davidson_vectors=10)
    print(
        f"  CASCI(16e,16o)  {dimension:,} determinants, "
        f"~{spaces.humanize_bytes(casci.batch_peak_bytes)} peak"
    )
    print("  CCSD / CCSD(T) in the active space: cheap (16 orbitals).")
    print("  CCSD(T) in the full aug-cc-pVQZ basis: 528 basis functions, 10 occupied.")
    print("    That is an HPC calculation, not a workstation one. It quantifies the")
    print("    active-space approximation, not SQD accuracy, and is opt-in here.")

    print("\ngeometry")
    print(f"  orientation is NOT published. Default: {geometry.DEFAULT_ORIENTATION}")
    for name, orient in sorted(geometry.ORIENTATIONS.items()):
        flag = "  <- default" if orient.is_default else ""
        print(f"    {name:<5} {orient.point_group:<5}{flag}")
    return 0


def cmd_geometry(args) -> int:
    distances = [args.distance] if args.distance else list(paper.FULL_TREATMENT_DISTANCES)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    orient = geometry.orientation(args.orientation)
    print(f"orientation {orient.name} ({orient.point_group})")
    print(f"  {orient.description}\n")

    for distance in distances:
        atoms = geometry.methane_dimer(
            distance, orientation_name=args.orientation, r_ch=args.r_ch
        )
        geometry.check_geometry(
            atoms, expected_distance=distance, expected_r_ch=args.r_ch
        )
        comment = (
            f"methane dimer {orient.point_group} "
            f"R(C-C)={distance:.3f} A r(C-H)={args.r_ch:.4f} A"
        )
        if out_dir:
            path = out_dir / f"dimer_{orient.name}_{distance:.3f}.xyz"
            path.write_text(geometry.to_xyz(atoms, comment), encoding="utf-8")
            print(f"  wrote {path}")
        elif args.distance:
            print(geometry.to_xyz(atoms, comment))
        else:
            print(f"  R = {distance:>7.3f} A   checks passed")

    if not args.distance and not out_dir:
        print(f"\n{len(distances)} geometries validated.")
    return 0


# --------------------------------------------------------------------------
# Commands that need PySCF / ffsim
# --------------------------------------------------------------------------

def _spec(args, distance: float):
    import chemistry

    return chemistry.SystemSpec(
        distance=distance,
        orientation=args.orientation,
        r_ch=args.r_ch,
        basis=args.basis,
        density_fit=not args.no_density_fit,
    )


def cmd_prepare(args) -> int:
    import chemistry

    chemistry.require_chemistry()
    for distance in _distances(args):
        spec = _spec(args, distance)
        print(f"R = {distance:.3f} A  [{spec.fingerprint}]")
        mol_data = chemistry.load_or_build(
            spec, CACHE, rebuild=args.rebuild, verbose=args.verbose
        )
        print(
            f"  HF = {mol_data.hf_energy:.10f} Ha   "
            f"active space ({sum(mol_data.nelec)}e,{mol_data.norb}o)   "
            f"{spaces.n_determinants(mol_data.norb, mol_data.nelec):,} determinants"
        )
    return 0


def cmd_reference(args) -> int:
    import chemistry
    import reference

    chemistry.require_chemistry()
    for distance in _distances(args):
        spec = _spec(args, distance)
        mol_data = chemistry.load_or_build(spec, CACHE, verbose=args.verbose)
        result = reference.compute(
            mol_data,
            distance,
            want_casci=not args.no_casci,
            want_ccsd_t=not args.no_ccsd_t,
            verbose=args.verbose,
        )
        print(reference.summarize(result))
        validation.save_result(
            {"kind": "reference", "spec": spec.to_dict(), **result.to_dict()},
            RESULTS / f"reference_{spec.label}.json",
        )
    return 0


def cmd_sqd(args) -> int:
    import chemistry
    import ansatz
    import reference
    import sampling
    import sqd

    chemistry.require_chemistry()
    rung = spaces.rung(args.rung)
    config = sqd.SqdConfig(
        samples_per_batch=args.samples_per_batch or rung.samples_per_batch,
        n_batches=args.batches or rung.n_batches,
        max_iterations=args.max_iterations,
        max_dim=args.max_dim if args.max_dim else rung.max_dim,
        seed=args.seed,
    )

    for distance in _distances(args):
        spec = _spec(args, distance)
        print(f"\nR = {distance:.3f} A   rung={rung.name}   [{spec.fingerprint}]")

        mol_data = chemistry.load_or_build(spec, CACHE, verbose=args.verbose)
        reference.run_ccsd(mol_data, store_amplitudes=True)

        layout = ansatz.heavy_hex_layout(mol_data.norb, n_reps=args.n_reps)
        ansatz.validate_layout(layout)
        operator = ansatz.build_operator(mol_data, layout, optimize=args.optimize_ansatz)
        circuit = ansatz.build_circuit(mol_data, operator)
        print(f"  circuit: {layout.n_qubits_total} qubits "
              f"({layout.n_occupation_qubits} + {layout.n_ancilla_qubits} ancilla)")

        shots = args.shots or paper.TOTAL_SAMPLES
        sampler = sampling.NoiselessSampler(mol_data.norb, mol_data.nelec, seed=args.seed)
        sampled = sampler.sample(circuit, shots)
        validity = sampling.valid_configuration_fraction(
            sampled.bit_array, mol_data.norb, mol_data.nelec
        )
        print(f"  sampled {shots:,} shots, {validity:.2%} valid "
              f"(random baseline "
              f"{sampling.random_validity_probability(mol_data.norb, mol_data.nelec):.2e})")

        result = sqd.run(mol_data, sampled.bit_array, config,
                         source=sampled.source, verbose=True)
        print(f"  SQD  E = {result.energy:.10f} Ha   "
              f"d = {result.subspace_dimension:,} "
              f"({result.subspace_fraction:.2%} of CAS)")

        payload = {
            "kind": "sqd",
            "spec": spec.to_dict(),
            "rung": rung.name,
            "layout": layout.to_dict(),
            "sampling": sampled.metadata | {"valid_fraction": validity, "shots": shots},
            **result.to_dict(),
        }
        validation.save_result(
            payload, RESULTS / f"sqd_{rung.name}_{spec.label}.json"
        )
    return 0


def cmd_ablation(args) -> int:
    """Uniform-sampling control at matched subspace dimension."""
    import chemistry
    import sampling
    import sqd

    chemistry.require_chemistry()
    rung = spaces.rung(args.rung)
    for distance in _distances(args):
        spec = _spec(args, distance)
        mol_data = chemistry.load_or_build(spec, CACHE, verbose=args.verbose)
        config = sqd.SqdConfig(
            samples_per_batch=args.samples_per_batch or rung.samples_per_batch,
            n_batches=args.batches or rung.n_batches,
            max_iterations=args.max_iterations,
            max_dim=args.max_dim if args.max_dim else rung.max_dim,
            seed=args.seed,
        )
        sampler = sampling.UniformSampler(mol_data.norb, mol_data.nelec, seed=args.seed)
        sampled = sampler.sample(shots=args.shots or paper.TOTAL_SAMPLES)
        result = sqd.run(mol_data, sampled.bit_array, config,
                         source="uniform", verbose=True)
        print(f"  ablation E = {result.energy:.10f} Ha   "
              f"d = {result.subspace_dimension:,}")
        validation.save_result(
            {"kind": "ablation", "spec": spec.to_dict(), "rung": rung.name,
             **result.to_dict()},
            RESULTS / f"ablation_{rung.name}_{spec.label}.json",
        )
    return 0


def cmd_verify(args) -> int:
    """The headline claim: SQD agrees with CASCI to 0.010 kcal/mol at 3.638 A."""
    results = validation.load_all(RESULTS)
    sqd_runs = {r["spec"]["distance"]: r for r in results if r.get("kind") == "sqd"}
    references = {r["spec"]["distance"]: r for r in results if r.get("kind") == "reference"}

    distance = paper.SQD_VS_CASCI_TARGET_DISTANCE
    if distance not in sqd_runs or distance not in references:
        print(f"nothing to verify at {distance} A yet. Run:")
        print(f"  python run.py reference --distance {distance}")
        print(f"  python run.py sqd       --distance {distance} --rung converged")
        return 1

    casci = references[distance].get("casci")
    if casci is None:
        print("reference exists but has no CASCI energy; rerun without --no-casci")
        return 1

    computed = sqd_runs[distance]["energy"]
    binding.variational_check(computed, casci)
    agreement = binding.Agreement(computed, casci)

    print(f"R = {distance} A")
    print(f"  CASCI  {casci:.10f} Ha")
    print(f"  SQD    {computed:.10f} Ha")
    print(f"  {agreement}")
    print(f"  subspace {sqd_runs[distance]['subspace_dimension']:,} "
          f"({sqd_runs[distance]['subspace_fraction']:.2%} of CAS)")

    target = paper.SQD_VS_CASCI_TARGET_KCAL
    passed = agreement.meets(target)
    print(f"\npaper's claim: agreement within {target} kcal/mol at |chi_b| = 20e3")
    print(f"reproduced:    {'YES' if passed else 'NO'}")
    return 0 if passed else 1


def cmd_validate(args) -> int:
    """End-to-end chemistry check on a tiny active space. Seconds, not hours."""
    import validate

    return 0 if validate.run(CACHE / "validation", verbose=args.verbose) else 1


def cmd_report(args) -> int:
    import report

    return report.generate(RESULTS, HERE / "report.md", HERE / "report.html")


# --------------------------------------------------------------------------

def _distances(args) -> list[float]:
    if args.distance is not None:
        return [args.distance]
    if getattr(args, "all", False):
        return list(paper.FULL_TREATMENT_DISTANCES)
    return [paper.EQUILIBRIUM_DISTANCE, paper.UNBOUND_DISTANCE]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, *, chemistry_flags: bool = True):
        p.add_argument("--distance", type=float, default=None,
                       help="C-C separation in Angstrom")
        p.add_argument("--all", action="store_true",
                       help="every distance in the published grid")
        p.add_argument("--orientation", default=geometry.DEFAULT_ORIENTATION,
                       choices=sorted(geometry.ORIENTATIONS),
                       help="dimer orientation (NOT published; see README)")
        p.add_argument("--r-ch", type=float, default=geometry.DEFAULT_CH_BOND,
                       dest="r_ch", help="monomer C-H bond length in Angstrom")
        if chemistry_flags:
            p.add_argument("--basis", default=paper.BASIS)
            p.add_argument("--no-density-fit", action="store_true",
                           help="exact four-index integrals; needs a large machine")
            p.add_argument("--verbose", type=int, default=0)

    p = sub.add_parser("selftest", help="verify the logic without chemistry packages")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("plan", help="cost model, ladder, and unpublished choices")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("geometry", help="build, validate and export structures")
    add_common(p, chemistry_flags=False)
    p.add_argument("--out", default=None, help="directory for XYZ files")
    p.set_defaults(func=cmd_geometry)

    p = sub.add_parser("prepare", help="RHF + AVAS -> cached active-space Hamiltonian")
    add_common(p)
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("reference", help="CASCI / CCSD / CCSD(T) in the active space")
    add_common(p)
    p.add_argument("--no-casci", action="store_true",
                   help="skip the 165M-determinant exact solve")
    p.add_argument("--no-ccsd-t", action="store_true")
    p.set_defaults(func=cmd_reference)

    for name, func, helptext in (
        ("sqd", cmd_sqd, "the 36-qubit SQD run"),
        ("ablation", cmd_ablation, "uniform-sampling control at matched dimension"),
    ):
        p = sub.add_parser(name, help=helptext)
        add_common(p)
        p.add_argument("--rung", default="extrapolation-low",
                       choices=[r.name for r in spaces.COST_LADDER])
        p.add_argument("--samples-per-batch", type=int, default=None)
        p.add_argument("--batches", type=int, default=None)
        p.add_argument("--max-iterations", type=int, default=paper.RECOVERY_STEPS)
        p.add_argument("--max-dim", type=int, default=None)
        p.add_argument("--shots", type=int, default=None)
        p.add_argument("--n-reps", type=int, default=2, dest="n_reps")
        p.add_argument("--optimize-ansatz", action="store_true")
        p.add_argument("--seed", type=int, default=12345)
        p.set_defaults(func=func)

    p = sub.add_parser(
        "validate",
        help="end-to-end chemistry check on a tiny active space (run this first)",
    )
    p.add_argument("--verbose", type=int, default=0)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("verify", help="check the paper's headline claim, pass/fail")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("report", help="write report.md and report.html")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
