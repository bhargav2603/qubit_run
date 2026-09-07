"""Classical references: CASCI, CCSD, CCSD(T).

The paper benchmarks SQD(16e,16o) against CASCI(16e,16o) in the same active
space, and separately against CCSD/CCSD(T) in the full aug-cc-pVQZ basis.
Those two comparisons answer different questions and cost wildly different
amounts:

    CASCI(16e,16o)      exact in the active space. 165,636,900 determinants.
                        This is the SQD accuracy claim, and it is the one that
                        matters. Expensive but reachable.

    CCSD(T)/aug-cc-pVQZ exact-ish in the full basis. Quantifies the *active
                        space* approximation, not SQD. 528 basis functions and
                        10 occupied orbitals put this out of reach of a single
                        commodity node; the paper used HPC.

`run.py` therefore treats the active-space references as required and the
full-basis ones as opt-in, and says so rather than quietly skipping them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import binding
import spaces


@dataclass
class ReferenceEnergies:
    """Classical results for one geometry, all in hartree."""

    distance: float
    hf: float
    casci: float | None = None
    ccsd: float | None = None
    ccsd_t: float | None = None
    n_determinants: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def correlation_energy(self) -> float | None:
        if self.casci is None:
            return None
        return self.casci - self.hf


def casci_memory_estimate(norb: int, nelec: tuple[int, int], davidson_vectors: int = 10):
    """Peak memory for a direct CASCI at this size, before running it."""
    cost = spaces.SubspaceCost(
        dimension=spaces.n_determinants(norb, nelec),
        n_batches=1,
        davidson_vectors=davidson_vectors,
    )
    return cost.batch_peak_bytes


def run_ccsd(mol_data, *, store_amplitudes: bool = True) -> float:
    """CCSD in the active space; optionally keep t1/t2 to seed the LUCJ ansatz.

    ``MolecularData.scf`` round-trips the active-space integrals through an
    FCIDUMP, so this is a 16-orbital CCSD -- cheap -- not a 528-orbital one.
    """
    mol_data.run_ccsd(store_t1=store_amplitudes, store_t2=store_amplitudes)
    return float(mol_data.ccsd_energy)


def run_ccsd_t(mol_data) -> float:
    """CCSD(T) in the active space."""
    import pyscf.cc

    scf = mol_data.scf.run()
    ccsd = pyscf.cc.CCSD(scf).run()
    return float(ccsd.e_tot + ccsd.ccsd_t())


def pyscf_fci_minimum_memory_mb(norb: int, nelec: tuple[int, int]) -> float:
    """The threshold PySCF itself checks before choosing a Davidson path.

    ``direct_spin1.kernel_ms1`` compares ``max_memory`` against
    ``civec_size * 6 * 8e-6`` MB and silently drops to a slower, more
    conservative algorithm below it. PySCF's default ``max_memory`` is 4000 MB,
    while (16e,16o) needs about 7950 MB -- so leaving the default in place would
    quietly change the algorithm on exactly the calculation that matters.
    """
    return spaces.n_determinants(norb, nelec) * 6 * 8e-6


def run_casci(mol_data, *, max_memory_mb: int | None = None, verbose: int = 0) -> float:
    """Exact diagonalization in the active space -- the SQD reference.

    At (16e,16o) this is a 165,636,900-dimensional eigenproblem. A single CI
    vector is 961 MiB and Davidson holds a dozen of them, so budget ~15 GiB.
    The estimate is printed before the solve starts, because discovering the
    requirement by being OOM-killed three hours in is the expensive way to
    learn it.
    """
    import pyscf.fci

    norb, nelec = mol_data.norb, mol_data.nelec
    dimension = spaces.n_determinants(norb, nelec)
    estimate = casci_memory_estimate(norb, nelec)
    required_mb = pyscf_fci_minimum_memory_mb(norb, nelec)

    if max_memory_mb is None:
        # Beat PySCF's internal threshold with headroom, but never ask for less
        # than its own 4000 MB default.
        max_memory_mb = int(max(4000.0, required_mb * 1.5))

    if verbose:
        print(
            f"  CASCI({sum(nelec)}e,{norb}o): {dimension:,} determinants, "
            f"~{spaces.humanize_bytes(estimate)} peak, "
            f"max_memory = {max_memory_mb} MB "
            f"(PySCF needs > {required_mb:.0f} MB for the fast path)"
        )

    solver = pyscf.fci.direct_spin1.FCI()
    solver.verbose = verbose
    solver.max_memory = max_memory_mb

    hamiltonian = mol_data.hamiltonian
    energy, _ = solver.kernel(
        hamiltonian.one_body_tensor,
        hamiltonian.two_body_tensor,
        norb,
        nelec,
    )
    # solve_sci and direct_spin1 both return the active-space electronic energy
    # with ecore=0, so the core energy is added here exactly once. Verified
    # against qiskit_addon_sqd.fermion.solve_sci, which builds its energy purely
    # from RDMs contracted with the one- and two-body tensors.
    return float(energy + mol_data.core_energy)


def compute(
    mol_data,
    distance: float,
    *,
    want_casci: bool = True,
    want_ccsd_t: bool = True,
    verbose: int = 0,
) -> ReferenceEnergies:
    """All active-space references for one geometry."""
    result = ReferenceEnergies(
        distance=distance,
        hf=float(mol_data.hf_energy),
        n_determinants=spaces.n_determinants(mol_data.norb, mol_data.nelec),
    )

    result.ccsd = run_ccsd(mol_data, store_amplitudes=True)
    if want_ccsd_t:
        result.ccsd_t = run_ccsd_t(mol_data)
    if want_casci:
        result.casci = run_casci(mol_data, verbose=verbose)
        # CCSD is not variational, so it may dip below CASCI; SQD may not.
        # Only the exact reference is checked for sanity here.
        if result.casci > result.hf:
            raise AssertionError(
                f"CASCI ({result.casci:.8f}) lies above HF ({result.hf:.8f}); "
                "the active space or the integrals are wrong."
            )
    return result


def summarize(result: ReferenceEnergies) -> str:
    lines = [f"R(C-C) = {result.distance:.3f} A"]
    lines.append(f"  RHF          {result.hf:.10f} Ha")
    if result.ccsd is not None:
        lines.append(f"  CCSD  (as)   {result.ccsd:.10f} Ha")
    if result.ccsd_t is not None:
        lines.append(f"  CCSD(T) (as) {result.ccsd_t:.10f} Ha")
    if result.casci is not None:
        lines.append(
            f"  CASCI (as)   {result.casci:.10f} Ha   "
            f"E_corr = {result.correlation_energy:.10f} Ha "
            f"({binding.hartree_to_kcal(result.correlation_energy):.3f} kcal/mol)"
        )
        lines.append(f"  determinants {result.n_determinants:,}")
    return "\n".join(lines)
