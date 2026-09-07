"""Geometry -> Hartree-Fock -> AVAS -> cached active-space Hamiltonian.

The only module that imports PySCF.  Everything downstream consumes an
``ffsim.MolecularData``, which carries the active-space integrals, the core
energy, and (once ``reference.py`` has run) the CCSD amplitudes that seed the
LUCJ ansatz.

Why AVAS lands on (16e,16o) by construction
-------------------------------------------
The paper's Table I gives the AVAS target orbitals as ``C[2s,2p], H[1s]``.
For a methane dimer that is 2 carbons x 4 valence AOs + 8 hydrogens x 1 = 16
reference AOs, and the 20 electrons minus the two carbon 1s pairs leaves 16
valence electrons.  The active space is therefore *determined* by the AO list,
not tuned to it -- which is why this module asserts (16, 16) rather than
searching for it.  If the assertion fires, the SCF or the AO labels are wrong,
and no amount of threshold tuning is the right response.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geometry
import paper

CACHE_VERSION = 3


# --------------------------------------------------------------------------
# Dependency guard
# --------------------------------------------------------------------------

_INSTALL_HINT = """\
This step needs PySCF and ffsim, and neither publishes a Windows wheel --
they ship manylinux and macOS only. Checked against PyPI:

    ffsim  -> manylinux_2_28_x86_64, manylinux_2_28_aarch64, macosx_11_0_arm64
    pyscf  -> manylinux_2_17_x86_64, manylinux2014_aarch64, macosx_*

So on Windows this is not a missing-package problem that `pip install` fixes.
Run the chemistry on one of:

    * Google Colab      (see colab.ipynb in this folder)
    * WSL2              wsl --install, then pip install -r requirements.txt
    * any Linux/macOS host

Everything that does not touch chemistry still runs here:

    python run.py selftest      # full logic check, no chemistry packages
    python run.py plan          # cost model and the sample-size ladder
    python run.py geometry      # build and inspect structures, write XYZ
"""


def require_chemistry() -> None:
    """Fail early and legibly when the chemistry stack is unavailable."""
    missing = []
    for module in ("pyscf", "ffsim"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return
    header = f"missing required package(s): {', '.join(missing)}"
    if platform.system() == "Windows":
        raise SystemExit(f"{header}\n\n{_INSTALL_HINT}")
    raise SystemExit(f"{header}\n\nInstall with:  pip install -r requirements.txt")


# --------------------------------------------------------------------------
# Specification and fingerprinting
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemSpec:
    """Everything that determines the Hamiltonian.

    The fingerprint of this object keys the integral cache.  Orientation and
    C-H bond length are included because they are *our* choices, not the
    paper's -- a cache built under D3d must never be silently reused for D3h.
    """

    distance: float
    orientation: str = geometry.DEFAULT_ORIENTATION
    r_ch: float = geometry.DEFAULT_CH_BOND
    basis: str = paper.BASIS
    density_fit: bool = True
    avas_threshold: float = 0.2
    avas_minao: str = "minao"
    # Overridable so `run.py validate` can exercise the identical code path on
    # a deliberately tiny active space. Defaults reproduce the paper exactly.
    avas_ao_labels: tuple[str, ...] = paper.AVAS_AO_LABELS
    expect_active_space: tuple[int, int] | None = (
        paper.N_ELECTRONS,
        paper.N_ORBITALS,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": round(self.distance, 6),
            "orientation": self.orientation,
            "r_ch": round(self.r_ch, 6),
            "basis": self.basis,
            "density_fit": self.density_fit,
            "avas_threshold": self.avas_threshold,
            "avas_minao": self.avas_minao,
            "avas_ao_labels": list(self.avas_ao_labels),
            "expect_active_space": list(self.expect_active_space)
            if self.expect_active_space
            else None,
            "cache_version": CACHE_VERSION,
        }

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def label(self) -> str:
        return f"{self.orientation}_{self.distance:.3f}A_{self.fingerprint}"

    def atoms(self) -> list[geometry.Atom]:
        atoms = geometry.methane_dimer(
            self.distance, orientation_name=self.orientation, r_ch=self.r_ch
        )
        geometry.check_geometry(
            atoms, expected_distance=self.distance, expected_r_ch=self.r_ch
        )
        return atoms


def cache_path(spec: SystemSpec, root: Path) -> Path:
    return root / f"hamiltonian_{spec.label}.json.xz"


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def build_scf(spec: SystemSpec, *, verbose: int = 0):
    """Run RHF in the full basis.

    aug-cc-pVQZ on a methane dimer is 528 basis functions. Conventional
    four-index integrals at that size would be ~78 GB, so density fitting is on
    by default; pass ``density_fit=False`` to force the exact treatment if you
    have the machine for it. The active-space integrals extracted afterwards
    involve only 16 orbitals and are exact either way.
    """
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.Mole()
    mol.atom = geometry.to_pyscf_atom(spec.atoms())
    mol.basis = spec.basis
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False        # the 48 A reference breaks the point group
    mol.verbose = verbose
    mol.build()

    expected = geometry.n_electrons(spec.atoms())
    if mol.nelectron != expected:
        raise AssertionError(
            f"PySCF counts {mol.nelectron} electrons, geometry implies {expected}"
        )

    mf = pyscf.scf.RHF(mol)
    if spec.density_fit:
        mf = mf.density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            f"RHF did not converge at R = {spec.distance} A. A non-converged "
            "reference silently poisons every downstream number; fix it rather "
            "than proceeding."
        )
    return mf


def select_active_space(mf, spec: SystemSpec):
    """AVAS on the spec's AO labels. Returns ``(norb, n_active_electrons, mo)``.

    ``pyscf.mcscf.avas.avas`` is an alias for ``kernel`` and returns exactly
    three values, with ``nelecas`` an ``int`` (not a tuple).
    """
    from pyscf.mcscf import avas

    norb, ne_act, orbitals = avas.avas(
        mf,
        list(spec.avas_ao_labels),
        threshold=spec.avas_threshold,
        minao=spec.avas_minao,
        canonicalize=True,
    )

    if spec.expect_active_space is not None:
        expected_e, expected_o = spec.expect_active_space
        if (ne_act, norb) != (expected_e, expected_o):
            raise AssertionError(
                f"AVAS produced ({ne_act}e,{norb}o); expected "
                f"({expected_e}e,{expected_o}o).\n"
                f"AO labels used: {list(spec.avas_ao_labels)}\n"
                "For a methane dimer, C[2s,2p] + H[1s] spans exactly 16 reference "
                "AOs and 16 valence electrons, so a mismatch means the SCF or the "
                "AO labels are wrong -- not that the threshold needs tuning."
            )
    return norb, ne_act, orbitals


def active_space_hamiltonian(spec: SystemSpec, *, verbose: int = 0):
    """Full pipeline: geometry -> RHF -> AVAS -> ffsim.MolecularData."""
    import ffsim

    mf = build_scf(spec, verbose=verbose)
    norb, ne_act, orbitals = select_active_space(mf, spec)

    n_core = (mf.mol.nelectron - ne_act) // 2
    mf.mo_coeff = orbitals
    active_space = range(n_core, n_core + norb)

    # AVAS returns mo = [frozen | inactive-occupied | active | virtual], and
    # orders the active block occupied-first. mf.mo_occ is NOT permuted by
    # AVAS, so it stays valid only because that ordering keeps every occupied
    # orbital below every virtual one. ffsim.MolecularData.from_scf derives the
    # electron count from mo_occ[active_space], so if that ever stopped holding
    # the active space would be silently wrong. Check it rather than assume it.
    occupancy = float(sum(mf.mo_occ[list(active_space)]))
    if abs(occupancy - ne_act) > 1e-9:
        raise AssertionError(
            f"mo_occ over the active window sums to {occupancy}, but AVAS "
            f"reports {ne_act} active electrons. The AVAS orbital ordering no "
            "longer matches mo_occ; the active space would be silently wrong."
        )

    mol_data = ffsim.MolecularData.from_scf(mf, active_space=active_space)

    if mol_data.norb != norb:
        raise AssertionError(f"norb is {mol_data.norb}, expected {norb}")
    if sum(mol_data.nelec) != ne_act:
        raise AssertionError(
            f"nelec is {mol_data.nelec} ({sum(mol_data.nelec)} electrons), "
            f"expected {ne_act}"
        )
    return mol_data


def load_or_build(
    spec: SystemSpec, root: Path, *, rebuild: bool = False, verbose: int = 0
):
    """Cached ``active_space_hamiltonian``.

    The cache key is the spec fingerprint, so changing the orientation, the
    bond length, the basis or the AVAS settings produces a different file
    rather than a stale hit.
    """
    import ffsim

    root.mkdir(parents=True, exist_ok=True)
    path = cache_path(spec, root)

    if path.exists() and not rebuild:
        return ffsim.MolecularData.from_json(str(path), compression="lzma")

    mol_data = active_space_hamiltonian(spec, verbose=verbose)
    mol_data.to_json(str(path), compression="lzma")
    return mol_data


def environment_report() -> dict[str, Any]:
    """Versions and platform, recorded in every result file for provenance."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module in ("numpy", "scipy", "pyscf", "ffsim", "qiskit", "qiskit_addon_sqd"):
        try:
            report[module] = __import__(module).__version__
        except Exception:
            report[module] = None
    return report
