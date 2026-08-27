"""Methane dimer geometries.

The paper fixes the C-C distance grid and states that the monomers are held
rigid, but publishes no coordinates and never names the relative orientation of
the two units.  See ``paper.GEOMETRY_GAP``.

This module makes the missing choice explicit and testable instead of implicit
and buried.  Three stationary-point orientations are implemented; ``run.py``
refuses to build a geometry unless one is named; and the choice is stamped into
the integral cache fingerprint, so a cache built under one orientation can
never be silently reused under another.

Frames
------
Both monomers are tetrahedral and rigid.  The intermolecular axis is +z, with
monomer A centred at the origin and monomer B centred at ``(0, 0, R)``.
Monomer B is always the mirror image of A through the plane ``z = R/2``,
optionally twisted about z.  That construction guarantees, by symmetry rather
than by arithmetic, that the two monomers are congruent at every R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Experimental equilibrium C-H bond length of methane, r_e = 1.0870 A.
# The paper says only "the equilibrium geometries"; this is ours, and is a flag.
DEFAULT_CH_BOND = 1.0870

Atom = tuple[str, float, float, float]


# --------------------------------------------------------------------------
# Monomer frames
# --------------------------------------------------------------------------

def _tetrahedron_c2_frame(r_ch: float) -> np.ndarray:
    """CH4 with a C2 axis along z.

    The canonical alternating-corners-of-a-cube tetrahedron.  Two hydrogens sit
    above the xy-plane and two below, so z is a two-fold axis.
    """
    directions = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )
    return directions / math.sqrt(3.0) * r_ch


def _tetrahedron_c3_frame(r_ch: float) -> np.ndarray:
    """CH4 with a C3 axis along z: one apex H on +z, a tripod below.

    The tripod sits at the tetrahedral angle arccos(-1/3) = 109.4712 deg from
    +z, at azimuths 0, 120 and 240 degrees.
    """
    cos_theta = -1.0 / 3.0
    sin_theta = math.sqrt(1.0 - cos_theta**2)   # = 2*sqrt(2)/3
    atoms = [[0.0, 0.0, 1.0]]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        atoms.append([sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta])
    return np.array(atoms) * r_ch


def _rotate_z(coords: np.ndarray, degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return coords @ rot.T


def _mirror_z(coords: np.ndarray, plane_z: float) -> np.ndarray:
    """Reflect through the plane z = plane_z."""
    out = coords.copy()
    out[:, 2] = 2.0 * plane_z - out[:, 2]
    return out


# --------------------------------------------------------------------------
# Orientations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Orientation:
    name: str
    point_group: str
    frame: str          # "c3" or "c2"
    twist_degrees: float
    description: str
    is_default: bool = field(default=False)


ORIENTATIONS: dict[str, Orientation] = {
    "d3d": Orientation(
        name="d3d",
        point_group="D3d",
        frame="c3",
        twist_degrees=60.0,
        description=(
            "C3 axes collinear with the intermolecular axis, one H of each monomer "
            "pointing at the other, staggered by 60 deg. Generally reported as the "
            "global minimum of the methane dimer, and consistent with the paper "
            "singling out 3.638 A as its extra distance."
        ),
        is_default=True,
    ),
    "d3h": Orientation(
        name="d3h",
        point_group="D3h",
        frame="c3",
        twist_degrees=0.0,
        description=(
            "As D3d but eclipsed. A low-lying saddle point; included so the "
            "sensitivity of the binding well to the twist angle can be measured "
            "rather than assumed."
        ),
    ),
    "d2d": Orientation(
        name="d2d",
        point_group="D2d",
        frame="c2",
        twist_degrees=90.0,
        description=(
            "C2 axes along the intermolecular axis, twisted 90 deg: an edge-on "
            "contact presenting two H atoms from each monomer rather than one."
        ),
    ),
}

DEFAULT_ORIENTATION = "d3d"


def orientation(name: str) -> Orientation:
    key = name.strip().lower()
    if key not in ORIENTATIONS:
        raise KeyError(
            f"unknown orientation {name!r}; choose from {sorted(ORIENTATIONS)}"
        )
    return ORIENTATIONS[key]


# --------------------------------------------------------------------------
# Dimer construction
# --------------------------------------------------------------------------

def methane_dimer(
    distance: float,
    *,
    orientation_name: str = DEFAULT_ORIENTATION,
    r_ch: float = DEFAULT_CH_BOND,
) -> list[Atom]:
    """Build a rigid methane dimer at a given C-C separation.

    Args:
        distance: C-C separation in Angstrom. This is the paper's PES coordinate.
        orientation_name: one of ``ORIENTATIONS``.
        r_ch: monomer C-H bond length in Angstrom, held fixed across the grid.

    Returns:
        Ten ``(symbol, x, y, z)`` tuples: C, 4H for monomer A, then C, 4H for B.
    """
    if distance <= 0.0:
        raise ValueError(f"C-C distance must be positive, got {distance}")
    if r_ch <= 0.0:
        raise ValueError(f"C-H bond length must be positive, got {r_ch}")

    orient = orientation(orientation_name)
    hydrogens = (
        _tetrahedron_c3_frame(r_ch) if orient.frame == "c3" else _tetrahedron_c2_frame(r_ch)
    )

    carbon_a = np.zeros(3)
    hydrogens_a = hydrogens

    # B is A mirrored through the midplane, then twisted about the axis.  The
    # mirror is what makes the apex hydrogens face each other.
    carbon_b = np.array([0.0, 0.0, distance])
    hydrogens_b = _mirror_z(hydrogens + carbon_a, distance / 2.0)
    hydrogens_b = _rotate_z(hydrogens_b - carbon_b, orient.twist_degrees) + carbon_b

    atoms: list[Atom] = [("C", *carbon_a.tolist())]
    atoms += [("H", *row.tolist()) for row in hydrogens_a]
    atoms += [("C", *carbon_b.tolist())]
    atoms += [("H", *row.tolist()) for row in hydrogens_b]
    return atoms


def to_pyscf_atom(atoms: list[Atom]) -> str:
    """Render to the ``atom=`` string PySCF's Mole accepts."""
    return "; ".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, x, y, z in atoms)


def to_xyz(atoms: list[Atom], comment: str = "") -> str:
    """Render to standard XYZ, so geometries can be inspected in any viewer."""
    lines = [str(len(atoms)), comment]
    lines += [f"{sym:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}" for sym, x, y, z in atoms]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------

def carbon_positions(atoms: list[Atom]) -> np.ndarray:
    return np.array([[x, y, z] for sym, x, y, z in atoms if sym == "C"])


def carbon_carbon_distance(atoms: list[Atom]) -> float:
    carbons = carbon_positions(atoms)
    if carbons.shape[0] != 2:
        raise ValueError(f"expected exactly 2 carbons, found {carbons.shape[0]}")
    return float(np.linalg.norm(carbons[1] - carbons[0]))


def ch_bond_lengths(atoms: list[Atom]) -> np.ndarray:
    """All C-H distances, each hydrogen assigned to its nearest carbon."""
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    symbols = [sym for sym, *_ in atoms]
    carbons = np.array([c for c, s in zip(coords, symbols) if s == "C"])
    lengths = []
    for position, symbol in zip(coords, symbols):
        if symbol != "H":
            continue
        lengths.append(float(np.min(np.linalg.norm(carbons - position, axis=1))))
    return np.array(sorted(lengths))


def check_geometry(
    atoms: list[Atom],
    *,
    expected_distance: float,
    expected_r_ch: float = DEFAULT_CH_BOND,
    tolerance: float = 1e-8,
) -> None:
    """Assert the structural invariants the paper's protocol demands.

    Raises ``AssertionError`` with a specific message on any violation.  Called
    for every geometry before an SCF is spent on it -- a malformed structure is
    much cheaper to catch here than three hours into a scan.
    """
    if len(atoms) != 10:
        raise AssertionError(f"methane dimer must have 10 atoms, got {len(atoms)}")

    symbols = [sym for sym, *_ in atoms]
    if symbols.count("C") != 2 or symbols.count("H") != 8:
        raise AssertionError(f"expected C2H8, got {symbols.count('C')}C {symbols.count('H')}H")

    actual = carbon_carbon_distance(atoms)
    if abs(actual - expected_distance) > tolerance:
        raise AssertionError(
            f"C-C distance is {actual:.10f} A, expected {expected_distance:.10f} A"
        )

    lengths = ch_bond_lengths(atoms)
    if len(lengths) != 8:
        raise AssertionError(f"expected 8 C-H bonds, found {len(lengths)}")
    spread = float(lengths.max() - lengths.min())
    if spread > tolerance:
        raise AssertionError(
            f"monomers are not rigid: C-H lengths span {spread:.2e} A "
            f"({lengths.min():.8f} to {lengths.max():.8f})"
        )
    if abs(float(lengths.mean()) - expected_r_ch) > tolerance:
        raise AssertionError(
            f"C-H length is {lengths.mean():.10f} A, expected {expected_r_ch:.10f} A"
        )


def n_electrons(atoms: list[Atom]) -> int:
    """Total electron count. Methane dimer: 2*6 + 8*1 = 20."""
    charges = {"H": 1, "C": 6}
    return sum(charges[sym] for sym, *_ in atoms)
