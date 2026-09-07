"""Provenance: fingerprints, receipts, and result files.

A number in this folder is only worth as much as the record of how it was
produced.  Every result carries:

  * the physics fingerprint  -- a hash over the modules that define the
    Hamiltonian and the energetics. Change one of them and old results are
    visibly stale rather than quietly incomparable.
  * the workflow fingerprint -- a hash over every module, for full provenance.
  * the environment          -- package versions and platform.
  * the geometry choice      -- orientation and bond length, which the paper
    does not publish and which therefore must never be implicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Modules that determine the physics. Editing report.py or visualize.py does
# not invalidate a result; editing any of these does.
PHYSICS_MODULES = (
    "paper.py",
    "geometry.py",
    "spaces.py",
    "binding.py",
    "chemistry.py",
    "reference.py",
    "ansatz.py",
    "sqd.py",
)


def _hash_files(names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = HERE / name
        if not path.exists():
            continue
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def physics_fingerprint() -> str:
    return _hash_files(PHYSICS_MODULES)


def workflow_fingerprint() -> str:
    return _hash_files(tuple(sorted(p.name for p in HERE.glob("*.py"))))


@dataclass
class Receipt:
    """Provenance stamp attached to every result."""

    physics_sha256: str = field(default_factory=physics_fingerprint)
    workflow_sha256: str = field(default_factory=workflow_fingerprint)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matches_current_physics(self) -> bool:
        return self.physics_sha256 == physics_fingerprint()


def save_result(payload: dict[str, Any], path: Path) -> Path:
    """Write a result with a provenance receipt attached."""
    import chemistry

    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = Receipt(environment=chemistry.environment_report())
    document = {"receipt": receipt.to_dict(), **payload}
    path.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
    return path


def load_result(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    stamped = document.get("receipt", {}).get("physics_sha256")
    if stamped and stamped != physics_fingerprint():
        document["_stale"] = True
    return document


def load_all(root: Path, pattern: str = "*.json") -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [load_result(p) for p in sorted(root.glob(pattern))]


def check_invariants() -> list[tuple[str, bool, str]]:
    """Assertions that must hold before any number is trusted.

    Returns ``(name, passed, detail)`` triples. Pure arithmetic and geometry --
    no chemistry packages, so this runs anywhere.
    """
    import geometry
    import paper
    import spaces

    checks: list[tuple[str, bool, str]] = []

    dimension = spaces.full_cas_dimension()
    checks.append((
        "CAS(16e,16o) dimension",
        dimension == 165_636_900,
        f"{dimension:,} determinants (C(16,8)^2)",
    ))

    total = paper.N_QUBITS_OCCUPATION + paper.N_QUBITS_ANCILLA
    checks.append((
        "qubit budget",
        total == paper.N_QUBITS_TOTAL == 36,
        f"{paper.N_QUBITS_OCCUPATION} occupation + {paper.N_QUBITS_ANCILLA} ancilla = {total}",
    ))

    checks.append((
        "AVAS AO count matches active space",
        len(paper.AVAS_AO_LABELS) == 3 and paper.N_ORBITALS == 16,
        "C[2s,2p] + H[1s] spans 2*4 + 8*1 = 16 reference AOs",
    ))

    atoms = geometry.methane_dimer(paper.EQUILIBRIUM_DISTANCE)
    checks.append((
        "methane dimer electron count",
        geometry.n_electrons(atoms) == 20,
        f"{geometry.n_electrons(atoms)} electrons; 20 - 4 core = 16 active",
    ))

    grid_ok = (
        paper.UNBOUND_DISTANCE in paper.FULL_TREATMENT_DISTANCES
        and paper.EQUILIBRIUM_DISTANCE in paper.FULL_TREATMENT_DISTANCES
        and len(paper.FULL_TREATMENT_DISTANCES) == len(paper.PES_DISTANCES) + 2
    )
    checks.append((
        "distance grid",
        grid_ok,
        f"{len(paper.PES_DISTANCES)} PES points + equilibrium + unbound "
        f"= {len(paper.FULL_TREATMENT_DISTANCES)}",
    ))

    fraction = spaces.subspace_fraction(paper.SUBSPACE_DIMENSION)
    checks.append((
        "paper subspace fraction",
        0.70 < fraction < 0.80,
        f"d = {paper.SUBSPACE_DIMENSION:,} is {fraction:.1%} of the full CAS",
    ))

    return checks
