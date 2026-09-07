#!/usr/bin/env python3
"""A structurally real active space that needs no quantum-chemistry package.

`selftest.py` has to prove the whole classical stack against exact answers on a
machine that may have neither PySCF nor a cached Hamiltonian. Random arrays will
not do: an `eri` without the eight-fold permutational symmetry of a real
two-electron integral tensor is not Hermitian in the determinant basis, and a
test that passes on it proves nothing about the code path a molecule takes.

So the integrals here are built the way a real density-fitted integral tensor is
built -- `eri[p,q,r,s] = sum_L B[L,p,q] B[L,r,s]` with each `B[L]` symmetric.
That gives, by construction rather than by patching:

* full eight-fold symmetry, (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq);
* a positive-semidefinite electron-electron interaction, so the spectrum is
  bounded below and the ground state is not an artefact;
* a spin-free Hamiltonian, so the exact ground state of a closed-shell sector is
  a genuine singlet and <S^2> is a meaningful check.

Ascending one-body orbital energies then make the aufbau determinant the sensible
reference, which is what HI-VQE's expansion step excites from.
"""

from __future__ import annotations

import numpy as np

from determinants import ActiveSpace


def synthetic_active_space(
    n_orbitals: int = 8,
    n_alpha: int = 4,
    n_beta: int = 4,
    seed: int = 0,
    coupling: float = 0.35,
    core_energy: float = -3.75,
    n_auxiliary: int | None = None,
) -> ActiveSpace:
    """An ActiveSpace with the symmetries of a real molecular Hamiltonian."""
    rng = np.random.default_rng(seed)
    m = n_orbitals
    auxiliary = m if n_auxiliary is None else n_auxiliary

    energies = np.linspace(-1.25, 1.35, m)
    hopping = rng.normal(scale=0.08, size=(m, m))
    h1e = np.diag(energies) + 0.5 * (hopping + hopping.T)

    factors = rng.normal(scale=1.0, size=(auxiliary, m, m))
    factors = 0.5 * (factors + factors.transpose(0, 2, 1))
    eri = coupling * np.einsum("Lpq,Lrs->pqrs", factors, factors) / auxiliary

    return ActiveSpace(
        n_orbitals=m,
        n_alpha=n_alpha,
        n_beta=n_beta,
        h1e=h1e,
        eri=eri,
        core_energy=core_energy,
    )


def check_eri_symmetry(eri: np.ndarray) -> dict[str, float]:
    """Residuals of the eight-fold permutational symmetry, for reporting."""
    return {
        "pq<->qp": float(np.abs(eri - eri.transpose(1, 0, 2, 3)).max()),
        "rs<->sr": float(np.abs(eri - eri.transpose(0, 1, 3, 2)).max()),
        "pq<->rs": float(np.abs(eri - eri.transpose(2, 3, 0, 1)).max()),
    }
