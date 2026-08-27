"""Where the configurations come from.

Three sources, one interface.  Each returns a Qiskit ``BitArray`` that feeds
straight into ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``,
so the SQD post-processing code is identical no matter which was used.  That
matters: it means the noiseless run and the hardware run differ in exactly one
object, and the ablation is a fair comparison rather than a separate code path.

    noiseless   ffsim's FfsimSampler. Exact sampling inside the (8,8) particle
                number sector -- 2.65 GB, not the 1.2 EB a dense 32-qubit
                simulator would need. No QPU, no queue, no noise.

    hardware    Qiskit Runtime SamplerV2 with gate twirling and dynamical
                decoupling, matching the paper's Methods section.

    uniform     Random particle-number-correct configurations. Not a way to
                compute anything -- it is the control that determines whether
                the quantum samples carried information. Run it at matched
                subspace dimension or the comparison means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

import paper


@dataclass(frozen=True)
class SamplingResult:
    bit_array: Any
    source: str
    shots: int
    metadata: dict[str, Any]


class Sampler(Protocol):
    def sample(self, circuit, shots: int) -> SamplingResult: ...


# --------------------------------------------------------------------------
# Noiseless -- ffsim
# --------------------------------------------------------------------------

class NoiselessSampler:
    """Exact fermionic sampling via ffsim.

    FfsimSampler implements Qiskit's SamplerV2 interface but simulates in the
    fermionic sector rather than the full qubit Hilbert space, which is the
    only reason 32 qubits is tractable here.
    """

    def __init__(self, norb: int, nelec: tuple[int, int], *, seed: int | None = None):
        self.norb = norb
        self.nelec = nelec
        self.seed = seed

    def sample(self, circuit, shots: int) -> SamplingResult:
        import ffsim

        sampler = ffsim.qiskit.FfsimSampler(
            default_shots=shots, norb=self.norb, nelec=self.nelec, seed=self.seed
        )
        result = sampler.run([circuit], shots=shots).result()[0]
        return SamplingResult(
            bit_array=result.data.meas,
            source="noiseless",
            shots=shots,
            metadata={"norb": self.norb, "nelec": list(self.nelec), "seed": self.seed},
        )


# --------------------------------------------------------------------------
# Hardware -- Qiskit Runtime
# --------------------------------------------------------------------------

class HardwareSampler:
    """Qiskit Runtime SamplerV2, configured as the paper's Methods describe.

    Gate twirling over random two-qubit Cliffords and dynamical decoupling are
    enabled; measurement twirling is explicitly *not*, matching the paper.
    """

    def __init__(self, backend, *, dd_sequence: str = "XY4"):
        self.backend = backend
        self.dd_sequence = dd_sequence

    def sample(self, circuit, shots: int) -> SamplingResult:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=self.backend)
        options = sampler.options
        options.dynamical_decoupling.enable = paper.DYNAMICAL_DECOUPLING
        options.dynamical_decoupling.sequence_type = self.dd_sequence
        options.twirling.enable_gates = paper.GATE_TWIRLING
        options.twirling.enable_measure = paper.MEASUREMENT_TWIRLING

        job = sampler.run([circuit], shots=shots)
        result = job.result()[0]
        return SamplingResult(
            bit_array=result.data.meas,
            source="hardware",
            shots=shots,
            metadata={
                "backend": getattr(self.backend, "name", str(self.backend)),
                "job_id": job.job_id(),
                "gate_twirling": paper.GATE_TWIRLING,
                "measurement_twirling": paper.MEASUREMENT_TWIRLING,
                "dynamical_decoupling": paper.DYNAMICAL_DECOUPLING,
                "dd_sequence": self.dd_sequence,
            },
        )


# --------------------------------------------------------------------------
# Uniform -- the ablation control
# --------------------------------------------------------------------------

class UniformSampler:
    """Random configurations with the correct particle number.

    This is the control that separates "the quantum layer contributed a better
    subspace" from "selected CI works". It bypasses the circuit entirely.
    """

    def __init__(self, norb: int, nelec: tuple[int, int], *, seed: int | None = None):
        self.norb = norb
        self.nelec = nelec
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def sample(self, circuit=None, shots: int = 1) -> SamplingResult:
        from qiskit.primitives.containers.bit_array import BitArray

        n_alpha, n_beta = self.nelec
        strings = [
            self._random_string(n_beta) + self._random_string(n_alpha)
            for _ in range(shots)
        ]
        return SamplingResult(
            bit_array=BitArray.from_samples(strings, num_bits=2 * self.norb),
            source="uniform",
            shots=shots,
            metadata={"norb": self.norb, "nelec": list(self.nelec), "seed": self.seed},
        )

    def _random_string(self, n_occupied: int) -> str:
        bits = np.zeros(self.norb, dtype=int)
        bits[self.rng.choice(self.norb, size=n_occupied, replace=False)] = 1
        return "".join(str(b) for b in bits[::-1])


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def valid_configuration_fraction(bit_array, norb: int, nelec: tuple[int, int]) -> float:
    """Fraction of raw shots with the correct alpha and beta electron counts.

    On a noiseless simulator this is exactly 1.0, because every gate in the
    LUCJ ansatz commutes with both number operators. On hardware it collapses.

    Interpret the number with care for *this* system. The often-quoted framing
    -- "2% valid on hardware versus 1e-6 at random, therefore quantum signal" --
    comes from the 52-qubit N2 case, which is far from half filling. The methane
    dimer is (8,8) in 16 orbitals, i.e. exactly half filled, and

        C(16,8)^2 / 2^32 = 3.86%

    of *random* 32-bit strings are already valid configurations. A hardware
    validity fraction of a few percent therefore demonstrates nothing here.
    Half filling is the worst case for this diagnostic.

    What does discriminate is the energy: run the uniform sampler at matched
    subspace dimension and compare. That is why `run.py ablation` exists and why
    the report leads with it rather than with this fraction.
    """
    array = bit_array.array
    counts = np.unpackbits(array.astype(np.uint8), axis=-1, bitorder="big")
    # Bits are packed MSB-first; the last 2*norb of them carry the register.
    counts = counts[..., -2 * norb:]
    beta = counts[..., : norb].sum(axis=-1)
    alpha = counts[..., norb:].sum(axis=-1)
    n_alpha, n_beta = nelec
    valid = np.logical_and(alpha == n_alpha, beta == n_beta)
    return float(valid.mean())


def random_validity_probability(norb: int, nelec: tuple[int, int]) -> float:
    """Probability a uniformly random 2*norb-bit string is a valid configuration.

    The baseline the hardware number should be compared against.
    """
    from math import comb

    n_alpha, n_beta = nelec
    return comb(norb, n_alpha) * comb(norb, n_beta) / float(1 << (2 * norb))
