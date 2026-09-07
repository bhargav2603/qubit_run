#!/usr/bin/env python3
"""An exact simulator that replays a Qiskit circuit inside the number sector.

Aer simulates 24 qubits as 16,777,216 amplitudes. The HI-VQE ansatz conserves
the alpha and the beta electron count separately, so its state never leaves a
sector of only C(12,6)^2 = 853,776 amplitudes -- a factor of twenty smaller, and
already in exactly the (alpha string, beta string) layout `determinants.py`
uses. Replaying the circuit there instead costs about a second where Aer costs
eleven, which is the difference between a dissociation scan you can iterate on
and one you start before lunch.

This is a *reimplementation risk*, and it is handled by refusing to reimplement
anything. The circuit is a real `qiskit.QuantumCircuit`; this module reads each
gate's own ``to_matrix()`` and applies that matrix, so Qiskit remains the sole
authority on what ``XXPlusYY(theta)`` and ``RZZ(theta)`` mean and there is no
hand-derived convention to drift. Under Jordan-Wigner a computational basis
state *is* the determinant with the same occupation, with coefficient +1, so no
fermionic phase enters either: the qubit-space matrix acts directly on the CI
matrix.

`validate` and `selftest` both check this simulator against Aer amplitude by
amplitude, and the run itself records which simulator produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from determinants import make_strings


ZERO_TOLERANCE = 1.0e-12


class UnsupportedCircuit(RuntimeError):
    """Raised when a circuit does something this simulator cannot represent."""


@dataclass(frozen=True)
class _GateSlot:
    """One gate, reduced to what the sector representation needs."""

    kind: str  # "alpha1", "beta1", "alpha2", "beta2", "cross_diag"
    qubits: tuple[int, ...]  # orbital indices within the relevant block(s)
    index: int  # position in the bound circuit's instruction list


def _group_by_block(slots: list[_GateSlot]) -> list[_GateSlot]:
    """Move alpha-block gates ahead of beta-block gates, segment by segment.

    Gates acting on disjoint qubits commute, and the alpha and beta blocks are
    disjoint, so this reordering leaves the circuit exactly equivalent. What it
    buys is locality: the CI matrix is alpha-major, so a beta gate is a strided
    column operation and an alpha gate is a contiguous row operation. Grouping
    lets the beta gates be applied to a single transposed copy instead of
    paying the stride on every one of them.

    Segments are cut at every cross-spin gate, which is the only place the two
    blocks genuinely meet and therefore the only place the reordering could not
    be justified.
    """
    reordered: list[_GateSlot] = []
    alpha: list[_GateSlot] = []
    beta: list[_GateSlot] = []
    for slot in slots:
        if slot.kind == "cross_diag":
            reordered += alpha + beta + [slot]
            alpha, beta = [], []
        elif slot.kind in ("alpha1", "alpha2"):
            alpha.append(slot)
        else:
            beta.append(slot)
    return reordered + alpha + beta


class SectorSimulator:
    """Evolve the CI matrix of one particle-number sector through a circuit."""

    def __init__(
        self,
        circuit: Any,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
    ) -> None:
        self.circuit = circuit
        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        if circuit.num_qubits != 2 * n_orbitals:
            raise UnsupportedCircuit(
                f"circuit has {circuit.num_qubits} qubits, expected {2 * n_orbitals}"
            )
        self.strings_a = make_strings(n_orbitals, n_alpha)
        self.strings_b = make_strings(n_orbitals, n_beta)
        self._bits_a = self._bit_table(self.strings_a)
        self._bits_b = self._bit_table(self.strings_b)
        self._pair_cache: dict[tuple[str, int, int], Any] = {}
        self._initial = self._parse_prefix()
        self._slots = self._parse_gates()

    # -- parsing ---------------------------------------------------------

    def _bit_table(self, strings: np.ndarray) -> np.ndarray:
        table = np.zeros((len(strings), self.n_orbitals), dtype=np.int8)
        for row, value in enumerate(strings):
            for orbital in range(self.n_orbitals):
                table[row, orbital] = (int(value) >> orbital) & 1
        return table

    def _qubit_index(self, qubit: Any) -> int:
        return int(self.circuit.find_bit(qubit).index)

    def _parse_prefix(self) -> tuple[int, int]:
        """Read the leading X layer as the reference determinant."""
        string_a = 0
        string_b = 0
        for instruction in self.circuit.data:
            name = instruction.operation.name
            if name == "barrier":
                continue
            if name != "x":
                break
            qubit = self._qubit_index(instruction.qubits[0])
            if qubit < self.n_orbitals:
                string_a ^= 1 << qubit
            else:
                string_b ^= 1 << (qubit - self.n_orbitals)
        if string_a.bit_count() != self.n_alpha or string_b.bit_count() != self.n_beta:
            raise UnsupportedCircuit(
                "the leading X layer does not prepare the requested electron count "
                f"(got {string_a.bit_count()} alpha, {string_b.bit_count()} beta; "
                f"expected {self.n_alpha}, {self.n_beta})"
            )
        return string_a, string_b

    def _parse_gates(self) -> list[_GateSlot]:
        slots: list[_GateSlot] = []
        seen_non_x = False
        for index, instruction in enumerate(self.circuit.data):
            name = instruction.operation.name
            if name == "barrier":
                continue
            if name == "x":
                if seen_non_x:
                    raise UnsupportedCircuit(
                        "X gates are only supported in the leading reference layer"
                    )
                continue
            seen_non_x = True
            qubits = [self._qubit_index(q) for q in instruction.qubits]
            if len(qubits) == 1:
                qubit = qubits[0]
                if qubit < self.n_orbitals:
                    slots.append(_GateSlot("alpha1", (qubit,), index))
                else:
                    slots.append(
                        _GateSlot("beta1", (qubit - self.n_orbitals,), index)
                    )
            elif len(qubits) == 2:
                first, second = qubits
                both_alpha = first < self.n_orbitals and second < self.n_orbitals
                both_beta = first >= self.n_orbitals and second >= self.n_orbitals
                if both_alpha:
                    slots.append(_GateSlot("alpha2", (first, second), index))
                elif both_beta:
                    slots.append(
                        _GateSlot(
                            "beta2",
                            (first - self.n_orbitals, second - self.n_orbitals),
                            index,
                        )
                    )
                else:
                    if first >= self.n_orbitals:
                        raise UnsupportedCircuit(
                            "cross-spin gates must list the alpha qubit first"
                        )
                    slots.append(
                        _GateSlot(
                            "cross_diag", (first, second - self.n_orbitals), index
                        )
                    )
            else:
                raise UnsupportedCircuit(
                    f"gate '{name}' acts on {len(qubits)} qubits; only 1 and 2 supported"
                )
        return _group_by_block(slots)

    # -- gate application ------------------------------------------------

    def _pair_layout(self, block: str, first: int, second: int):
        """Index arrays splitting a string set by the occupation of two orbitals.

        Returns ``(mask00, mask11, rows_10, rows_01)`` where ``rows_10[k]`` and
        ``rows_01[k]`` are the two strings that differ only by which of the pair
        holds the electron, aligned so that the 2x2 mixing block applies
        elementwise.
        """
        key = (block, first, second)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        strings = self.strings_a if block == "a" else self.strings_b
        bits = self._bits_a if block == "a" else self._bits_b
        n_i = bits[:, first]
        n_j = bits[:, second]
        mask00 = (n_i == 0) & (n_j == 0)
        mask11 = (n_i == 1) & (n_j == 1)
        rows_10 = np.nonzero((n_i == 1) & (n_j == 0))[0]
        partners = strings[rows_10] ^ (1 << first) ^ (1 << second)
        rows_01 = np.searchsorted(strings, partners)
        if len(rows_01) and np.any(strings[rows_01] != partners):  # pragma: no cover
            raise RuntimeError("partner string missing from the sector")
        layout = (mask00, mask11, rows_10, rows_01)
        self._pair_cache[key] = layout
        return layout

    @staticmethod
    def _check_number_preserving(matrix: np.ndarray) -> None:
        offenders = [
            matrix[0, 1],
            matrix[0, 2],
            matrix[0, 3],
            matrix[1, 0],
            matrix[1, 3],
            matrix[2, 0],
            matrix[2, 3],
            matrix[3, 0],
            matrix[3, 1],
            matrix[3, 2],
        ]
        if max(abs(value) for value in offenders) > ZERO_TOLERANCE:
            raise UnsupportedCircuit(
                "a two-qubit gate in this circuit does not conserve the electron "
                "count of the pair it acts on; run with --simulator aer instead"
            )

    def _apply_two_qubit(
        self, state: np.ndarray, matrix: np.ndarray, block: str, first: int, second: int
    ) -> np.ndarray:
        """Apply a mixing gate as a ROW operation.

        `state` must already be oriented so that the gate's spin block indexes
        its rows -- `run` handles that by transposing once per block switch.
        """
        self._check_number_preserving(matrix)
        mask00, mask11, rows_10, rows_01 = self._pair_layout(block, first, second)
        # Qiskit indexes a two-qubit matrix as (n_second << 1) | n_first.
        u00 = matrix[0, 0]
        u11 = matrix[3, 3]
        a, b = matrix[1, 1], matrix[1, 2]
        c, d = matrix[2, 1], matrix[2, 2]
        # XXPlusYY leaves the doubly-empty and doubly-occupied pairs alone, and
        # those two masks touch every amplitude in the array. Skipping them when
        # they are the identity removes two full passes per Givens rotation.
        if not (
            abs(u00 - 1.0) <= ZERO_TOLERANCE and abs(u11 - 1.0) <= ZERO_TOLERANCE
        ):
            state[mask00] *= u00
            state[mask11] *= u11

        x = state[rows_10]
        y = state[rows_01]
        # new_10 = a x + b y and new_01 = c x + d y, with one scratch buffer
        # instead of the six temporaries the plain expression would allocate.
        scratch = x * a
        scratch += b * y
        y *= d
        y += c * x
        state[rows_10] = scratch
        state[rows_01] = y
        return state

    def _one_qubit_factors(
        self, matrix: np.ndarray, block: str, orbital: int
    ) -> np.ndarray:
        if abs(matrix[0, 1]) > ZERO_TOLERANCE or abs(matrix[1, 0]) > ZERO_TOLERANCE:
            raise UnsupportedCircuit(
                "a single-qubit gate in this circuit is not diagonal, so it does "
                "not conserve the electron count; run with --simulator aer instead"
            )
        bits = self._bits_a if block == "a" else self._bits_b
        occupation = bits[:, orbital]
        return np.where(occupation == 1, matrix[1, 1], matrix[0, 0])

    def _two_qubit_diagonal_factors(
        self, matrix: np.ndarray, block: str, first: int, second: int
    ) -> np.ndarray | None:
        """Phases for a same-block diagonal gate, or None if it mixes."""
        if abs(matrix[1, 2]) > ZERO_TOLERANCE or abs(matrix[2, 1]) > ZERO_TOLERANCE:
            return None
        bits = self._bits_a if block == "a" else self._bits_b
        index = (bits[:, second].astype(np.int64) << 1) | bits[:, first].astype(
            np.int64
        )
        diagonal = np.array(
            [matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[3, 3]]
        )
        return diagonal[index]

    def _cross_index(self, alpha: int, beta: int) -> np.ndarray:
        """Which of the four (n_alpha, n_beta) cases each amplitude falls into.

        Depends only on the orbital pair, never on the gate angle, so it is
        built once and reused. Rebuilding it per gate means allocating a full
        (n_a, n_b) integer array for every cross-spin term of every circuit
        evaluation, which the profiler put among the top costs of a run.
        """
        key = ("cross", alpha, beta)
        cached = self._pair_cache.get(key)
        if cached is None:
            n_alpha = self._bits_a[:, alpha].astype(np.int64)
            n_beta = self._bits_b[:, beta].astype(np.int64)
            # Qiskit index = (n_beta << 1) | n_alpha for qubit order (alpha, beta).
            cached = (n_beta[None, :] << 1) | n_alpha[:, None]
            self._pair_cache[key] = cached
        return cached

    def _cross_diagonal_factors(
        self, matrix: np.ndarray, alpha: int, beta: int
    ) -> np.ndarray:
        self._check_number_preserving(matrix)
        if abs(matrix[1, 2]) > ZERO_TOLERANCE or abs(matrix[2, 1]) > ZERO_TOLERANCE:
            raise UnsupportedCircuit(
                "cross-spin gates must be diagonal; a hopping term between the "
                "alpha and beta blocks changes S_z"
            )
        diagonal = np.array(
            [matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[3, 3]]
        )
        return diagonal[self._cross_index(alpha, beta)]

    # -- public API ------------------------------------------------------

    def run(self, angles: Sequence[float], parameters: Any = None) -> np.ndarray:
        """Return the CI matrix, complex, indexed by (alpha string, beta string)."""
        if parameters is None:
            parameters = list(self.circuit.parameters)
        bound = self.circuit.assign_parameters(
            {p: float(v) for p, v in zip(parameters, angles)}
        )
        data = list(bound.data)

        n_a, n_b = len(self.strings_a), len(self.strings_b)
        state = np.zeros((n_a, n_b), dtype=np.complex128)
        row = int(np.searchsorted(self.strings_a, self._initial[0]))
        column = int(np.searchsorted(self.strings_b, self._initial[1]))
        state[row, column] = 1.0

        # Diagonal gates are fused before being applied. A Jastrow layer is
        # 58 consecutive diagonal gates, and applying each one separately means
        # 58 full passes over the state; folding them into one alpha vector, one
        # beta vector and one cross-spin array first means three.
        pending_a = np.ones(n_a, dtype=np.complex128)
        pending_b = np.ones(n_b, dtype=np.complex128)
        pending_cross: np.ndarray | None = None
        pending = False
        # False: rows are alpha strings (canonical). True: rows are beta strings.
        transposed = False

        def orient(current: np.ndarray, want_transposed: bool) -> np.ndarray:
            nonlocal transposed
            if transposed == want_transposed:
                return current
            transposed = want_transposed
            return np.ascontiguousarray(current.T)

        def flush(current: np.ndarray) -> np.ndarray:
            nonlocal pending_a, pending_b, pending_cross, pending
            if not pending:
                return current
            current = orient(current, False)
            if pending_cross is not None:
                current *= pending_cross
            current *= pending_a[:, None]
            current *= pending_b[None, :]
            pending_a = np.ones(n_a, dtype=np.complex128)
            pending_b = np.ones(n_b, dtype=np.complex128)
            pending_cross = None
            pending = False
            return current

        for slot in self._slots:
            matrix = np.asarray(data[slot.index].operation.to_matrix(), dtype=complex)
            if slot.kind == "alpha1":
                pending_a *= self._one_qubit_factors(matrix, "a", slot.qubits[0])
                pending = True
            elif slot.kind == "beta1":
                pending_b *= self._one_qubit_factors(matrix, "b", slot.qubits[0])
                pending = True
            elif slot.kind == "cross_diag":
                factors = self._cross_diagonal_factors(
                    matrix, slot.qubits[0], slot.qubits[1]
                )
                pending_cross = (
                    factors if pending_cross is None else pending_cross * factors
                )
                pending = True
            else:
                block = "a" if slot.kind == "alpha2" else "b"
                diagonal = self._two_qubit_diagonal_factors(
                    matrix, block, slot.qubits[0], slot.qubits[1]
                )
                if diagonal is not None:
                    if block == "a":
                        pending_a *= diagonal
                    else:
                        pending_b *= diagonal
                    pending = True
                    continue
                state = flush(state)
                state = orient(state, block == "b")
                state = self._apply_two_qubit(
                    state, matrix, block, slot.qubits[0], slot.qubits[1]
                )
        state = flush(state)
        return orient(state, False)

    def probabilities(self, angles: Sequence[float], parameters: Any = None) -> np.ndarray:
        state = self.run(angles, parameters)
        probability = np.abs(state) ** 2
        total = probability.sum()
        if total <= 0:  # pragma: no cover
            raise RuntimeError("the circuit produced a zero state")
        return probability / total
