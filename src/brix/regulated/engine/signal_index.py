"""Pre-compiled Aho-Corasick signal index for O(n) multi-pattern matching.

All patterns from circuit breakers and risk signals are compiled into a
single automaton at initialization. Per-request evaluation runs in O(n)
where n is the query length, not O(patterns x query). The automaton is
rebuilt only when the specification changes.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import ahocorasick

from brix.regulated.spec.models import SpecModel


def _normalize(text: str) -> str:
    """Apply Unicode NFKC normalization and collapse whitespace.

    Converts non-breaking spaces, soft hyphens, and other Unicode
    variants to their canonical forms, then collapses all whitespace
    runs (including line breaks) into single spaces.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


@dataclass(frozen=True, slots=True)
class SignalMatch:
    """A single pattern match found during query scanning."""

    signal_name: str
    signal_type: str  # "circuit_breaker" or "risk_signal"
    pattern: str
    # NOTE: end_position refers to position in the normalized text,
    # not the original input text.
    end_position: int


class SignalIndex:
    """Pre-compiled Aho-Corasick automaton for efficient signal matching.

    Builds a single automaton containing all patterns from both circuit
    breakers and risk signals. The scan() method performs a single O(n)
    pass over the query text and returns all matches.
    """

    def __init__(self, spec: SpecModel) -> None:
        self._automaton = ahocorasick.Automaton()
        self._build(spec)

    def _build(self, spec: SpecModel) -> None:
        """Compile all patterns into the Aho-Corasick automaton.

        Accumulates all values per normalized key in a plain dict first,
        then calls add_word() once per key with the complete final list,
        then make_automaton(). No mutation between add_word and make_automaton.
        """
        pending: dict[str, list[tuple[str, str, str]]] = {}

        for cb in spec.circuit_breakers:
            for pattern in cb.patterns:
                key = _normalize(pattern).lower()
                value = (cb.name, "circuit_breaker", pattern)
                pending.setdefault(key, []).append(value)

        for signal in spec.risk_signals:
            for pattern in signal.patterns:
                key = _normalize(pattern).lower()
                value = (signal.name, "risk_signal", pattern)
                pending.setdefault(key, []).append(value)

        for key, values in pending.items():
            self._automaton.add_word(key, values)

        self._automaton.make_automaton()

    def scan(self, text: str) -> list[SignalMatch]:
        """Scan text for all matching patterns in a single O(n) pass.

        Args:
            text: The query text to scan.

        Returns:
            List of all signal matches found in the text.
        """
        if not self._automaton:
            return []

        matches: list[SignalMatch] = []
        lowered = _normalize(text).lower()

        for end_pos, values in self._automaton.iter(lowered):
            for signal_name, signal_type, pattern in values:
                matches.append(
                    SignalMatch(
                        signal_name=signal_name,
                        signal_type=signal_type,
                        pattern=pattern,
                        end_position=end_pos,
                    )
                )

        return matches

    def rebuild(self, spec: SpecModel) -> None:
        """Rebuild the automaton from a new specification.

        Called only when the specification changes, never per-request.
        """
        self._automaton = ahocorasick.Automaton()
        self._build(spec)
