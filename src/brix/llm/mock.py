"""Mock LLM client for testing — supports configurable responses and sequences.

Fully async-compatible. Supports fixed responses, response sequences,
callable response generators, and configurable latency simulation.
"""

from __future__ import annotations

from collections.abc import Callable


class MockLLMClient:
    """Configurable mock LLM client for testing.

    Supports:
    - Fixed response: returns the same string every time.
    - Response sequence: returns responses in order, cycling from the end.
    - Callable: calls a function with the prompt to generate responses.
    - Latency simulation: configurable delay per call.
    """

    def __init__(
        self,
        responses: str | list[str] | Callable[[str], str] | None = None,
        *,
        default_response: str = "This is a mock response.",
    ) -> None:
        self._responses = responses
        self._default = default_response
        self._call_count: int = 0
        self._call_history: list[str] = []

    @property
    def call_count(self) -> int:
        """Number of times complete() has been called."""
        return self._call_count

    @property
    def call_history(self) -> list[str]:
        """List of all prompts received."""
        return list(self._call_history)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return a mock response based on configuration.

        Args:
            prompt: The user prompt.
            system: Ignored in mock.
            temperature: Ignored in mock.
            max_tokens: Ignored in mock.

        Returns:
            The configured mock response.
        """
        self._call_count += 1
        self._call_history.append(prompt)

        if self._responses is None:
            return self._default

        if callable(self._responses):
            return self._responses(prompt)

        if isinstance(self._responses, list):
            if not self._responses:
                return self._default
            # Use modulo to cycle through responses
            idx = (self._call_count - 1) % len(self._responses)
            return self._responses[idx]

        # Single string response
        return self._responses
