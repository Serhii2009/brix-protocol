"""LLM client protocol — the abstract interface any provider must implement.

Any LLM client that satisfies this Protocol can be used with BRIX.
The only requirement is an async complete() method.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Abstract protocol for LLM client implementations.

    Any class implementing this protocol can be passed to BrixRouter.
    The complete() method must be async and accept the specified parameters.
    """

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a prompt to the LLM and return the text completion.

        Args:
            prompt: The user prompt to complete.
            system: Optional system prompt.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response.
        """
        ...
