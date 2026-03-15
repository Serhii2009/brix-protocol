"""Anthropic adapter for the BRIX LLM client protocol.

Requires the 'anthropic' optional dependency: pip install brix-protocol[anthropic]
"""

from __future__ import annotations

from typing import Any


class AnthropicClient:
    """Production-ready Anthropic adapter implementing the LLMClient protocol.

    Wraps the official Anthropic Python SDK's async client. The API key must
    be provided via the ANTHROPIC_API_KEY environment variable or passed
    directly to the Anthropic client.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20250514",
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic adapter requires the 'anthropic' package. "
                "Install it with: pip install brix-protocol[anthropic]"
            ) from exc

        self._model = model
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            self._client = AsyncAnthropic(**kwargs)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a prompt to Anthropic and return the text completion.

        Args:
            prompt: The user prompt to complete.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        # Extract text from the first content block
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
