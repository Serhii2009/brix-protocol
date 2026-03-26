"""OpenAI adapter for the BRIX LLM client protocol.

Requires the 'openai' optional dependency: pip install brix-protocol[openai]
"""

from __future__ import annotations

from typing import Any


class OpenAIClient:
    """Production-ready OpenAI adapter implementing the LLMClient protocol.

    Wraps the official OpenAI Python SDK's async client. The API key must
    be provided via the OPENAI_API_KEY environment variable or passed
    directly to the OpenAI client.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI adapter requires the 'openai' package. "
                "Install it with: pip install brix-protocol[openai]"
            ) from exc

        self._model = model
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            self._client = AsyncOpenAI(**kwargs)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a prompt to OpenAI and return the text completion.

        Args:
            prompt: The user prompt to complete.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
