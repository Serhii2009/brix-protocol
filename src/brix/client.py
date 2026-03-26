"""BrixClient and BRIX factory — the public entry point for the BRIX library.

Usage::

    import openai
    from brix import BRIX

    client = BRIX.wrap(
        openai.OpenAI(),
        regulated_spec="medical",   # activates RegulatedGuard
        log_path="./traces",
    )

    response = await client.complete([{"role": "user", "content": "What is aspirin?"}])
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from brix.chain import InterceptorChain
from brix.context import CallRecord, ExecutionContext
from brix.exceptions import BrixConfigurationError
from brix.guards.protocol import CallRequest, CallResponse, Guard
from brix.settings import get_settings


class BrixClient:
    """Wraps any LLM client and routes calls through the Guard chain.

    Do not instantiate directly — use ``BRIX.wrap()`` instead.

    The client maintains one :class:`~brix.context.ExecutionContext` for its
    entire lifetime. Session cost, call count, and call history accumulate
    across all calls to ``complete()``.

    Args:
        llm_client: Any LLM client. Supported interfaces:
            - OpenAI SDK (``openai.OpenAI`` or ``openai.AsyncOpenAI``)
            - Anthropic SDK (``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic``)
            - Any object implementing the BRIX ``LLMClient`` protocol
              (``async def complete(prompt, *, system, temperature, max_tokens) -> str``)
        guards: Ordered list of Guards to run on every call.
        log_path: Optional path for JSONL call logging.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        guards: list[Guard],
        log_path: str | Path | None = None,
    ) -> None:
        self._llm = llm_client
        self._chain = InterceptorChain(guards)
        self._context = ExecutionContext.new_session()
        self._settings = get_settings()
        resolved_log = log_path or self._settings.log_path
        self._log_path = Path(resolved_log) if resolved_log else None
        self._llm_callable = self._build_llm_callable()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send a chat request through the Guard chain and return the response.

        Args:
            messages: List of message dicts in OpenAI chat format, e.g.
                ``[{"role": "user", "content": "Hello"}]``.
            model: Model identifier. Falls back to ``BRIX_DEFAULT_MODEL``
                env var, then to an empty string (client default).
            **kwargs: Extra keyword arguments forwarded to the LLM client
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            The response content. Typically a ``str``, but SchemaGuard
            (Sprint 2) may return a Pydantic model instance.

        Raises:
            BrixGuardBlockedError: If a Guard blocks the request.
            BrixConfigurationError: If the LLM client interface is not recognized.
        """
        resolved_model = model or self._settings.default_model or ""
        request = CallRequest(
            messages=messages,
            model=resolved_model,
            kwargs=kwargs,
        )

        # Refresh run_id and increment call counter
        self._context.run_id = str(uuid4())
        self._context.call_count += 1

        t0 = time.perf_counter()
        response = await self._chain.execute(request, self._context, self._llm_callable)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Record in session history
        self._context.call_history.append(
            CallRecord(
                request=request,
                response=response,
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms,
            )
        )

        return response.content

    # Alias for explicit async naming preference
    acomplete = complete

    @property
    def context(self) -> ExecutionContext:
        """The session-scoped ExecutionContext for this client."""
        return self._context

    @property
    def chain(self) -> InterceptorChain:
        """The InterceptorChain holding all registered Guards."""
        return self._chain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_llm_callable(self) -> Callable[[CallRequest], Awaitable[CallResponse]]:
        """Detect the LLM client interface and return the appropriate adapter."""
        client = self._llm

        # OpenAI SDK: client.chat.completions.create(...)
        if hasattr(client, "chat") and hasattr(getattr(client, "chat", None), "completions"):
            return self._openai_callable

        # Anthropic SDK: client.messages.create(...)
        if hasattr(client, "messages") and hasattr(
            getattr(client, "messages", None), "create"
        ):
            return self._anthropic_callable

        # BRIX legacy LLMClient protocol: async def complete(prompt, ...) -> str
        if callable(getattr(client, "complete", None)):
            return self._legacy_callable

        raise BrixConfigurationError(
            "llm_client does not implement a recognized interface. "
            "Expected: OpenAI SDK, Anthropic SDK, or brix.LLMClient protocol "
            "(async def complete(prompt, *, system, temperature, max_tokens) -> str)."
        )

    async def _openai_callable(self, request: CallRequest) -> CallResponse:
        """Adapter for the OpenAI SDK (openai>=1.0)."""
        model = request.model or "gpt-4o"
        temperature = request.kwargs.get("temperature", 0.7)
        max_tokens = request.kwargs.get("max_tokens", 1024)

        # Support both sync and async OpenAI clients
        create = self._llm.chat.completions.create
        if callable(create):
            import inspect

            if inspect.iscoroutinefunction(create):
                raw = await create(
                    model=model,
                    messages=request.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raw = create(
                    model=model,
                    messages=request.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        else:
            raise BrixConfigurationError("OpenAI client.chat.completions.create is not callable.")

        content: str = raw.choices[0].message.content or ""
        usage = None
        if hasattr(raw, "usage") and raw.usage is not None:
            usage = {
                "prompt_tokens": raw.usage.prompt_tokens,
                "completion_tokens": raw.usage.completion_tokens,
                "total_tokens": raw.usage.total_tokens,
            }
        return CallResponse(content=content, usage=usage, raw=raw)

    async def _anthropic_callable(self, request: CallRequest) -> CallResponse:
        """Adapter for the Anthropic SDK (anthropic>=0.30)."""
        model = request.model or "claude-3-5-sonnet-20241022"
        max_tokens = request.kwargs.get("max_tokens", 1024)

        # Split messages into system and human/assistant turns
        system_parts = [
            m["content"] for m in request.messages if m.get("role") == "system"
        ]
        non_system = [m for m in request.messages if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": non_system,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        create = self._llm.messages.create
        import inspect

        if inspect.iscoroutinefunction(create):
            raw = await create(**kwargs)
        else:
            raw = create(**kwargs)

        content = raw.content[0].text if raw.content else ""
        usage = None
        if hasattr(raw, "usage") and raw.usage is not None:
            usage = {
                "input_tokens": raw.usage.input_tokens,
                "output_tokens": raw.usage.output_tokens,
            }
        return CallResponse(content=content, usage=usage, raw=raw)

    async def _legacy_callable(self, request: CallRequest) -> CallResponse:
        """Adapter for the BRIX LLMClient protocol (prompt-based, not messages-based).

        Extracts the last user message as the prompt and any system message
        as the system argument.
        """
        prompt = ""
        system: str | None = None

        for msg in reversed(request.messages):
            if msg.get("role") == "user" and not prompt:
                prompt = msg.get("content", "")
            if msg.get("role") == "system" and system is None:
                system = msg.get("content")

        temperature = request.kwargs.get("temperature", 0.7)
        max_tokens = request.kwargs.get("max_tokens", 1024)

        content = await self._llm.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return CallResponse(content=str(content), usage=None, raw=content)


class BRIX:
    """Factory for creating a :class:`BrixClient` with configured Guards.

    Usage::

        client = BRIX.wrap(my_llm, regulated_spec="medical")
        response = await client.complete([{"role": "user", "content": "..."}])
    """

    @classmethod
    def wrap(
        cls,
        llm_client: Any,
        *,
        # Resource guards (Sprint 2)
        max_cost_usd: float | None = None,
        max_time_seconds: float | None = None,
        max_retries: int | None = None,
        rate_limit_rpm: int | None = None,
        # Safety guards (Sprint 2)
        loop_detection: bool = False,
        max_context_tokens: int | None = None,
        response_schema: type | None = None,
        # Observability
        log_path: str | Path | None = None,
        # Regulated domain guard
        regulated_spec: str | Path | Any | None = None,
    ) -> BrixClient:
        """Wrap an LLM client with BRIX reliability Guards.

        Each parameter activates exactly one Guard. Parameters whose Guards
        are not yet implemented emit a ``UserWarning`` and are ignored.

        Args:
            llm_client: Any supported LLM client (see :class:`BrixClient`).
            max_cost_usd: Maximum cumulative session cost in USD (BudgetGuard, Sprint 2).
            max_time_seconds: Maximum wall-clock time per call in seconds
                (TimeoutGuard, Sprint 2).
            max_retries: Maximum retries for transient failures (RetryGuard, Sprint 2).
            rate_limit_rpm: Maximum requests per minute (RateLimitGuard, Sprint 2).
            loop_detection: Enable agent loop detection (LoopGuard, Sprint 2).
            max_context_tokens: Maximum context window tokens before compression
                (ContextGuard, Sprint 2).
            response_schema: Pydantic model class for guaranteed structured output
                (SchemaGuard, Sprint 2).
            log_path: Directory path for JSONL call logs.
            regulated_spec: Spec path, spec name (e.g. ``"medical"``), or
                :class:`~brix.regulated.spec.models.SpecModel` for regulated-domain
                analysis (activates RegulatedGuard).

        Returns:
            A configured :class:`BrixClient` ready to use.
        """
        guards: list[Guard] = []

        # Warn for unimplemented guards
        _unimplemented: list[tuple[str, Any]] = [
            ("max_cost_usd (BudgetGuard)", max_cost_usd),
            ("max_time_seconds (TimeoutGuard)", max_time_seconds),
            ("max_retries (RetryGuard)", max_retries),
            ("rate_limit_rpm (RateLimitGuard)", rate_limit_rpm),
            ("loop_detection (LoopGuard)", loop_detection),
            ("max_context_tokens (ContextGuard)", max_context_tokens),
            ("response_schema (SchemaGuard)", response_schema),
        ]
        for param_name, value in _unimplemented:
            if value is not None and value is not False:
                warnings.warn(
                    f"BRIX: {param_name} is not yet implemented and will be ignored. "
                    "It will be active in a future sprint.",
                    stacklevel=2,
                )

        # Activate RegulatedGuard if regulated_spec is provided
        if regulated_spec is not None:
            try:
                from brix.regulated._guard import RegulatedGuard
            except ImportError as exc:
                raise BrixConfigurationError(
                    "regulated_spec requires the 'regulated' extra. "
                    "Install with: pip install 'brix-protocol[regulated]'"
                ) from exc

            settings = get_settings()
            guards.append(
                RegulatedGuard(
                    llm_client=llm_client,
                    spec=regulated_spec,
                    embedding_model=settings.embedding_model,
                    log_path=Path(log_path) if log_path else None,
                )
            )

        return BrixClient(llm_client, guards=guards, log_path=log_path)


__all__ = ["BRIX", "BrixClient"]
