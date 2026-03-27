"""SchemaGuard — guaranteed structured output with self-healing validation.

The 100% guarantee: if ``complete()`` returns without raising, the return value
is a valid instance of ``response_schema``. A partially valid or unvalidated
object cannot be returned. If validation fails after all retries (or after the
optional time budget), :class:`~brix.exceptions.BrixSchemaError` is raised
with the complete attempt history.

Three phases:
1. **Prevention (pre_call):** Injects the JSON schema into the system prompt
   before the first call. The model sees exactly what structure is expected.
   Schema injection is idempotent — a sentinel marker prevents double-injection
   across multiple calls in the same session.
2. **Detection (post_call):** Extracts JSON from the response (handling markdown
   code blocks and surrounding text) and validates against the Pydantic model.
   On success, ``response.content`` is replaced with the validated model instance.
3. **Self-healing (post_call retry loop):** If validation fails, a precise
   feedback message is constructed and sent back to the model. This loop runs
   up to ``max_retries`` times and is bounded by ``max_healing_seconds``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from json import JSONDecodeError
from typing import Any

from pydantic import BaseModel, ValidationError

from brix.context import ExecutionContext
from brix.exceptions import BrixSchemaError
from brix.guards.protocol import CallRequest, CallResponse


def _extract_json(text: str) -> str:
    """Extract JSON from an LLM response string.

    Handles (in order of priority):
    1. ```json ... ``` or ``` ... ``` markdown code blocks
    2. Raw JSON starting with ``{`` or ``[`` (finds first occurrence,
       returns the substring up to the matching closing bracket)
    3. Fallback: returns the stripped input as-is

    Args:
        text: Raw LLM response content.

    Returns:
        Best-effort JSON string for further parsing.
    """
    stripped = text.strip()

    # 1. Markdown code block (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Find first JSON object or array and balance brackets
    for open_c, close_c in [('{', '}'), ('[', ']')]:
        start = stripped.find(open_c)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(stripped[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return stripped[start:i + 1]
        break

    # 3. Fallback
    return stripped


def _build_feedback(exc: Exception, schema_json: str) -> str:
    """Build a precise re-prompt message from a validation error.

    Args:
        exc: The ``ValidationError`` or ``JSONDecodeError`` that was raised.
        schema_json: The expected JSON schema (for inclusion in the feedback).

    Returns:
        A targeted feedback message for the model.
    """
    if isinstance(exc, ValidationError):
        error_lines: list[str] = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err["loc"]) if err["loc"] else "(root)"
            msg = err["msg"]
            error_lines.append(f"  - field '{loc}': {msg}")
        errors_str = "\n".join(error_lines)
        return (
            "Your response did not match the required JSON schema. "
            "Please correct these specific errors and respond with valid JSON only:\n"
            f"{errors_str}\n\n"
            f"Required schema:\n{schema_json}"
        )
    if isinstance(exc, JSONDecodeError):
        location = f"line {exc.lineno}, column {exc.colno}" if exc.lineno else "unknown location"
        return (
            f"Your response was not valid JSON (parse error at {location}: {exc.msg}). "
            "Please respond with valid JSON only, matching this schema:\n"
            f"{schema_json}"
        )
    return (
        f"Your response could not be validated: {exc}. "
        "Please respond with valid JSON matching this schema:\n"
        f"{schema_json}"
    )


class SchemaGuard:
    """Guard that enforces structured output through schema injection and self-healing.

    Args:
        llm_callable: The async callable used for self-healing re-prompts. Pass the
            same ``build_llm_callable(llm_client)`` result as RetryGuard receives.
        schema: The Pydantic model class that defines the expected output structure.
        max_retries: Maximum self-healing re-prompt attempts. Total validation
            attempts = max_retries + 1. Default 2.
        inject_schema: Whether to inject the JSON schema into the system prompt
            during pre_call (Phase 1 prevention). Default True.
        max_healing_seconds: Optional wall-clock time budget for the entire
            post_call validation and re-prompt loop. If exceeded, raises
            :class:`~brix.exceptions.BrixSchemaError` immediately. Default None
            (no time limit, only ``max_retries`` applies).
    """

    name: str = "schema"
    _MARKER: str = "# [BRIX-SCHEMA-v1]"

    def __init__(
        self,
        llm_callable: Callable[[CallRequest], Awaitable[CallResponse]],
        schema: type[BaseModel],
        *,
        max_retries: int = 2,
        inject_schema: bool = True,
        max_healing_seconds: float | None = None,
    ) -> None:
        self._llm_callable = llm_callable
        self._schema = schema
        self._max_retries = max_retries
        self._inject = inject_schema
        self._max_healing_seconds = max_healing_seconds
        self._schema_json = json.dumps(schema.model_json_schema(), indent=2)

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest:
        """Inject the JSON schema into the system prompt (idempotent).

        Args:
            request: The outbound request.
            context: Mutable session state.

        Returns:
            Modified request with schema injected, or the unmodified request
            if ``inject_schema=False`` or the schema is already present.
        """
        if not self._inject:
            return request

        # Idempotency: skip if the marker is already in any system message
        for msg in request.messages:
            if msg.get("role") == "system" and self._MARKER in msg.get("content", ""):
                return request

        injection = (
            f"{self._MARKER}\n"
            f"You MUST respond with valid JSON matching this exact schema:\n"
            f"{self._schema_json}"
        )

        messages = list(request.messages)
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = {**msg, "content": msg["content"] + "\n\n" + injection}
                break
        else:
            messages.insert(0, {"role": "system", "content": injection})

        return CallRequest(messages=messages, model=request.model, kwargs=request.kwargs)

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Validate the response and self-heal if needed.

        On success: returns a ``CallResponse`` with ``content`` set to the validated
        Pydantic model instance. ``BrixClient.complete()`` then returns this instance
        directly (return type: Any).

        On failure after all retries (or after ``max_healing_seconds``): raises
        :class:`~brix.exceptions.BrixSchemaError` with the full attempt history.

        Args:
            request: The final request that was sent (used as base for re-prompts).
            response: The response from the LLM (or from RetryGuard's short-circuit).
            context: Mutable session state.

        Returns:
            CallResponse with ``content`` = validated Pydantic model instance.

        Raises:
            BrixSchemaError: If all validation attempts fail.
        """
        history: list[dict[str, Any]] = []
        current: str = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        healing_start = time.perf_counter()

        for attempt in range(self._max_retries + 1):
            # Time-budget check before each attempt (including first)
            if self._max_healing_seconds is not None:
                elapsed = time.perf_counter() - healing_start
                if elapsed > self._max_healing_seconds:
                    raise BrixSchemaError(
                        reason=(
                            f"healing time limit ({self._max_healing_seconds:.1f}s) exceeded "
                            f"after {attempt} attempt(s); history: {history}"
                        )
                    )

            try:
                instance = self._schema.model_validate_json(_extract_json(current))
                context.metadata["schema_validated"] = True
                context.metadata["schema_attempts"] = attempt + 1
                return CallResponse(
                    content=instance,
                    usage=response.usage,
                    raw=response.raw,
                )
            except (ValidationError, JSONDecodeError, ValueError) as exc:
                history.append(
                    {"attempt": attempt, "response": current, "error": str(exc)}
                )
                if attempt >= self._max_retries:
                    break

                # Build precise feedback and re-prompt
                feedback = _build_feedback(exc, self._schema_json)
                re_response = await self._llm_callable(
                    CallRequest(
                        messages=list(request.messages)
                        + [
                            {"role": "assistant", "content": current},
                            {"role": "user", "content": feedback},
                        ],
                        model=request.model,
                        kwargs=request.kwargs,
                    )
                )
                current = (
                    re_response.content
                    if isinstance(re_response.content, str)
                    else str(re_response.content)
                )

        raise BrixSchemaError(
            reason=f"validation failed after {len(history)} attempt(s); history: {history}"
        )


__all__ = ["SchemaGuard"]
