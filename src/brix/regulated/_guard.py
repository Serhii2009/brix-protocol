"""RegulatedGuard — wraps BrixRouter as a BRIX Guard.

Integrates the full regulated-domain analysis pipeline (two-track evaluation,
adaptive sampling, semantic consistency, uncertainty classification) into the
BRIX Interceptor Chain.

RegulatedGuard short-circuits the chain's own LLM call: BrixRouter.process()
handles all sampling and response generation internally, so the final response
is delivered via a CallResponse from pre_call rather than by the chain's
llm_callable. This avoids double LLM calls.

Usage (via BRIX.wrap)::

    client = BRIX.wrap(my_llm, regulated_spec="medical")

Usage (direct)::

    from brix.regulated import RegulatedGuard

    guard = RegulatedGuard(my_llm, spec="medical")
    chain = InterceptorChain([guard])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brix.exceptions import BrixGuardBlockedError
from brix.guards.protocol import CallRequest, CallResponse
from brix.context import ExecutionContext
from brix.regulated.core.router import BrixRouter
from brix.regulated.llm.protocol import LLMClient
from brix.regulated.retrieval.protocol import RetrievalProvider
from brix.regulated.spec.models import SpecModel


def _extract_last_user_message(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last user-role message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return str(content) if content else ""
    return ""


def _extract_system_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the content of the first system-role message, or None."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            return str(content) if content else None
    return None


class RegulatedGuard:
    """Guard that runs the full regulated-domain analysis pipeline.

    Wraps :class:`~brix.regulated.core.router.BrixRouter` as a BRIX Guard.
    On each call, it:

    1. Extracts the last user message as the query.
    2. Runs the full BrixRouter pipeline (two-track evaluation, sampling,
       classification, action execution).
    3. Stores the :class:`~brix.regulated.core.result.StructuredResult` in
       ``context.metadata["regulated_result"]`` for downstream inspection.
    4. If a circuit breaker fired with mandatory intervention, raises
       :class:`~brix.exceptions.BrixGuardBlockedError`.
    5. Otherwise returns a ``CallResponse`` to short-circuit the chain's own
       LLM call (BrixRouter already produced the final response).

    Args:
        llm_client: LLM client implementing the BRIX LLMClient protocol.
        spec: Spec path, built-in spec name (e.g. ``"medical"``), or
            :class:`~brix.regulated.spec.models.SpecModel`. Defaults to the
            built-in general v1.0.0 spec.
        embedding_model: Sentence-transformers model for semantic analysis.
        log_path: Optional path for JSONL audit logging.
        system_prompt: Optional system prompt for all sample LLM calls.
        enable_output_guard: Enable response-side output signal scanning.
        retrieval_provider: Optional RAG provider for epistemic queries.
    """

    name: str = "regulated"

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        spec: SpecModel | str | Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        log_path: Path | None = None,
        system_prompt: str | None = None,
        enable_output_guard: bool = False,
        retrieval_provider: RetrievalProvider | None = None,
    ) -> None:
        self._router = BrixRouter(
            llm_client=llm_client,
            spec=spec,
            embedding_model=embedding_model,
            log_path=log_path,
            system_prompt=system_prompt,
            enable_output_guard=enable_output_guard,
            retrieval_provider=retrieval_provider,
        )

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest | CallResponse | None:
        """Run the regulated analysis pipeline and short-circuit the LLM call.

        Returns:
            A ``CallResponse`` containing the BrixRouter's final response.
            The chain will skip its own LLM call.

        Raises:
            BrixGuardBlockedError: If a circuit breaker fires and the router
                determines the query must be blocked.
        """
        query = _extract_last_user_message(request.messages)

        if not query:
            # No user message to analyse — pass through to the chain's LLM call
            return request

        ctx_str = _extract_system_message(request.messages)
        retrieval_score: float | None = request.kwargs.get("retrieval_score")

        result = await self._router.process(
            query,
            context=ctx_str,
            retrieval_score=retrieval_score,
        )

        # Store for downstream inspection (e.g. logging, other guards)
        context.metadata["regulated_result"] = result

        # Circuit breaker with mandatory intervention → block the call
        if result.circuit_breaker_hit and result.intervention_necessary:
            raise BrixGuardBlockedError(
                self.name,
                f"circuit breaker '{result.circuit_breaker_name}' fired",
            )

        # BrixRouter produced the final response — short-circuit the chain's LLM call
        return CallResponse(
            content=result.response,
            usage=None,
            raw=result,
        )

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Pass-through post_call hook.

        Future: run OutputGuard on the final response here if needed.
        """
        return response
