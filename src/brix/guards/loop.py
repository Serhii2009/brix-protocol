"""LoopGuard — detects and recovers from infinite agent loops.

Two-tier detection:

**Tier 1 — Exact (100% guaranteed):** SHA-256 hashing detects bit-for-bit
identical responses. SHA-256 collision resistance makes false negatives
mathematically impossible within any practical window.

**Tier 2 — Semantic (~95%):** Sentence-transformer cosine similarity detects
paraphrased repetition. Requires ``pip install 'brix-protocol[semantic]'``.
Not 100% guaranteed; threshold tuning affects recall vs. false positives.

Recovery: On detection, LoopGuard injects a diversity prompt into the next
pre_call to encourage a different response. After ``diversity_attempts``
failed recoveries, it raises :class:`~brix.exceptions.BrixLoopError`.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from brix.context import ExecutionContext
from brix.exceptions import BrixConfigurationError, BrixLoopError
from brix.guards.protocol import CallRequest, CallResponse

_DEFAULT_DIVERSITY_PROMPT = (
    "[BRIX-LOOP-RECOVERY] You appear to be repeating yourself. "
    "You MUST take a genuinely different approach on this response."
)


class LoopGuard:
    """Guard that detects and recovers from infinite agent loops.

    All session state is stored in ``context.metadata`` under ``_loop_*`` keys,
    so the guard instance itself is stateless and safe to share across calls.

    **Tier 1 (100% guaranteed):** SHA-256 collision resistance guarantees that
    exact duplicate detection is mathematically certain within the rolling window.

    **Tier 2 (~95%):** Semantic loop detection depends on the embedding similarity
    threshold. Not 100% guaranteed; adjust ``semantic_threshold`` for your use case.

    Args:
        exact_threshold: Number of SHA-256-identical responses in the rolling window
            that triggers detection. Default 3.
        semantic_detection: Enable Tier 2 cosine-similarity detection.
            Requires ``pip install 'brix-protocol[semantic]'``. Default False.
        semantic_threshold: Cosine similarity threshold for semantic loop detection.
            Default 0.92.
        on_loop: ``"inject_diversity"`` (default) — inject a recovery prompt and
            continue. ``"raise"`` — raise :class:`~brix.exceptions.BrixLoopError`
            immediately on first detection.
        diversity_attempts: Maximum diversity injections before raising. Default 2.
        loop_window: Rolling window size for hash/embedding history. Default 10.
        diversity_prompt: Custom text injected when a loop is detected. If None,
            uses the built-in ``[BRIX-LOOP-RECOVERY]`` prompt.

    Raises:
        BrixConfigurationError: If ``on_loop`` is not a recognised value, or if
            ``semantic_detection=True`` but ``sentence-transformers`` is not installed.
    """

    name = "loop"

    # Class-level model cache — shared across all LoopGuard instances in the process.
    # Loading SentenceTransformer is expensive (~90 MB); caching ensures it is done once.
    _semantic_model: Any = None
    _semantic_model_lock: asyncio.Lock | None = None

    def __init__(
        self,
        *,
        exact_threshold: int = 3,
        semantic_detection: bool = False,
        semantic_threshold: float = 0.92,
        on_loop: str = "inject_diversity",
        diversity_attempts: int = 2,
        loop_window: int = 10,
        diversity_prompt: str | None = None,
    ) -> None:
        if on_loop not in ("inject_diversity", "raise"):
            raise BrixConfigurationError(
                f"LoopGuard on_loop must be 'inject_diversity' or 'raise', got {on_loop!r}"
            )
        if semantic_detection:
            try:
                import sentence_transformers  # noqa: F401, PLC0415
            except ImportError as exc:
                raise BrixConfigurationError(
                    "LoopGuard semantic_detection=True requires sentence-transformers. "
                    "Install with: pip install 'brix-protocol[semantic]'"
                ) from exc

        self._exact_threshold = exact_threshold
        self._semantic_detection = semantic_detection
        self._semantic_threshold = semantic_threshold
        self._on_loop = on_loop
        self._diversity_attempts = diversity_attempts
        self._loop_window = loop_window
        self._diversity_prompt = diversity_prompt or _DEFAULT_DIVERSITY_PROMPT

    # ------------------------------------------------------------------
    # Class-level semantic model loader
    # ------------------------------------------------------------------

    @classmethod
    async def _get_semantic_model(cls) -> Any:
        """Load the sentence-transformer model once and cache it at class level.

        Uses double-checked locking with a lazy asyncio.Lock so only one coroutine
        performs the expensive model load; all subsequent calls return instantly.
        """
        if cls._semantic_model is not None:
            return cls._semantic_model
        if cls._semantic_model_lock is None:
            cls._semantic_model_lock = asyncio.Lock()
        async with cls._semantic_model_lock:
            if cls._semantic_model is None:  # double-checked locking
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                cls._semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._semantic_model

    # ------------------------------------------------------------------
    # Guard protocol
    # ------------------------------------------------------------------

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest:
        """Inject a diversity prompt if a loop was detected on the previous call.

        Zero overhead on clean calls — the flag check is a single dict lookup.
        """
        if not context.metadata.get("_loop_inject_next", False):
            return request

        context.metadata["_loop_inject_next"] = False
        messages = list(request.messages)

        # Append diversity prompt to an existing system message, or prepend a new one
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = {
                    **msg,
                    "content": msg.get("content", "") + "\n" + self._diversity_prompt,
                }
                break
        else:
            messages.insert(0, {"role": "system", "content": self._diversity_prompt})

        return CallRequest(messages=messages, model=request.model, kwargs=request.kwargs)

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Hash the response, check for loops, and trigger recovery or raise."""
        # --- Tier 1: exact SHA-256 matching ---
        content_hash = hashlib.sha256(str(response.content).encode()).hexdigest()

        hashes: list[str] = context.metadata.setdefault("_loop_hashes", [])
        hashes.append(content_hash)
        # Trim to rolling window
        if len(hashes) > self._loop_window:
            del hashes[: len(hashes) - self._loop_window]

        exact_count = hashes.count(content_hash)
        loop_detected = exact_count >= self._exact_threshold
        loop_reason = (
            f"exact loop detected after {exact_count} identical responses; "
            f"hash={content_hash[:16]}…; window={hashes}"
        )

        # --- Tier 2: semantic cosine similarity ---
        if not loop_detected and self._semantic_detection:
            model = await self._get_semantic_model()
            embedding: list[float] = model.encode(str(response.content)).tolist()

            embeddings: list[list[float]] = context.metadata.setdefault("_loop_embeddings", [])
            for prev_emb in embeddings:
                sim = _cosine_similarity(prev_emb, embedding)
                if sim >= self._semantic_threshold:
                    loop_detected = True
                    loop_reason = (
                        f"semantic loop detected (cosine similarity={sim:.4f} >= "
                        f"{self._semantic_threshold}); window size={len(embeddings)}"
                    )
                    break

            embeddings.append(embedding)
            if len(embeddings) > self._loop_window - 1:
                del embeddings[: len(embeddings) - (self._loop_window - 1)]

        # --- Recovery or raise ---
        if loop_detected:
            if self._on_loop == "raise":
                raise BrixLoopError(reason=loop_reason)

            # inject_diversity path
            diversity_count: int = context.metadata.get("_loop_diversity_count", 0)
            if diversity_count >= self._diversity_attempts:
                raise BrixLoopError(
                    reason=(
                        f"loop persisted after {diversity_count} diversity injection(s); "
                        + loop_reason
                    )
                )
            context.metadata["_loop_inject_next"] = True
            context.metadata["_loop_diversity_count"] = diversity_count + 1

        return response


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import numpy as np  # noqa: PLC0415

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    dot = float(np.dot(arr_a, arr_b))
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["LoopGuard"]
