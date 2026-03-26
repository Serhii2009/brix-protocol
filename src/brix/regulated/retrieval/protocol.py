"""RetrievalProvider protocol and RetrievalResult data class."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievalResult:
    """Result returned by a RetrievalProvider."""

    content: str
    score: float
    sources: list[str] = field(default_factory=list)
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@runtime_checkable
class RetrievalProvider(Protocol):
    """Protocol for retrieval augmentation providers.

    Implement this with a single async retrieve() method to plug
    real RAG into the BRIX pipeline.
    """

    async def retrieve(self, query: str, *, max_results: int = 3) -> RetrievalResult:
        """Retrieve relevant content for a query.

        Args:
            query: The user query to retrieve content for.
            max_results: Maximum number of results to return.

        Returns:
            RetrievalResult with retrieved content, quality score, and sources.
        """
        ...
