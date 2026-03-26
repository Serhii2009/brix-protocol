"""OutputGuard — async response-side scanning for BRIX.

Can be used standalone or integrated into BrixRouter via enable_output_guard.
"""

from __future__ import annotations

from brix.regulated.analysis.consistency import SemanticConsistencyAnalyzer
from brix.regulated.output.analyzer import OutputAnalyzer
from brix.regulated.output.result import OutputResult
from brix.regulated.spec.models import SpecModel


class OutputGuard:
    """Async output guard for scanning LLM responses.

    When used inside BrixRouter, pass _analyzer to reuse the already-loaded
    embedding model. When used standalone, a new model is loaded at init.
    """

    def __init__(
        self,
        spec: SpecModel,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        _analyzer: SemanticConsistencyAnalyzer | None = None,
    ) -> None:
        """Initialize the output guard.

        Args:
            spec: The BRIX specification containing output_signals.
            embedding_model: Sentence-transformers model name (used if _analyzer is None).
            _analyzer: Injected analyzer to reuse (avoids double model load).
        """
        if _analyzer is not None:
            self._analyzer = _analyzer
        else:
            self._analyzer = SemanticConsistencyAnalyzer(embedding_model)
        self._output_analyzer = OutputAnalyzer(spec)

    async def analyze(
        self,
        response: str,
        *,
        query: str | None = None,
        context: str | None = None,
    ) -> OutputResult:
        """Analyze a response for output-side signals.

        Args:
            response: The LLM response text to scan.
            query: Original query (for exclude_context filtering).
            context: Optional context (for exclude_context filtering).

        Returns:
            OutputResult with blocking status and risk assessment.
        """
        return self._output_analyzer.analyze(response, query=query, context=context)
