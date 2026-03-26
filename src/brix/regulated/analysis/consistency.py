"""Semantic consistency analyzer using sentence-transformers.

Uses the all-MiniLM-L6-v2 model to compute pairwise cosine similarity
between collected response samples. The model is loaded ONCE at
initialization and never reloaded per request.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """Result of semantic consistency analysis."""

    mean_similarity: float
    variance: float
    pairwise_similarities: list[float]


class SemanticConsistencyAnalyzer:
    """Computes pairwise semantic similarity between response samples.

    The sentence-transformers model is loaded exactly once during __init__
    and reused for all subsequent calls. No per-request model loading.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name)

    def analyze(self, samples: list[str]) -> ConsistencyResult:
        """Compute pairwise cosine similarity between all samples.

        Args:
            samples: List of response texts (must have at least 2).

        Returns:
            ConsistencyResult with mean similarity, variance, and all
            pairwise similarity values.
        """
        if len(samples) < 2:
            return ConsistencyResult(
                mean_similarity=1.0,
                variance=0.0,
                pairwise_similarities=[1.0],
            )

        # Encode all samples at once (batch operation)
        embeddings = self._model.encode(samples, convert_to_numpy=True)
        embeddings = np.array(embeddings)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        # Compute pairwise cosine similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)

        # Extract upper triangle (excluding diagonal)
        n = len(samples)
        pairwise: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise.append(float(sim_matrix[i, j]))

        if not pairwise:
            return ConsistencyResult(
                mean_similarity=1.0,
                variance=0.0,
                pairwise_similarities=[1.0],
            )

        mean_sim = float(np.mean(pairwise))
        var_sim = float(np.var(pairwise))

        return ConsistencyResult(
            mean_similarity=mean_sim,
            variance=var_sim,
            pairwise_similarities=pairwise,
        )
