"""Action executor — dispatches response strategies per uncertainty type.

Each uncertainty type produces a meaningfully different response:
  EPISTEMIC      → force retrieval augmentation
  CONTRADICTORY  → explicit conflict resolution
  OPEN_ENDED     → distribution of outcomes
  CERTAIN        → passthrough (no intervention)
"""

from __future__ import annotations

from dataclasses import dataclass

from brix.core.result import ActionTaken, UncertaintyType
from brix.llm.protocol import LLMClient
from brix.spec.models import SpecModel, UncertaintyTypeDef


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result of action execution."""

    action_taken: ActionTaken
    response: str
    intervention_necessary: bool
    cost_tokens_extra: int


class ActionExecutor:
    """Executes the appropriate response strategy for each uncertainty type.

    Each type produces a meaningfully different response, not just a
    different label on the same output.
    """

    def __init__(self, spec: SpecModel, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._type_configs: dict[str, UncertaintyTypeDef] = {
            t.name: t for t in spec.uncertainty_types
        }

    async def execute(
        self,
        uncertainty_type: UncertaintyType,
        samples: list[str],
        query: str,
        force_retrieval: bool = False,
    ) -> ActionResult:
        """Execute the action for the classified uncertainty type.

        Args:
            uncertainty_type: The classified uncertainty type.
            samples: Collected response samples.
            query: Original user query.
            force_retrieval: Whether to force retrieval (from CB hit).

        Returns:
            ActionResult with the final response and action metadata.
        """
        if uncertainty_type == UncertaintyType.CERTAIN and not force_retrieval:
            return ActionResult(
                action_taken=ActionTaken.NONE,
                response=samples[0] if samples else "",
                intervention_necessary=False,
                cost_tokens_extra=0,
            )

        if uncertainty_type == UncertaintyType.EPISTEMIC or force_retrieval:
            return await self._handle_epistemic(samples, query)

        if uncertainty_type == UncertaintyType.CONTRADICTORY:
            return await self._handle_contradictory(samples, query)

        if uncertainty_type == UncertaintyType.OPEN_ENDED:
            return await self._handle_open_ended(samples, query)

        # Fallback: treat as epistemic
        return await self._handle_epistemic(samples, query)

    async def _handle_epistemic(
        self, samples: list[str], query: str
    ) -> ActionResult:
        """Handle epistemic uncertainty — signal retrieval augmentation needed."""
        config = self._type_configs.get("epistemic")
        template = config.action_config.message_template if config else ""

        # Build a retrieval-signaling response from the samples
        sample_summary = samples[0] if samples else "No response available."
        response = (
            f"{template.strip()}\n\n"
            f"Based on initial analysis: {sample_summary}\n\n"
            f"[RETRIEVAL_NEEDED] This response requires verification through "
            f"retrieval augmentation. The query '{query}' touches on knowledge "
            f"that may not be reliably represented in the model's training data."
        )

        return ActionResult(
            action_taken=ActionTaken.FORCE_RETRIEVAL,
            response=response,
            intervention_necessary=True,
            cost_tokens_extra=self._estimate_extra_tokens(samples),
        )

    async def _handle_contradictory(
        self, samples: list[str], query: str
    ) -> ActionResult:
        """Handle contradictory uncertainty — explicit conflict resolution."""
        config = self._type_configs.get("contradictory")
        template = config.action_config.message_template if config else ""

        # Build conflict resolution from divergent samples
        conflict_parts: list[str] = []
        for i, sample in enumerate(samples, 1):
            conflict_parts.append(f"Position {i}: {sample.strip()}")

        conflicts = "\n\n".join(conflict_parts)
        response = (
            f"{template.strip()}\n\n"
            f"Multiple responses to '{query}' produced conflicting information:\n\n"
            f"{conflicts}\n\n"
            f"[CONFLICT_DETECTED] These positions contain material differences "
            f"that require resolution. The correct answer may depend on specific "
            f"context, jurisdiction, or conditions not specified in the query."
        )

        return ActionResult(
            action_taken=ActionTaken.CONFLICT_RESOLUTION,
            response=response,
            intervention_necessary=True,
            cost_tokens_extra=self._estimate_extra_tokens(samples),
        )

    async def _handle_open_ended(
        self, samples: list[str], query: str
    ) -> ActionResult:
        """Handle open-ended uncertainty — distribution of outcomes."""
        config = self._type_configs.get("open_ended")
        template = config.action_config.message_template if config else ""

        # Build distribution response from varied samples
        perspective_parts: list[str] = []
        for i, sample in enumerate(samples, 1):
            perspective_parts.append(f"Perspective {i}: {sample.strip()}")

        perspectives = "\n\n".join(perspective_parts)
        response = (
            f"{template.strip()}\n\n"
            f"The query '{query}' has multiple valid answers:\n\n"
            f"{perspectives}\n\n"
            f"[MULTIPLE_PERSPECTIVES] This question does not have a single "
            f"deterministically correct answer. The above perspectives represent "
            f"the distribution of plausible responses."
        )

        return ActionResult(
            action_taken=ActionTaken.DISTRIBUTION_RESPONSE,
            response=response,
            intervention_necessary=True,
            cost_tokens_extra=self._estimate_extra_tokens(samples),
        )

    @staticmethod
    def _estimate_extra_tokens(samples: list[str]) -> int:
        """Estimate extra token cost from additional samples."""
        if len(samples) <= 1:
            return 0
        extra_chars = sum(len(s) for s in samples[1:])
        return extra_chars // 4  # rough token estimate
