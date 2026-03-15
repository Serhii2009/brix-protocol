"""Refusal detection heuristics for LLM response samples.

Detects common patterns indicating the model is refusing to answer,
acknowledging limitations, or deferring to human expertise.
"""

from __future__ import annotations

# Common refusal/deferral phrases found in LLM responses
_REFUSAL_PATTERNS: list[str] = [
    "i cannot",
    "i can't",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i don't have",
    "i do not have",
    "i shouldn't",
    "i should not",
    "consult a professional",
    "consult a doctor",
    "consult a lawyer",
    "consult an attorney",
    "consult a financial advisor",
    "seek professional",
    "seek medical",
    "seek legal",
    "not qualified to",
    "beyond my capabilities",
    "outside my expertise",
    "i'm not a doctor",
    "i'm not a lawyer",
    "i'm not a financial",
    "not medical advice",
    "not legal advice",
    "not financial advice",
    "please consult",
    "please seek",
    "i must decline",
    "i have to decline",
    "i apologize, but i cannot",
    "as an ai",
    "as a language model",
]


def detect_refusal(text: str) -> bool:
    """Check if a single response contains refusal language.

    Args:
        text: A single LLM response text.

    Returns:
        True if refusal language is detected.
    """
    lowered = text.lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def count_refusals(samples: list[str]) -> int:
    """Count how many samples in a list contain refusal language.

    Args:
        samples: List of LLM response texts.

    Returns:
        Number of samples containing refusal patterns.
    """
    return sum(1 for s in samples if detect_refusal(s))
