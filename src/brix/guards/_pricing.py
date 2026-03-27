"""Model pricing table for BudgetGuard cost estimation.

Prices are in USD per token (not per 1K or 1M tokens).
Keep this file up to date as provider prices change.

Source: Provider pricing pages as of 2026-03.
"""

from __future__ import annotations

# Maps model name substring (lowercase) → (input_price_per_token, output_price_per_token)
# Keys must be substrings that uniquely identify a model family.
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI — https://openai.com/pricing
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4-turbo": (10.00e-6, 30.00e-6),
    "gpt-4": (30.00e-6, 60.00e-6),
    "gpt-3.5-turbo": (0.50e-6, 1.50e-6),
    # Anthropic — https://anthropic.com/pricing
    "claude-3-5-sonnet": (3.00e-6, 15.00e-6),
    "claude-3-5-haiku": (0.80e-6, 4.00e-6),
    "claude-3-opus": (15.00e-6, 75.00e-6),
    "claude-3-haiku": (0.25e-6, 1.25e-6),
    # Google — https://ai.google.dev/pricing
    "gemini-1.5-pro": (1.25e-6, 5.00e-6),
    "gemini-1.5-flash": (0.075e-6, 0.30e-6),
}

# Used when the model is not found in the table.
# A zero price means cost tracking will under-count, but it is better than
# crashing or blocking a call because of an unknown model name.
_FALLBACK_PRICE: tuple[float, float] = (0.0, 0.0)


def get_price(model: str) -> tuple[float, float]:
    """Return (input_price_per_token, output_price_per_token) for a model.

    Matches by substring, case-insensitive. Keys are iterated longest-first
    so that "gpt-4o-mini" matches before "gpt-4", preventing silent wrong-price
    bugs from prefix collisions.

    Args:
        model: Model identifier string (e.g. "gpt-4o-mini", "claude-3-5-sonnet-20241022").

    Returns:
        Tuple of (input_price_per_token, output_price_per_token) in USD.
        Returns (0.0, 0.0) for unknown models.
    """
    model_lower = model.lower()
    for key in sorted(PRICES.keys(), key=len, reverse=True):
        if key in model_lower:
            return PRICES[key]
    return _FALLBACK_PRICE


__all__ = ["PRICES", "get_price"]
