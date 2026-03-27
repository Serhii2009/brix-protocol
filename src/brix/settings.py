"""BRIX library-level configuration via environment variables.

All settings use the ``BRIX_`` prefix. Engineers can set these once at the
environment level rather than in every wrap() call.

Example::

    export BRIX_LOG_PATH=./traces
    export BRIX_DEFAULT_MODEL=gpt-4o
    export BRIX_MAX_RETRIES=5
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrixSettings(BaseSettings):
    """Library-level configuration sourced from environment variables.

    All variables use the ``BRIX_`` prefix. Unset variables fall back to
    the defaults defined here.

    Attributes:
        log_path: Default path for JSONL audit logs. Overridden per-client
            by the ``log_path`` argument to BRIX.wrap().
        default_model: Default model identifier passed to the LLM client
            when no model is specified in complete().
        max_retries: Default maximum retry count for RetryGuard (Sprint 2).
        embedding_model: Sentence-transformers model used by RegulatedGuard
            for semantic consistency analysis.
    """

    model_config = SettingsConfigDict(
        env_prefix="BRIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_path: str | None = Field(default=None, description="Default JSONL log path")
    default_model: str | None = Field(default=None, description="Default LLM model ID")
    max_retries: int = Field(default=3, ge=0, description="Default retry count")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for RegulatedGuard",
    )
    trace_buffer_size: int = Field(
        default=1000,
        ge=0,
        description="Number of trace entries held in memory by ObservabilityGuard",
    )


@lru_cache(maxsize=1)
def get_settings() -> BrixSettings:
    """Return the cached singleton BrixSettings instance.

    The settings are loaded once from environment variables and cached.
    Call ``get_settings.cache_clear()`` in tests to reload from a fresh env.
    """
    return BrixSettings()


__all__ = ["BrixSettings", "get_settings"]
