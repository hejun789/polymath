"""Centralized settings loaded from environment / .env via pydantic-settings.

Never call os.getenv() elsewhere — import `settings` from here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str = ""
    tavily_api_key: str = ""
    anthropic_api_key: str = ""  # optional fallback
    chroma_persist_dir: str = "./.chroma"
    log_level: str = "INFO"

    # OpenRouter endpoint (OpenAI-compatible).
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


settings = Settings()
