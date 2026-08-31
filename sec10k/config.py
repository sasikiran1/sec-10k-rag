"""Typed application settings, loaded once from the environment / .env file.

Nothing else in the codebase should read os.environ directly — import `settings`
from here instead, so every config value has one definition and one type.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the app.

    Values come from environment variables (case-insensitive), falling back to a
    local .env file. Fields without a default are required — the app should fail
    loudly at startup if they're missing, not deep inside a request.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["groq", "gemini"] = "groq"

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"

    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-2.0-flash"

    database_url: str = "postgresql://sec10k:sec10k@localhost:5433/sec10k"

    def active_api_key(self) -> str:
        """Return the API key for whichever provider `llm_provider` selects.

        Raise RuntimeError if that provider's key is empty — this is the "fail at
        startup, not mid-request" check.
        """
        if self.llm_provider == "groq":
            key = self.groq_api_key
        else:
            key = self.gemini_api_key

        if not key:
            raise RuntimeError(
                f"No API key set for provider '{self.llm_provider}'. Add it to .env."
            )
        return key

    def active_base_url(self) -> str:
        """Return the base_url for the selected provider."""
        if self.llm_provider == "groq":
            return self.groq_base_url
        return self.gemini_base_url

    def active_model(self) -> str:
        """Return the model name for the selected provider."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.gemini_model


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide singleton Settings instance.

    lru_cache means the .env file is read once, not on every import/call.
    """
    return Settings()
