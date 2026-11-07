
"""Tiny env-based config. Copy .env.example to .env."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 600
    retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        return cls(
            model=os.getenv("AV_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("OPENAI_API_KEY") or None,
            temperature=float(os.getenv("AV_TEMPERATURE", "0")),
        )
