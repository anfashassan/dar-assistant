"""Runtime configuration (12-factor: everything comes from environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    # Optional explicit model list (comma separated). Empty -> auto-discover free models.
    models: list[str] = field(default_factory=lambda: _list("OPENROUTER_MODELS"))
    app_url: str = field(default_factory=lambda: os.getenv("APP_URL", "https://huggingface.co/spaces"))
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "DarGlobal x Wasalt Property Assistant"))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT / "data"))))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "6")))
    max_listings_in_context: int = field(default_factory=lambda: int(os.getenv("MAX_LISTINGS", "6")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "900")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "6")))

    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN", "12")))
    cache_ttl_s: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_S", "3600")))
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "60")))


settings = Settings()
