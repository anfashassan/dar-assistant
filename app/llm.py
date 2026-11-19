"""OpenRouter client with free-model auto-discovery and layered fallback.

Free models on OpenRouter rotate frequently, so the model list is not hard-coded:
1. on startup (and every 6h) the public /models catalogue is fetched and filtered to
   zero-priced text models, ranked by a preference list then by context length;
2. each request sends the top candidates in OpenRouter's `models` array so the
   router itself fails over on rate limits / outages;
3. if the whole request fails before streaming starts, the client retries with the
   next batch of candidates and finally the `openrouter/free` router.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .config import Settings

log = logging.getLogger("app.llm")

# Rough quality preference among free models seen on OpenRouter in 2026. Anything that
# matches earlier patterns is tried first; unknown free models are still used.
PREFERENCE = [
    r"nemotron-3-ultra", r"gpt-oss-120b", r"gemma-4-31b", r"deepseek", r"qwen3.*(235b|max|next)",
    r"llama-4-maverick", r"nemotron-3\.5-super", r"glm", r"kimi", r"mistral-(large|medium)",
    r"gemma-4", r"llama-3\.3-70b", r"qwen", r"nemotron", r"llama", r"mistral",
]
EXCLUDE = re.compile(r"(embed|guard|moderation|tts|whisper|audio|image|vision-only|riverflow|"
                     r"coder|lyria|sora|veo|flux|ocr)", re.I)
FREE_ROUTER = "openrouter/free"
# used when the catalogue can't be fetched; the router alone sometimes returns empty answers
DEFAULT_FREE = ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3.5-super-vl:free"]


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class ModelRegistry:
    candidates: list[str] = field(default_factory=list)
    refreshed_at: float = 0.0
    source: str = "none"

    def rank(self, models: list[dict]) -> list[str]:
        free = []
        for m in models:
            mid = m.get("id", "")
            pricing = m.get("pricing") or {}
            is_free = mid.endswith(":free") or (str(pricing.get("prompt")) in ("0", "0.0")
                                                and str(pricing.get("completion")) in ("0", "0.0"))
            if not is_free or mid == FREE_ROUTER or mid.startswith("openrouter/"):
                continue
            arch = m.get("architecture") or {}
            if "text" not in (arch.get("output_modalities") or ["text"]):
                continue
            if "text" not in (arch.get("input_modalities") or ["text"]):
                continue
            if EXCLUDE.search(mid) or (m.get("context_length") or 0) < 16000:
                continue
            free.append(m)

        def score(m):
            mid = m["id"].lower()
            pref = next((i for i, p in enumerate(PREFERENCE) if re.search(p, mid)), len(PREFERENCE))
            return (pref, -(m.get("context_length") or 0))

        return [m["id"] for m in sorted(free, key=score)]


class OpenRouterClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.registry = ModelRegistry()
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base,
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": settings.app_url,
                "X-Title": settings.app_name,
            },
        )

    @property
    def enabled(self) -> bool:
        return bool(self.s.openrouter_api_key)

    async def close(self):
        await self._client.aclose()

    # ------------------------------------------------------------ discovery
    async def refresh_models(self) -> list[str]:
        if self.s.models:  # explicit configuration wins
            self.registry.candidates = list(self.s.models)
            self.registry.source = "env"
        else:
            try:
                r = await self._client.get("/models")
                r.raise_for_status()
                ranked = self.registry.rank(r.json().get("data", []))
                if ranked:
                    self.registry.candidates = ranked[:8]
                    self.registry.source = "auto-discovered"
            except Exception as e:  # keep the previous list, or the defaults; retry soon
                log.warning("model discovery failed: %s", e)
                if not [m for m in self.registry.candidates if m != FREE_ROUTER]:
                    self.registry.candidates = list(DEFAULT_FREE)
                    self.registry.source = "defaults"
        if FREE_ROUTER not in self.registry.candidates:
            self.registry.candidates.append(FREE_ROUTER)
        self.registry.refreshed_at = time.time()
        log.info("model candidates (%s): %s", self.registry.source, self.registry.candidates)
        return self.registry.candidates

    async def maybe_refresh(self, max_age_s: int = 6 * 3600):
        if self.registry.source in ("none", "defaults"):
            max_age_s = 60   # discovery failed last time
        if time.time() - self.registry.refreshed_at > max_age_s:
            await self.refresh_models()

    def batches(self) -> list[list[str]]:
        c = [m for m in self.registry.candidates if m != FREE_ROUTER]
        out = [c[i:i + 3] for i in range(0, min(len(c), 6), 3)]
        out.append([FREE_ROUTER])
        return out

    # ------------------------------------------------------------- inference
    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[dict]:
        """Yield {"type": "delta", "text"} ... then {"type": "done", "model"}.

        Raises LLMError if every candidate fails before any text was produced.
        """
        if not self.enabled:
            raise LLMError("OPENROUTER_API_KEY is not configured")
        await self.maybe_refresh()
        last_err: LLMError | None = None
        for batch in self.batches():
            produced = False
            try:
                try:
                    async for ev in self._stream_once(messages, batch):
                        produced = produced or ev["type"] == "delta"
                        yield ev
                except httpx.HTTPError as e:
                    # treat timeouts / dropped connections like HTTP errors
                    raise LLMError(f"network error: {e.__class__.__name__}: {e}") from e
                return
            except LLMError as e:
                last_err = e
                if produced:  # mid-answer failure: surface it rather than restarting
                    raise
                log.warning("models %s failed: %s", batch, e)
                if e.status in (401, 403):
                    break  # bad key: no point trying other models
                await asyncio.sleep(0.4)
        raise last_err or LLMError("all models failed")

    async def _stream_once(self, messages: list[dict], models: list[str]) -> AsyncIterator[dict]:
        body = {
            "model": models[0],
            "messages": messages,
            "stream": True,
            "temperature": self.s.temperature,
            "max_tokens": self.s.max_tokens,
            "reasoning": {"exclude": True},
        }
        if len(models) > 1:
            body["models"] = models
        used_model, finish = models[0], None
        async with self._client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode("utf-8", "replace")[:300]
                raise LLMError(f"HTTP {r.status_code}: {text}", r.status_code)
            got_text = False
            async for line in r.aiter_lines():
                if not line or line.startswith(":"):  # keep-alive comments
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if "error" in chunk:
                    err = chunk["error"]
                    raise LLMError(str(err.get("message", err)), err.get("code") if isinstance(err.get("code"), int) else None)
                used_model = chunk.get("model") or used_model
                for choice in chunk.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        got_text = True
                        yield {"type": "delta", "text": delta}
                    if choice.get("finish_reason") == "error":
                        raise LLMError("provider error during generation")
                    finish = choice.get("finish_reason") or finish
            if not got_text:
                raise LLMError("empty completion")
        # "length" = hit max_tokens; None = stream ended without saying why
        yield {"type": "done", "model": used_model, "finish": finish}
