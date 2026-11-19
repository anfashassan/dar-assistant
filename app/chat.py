"""Chat orchestration, independent of the web framework.

question -> query understanding -> hybrid retrieval + listing search -> prompt ->
LLM stream (with cache and graceful degradation) -> events for the UI.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import AsyncIterator

from .config import Settings
from .knowledge import KnowledgeBase, fmt_sar
from .listings import ListingResult, parse_query, search_listings
from .llm import LLMError, OpenRouterClient
from .prompts import build_context, build_messages, build_smalltalk_messages, detect_lang
from .retrieval import Chunk, normalize

log = logging.getLogger("app.chat")

FOLLOWUP = re.compile(r"\b(it|its|they|them|their|those|these|same|which one|what about|how about|"
                      r"the (first|second|third|last) one|cheaper|bigger|smaller)\b|"
                      r"هذا|هذه|ذلك|تلك|نفس|ماذا عن|وماذا", re.I)
PLACES = (r"saudi|ksa|riyadh|jeddah|oman|muscat|uae|emirates|dubai|ras al khaimah|rak|qatar|doha|"
          r"spain|marbella|uk|united kingdom|london|abroad|gcc|middle east")
AGG_PROJECTS = re.compile(r"\b(projects?|portfolio|developments?|properties)\b.*\b(list|which|all|how many|where|"
                          r"countries|overview)\b|\b(list|which|what|all|how many|overview)\b.*\b(projects|developments)\b|"
                          rf"\b(projects|developments)\b.*\bin\s+(the\s+)?({PLACES})\b|"
                          r"مشاريع|مشروع", re.I)
AGG_MARKET = re.compile(r"\b(average|median|typical|price range|market|how many listings|price per|cost of)\b|"
                        r"متوسط|اسعار|أسعار|السوق", re.I)
DARGLOBAL_HINT = re.compile(r"dar ?global|trump|aida|neptune|mouawad|fendi|missoni|pagani|aston martin|elie saab|"
                            r"lamborghini|amaya|rayana|astera|marea|tierra viva|les vagues|mulliner|دار جلوبال|ترامب", re.I)


# Whole-message small talk / questions about the assistant itself. Answered conversationally
# without retrieval (otherwise "who are you?" pulls random pages and gets cited answers).
SMALLTALK = re.compile(
    r"(hi|hello|hey|hiya|yo|salam|salaam|good (morning|afternoon|evening|night)|"
    r"thanks?|thank you( so much)?|thx|ok(ay)?|cool|great|nice|awesome|perfect|bye|goodbye|see you|"
    r"who are you|what are you|who made you|who built you|what can you do|what do you do|help|"
    r"how are you( doing)?|what('?s| is) your name|are you (a )?(bot|human|robot|ai|real|chatgpt|claude)|"
    r"why did you (message|text|contact) me|"
    r"مرحبا|مرحباً|اهلا|أهلا|هلا|السلام عليكم|سلام|شكرا|شكراً|من انت|من أنت|كيف حالك|مع السلامة|ماذا تستطيع)"
    r"( there| again| dar assistant| bot)?", re.I)


def is_smalltalk(text: str) -> bool:
    t = re.sub(r"[^\w\s'؀-ۿ]|[؟،؛]", " ", text.strip().lower())   # incl. Arabic ? , ;
    t = re.sub(r"\s+", " ", t).strip()
    return bool(t) and len(t) <= 60 and SMALLTALK.fullmatch(t) is not None


@dataclass
class Retrieved:
    chunks: list[tuple[Chunk, float]]
    listings: ListingResult | None
    search_query: str


class TTLCache:
    def __init__(self, ttl: int, size: int = 300):
        self.ttl, self.size = ttl, size
        self._d: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: str):
        item = self._d.get(key)
        if not item or time.time() - item[0] > self.ttl:
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return item[1]

    def set(self, key: str, value: dict):
        self._d[key] = (time.time(), value)
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


class RateLimiter:
    """Sliding-window limiter per client key (protects the free model quota)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False, int(60 - (now - q[0])) + 1
        q.append(now)
        return True, 0


def _last_user(history: list[dict]) -> str:
    for m in reversed(history):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


class ChatService:
    def __init__(self, settings: Settings, kb: KnowledgeBase, llm: OpenRouterClient):
        self.s = settings
        self.kb = kb
        self.llm = llm
        self.cache = TTLCache(settings.cache_ttl_s)
        self._index_district_city()

    def swap_kb(self, kb: KnowledgeBase):
        self.kb = kb
        self._index_district_city()
        self.cache.clear()

    def _index_district_city(self):
        self.districts = sorted({l["district"] for l in self.kb.listings if l.get("district")})
        self.cities = sorted({l["city"] for l in self.kb.listings if l.get("city")})

    # ----------------------------------------------------------- retrieval
    def retrieve(self, question: str, history: list[dict]) -> Retrieved:
        prev = _last_user(history)
        is_followup = bool(prev) and (len(question.split()) <= 5 or bool(FOLLOWUP.search(question)))
        search_query = f"{prev}\n{question}" if is_followup else question

        chunks = self.kb.retriever.search(search_query, k=self.s.top_k) if self.kb.retriever else []
        chunks = self._add_summaries(question, chunks)

        lq = parse_query(question, self.districts, self.cities)
        if is_followup and prev:
            lq = parse_query(prev, self.districts, self.cities).merged_with(lq)
        about_darglobal = bool(DARGLOBAL_HINT.search(question)) and not re.search(r"wasalt|وصلت", question, re.I)
        listings = None
        if self.kb.listings and lq.intent and not about_darglobal:
            listings = search_listings(self.kb.listings, lq, limit=self.s.max_listings_in_context)
            # listing cards are shown separately; drop duplicate listing chunks from documents
            # the structured search is authoritative for listings: drop listing chunks
            # (which may violate the filters) and keep only a few supporting documents
            chunks = [(c, s) for c, s in chunks if c.kind != "listing"][:4]
        return Retrieved(chunks=chunks, listings=listings, search_query=search_query)

    def _add_summaries(self, question: str, chunks):
        extra = []
        if AGG_PROJECTS.search(question) or (DARGLOBAL_HINT.search(question) and re.search(r"\b(list|all|which|how many)\b", question, re.I)):
            extra += [c for c in self.kb.chunks if c.kind == "summary" and c.site == "darglobal"]
        if AGG_MARKET.search(question):
            extra += [c for c in self.kb.chunks if c.kind == "summary" and c.site == "wasalt"]
        ids = {c.id for c in extra}
        merged = [(c, 1.0) for c in extra] + [(c, s) for c, s in chunks if c.id not in ids]
        return merged[: self.s.top_k + 2]

    # ------------------------------------------------------------ payloads
    @staticmethod
    def source_payload(chunks) -> list[dict]:
        out = []
        for i, (c, score) in enumerate(chunks, 1):
            body = c.text.split("\n", 1)[-1]
            out.append({"n": i, "title": c.title, "url": c.url, "site": c.site, "kind": c.kind,
                        "snippet": re.sub(r"\s+", " ", body)[:220], "score": score})
        return out

    @staticmethod
    def listing_payload(lr: ListingResult | None) -> dict | None:
        if not lr:
            return None
        return {"filters": lr.query.active(), "total": lr.total, "relaxed": lr.relaxed,
                "stats": lr.stats, "items": [
                    {"n": i, **{k: l.get(k) for k in ("title", "url", "purpose", "property_type", "city", "district",
                                                      "price_sar", "rent_period", "bedrooms", "bathrooms",
                                                      "built_area_sqm", "land_area_sqm", "building_year",
                                                      "image", "last_updated")}}
                    for i, l in enumerate(lr.matches, 1)]}

    def _cache_key(self, question: str, history: list[dict]) -> str | None:
        if history:
            return None
        return hashlib.sha1(normalize(re.sub(r"\s+", " ", question.strip())).encode()).hexdigest()

    # --------------------------------------------------------------- answer
    async def answer(self, question: str, history: list[dict]) -> AsyncIterator[dict]:
        if is_smalltalk(question):
            async for ev in self._smalltalk(question, history):
                yield ev
            return
        t0 = time.time()
        r = self.retrieve(question, history)
        yield {"type": "meta", "sources": self.source_payload(r.chunks),
               "listings": self.listing_payload(r.listings), "lang": detect_lang(question)}

        key = self._cache_key(question, history)
        cached = self.cache.get(key) if key else None
        if cached:
            yield {"type": "delta", "text": cached["text"]}
            yield {"type": "done", "model": cached["model"], "cached": True,
                   "latency_ms": int((time.time() - t0) * 1000)}
            return

        messages = build_messages(question, history, build_context(r.chunks, r.listings), self.s.history_turns,
                                  scraped_at=self.kb.manifest.get("scraped_at", ""))
        parts: list[str] = []
        try:
            async for ev in self.llm.stream_chat(messages):
                if ev["type"] == "delta":
                    parts.append(ev["text"])
                    yield ev
                elif ev["type"] == "done":
                    text = "".join(parts)
                    if key and len(text) > 40:
                        self.cache.set(key, {"text": text, "model": ev["model"]})
                    yield {"type": "done", "model": ev["model"], "cached": False,
                           "latency_ms": int((time.time() - t0) * 1000)}
            return
        except LLMError as e:
            log.warning("LLM unavailable: %s", e)
            if parts:  # partial answer already streamed
                yield {"type": "delta", "text": "\n\n_(The response was interrupted by the model provider.)_"}
                yield {"type": "done", "model": "partial", "cached": False,
                       "latency_ms": int((time.time() - t0) * 1000)}
                return
            yield {"type": "delta", "text": self.fallback_answer(question, r, e)}
            yield {"type": "done", "model": "retrieval-only (LLM unavailable)", "cached": False,
                   "latency_ms": int((time.time() - t0) * 1000), "degraded": True}

    async def _smalltalk(self, question: str, history: list[dict]) -> AsyncIterator[dict]:
        t0 = time.time()
        ar = detect_lang(question) == "ar"
        yield {"type": "meta", "sources": [], "listings": None, "lang": "ar" if ar else "en"}
        try:
            async for ev in self.llm.stream_chat(build_smalltalk_messages(question, history, self.s.history_turns)):
                if ev["type"] == "delta":
                    yield ev
                elif ev["type"] == "done":
                    yield {"type": "done", "model": ev["model"], "cached": False,
                           "latency_ms": int((time.time() - t0) * 1000)}
            return
        except LLMError as e:
            log.warning("LLM unavailable for small talk: %s", e)
        yield {"type": "delta", "text": (
            "أنا Dar Assistant، أساعدك في استكشاف مشاريع دار جلوبال وعقارات وصلت. اسألني مثلاً: "
            "«ما هي مشاريع دار جلوبال في الرياض؟»" if ar else
            "I'm Dar Assistant - I help you explore DarGlobal's projects and Wasalt property listings. "
            "Try: *\"Which DarGlobal projects are in Saudi Arabia?\"* or *\"Villas for sale in Riyadh under 2M SAR\"*.")}
        yield {"type": "done", "model": "canned (LLM unavailable)", "cached": False,
               "latency_ms": int((time.time() - t0) * 1000), "degraded": True}

    @staticmethod
    def fallback_answer(question: str, r: Retrieved, err: Exception) -> str:
        """Extractive answer used when no model is reachable (quota exhausted, outage, no key)."""
        ar = detect_lang(question) == "ar"
        lines = ["⚠️ " + ("نموذج اللغة غير متاح حاليًا، إليك أكثر المعلومات صلة من البيانات:" if ar else
                         "The language model is temporarily unavailable (free-tier limit or provider outage). "
                         "Here is the most relevant information from the scraped data:")]
        if r.listings and r.listings.matches:
            lines.append("")
            for i, l in enumerate(r.listings.matches[:5], 1):
                lines.append(f"- **{l.get('property_type') or 'Property'}** in {l.get('district') or '?'}, "
                             f"{l.get('city') or '?'} — {fmt_sar(l.get('price_sar'))}"
                             + (f", {l['bedrooms']} bed" if l.get("bedrooms") is not None else "") + f" [L{i}]")
        for i, (c, _) in enumerate(r.chunks[:4], 1):
            body = re.sub(r"\s+", " ", c.text.split("\n", 1)[-1])[:260]
            lines.append(f"\n**{c.title}** [{i}]\n> {body}…")
        if len(lines) == 1:
            lines.append("No matching information was found." if not ar else "لم يتم العثور على معلومات مطابقة.")
        return "\n".join(lines)
