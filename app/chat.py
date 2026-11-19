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
from .verify import verify_answer

log = logging.getLogger("app.chat")

FOLLOWUP = re.compile(r"\b(it|its|they|them|their|those|these|same|which one|what about|how about|"
                      r"the (villas?|project|tower|units?|apartments?|residences?|development|mansions?|penthouses?)|"
                      r"the (first|second|third|last) one|cheaper|bigger|smaller)\b|"
                      r"هذا|هذه|ذلك|تلك|نفس|ماذا عن|وماذا", re.I)
PLACES = (r"saudi|ksa|riyadh|jeddah|oman|muscat|uae|emirates|dubai|ras al khaimah|rak|qatar|doha|"
          r"spain|marbella|uk|united kingdom|london|abroad|gcc|middle east")
AGG_PROJECTS = re.compile(r"\b(projects?|portfolio|developments?|properties)\b.*\b(list|which|all|how many|where|"
                          r"countries|overview)\b|\b(list|which|what|all|how many|overview)\b.*\b(projects|developments)\b|"
                          rf"\b(projects?|developments?)\b.*\bin\s+(the\s+)?({PLACES}|[a-z]{{4,}})\b|"
                          r"مشاريع|مشروع", re.I)
AGG_MARKET = re.compile(r"\b(average|median|typical|price range|market|how many listings|price per|cost of|"
                        r"how much (is|are|does|do)|price of)\b|متوسط|اسعار|أسعار|السوق|كم سعر|بكم", re.I)
SAME_PLACE = re.compile(r"\b(same (city|area|place|location|neighbou?rhood)|nearby|near (it|there|by)|close to it|"
                        r"in that (city|area))\b|نفس المدينة|نفس المنطقة|قريب", re.I)
PRICE_Q = re.compile(r"\bhow much\b|\bprice of\b|\bcost of\b|\b(median|average|typical)\b.*\bprices?\b|"
                     r"كم سعر|بكم|كم يكلف|متوسط سعر|متوسط اسعار|متوسط أسعار", re.I)
DARGLOBAL_HINT = re.compile(r"dar ?global|trump|aida|neptune|mouawad|fendi|missoni|pagani|aston martin|elie saab|"
                            r"lamborghini|amaya|rayana|astera|marea|tierra viva|les vagues|mulliner|دار جلوبال|ترامب", re.I)


# answered without retrieval
_SMALLTALK_PHRASE = (
    r"(hi|hello|hey|hiya|yo|salam|salaam|good (morning|afternoon|evening|night)|"
    r"thanks?|thank you( so much)?|thx|ok(ay)?|cool|great|nice|awesome|perfect|bye|goodbye|see you|"
    r"who are you|what are you|who made you|who built you|what can you do|what do you do|help|"
    r"how are you( doing)?|what('?s| is) your name|are you (a )?(real )?(bot|human|person|robot|ai|real|chatgpt|claude)|"
    r"why did you (message|text|contact) me|"
    r"مرحبا|مرحباً|اهلا|أهلا|هلا|السلام عليكم|سلام|شكرا|شكراً|من انت|من أنت|كيف حالك|مع السلامة|ماذا تستطيع)"
    r"( there| again| haya| bot)?")
# one or more phrases: "hi", "hi how are you", "hello thanks"
SMALLTALK = re.compile(rf"{_SMALLTALK_PHRASE}( (and )?{_SMALLTALK_PHRASE}){{0,3}}", re.I)


def is_smalltalk(text: str) -> bool:
    t = re.sub(r"[^\w\s'؀-ۿ]|[؟،؛]", " ", text.strip().lower())   # incl. Arabic ? , ;
    t = re.sub(r"\s+", " ", t).strip()
    return bool(t) and len(t) <= 60 and SMALLTALK.fullmatch(t) is not None


COUNTRIES = {
    "Saudi Arabia": r"saudi|ksa|riyadh|jeddah|السعودية|المملكة|الرياض|جدة|جده",
    "United Arab Emirates": r"\buae\b|emirates|dubai|ras al khaimah|\brak\b|الامارات|الإمارات|دبي",
    "Oman": r"\boman\b|muscat|\baida\b|عمان|عُمان|مسقط",
    "Qatar": r"qatar|doha|قطر|الدوحة",
    "Spain": r"spain|marbella|benahav|اسبانيا|إسبانيا",
    "United Kingdom": r"\buk\b|united kingdom|britain|england|london|بريطانيا|لندن",
}


def detect_country(text: str) -> str | None:
    for country, pat in COUNTRIES.items():
        if re.search(pat, text, re.I):
            return country
    return None


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
        cards = {c.url: c for c in self.kb.chunks if c.kind == "project_card"}
        self.project_keys: list[tuple[re.Pattern, Chunk]] = []
        for p in self.kb.projects:
            card = cards.get(p.get("url"))
            core = re.sub(r"^the\s+", "", normalize(re.split(r",|\s[-–|]\s", p.get("name", ""))[0]).strip())
            if card and (len(core) >= 4 or re.search(r"\d", core)):
                self.project_keys.append((re.compile(rf"\b{re.escape(core)}\b"), card))

    # ----------------------------------------------------------- retrieval
    def retrieve(self, question: str, history: list[dict]) -> Retrieved:
        prev = _last_user(history)
        is_followup = bool(prev) and (len(question.split()) <= 5 or bool(FOLLOWUP.search(question)))
        search_query = f"{prev}\n{question}" if is_followup else question

        chunks = self.kb.retriever.search(search_query, k=self.s.top_k) if self.kb.retriever else []
        chunks = self._add_summaries(question, chunks, context=search_query)

        lq = parse_query(question, self.districts, self.cities)
        if is_followup and prev:
            base = None
            for earlier in [m["content"] for m in history if m.get("role") == "user" and m.get("content")][-3:]:
                q_ = parse_query(earlier, self.districts, self.cities)
                base = q_ if base is None else base.merged_with(q_)
            lq = base.merged_with(lq)
        if lq.purpose is None and PRICE_Q.search(question):
            lq.purpose = "sale"   # "how much is a villa?" means buying; rents are asked for explicitly
        # "same city" -> city of a project mentioned earlier, only if the user is searching
        own = parse_query(question, self.districts, self.cities)
        wants_search = bool(own.purpose or own.price_max or own.price_min or SAME_PLACE.search(question))
        if not lq.city and is_followup and wants_search:
            recent = " ".join(m["content"] for m in history if m.get("role") == "user" and m.get("content"))
            for card in self._linked_projects(recent[-1500:]):
                city = parse_query(card.meta["project"].get("location", ""), [], self.cities).city
                if city:
                    lq.city = city
                    break
        about_darglobal = bool(DARGLOBAL_HINT.search(question)) and not re.search(r"wasalt|وصلت", question, re.I)
        listings = None
        refines = bool(own.intent or wants_search or own.city or own.district or own.sort != "relevance"
                       or own.bedrooms_exact is not None or own.bedrooms_min is not None
                       or re.search(r"\b(cheaper|bigger|larger|smaller|newer|newest|older)\b|ارخص|أرخص|اكبر|أكبر|احدث|أحدث", question, re.I))
        if is_followup and not refines:
            lq = own   # "how many bedrooms do the villas have?" asks about the project, not a new search
        if self.kb.listings and lq.intent and lq.concrete() and not about_darglobal:
            listings = search_listings(self.kb.listings, lq, limit=self.s.max_listings_in_context)
            # listing cards come from the structured search; keep a few supporting docs
            chunks = [(c, s) for c, s in chunks if c.kind != "listing"][:4]
        return Retrieved(chunks=chunks, listings=listings, search_query=search_query)

    def _linked_projects(self, text: str) -> list:
        nt = normalize(text)
        return [card for pat, card in self.project_keys if pat.search(nt)]

    def _add_summaries(self, question: str, chunks, context: str = ""):
        extra = []
        nq = normalize(question)
        # always include projects the user names
        named = [card for pat, card in self.project_keys if pat.search(nq)]
        if not named and context and context != question:   # follow-up: projects named earlier
            named = [card for pat, card in self.project_keys if pat.search(normalize(context))]
        extra += named[:4]
        country_q = bool(DARGLOBAL_HINT.search(question)) and bool(detect_country(question)) and not named
        if country_q or AGG_PROJECTS.search(question) or (DARGLOBAL_HINT.search(question) and re.search(r"\b(list|all|which|how many)\b", question, re.I)):
            country = detect_country(question)
            port = [c for c in self.kb.chunks if c.kind == "summary" and c.site == "darglobal"]
            focused = [c for c in port if c.meta.get("country") == country] if country else []
            extra += focused or [c for c in port if not c.meta.get("country")]
        if AGG_MARKET.search(question):
            extra += [c for c in self.kb.chunks if c.kind == "summary" and c.site == "wasalt"]
        extra = list({c.id: c for c in extra}.values())
        ids = {c.id for c in extra}
        # country chunks are only added on purpose (above)
        chunks = [(c, s) for c, s in chunks if not c.meta.get("country")]
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
    async def answer(self, question: str, history: list[dict], use_cache: bool = True) -> AsyncIterator[dict]:
        if is_smalltalk(question):
            async for ev in self._smalltalk(question, history):
                yield ev
            return
        t0 = time.time()
        r = self.retrieve(question, history)
        yield {"type": "meta", "mode": "search", "sources": self.source_payload(r.chunks),
               "listings": self.listing_payload(r.listings), "lang": detect_lang(question)}

        context = build_context(r.chunks, r.listings)
        key = self._cache_key(question, history)
        cached = self.cache.get(key) if key and use_cache else None
        if cached:
            yield {"type": "delta", "text": cached["text"]}
            yield verify_answer(cached["text"], context).as_event()
            yield {"type": "done", "model": cached["model"], "cached": True,
                   "latency_ms": int((time.time() - t0) * 1000)}
            return

        messages = build_messages(question, history, context, self.s.history_turns,
                                  scraped_at=self.kb.manifest.get("scraped_at", ""))
        parts: list[str] = []
        try:
            async for ev in self.llm.stream_chat(messages):
                if ev["type"] == "delta":
                    parts.append(ev["text"])
                    yield ev
                elif ev["type"] == "done":
                    text = "".join(parts)
                    complete = ev.get("finish") == "stop"
                    if not complete:
                        log.warning("answer not finished (finish_reason=%s)", ev.get("finish"))
                        yield {"type": "delta", "text": "\n\n_" + (
                            "الإجابة لم تكتمل، اضغط إعادة التوليد." if detect_lang(question) == "ar"
                            else "The answer was cut short. Tap regenerate to try again.") + "_"}
                    elif key and len(text) > 40:
                        self.cache.set(key, {"text": text, "model": ev["model"]})
                    check = verify_answer(text, context)
                    if check.unsupported:
                        log.warning("figures not found in sources: %s", check.unsupported)
                    yield check.as_event()
                    yield {"type": "done", "model": ev["model"], "cached": False,
                           "latency_ms": int((time.time() - t0) * 1000), "truncated": not complete}
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
        yield {"type": "meta", "mode": "chat", "sources": [], "listings": None, "lang": "ar" if ar else "en"}
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
            "أنا هيا، أساعدك في استكشاف مشاريع دار جلوبال وعقارات وصلت. اسألني مثلاً: "
            "«ما هي مشاريع دار جلوبال في الرياض؟»" if ar else
            "I'm Haya - I help you explore DarGlobal's projects and Wasalt property listings. "
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
