"""FastAPI application: chat API (streaming + JSON), health/stats, static UI."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat import ChatService, RateLimiter
from .config import settings
from .knowledge import KnowledgeBase
from .listings import parse_query, search_listings
from .llm import OpenRouterClient
from .refresh import Refresher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"
STARTED = time.time()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[Message] = Field(default_factory=list, max_length=40)
    regenerate: bool = Field(default=False, description="skip the answer cache (UI 'Regenerate')")


class State:
    kb: KnowledgeBase
    llm: OpenRouterClient
    chat: ChatService
    refresher: Refresher
    limiter: RateLimiter


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.kb = KnowledgeBase.load(settings.data_dir)
    state.llm = OpenRouterClient(settings)
    state.chat = ChatService(settings, state.kb, state.llm)
    state.limiter = RateLimiter(settings.rate_limit_per_min)

    def swap(kb: KnowledgeBase):
        state.kb = kb
        state.chat.swap_kb(kb)

    state.refresher = Refresher(settings.data_dir, swap, size_fn=lambda: len(state.kb.pages))
    state.refresher.start(state.kb)
    if state.llm.enabled:
        await state.llm.refresh_models()
    else:
        log.warning("OPENROUTER_API_KEY missing: running in retrieval-only mode")
    yield
    await state.refresher.stop()
    await state.llm.close()


app = FastAPI(
    title="Haya - Property Assistant API",
    description="RAG chatbot over publicly available DarGlobal and Wasalt data, powered by free OpenRouter models.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "anon")


def check_rate(request: Request):
    ok, retry = state.limiter.allow(client_key(request))
    if not ok:
        raise HTTPException(429, f"Too many messages - please wait {retry}s (protects the free model quota).",
                            headers={"Retry-After": str(retry)})


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ chat API
@app.post("/api/chat", summary="Chat (Server-Sent Events stream)")
async def chat_stream(req: ChatRequest, request: Request):
    check_rate(request)
    history = [m.model_dump() for m in req.history]

    async def gen():
        try:
            async for ev in state.chat.answer(req.message.strip(), history, use_cache=not req.regenerate):
                if await request.is_disconnected():
                    break
                yield sse(ev)
        except Exception as e:  # never leave the client hanging
            log.exception("chat failed")
            yield sse({"type": "error", "message": f"Internal error: {e.__class__.__name__}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat/sync", summary="Chat (single JSON response, handy for curl/tests)")
async def chat_sync(req: ChatRequest, request: Request):
    check_rate(request)
    out = {"answer": "", "sources": [], "listings": None, "model": None}
    async for ev in state.chat.answer(req.message.strip(), [m.model_dump() for m in req.history]):
        if ev["type"] == "meta":
            out["sources"], out["listings"] = ev["sources"], ev["listings"]
        elif ev["type"] == "delta":
            out["answer"] += ev["text"]
        elif ev["type"] == "verify":
            out["verification"] = {"checked": ev["checked"], "unsupported": ev["unsupported"]}
        elif ev["type"] == "done":
            out.update(model=ev["model"], latency_ms=ev["latency_ms"], cached=ev.get("cached", False))
    return out


# ------------------------------------------------------------- data endpoints
@app.get("/api/health", summary="Health, dataset and model status")
async def health():
    return {
        "status": "ok",
        "uptime_s": int(time.time() - STARTED),
        "llm": {"configured": state.llm.enabled, "candidates": state.llm.registry.candidates,
                "discovery": state.llm.registry.source},
        "data": state.kb.stats(),
        "refresh": state.refresher.status,
    }


@app.get("/api/search", summary="Inspect hybrid retrieval for a query (explainability)")
async def search(q: str, k: int = 8):
    hits = state.kb.retriever.search(q, k=min(k, 20)) if state.kb.retriever else []
    return {"query": q, "results": ChatService.source_payload(hits)}


@app.get("/api/listings/search", summary="Structured Wasalt listing search from natural language")
async def listing_search(q: str, limit: int = 12):
    lq = parse_query(q, state.chat.districts, state.chat.cities)
    return ChatService.listing_payload(search_listings(state.kb.listings, lq, limit=min(limit, 50)))


@app.get("/api/projects", summary="Structured DarGlobal projects")
async def projects():
    return state.kb.projects


@app.get("/api/suggestions")
async def suggestions(lang: str = "en"):
    if lang == "ar":
        return {"suggestions": [
            "ما هي مشاريع دار جلوبال في السعودية؟",
            "أخبرني عن برج ترامب جدة",
            "من هم شركاء دار جلوبال من العلامات التجارية؟",
            "أرني فلل للبيع في الرياض بأقل من 2 مليون ريال",
            "ما متوسط سعر الفلل في الرياض على وصلت؟",
            "What DarGlobal projects are in Oman?",
        ]}
    s = [
        "Which DarGlobal projects are in Saudi Arabia?",
        "Tell me about Trump Tower Jeddah",
        "What brand partners does DarGlobal work with?",
        "Show me villas for sale in Riyadh under 2 million SAR",
        "What is the median villa price in Riyadh on Wasalt?",
        "ما هي مشاريع دار جلوبال في عُمان؟",
    ]
    return {"suggestions": s}


@app.post("/api/admin/refresh", summary="Trigger a data re-scrape (requires ADMIN_TOKEN)")
async def admin_refresh(x_admin_token: str = Header(default="")):
    if not settings.admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(403, "invalid admin token")
    return await state.refresher.run_once(current_pages=len(state.kb.pages))


# ---------------------------------------------------------------------- UI
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)
