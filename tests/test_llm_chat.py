import asyncio
import json
import sys
from pathlib import Path

import pytest

from app.chat import ChatService, RateLimiter, TTLCache
from app.config import Settings
from app.knowledge import KnowledgeBase
from app.llm import FREE_ROUTER, LLMError, ModelRegistry, OpenRouterClient
from app.refresh import Refresher

DATA = Path(__file__).resolve().parent.parent / "data"


def sse(*texts, model="m/a:free"):
    lines = [": OPENROUTER PROCESSING"]
    lines += ["data: " + json.dumps({"model": model, "choices": [{"delta": {"content": t}}]}) for t in texts]
    return lines + ["data: [DONE]"]


class FakeResp:
    def __init__(self, status, lines=(), body=None):
        self.status_code, self.lines, self.body = status, list(lines), body

    async def aread(self):
        return "\n".join(self.lines).encode()

    async def aiter_lines(self):
        for l in self.lines:
            yield l

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeHTTP:
    def __init__(self, streams, models=None):
        self.streams, self.models, self.calls = list(streams), models or {"data": []}, []

    async def get(self, url, **kw):
        return FakeResp(200, body=self.models)

    def stream(self, method, url, json=None, **kw):
        self.calls.append(json)
        resp = self.streams.pop(0)

        class Ctx:
            async def __aenter__(s):
                if isinstance(resp, Exception):  # simulate a network failure
                    raise resp
                return resp

            async def __aexit__(s, *a):
                return False
        return Ctx()

    async def aclose(self):
        pass


def make_client(streams, models=None, **settings):
    c = OpenRouterClient(Settings(openrouter_api_key="k", **settings))
    c._client = FakeHTTP(streams, models)
    return c


async def collect(agen):
    return [e async for e in agen]


MODELS = {"data": [
    {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "context_length": 128000},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 128000},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 262000},
    {"id": "x/tiny:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 4000},
    {"id": "x/text-embed:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 32000},
    {"id": "x/img:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 32000,
     "architecture": {"output_modalities": ["image"]}},
]}


def test_model_ranking_filters_and_prefers():
    ranked = ModelRegistry().rank(MODELS["data"])
    assert ranked == ["nvidia/nemotron-3-ultra-550b-a55b:free", "meta-llama/llama-3.3-70b-instruct:free"]


def test_discovery_adds_free_router():
    c = make_client([], MODELS)
    cands = asyncio.run(c.refresh_models())
    assert cands[0].startswith("nvidia/") and cands[-1] == FREE_ROUTER


def test_stream_success_and_models_array():
    c = make_client([FakeResp(200, sse("Hello ", "world"))], MODELS)
    asyncio.run(c.refresh_models())
    evs = asyncio.run(collect(c.stream_chat([{"role": "user", "content": "hi"}])))
    assert "".join(e["text"] for e in evs if e["type"] == "delta") == "Hello world"
    assert evs[-1] == {"type": "done", "model": "m/a:free"}
    body = c._client.calls[0]
    assert body["stream"] and body["models"][0] == body["model"]


def test_fallback_to_next_batch_on_429():
    c = make_client([FakeResp(429, ["rate limited"]), FakeResp(200, sse("ok"))], MODELS)
    asyncio.run(c.refresh_models())
    evs = asyncio.run(collect(c.stream_chat([{"role": "user", "content": "hi"}])))
    assert evs[0]["text"] == "ok"
    assert c._client.calls[1]["model"] == FREE_ROUTER


def test_network_error_tries_next_batch():
    import httpx
    c = make_client([httpx.ConnectTimeout("slow network"), FakeResp(200, sse("ok"))], MODELS)
    asyncio.run(c.refresh_models())
    evs = asyncio.run(collect(c.stream_chat([{"role": "user", "content": "hi"}])))
    assert evs[0]["text"] == "ok"


def test_network_outage_degrades_to_retrieval_only(kb):
    """Seen live: httpx.ConnectTimeout used to escape as 'Internal error: ConnectTimeout'."""
    import httpx
    llm = make_client([httpx.ConnectTimeout("x"), httpx.ReadTimeout("y"), httpx.ConnectError("z")], MODELS)
    asyncio.run(llm.refresh_models())
    svc = ChatService(Settings(), kb, llm)
    evs = asyncio.run(collect(svc.answer("Tell me about Trump Tower Jeddah", [])))
    text = "".join(e.get("text", "") for e in evs if e["type"] == "delta")
    assert "temporarily unavailable" in text and evs[-1]["degraded"]


def test_smalltalk_detection():
    from app.chat import is_smalltalk
    for q in ("hi", "Hello!", "who are you", "Who are you?", "thanks", "thank you so much", "why did you message me",
              "what can you do?", "مرحبا", "من أنت؟", "شكرا"):
        assert is_smalltalk(q), q
    for q in ("hi, show me villas in Riyadh", "who are DarGlobal's partners?", "thanks, and what about Jeddah?",
              "where is dar global located", "what is difference between burger and pizza", "help me find a villa"):
        assert not is_smalltalk(q), q


def test_smalltalk_skips_retrieval_and_sources(kb):
    llm = make_client([FakeResp(200, sse("I'm Dar Assistant."))], MODELS)
    asyncio.run(llm.refresh_models())
    svc = ChatService(Settings(), kb, llm)
    evs = asyncio.run(collect(svc.answer("who are you?", [])))
    assert evs[0]["type"] == "meta" and evs[0]["sources"] == [] and evs[0]["listings"] is None
    sent = llm._client.calls[0]["messages"]
    assert "REFERENCE NOTES" not in sent[-1]["content"] and "small talk" in sent[0]["content"]


def test_smalltalk_without_llm_is_friendly(kb):
    svc = ChatService(Settings(), kb, OpenRouterClient(Settings(openrouter_api_key="")))
    evs = asyncio.run(collect(svc.answer("hello", [])))
    text = "".join(e.get("text", "") for e in evs if e["type"] == "delta")
    assert "Dar Assistant" in text and "temporarily unavailable" not in text


def test_prompt_speaks_naturally():
    from app.prompts import SYSTEM_PROMPT, build_messages
    assert "the context" in SYSTEM_PROMPT            # only inside the "never say ..." rule
    assert "Never refer to \"the context\"" in SYSTEM_PROMPT
    msgs = build_messages("q?", [], "[1] x", 3, scraped_at="2026-11-18T20:10:00+00:00")
    assert "REFERENCE NOTES (website snapshot from 2026-11-18)" in msgs[-1]["content"]


def test_bad_key_stops_immediately():
    c = make_client([FakeResp(401, ["bad key"]), FakeResp(200, sse("never"))], MODELS)
    asyncio.run(c.refresh_models())
    with pytest.raises(LLMError):
        asyncio.run(collect(c.stream_chat([{"role": "user", "content": "hi"}])))
    assert len(c._client.calls) == 1


def test_error_event_in_stream():
    lines = ["data: " + json.dumps({"error": {"message": "upstream down", "code": 502}})]
    c = make_client([FakeResp(200, lines), FakeResp(200, sse("recovered"))], MODELS)
    asyncio.run(c.refresh_models())
    evs = asyncio.run(collect(c.stream_chat([{"role": "user", "content": "hi"}])))
    assert evs[0]["text"] == "recovered"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load(DATA, raw_dir=DATA / "seed")  # pinned: see tests/test_rag.py


def test_chat_service_end_to_end_and_cache(kb):
    llm = make_client([FakeResp(200, sse("Here are the best villas within budget: ", "Al Nahdah [L1] and Al Wasam [L2] [1]."))], MODELS)
    asyncio.run(llm.refresh_models())
    svc = ChatService(Settings(openrouter_api_key="k"), kb, llm)
    evs = asyncio.run(collect(svc.answer("villas for sale in Riyadh under 2 million", [])))
    meta = evs[0]
    assert meta["type"] == "meta" and meta["listings"]["total"] > 0
    assert all(i["price_sar"] <= 2e6 for i in meta["listings"]["items"])
    assert evs[-1]["type"] == "done" and not evs[-1]["cached"]
    again = asyncio.run(collect(svc.answer("Villas for sale in Riyadh under 2 million", [])))
    assert again[-1]["cached"] is True  # no second LLM call
    prompt = llm._client.calls[0]["messages"][-1]["content"]
    assert "WASALT LISTINGS" in prompt and "USER MESSAGE:" in prompt


def test_chat_degrades_gracefully_without_llm(kb):
    llm = OpenRouterClient(Settings(openrouter_api_key=""))
    svc = ChatService(Settings(), kb, llm)
    evs = asyncio.run(collect(svc.answer("Tell me about Trump Tower Jeddah", [])))
    text = "".join(e.get("text", "") for e in evs if e["type"] == "delta")
    assert "temporarily unavailable" in text and "Trump Tower Jeddah" in text
    assert evs[-1]["degraded"]


def test_followup_inherits_filters(kb):
    svc = ChatService(Settings(), kb, OpenRouterClient(Settings()))
    hist = [{"role": "user", "content": "villas for sale in Riyadh under 2 million"},
            {"role": "assistant", "content": "..."}]
    r = svc.retrieve("what about the cheapest ones?", hist)
    f = r.listings.query.active()
    assert f["city"] == "Riyadh" and f["price_max"] == 2e6 and f["sort"] == "price_asc"


def test_aggregate_question_pulls_portfolio_summary(kb):
    svc = ChatService(Settings(), kb, OpenRouterClient(Settings()))
    r = svc.retrieve("Which DarGlobal projects are in Oman?", [])
    assert r.chunks[0][0].kind == "summary"
    # natural phrasings that used to miss the summary (answer named 1 of 8 Saudi projects)
    for q in ("What DarGlobal projects are there in Saudi Arabia?", "Tell me about the projects in Dubai"):
        assert svc.retrieve(q, []).chunks[0][0].kind == "summary", q
    # a single-project question doesn't need the portfolio overview
    assert all(c.kind != "summary" for c, _ in svc.retrieve("When will Trump Tower Jeddah be completed?", []).chunks)


def test_rate_limiter_and_cache():
    rl = RateLimiter(2)
    assert rl.allow("a")[0] and rl.allow("a")[0] and not rl.allow("a")[0] and rl.allow("b")[0]
    c = TTLCache(ttl=100, size=2)
    c.set("x", {"v": 1}); c.set("y", {"v": 2}); c.set("z", {"v": 3})
    assert c.get("x") is None and c.get("z") == {"v": 3}


def _fake_scraper(src: Path, pages: int):
    code = (
        "import json,sys,shutil,pathlib;out=pathlib.Path(sys.argv[1]);src=pathlib.Path(sys.argv[2]);"
        "[shutil.copy(p,out/p.name) for p in src.glob('*') if p.is_file()];"
        f"(out/'manifest.json').write_text(json.dumps({{'pages':{pages},'projects':38,'listings':11,"
        "'scraped_at':'2026-11-18T00:00:00+00:00'}))")
    return lambda out: [sys.executable, "-c", code, str(out), str(src)]


def test_refresher_swaps_on_success_and_keeps_data_on_failure(tmp_path, kb):
    swapped = []
    r = Refresher(DATA, swapped.append, interval_h=24, work_dir=tmp_path / "w", size_fn=lambda: 8)
    r.cmd_override = _fake_scraper(DATA / "seed", pages=25)
    status = asyncio.run(r.run_once())
    assert status["last_status"] == "ok" and len(swapped) == 1
    assert swapped[0].stats()["projects"] == 38

    r.cmd_override = _fake_scraper(DATA / "seed", pages=3)   # suspiciously small scrape
    status = asyncio.run(r.run_once())
    assert status["last_status"] == "failed" and "incomplete" in status["last_message"]
    assert len(swapped) == 1  # old data kept


def test_refresher_starts_from_current_dataset(tmp_path):
    """A refresh must see the current data, so a site that fails (e.g. DarGlobal's bot
    protection without a browser) keeps its pages/projects instead of vanishing."""
    import shutil
    data = tmp_path / "data"
    shutil.copytree(DATA / "seed", data / "raw")
    (data / "raw" / "embeddings.npz").write_bytes(b"vector cache")  # so refreshes embed only new chunks
    # fake scraper that does nothing but write a manifest: whatever is in its out dir
    # at that point is exactly what was carried over from the current dataset
    code = ("import sys,json,pathlib; out=pathlib.Path(sys.argv[1]);"
            "assert (out/'projects.json').exists() and (out/'pages.jsonl').exists();"
            "assert (out/'embeddings.npz').read_bytes() == b'vector cache';"
            "(out/'manifest.json').write_text(json.dumps({'pages':25}))")
    swapped = []
    r = Refresher(data, swapped.append, interval_h=24, work_dir=tmp_path / "w", size_fn=lambda: 8)
    r.cmd_override = lambda out: [sys.executable, "-c", code, str(out)]
    status = asyncio.run(r.run_once())
    assert status["last_status"] == "ok", status["last_message"]
    assert swapped[0].stats()["projects"] == len(json.loads((DATA / "seed" / "projects.json").read_text()))


def test_merge_records_never_shrinks():
    from scraper.run import merge_records
    old = [{"id": "1", "price": 10}, {"id": "2", "price": 20}, {"id": "3", "price": 30}]
    new = [{"id": "2", "price": 25}, {"id": "4", "price": 40}]
    merged = {r["id"]: r["price"] for r in merge_records(old, new, "id")}
    assert merged == {"1": 10, "2": 25, "3": 30, "4": 40}   # updated + added, none lost
    assert merge_records(old, [], "id") == old
