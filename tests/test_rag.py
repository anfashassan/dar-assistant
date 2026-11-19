from pathlib import Path

import pytest

from app.knowledge import KnowledgeBase, pack, split_sections
from app.listings import parse_query, search_listings
from app.prompts import build_context, detect_lang

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def kb():
    # pinned to the bundled seed snapshot so results don't change with each live scrape;
    # retrieval quality on live data is measured by scripts/eval.py
    return KnowledgeBase.load(DATA, raw_dir=DATA / "seed")


def test_official_project_page_wins_near_tie():
    """A third-party blog post about a project shouldn't outrank the developer's own page."""
    from app.retrieval import Chunk, HybridRetriever
    filler = [Chunk(f"f{i}", f"Unrelated text about apartments number {i} in Riyadh", f"Filler {i}",
                    f"https://x/{i}", "wasalt", "page") for i in range(20)]
    project = Chunk("p", "Trump Tower Jeddah seafront tower with 47 floors, completion December 2029",
                    "Trump Tower Jeddah", "https://darglobal.co.uk/trump-tower-jeddah", "darglobal", "project")
    blog = Chunk("b", "Jeddah Trump Tower: a stunning collaboration, completion expected soon in Jeddah",
                 "Jeddah Trump Tower blog", "https://blog.wasalt.sa/en/jeddah-trump-tower/", "wasalt", "blog")
    r = HybridRetriever(filler + [blog, project])
    top = r.search("When will Trump Tower Jeddah be completed?", k=3)
    assert top[0][0].id == "p"
    assert r.search("apartments number 7", k=1)[0][0].id == "f7"   # prior doesn't hijack other queries


def test_summary_chunks_reach_prompt_in_full():
    """The portfolio summary's last section (Saudi Arabia) used to be clipped away."""
    from app.knowledge import SUMMARY_CHARS
    from app.retrieval import Chunk
    text = ("Oman (13 projects):\n" + "- some project line here\n" * 80)[:SUMMARY_CHARS - 40] + "\nSaudi Arabia: Neptune"
    summary = Chunk("s", text, "DarGlobal project portfolio overview", "https://darglobal.co.uk/projects", "darglobal", "summary")
    page = Chunk("p", "x " * 1500 + "TAIL", "Some page", "https://x", "darglobal", "page")
    ctx = build_context([(summary, 1.0), (page, 0.5)], None)
    assert "Saudi Arabia: Neptune" in ctx       # summary kept whole
    assert "TAIL" not in ctx                    # ordinary chunks still clipped


def test_intent_keywords_reach_near_empty_pages():
    """Live /careers page is ~118 chars with no word 'job'; intent keywords bridge that."""
    from app.retrieval import Chunk, HybridRetriever, expand_query
    filler = [Chunk(f"f{i}", f"Investment opportunities in Saudi real estate, report {i}", f"Blog {i}",
                    f"https://x/{i}", "darglobal", "blog") for i in range(20)]
    careers = Chunk("c", "Find your next great opportunity. Filter by. Showing 0 results",
                    "Careers", "https://darglobal.co.uk/careers", "darglobal", "page")
    r = HybridRetriever(filler + [careers])
    for q in ("are there any job opportunities at DarGlobal?", "هل توجد وظائف في دار جلوبال؟"):
        assert "c" in [c.id for c, _ in r.search(q, k=3)], q
    assert "headquartered" in expand_query("where is dar global located")
    assert "careers" not in expand_query("Tell me about Trump Tower Jeddah")   # no false triggers


def test_kb_builds(kb):
    s = kb.stats()
    assert s["projects"] >= 30 and s["chunks"] > 20
    assert any(c.kind == "summary" and c.site == "darglobal" for c in kb.chunks)


@pytest.mark.parametrize("q,expected", [
    ("Trump Tower Jeddah completion", "Trump Tower Jeddah"),
    ("Neptune Mouawad villas", "Neptune"),
    ("Lamborghini inspired villas in Spain", "Tierra Viva"),
    ("what is wasalt", "Wasalt"),
    ("متى موعد تسليم برج ترامب جدة؟", "Trump Tower Jeddah"),   # Arabic -> English retrieval
    ("trump towr jedah", "Trump Tower Jeddah"),                  # typos
])
def test_retrieval_top3(kb, q, expected):
    titles = [c.title for c, _ in kb.retriever.search(q, k=3)]
    assert any(expected.lower() in t.lower() for t in titles), titles


@pytest.mark.parametrize("q,expect", [
    ("3 bedroom villa in Riyadh under 2 million", dict(property_type="Villa", city="Riyadh", bedrooms_exact=3, price_max=2e6)),
    ("apartments for rent in Jeddah below 60k", dict(purpose="rent", property_type="Apartment", city="Jeddah", price_max=6e4)),
    ("شقة للإيجار في جدة اقل من 50 الف", dict(purpose="rent", property_type="Apartment", city="Jeddah", price_max=5e4)),
    ("villa between 1.5m and 3m with pool", dict(price_min=1.5e6, price_max=3e6, amenities=["pool"])),
    ("4+ bedrooms villa", dict(bedrooms_min=4)),
    ("cheapest land in Riyadh", dict(property_type="Land", sort="price_asc")),
])
def test_query_parser(q, expect):
    got = parse_query(q).active()
    for k, v in expect.items():
        assert got.get(k) == v, (k, got)


def test_parser_no_false_intent():
    assert not parse_query("tell me about trump tower jeddah").intent


def test_fuzzy_district():
    assert parse_query("villa in al yasmin", ["Al Yasmeen", "Tuwaiq"]).district == "Al Yasmeen"


def test_currency_word_is_not_a_district():
    # "SAR" used to fuzzy-match the live district "Al Samer" (ratio 0.75), silently
    # filtering every price question to one district
    districts = ["Al Samer", "Al Yasmeen", "Al Sahafa", "Al Malqa"]
    q = parse_query("Show me villas for sale in Riyadh under 2 million SAR", districts)
    assert q.district is None and q.price_max == 2_000_000 and q.city == "Riyadh"
    assert parse_query("apartments in al samer under 900k sar", districts).district == "Al Samer"
    assert parse_query("villa in malqa", districts).district == "Al Malqa"


def test_listing_search_and_relaxation(kb):
    r = search_listings(kb.listings, parse_query("villa in Riyadh under 2 million"))
    assert r.total > 0 and all(l["price_sar"] <= 2e6 for l in r.matches)
    r = search_listings(kb.listings, parse_query("villa in Riyadh with pool under 1.2m"))
    assert r.relaxed  # nothing with a pool -> filters relaxed, closest options returned
    r = search_listings(kb.listings, parse_query("cheapest villa in Riyadh"))
    prices = [l["price_sar"] for l in r.matches]
    assert prices == sorted(prices)


def test_context_and_lang(kb):
    hits = kb.retriever.search("Trump Tower Jeddah", k=2)
    lr = search_listings(kb.listings, parse_query("villa in Riyadh under 2m"))
    ctx = build_context(hits, lr)
    assert "[1]" in ctx and "[L1]" in ctx and "WASALT LISTINGS" in ctx
    assert detect_lang("ما هي مشاريع دار جلوبال؟") == "ar" and detect_lang("hello") == "en"


def test_chunking():
    secs = split_sections("# A\nintro\n## B\nbody b\n## C\nbody c")
    assert [s[0] for s in secs] == ["A", "A > B", "A > C"]
    parts = pack("para one. " * 300, size=500)
    assert all(len(p) <= 520 for p in parts) and len(parts) > 3
