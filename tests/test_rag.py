from pathlib import Path

import pytest

from app.knowledge import KnowledgeBase, pack, split_sections
from app.listings import parse_query, search_listings
from app.prompts import build_context, detect_lang

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def kb():
    # seed snapshot keeps these stable; scripts/eval.py measures the scraped data
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
    """Summaries must not be clipped like ordinary chunks."""
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


def test_city_name_is_not_a_district():
    # some listings have "Riyadh" as their district
    q = parse_query("villas for sale in Riyadh under 2 million", ["Riyadh", "Al Narjis", "Al Malqa"])
    assert q.city == "Riyadh" and q.district is None


def test_budget_results_are_diverse():
    from app.listings import ListingQuery
    listings = [{"id": str(i), "url": f"u{i}", "city": "Riyadh", "property_type": "Villa", "purpose": "sale",
                 "district": "Al-Mahdiyah" if i < 20 else f"D{i % 7}", "price_sar": 2_000_000 - i * 10_000}
                for i in range(80)]
    r = search_listings(listings, ListingQuery(city="Riyadh", property_type="Villa", price_max=2_000_000, intent=True))
    districts = [l["district"] for l in r.matches]
    prices = [l["price_sar"] for l in r.matches]
    assert len(r.matches) == 6 and r.total == 80
    assert max(districts.count(d) for d in set(districts)) <= 2          # no single district dominates
    assert max(prices) - min(prices) >= 300_000                           # spread across the budget


def test_implausible_prices_are_dropped(tmp_path):
    """Live data: Dammam land 'for sale' at SAR 1,828 (per-m2 price) and a 'rental' at SAR 1.9B."""
    import json, shutil
    from app.knowledge import plausible_price
    assert not plausible_price({"purpose": "sale", "price_sar": 1828})
    assert not plausible_price({"purpose": "rent", "price_sar": 1_906_550_000})
    assert plausible_price({"purpose": "sale", "price_sar": 775_000}) and plausible_price({"purpose": "rent", "price_sar": 38_000})
    assert plausible_price({"purpose": "sale", "price_sar": None})          # price on request is kept
    raw = tmp_path / "raw"
    shutil.copytree(DATA / "seed", raw)
    good = json.loads((raw / "listings.json").read_text())
    bad = [{**good[0], "id": "x1", "url": "u-x1", "price_sar": 1828}]
    (raw / "listings.json").write_text(json.dumps(good + bad))
    kb = KnowledgeBase.load(DATA, raw_dir=raw, embedder=None)
    assert len(kb.listings) == len(good) and all(l["id"] != "x1" for l in kb.listings)


def test_district_articles_and_number_words():
    d = ["An-Narjis", "Al Yasmeen", "City Center"]
    assert parse_query("villa in al narjis riyadh with at least 5 bedrooms", d).district == "An-Narjis"
    assert parse_query("same city under 3 million", d).district is None          # generic words never match
    assert parse_query("three bedroom apartment in Jeddah").bedrooms_exact == 3
    assert parse_query("شقق للإيجار في جدة بثلاث غرف").bedrooms_exact == 3
    assert parse_query("شقة غرفتين للبيع في الرياض").bedrooms_exact == 2


def test_more_district_traps_from_live_data():
    d = ["Al-Riyadh", "The Wasam", "Al Samer", "An-Narjis"]   # Al-Riyadh is a district of Jeddah
    assert parse_query("villas for sale in Riyadh", d).district is None
    assert parse_query("villas in the same city under 3 million", d).district is None
    assert parse_query("villa in al samer", d).district == "Al Samer"
    assert parse_query("I have 1.5 million riyals, what can I buy in Jeddah?").price_max == 1_500_000


def test_property_word_is_not_a_district():
    # there is a Jeddah district called "The Apartment"
    q = parse_query("What's the median apartment price in Jeddah?", ["The Apartment", "Al Rayaan"])
    assert q.district is None and q.property_type == "Apartment" and q.city == "Jeddah"
