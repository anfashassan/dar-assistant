"""Loads scraped data, turns it into retrievable chunks and builds derived
knowledge (portfolio catalogue, market statistics) so aggregate questions
("which DarGlobal projects are in Saudi?", "median villa price in Riyadh?")
can be answered from a single retrieved chunk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import CACHE_FILE, VectorCache, chunk_text_for_embedding, get_embedder, text_key
from .retrieval import Chunk, HybridRetriever

log = logging.getLogger("app.knowledge")
CHUNK_CHARS = 1100
SUMMARY_CHARS = 2400  # max size of a generated summary chunk; prompts.py never clips below this
SITE_LABEL = {"darglobal": "DarGlobal", "wasalt": "Wasalt"}


# Sanity bounds for listing prices. Live Wasalt data has a few impossible values: land "for
# sale" at SAR 1,100-9,375 (the per-m2 price shown as the total) and "rentals" of SAR 3M-1.9B
# (sale prices / typos). They would win "cheapest land" questions and skew the medians.
SALE_MIN, SALE_MAX = 50_000, 500_000_000
RENT_MIN, RENT_MAX = 500, 2_000_000   # per period; daily/monthly rents sit well inside this


def plausible_price(l: dict) -> bool:
    p = l.get("price_sar")
    if p is None:
        return True   # "price on request" listings are still useful
    if l.get("purpose") == "rent":
        return RENT_MIN <= p <= RENT_MAX
    return SALE_MIN <= p <= SALE_MAX


def fmt_sar(v) -> str:
    if v in (None, ""):
        return "price on request"
    v = float(v)
    if v >= 1_000_000:
        return f"SAR {v:,.0f} (~{v / 1_000_000:.2f}M)"
    return f"SAR {v:,.0f}"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ------------------------------------------------------------------ chunking
def split_sections(md: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) sections."""
    sections, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path), body))
        buf.clear()

    for line in md.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


def pack(text: str, size: int = CHUNK_CHARS) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph -> hard split on sentence boundary
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(cur) + len(p) + 2 > size and cur:
            out.append(cur)
            cur = cur.split("\n\n")[-1] if len(cur.split("\n\n")[-1]) < size // 4 else ""  # 1-para overlap
        cur = f"{cur}\n\n{p}".strip() if cur else p
    if cur:
        out.append(cur)
    return out


def page_chunks(page: dict) -> list[Chunk]:
    title = page.get("h1") or page.get("title") or page["url"]
    site = page.get("site", "")
    header = f"{SITE_LABEL.get(site, site)} | {title}"
    chunks = []
    text = page.get("text", "")
    if page.get("description") and page["description"][:60] not in text:
        text = page["description"] + "\n\n" + text
    for sec_title, body in split_sections(text) or [("", text)]:
        for part in pack(body):
            if len(part) < 60:
                continue
            ctx = f"{header}" + (f" > {sec_title}" if sec_title and sec_title != title else "")
            chunks.append(Chunk(id="", text=f"{ctx}\n{part}", title=title, url=page["url"],
                                site=site, kind=page.get("type", "page"),
                                meta={"published": page.get("published", "")}))
    return chunks


# ---------------------------------------------------------- structured docs
def listing_text(l: dict) -> str:
    bits = [f"{l.get('property_type') or 'Property'} for {l.get('purpose') or 'sale'}"
            f" in {l.get('district') or '?'}, {l.get('city') or 'Saudi Arabia'}"]
    bits.append(f"Price: {fmt_sar(l.get('price_sar'))}" + (f" ({l['rent_period']})" if l.get("rent_period") else ""))
    for k, label in (("bedrooms", "bedrooms"), ("bathrooms", "bathrooms"), ("living_rooms", "living rooms")):
        if l.get(k) is not None:
            bits.append(f"{l[k]} {label}")
    if l.get("land_area_sqm"):
        bits.append(f"land area {l['land_area_sqm']:.0f} sqm")
    if l.get("built_area_sqm"):
        bits.append(f"built-up area {l['built_area_sqm']:.0f} sqm")
    if l.get("building_year"):
        bits.append(f"built {l['building_year']}")
    for k in ("furnishing", "facade", "usage"):
        if l.get(k):
            bits.append(f"{k.replace('_', ' ')}: {l[k]}")
    if l.get("amenities"):
        bits.append("amenities: " + ", ".join(l["amenities"][:12]))
    if l.get("fixtures"):
        bits.append("fixtures: " + ", ".join(l["fixtures"][:8]))
    if l.get("last_updated"):
        bits.append(f"last updated {l['last_updated']}")
    return f"Wasalt listing: {l.get('title', '')}\n" + "; ".join(bits)


def project_text(p: dict) -> str:
    rows = [("Location", p.get("location")), ("Country", p.get("country")),
            ("Property type", p.get("property_type")), ("Unit types", p.get("unit_types")),
            ("Bedrooms", p.get("bedrooms")), ("Area (sqm)", p.get("area_sqm")),
            ("Status", p.get("status")), ("Completion", p.get("completion")),
            ("Floors", p.get("floors")), ("Units", p.get("units")),
            ("Starting price", p.get("starting_price")), ("Payment plan", p.get("payment_plan"))]
    lines = [f"DarGlobal project: {p.get('name')}"] + [f"- {k}: {v}" for k, v in rows if v]
    if p.get("description"):
        lines.append(p["description"])
    for f in p.get("faqs", [])[:6]:
        lines.append(f"Q: {f['q']}\nA: {f['a']}")
    return "\n".join(lines)


def portfolio_summary(projects: list[dict]) -> list[Chunk]:
    if not projects:
        return []
    by_country = defaultdict(list)
    for p in projects:
        by_country[p.get("country") or "Other / unspecified"].append(p)
    lines = [f"DarGlobal project portfolio overview ({len(projects)} projects listed on darglobal.co.uk/projects), grouped by country:"]
    for country in sorted(by_country, key=lambda c: (-len(by_country[c]), c)):
        items = by_country[country]
        lines.append(f"\n{country} ({len(items)} projects):")
        for p in sorted(items, key=lambda x: x.get("name", "")):
            extra = ", ".join(x for x in (p.get("location"), p.get("property_type"),
                                           f"completion {p['completion']}" if p.get("completion") else "") if x)
            lines.append(f"- {p.get('name')}" + (f" — {extra}" if extra else ""))
    text = "\n".join(lines)
    out = [Chunk(id="", text=part if i == 0 else "DarGlobal project portfolio overview (continued)\n" + part,
                 title="DarGlobal project portfolio overview", url="https://darglobal.co.uk/projects",
                 site="darglobal", kind="summary") for i, part in enumerate(pack(text, SUMMARY_CHARS))]
    # one focused chunk per country: "projects in Oman?" gets exactly the 13 Oman projects
    # (with the full overview a free model once listed Saudi and Spanish projects under Oman)
    for country, items in by_country.items():
        body = [f"DarGlobal projects in {country}: {len(items)} projects listed on darglobal.co.uk/projects "
                f"(complete list; DarGlobal has no other projects in {country}):"]
        for p in sorted(items, key=lambda x: x.get("name", "")):
            extra = ", ".join(x for x in (p.get("location"), p.get("property_type"), p.get("status"),
                                          f"completion {p['completion']}" if p.get("completion") else "") if x)
            body.append(f"- {p.get('name')}" + (f" - {extra}" if extra else ""))
        out.append(Chunk(id="", text="\n".join(body)[:SUMMARY_CHARS], title=f"DarGlobal projects in {country}",
                         url="https://darglobal.co.uk/projects", site="darglobal", kind="summary",
                         meta={"country": country}))
    return out


def market_summary(listings: list[dict]) -> list[Chunk]:
    groups = defaultdict(list)
    for l in listings:
        if l.get("price_sar"):
            groups[(l.get("city") or "Unknown", l.get("purpose") or "sale", l.get("property_type") or "Property")].append(l)
    if not groups:
        return []
    lines = [f"Wasalt market snapshot computed from {len(listings)} scraped Wasalt listings "
             "(statistics of the scraped sample only, not the whole market):"]
    for (city, purpose, ptype), items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        prices = [i["price_sar"] for i in items]
        ppsqm = [i["price_sar"] / (i.get("built_area_sqm") or i.get("land_area_sqm"))
                 for i in items if (i.get("built_area_sqm") or i.get("land_area_sqm"))]
        line = (f"- {ptype} for {purpose} in {city}: {len(items)} listings, median {fmt_sar(statistics.median(prices))}, "
                f"range {fmt_sar(min(prices))} – {fmt_sar(max(prices))}")
        if ppsqm:
            line += f", median ~SAR {statistics.median(ppsqm):,.0f}/sqm"
        lines.append(line)
    cities = defaultdict(int)
    for l in listings:
        cities[l.get("city") or "Unknown"] += 1
    lines.append("Listings per city: " + ", ".join(f"{c} {n}" for c, n in sorted(cities.items(), key=lambda x: -x[1])))
    return [Chunk(id="", text=part, title="Wasalt market snapshot (scraped sample)",
                  url="https://wasalt.sa/en", site="wasalt", kind="summary")
            for part in pack("\n".join(lines), SUMMARY_CHARS)]


# ------------------------------------------------------------- knowledge base
@dataclass
class KnowledgeBase:
    pages: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    listings: list[dict] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    retriever: HybridRetriever | None = None
    source: dict = field(default_factory=dict)  # which dataset each site came from
    manifest: dict = field(default_factory=dict)

    @classmethod
    def load(cls, data_dir: Path, raw_dir: Path | None = None, embedder="auto") -> "KnowledgeBase":
        """embedder: "auto" = process-wide model if available, None = lexical only,
        or any object with embed()/embed_query() (tests use a fake)."""
        raw, seed = raw_dir or data_dir / "raw", data_dir / "seed"
        kb = cls()
        kb._vector_cache_path = raw / CACHE_FILE
        kb._embedder = get_embedder() if embedder == "auto" else embedder
        raw_pages = _read_jsonl(raw / "pages.jsonl")
        seed_pages = _read_jsonl(seed / "pages.jsonl")
        for site in ("darglobal", "wasalt"):
            sp = [p for p in raw_pages if p.get("site") == site]
            if sp:
                kb.pages += sp
                kb.source[site] = "scraped"
            else:
                kb.pages += [p for p in seed_pages if p.get("site") == site]
                kb.source[site] = "seed" if any(p.get("site") == site for p in seed_pages) else "none"
        kb.projects = _read_json(raw / "projects.json") or _read_json(seed / "projects.json")
        listings = _read_json(raw / "listings.json") or _read_json(seed / "listings.json")
        kb.listings = [l for l in listings if plausible_price(l)]
        if len(kb.listings) < len(listings):
            log.info("dropped %d listings with implausible prices (e.g. per-m2 shown as total)",
                     len(listings) - len(kb.listings))
        mf = raw / "manifest.json"
        kb.manifest = json.loads(mf.read_text()) if mf.exists() else {}
        kb.build()
        return kb

    def build(self) -> None:
        chunks: list[Chunk] = []
        listing_urls = {l["url"] for l in self.listings}
        for p in self.pages:
            if p["url"] in listing_urls:  # represented by the structured listing doc instead
                continue
            chunks += page_chunks(p)
        for p in self.projects:
            chunks.append(Chunk(id="", text=project_text(p), title=p.get("name", ""), url=p["url"],
                                site="darglobal", kind="project_card", meta={"project": p}))
        for l in self.listings:
            chunks.append(Chunk(id="", text=listing_text(l), title=l.get("title", ""), url=l["url"],
                                site="wasalt", kind="listing", meta={"listing": l}))
        chunks += portfolio_summary(self.projects) + market_summary(self.listings)

        seen, uniq = set(), []
        for c in chunks:  # exact-duplicate removal (same boilerplate section repeated on many pages)
            body = c.text.split("\n", 1)[-1] if c.kind not in ("project_card", "listing", "summary") else c.text
            h = hashlib.md5(re.sub(r"\W+", "", body.lower()).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            c.id = f"c{len(uniq)}"
            uniq.append(c)
        self.chunks = uniq
        dense, emb = self._dense_vectors(uniq)
        self.retriever = HybridRetriever(uniq, dense=dense, embedder=emb)
        log.info("knowledge base: %d pages, %d projects, %d listings -> %d chunks (%s, %s)",
                 len(self.pages), len(self.projects), len(self.listings), len(uniq), self.source,
                 "hybrid lexical+dense" if dense is not None else "lexical only")

    def _dense_vectors(self, chunks: list[Chunk]):
        emb = getattr(self, "_embedder", None)
        if emb is None or not chunks:
            return None, None
        path = getattr(self, "_vector_cache_path", None)
        try:
            t0 = time.time()
            cache = VectorCache.load(path, getattr(emb, "model_name", "custom")) if path else VectorCache("custom")
            texts = [chunk_text_for_embedding(c.title, c.text) for c in chunks]
            mat, n_new = cache.matrix_for(texts, emb)
            if n_new and path:
                try:  # keep only vectors still in use so the file doesn't grow forever
                    cache.save(path, keep={text_key(t) for t in texts})
                except OSError as e:
                    log.warning("could not write vector cache %s: %s", path, e)
            log.info("dense vectors: %d chunks (%d newly embedded) in %.1fs", len(chunks), n_new, time.time() - t0)
            return mat, emb
        except Exception as e:  # dense retrieval is optional: fall back to lexical
            log.warning("dense vectors unavailable, lexical only: %s", e)
            return None, None

    def stats(self) -> dict:
        kinds = defaultdict(int)
        for p in self.pages:
            kinds[f"{p.get('site')}:{p.get('type')}"] += 1
        return {"pages": len(self.pages), "projects": len(self.projects), "listings": len(self.listings),
                "chunks": len(self.chunks), "pages_by_type": dict(kinds), "data_source": self.source,
                "scraped_at": self.manifest.get("scraped_at")}
