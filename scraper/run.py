"""Scraper entry point.

Usage:
    python -m scraper.run                       # scrape everything with defaults
    python -m scraper.run --sites darglobal     # one site only
    python -m scraper.run --max-listings 800 --delay 1.0
    python -m scraper.run --render              # use Playwright for JS-only pages

Output (data/raw/):
    pages.jsonl      every scraped page (clean markdown text + metadata)
    projects.json    structured DarGlobal project records
    listings.json    structured Wasalt property listings
    manifest.json    counts + timestamps of the run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from . import darglobal, wasalt
from .extract import normalize_url, parse_html
from .http import PoliteSession, is_challenge

log = logging.getLogger("scraper")
LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)
MIN_TEXT = 80          # pages with less extracted text than this are treated as empty
MAX_RENDER_FAILS = 8   # stop rendering a host after this many failures in a row


def has_content(page: dict | None) -> bool:
    return bool(page) and len(page.get("text", "")) >= MIN_TEXT


# ----------------------------------------------------------------- sitemaps
def sitemap_urls(sess: PoliteSession, base: str, limit: int = 5000,
                 fallback: list[str] | None = None) -> list[str]:
    """Page URLs from sitemaps. `fallback` is used when robots.txt lists none."""
    roots = list(sess.sitemaps_from_robots(base) or fallback or [])
    roots += [urljoin(base, "/sitemap.xml"), urljoin(base, "/sitemap_index.xml")]
    seen_maps, out, queue = set(), [], deque(dict.fromkeys(roots))
    while queue and len(out) < limit and len(seen_maps) < 60:
        sm = queue.popleft()
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        body = sess.get_text(sm)
        if not body or "<loc>" not in body.lower():
            continue
        locs = [l.replace("&amp;", "&") for l in LOC_RE.findall(body)]
        if "<sitemapindex" in body.lower():
            queue.extend(l for l in locs if l not in seen_maps)
        else:
            out.extend(locs)
    log.info("sitemaps for %s: %d maps, %d urls", base, len(seen_maps), len(out))
    return out[:limit]


# -------------------------------------------------------------------- crawl
class Crawler:
    def __init__(self, sess: PoliteSession, workers: int, render: bool):
        self.sess = sess
        self.workers = workers
        self.render = render
        self._pw = None
        self._render_fails: dict[str, int] = {}

    def fetch_page(self, url: str) -> dict | None:
        html = self.sess.get_text(url)
        if not html:
            return None
        try:
            return parse_html(html, url)
        except Exception as e:  # never let one bad page kill the run
            log.warning("parse failed %s: %s", url, e)
            return None

    def _start_browser(self) -> bool:
        """Start Chromium once; if that fails, turn rendering off for the run."""
        ctx = None
        try:
            from playwright.sync_api import sync_playwright
            ctx = sync_playwright().start()
            self._browser = ctx.chromium.launch()
            self._pw = self._browser.new_page(user_agent=self.sess.user_agent)
            self._pw_ctx = ctx
            return True
        except Exception as e:
            log.error("cannot start headless browser, rendering disabled "
                      "(run: python -m playwright install chromium): %s", str(e).splitlines()[0])
            if ctx is not None:
                try:
                    ctx.stop()
                except Exception:
                    pass
            self.render = False
            return False

    def render_page(self, url: str) -> dict | None:
        """Headless-browser fallback for pages whose HTML has no server-rendered text."""
        if not self.render:
            return None
        host = urlparse(url).netloc
        if self._render_fails.get(host, 0) >= MAX_RENDER_FAILS or not self.sess.allowed(url):
            return None
        if self._pw is None and not self._start_browser():
            return None
        page = None
        try:
            self.sess.throttle(url)  # same per-host politeness as plain requests
            self._pw.goto(url, wait_until="domcontentloaded", timeout=45000)
            try:  # sites with analytics/websockets never go fully idle, so cap the wait
                self._pw.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            html = self._pw.content()
            if is_challenge(html):
                log.warning("render blocked by bot protection: %s", url)
            else:
                page = parse_html(html, url)
        except Exception as e:
            log.warning("render failed %s: %s", url, e)
        ok = has_content(page)
        self._render_fails[host] = 0 if ok else self._render_fails.get(host, 0) + 1
        if self._render_fails[host] == MAX_RENDER_FAILS:
            log.error("rendering %s failed %d times in a row - giving up on this host", host, MAX_RENDER_FAILS)
        return page if ok else None

    def close(self):
        if self._pw is not None:
            self._browser.close()
            self._pw_ctx.stop()

    def crawl(self, seeds: list[str], *, allow, priority, max_pages: int,
              max_priority: int | None = None, drop_query: bool = False) -> list[dict]:
        """BFS crawl. `priority(url)` URLs (e.g. listings) are fetched first.
        `drop_query` strips query strings (tracking parameters) instead of skipping those links."""
        hi, lo = deque(), deque()
        seen: set[str] = set()
        n_priority = 0

        def push(u: str):
            u = normalize_url(u)
            if drop_query:
                u = u.split("?", 1)[0]
            if u in seen or not allow(u):
                return
            seen.add(u)
            (hi if priority(u) else lo).append(u)

        for s in seeds:
            push(s)
        pages: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while (hi or lo) and len(pages) < max_pages:
                batch = []
                while (hi or lo) and len(batch) < self.workers * 2:
                    if hi and (max_priority is None or n_priority < max_priority):
                        batch.append(hi.popleft())
                        n_priority += 1
                    elif lo:
                        batch.append(lo.popleft())
                    else:
                        hi.clear()
                        break
                if not batch:
                    break
                for url, page in zip(batch, pool.map(self.fetch_page, batch)):
                    if (page and len(page["text"]) < 250) or (page is None and self.sess.challenged(url)):
                        page = self.render_page(url) or page
                    if not page:
                        continue
                    for link in page["links"]:
                        push(link)
                    if has_content(page):  # thin pages still feed link discovery
                        pages.append(page)
                log.info("  crawled %d pages (queue: %d priority / %d other)", len(pages), len(hi), len(lo))
        return pages


def merge_records(old: list[dict], new: list[dict], key: str) -> list[dict]:
    """Merge by `key`; new records win, old ones not seen this run are kept."""
    merged = {r.get(key) or r.get("url"): r for r in old}
    merged.update({r.get(key) or r.get("url"): r for r in new})
    return list(merged.values())


def slim(page: dict, site: str, kind: str) -> dict:
    return {
        "site": site, "type": kind, "url": page["url"], "title": page.get("title", ""),
        "h1": page.get("h1", ""), "description": page.get("description", ""),
        "lang": page.get("lang", ""), "published": page.get("published", ""),
        "text": page.get("text", ""),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# -------------------------------------------------------------- site drivers
def scrape_darglobal(c: Crawler, max_pages: int) -> tuple[list[dict], list[dict]]:
    base = darglobal.BASE
    log.info("DarGlobal: discovering pages")
    project_paths = set(darglobal.KNOWN_PROJECT_PATHS)
    projects_url = urljoin(base, "/projects")
    listing_page = c.fetch_page(projects_url)
    if not has_content(listing_page):
        listing_page = c.render_page(projects_url) or listing_page
    if listing_page:  # pick up any project added after this code was written
        for link in listing_page["links"]:
            path = urlparse(link).path
            if darglobal.is_project_path(path):
                project_paths.add(path.rstrip("/"))
    seeds = [urljoin(base, p) for p in darglobal.SEEDS + sorted(project_paths)]
    seeds += [u for u in sitemap_urls(c.sess, base) if "/ar/" not in u]

    def allow(u: str) -> bool:
        p = urlparse(u)
        return (p.netloc.removeprefix("www.") == "darglobal.co.uk"
                and not re.match(r"^/(ar|fr|ru|zh|de|es|it)(/|$)", p.path))

    # project cross-links carry "?slug=..."; crawl the clean path instead of skipping them
    pages = c.crawl(seeds, allow=allow, max_pages=max_pages, drop_query=True,
                    priority=lambda u: urlparse(u).path in project_paths)
    docs, projects = [], []
    for p in pages:
        kind = darglobal.page_type(p["url"], project_paths)
        docs.append(slim(p, "darglobal", kind))
        if kind == "project":
            rec = darglobal.parse_project(p)
            if darglobal.is_valid_project(rec):
                projects.append(rec)
            else:
                log.warning("skipping project page with no usable data: %s", p["url"])
    log.info("DarGlobal: %d pages, %d projects", len(docs), len(projects))
    return docs, projects


def scrape_wasalt(c: Crawler, max_pages: int, max_listings: int) -> tuple[list[dict], list[dict]]:
    base = wasalt.BASE
    seeds = [urljoin(base, s) for s in wasalt.SEEDS]
    sm = [u for u in sitemap_urls(c.sess, base, limit=100000, fallback=wasalt.SITEMAPS)
          if wasalt.is_crawlable(u)]
    props = [u for u in sm if wasalt.is_property_url(u)]
    # the sitemap is newest-first; sample across it for variety
    seeds += wasalt.spread_sample(props, max_listings) + [u for u in sm if not wasalt.is_property_url(u)]
    pages = c.crawl(seeds, allow=wasalt.is_crawlable, max_pages=max_pages,
                    priority=wasalt.is_property_url, max_priority=max_listings)
    docs, listings = [], []
    for p in pages:
        if wasalt.is_property_url(p["url"]):
            rec = wasalt.parse_listing(p)
            if rec:  # keep only the structured record: listing pages also carry broker
                listings.append(rec)  # names/phone numbers, which we deliberately don't store
        else:
            docs.append(slim(p, "wasalt", "page"))
    log.info("Wasalt: %d pages, %d listings parsed", len(docs), len(listings))
    return docs, listings


def scrape_wasalt_blog(c: Crawler, max_posts: int) -> list[dict]:
    posts: list[dict] = []
    for b in wasalt.BLOG_BASES:
        page = 1
        while len(posts) < max_posts:
            data = c.sess.get_json(f"{b}/wp-json/wp/v2/posts?per_page=50&page={page}"
                                   "&_fields=link,title,content,excerpt,date")
            if not isinstance(data, list) or not data:
                break
            posts.extend(data)
            page += 1
        if posts:
            break
    docs = [slim(wasalt.wp_post_to_page(p), "wasalt", "blog") for p in posts[:max_posts]]
    if not docs:  # REST API closed -> crawl the blog sitemap instead
        log.info("Wasalt blog API unavailable, falling back to sitemap crawl")
        urls = [u for u in sitemap_urls(c.sess, "https://blog.wasalt.sa") if "/en/" in u]
        pages = c.crawl(urls or ["https://blog.wasalt.sa/en/"],
                        allow=lambda u: "blog.wasalt.sa" in u and "/en/" in u,
                        priority=lambda u: False, max_pages=max_posts)
        docs = [slim(p, "wasalt", "blog") for p in pages]
    log.info("Wasalt blog: %d posts", len(docs))
    return docs


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape public DarGlobal + Wasalt data")
    ap.add_argument("--sites", nargs="+", default=["darglobal", "wasalt", "blog"],
                    choices=["darglobal", "wasalt", "blog"])
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-pages", type=int, default=250, help="max DarGlobal pages")
    ap.add_argument("--max-wasalt-pages", type=int, default=700)
    ap.add_argument("--max-listings", type=int, default=500)
    ap.add_argument("--max-posts", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.75, help="seconds between hits per host")
    ap.add_argument("--render", action="store_true", help="Playwright fallback for JS pages")
    ap.add_argument("--ignore-robots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = PoliteSession(delay=a.delay, respect_robots=not a.ignore_robots)
    crawler = Crawler(sess, workers=a.workers, render=a.render)
    t0 = time.time()

    # merge with an existing run so scraping one site doesn't erase the others
    pages_path = out / "pages.jsonl"
    old = []
    if pages_path.exists():
        old = [json.loads(l) for l in pages_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    def group(p: dict) -> str:
        if p.get("site") == "darglobal":
            return "darglobal"
        return "blog" if p.get("type") == "blog" else "wasalt"

    def load_json(name: str) -> list:
        f = out / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

    old_projects, old_listings = load_json("projects.json"), load_json("listings.json")
    fresh: dict[str, list[dict]] = {}
    projects, listings = old_projects, old_listings
    try:
        if "darglobal" in a.sites:
            fresh["darglobal"], new_projects = scrape_darglobal(crawler, a.max_pages)
            projects = merge_records(old_projects, new_projects, "id")
        if "wasalt" in a.sites:
            fresh["wasalt"], new_listings = scrape_wasalt(crawler, a.max_wasalt_pages, a.max_listings)
            listings = merge_records(old_listings, new_listings, "id")
        if "blog" in a.sites:
            fresh["blog"] = scrape_wasalt_blog(crawler, a.max_posts)
    finally:
        crawler.close()
    # drop records from older runs that the current rules no longer treat as projects
    projects = [p for p in projects if darglobal.is_project_path(p.get("id", ""))]

    # merge by URL: a failed or partial run never removes data
    all_pages = []
    for g in ("darglobal", "wasalt", "blog"):
        prev = [p for p in old if group(p) == g]
        if not fresh.get(g) and g in a.sites:
            log.warning("%s: no pages scraped this run - keeping previous data (%d pages)", g, len(prev))
        all_pages += merge_records(prev, fresh.get(g) or [], "url")

    uniq = {}
    for p in all_pages:  # de-duplicate by URL, drop near-empty pages
        if len(p.get("text", "")) >= 80:
            uniq[p["url"]] = p
    if not any(fresh.values()):
        log.error("No pages scraped - check your network connection, or try --render / --delay 2")
        return 1
    with pages_path.open("w", encoding="utf-8") as f:
        for p in uniq.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "projects.json").write_text(json.dumps(projects, ensure_ascii=False, indent=1), encoding="utf-8")
    listings = list({(l.get("id") or l["url"]): l for l in listings}.values())
    for l in listings:  # also normalises records kept from older runs
        l["city"] = wasalt.canon_city(l.get("city", ""))
        l["purpose"] = wasalt.purpose_from_url(l.get("url", "")) or l.get("purpose", "")
    (out / "listings.json").write_text(json.dumps(listings, ensure_ascii=False, indent=1), encoding="utf-8")

    by = {}
    for p in uniq.values():
        k = f"{p['site']}:{p['type']}"
        by[k] = by.get(k, 0) + 1
    manifest = {
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0, 1),
        "pages": len(uniq), "pages_by_type": by,
        "projects": len(projects), "listings": len(listings),
        "http": sess.stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("done: %s", json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
