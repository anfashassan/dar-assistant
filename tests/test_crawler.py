"""Crawler integration test against a local mini-site (robots.txt, sitemap index, links)."""
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scraper.http import PoliteSession
from scraper.run import Crawler, sitemap_urls

PAGE = "<html><head><title>{t}</title></head><body><nav><a href='/'>Home</a></nav><main><h1>{t}</h1><p>{body}</p>{links}</main></body></html>"


@pytest.fixture()
def site(tmp_path):
    long = "Luxury residences with sea views and private amenities. " * 8
    files = {
        "robots.txt": "User-agent: *\nDisallow: /private\nSitemap: http://127.0.0.1:{port}/sitemap_index.xml\n",
        "sitemap_index.xml": "<sitemapindex><sitemap><loc>http://127.0.0.1:{port}/pages.xml</loc></sitemap></sitemapindex>",
        "pages.xml": "<urlset><url><loc>http://127.0.0.1:{port}/a.html</loc></url><url><loc>http://127.0.0.1:{port}/private/x.html</loc></url></urlset>",
        "a.html": PAGE.format(t="Page A", body=long, links="<a href='/b.html'>B</a><a href='/doc.pdf'>pdf</a>"),
        "b.html": PAGE.format(t="Page B", body=long, links="<a href='/a.html'>A</a><a href='https://other.com/x'>ext</a>"
                                                           "<a href='/c.html?slug=b'>C</a>"),
        "c.html": PAGE.format(t="Page C", body=long, links=""),
        "extra.xml": "<urlset><url><loc>http://127.0.0.1:{port}/b.html</loc></url></urlset>",
        "challenge.html":(Path(__file__).parent / "fixtures" / "incapsula_challenge.html").read_text(),
    }
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "x.html").write_text(PAGE.format(t="Secret", body=long, links=""))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path)))
    port = srv.server_address[1]
    for name, content in files.items():
        (tmp_path / name).write_text(content.replace("{port}", str(port)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def test_crawl_respects_robots_and_follows_links(site):
    sess = PoliteSession(delay=0)
    urls = sitemap_urls(sess, site)
    assert f"{site}/a.html" in urls
    pages = Crawler(sess, workers=2, render=False).crawl(
        urls, allow=lambda u: u.startswith(site), priority=lambda u: False, max_pages=10)
    titles = {p["title"] for p in pages}
    assert {"Page A", "Page B"} <= titles             # B discovered via link
    assert "Secret" not in titles                     # /private blocked by robots.txt
    assert sess.stats["blocked_by_robots"] >= 1
    a = next(p for p in pages if p["title"] == "Page A")
    assert "Home" not in a["text"]                     # nav stripped
    assert not any(l.endswith(".pdf") or "other.com" in l for l in a["links"])


def test_partial_rerun_keeps_previous_pages(tmp_path, monkeypatch):
    """An interrupted re-scrape updates pages it reached and keeps the rest."""
    import json
    from scraper import run

    def page(url, text):
        return {"site": "darglobal", "type": "news", "url": url, "text": text}
    body = "x" * 100
    (tmp_path / "pages.jsonl").write_text("\n".join(json.dumps(page(f"https://d/{i}", body)) for i in range(5)))
    monkeypatch.setattr(run, "scrape_darglobal", lambda c, n: ([page("https://d/0", "new " + body)], []))
    assert run.main(["--sites", "darglobal", "--out", str(tmp_path)]) == 0
    pages = {p["url"]: p["text"] for p in map(json.loads, (tmp_path / "pages.jsonl").read_text().splitlines())}
    assert len(pages) == 5 and pages["https://d/0"].startswith("new")


def test_sitemap_fallback_when_robots_lists_none(site):
    """If robots.txt can't be read (network blip), known sitemaps are still crawled."""
    sess = PoliteSession(delay=0)
    sess.sitemaps_from_robots = lambda base: []          # simulate robots.txt failure
    assert f"{site}/b.html" not in sitemap_urls(sess, site)   # only reachable via extra.xml
    assert f"{site}/b.html" in sitemap_urls(sess, site, fallback=[f"{site}/extra.xml"])


def test_challenge_page_is_not_a_page(site):
    """HTTP 200 bot-protection interstitials must count as blocked, not as content."""
    sess = PoliteSession(delay=0)
    pages = Crawler(sess, workers=2, render=False).crawl(
        [f"{site}/challenge.html", f"{site}/a.html"], allow=lambda u: u.startswith(site),
        priority=lambda u: False, max_pages=10)
    assert all(p["url"] != f"{site}/challenge.html" for p in pages)
    assert sess.stats["challenged"] == 1 and sess.challenged(site + "/anything")


def test_browser_launch_failure_disables_render(monkeypatch):
    """If Chromium is missing, rendering turns off cleanly instead of erroring on every page."""
    pw_api = pytest.importorskip("playwright.sync_api")  # optional dependency

    class BrokenPW:
        def start(self):
            raise RuntimeError("Executable doesn't exist")

    monkeypatch.setattr(pw_api, "sync_playwright", lambda: BrokenPW())
    crawler = Crawler(PoliteSession(delay=0), workers=1, render=True)
    assert crawler.render_page("http://127.0.0.1:9/x") is None
    assert crawler.render is False
    assert crawler.render_page("http://127.0.0.1:9/y") is None   # no second launch attempt
    crawler.close()


def test_challenged_host_falls_back_to_render(site, monkeypatch):
    sess = PoliteSession(delay=0)
    crawler = Crawler(sess, workers=1, render=True)
    rendered = []

    def fake_render(url):
        rendered.append(url)
        return {"url": url, "title": "Rendered", "text": "Real content after JS. " * 10, "links": []}

    monkeypatch.setattr(crawler, "render_page", fake_render)
    pages = crawler.crawl([f"{site}/challenge.html"], allow=lambda u: u.startswith(site),
                          priority=lambda u: False, max_pages=5)
    assert rendered == [f"{site}/challenge.html"]
    assert [p["title"] for p in pages] == ["Rendered"]


def test_query_links_followed_as_clean_paths(site):
    """DarGlobal project cross-links carry ?slug=...; with drop_query they are crawled, not skipped."""
    pages = Crawler(PoliteSession(delay=0), workers=2, render=False).crawl(
        [f"{site}/b.html"], allow=lambda u: u.startswith(site), priority=lambda u: False,
        max_pages=10, drop_query=True)
    assert f"{site}/c.html" in {p["url"] for p in pages}
