"""Generic, site-agnostic HTML extraction.

Turns a raw HTML page into:
- clean markdown-ish main text (headings + lists preserved, boilerplate removed)
- metadata (title, description, canonical, language, og:image)
- JSON-LD blocks and the Next.js __NEXT_DATA__ payload (if present)
- internal links for crawling
"""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup, Comment, Tag

try:  # markdownify gives nicer structure; plain-text fallback otherwise
    from markdownify import MarkdownConverter
except ImportError:  # pragma: no cover
    MarkdownConverter = None

BOILERPLATE_TAGS = ["script", "style", "noscript", "svg", "iframe", "canvas", "template",
                    "header", "nav", "footer", "form", "button", "select", "input", "picture",
                    "video", "audio", "img"]
BOILERPLATE_HINT = re.compile(
    r"(cookie|consent|gdpr|newsletter|modal|popup|offcanvas|breadcrumb|share-|whatsapp|"
    r"chat-widget|skip-link|navbar|site-header|site-footer|main-menu|nav-menu|mega-menu)", re.I)
ASSET_EXT = re.compile(r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|webm|mp3|zip|woff2?|ttf|pdf|xml|json)(\?|$)", re.I)


def _clean_ws(text: str) -> str:
    text = text.replace("\u200f", "").replace("\u200e", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _dedupe_lines(md: str, min_len: int = 2) -> str:
    """Drop repeated lines (menus/cards repeated many times) and empty markdown noise."""
    out, seen = [], set()
    for line in md.splitlines():
        s = line.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if re.fullmatch(r"[#\-*|>\s]+", s) or len(s) < min_len:
            continue
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key in seen and len(key) < 400:
            continue
        seen.add(key)
        out.append(line.rstrip())
    return "\n".join(out).strip()


def extract_jsonld(soup: BeautifulSoup) -> list[dict]:
    blocks = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        try:
            data = json.loads(raw.strip())
        except (ValueError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict) and "@graph" in it and isinstance(it["@graph"], list):
                blocks.extend(x for x in it["@graph"] if isinstance(x, dict))
            elif isinstance(it, dict):
                blocks.append(it)
    return blocks


def extract_next_data(soup: BeautifulSoup) -> dict | None:
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag:
        return None
    try:
        return json.loads(tag.string or tag.get_text())
    except (ValueError, TypeError):
        return None


def _meta(soup: BeautifulSoup, *names: str) -> str:
    for n in names:
        t = soup.find("meta", attrs={"name": n}) or soup.find("meta", attrs={"property": n})
        if t and t.get("content"):
            return t["content"].strip()
    return ""


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    path = re.sub(r"/{2,}", "/", p.path) or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    q = f"?{p.query}" if p.query else ""
    return f"{p.scheme}://{p.netloc.lower()}{path}{q}"


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    base_host = urlparse(base_url).netloc.lower().removeprefix("www.")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#", "whatsapp:")):
            continue
        absu = urljoin(base_url, href)
        p = urlparse(absu)
        if p.scheme not in ("http", "https"):
            continue
        if p.netloc.lower().removeprefix("www.") != base_host:
            continue
        if ASSET_EXT.search(p.path):
            continue
        links.add(normalize_url(absu))
    return sorted(links)


def _main_node(soup: BeautifulSoup) -> Tag:
    for sel in ("main", "article", "[role=main]", "#__next", "#root", "body"):
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) > 200:
            return node
    return soup.body or soup


INLINE_TAGS = ["a", "span", "label", "strong", "b", "em", "small", "sup", "sub"]


def html_to_markdown(node: Tag) -> str:
    for t in node.find_all(INLINE_TAGS):  # keep adjacent inline texts apart
        t.insert_after(" ")
    if MarkdownConverter is not None:
        md = MarkdownConverter(heading_style="ATX", strip=["a", "img"], bullets="-").convert_soup(node)
    else:  # pragma: no cover
        md = node.get_text("\n")
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)          # leftover images
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)       # leftover links -> text
    md = re.sub(r"\\([_*\[\]()#+\-.!])", r"\1", md)         # markdownify escapes
    return _dedupe_lines(_clean_ws(md))


def parse_html(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    jsonld = extract_jsonld(soup)
    next_data = extract_next_data(soup)

    title = (_meta(soup, "og:title") or (soup.title.get_text(strip=True) if soup.title else "")).strip()
    h1 = soup.find("h1")
    lang = (soup.html.get("lang") if soup.html else "") or ""
    canonical = soup.find("link", rel="canonical")
    links = extract_links(soup, url)

    # strip boilerplate, then pick the main content node
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for t in soup.find_all(BOILERPLATE_TAGS):
        t.decompose()
    for t in soup.find_all(attrs={"role": ["navigation", "banner", "contentinfo", "dialog"]}):
        t.decompose()
    for t in list(soup.find_all(True)):
        if getattr(t, "decomposed", False):
            continue
        attrs = getattr(t, "attrs", None)
        if not attrs:
            continue
        ident = " ".join([str(attrs.get("id", "")), " ".join(attrs.get("class", []) or [])])
        if ident.strip() and BOILERPLATE_HINT.search(ident) and t.name not in ("body", "main", "html"):
            # keep big content containers that merely have a matching class name
            if len(t.get_text(" ", strip=True)) < 1500:
                t.decompose()

    text = html_to_markdown(_main_node(soup))
    return {
        "url": normalize_url(url),
        "canonical": canonical.get("href") if canonical else "",
        "title": title,
        "h1": h1.get_text(" ", strip=True) if h1 else "",
        "description": _meta(soup, "description", "og:description"),
        "image": _meta(soup, "og:image"),
        "lang": lang.lower()[:2],
        "text": text,
        "jsonld": jsonld,
        "next_data": next_data,
        "links": links,
    }


# --------------------------------------------------------------- JSON helpers
PRICE_KEYS = ("price", "Price", "salePrice", "rentPrice", "amount", "priceValue")
ROOM_KEYS = ("bedrooms", "bedroom", "noOfBedrooms", "beds", "rooms", "numberOfRooms", "bedroomCount")


def iter_dicts(obj, depth: int = 0, max_depth: int = 14):
    """Yield every dict nested in a JSON structure (bounded depth)."""
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v, depth + 1, max_depth)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_dicts(v, depth + 1, max_depth)


def find_listing_like(obj) -> list[dict]:
    """Return dicts that look like property listings (have a price and room/area info)."""
    found = []
    for d in iter_dicts(obj):
        keys = set(d.keys())
        if keys & set(PRICE_KEYS) and (keys & set(ROOM_KEYS) or any("area" in k.lower() for k in keys)):
            found.append(d)
    return found
