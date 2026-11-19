"""Wasalt-specific parsing (wasalt.sa property portal + blog.wasalt.sa).

Wasalt listing pages render a templated description, e.g.:
    Villa for Sale in Al Yasmeen, Riyadh
    Land Area: 360 SQM
    The property has 3 floors, 7 bedrooms, 8 bathrooms, 5+ living rooms, 3 guest rooms
    Building year: 2024
    Property amenities:  - Maids Room - Garden ...
    Price: 4800000 SAR
    Ref no. 944378 | Last updated: 40 days ago
The parser reads those fields from the extracted text and falls back to JSON-LD /
Next.js data when available.
"""
from __future__ import annotations

import html as htmllib
import re
from urllib.parse import unquote, urlparse

from .extract import find_listing_like

BASE = "https://wasalt.sa"
SEEDS = [
    "/en", "/en/s/wasalt-business", "/en/corporate", "/en/about",
    "/en/sale", "/en/rent", "/en/projects", "/en/daily-rentals", "/en/auction",
]
# from robots.txt; used if robots.txt can't be fetched
SITEMAPS = [
    "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
]
BLOG_BASES =["https://blog.wasalt.sa/en", "https://blog.wasalt.sa"]

PROPERTY_TYPES = [
    "Rest House", "Apartment", "Penthouse", "Townhouse", "Duplex", "Villa", "Floor", "Land",
    "Building", "Office", "Shop", "Showroom", "Warehouse", "Farm", "Chalet", "Studio",
    "Room", "Compound", "Istiraha",
]
_TYPE_RE = "|".join(re.escape(t) for t in PROPERTY_TYPES)


def is_property_url(url: str) -> bool:
    path = unquote(urlparse(url).path)
    return path.startswith("/en/property/") and re.search(r"-\d{4,}$", path) is not None


CITY_CANON = {"aldammam": "Dammam", "al dammam": "Dammam", "dammam": "Dammam",
              "khobar": "Al Khobar", "alkhobar": "Al Khobar", "al khobar": "Al Khobar",
              "almadinah": "Madinah", "al madinah": "Madinah", "madina": "Madinah",
              "medina": "Madinah", "makkah": "Makkah", "mecca": "Makkah", "alkharj": "Al Kharj",
              "makkah al mukarramah": "Makkah", "bariduh": "Buraydah", "alttayif": "Taif",
              "tbwk": "Tabuk", "eanizah": "Unaizah", "alzahran": "Dhahran", "hayil": "Hail",
              "earear": "Arar", "aljumum": "Al Jumum", "almuzahimih": "Al Muzahimiyah",
              "alhufuf": "Al Hofuf"}


def canon_city(city: str) -> str:
    c = re.sub(r"[\s\-]+", " ", (city or "").strip()).lower()
    return CITY_CANON.get(c, CITY_CANON.get(c.replace(" ", ""), (city or "").strip()))


def spread_sample(urls: list[str], n: int) -> list[str]:
    """`n` evenly spaced items, in order."""
    if n <= 0 or len(urls) <= n:
        return list(urls)
    step = len(urls) / n
    return [urls[int(i * step)] for i in range(n)]


def purpose_from_url(url: str) -> str:
    """'sale' / 'rent' from a listing URL (/en/property/<purpose>/...), else ''."""
    m = re.match(r"/en/property/(sale|rent)/", unquote(urlparse(url or "").path).lower())
    return m.group(1) if m else ""


def is_crawlable(url: str) -> bool:
    p = urlparse(url)
    if p.netloc.lower().removeprefix("www.") != "wasalt.sa":
        return False
    path = unquote(p.path)
    if not path.startswith("/en"):
        return False  # English pages only; the LLM answers in Arabic when asked
    if re.search(r"/(login|signup|register|account|profile|favourites?|my-|checkout|payment)", path):
        return False
    return True


def _num(s: str | None) -> float | None:
    if not s:
        return None
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _list_after(text: str, header: str) -> list[str]:
    m = re.search(rf"{header}\s*:?\s*\n((?:\s*[-*•]\s*.+\n?)+)", text, re.I)
    if not m:
        return []
    items = [re.sub(r"^\s*[-*•]\s*", "", l).strip() for l in m.group(1).splitlines()]
    return [i for i in items if i and len(i) < 60]


def parse_listing(page: dict) -> dict | None:
    url = page["url"]
    text = page.get("text", "") or ""
    title = (page.get("h1") or page.get("title") or "").strip()
    title = re.sub(r"\s*[-|]\s*wasalt.*$", "", title, flags=re.I)

    rec: dict = {
        "id": "", "url": url, "title": title, "purpose": "", "property_type": "",
        "city": "", "district": "", "price_sar": None, "rent_period": "",
        "bedrooms": None, "bathrooms": None, "living_rooms": None,
        "land_area_sqm": None, "built_area_sqm": None, "building_year": None,
        "furnishing": "", "facade": "", "street_width_m": None, "usage": "",
        "amenities": [], "fixtures": [], "ref_no": "", "last_updated": "",
        "image": page.get("image", ""),
    }
    m = re.search(r"-(\d{4,})$", unquote(urlparse(url).path))
    if m:
        rec["id"] = m.group(1)

    # trust the URL: every page header says "Properties for Sale"
    rec["purpose"] = purpose_from_url(url)
    if not rec["purpose"]:
        has_sale = re.search(r"\bfor sale\b", text, re.I)
        has_rent = re.search(r"\bfor rent\b", text, re.I)
        rec["purpose"] = "rent" if has_rent else ("sale" if has_sale else "")
    url_purpose = rec["purpose"] if purpose_from_url(url) else ""

    m = re.search(rf"(?:(Furnished|Un-Furnished|Semi-Furnished)\s+)?({_TYPE_RE})\s+for\s+(Sale|Rent)\s+in\s+([^,\n]+),\s*([^\n]+)", text, re.I)
    if m:
        if m.group(1):
            rec["furnishing"] = m.group(1)
        rec["property_type"] = m.group(2).title()
        rec["purpose"] = url_purpose or m.group(3).lower()
        rec["district"] = m.group(4).strip()
        rec["city"] = m.group(5).strip().rstrip(".")
    else:
        t = re.search(rf"\b({_TYPE_RE})\b", title, re.I)
        if t:
            rec["property_type"] = t.group(1).title()

    m = re.search(r"Price\s*:\s*([\d,\.]+)\s*SAR", text, re.I) or \
        re.search(r"\n\s*([\d]{1,3}(?:,\d{3})+)\s*\n\s*SAR", text)
    rec["price_sar"] = _num(m.group(1)) if m else None
    if rec["purpose"] == "rent":
        # the period sits on its own line under the price: "38,000\n/Year"
        m = re.search(r"^\s*/\s*(year|month|week|day)\b", text, re.I | re.M) or \
            re.search(r"\b(yearly|annually|monthly|weekly|daily|per year|per month)\b", text, re.I)
        if m:
            p = m.group(1).lower().replace("per ", "").strip()
            rec["rent_period"] = {"year": "yearly", "annually": "yearly", "month": "monthly",
                                  "week": "weekly", "day": "daily"}.get(p, p)

    for key, word in (("bedrooms", "bedroom"), ("bathrooms", "bathroom"), ("living_rooms", "living room")):
        m = re.search(rf"(\d+)\+?\s*{word}s?\b", text, re.I)
        if m:
            rec[key] = int(m.group(1))
    m = re.search(r"Land Area\s*:\s*([\d,\.]+)\s*SQM", text, re.I)
    rec["land_area_sqm"] = _num(m.group(1)) if m else None
    m = re.search(r"Built[- ]?up Area\s*:\s*([\d,\.]+)\s*SQM", text, re.I)
    rec["built_area_sqm"] = _num(m.group(1)) if m else None
    m = re.search(r"Building year\s*:\s*(\d{4})", text, re.I)
    rec["building_year"] = int(m.group(1)) if m else None
    m = re.search(r"Facade\s*:?[ \t]*([A-Za-z0-9 ]{4,20}?)[ \t]*(?:\n|$)", text) or \
        re.search(r"Facade\s*:?\s*([A-Za-z ]{4,20}?)\s*(?:\n|Street|$)", text)
    rec["facade"] = m.group(1).strip() if m else ""
    m = re.search(r"Street width\s*:?\s*(\d+)\s*m", text, re.I)
    rec["street_width_m"] = int(m.group(1)) if m else None
    m = re.search(r"Property usage\s*:?\s*(Residential|Commercial|Agricultural|Industrial|Mixed)", text, re.I)
    rec["usage"] = m.group(1).title() if m else ""
    if not rec["furnishing"]:
        m = re.search(r"\b(Un-Furnished|Semi-Furnished|Furnished)\b", text)
        rec["furnishing"] = m.group(1) if m else ""
    rec["amenities"] = _list_after(text, r"Property amenities")
    rec["fixtures"] = _list_after(text, r"Property fixtures")
    m = re.search(r"Ref no\.?\s*([\w-]+)\s*\|\s*Last updated\s*:\s*([^\n]+)", text, re.I)
    if m:
        rec["ref_no"], rec["last_updated"] = m.group(1), m.group(2).strip()

    _merge_structured(rec, page)
    rec["city"] = canon_city(rec["city"])
    if rec["price_sar"] is None and not rec["property_type"]:
        return None
    return rec


def _merge_structured(rec: dict, page: dict) -> None:
    """Fill gaps from JSON-LD (schema.org) or Next.js page data."""
    for block in page.get("jsonld") or []:
        offers = block.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        if rec["price_sar"] is None and isinstance(offers, dict):
            rec["price_sar"] = _num(str(offers.get("price", "")))
        if rec["bedrooms"] is None and block.get("numberOfRooms"):
            rec["bedrooms"] = int(_num(str(block["numberOfRooms"])) or 0) or None
        addr = block.get("address") or {}
        if isinstance(addr, dict):
            rec["city"] = rec["city"] or addr.get("addressLocality", "")
            rec["district"] = rec["district"] or addr.get("streetAddress", "")
    nd = page.get("next_data")
    if nd and (rec["price_sar"] is None or not rec["city"]):
        cands = find_listing_like(nd)
        if cands:
            n = normalize_json_listing(cands[0])
            for k, v in n.items():
                if v not in (None, "", []) and rec.get(k) in (None, "", []):
                    rec[k] = v


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            v = v.get("name") or v.get("en") or v.get("title") or v.get("value")
        if v not in (None, "", []):
            return v
    return None


def normalize_json_listing(d: dict) -> dict:
    """Best-effort mapping of an arbitrary listing JSON object to our schema."""
    price = _first(d, "price", "salePrice", "rentPrice", "priceValue", "amount")
    purpose = str(_first(d, "purpose", "listingType", "type", "propertyFor") or "").lower()
    return {
        "id": str(_first(d, "id", "propertyId", "listingId") or ""),
        "title": str(_first(d, "title", "name", "propertyTitle") or ""),
        "price_sar": _num(str(price)) if price is not None else None,
        "purpose": "rent" if "rent" in purpose else ("sale" if "sale" in purpose or "sell" in purpose else ""),
        "property_type": str(_first(d, "propertyType", "propertyTypeName", "category", "subType") or ""),
        "city": str(_first(d, "city", "cityName") or ""),
        "district": str(_first(d, "district", "districtName", "zone", "neighborhood") or ""),
        "bedrooms": _first(d, "bedrooms", "noOfBedrooms", "beds", "bedroomCount"),
        "bathrooms": _first(d, "bathrooms", "noOfBathrooms", "baths", "bathroomCount"),
        "built_area_sqm": _num(str(_first(d, "builtUpArea", "builtArea", "area", "size") or "")),
    }


# ------------------------------------------------------------------- blog API
def wp_post_to_page(post: dict) -> dict:
    """Convert a WordPress REST API post into the common page format."""
    from bs4 import BeautifulSoup
    from .extract import html_to_markdown

    content_html = (post.get("content") or {}).get("rendered", "")
    soup = BeautifulSoup(f"<div>{content_html}</div>", "lxml")
    text = html_to_markdown(soup.div or soup)
    title = htmllib.unescape(re.sub(r"<[^>]+>", "", (post.get("title") or {}).get("rendered", "")))
    excerpt = htmllib.unescape(re.sub(r"<[^>]+>", "", (post.get("excerpt") or {}).get("rendered", ""))).strip()
    return {
        "url": post.get("link", ""),
        "title": title,
        "h1": title,
        "description": excerpt[:300],
        "lang": "ar" if re.search(r"[؀-ۿ]", title) else "en",
        "text": f"# {title}\n\n{text}",
        "published": post.get("date", ""),
        "image": "",
        "jsonld": [],
        "next_data": None,
        "links": [],
    }
