"""DarGlobal-specific parsing (darglobal.co.uk).

Project pages expose a spec block such as:
    Property Type   Seafront Residences
    Unit Type       1, 2, 3, 4 Bedroom Apartments & 3, 4 Bedroom Penthouses
    Location        Jeddah Corniche, Saudi Arabia
    Completion Date December 2029
    Number of Floors 47
    Area (sqm)      70 to 365
The markup varies between projects, so the parser works on the extracted text and
accepts "Label value", "Label: value" and "Label\\nvalue" layouts.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

BASE = "https://darglobal.co.uk"
SEEDS = [
    "/", "/projects", "/about", "/investor", "/why-invest", "/press", "/blog",
    "/luxury-golf-communities", "/hospitality", "/exclusive-membership", "/commercial",
    "/tokenization",
    "/partners/aston-martin", "/partners/elie-saab", "/partners/fendi-casa",
    "/partners/lamborghini", "/partners/marriott-residences-aida-oman", "/partners/missoni",
    "/partners/mouawad", "/partners/pagani", "/partners/the-trump-organization",
    "/partners/w-hotels", "/partners/paramount",
]
# fallback list; new projects are also discovered from /projects
KNOWN_PROJECT_PATHS = [
    "/trump-international-hotel-and-tower-dubai", "/d-villas-at-jge", "/the-astera", "/dg1",
    "/davinci-tower-by-pagani", "/urban-oasis-by-missoni", "/w-residences",
    "/one-of-one/project-pagani", "/one-of-one/top-of-the-astera", "/tierra-viva",
    "/one-of-one/painite-villas-design-inspired-by-Automobili-Lamborghini", "/marea",
    "/les-vagues", "/lp/trump-international-qatar", "/aida-trump-international-hotel",
    "/aida-trump-international-cliff-villas", "/aida-trump-villas", "/landing-page/aida",
    "/fairway-villas-aida-oman", "/one-of-one/fendi-casa-azure-oceanfront-villas",
    "/amour-sans-detour", "/the-great-escape-1-aida-oman", "/the-great-escape-2-aida-oman",
    "/aida-coastal-investment-villas", "/Sunrise-Haven-Luxury-Villas",
    "/marriott-residences-aida-oman", "/amour-sans-detour-2", "/play-project/nickelodeon",
    "/the-mulliner", "/one-of-one/THE-ALBERT-HALL-MANSION", "/trump-plaza/park-residences",
    "/rayana/trump-mansions", "/trump-plaza/executive-residences", "/amaya/amaya-plots",
    "/amaya/padel-living", "/neptune", "/trump-tower-jeddah",
    "/rayana/darglobal-mansions-fendi-casa",
]
NON_PROJECT_PREFIXES = ("/press", "/blog", "/partners", "/privacy", "/terms", "/cookie",
                        "/about", "/investor", "/why-invest", "/careers", "/contact",
                        "/projects", "/ar", "/fr", "/ru", "/zh", "/de", "/es", "/it",
                        "/faq", "/get-in-touch", "/insights", "/pay-online", "/careers",
                        "/media", "/news", "/events", "/sitemap", "/search")
NON_PROJECT_PATHS = {"/one-of-one", "/trump-plaza", "/rayana", "/amaya", "/play-project",
                     "/lp", "/landing-page", "/landing-page/aida"}


def is_project_path(path: str) -> bool:
    """Could this path be a project page? Used for discovery and to clean merged data."""
    p = "/" + path.strip("/")
    low = p.lower()
    if re.match(r"^/(terms|privacy|cookie|legal|disclaimer)", low):  # /terms-conditions, /privacy-policy...
        return False
    return (p != "/" and low not in NON_PROJECT_PATHS
            and not any(low == x or low.startswith(x + "/") for x in NON_PROJECT_PREFIXES))

SPEC_LABELS = [
    "Property Type", "Unit Types", "Unit Type", "Location", "Completion Date", "Completion",
    "Handover", "Delivery", "Number of Floors", "Floors", "Area (sqm)", "Area (SQM)",
    "Area", "Plot Size", "Status", "Starting Price", "Starting From", "Prices From",
    "Price", "Payment Plan", "Bedrooms", "Number of Units", "Units", "Interiors", "Developer",
]
_LABELS_SORTED = sorted(SPEC_LABELS, key=len, reverse=True)

COUNTRY_HINTS = [
    ("Saudi Arabia", r"\b(ksa|saudi|riyadh|jeddah|wadi safar|rayana|amaya|makkah|madinah)\b"),
    ("United Arab Emirates", r"\b(uae|dubai|ras al khaimah|rak|marjan|abu dhabi|jumeirah)\b"),
    ("Oman", r"\b(oman|muscat|aida)\b"),
    ("Qatar", r"\b(qatar|doha|simaisma)\b"),
    ("Spain", r"\b(spain|marbella|benahav[ií]s|cortesin|estepona)\b"),
    ("United Kingdom", r"\b(london|uk|england|knightsbridge|hyde park|park lane)\b"),
    ("Greece", r"\b(greece|athens)\b"),
    ("United States", r"\b(usa|new york|miami)\b"),
]


def page_type(url: str, project_paths: set[str]) -> str:
    path = urlparse(url).path or "/"
    if path.rstrip("/").lower() in {p.lower() for p in project_paths}:
        return "project"
    for prefix, kind in (("/press", "news"), ("/blog", "blog"), ("/partners", "partner")):
        if path.lower().startswith(prefix + "/"):
            return kind
    return "page"


def infer_country(*texts: str) -> str:
    blob = " ".join(t for t in texts if t).lower()
    for country, pat in COUNTRY_HINTS:
        if re.search(pat, blob):
            return country
    return ""


def _strip_md(line: str) -> str:
    return re.sub(r"^[#>\-*\s|]+", "", line).strip().strip("|").strip()


def parse_specs(text: str) -> dict[str, str]:
    lines = [_strip_md(l) for l in text.splitlines()]
    lines = [l for l in lines if l]
    specs: dict[str, str] = {}
    for i, line in enumerate(lines):
        low = line.lower()
        for label in _LABELS_SORTED:
            if not low.startswith(label.lower()):
                continue
            rest = line[len(label):]
            # "Area" must not swallow e.g. "Areas of interest"
            if rest and rest[0].isalpha():
                continue
            value = rest.lstrip(" :–-|").strip()
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            nxt_is_label = any(nxt.lower().startswith(l.lower()) for l in SPEC_LABELS)
            if not value and nxt and not nxt_is_label:
                value = nxt
            elif value and re.search(r"(&|,|\band)$", value) and nxt and not nxt_is_label:
                value = f"{value} {nxt}"  # value wrapped onto the next line
            key = label.lower().replace(" (sqm)", "").replace(" ", "_")
            if re.search(r"frequently asked|\bfaqs?\b", value, re.I):
                value = ""  # a "Location" heading followed by the FAQ heading isn't a value
            if value and len(value) <= 140 and key not in specs:
                specs[key] = value
            break
    return specs


def is_valid_project(rec: dict) -> bool:
    """Needs a name and at least one real field."""
    content_fields = ("location", "property_type", "unit_types", "completion", "area_sqm",
                      "description", "starting_price", "floors", "faqs")
    return bool(rec.get("name")) and any(rec.get(f) for f in content_fields)


def parse_project(page: dict) -> dict:
    """Build a structured project record from a parsed project page."""
    text = page.get("text", "")
    name = (page.get("h1") or page.get("title") or "").strip()
    name = re.split(r"\s+[|–-]\s+(DarGlobal|Dar Global)", name, flags=re.I)[0].strip()
    # H1 is sometimes a tagline ("The place to be"); prefer the <title> then
    title = re.split(r"\s+[|–]\s+", page.get("title") or "")[0].strip()
    slug_words = set(re.findall(r"[a-z0-9]+", urlparse(page["url"]).path.lower())) - {"the", "by", "at", "of"}
    if title and title != name and not slug_words & set(re.findall(r"[a-z0-9]+", name.lower())):
        name = title
    specs = parse_specs(text)

    # subtitle right after the H1 is usually the location ("Riyadh | Saudi Arabia")
    subtitle = ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, l in enumerate(lines):
        if l.startswith("# ") and i + 1 < len(lines):
            cand = _strip_md(lines[i + 1])
            if (2 < len(cand) < 60 and not cand.lower().startswith(tuple(s.lower() for s in SPEC_LABELS))
                    and not re.search(r"frequently asked|\bfaqs?\b", cand, re.I)):
                subtitle = cand
            break
    location = specs.get("location") or subtitle
    faqs = re.findall(r"(?m)^#*\s*((?:What|Where|When|How|Who|Is|Are|Can|Does|Why)[^\n?]{5,160}\?)\s*\n+([^\n#]{20,800})", text)
    return {
        "id": urlparse(page["url"]).path.strip("/") or "home",
        "name": name,
        "url": page["url"],
        "location": location,
        "country": (infer_country(location) or infer_country(name)
                    or infer_country(page.get("description", ""), text[:600])),
        "property_type": specs.get("property_type", ""),
        "unit_types": specs.get("unit_types") or specs.get("unit_type", ""),
        "bedrooms": specs.get("bedrooms", ""),
        "area_sqm": specs.get("area", "") or specs.get("plot_size", ""),
        "completion": specs.get("completion_date") or specs.get("completion") or specs.get("handover") or specs.get("delivery", ""),
        "status": specs.get("status", ""),
        "floors": specs.get("number_of_floors") or specs.get("floors", ""),
        "units": specs.get("number_of_units") or specs.get("units", ""),
        "starting_price": specs.get("starting_price") or specs.get("starting_from") or specs.get("prices_from") or specs.get("price", ""),
        "payment_plan": specs.get("payment_plan", ""),
        "description": page.get("description", ""),
        "faqs": [{"q": q.strip(), "a": a.strip()} for q, a in faqs[:12]],
        "image": page.get("image", ""),
    }
