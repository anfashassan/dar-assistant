"""Structured property search over scraped Wasalt listings.

Natural-language requests ("3 bedroom villa in Riyadh under 2 million",
"شقة للإيجار في جدة") are parsed into filters with deterministic rules, applied
to the listings table, and progressively relaxed when nothing matches. Prices in
answers therefore come from data, not from the language model.
"""
from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher
from dataclasses import dataclass, field

from .retrieval import expand_query, normalize

TYPE_SYNONYMS = {
    "Villa": ["villa", "villas", "فيلا", "فلل", "فله"],
    "Apartment": ["apartment", "apartments", "flat", "flats", "شقة", "شقه", "شقق"],
    "Floor": ["floor", "دور", "ادوار"],
    "Land": ["land", "plot", "plots", "ارض", "اراضي"],
    "Duplex": ["duplex", "دوبلكس", "دبلكس"],
    "Townhouse": ["townhouse", "town house", "تاون هاوس"],
    "Rest House": ["rest house", "resthouse", "istiraha", "استراحة", "استراحه"],
    "Building": ["building", "عمارة", "عماره", "مبنى"],
    "Office": ["office", "offices", "مكتب", "مكاتب"],
    "Shop": ["shop", "shops", "retail", "محل", "محلات"],
    "Warehouse": ["warehouse", "مستودع"],
    "Farm": ["farm", "مزرعة", "مزرعه"],
    "Chalet": ["chalet", "شاليه"],
    "Penthouse": ["penthouse", "بنتهاوس"],
    "Studio": ["studio", "ستوديو"],
}
CITY_ALIASES = {
    "Riyadh": ["riyadh", "riyad", "الرياض"],
    "Jeddah": ["jeddah", "jedda", "jidda", "جدة", "جده"],
    "Dammam": ["dammam", "الدمام"],
    "Al Khobar": ["khobar", "al khobar", "alkhobar", "الخبر"],
    "Makkah": ["makkah", "mecca", "مكة", "مكه"],
    "Madinah": ["madinah", "medina", "المدينة", "المدينه"],
    "Taif": ["taif", "الطائف"],
    "Abha": ["abha", "ابها"],
    "Dhahran": ["dhahran", "الظهران"],
}
RENT_WORDS = r"\b(rent|rental|renting|lease|leasing|to let|monthly|yearly)\b|ايجار|للايجار|استئجار"
SALE_WORDS = r"\b(buy|buying|sale|purchase|own|for sale)\b|للبيع|شراء|بيع|تملك"
MAX_WORDS = r"(under|below|less than|max(?:imum)?|up to|upto|within|budget(?: of| is)?|cheaper than|no more than|<=?|اقل من|تحت|حد اقصى|بحدود|ميزانيه)"
MIN_WORDS = r"(over|above|more than|min(?:imum)?|at least|from|starting|>=?|اكثر من|فوق|على الاقل)"
AMENITIES = {
    "pool": ["pool", "swimming", "مسبح"], "garden": ["garden", "حديقة"], "elevator": ["elevator", "lift", "مصعد"],
    "maids room": ["maid", "خادمة"], "driver room": ["driver", "سائق"], "gym": ["gym", "نادي"],
    "parking": ["parking", "garage", "موقف"], "annex": ["annex", "ملحق"], "rooftop": ["rooftop", "roof", "سطح"],
    "furnished": ["furnished", "مفروش", "مفروشة"], "smart home": ["smart home", "منزل ذكي"],
    "central ac": ["central ac", "تكييف مركزي"], "basement": ["basement", "قبو"],
}
LISTING_INTENT = r"\b(listings?|propert(y|ies)|homes?|houses?|for sale|for rent|available|cheapest|wasalt|budget|sqm|bedrooms?|beds?)\b|عقار|عقارات|وصلت|للبيع|للايجار"


@dataclass
class ListingQuery:
    purpose: str | None = None
    property_type: str | None = None
    city: str | None = None
    district: str | None = None
    bedrooms_min: int | None = None
    bedrooms_exact: int | None = None
    price_min: float | None = None
    price_max: float | None = None
    area_min: float | None = None
    amenities: list[str] = field(default_factory=list)
    sort: str = "relevance"
    intent: bool = False

    def active(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, [], "relevance", False)}
        d.pop("intent", None)
        return d

    def merged_with(self, newer: "ListingQuery") -> "ListingQuery":
        """Follow-up questions inherit earlier filters unless they override them."""
        out = ListingQuery(**{k: v for k, v in self.__dict__.items()})
        for k, v in newer.__dict__.items():
            if v not in (None, [], "relevance", False):
                setattr(out, k, v)
        out.intent = self.intent or newer.intent
        return out


def _money(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").lower().strip()
    if unit in ("m", "mn", "mil", "million", "millions", "مليون", "ملايين"):
        v *= 1_000_000
    elif unit in ("k", "thousand", "الف", "ألف", "آلاف", "الاف"):
        v *= 1_000
    elif v < 100:  # "under 2" in a property context almost always means millions
        v *= 1_000_000
    return v


MONEY = r"(?:sar|sr|ريال)?\s*([\d][\d,]*(?:\.\d+)?)\s*(m|mn|mil|millions?|k|thousand|مليون|ملايين|الف|ألف|آلاف|الاف)?\b\s*(?:sar|sr|riyals?|ريال)?"


FUZZY_STOPWORDS = {"sale", "rent", "sell", "riyal", "riyals", "million", "millions", "price",
                   "villa", "villas", "house", "homes", "under", "below", "above", "show",
                   "search", "budget", "cheap", "apartment", "apartments", "bedrooms", "bedroom"}


def parse_query(text: str, known_districts: list[str] | None = None,
                known_cities: list[str] | None = None) -> ListingQuery:
    q = ListingQuery()
    t = normalize(expand_query(text))
    t = re.sub(r"(\d)\s*,\s*(\d{3})", r"\1,\2", t)

    if re.search(RENT_WORDS, t):
        q.purpose = "rent"
    elif re.search(SALE_WORDS, t):
        q.purpose = "sale"

    for canon, words in TYPE_SYNONYMS.items():
        if any(re.search(rf"(?<![\w؀-ۿ]){re.escape(normalize(w))}(?![\w؀-ۿ])", t) for w in words):
            q.property_type = canon
            break

    cities = dict(CITY_ALIASES)
    for c in known_cities or []:
        cities.setdefault(c, [c.lower()])
    for canon, words in cities.items():
        if any(re.search(rf"(?<![\w؀-ۿ]){re.escape(normalize(w))}(?![\w؀-ۿ])", t) for w in words):
            q.city = canon
            break

    best, words = "", re.findall(r"[a-z\u0600-\u06FF]+", t)
    for d in known_districts or []:
        nd = normalize(d)
        core = re.sub(r"^(al|al-|ال)\s*", "", nd).strip()
        if len(core) < 4:
            continue
        hit = re.search(rf"\b(al[\s-]?)?{re.escape(core)}\b", t)
        if not hit and len(core) >= 5:  # fuzzy: "yasmin" ~ "yasmeen", "malqa" ~ "malga"
            n = len(core.split())
            grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
            # short / common query words must not fuzzy-match a district: "sar" (the
            # currency) used to match "Al Samer" on every price question
            hit = any(g[:2] == core[:2] and len(g) >= 4 and abs(len(g) - len(core)) <= 2
                      and g not in FUZZY_STOPWORDS
                      and SequenceMatcher(None, core, g).ratio() >= 0.75 for g in grams)
        if hit and len(d) > len(best):
            best = d
    q.district = best or None

    m = re.search(r"(\d+)\s*\+?\s*(?:-\s*)?(?:bed(?:room)?s?|br|bhk|غرف|غرفه|غرفة)", t)
    if m:
        n = int(m.group(1))
        if re.search(rf"(at least|minimum|min|more than|or more|\+|على الاقل)\s*{n}|{n}\s*(\+|or more)", t):
            q.bedrooms_min = n
        else:
            q.bedrooms_exact = n
    elif re.search(r"\bstudio\b|ستوديو", t):
        q.bedrooms_exact = 1

    m = re.search(rf"between\s+{MONEY}\s+(?:and|to|-)\s+{MONEY}|بين\s+{MONEY}\s+و\s*{MONEY}", t)
    if m:
        g = [x for x in m.groups()]
        a = _money(g[0] or g[4], g[1] or g[5]) if (g[0] or g[4]) else None
        b = _money(g[2] or g[6], g[3] or g[7]) if (g[2] or g[6]) else None
        if a and b:
            q.price_min, q.price_max = min(a, b), max(a, b)
    else:
        m = re.search(rf"{MAX_WORDS}\s*{MONEY}", t)
        if m and not re.match(r"\s*\d+\s*(bed|br|sqm|m2|متر)", t[m.end(1):m.end(1) + 12]):
            q.price_max = _money(m.group(2), m.group(3))
        m = re.search(rf"{MIN_WORDS}\s*{MONEY}(?!\s*(?:sqm|m2|square|متر|bed))", t)
        if m and re.search(r"(m|million|k|thousand|مليون|الف|sar|ريال|\d{5,})", m.group(0)):
            q.price_min = _money(m.group(2), m.group(3))
        m = re.search(rf"(?:around|about|approx(?:imately)?|حوالي|تقريبا)\s*{MONEY}", t)
        if m and not (q.price_max or q.price_min):
            v = _money(m.group(1), m.group(2))
            q.price_min, q.price_max = v * 0.8, v * 1.2

    m = re.search(r"(?:at least|over|above|min(?:imum)?|more than|اكثر من)?\s*(\d{2,5})\s*(?:sqm|sq m|m2|square met(?:er|re)s?|متر)", t)
    if m:
        q.area_min = float(m.group(1)) * (1.0 if re.search(r"at least|over|above|min|more than|اكثر", m.group(0)) else 0.85)

    for canon, words in AMENITIES.items():
        if any(normalize(w) in t for w in words):
            q.amenities.append(canon)

    if re.search(r"cheapest|lowest price|most affordable|ارخص|الارخص", t):
        q.sort = "price_asc"
    elif re.search(r"most expensive|luxur|highest price|اغلى", t):
        q.sort = "price_desc"
    elif re.search(r"largest|biggest|most space|اكبر", t):
        q.sort = "area_desc"
    elif re.search(r"newest|new build|latest|احدث", t):
        q.sort = "year_desc"

    q.intent = bool(re.search(LISTING_INTENT, t)) or any(
        getattr(q, k) is not None for k in ("property_type", "bedrooms_exact", "bedrooms_min", "price_max", "price_min"))
    return q


def _norm(s) -> str:
    return normalize(str(s or ""))


def _match(l: dict, q: ListingQuery, *, skip: set[str]) -> bool:
    if q.purpose and "purpose" not in skip and l.get("purpose") and l["purpose"] != q.purpose:
        return False
    if q.property_type and "property_type" not in skip and _norm(q.property_type) not in _norm(l.get("property_type")) \
            and _norm(q.property_type) not in _norm(l.get("title")):
        return False
    if q.city and "city" not in skip and _norm(q.city).replace("al ", "") not in _norm(l.get("city")):
        return False
    if q.district and "district" not in skip:
        core = re.sub(r"^(al|al-)\s*", "", _norm(q.district))
        if core not in _norm(l.get("district")) and core not in _norm(l.get("title")):
            return False
    beds = l.get("bedrooms")
    if q.bedrooms_exact is not None and "bedrooms" not in skip and beds != q.bedrooms_exact:
        return False
    if q.bedrooms_min is not None and "bedrooms" not in skip and (beds is None or beds < q.bedrooms_min):
        return False
    price = l.get("price_sar")
    slack = 1.15 if "price" in skip else 1.0
    if q.price_max is not None and (price is None or price > q.price_max * slack):
        return False
    if q.price_min is not None and (price is None or price < q.price_min / slack):
        return False
    area = l.get("built_area_sqm") or l.get("land_area_sqm")
    if q.area_min and "area" not in skip and (area is None or area < q.area_min):
        return False
    if q.amenities and "amenities" not in skip:
        blob = _norm(" ".join(l.get("amenities", []) + l.get("fixtures", []) + [l.get("furnishing", ""), l.get("title", "")]))
        for a in q.amenities:
            if a == "furnished":
                if "furnished" not in blob or "un-furnished" in blob or "unfurnished" in blob:
                    return False
            elif not any(_norm(w) in blob for w in AMENITIES[a]):
                return False
    return True


@dataclass
class ListingResult:
    query: ListingQuery
    matches: list[dict]
    total: int
    relaxed: list[str]
    stats: dict


RELAX_ORDER = [["amenities"], ["district"], ["bedrooms"], ["price"], ["area"], ["property_type"]]


def search_listings(listings: list[dict], q: ListingQuery, limit: int = 6) -> ListingResult:
    skip: set[str] = set()
    relaxed: list[str] = []
    found = [l for l in listings if _match(l, q, skip=skip)]
    for step in RELAX_ORDER:
        if found:
            break
        skip.update(step)
        relaxed += step
        found = [l for l in listings if _match(l, q, skip=skip)]

    key = {
        "price_asc": lambda l: l.get("price_sar") or 9e18,
        "price_desc": lambda l: -(l.get("price_sar") or 0),
        "area_desc": lambda l: -((l.get("built_area_sqm") or l.get("land_area_sqm")) or 0),
        "year_desc": lambda l: -(l.get("building_year") or 0),
    }.get(q.sort)
    if key:
        found.sort(key=key)
    elif q.price_max:  # best use of budget first
        found.sort(key=lambda l: -(l.get("price_sar") or 0))
    else:
        found.sort(key=lambda l: (l.get("price_sar") is None, l.get("price_sar") or 0))

    prices = [l["price_sar"] for l in found if l.get("price_sar")]
    stats = {}
    if prices:
        stats = {"count": len(found), "median_price": statistics.median(prices),
                 "min_price": min(prices), "max_price": max(prices)}
    return ListingResult(q, found[:limit], len(found), relaxed, stats)
