"""Hybrid retrieval: BM25 (word level) + TF-IDF over character n-grams + dense
multilingual embeddings, fused with Reciprocal Rank Fusion.

Why this design
- BM25 is strong on exact terms (project names, districts, brand partners).
- Character n-grams are robust to typos, plurals, transliteration variants
  ("Jeddah"/"Jedda", "Al Yasmeen"/"Al Yasmin") and to Arabic morphology.
- Dense embeddings (app/embeddings.py) match on meaning: "where is it located?" finds
  "headquartered in London", and Arabic questions reach English text.
- RRF combines the lists without score calibration. The dense list is optional: without
  it (no model available) retrieval runs lexical-only, exactly as before.
"""
from __future__ import annotations

import logging
import math
import os
from difflib import get_close_matches
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger("app.retrieval")

AR_DIACRITICS =re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
TOKEN_RE = re.compile(r"[a-z0-9؀-ۿ]+")
EN_STOP = set("""a an the and or of to in on for with by at from is are was were be been it its this that
these those as about into than then there their they them what which who whom whose when where why how
do does did can could should would will shall may might me my we our you your i he she his her not no yes
any all some more most other such only own same so too very just also please tell show give find list
much many get got have has had""".split())
AR_STOP = set("""في من على إلى الى عن مع هذا هذه ذلك التي الذي ما ماذا هل كم كيف أين متى او أو و ثم لا نعم
كل بعض هو هي هم انا أنا نحن انت أنت لي لك عند""".split())


# Arabic -> English glossary so Arabic questions retrieve English source text.
# (Deterministic and free; avoids spending an LLM call on query translation.)
AR_EN = {
    "دار جلوبال": "DarGlobal Dar Global", "دار غلوبال": "DarGlobal", "دارجلوبال": "DarGlobal",
    "وصلت": "Wasalt", "دار الاركان": "Dar Al Arkan", "ترامب": "Trump", "برج": "tower",
    "مشاريع": "projects", "مشروع": "project", "فيلا": "villa", "فلل": "villas", "شقة": "apartment",
    "شقق": "apartments", "ارض": "land plot", "عقار": "property", "عقارات": "properties",
    "للبيع": "for sale", "بيع": "sale", "للايجار": "for rent", "ايجار": "rent", "سعر": "price",
    "اسعار": "prices", "متوسط": "average median", "الرياض": "Riyadh", "جدة": "Jeddah", "جده": "Jeddah",
    "الدمام": "Dammam", "الخبر": "Khobar", "مكة": "Makkah", "المدينة": "Madinah", "عمان": "Oman Muscat",
    "مسقط": "Muscat", "دبي": "Dubai", "الامارات": "UAE Dubai", "قطر": "Qatar Doha", "الدوحة": "Doha",
    "اسبانيا": "Spain", "لندن": "London", "السعودية": "Saudi Arabia KSA", "المملكة": "Saudi Arabia KSA",
    "كورنيش": "Corniche", "تسليم": "completion handover", "اكتمال": "completion", "موعد": "date",
    "غرف": "bedrooms", "غرفة": "bedroom", "حمام": "bathroom", "مساحة": "area sqm", "متر": "sqm",
    "شركاء": "partners", "شريك": "partner", "علامات": "brands", "فندق": "hotel", "جولف": "golf",
    "مزاد": "auction", "مزادات": "auctions", "ترخيص": "license", "مستثمر": "investor", "استثمار": "investment",
    "اخبار": "news", "اطلاق": "launch", "مفروش": "furnished", "مسبح": "pool", "حي": "district",
    "ياسمين": "Yasmeen", "الياسمين": "Al Yasmeen", "طويق": "Tuwaiq", "النرجس": "Al Narjis", "الملقا": "Al Malqa",
    "نبتون": "Neptune", "معوض": "Mouawad", "فندي": "Fendi", "ايدا": "AIDA", "امايا": "Amaya", "ريانة": "Rayana",
}


# Proper nouns rewritten to the spelling the source pages use, for the dense model only.
# Measured on live data: "where is DarGlobal located" ranks the FAQ #1, "where is dar
# global located" #22, and the Arabic "اين يقع مقر دار جلوبال" #346 -> #1 once the brand is
# written as on the site. (The model knows the words, not how this brand is spelled.)
BRANDS = [
    (r"dar\s*-?\s*global|دار\s*[جغ]لوبال", "DarGlobal"), (r"وصلت", "Wasalt"), (r"ترامب", "Trump"),
    (r"نبتون", "Neptune"), (r"معوض", "Mouawad"), (r"فندي", "Fendi"), (r"ايدا|أيدا", "AIDA"),
    (r"امايا|أمايا", "Amaya"), (r"ريانة|ريانه", "Rayana"), (r"باغاني|باجاني", "Pagani"),
    (r"لامبورغيني|لامبورجيني", "Lamborghini"), (r"استون مارتن|أستون مارتن", "Aston Martin"),
]
_BRAND_RES = [(re.compile(p, re.I), canon) for p, canon in BRANDS]


def dense_query(q: str) -> str:
    for pat, canon in _BRAND_RES:
        q = pat.sub(canon, q)
    return q


# Topic keywords: words the site uses for a few common intents. Needed where a page has
# too little text for either lexical or semantic matching - DarGlobal's /careers page is
# just "Find your next great opportunity ... Showing 0 results", so "jobs" can't reach it.
# These only ADD search terms; nothing is removed or re-ranked.
INTENTS = [
    (r"\b(jobs?|vacanc(y|ies)|hiring|recruit\w*|careers?|internships?|employment|openings?|"
     r"work (at|for|with)|apply for)\b|وظائف|وظيفه|وظيفة|توظيف|تدريب|فرص عمل",
     "careers internships vacancies opportunity"),
    (r"\bwhere\b.*\b(located|based|situated)\b|\b(headquarters?|hq|head office|main office|offices?)\b|"
     r"\blocation of\b|اين يقع|أين يقع|مقر|مكتب|مكاتب",
     "headquartered head office offices"),
    (r"\b(contact|phone|email|e-mail|call|reach|whatsapp)\b|تواصل|اتصال|هاتف|ايميل|بريد",
     "contact get in touch"),
]
_INTENT_RES = [(re.compile(p, re.I), words) for p, words in INTENTS]


def expand_query(q: str) -> str:
    nq = normalize(q)
    extra = [en for ar, en in AR_EN.items() if normalize(ar) in nq]
    extra += [words for pat, words in _INTENT_RES if pat.search(q) or pat.search(nq)]
    return q + (" " + " ".join(extra) if extra else "")


def normalize(text: str) -> str:
    t = text.lower()
    t = AR_DIACRITICS.sub("", t)
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("&", " and ")
    return t


def _stem(tok: str) -> str:
    if tok.isdigit() or len(tok) <= 3:
        return tok
    if re.match(r"[؀-ۿ]", tok):
        tok = re.sub(r"^(وال|بال|كال|فال|ال)", "", tok) if len(tok) > 4 else tok
        return tok
    for suf in ("ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(normalize(text))
    return [_stem(t) for t in toks if t not in EN_STOP and t not in AR_STOP]


@dataclass
class Chunk:
    id: str
    text: str
    title: str
    url: str
    site: str           # darglobal | wasalt
    kind: str           # project | news | blog | page | listing | summary ...
    meta: dict = field(default_factory=dict)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.lens = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.lens.mean()) if self.N else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


class HybridRetriever:
    # Source-authority prior: the developer's own project pages are the primary source for
    # project facts (completion, specs), so they win near-ties against third-party blog
    # posts or partner pages about the same project. +5% only reorders close RRF scores;
    # it can't pull in a project chunk that doesn't match the query.
    AUTHORITY = {"project": 1.05, "project_card": 1.05}
    # result slots (of k) reserved for the best semantic matches; see search()
    DENSE_SLOTS = int(os.getenv("DENSE_SLOTS", "2"))

    def __init__(self, chunks: list[Chunk], *, dense: np.ndarray | None = None, embedder=None):
        self.chunks = chunks
        # dense: (n_chunks, dim) L2-normalised vectors row-aligned with `chunks`;
        # embedder: anything with embed_query(str) -> unit vector
        self.dense = dense if dense is not None and len(dense) == len(chunks) else None
        self.embedder = embedder if self.dense is not None else None
        corpus = [f"{c.title}\n{c.text}" for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in corpus])
        self.vocab = [t for t in self.bm25.postings if len(t) >= 4 and t.isalpha()]
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                    sublinear_tf=True, preprocessor=normalize, max_features=250_000)
        self.char_matrix = self.char.fit_transform(corpus) if corpus else None

    def correct(self, toks: list[str]) -> list[str]:
        """Typo tolerance: map unknown Latin tokens to the closest corpus term."""
        out = []
        for t in toks:
            out.append(t)
            if len(t) >= 4 and t.isascii() and t.isalpha() and t not in self.bm25.postings:
                out += get_close_matches(t, self.vocab, n=1, cutoff=0.8)
        return out

    def _ranks(self, scores: np.ndarray, limit: int) -> list[int]:
        idx = np.argsort(-scores)[:limit]
        return [int(i) for i in idx if scores[i] > 0]

    def search(self, query: str, k: int = 6, *, site: str | None = None, kinds: set[str] | None = None,
               per_url: int = 2, rrf_k: int = 60) -> list[tuple[Chunk, float]]:
        if not self.chunks or not query.strip():
            return []
        raw_query = query
        query = expand_query(query)
        pool = max(50, k * 8)
        lists = [self._ranks(self.bm25.scores(self.correct(tokenize(query))), pool)]
        if self.char_matrix is not None:
            q = self.char.transform([query])
            lists.append(self._ranks((self.char_matrix @ q.T).toarray().ravel(), pool))
        dense_ranked: list[int] = []
        if self.embedder is not None:
            try:  # the raw question (the model is multilingual) with brand spellings fixed
                qv = self.embedder.embed_query(dense_query(raw_query))
                dense_ranked = self._ranks(self.dense @ qv, pool)
            except Exception as e:  # never let the optional dense path break search
                log.warning("dense retrieval failed: %s", e)

        fused: dict[int, float] = defaultdict(float)
        for ranked in lists:
            for r, i in enumerate(ranked):
                fused[i] += 1.0 / (rrf_k + r + 1)
        for i in fused:
            fused[i] *= self.AUTHORITY.get(self.chunks[i].kind, 1.0)
        lexical = [(i, s) for i, s in sorted(fused.items(), key=lambda x: -x[1])]
        dense = [(i, 1.0 / (rrf_k + r + 1)) for r, i in enumerate(dense_ranked)]

        # Reserved slots instead of one fused ranking: lexical keeps the top positions
        # (precise on names, prices, typos) and the best semantic matches fill the last
        # `reserve` slots. Measured: fusing dense into RRF at any weight cost hit@1
        # (0.89 -> 0.72) on name questions; reserved slots keep that and still put
        # paraphrase answers ("located" -> "headquartered") in front of the LLM.
        reserve = min(self.DENSE_SLOTS, k // 3) if dense else 0
        out, per, taken = [], Counter(), set()

        def take(cands, limit):
            for i, score in cands:
                if len(out) >= limit:
                    return
                c = self.chunks[i]
                if i in taken or (site and c.site != site) or (kinds and c.kind not in kinds):
                    continue
                if per[c.url] >= per_url:  # diversity: avoid 6 chunks of one page
                    continue
                per[c.url] += 1
                taken.add(i)
                out.append((c, round(score, 5)))

        take(lexical, k - reserve)
        take(dense, k)
        take(lexical, k)   # if dense had too few eligible hits, lexical fills the rest
        return out
