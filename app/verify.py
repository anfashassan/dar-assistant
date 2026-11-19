"""Check that money amounts and years in an answer appear in the sources it was given."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
_UNITS = {"k": 1e3, "thousand": 1e3, "ألف": 1e3, "الف": 1e3, "آلاف": 1e3,
          "m": 1e6, "mn": 1e6, "million": 1e6, "مليون": 1e6, "ملايين": 1e6,
          "b": 1e9, "bn": 1e9, "billion": 1e9, "مليار": 1e9}
_NUMBER = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
    r"(k|mn|m|bn|b|thousand|million|billion|ألف|الف|آلاف|مليون|ملايين|مليار)?(?![a-z])", re.I)
_CURRENCY = re.compile(r"sar|\bsr\b|riyal|ريال|ر\.س|usd|aed|gbp|\$", re.I)
# "above 5 million", "up to SAR 2M": thresholds the model uses to describe a range, not facts
_THRESHOLD = re.compile(r"(above|over|under|below|less than|more than|up to|at least|"
                        r"أعلى من|أقل من|اقل من|أكثر من|اكثر من|حتى|فوق|تحت)\s*(sar|riyals?|ريال)?\s*$", re.I)
_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20\d\d)(?!\d)")
_CITATION = re.compile(r"\[L?\d{1,2}\]")


@dataclass
class Verification:
    checked: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def as_event(self) -> dict:
        return {"type": "verify", "checked": len(self.checked), "unsupported": self.unsupported}


def _values(text: str, money_only: bool) -> list[tuple[float, str]]:
    out = []
    for m in _NUMBER.finditer(text):
        num, unit = m.group(1), (m.group(2) or "").lower()
        value = float(num.replace(",", "")) * _UNITS.get(unit, 1)
        if money_only:
            near = text[max(0, m.start() - 12): m.end() + 12]
            if value < 10_000 or not (unit or _CURRENCY.search(near)):
                continue
            if _THRESHOLD.search(text[max(0, m.start() - 20): m.start()]):
                continue
        out.append((value, m.group(0).strip()))
    return out


def verify_answer(answer: str, context: str, tolerance: float = 0.03) -> Verification:
    """Amounts must match a number in the context within `tolerance` (answers round, e.g.
    1,187,040 -> "1.19M"); years must appear in the context. Uncited answers (small talk,
    general knowledge) are not checked."""
    result = Verification()
    answer, context = answer.translate(_AR_DIGITS), context.translate(_AR_DIGITS)
    if not _CITATION.search(answer):
        return result
    known = [v for v, _ in _values(context, money_only=False)]
    for value, raw in _values(answer, money_only=True):
        result.checked.append(raw)
        if not any(abs(value - k) <= tolerance * max(value, k) for k in known):
            result.unsupported.append(raw)
    years = set(_YEAR.findall(context))
    for year in dict.fromkeys(_YEAR.findall(answer)):
        result.checked.append(year)
        if year not in years:
            result.unsupported.append(year)
    return result
