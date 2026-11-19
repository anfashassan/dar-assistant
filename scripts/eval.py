"""Quick quality check.

Offline (default): retrieval hit@1 / hit@3 / MRR on a small labelled question set,
including Arabic and misspelled questions.
    python scripts/eval.py

Live: send questions to a running deployment and print answers, models and latency.
    python scripts/eval.py --live https://<your-space>.hf.space
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL = [  # (question, URL substring of a relevant source; "a|b" = any of several correct sources)
    ("When will Trump Tower Jeddah be completed?", "trump-tower-jeddah"),
    ("How many floors does Trump Tower Jeddah have?", "trump-tower-jeddah"),
    ("Tell me about the Neptune villas by Mouawad", "neptune"),
    ("Mouawad partnership riyadh", "mouawad"),
    # both Tierra Viva and Painite Villas are Lamborghini-inspired villas in Spain; the
    # /insights/ Q&A page answers the question directly and names Tierra Viva
    ("Lamborghini inspired villas in Spain", "tierra-viva|painite|lamborghini-inspired-villas-spain"),
    ("Aston Martin tower in Ras Al Khaimah", "astera|aston-martin-interiors-apartments-ras-al-khaimah"),
    ("Missoni apartments in Dubai", "urban-oasis"),
    ("Elie Saab project in Doha", "les-vagues"),
    ("Trump golf villas Oman", "aida-trump-villas"),
    ("Pagani tower Dubai", "davinci"),
    ("What is Wasalt?", "wasalt.sa"),
    ("trump towr jedah", "trump-tower-jeddah"),
    # paraphrases: the answer uses different words than the question ("located" vs
    # "headquartered") - what the dense (semantic) list is for
    ("where is dar global located", "co.uk/faq|co.uk/investor|co.uk/about"),
    ("Which cities does DarGlobal have sales offices in?", "co.uk/faq|co.uk/investor|co.uk/about"),
    ("Is DarGlobal a publicly traded company?", "co.uk/faq|co.uk/investor|co.uk/about"),
    ("اين يقع مقر دار جلوبال؟", "co.uk/faq|co.uk/investor|co.uk/about"),
    # jobs: /careers is nearly empty ("Showing 0 results"), so only intent keywords reach it
    ("are there any job opportunities at DarGlobal?", "co.uk/careers|co.uk/internships"),
    ("tell me more about dar global and are there any job oppurtunities in it", "co.uk/careers|co.uk/internships"),
    ("هل توجد وظائف في دار جلوبال؟", "co.uk/careers|co.uk/internships"),
    ("متى يكتمل برج ترامب جدة؟", "trump-tower-jeddah"),
    ("فلل نبتون معوض في الرياض", "neptune"),
]
LIVE = [
    "Which DarGlobal projects are in Saudi Arabia?",
    "Tell me about Trump Tower Jeddah",
    "Show me villas for sale in Riyadh under 2 million SAR",
    "What is the median villa price in Riyadh on Wasalt?",
    "ما هي مشاريع دار جلوبال في عُمان؟",
    "What is the weather in London?",  # out-of-scope: should decline politely
]


def offline():
    from app.knowledge import KnowledgeBase
    kb = KnowledgeBase.load(ROOT / "data")
    print(f"dataset: {kb.stats()['data_source']}, {len(kb.chunks)} chunks\n")
    h1 = h3 = ctx = mrr = 0.0
    for q, frag in EVAL:
        hits = kb.retriever.search(q, k=6)   # 6 = TOP_K: exactly what the LLM gets to read
        ranks = [i for i, (c, _) in enumerate(hits, 1)
                 if any(f in c.url.lower() for f in frag.lower().split("|"))]
        r = ranks[0] if ranks else None
        h1 += r == 1
        h3 += bool(r and r <= 3)
        ctx += bool(r)
        mrr += 1 / r if r else 0
        print(f"{'OK ' if r else 'MISS'} rank={r or '-':<2} {q}")
    n = len(EVAL)
    print(f"\nhit@1={h1 / n:.2f}  hit@3={h3 / n:.2f}  ctx@6={ctx / n:.2f}  MRR={mrr / n:.2f}  (n={n})"
          f"   [ctx@6 = answer is in the context the LLM receives]")


def live(base: str):
    import requests
    for q in LIVE:
        t = time.time()
        r = requests.post(f"{base.rstrip('/')}/api/chat/sync", json={"message": q}, timeout=120)
        d = r.json()
        print(f"\n=== {q}\n[{d.get('model')} | {time.time() - t:.1f}s | {len(d.get('sources', []))} sources"
              f" | {(d.get('listings') or {}).get('total', 0)} listings]\n{d.get('answer', json.dumps(d))[:900]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", help="base URL of a running deployment")
    a = ap.parse_args()
    live(a.live) if a.live else offline()
