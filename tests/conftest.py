import os
from pathlib import Path

import pytest

# Unit tests run lexical-only: fast, deterministic, no model download. Dense retrieval is
# tested with a fake embedder (test_embeddings.py) and measured with scripts/eval.py.
os.environ.setdefault("EMBEDDINGS", "0")

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def kb():
    from app.knowledge import KnowledgeBase
    # the seed snapshot keeps results stable; scripts/eval.py measures the scraped data
    return KnowledgeBase.load(DATA, raw_dir=DATA / "seed")
