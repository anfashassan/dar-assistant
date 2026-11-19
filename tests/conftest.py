import os

# Unit tests run lexical-only by default: fast, deterministic, no model download.
# Dense retrieval is tested with a fake embedder (tests/test_embeddings.py) and measured
# on real data with `python scripts/eval.py`.
os.environ.setdefault("EMBEDDINGS", "0")
