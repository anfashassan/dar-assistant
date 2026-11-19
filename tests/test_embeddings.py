"""Dense retrieval wiring, tested with a deterministic fake embedder (no model download)."""
import re
from pathlib import Path

import numpy as np

from app.embeddings import VectorCache
from app.knowledge import KnowledgeBase
from app.retrieval import Chunk, HybridRetriever

DATA = Path(__file__).resolve().parent.parent / "data"

# words with the same meaning share a dimension, like a real sentence-embedding model would
CONCEPTS = {"located": 0, "headquartered": 0, "based": 0, "hq": 0, "office": 0, "offices": 0, "sit": 0,
            "london": 1, "villa": 2, "villas": 2, "price": 3, "cost": 3}


class FakeEmbedder:
    model_name = "fake-concepts-v1"

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(16, dtype=np.float32)
        v[15] = 0.05  # tiny shared component so no vector is all-zero
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in CONCEPTS:  # meaning-bearing words only, like a semantic model
                v[CONCEPTS[w]] += 1.0
        return v / max(np.linalg.norm(v), 1e-9)

    def embed(self, texts):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def _chunks():
    # like the live data: many chunks mention DarGlobal, so the name alone can't pick the
    # answer; only the meaning ("located" ~ "headquartered") can
    filler = [Chunk(f"f{i}", f"DarGlobal news {i}: DarGlobal launches apartments with balcony", f"DarGlobal news {i}",
                    f"https://x/{i}", "wasalt", "listing") for i in range(30)]
    hq = Chunk("hq", "DarGlobal is headquartered in London and operates in 9 countries.",
               "Frequently asked questions", "https://darglobal.co.uk/faq", "darglobal", "page")
    return filler + [hq]


def test_dense_list_bridges_paraphrase():
    chunks = _chunks()
    # a paraphrase the intent keywords don't cover, so only the semantic list can bridge it
    question = "where does DarGlobal sit?"
    lexical = HybridRetriever(chunks).search(question, k=6)
    assert "hq" not in [c.id for c, _ in lexical]         # words alone never reach the answer

    emb = FakeEmbedder()
    dense = emb.embed([f"{c.title}\n{c.text}" for c in chunks])
    hybrid = HybridRetriever(chunks, dense=dense, embedder=emb).search(question, k=6)
    ids = [c.id for c, _ in hybrid]
    assert "hq" in ids[4:]                                 # arrives via a reserved semantic slot
    assert ids[:4] == [c.id for c, _ in lexical[:4]]       # lexical top positions untouched


def test_dense_failure_falls_back_to_lexical():
    class Broken(FakeEmbedder):
        def embed_query(self, text):
            raise RuntimeError("onnx exploded")
    chunks = _chunks()
    emb = Broken()
    r = HybridRetriever(chunks, dense=emb.embed([c.text for c in chunks]), embedder=emb)
    assert r.search("headquartered London", k=1)[0][0].id == "hq"   # still answers, lexically


def test_misaligned_vectors_are_ignored():
    chunks = _chunks()
    r = HybridRetriever(chunks, dense=np.zeros((3, 16), dtype=np.float32), embedder=FakeEmbedder())
    assert r.embedder is None and r.dense is None


def test_kb_embeds_once_then_reuses_cache(tmp_path):
    import shutil
    raw = tmp_path / "raw"
    shutil.copytree(DATA / "seed", raw)
    emb = FakeEmbedder()
    kb = KnowledgeBase.load(DATA, raw_dir=raw, embedder=emb)
    first = emb.calls
    assert first == len(kb.chunks) and (raw / "embeddings.npz").exists()
    assert kb.retriever.embedder is emb

    emb2 = FakeEmbedder()
    KnowledgeBase.load(DATA, raw_dir=raw, embedder=emb2)
    assert emb2.calls == 0                                   # everything came from the cache

    class OtherModel(FakeEmbedder):
        model_name = "another-model"
    emb3 = OtherModel()
    KnowledgeBase.load(DATA, raw_dir=raw, embedder=emb3)
    assert emb3.calls == first                               # stale cache for a different model ignored


def test_kb_without_embedder_is_lexical_only():
    kb = KnowledgeBase.load(DATA, raw_dir=DATA / "seed", embedder=None)
    assert kb.retriever.embedder is None and kb.retriever.search("Trump Tower Jeddah", k=1)


def test_vector_cache_roundtrip(tmp_path):
    c = VectorCache("m", {"a": np.ones(4, dtype=np.float32)})
    c.save(tmp_path / "v.npz")
    back = VectorCache.load(tmp_path / "v.npz", "m")
    assert list(back.vectors) == ["a"] and np.allclose(back.vectors["a"], 0.5)   # unit-normalised
    assert VectorCache.load(tmp_path / "missing.npz", "m").vectors == {}


def test_int8_storage_keeps_ranking(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    VectorCache("m", {str(i): v for i, v in enumerate(vecs)}).save(tmp_path / "v.npz")
    back = VectorCache.load(tmp_path / "v.npz", "m")
    restored = np.stack([back.vectors[str(i)] for i in range(200)])
    # random vectors have near-tied similarities below the top few (real embeddings are far
    # less tied), so require the top 3 to be identical and the top 10 to overlap almost fully
    for qi in (7, 42, 150):
        a = np.argsort(-(vecs @ vecs[qi]))[:10].tolist()
        b = np.argsort(-(restored @ vecs[qi]))[:10].tolist()
        assert a[:3] == b[:3] and len(set(a) & set(b)) >= 9
    assert np.abs(restored - vecs).max() < 0.02
