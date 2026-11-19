"""Multilingual sentence embeddings (fastembed / ONNX, CPU only).

Vectors are cached by chunk-text hash in `embeddings.npz`, so startup and refreshes only
embed new chunks. If the model is unavailable, `get_embedder()` returns None and
retrieval runs lexical-only.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("app.embeddings")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_FILE = "embeddings.npz"


def text_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


class Embedder:
    """Thin wrapper so tests can swap in a fake with the same two methods."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: optional dependency
        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.stack(list(self._model.embed(texts, batch_size=batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_EMBEDDER: Embedder | None = None
_TRIED = False


def get_embedder() -> Embedder | None:
    """Process-wide embedder, or None when disabled / not installed / model missing."""
    global _EMBEDDER, _TRIED
    if _TRIED:
        return _EMBEDDER
    _TRIED = True
    if os.getenv("EMBEDDINGS", "1").lower() in ("0", "false", "off", "no"):
        log.info("embeddings disabled (EMBEDDINGS=0): lexical retrieval only")
        return None
    try:
        _EMBEDDER = Embedder(os.getenv("EMBED_MODEL", DEFAULT_MODEL), os.getenv("FASTEMBED_CACHE_PATH"))
        log.info("embeddings: %s", _EMBEDDER.model_name)
    except Exception as e:  # missing package, no network for the first download, ...
        log.warning("embeddings unavailable (%s): lexical retrieval only", str(e).splitlines()[0][:200])
        _EMBEDDER = None
    return _EMBEDDER


class VectorCache:
    """text-hash -> vector, persisted as a compressed .npz."""

    def __init__(self, model_name: str, vectors: dict[str, np.ndarray] | None = None):
        self.model_name = model_name
        self.vectors = vectors or {}

    @classmethod
    def load(cls, path: Path, model_name: str) -> "VectorCache":
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["model"]) != model_name:
                    log.info("vector cache %s is for another model, ignoring", path)
                    return cls(model_name)
                vecs = z["vectors"].astype(np.float32)
                if "scales" in z:  # int8 storage: undo the per-row scaling
                    vecs = vecs * (z["scales"].astype(np.float32)[:, None] / 127.0)
                return cls(model_name, dict(zip(z["keys"].tolist(), _normalize(vecs))))
        except FileNotFoundError:
            return cls(model_name)
        except Exception as e:
            log.warning("could not read vector cache %s: %s", path, e)
            return cls(model_name)

    def save(self, path: Path, keep: set[str] | None = None) -> None:
        keys = [k for k in self.vectors if keep is None or k in keep]
        if not keys:
            return
        mat = np.stack([self.vectors[k] for k in keys]).astype(np.float32)
        # int8 + per-row scale: 4x smaller, ranking unchanged
        scales = np.maximum(np.abs(mat).max(axis=1), 1e-12)
        q = np.round(mat / scales[:, None] * 127.0).astype(np.int8)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, model=np.array(self.model_name), keys=np.array(keys),
                            vectors=q, scales=scales.astype(np.float32))
        os.replace(tmp, path)

    def matrix_for(self, texts: list[str], embedder: Embedder) -> tuple[np.ndarray, int]:
        """Vectors for `texts` (row-aligned), embedding only the ones not cached yet."""
        keys = [text_key(t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self.vectors))
        if missing:
            by_key = {k: t for k, t in zip(keys, texts)}
            new = embedder.embed([by_key[k] for k in missing])
            self.vectors.update(zip(missing, new))
        return _normalize(np.stack([self.vectors[k] for k in keys])), len(missing)


def chunk_text_for_embedding(title: str, text: str) -> str:
    # the model only reads ~128 tokens, so the title goes first
    return f"{title}\n{text}"[:1200]
