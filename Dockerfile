# ---- DarGlobal x Wasalt Property Assistant ----------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    FASTEMBED_CACHE_PATH=/app/models

# Hugging Face Spaces runs containers as uid 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the embedding model (~220 MB) into the image: no download at startup, and the
# Space still works if the model hub is unreachable
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/models')" \
 && chown -R app:app /app/models

# app code + the scraped data snapshot (data/raw) and seed fallback (data/seed)
COPY --chown=app:app app ./app
COPY --chown=app:app scraper ./scraper
COPY --chown=app:app data ./data

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','7860'), timeout=4)" || exit 1

# single worker on purpose: the index and refresh scheduler live in-process
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
