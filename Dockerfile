FROM python:3.11-slim

# libreoffice: converts Office docs (docx/pptx/xlsx) for extraction.
# libmagic1: content-type sniffing. curl: container healthcheck.
RUN apt-get update && apt-get install -y \
    libreoffice \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# --timeout/--retries: the raganything[all] wheels are large; bump pip's low
# default per-read timeout so a slow mirror read doesn't abort the whole install.
# lightrag-hku is pinned in requirements.txt to a release carrying the upstream
# AGE edge-property fix (HKUDS/LightRAG#3052) — no vendored patch needed.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

WORKDIR /app
COPY server.py .

EXPOSE 9622
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:9622/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9622"]
