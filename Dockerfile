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

# This image is CPU-only. torch's default PyPI wheel bundles the full CUDA/cuDNN
# runtime (~2 GB of nvidia-* + triton wheels) that a CPU host never uses, so install
# the CPU-only torch/torchvision build FIRST from PyTorch's CPU index. mineru /
# raganything only need torch for document layout + OCR inference, which runs fine on
# CPU. Pre-installing it satisfies the transitive constraint so the resolve below reuses
# it instead of pulling the CUDA variant. Keep these pins in sync with the versions
# raganything[all] resolves to (bump together when upgrading the stack).
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

WORKDIR /app
COPY server ./server

EXPOSE 9622
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:9622/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9622"]
