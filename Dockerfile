# syntax=docker/dockerfile:1

# ---- builder: resolve dependencies from uv.lock into /app/.venv -----------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# torch comes from the PyTorch CPU index (see [tool.uv.sources]), so no CUDA libraries land here.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev

# ---- runtime: slim image, non-root user, no build tools --------------------------------------
FROM python:3.12-slim AS runtime

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts
COPY data/kb ./data/kb

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    APP_ENV=uat

# Writable locations for the vector index, the rewrite cache, and the model cache.
RUN mkdir -p /app/data/chroma /app/data/cache /home/app/.cache/huggingface \
    && chown -R app:app /app/data /home/app

USER app
VOLUME ["/app/data/chroma", "/home/app/.cache/huggingface"]
EXPOSE 8000

# Model loading can take a while on first start (downloads into the Hugging Face volume).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
