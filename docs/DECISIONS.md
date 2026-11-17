# Decisions

Deviations from `support-query-transformer-workflow.md` on non-binding points. One line each: what, why.

| # | What | Why |
|---|---|---|
| 1 | `/health` returns `ApiResponse[HealthResult]` instead of the bare JSON in doc 7.4 | Binding rule: every endpoint uses the shared envelope |
| 2 | The single exception handler is also registered for Starlette `HTTPException` | Unknown routes / wrong methods (404/405) must use the envelope too |
| 3 | `ChunkHit` and `StoredChunk` carry a `heading` field and live in `core/schemas/chunk.py` | The reranker and BM25 need `title — heading: chunk`; infrastructure must not import domain modules for its own data shape |
| 4 | Added `core/logging.py`, `core/health.py`, `core/schemas/health.py`, `utils/env.py`, `llm/call_budget.py` | Structured JSON logging with request ID, health endpoint, typed env parsing (functions, not a settings class), and the 50-call budget guard need a home |
| 5 | torch comes from the PyTorch CPU index via `[tool.uv.sources]` | CPU-only wheels on every platform; keeps the Docker image free of CUDA libraries |
| 6 | Request-ID middleware is pure ASGI, not `BaseHTTPMiddleware` | Current Starlette guidance; no response-streaming or contextvar pitfalls |
| 7 | Services are `async` and push Protocol calls through `asyncio.to_thread` | Protocols stay synchronous (as in doc 5.2) while CPU-bound work never blocks the event loop |
