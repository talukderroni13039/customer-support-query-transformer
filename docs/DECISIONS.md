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
| 8 | Dependency is `instructor[anthropic]`, which pins `anthropic==0.93.0`, not the latest Anthropic SDK | It is the only SDK version Instructor 1.17 declares support for; a mismatch could not be debugged under the 50-call cap |
| 9 | BM25 uses a `BM25Okapi` subclass with the Lucene IDF `ln(1 + (N - n + 0.5) / (n + 0.5))` | Stock rank_bm25 IDF is zero or negative for terms in half or more of the chunks, so true matches score ≤ 0 on a small knowledge base |
| 10 | Tokenizer keeps hyphenated tokens whole only when they contain a digit (`tb-pro-22`); `water-resistant` is split | Codes stay exact, while ordinary hyphenated words still match their unhyphenated spelling |
| 11 | ChromaDB collection uses `configuration={"hnsw": {"space": "cosine"}}` and `embedding_function=None` | Current ChromaDB 1.x API; embeddings are always supplied, so the default ONNX embedding model is never downloaded |
| 12 | Chunks are stored as raw contiguous slices, heading line included | `"".join(chunks) == content`, so `GET /articles/{id}` rebuilds the article exactly; the heading line is dropped only when building the embedded / BM25 / reranker text |
| 13 | LLM calls are counted in an httpx request hook, and the SDK keeps one transport retry | Counts every real HTTP attempt (Instructor retries and SDK retries for 429/5xx alike); the first live run failed twice on transient provider errors with SDK retries off |
| 14 | In Instructor 1.17 `max_retries=N` means N retries after the first attempt | Differs from the older "total attempts" meaning; live tests with `max_retries=1` can make up to 2 calls each |
| 15 | With `rerank: true` the result contains only the reranked candidates (top `RERANK_CANDIDATES` chunks) | Rerank scores and fused scores are not comparable, so un-reranked chunks are not appended below reranked ones |
| 16 | `rewrite_source` has values `none`, `skipped`, `llm`, `cache`, `fallback`, `disabled`; a missing API key or `REWRITE_PROVIDER=none` yields `disabled` with `rewrite_fallback: true` | The doc only shows `llm`; clients need to tell a skipped rewrite from a failed or disabled one |
| 17 | `RewrittenQuery.from_cache` is a `SkipJsonSchema` field excluded from dumps | Lets the cache decorator report cache hits through the same Protocol without exposing the field to Claude |
| 18 | The rewrite cache stores hashed keys plus the rewritten query and the original call latency, never the question or history | Privacy (conversations are never stored) and honest benchmark latency on cached re-runs |
| 19 | Benchmark runs on an in-memory ChromaDB seeded from `data/kb`, and requests `top_k=10` | Reproducible without touching `data/chroma`; MRR needs ranks beyond 3 |
| 20 | `top_k` in the request is optional and defaults to `DEFAULT_TOP_K` on the server | Honors the env default from doc section 8 instead of hardcoding 3 in the schema |
| 21 | Added `LLM_BUDGET_PATH` env variable | The budget file location must be configurable for tests and containers |
| 22 | The KB-empty and reranker-disabled checks run before the query stage | A request that cannot succeed must never cost a Claude call |
| 23 | The candidate window per retriever starts at `CANDIDATES_PER_RETRIEVER` and doubles (up to 320) until it holds `top_k` distinct articles or the retrievers are exhausted | QA: results are grouped by article, so a fixed 20-chunk window returned 8 articles for `top_k=10` |
| 24 | The rerank pool stays at `RERANK_CANDIDATES` chunks, but duplicate-article chunks give way to the best chunk of unseen articles when the top chunks cover fewer than `top_k` articles | QA: `top_k=10` with rerank returned 4 articles; benchmark quality is unchanged by this |
| 25 | `top_k`, `rerank`, and `include_debug` are strict types (`"3"`, `"false"`, `1` → 422) | QA: Pydantic silently coerced wrongly typed switches |
| 26 | Added `RERANK_MIN_SCORE` (opt-in, default 0 = off) | QA: off-topic questions return near-zero rerank scores, but a 0.01 cut-off also removed correct articles on the benchmark (hit@3 0.97 → 0.90), so it is not enabled by default |
