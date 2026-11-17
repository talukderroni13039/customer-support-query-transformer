# CLAUDE.md

## 1. Project summary

Customer Support Query Transformer: a FastAPI service that finds the right support articles for a chat follow-up.
Pipeline: **rewrite** (Claude + Instructor) → **hybrid retrieval** (ChromaDB vector + BM25, fused with RRF) → **rerank** (cross-encoder) → group by article.
It never writes the answer and never stores conversations.

## 2. Commands (PowerShell)

```powershell
uv sync                                              # install (Python 3.12, uv.lock)
uv run python -m scripts.seed_kb                     # seed data/kb/*.md into ChromaDB
$env:APP_ENV = "dev"; uv run uvicorn src.main:app --reload    # run dev  (loads .env.dev)
$env:APP_ENV = "uat"; uv run uvicorn src.main:app             # run uat  (loads .env.uat)
uv run pytest                                        # default tests: fakes only, 0 API calls, coverage >= 80%
uv run pytest -m model --no-cov                      # real-model smoke tests (downloads models; run once)
uv run pytest -m live --no-cov                       # live Claude tests (max 5 calls; run once)
uv run python -m scripts.run_benchmark               # six configs x eval set -> docs/BENCHMARK.md
uv run ruff check .                                  # lint
uv run ruff format .                                 # format   (CI uses: ruff format --check .)
uv run mypy src                                      # type check (strict)
docker build -t support-query-transformer .
docker run --rm -p 8000:8000 --env-file .env.uat -v sqt-chroma:/app/data/chroma -v sqt-hf:/home/app/.cache/huggingface support-query-transformer
```

## 3. Architecture rules

- `src/main.py`: env loading (`.env.{APP_ENV}`), app creation, middleware, handlers, routers. `src/api_router.py`: central `/api/v1` router.
- `src/core/`: cross-cutting code only: `dependencies.py` (ALL DI providers), `lifespan.py` (models, ChromaDB, BM25, rewriter created once), `exceptions.py`, `exception_handlers.py` (ONE handler), `middleware.py`, `logging.py`, `health.py`, `schemas/` (envelope, `ChunkHit`/`StoredChunk`, health).
- `src/domains/<domain>/routers/`: HTTP only. `schemas/`: Pydantic request/response/LLM shapes. `services/`: business logic. `prompts/`: prompt text + `PROMPT_VERSION`. `chunking/`: content → chunks.
- Infrastructure folders, each with `base.py` (Protocol) + one implementation: `llm/` (rewriter, cache decorator, call budget), `embeddings/`, `vectorstore/`, `keyword_search/` (tokenizer + BM25), `reranking/`. `utils/`: pure helpers (`rank_fusion.py`, `hashing.py`, `env.py`).
- **Layer rule:** router → service → Protocols. Services never import `chromadb`, `rank_bm25`, `sentence_transformers`, `anthropic`, or `instructor`. Only `core/lifespan.py` builds concrete implementations.
- No domain-level `dependencies.py` or `constants.py`. No `config.py` / settings class: read env with `utils/env.py` helpers where the value is used at startup; every non-secret value has a default in code.

## 4. Pipeline rules

- **Query stage** (`query_service.py`): `question_only` → question; `full_history` → last 12 history items + question; `rewrite` → empty history skips Claude, otherwise `CachedRewriter` → Claude. Rewrite error → question + `rewrite_fallback=true`. `needs_retrieval=false` → empty results, no retrieval.
- **Retrieval stage** (`retrieval_service.py`): `vector` | `keyword` | `hybrid` (both in parallel threads), top `CANDIDATES_PER_RETRIEVER` each.
- **Fusion:** `utils/rank_fusion.reciprocal_rank_fusion` (pure, `RRF_K=60`), ranks only, never raw scores.
- **Rerank:** only the top `RERANK_CANDIDATES` of the ranked list; error → keep prior order + `rerank_fallback=true`.
- **Grouping** (`support_search_service.py`): best chunk per article → `top_k` articles.
- `ChunkHit` is the one shape passed between stages. `score_type` says which stage produced `score` (`cosine` / `bm25` / `rrf` / `rerank`).
- **Any article write (create/update/delete/seed) must rebuild BM25** from ChromaDB (atomic swap, under the index write lock).
- CPU-bound work (embedding, Chroma calls, BM25 search/rebuild, reranking) always runs via `asyncio.to_thread`.

## 5. Naming conventions

- snake_case modules with role suffixes: `*_service.py`, `*_schemas.py`, `*_prompt.py`, `api_endpoints.py`, `base.py` for Protocols.
- PascalCase classes. Domain data returned in `ApiResponse.data` is named `*Result` (`ArticleResult`, `SupportSearchResult`, `HealthResult`).
- Requests: `*Create`, `*Update`, `*Request`. Enums: `QueryMode`, `RetrievalMode`, `ScoreType`.
- Tests mirror modules: `tests/unit/test_<module>.py`, `tests/integration/test_<area>_api.py`.

## 6. Response contract

- Every endpoint returns `ApiResponse[<DomainResult>]`: `{"success": true, "data": ..., "meta": {"request_id", "processing_time_ms"}}`.
- Errors are raised as `AppError` subclasses (`core/exceptions.py`) and rendered as `ErrorResponse` by the single handler in `core/exception_handlers.py`. Never return raw dicts, never raise `HTTPException` in services.

## 7. Adding a retrieval stage or query mode (e.g. multi-query, HyDE)

1. New infrastructure → Protocol in `<folder>/base.py` + implementation; otherwise a method on `QueryService` / `RetrievalService`.
2. Add the enum value (`QueryMode` / `RetrievalMode`) in `search_schemas.py`; extend `StageDebug` if the stage has a score.
3. Wire it in `core/lifespan.py` (create once) and `core/dependencies.py` (provider).
4. Tests: fake for the Protocol in `tests/fakes.py`, unit tests for the service branch, one API test for the new enum value.
5. Add a config row in `scripts/run_benchmark.py` and re-run the benchmark.

## 8. Adding a new domain

Create `src/domains/<name>/{routers,schemas,services,prompts,chunking}/` (only the ones needed), define schemas (`*Result`), services depending on Protocols, a router in `routers/api_endpoints.py`, include it in `src/api_router.py`, add `AppError` subclasses to `core/exceptions.py`, providers to `core/dependencies.py`, and tests under `tests/unit` + `tests/integration`.

## 9. Adding a database later

- `models/` = SQLAlchemy ORM classes; `schemas/` stays Pydantic. Add `repositories/` per domain (services depend on repository Protocols), `core/database.py` (engine + session provider), Alembic for migrations.
- Articles would move to PostgreSQL as the source of truth; ChromaDB and BM25 become pure search indexes rebuilt from the database.

## 10. Testing rules

- Default `uv run pytest` uses `FakeRewriter`, `FakeEmbedder`, `FakeReranker`, Chroma `EphemeralClient()`, and the real `BM25Index`. Zero API calls, zero model downloads.
- `@pytest.mark.model` (real models) and `@pytest.mark.live` (real Claude, max 5 cases, `max_retries=1`) are excluded by default and run once.
- Hard cap: 50 real Anthropic calls for the whole build, enforced by `LlmCallBudget` (`data/cache/llm_call_budget.json`). Never debug with live calls. Never delete `data/cache/`.
- Coverage threshold: 80% on `src/` (enforced in `pyproject.toml`).

## 11. Security, privacy, cost

- API key only via `os.getenv("ANTHROPIC_API_KEY")`; never printed, logged, hardcoded, or committed. `.env.dev` / `.env.uat` are gitignored and never edited by tooling; `.env.example` has placeholders only.
- Never log questions, history, or article content at INFO or above; log lengths, modes, counts, and timings instead.
- Conversations are never stored. The rewrite cache (`data/cache/rewrites.json`) is gitignored.
- Rewrites are cached; empty history skips Claude; only the last 12 history items are sent.

## 12. Do not

- Add a settings class, `config.py`, or a DI container.
- Put logic in routers or return raw dicts.
- Log user text.
- Load models per request (only in `core/lifespan.py`).
- Forget the BM25 rebuild after an article write.
- Call real APIs or real models in default tests.
- Commit to or push `main`.
