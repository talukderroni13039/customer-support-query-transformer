# customer-support-query-transformer

Finds the right support articles for a customer's follow-up question in a chat.

```
Customer: Do you sell the Trailblazer jacket in waterproof?
Agent:    Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.
Customer: How long do I have to return it?
```

Searching the knowledge base with only "How long do I have to return it?" fails, because the search engine does not know what "it" is. This service:

1. **Rewrites** the question into a standalone query using the recent conversation (Claude `claude-haiku-5-5` + Instructor): `"Return window for the Trailblazer Pro TB-PRO-22 jacket"`.
2. **Searches two ways**: by meaning (vector search, `bge-small-en-v1.5` + ChromaDB) and by exact words (BM25), then merges both lists with Reciprocal Rank Fusion.
3. **Reranks** the top candidates with a cross-encoder (`bge-reranker-base`).
4. **Benchmarks** every stage against naive vector search (`docs/BENCHMARK.md`).

It does **not** write the support answer, and it does **not** store conversations: the client sends the question and recent history with every search.

## How it works

**Store an article**

```
POST /articles {title, content, category}
  → chunk (by headings / paragraphs)
  → embed chunks → save in ChromaDB            (vector index, source of truth)
  → rebuild BM25 from all chunks               (keyword index, in memory)
```

**Search**

```
POST /search {question, history}
  1. QUERY     history empty → use the question
               else → Claude rewrite (last 12 history items) → standalone query
               small talk → stop, empty results
  2. RETRIEVE  vector: embed query → ChromaDB → top 20 chunks
               BM25:   tokenize query → score all chunks → top 20      (in parallel)
  3. FUSE      RRF: score(chunk) = Σ 1 / (60 + rank in each list)
  4. RERANK    top 10 fused chunks → cross-encoder (query, chunk) → re-sort
  5. GROUP     best chunk per article → top_k articles
```

Each stage can be switched off per request, which is how the benchmark compares them.

| Data | Where | Lifetime |
|---|---|---|
| Articles (chunks + vectors + metadata) | ChromaDB on disk (`data/chroma`) | permanent; source of truth |
| BM25 keyword index | memory | rebuilt from ChromaDB at startup and after every article change |
| Conversations | nowhere | sent with each request, discarded |
| Rewrites | `data/cache/rewrites.json` | cache only |

## API

Base path `/api/v1/support`. Every response uses one envelope:

```json
{"success": true, "data": {}, "meta": {"request_id": "...", "processing_time_ms": 12.3}}
{"success": false, "error": {"code": "ARTICLE_NOT_FOUND", "message": "...", "details": null}, "meta": {}}
```

### Articles

| Method | Path | Body | Success |
|---|---|---|---|
| `POST` | `/articles` | `{title, content, category?}` | `201` → `ArticleResult` |
| `GET` | `/articles?category=returns` | | `200` → list of `ArticleSummary` |
| `GET` | `/articles/{article_id}` | | `200` → `ArticleResult` (full content) |
| `PUT` | `/articles/{article_id}` | `{title, content, category?}` | `200` → `ArticleResult` |
| `DELETE` | `/articles/{article_id}` | | `200` → `{"id": "KB-...", "deleted": true}` |

Validation: `title` 3–200 chars; `content` 20–20,000 chars; `category` optional, lowercase letters/digits/hyphens, ≤ 50 chars.

### Search: `POST /search`

```json
{
  "question": "How long do I have to return it?",
  "history": [
    "Do you sell the Trailblazer jacket in waterproof?",
    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof."
  ],
  "top_k": 3,
  "query_mode": "rewrite",
  "retrieval": "hybrid",
  "rerank": true,
  "include_debug": false
}
```

| Field | Values | Default |
|---|---|---|
| `question` | 1–2,000 chars, required | |
| `history` | plain strings, oldest first, ≤ 100 items (only the last 12 are used) | `[]` |
| `top_k` | 1–10 | `3` |
| `query_mode` | `rewrite` · `question_only` · `full_history` | `rewrite` |
| `retrieval` | `hybrid` · `vector` · `keyword` | `hybrid` |
| `rerank` | `true` · `false` | `true` |
| `include_debug` | `true` · `false` | `false` |

Response `data` (`SupportSearchResult`): `question`, `search_query`, `query_mode`, `retrieval`, `rerank`, `needs_retrieval`, `rewrite_source`, `rewrite_fallback`, `rerank_fallback`, `score_type`, and `results[]` of `{article_id, title, category, score, snippet, debug?}`.

`score_type` tells you which stage produced `score`: `cosine` (vector only), `bm25` (keyword only), `rrf` (hybrid, no rerank), `rerank` (reranked). With `include_debug: true` each result carries `vector_rank`, `keyword_rank`, `rrf_score`, `rerank_score`.

### Health: `GET /health`

`data`: `status`, `environment`, `embedding_model_loaded`, `reranker_loaded`, `vector_store`, `keyword_index_chunks`, `rewrite_provider`.

### Errors

| HTTP | `code` | When |
|---|---|---|
| 422 | `VALIDATION_ERROR` | Bad body or parameters |
| 404 | `ARTICLE_NOT_FOUND` | Unknown article ID |
| 409 | `KB_EMPTY` | Search with no articles |
| 409 | `RERANKER_DISABLED` | `rerank: true` while `RERANK_ENABLED=false` |
| 503 | `MODEL_NOT_READY` | Models not loaded |
| 503 | `VECTOR_STORE_ERROR` | ChromaDB failure |
| 502 / 504 | `LLM_ERROR` / `LLM_TIMEOUT` | Only when `REWRITE_STRICT=true` |
| 500 | `INTERNAL_ERROR` | Anything else (logged, generic message) |

A failed rewrite never fails a search (falls back to the question, `rewrite_fallback: true`). A failed reranker returns the fused order (`rerank_fallback: true`).

## Setup (PowerShell)

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you) and, for the rewrite stage, an Anthropic API key.

```powershell
uv sync                                   # install from uv.lock (CPU-only torch)
Copy-Item .env.example .env.dev           # then put your key in ANTHROPIC_API_KEY
uv run python -m scripts.seed_kb          # load the 40 starter articles from data/kb
$env:APP_ENV = "dev"; uv run uvicorn src.main:app --reload
```

Open http://127.0.0.1:8000/docs. The first start downloads `BAAI/bge-small-en-v1.5` (~130 MB) and `BAAI/bge-reranker-base` (~1.1 GB) into the Hugging Face cache; afterwards set `HF_HUB_OFFLINE=1` in `.env.dev`.

Environment is loaded from `.env.{APP_ENV}` (`.env.dev`, `.env.uat`), both gitignored. Every non-secret value has a default in code; `.env.example` lists them all. Without a key, or with `REWRITE_PROVIDER=none`, everything works except the rewrite (searches fall back to the question and report `rewrite_fallback: true`).

## Try it (PowerShell)

```powershell
$base = "http://127.0.0.1:8000/api/v1/support"

# Create, read, update, delete an article
$body = @{ title = "Snowshoe Rental"; category = "rentals"
           content = "## Prices`nRent MSR-EVO-9 snowshoes for 15 dollars per day at the flagship store." } | ConvertTo-Json
$created = Invoke-RestMethod "$base/articles" -Method Post -ContentType "application/json" -Body $body
$id = $created.data.id
Invoke-RestMethod "$base/articles/$id"
Invoke-RestMethod "$base/articles?category=rentals"
Invoke-RestMethod "$base/articles/$id" -Method Put -ContentType "application/json" -Body $body
Invoke-RestMethod "$base/articles/$id" -Method Delete

# Search: the full pipeline (rewrite → hybrid → rerank)
$search = @{
  question = "How long do I have to return it?"
  history  = @("Do you sell the Trailblazer jacket in waterproof?",
               "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.")
  include_debug = $true
} | ConvertTo-Json
(Invoke-RestMethod "$base/search" -Method Post -ContentType "application/json" -Body $search).data
```

Change one switch at a time (in Swagger or in the body above) and compare `search_query`, `score_type`, and the debug ranks:

| Body | Pipeline | `score_type` |
|---|---|---|
| `"query_mode": "question_only", "retrieval": "vector", "rerank": false` | naive vector search | `cosine` |
| `"query_mode": "question_only", "retrieval": "keyword", "rerank": false` | BM25 only | `bm25` |
| `"query_mode": "question_only", "retrieval": "hybrid", "rerank": false` | vector + BM25 fused with RRF | `rrf` |
| `"query_mode": "full_history", "retrieval": "vector", "rerank": false` | naive with the last 12 history items pasted in | `cosine` |
| `"query_mode": "rewrite", "retrieval": "hybrid", "rerank": false` | Claude rewrite + hybrid | `rrf` |
| `"query_mode": "rewrite", "retrieval": "hybrid", "rerank": true` (defaults) | full pipeline | `rerank` |

`rewrite_source` tells you what the query stage did: `llm`, `cache`, `skipped` (no history, Claude not called), `fallback` (Claude failed), `disabled` (no provider or key), or `none` (a baseline `query_mode`).

## Benchmark

`uv run python -m scripts.run_benchmark` runs the 33 eval conversations through six configurations and regenerates the tables in [`docs/BENCHMARK.md`](docs/BENCHMARK.md). Summary of the committed run (30 retrieval cases, 3 small-talk cases):

| # | Config | hit@1 | hit@3 | MRR | small talk | avg latency |
|---|---|---|---|---|---|---|
| 1 | Naive (question only, vector) | 0.63 | 0.83 | 0.767 | 0/3 | ~0.1 s |
| 3 | + Rewrite | 0.70 | 0.97 | 0.831 | 3/3 | ~0.8 s |
| 4 | + Hybrid (BM25 + RRF) | 0.80 | 0.93 | 0.870 | 3/3 | ~0.8 s |
| 5 | Full pipeline (+ rerank) | 0.77 | 0.97 | 0.864 | 3/3 | ~3.3 s |

- The rewrite fixes follow-ups (follow-up hit@1 0.53 → 0.65, corrections 0.50 → 1.00) and detects small talk.
- BM25 fixes exact product codes (`exact_code` hit@1 0.80 → 1.00).
- Reranking improves hit@3 but **not** hit@1 or MRR on this set, and it pushes p95 latency to about 4.7 s on the laptop CPU used, above the 3 s target. Both shortfalls, their causes, and the options are explained in `docs/BENCHMARK.md`.

## Cost and latency notes

- Claude is called only for `query_mode: "rewrite"` with non-empty history, with at most the last 12 history items, and results are cached in `data/cache/rewrites.json` (keys are hashes; the question and history are not written to disk).
- A hard cap on real Claude calls is enforced in code: every HTTP attempt increments `data/cache/llm_call_budget.json`, and at `LLM_CALL_BUDGET` (50) the rewriter refuses with "LLM budget exhausted" and searches fall back to the question. The build used 33 calls and the later QA pass 6 more (39 in total). Raise `LLM_CALL_BUDGET` or delete that one file to reset the counter for your own use.
- Reranking is the slow stage on CPU (about 0.3 s per candidate on a 12th-gen i5 laptop). `RERANK_CANDIDATES` trades quality for latency; `"rerank": false` skips it per request; `RERANK_ENABLED=false` skips loading the model.
- There is no relevance cut-off by default: an off-topic question still returns the nearest articles, with low scores. `RERANK_MIN_SCORE` is an opt-in filter on rerank scores; on the benchmark a value of 0.01 also removed correct articles, so calibrate it on your own data before enabling it.
- `top_k`, `rerank`, and `include_debug` are strictly typed: `"3"` or `"false"` is a 422, not a silent conversion.
- Questions, history, and article content are never logged; logs carry lengths, modes, counts, timings, and a request ID.

## Tests and quality gates

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest                      # default suite, fakes only: no API calls, no model downloads
uv run pytest -m model --no-cov    # real models (downloads on first run); run once
uv run pytest -m live --no-cov     # 5 real Claude calls (up to 10 with retries); run once
uv run pre-commit install          # optional: run ruff + mypy before each commit
```

## Docker

```powershell
docker build -t support-query-transformer .
docker run --rm -p 8000:8000 --env-file .env.uat `
  -v sqt-chroma:/app/data/chroma `
  -v sqt-hf:/home/app/.cache/huggingface `
  support-query-transformer
# seed the container's knowledge base (once per chroma volume)
docker run --rm --env-file .env.uat -v sqt-chroma:/app/data/chroma -v sqt-hf:/home/app/.cache/huggingface `
  support-query-transformer python -m scripts.seed_kb
```

The image is multi-stage, CPU-only, and runs as a non-root user. The vector index and the Hugging Face model cache live in volumes; `.env*` files are never copied into the image. `GET /health` reports `embedding_model_loaded`, `reranker_loaded`, `vector_store`, `keyword_index_chunks`, and `rewrite_provider`.

## Frontend (Angular)

`frontend/` holds a small Angular 17 app with a search screen (all switches, debug ranks) and an articles screen (upload a `.md` / `.txt` file, list, delete). With the backend running on port 8000:

```powershell
cd frontend
npm install     # first time only
npm start       # http://localhost:4200, proxies /api and /health to the backend
```

Details are in [`frontend/README.md`](frontend/README.md).

## Project layout

```
frontend/                        Angular app (search + article upload), dev proxy to the API
src/
  main.py, api_router.py         app creation, central /api/v1 router
  core/                          DI providers, lifespan, exceptions + ONE handler, middleware, envelope
  domains/support_search/        routers/ schemas/ services/ prompts/ chunking/
  llm/ embeddings/ vectorstore/ keyword_search/ reranking/    Protocol (base.py) + one implementation each
  utils/                         rank_fusion.py (RRF), hashing.py, env.py
scripts/                         seed_kb.py, run_benchmark.py
data/kb/, data/eval/             starter articles and eval conversations (committed)
tests/                           unit/, integration/, model/ (marker), live/ (marker), fakes.py
docs/                            DECISIONS.md (deviations from the design doc), BENCHMARK.md
```

Conventions and "how to extend" notes are in [`CLAUDE.md`](CLAUDE.md); the design baseline is [`support-query-transformer-workflow.md`](support-query-transformer-workflow.md).
