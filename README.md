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

## Setup, usage, and benchmark

See the sections below (completed in the final build step).
