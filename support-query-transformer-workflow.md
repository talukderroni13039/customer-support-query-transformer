# Customer Support Query Transformer — Project Workflow

Week 3 · Advanced RAG Patterns · Practical Assessment

---

## 1. Overview

### 1.1 The problem

In a support chat, customers ask follow-up questions that make no sense on their own:

```
Customer: Do you sell the Trailblazer jacket in waterproof?
Agent:    Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof.
Customer: How long do I have to return it?
```

Searching the knowledge base with only **"How long do I have to return it?"** fails: the search engine doesn't know what "it" is.

### 1.2 The goal

1. **Rewrite** the question into one clear, standalone search query using the recent conversation:
   `"Return window for the Trailblazer Pro TB-PRO-22 jacket"`
2. **Search two ways** — by meaning (vector search) and by exact words (BM25 keyword search) — and merge the results (RRF).
3. **Rerank** the top candidates with a cross-encoder so the best article comes first.
4. **Benchmark** this advanced pipeline against naive vector search.

**Analogy:** a receptionist turns your vague question into a clear request (rewrite). Two librarians search: one understands topics (vector), one checks exact titles and part numbers (BM25). Their lists are merged (RRF). A senior librarian reads the top 10 carefully and puts the best one first (reranker).

### 1.3 What you build

1. **Articles API** — standard REST endpoints to add, read, update, and delete support articles. Every change updates both the vector index and the keyword index.
2. **Query rewriter** — question + recent history → standalone query (Claude + Instructor).
3. **Hybrid retrieval** — vector search + BM25, fused with Reciprocal Rank Fusion.
4. **Reranker** — `bge-reranker-base` re-scores the top candidates.
5. **Search API** — one endpoint with switches for each stage, so every combination can be tested.
6. **Benchmark** — naive vector search vs each added stage vs the full pipeline.

### 1.4 What it does NOT do

- It does **not** write the support answer. It finds the right articles.
- It does **not** store conversations. The client sends the question and recent history with every search.

### 1.5 How it maps to Week 3 topics

| Week 3 topic | Where it appears |
|---|---|
| Query transformation (Mon–Tue) | Conversational query rewriting |
| Hybrid search + RRF (Wed–Thu) | BM25 + vector search, fused with RRF |
| Cross-encoder reranking (Fri) | `bge-reranker-base` on fused candidates |
| Benchmark vs naive (Weekend) | Stage-by-stage benchmark table |
| Multi-Query, Step-back, HyDE | Stretch goals |

---

## 2. How It Works

### 2.1 BM25 in plain language

BM25 is a **points game for words**. For every chunk, it adds up points for each query word the chunk contains:

| Rule | Meaning | Example |
|---|---|---|
| **Rare words score more** (IDF) | A word in few chunks is a strong signal | `tb-pro-22` appears in 1 chunk → many points; `jacket` appears in 30 → few |
| **Repeats help, but less each time** (TF saturation) | Mentioning a word 3× beats 1×, but 20× isn't 20× better | stops keyword-stuffed chunks from winning |
| **Long chunks are penalised a little** (length normalisation) | A match in a short, focused chunk counts more | a 2-line chunk about returns beats a 50-line mixed chunk |

Example — query `"Is TB-PRO-22 waterproof?"` → tokens `[tb-pro-22, waterproof]` (stop word `is` removed):

| Chunk | Contains | BM25 score |
|---|---|---|
| "Trailblazer Pro **TB-PRO-22** is fully **waterproof**..." | both, one rare | **high** |
| "Summit Shell TB-PRO-23 is **waterproof**..." | only the common word | low |
| "Our **waterproof** rating guide..." | only the common word | low |

Vector search would rate the first two almost equal (both "waterproof jackets"). BM25 separates them because the exact code matches.

### 2.2 RRF (Reciprocal Rank Fusion) in plain language

Vector scores (0–1) and BM25 scores (0–20+) are on different scales, so you can't add them. RRF ignores scores and uses **positions**:

```
rrf_score(chunk) = Σ  1 / (60 + rank in each list)
```

A chunk ranked #1 in vector and #3 in BM25 gets `1/61 + 1/63`. A chunk found by only one list gets points from only that list. Chunks found by **both** float to the top.

### 2.3 Reranker in plain language

The first searches are fast but rough: they score each chunk without really reading it against the question. The reranker (a **cross-encoder**) reads the question and each candidate **together** and outputs a relevance score. It's slower, so it only checks the top 10 fused candidates.

**Analogy:** a recruiter skims 500 CVs and shortlists 10 (vector + BM25 + RRF). An interviewer meets the 10 properly and ranks them (reranker).

### 2.4 Request flows

**Store an article**
```
POST /articles {title, content, category}
  → chunk (by headings / paragraphs)
  → embed chunks (bge-small) → save in ChromaDB          (vector index)
  → rebuild BM25 from all chunks                          (keyword index, in memory, ~ms)
  → return article ID
```

**Search (full pipeline)**
```
POST /search {question, history}
  │
  ├─ 1. QUERY      history empty → use question
  │                else → Claude rewrite (last 12 history items) → standalone query
  │                small talk → stop, empty results
  │
  ├─ 2. RETRIEVE   ┌─ vector: embed query → ChromaDB → top 20 chunks (meaning)
  │   (parallel)   └─ BM25:   tokenize query → score all chunks → top 20 (exact words)
  │
  ├─ 3. FUSE       RRF merges both lists → one ranked list
  │
  ├─ 4. RERANK     top 10 fused chunks → cross-encoder scores (query, chunk) → re-sort
  │
  └─ 5. GROUP      best chunk per article → top_k articles → response
```

Each stage can be switched off in the request, which is how the benchmark compares them.

### 2.5 What is stored where

| Data | Where | Lifetime |
|---|---|---|
| Articles (chunks + vectors + metadata) | ChromaDB on disk | permanent; source of truth |
| BM25 keyword index | Memory | rebuilt from ChromaDB at startup and after every article change |
| Conversations | Nowhere | sent with each request, discarded |
| Rewrites | `data/cache/rewrites.json` | cache only |

### 2.6 Rewrite rules

- Replace pronouns and vague references with the real thing from the history.
- Keep product names, SKUs, and order numbers **exactly** as written (BM25 depends on this).
- Do not answer. Do not add facts not in the conversation.
- Small talk → `needs_retrieval = false`.
- Already-clear question → nearly unchanged.

### 2.7 Cost and speed safeguards

| Safeguard | Effect |
|---|---|
| Last 12 history items only | Long chats stay fast and cheap |
| No history → skip Claude | First questions cost nothing |
| Rewrite cache | Repeated questions cost nothing |
| Rewrite fallback | Claude down → search with the question alone |
| Rerank only top 10 | Keeps the cross-encoder under ~0.5 s on CPU |

Typical latency: rewrite ~0.5–1.5 s · vector ~30 ms · BM25 ~5 ms · RRF < 1 ms · rerank ~0.2–0.6 s.

---

## 3. Tools — What and Why

| Tool | Why it's needed |
|---|---|
| **FastAPI** | Builds the REST API with automatic validation and Swagger docs at `/docs`. Same framework and structure as Weeks 1–2. |
| **Anthropic SDK** (`claude-haiku-5-5`) | Calls Claude to rewrite vague follow-ups into standalone queries. Rewriting is a small language task, so the cheapest fast model is enough. |
| **Instructor** | Makes Claude return a typed object (`standalone_query`, `needs_retrieval`) instead of free text. It validates the output and retries automatically, like Week 1. |
| **Pydantic** | Defines and validates every request, response, and LLM output shape. FastAPI and Instructor are both built on it. |
| **sentence-transformers** + `bge-small-en-v1.5` | Turns chunks and queries into meaning vectors locally at no cost. The small model is fast on CPU and plenty for ~40 articles. |
| **ChromaDB** | Stores chunks, vectors, and metadata, and runs the vector search. It runs inside Python with no server. |
| **rank_bm25** | Keyword search over the same chunks (`BM25Okapi`). It catches exact words and codes like `TB-PRO-22` that meaning search blurs together. |
| **sentence-transformers `CrossEncoder`** + `BAAI/bge-reranker-base` | Reads the query and each top candidate together and re-scores them. It runs locally (one-time ~1 GB download), so reranking adds no cost. |
| **python-dotenv** | Loads settings and the API key from `.env.dev` / `.env.uat`. Keeps secrets out of code. |
| **PyYAML** | Reads front matter of the seed article files. Used only by the seed script. |
| **uv** | Installs dependencies with a lockfile. Faster than pip and keeps versions identical. |
| **ruff** | Lints and formats in one tool. Catches mistakes before commit. |
| **mypy** | Checks type hints. Catches wrong types before runtime. |
| **pytest** (+ `pytest-asyncio`, `pytest-cov`) | Runs tests and measures coverage. Tests use fakes: no API calls, no model downloads. |
| **pre-commit** | Runs ruff and mypy before every commit. Broken code can't slip in. |
| **Docker** | Same runtime locally and in UAT. Volumes keep the vector DB and model cache outside the image. |

**Not used:** `cohere` (needs another paid account; `bge-reranker` does the same locally) and LangChain / LlamaIndex (building each stage yourself is the learning goal; each is small).

---

## 4. Requirements

### 4.1 Functional

| ID | Requirement |
|---|---|
| FR1 | `POST /articles` creates an article, chunks + embeds + indexes it in ChromaDB and BM25, returns its ID |
| FR2 | `GET /articles` (optional `category` filter) and `GET /articles/{id}` (full content) |
| FR3 | `PUT /articles/{id}` replaces and re-indexes; `DELETE /articles/{id}` removes from both indexes |
| FR4 | Seed script loads `data/kb/*.md` with fixed IDs |
| FR5 | `POST /search` accepts `question`, optional `history`, `top_k`, `query_mode`, `retrieval`, `rerank`, `include_debug` |
| FR6 | `query_mode`: `rewrite` (default) · `question_only` · `full_history` |
| FR7 | `retrieval`: `hybrid` (default) · `vector` · `keyword` |
| FR8 | `rerank`: `true` (default) re-scores the top `RERANK_CANDIDATES` with the cross-encoder |
| FR9 | Hybrid fuses vector and BM25 top `CANDIDATES_PER_RETRIEVER` with RRF (`RRF_K` = 60) |
| FR10 | BM25 index built from ChromaDB at startup and rebuilt after every create/update/delete |
| FR11 | BM25 tokenizer lowercases, keeps hyphenated codes as one token (`tb-pro-22`), removes common stop words |
| FR12 | Rewrite: empty history skips Claude; only the last `MAX_HISTORY_MESSAGES` (12) sent; small talk → no results |
| FR13 | Rewrites cached; Claude failure → fall back to `question_only`, flag `rewrite_fallback` |
| FR14 | Results grouped by article (best chunk per article), with `score` and `score_type` (`cosine` / `bm25` / `rrf` / `rerank`) |
| FR15 | `include_debug=true` adds per-result stage details: vector rank, BM25 rank, RRF score, rerank score |
| FR16 | Benchmark script compares pipeline configurations (section 9.3) |
| FR17 | `GET /health` reports both models, vector store, BM25 index size, rewrite provider |
| FR18 | All responses use the shared `ApiResponse` / `ErrorResponse` envelope |

### 4.2 Non-Functional

| Area | Requirement |
|---|---|
| Latency | Full pipeline p95 < 3 s on CPU; without rewrite < 1 s; article create/update < 2 s |
| Cost | Hard cap: **50 real Anthropic calls** for the whole build; cache makes benchmark re-runs free |
| Resilience | Rewrite failure never fails a search; reranker failure → return fused order and flag `rerank_fallback` |
| Consistency | BM25 rebuild is atomic (build new index, then swap the reference) so searches never see a half-built index |
| Input limits | `history` ≤ 100 items (last 12 used); each item and `question` ≤ 2,000 chars; article `content` ≤ 20,000 chars |
| Privacy | Questions and history never logged at INFO or above; API key only in env files |
| Testability | Protocols for rewriter, embedder, vector store, keyword index, reranker; default tests use fakes |
| Maintainability | Same structure and conventions as Weeks 1–2; ruff + mypy clean; coverage ≥ 80% |
| Portability | Docker with volumes for ChromaDB and the Hugging Face model cache |

---

## 5. Architecture

Same conventions as Weeks 1–2: domain-based layout, `core/` for cross-cutting code, shared response envelope, one global exception handler, all DI in `core/dependencies.py`, no `config.py`, infrastructure behind Protocols.

### 5.1 Folder Structure

```
support-query-transformer/
├── src/
│   ├── __init__.py
│   ├── main.py                          # env loading, app creation, lifespan, router + handler + middleware
│   ├── api_router.py                    # central /api/v1 router
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dependencies.py              # ALL DI providers
│   │   ├── lifespan.py                  # load models, open ChromaDB, build BM25 once at startup
│   │   ├── exceptions.py                # AppError + subclasses
│   │   ├── exception_handlers.py        # ONE global handler
│   │   ├── middleware.py                # request ID + start time
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── api_response.py          # ApiResponse[T], ErrorResponse, ResponseMeta
│   │
│   ├── domains/
│   │   ├── __init__.py
│   │   └── support_search/
│   │       ├── __init__.py
│   │       ├── routers/
│   │       │   ├── __init__.py
│   │       │   └── api_endpoints.py             # /articles (CRUD) + /search
│   │       ├── schemas/
│   │       │   ├── __init__.py                  # re-exports
│   │       │   ├── article_schemas.py           # ArticleCreate, ArticleUpdate, ArticleResult, ArticleSummary
│   │       │   ├── search_schemas.py            # SearchRequest, QueryMode, RetrievalMode, RetrievedArticle, StageDebug, SupportSearchResult
│   │       │   └── rewrite_schemas.py           # RewrittenQuery (Instructor response_model)
│   │       ├── services/
│   │       │   ├── __init__.py
│   │       │   ├── article_service.py           # CRUD → chunk + embed + Chroma + BM25 rebuild
│   │       │   ├── query_service.py             # question + history → search query (rewrite / baselines)
│   │       │   ├── retrieval_service.py         # vector / BM25 / RRF / rerank → ranked chunks
│   │       │   └── support_search_service.py    # orchestrates query → retrieval → group by article
│   │       ├── prompts/
│   │       │   ├── __init__.py
│   │       │   └── rewrite_prompt.py            # system prompt + PROMPT_VERSION
│   │       └── chunking/
│   │           ├── __init__.py
│   │           └── article_chunker.py           # content → chunks (headings, else paragraphs)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                      # QueryRewriter Protocol
│   │   ├── anthropic_rewriter.py        # Instructor + AsyncAnthropic
│   │   └── cached_rewriter.py           # file cache wrapper (same Protocol)
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                      # Embedder Protocol
│   │   └── bge_embedder.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py                      # VectorStore Protocol
│   │   └── chroma_store.py
│   │
│   ├── keyword_search/
│   │   ├── __init__.py
│   │   ├── base.py                      # KeywordIndex Protocol
│   │   ├── tokenizer.py                 # lowercase, keep hyphenated codes, stop words
│   │   └── bm25_index.py                # rank_bm25 BM25Okapi, atomic rebuild
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── base.py                      # Reranker Protocol
│   │   └── bge_reranker.py              # sentence-transformers CrossEncoder
│   │
│   └── utils/
│       ├── __init__.py
│       ├── hashing.py                   # stable hash for cache keys
│       └── rank_fusion.py               # reciprocal_rank_fusion() — pure function
│
├── scripts/
│   ├── seed_kb.py                       # load data/kb/*.md through ArticleService (fixed IDs)
│   └── run_benchmark.py                 # pipeline configs × eval set → metrics tables
│
├── data/
│   ├── kb/                              # starter articles (committed)
│   ├── eval/conversations.json          # benchmark cases (committed)
│   ├── cache/                           # rewrite cache + LLM call budget (gitignored)
│   └── chroma/                          # vector index (gitignored)
│
├── tests/
│   ├── conftest.py                      # fakes for every Protocol, in-memory Chroma, overrides
│   ├── fixtures/
│   ├── unit/
│   │   ├── test_article_chunker.py
│   │   ├── test_tokenizer.py
│   │   ├── test_bm25_index.py
│   │   ├── test_rank_fusion.py
│   │   ├── test_rewrite_schemas.py
│   │   ├── test_cached_rewriter.py
│   │   ├── test_chroma_store.py
│   │   ├── test_article_service.py
│   │   ├── test_query_service.py
│   │   ├── test_retrieval_service.py
│   │   └── test_support_search_service.py
│   ├── integration/
│   │   ├── test_articles_api.py
│   │   └── test_search_api.py
│   └── live/
│       └── test_rewrite_live.py         # @pytest.mark.live, max 5
│
├── docs/
│   ├── DECISIONS.md
│   └── BENCHMARK.md
├── .env.example
├── .env.dev
├── .env.uat
├── .gitignore
├── .dockerignore
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── CLAUDE.md
└── README.md
```

### 5.2 Protocols

```python
# llm/base.py
class QueryRewriter(Protocol):
    async def rewrite(self, question: str, history: list[str]) -> RewrittenQuery: ...


# embeddings/base.py
class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# vectorstore/base.py
class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...
    def delete_article(self, article_id: str) -> int: ...
    def get_article_chunks(self, article_id: str) -> list[StoredChunk]: ...
    def get_all_chunks(self) -> list[StoredChunk]: ...  # feeds the BM25 rebuild
    def list_articles(self, category: str | None = None) -> list[ArticleSummary]: ...
    def query(self, embedding: list[float], top_k: int) -> list[ChunkHit]: ...
    def count(self) -> int: ...


# keyword_search/base.py
class KeywordIndex(Protocol):
    def rebuild(self, chunks: list[StoredChunk]) -> None: ...
    def search(self, query: str, top_k: int) -> list[ChunkHit]: ...
    def size(self) -> int: ...


# reranking/base.py
class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, candidates: list[ChunkHit]) -> list[ChunkHit]: ...
```

`ChunkHit` = `chunk_id`, `article_id`, `title`, `category`, `text`, `score`. Shared by vector search, BM25, RRF, and the reranker, so every stage passes the same shape to the next.

### 5.3 Responsibility of Each Module

| Module | Does | Does NOT |
|---|---|---|
| `routers/api_endpoints.py` | Receives requests, calls services, wraps in `ApiResponse` | Any search or indexing logic |
| `services/article_service.py` | IDs, chunking, embedding, Chroma writes, triggers BM25 rebuild | Know HTTP or library APIs |
| `services/query_service.py` | Builds the search query per `query_mode`; trimming, shortcuts, cache, fallback | Retrieval |
| `services/retrieval_service.py` | Runs vector / BM25 / both, fuses with RRF, reranks, records stage debug | Query building, HTTP |
| `services/support_search_service.py` | Calls query → retrieval → groups chunks by article → result | Library details |
| `keyword_search/tokenizer.py` | Text → tokens (same function for chunks and queries) | Scoring |
| `keyword_search/bm25_index.py` | Builds `BM25Okapi`, maps scores back to chunks, atomic swap | Know about articles API |
| `utils/rank_fusion.py` | `reciprocal_rank_fusion(lists, k)` → fused ranking | I/O of any kind |
| `reranking/bge_reranker.py` | Scores `(query, title + chunk)` pairs, sigmoid → 0–1, sorts | Pick candidates |
| `core/lifespan.py` | Loads embedder + reranker, opens Chroma, builds BM25 from Chroma, creates rewriter | Per-request work |

### 5.4 Detailed Search Flow

```
POST /api/v1/support/search
 → middleware (request_id, start_time)
 → api_endpoints → Depends(get_support_search_service)
 → support_search_service.search(req)

   1. query_service.build_query(req)
        history = req.history[-12:]
        question_only → query = question
        full_history  → query = history + question joined
        rewrite       → no history → question ("skipped")
                        else cached_rewriter → Claude → standalone_query
                        error → question, rewrite_fallback = true
                        needs_retrieval = false → return empty result

   2. retrieval_service.retrieve(query, req.retrieval, req.rerank)
        vector  → embedder.embed_query → vector_store.query(top 20)
        keyword → bm25_index.search(top 20)
        hybrid  → both (in parallel threads) → reciprocal_rank_fusion([vector, keyword], k=60)
        rerank  → reranker.rerank(query, top 10 of the ranked list)
                  error → keep fused order, rerank_fallback = true

   3. group by article_id → best chunk per article → top_k
 → SupportSearchResult → ApiResponse

Any exception → core/exception_handlers.py → ErrorResponse
```

### 5.5 Article Change → Index Sync

```
create / update / delete (article_service)
   → Chroma write (source of truth)
   → chunks = vector_store.get_all_chunks()
   → bm25_index.rebuild(chunks)       # builds a new BM25Okapi, then swaps the reference
```

With ~40 articles (~150 chunks) a full rebuild takes milliseconds, so incremental BM25 updates aren't needed. A lock serialises concurrent rebuilds.

---

## 6. Data

### 6.1 Starter Knowledge Base (`data/kb/KB-012.md`)

Fictional store: **Northwind Gear**. Loaded with `scripts/seed_kb.py` through `ArticleService` (fixed IDs from front matter).

```markdown
---
id: KB-012
title: Returns for Outerwear
category: returns
---

## Return window
Outerwear, including the Trailblazer Pro (TB-PRO-22) and Summit Shell (SS-110),
can be returned within 60 days of delivery...
```

Cover 30–40 articles: returns, refunds, exchanges, shipping (by country), warranty, sizing, payment, order changes, account, and 6–8 product articles **with SKUs**. Deliberately include:
- **Near-duplicate topics** (returns for outerwear vs footwear vs sale items) → tests rewriting and reranking.
- **Near-duplicate codes** (`TB-PRO-22` vs `TB-PRO-23`, `SS-110` vs `SS-111`) on different articles → tests BM25.

### 6.2 Evaluation Set (`data/eval/conversations.json`)

30–35 cases, same shape as the search payload:

```json
{
  "id": "C-007",
  "pattern": "pronoun",
  "question": "How long do I have to return it?",
  "history": [
    "Do you sell the Trailblazer jacket in waterproof?",
    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof."
  ],
  "expected_article_ids": ["KB-012"]
}
```

| Pattern | Example question | Stage it tests |
|---|---|---|
| `pronoun` | "How long do I have to return it?" | rewrite |
| `ellipsis` | "What about Canada?" | rewrite |
| `correction` | "No, I meant the backpack, not the tent." | rewrite |
| `multi_reference` | "Which of those two has a longer warranty?" | rewrite |
| `exact_code` | "Is TB-PRO-23 machine washable?" | BM25 |
| `paraphrase` | "Can I send back something I bought cheap in a sale?" | vector, rerank |
| `already_standalone` | "What is your refund policy for sale items?" | all equal-ish |
| `small_talk` | "Thanks, that's all!" → `expected_article_ids: []` | rewrite |

3–5 cases per pattern.

### 6.3 Chunking and Tokenizing

- `##` headings → one chunk per section (long sections split by paragraph); no headings → paragraphs up to `CHUNK_MAX_CHARS` (1,200).
- **Stored** text = raw chunk (so `GET /articles/{id}` rebuilds content exactly).
- **Embedded** text = `"{title} — {heading}: {chunk}"`.
- **BM25** text = same as embedded text, tokenized: lowercase → regex `[a-z0-9]+(?:-[a-z0-9]+)*` (keeps `tb-pro-22` whole) → drop stop words.
- **Reranker** input = `(query, "{title} — {heading}: {chunk}")`.

---

## 7. API Contract

Base path: `/api/v1/support`. All responses use the shared envelope:
`{"success": true, "data": {...}, "meta": {"request_id": "...", "processing_time_ms": 0}}`

### 7.1 Articles

| Method | Path | Body | Success |
|---|---|---|---|
| `POST` | `/articles` | `ArticleCreate` | `201` → `ArticleResult` |
| `GET` | `/articles?category=returns` | — | `200` → `list[ArticleSummary]` |
| `GET` | `/articles/{article_id}` | — | `200` → `ArticleResult` |
| `PUT` | `/articles/{article_id}` | `ArticleUpdate` | `200` → `ArticleResult` |
| `DELETE` | `/articles/{article_id}` | — | `200` → `{ "id": "KB-...", "deleted": true }` |

**Create / update payload:**
```json
{
  "title": "Returns for Outerwear",
  "content": "## Return window\nOuterwear can be returned within 60 days...\n\n## Condition requirements\nTags must be attached...",
  "category": "returns"
}
```
Validation: `title` 3–200 chars; `content` 20–20,000 chars; `category` optional, lowercase letters/digits/hyphens, ≤ 50 chars.

**`ArticleResult`:**
```json
{
  "id": "KB-7f3a9c",
  "title": "Returns for Outerwear",
  "category": "returns",
  "content": "## Return window\n...",
  "chunk_count": 2,
  "created_at": "2026-11-17T10:42:00Z",
  "updated_at": "2026-11-17T10:42:00Z"
}
```

### 7.2 Search — `POST /search`

**Minimal payload:**
```json
{
  "question": "How long do I have to return it?",
  "history": [
    "Do you sell the Trailblazer jacket in waterproof?",
    "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof."
  ]
}
```

**All options (defaults shown):**
```json
{
  "question": "...",
  "history": [],
  "top_k": 3,
  "query_mode": "rewrite",
  "retrieval": "hybrid",
  "rerank": true,
  "include_debug": false
}
```

| Field | Values | Default |
|---|---|---|
| `question` | 1–2,000 chars, required | — |
| `history` | plain strings, oldest first, ≤ 100 (last 12 used) | `[]` |
| `top_k` | 1–10 | `3` |
| `query_mode` | `rewrite` · `question_only` · `full_history` | `rewrite` |
| `retrieval` | `hybrid` · `vector` · `keyword` | `hybrid` |
| `rerank` | `true` · `false` | `true` |
| `include_debug` | `true` · `false` | `false` |

**Response `data` — `SupportSearchResult` (with `include_debug: true`):**
```json
{
  "question": "How long do I have to return it?",
  "search_query": "Return window for the Trailblazer Pro TB-PRO-22 jacket",
  "query_mode": "rewrite",
  "retrieval": "hybrid",
  "rerank": true,
  "needs_retrieval": true,
  "rewrite_source": "llm",
  "rewrite_fallback": false,
  "rerank_fallback": false,
  "score_type": "rerank",
  "results": [
    {
      "article_id": "KB-012",
      "title": "Returns for Outerwear",
      "category": "returns",
      "score": 0.94,
      "snippet": "Outerwear, including the Trailblazer Pro (TB-PRO-22)... within 60 days...",
      "debug": {
        "vector_rank": 2,
        "keyword_rank": 1,
        "rrf_score": 0.0325,
        "rerank_score": 0.94
      }
    }
  ]
}
```
`score_type` tells you which stage produced `score`: `cosine` (vector only), `bm25` (keyword only), `rrf` (hybrid, no rerank), `rerank` (reranked).

### 7.3 Testing in Swagger

1. `POST /articles` a few times (or run `seed_kb.py`).
2. Same search body, change one switch at a time with `include_debug: true`:
   - `"retrieval": "vector", "rerank": false` → naive
   - `"retrieval": "keyword", "rerank": false` → BM25 only
   - `"retrieval": "hybrid", "rerank": false` → RRF
   - `"retrieval": "hybrid", "rerank": true` → full
3. Compare `search_query`, ranks, and which article came first.

### 7.4 Health — `GET /health`

```json
{
  "status": "ok",
  "environment": "dev",
  "embedding_model_loaded": true,
  "reranker_loaded": true,
  "vector_store": "ok",
  "keyword_index_chunks": 152,
  "rewrite_provider": "anthropic"
}
```

### 7.5 Errors

| Exception | HTTP | `code` | When |
|---|---|---|---|
| *(request validation)* | 422 | `VALIDATION_ERROR` | Bad body (same envelope) |
| `ArticleNotFoundError` | 404 | `ARTICLE_NOT_FOUND` | Unknown article ID |
| `KnowledgeBaseEmptyError` | 409 | `KB_EMPTY` | Search with no articles |
| `RerankerUnavailableError` | 409 | `RERANKER_DISABLED` | `rerank: true` while `RERANK_ENABLED=false` |
| `ModelNotReadyError` | 503 | `MODEL_NOT_READY` | Embedding model not loaded |
| `VectorStoreError` | 503 | `VECTOR_STORE_ERROR` | Chroma failure |
| *(anything else)* | 500 | `INTERNAL_ERROR` | Logged; generic message |

LLM and reranker runtime errors are caught and turned into fallbacks. LLM errors surface as `504` / `502` only when `REWRITE_STRICT=true` (live tests).

---

## 8. Environment

```dotenv
# .env.example
APP_ENV=dev
LOG_LEVEL=DEBUG

# Rewrite
ANTHROPIC_API_KEY=your-key-here
REWRITE_PROVIDER=anthropic          # anthropic | none
LLM_MODEL=claude-haiku-5-5
LLM_MAX_TOKENS=300
LLM_MAX_RETRIES=2
LLM_TIMEOUT_SECONDS=10
MAX_HISTORY_MESSAGES=12
REWRITE_STRICT=false
REWRITE_CACHE_PATH=./data/cache/rewrites.json
LLM_CALL_BUDGET=50

# Embeddings + vector store
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
CHROMA_PATH=./data/chroma
CHROMA_COLLECTION=support_kb
CHUNK_MAX_CHARS=1200

# Hybrid + rerank
CANDIDATES_PER_RETRIEVER=20
RRF_K=60
RERANK_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-base
RERANK_CANDIDATES=10

# General
HF_HUB_OFFLINE=0                    # 1 after both models are downloaded
KB_DIR=./data/kb
DEFAULT_TOP_K=3
```

`.env.dev` and `.env.uat` are gitignored. Every non-secret value has a default in code.

---

## 9. Quality Assurance

### 9.1 Test Layers

| Layer | Tool | API calls | Model download |
|---|---|---|---|
| Unit | pytest + fakes | 0 | No |
| Integration (API) | `TestClient` + `dependency_overrides` | 0 | No |
| Model smoke | `@pytest.mark.model` (run once) | 0 | Yes (both models) |
| Live rewrite | `@pytest.mark.live` (max 5) | ≤ 5 per run | No |
| Benchmark | `scripts/run_benchmark.py` | ≤ 35 first run, 0 after (cache) | Yes |

Fakes: `FakeRewriter`, `FakeEmbedder`, `FakeReranker` (e.g. reverses or scores by word overlap), real `BM25Index` on tiny fixtures (it's pure Python and fast), Chroma `EphemeralClient()`.

Budget guard: every real Anthropic attempt increments `data/cache/llm_call_budget.json`; at 50 the rewriter refuses with "LLM budget exhausted".

### 9.2 Minimum Test Cases

- **Tokenizer:** lowercases; `TB-PRO-22` → `tb-pro-22` (one token); punctuation stripped; stop words removed; same output for chunks and queries
- **BM25 index:** exact-code query ranks the matching chunk first; `TB-PRO-22` query does not rank the `TB-PRO-23` chunk first; rebuild after delete removes the chunk; empty index → empty results; concurrent search during rebuild never errors
- **RRF:** known two-list example gives expected order; item in both lists beats item in one; works with one empty list; `k` changes scores as expected
- **Reranker (fake + model test):** returns same items re-ordered; scores in 0–1; failure → fused order kept, `rerank_fallback=true`
- **Retrieval service:** `vector` never calls BM25; `keyword` never calls Chroma; `hybrid` calls both and fuses; `rerank=false` never calls the reranker; reranker receives at most `RERANK_CANDIDATES`; `score_type` correct for every combination; debug ranks filled when requested
- **Query service:** baselines never call the rewriter; empty history → skipped; 100 history items → last 12 sent; small talk → empty; rewriter error → fallback
- **Article service:** create returns ID + chunk count; get rebuilds exact content; update replaces chunks; delete removes chunks; **every change rebuilds BM25** (new article immediately findable by keyword)
- **API:** article CRUD status codes + envelope; 404 unknown ID; search returns envelope; invalid enum → 422; `rerank: true` with reranker disabled → 409; `X-Request-ID` present
- **Model smoke:** embedder 384 dims normalized; reranker scores a relevant pair higher than an irrelevant one
- **Live (max 5):** pronoun, ellipsis, correction, small talk, code preserved exactly in rewrite (`TB-PRO-22` unchanged)

### 9.3 Benchmark (main deliverable)

`scripts/run_benchmark.py` runs every eval case through each configuration and writes `docs/BENCHMARK.md`:

| # | Config | query_mode | retrieval | rerank |
|---|---|---|---|---|
| 1 | **Naive** | question_only | vector | no |
| 2 | Naive + history | full_history | vector | no |
| 3 | + Rewrite | rewrite | vector | no |
| 4 | + Hybrid | rewrite | hybrid | no |
| 5 | **Full pipeline** | rewrite | hybrid | yes |
| 6 | Keyword only (reference) | rewrite | keyword | no |

Metrics per config: **hit@1, hit@3, MRR, avg latency**, plus a **per-pattern breakdown** (pronoun, exact_code, paraphrase…). The story the tables should tell: rewrite fixes follow-ups, BM25 fixes exact codes, rerank fixes ordering.

- **hit@k:** share of cases with an expected article in the top k
- **MRR:** average of `1 / rank` of the first correct article (0 if missing)
- Small-talk cases scored separately: correct if `needs_retrieval=false`

Rewrites are cached, so configs 3–6 reuse the same ~30 Claude calls.

### 9.4 Quality Gates (every step)

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=src --cov-fail-under=80
```

### 9.5 Acceptance Criteria

- All article endpoints work; a new article is immediately findable by both vector and keyword search
- Full pipeline (config 5) beats naive (config 1) on hit@1, hit@3, and MRR
- `exact_code` pattern: hybrid (config 4) beats vector-only (config 3)
- Follow-up patterns: rewrite (config 3) beats naive (config 1)
- Rerank (config 5) improves hit@1 or MRR over config 4
- Full pipeline p95 latency < 3 s on CPU
- Total real Anthropic calls ≤ 50 for the whole build
- All quality gates green

---

## 10. Build Workflow (SDLC)

Branch: `feature/query-transformer`. Commit after each step: `step N: <name>`.

### Phase 1 — Design
**Step 1. README, CLAUDE.md, DECISIONS.md**
- Sections 1, 2, 4, 7 → `README.md`; structure rules → `CLAUDE.md`.
- ✅ Done when: all payloads and pipeline stages are clear without code.

### Phase 2 — Setup
**Step 2. Skeleton**
```powershell
uv init support-query-transformer
cd support-query-transformer
uv add "fastapi[standard]" anthropic instructor sentence-transformers chromadb rank-bm25 python-dotenv pyyaml
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy pre-commit
```
- Folders from 5.1, `.env.example`, `.gitignore`, `/health`.
- ✅ Done when: `/docs` opens.

**Step 3. Core (same as Weeks 1–2)**
- Envelope, exceptions, one global handler (also for `RequestValidationError`), middleware.
- ✅ Done when: a bad request returns `ErrorResponse` with `request_id`.

### Phase 3 — Indexing
**Step 4. Chunker, embedder, vector store**
- `article_chunker.py`, `bge_embedder.py`, `chroma_store.py` + tests.
- ✅ Done when: store tests pass (upsert, delete, get chunks, get all, list, query).

**Step 5. Keyword index**
- `tokenizer.py`, `bm25_index.py` (atomic rebuild + lock) + tests.
- ✅ Done when: exact-code tests pass, including the near-duplicate code case.

**Step 6. Article service + Articles API**
- `article_service.py` (Chroma write → BM25 rebuild), article routes, DI, lifespan (build BM25 from Chroma on startup).
- ✅ Done when: create → keyword-findable immediately → update → delete works in Swagger.

**Step 7. Starter data**
- 30–40 articles with SKUs and near-duplicates (6.1), `seed_kb.py`, 30–35 eval cases (6.2).
- ✅ Done when: seeding loads everything; every expected article ID exists.

### Phase 4 — Retrieval
**Step 8. Retrieval service: vector, keyword, hybrid (RRF)**
- `rank_fusion.py`, `retrieval_service.py`, search route with `query_mode=question_only` only.
- ✅ Done when: all three `retrieval` values work in Swagger with debug ranks.

**Step 9. Reranker**
- `bge_reranker.py`, `RERANK_ENABLED`, fallback, `score_type`.
- Run the `model` smoke tests once (downloads both models), then set `HF_HUB_OFFLINE=1`.
- ✅ Done when: `rerank: true` re-orders results; latency logged.

### Phase 5 — Query Transformation
**Step 10. Rewriter**
- `rewrite_schemas.py`, `rewrite_prompt.py`, `anthropic_rewriter.py` (history formatting, budget guard), `cached_rewriter.py`, `query_service.py`.
- Unit tests with fakes; then run the 5 live tests **once**.
- ✅ Done when: live cases pass; budget file shows ≤ 5 calls.

**Step 11. Full pipeline**
- `support_search_service.py` wires query → retrieval → grouping; all modes and switches.
- ✅ Done when: all service and API tests pass with fakes.

### Phase 6 — Benchmark
**Step 12. Benchmark + analysis**
- `run_benchmark.py` → `docs/BENCHMARK.md` (config table, per-pattern table, 5–8 lines of findings).
- Tune only if needed: prompt (bump `PROMPT_VERSION`), `CANDIDATES_PER_RETRIEVER`, `RERANK_CANDIDATES`, `RRF_K`.
- ✅ Done when: acceptance criteria 9.5 met; total LLM calls ≤ 50.

### Phase 7 — Delivery
**Step 13. Docker**
- Multi-stage, CPU-only torch, non-root; volumes for `data/chroma` and the Hugging Face cache (both models).
- ✅ Done when: `/health` shows both models loaded with `--env-file .env.uat`.

**Step 14. Finalize docs**
- README: setup, seed, article CRUD, search examples per switch, benchmark summary, cost and latency notes.
- ✅ Done when: a teammate can clone, seed, add an article, and run all search modes in 10 minutes.

---

## 11. Stretch Goals (after Step 14)

| Goal | Week 3 skill |
|---|---|
| `query_mode = multi_query`: 3 query variants, retrieve each, fuse all lists with RRF | Multi-Query expansion |
| `query_mode = step_back`: add a broader query alongside the specific one | Step-back prompting |
| `query_mode = hyde`: embed a hypothetical answer instead of the query | HyDE |
| Weighted RRF (e.g. BM25 weight 0.7 vs vector 1.0) | Fusion tuning |
| Context compression: return only the best sentences of each chunk | Context compression |

---

## 12. FAQ

**Does it use the whole conversation?**
Only to rewrite the question, and only the last 12 history items. All search stages use one short rewritten query.

**What if the chat has 100 messages?**
Accepted, but only the last 12 go to Claude. Fast (~1–2 s for the rewrite) and cheap.

**Why no `role` field in history?**
Simpler payload. Claude reads history as a numbered transcript and can tell questions from answers well enough.

**Why both vector and BM25 search?**
Vector search understands meaning ("send back" ≈ "return") but blurs exact codes (`TB-PRO-22` ≈ `TB-PRO-23`). BM25 nails exact words but misses synonyms. Together they cover each other's gaps.

**Why RRF instead of adding scores?**
Vector scores (0–1) and BM25 scores (0–20+) aren't comparable. RRF only uses rank positions, so no score calibration is needed.

**Why rerank only 10 candidates?**
A cross-encoder reads every (query, chunk) pair, which is slow. Ten candidates keeps it under ~0.5 s on CPU while still fixing the top of the list.

**Why is BM25 in memory, not in ChromaDB?**
`rank_bm25` is an in-memory library. ChromaDB is the source of truth; BM25 is rebuilt from it at startup and after every article change, which takes milliseconds at this size.

**Why is the vector DB also the article store?**
One storage system keeps the project simple. A production system would usually keep articles in PostgreSQL and treat ChromaDB and BM25 as search indexes.
