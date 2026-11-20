# customer-support-query-transformer

A support assistant for a store's help center. You ask a question in a chat, it finds the right help articles and writes one answer from them.

It has two parts:

- **Backend** (`src/`): a FastAPI service. It rewrites follow-up questions using the conversation, searches the articles by meaning and by exact words, reranks the results, and has Claude write the answer.
- **Frontend** (`frontend/`): an Angular app with a chat screen and a screen for uploading articles.

## Run it

All commands are PowerShell, run from the repository root unless stated.

### 1. What you need

| Tool | Used for | Check |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Backend. It installs Python 3.12 and the dependencies for you | `uv --version` |
| Node.js 20 or newer, with npm | Frontend | `node --version` |
| An Anthropic API key | Rewriting follow-ups and writing answers | |

You also need about 2 GB of free disk space for the two search models and about 2 GB of free memory while the backend runs.

### 2. Start the backend

```powershell
uv sync                                     # install dependencies (first time only)
Copy-Item .env.example .env.dev             # first time only
```

Open `.env.dev` and set your key on the `ANTHROPIC_API_KEY=` line. Then:

```powershell
uv run python -m scripts.seed_kb            # load the 40 starter articles (first time only)
$env:APP_ENV = "dev"; uv run uvicorn src.main:app
```

The backend is ready when it prints `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal open.

The first start downloads two models from Hugging Face (about 1.2 GB), so it takes a few minutes. Later starts take 20 to 30 seconds. Once the models are downloaded you can add `HF_HUB_OFFLINE=1` to `.env.dev` so the backend never contacts Hugging Face again.

### 3. Start the frontend

In a **second terminal**:

```powershell
cd frontend
npm install                                 # first time only
npm start
```

The frontend is ready when it prints `Local: http://localhost:4200/`.

### 4. Use it

Open **http://localhost:4200**.

- **Chat tab:** type a question, for example `How long does shipping to Canada take?`. You get one answer, with the article it came from underneath. Follow-ups such as `What about Australia?` work because the earlier messages are sent along.
- **Articles tab:** upload a `.md` or `.txt` file to add an article, and see or delete the stored ones. `frontend/sample-upload/tent-repair-service.md` is a ready-made file to try.

[`docs/SAMPLE_QUESTIONS.md`](docs/SAMPLE_QUESTIONS.md) has questions and whole conversations to try, including unrelated ones.

Other addresses:

| Address | What |
|---|---|
| http://localhost:4200 | The app |
| http://127.0.0.1:8000/docs | Swagger page for calling the API directly |
| http://127.0.0.1:8000/health | Backend status: models loaded, number of indexed chunks |

### 5. Stop it

Press `Ctrl+C` in each terminal.

### If something goes wrong

| What you see | Cause and fix |
|---|---|
| The app shows "offline", or "Cannot reach the support service" | The backend is not running, or is still loading its models. Check the first terminal and http://127.0.0.1:8000/health |
| "There are no help articles yet" | The knowledge base is empty. Run `uv run python -m scripts.seed_kb`, or upload an article |
| Answers quote an article instead of being written out | Claude was not reachable: the key is missing or wrong in `.env.dev`, or the call budget is used up (see "Cost" below). Restart the backend after changing `.env.dev` |
| `Port 4200 is already in use` or port 8000 in use | An earlier copy is still running. Close its terminal, or stop the old `node` / `python` process |
| The backend stops or fails with a memory error | It needs about 2 GB of free memory. Close other large programs and start it again |
| Answers take several seconds | Expected on a laptop CPU: reranking takes 3 to 5 seconds. See "Cost and speed" |
| First start seems stuck | It is downloading the models (about 1.2 GB) |

### Settings

The backend reads `.env.dev` when `APP_ENV` is `dev`, and `.env.uat` when it is `uat`. Both files are ignored by git. Every setting except the API key has a default in code; `.env.example` lists them all. The ones you are most likely to change:

| Setting | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Your key. Without it the app still runs, but answers fall back to quoting the best article |
| `LLM_CALL_BUDGET` | `500` | Hard cap on Claude calls. See "Cost and speed" |
| `RERANK_ENABLED` | `true` | Set to `false` to skip loading the reranker: faster and lighter, slightly less accurate |
| `CHAT_MIN_RERANK_SCORE` | `0.05` | Relevance gate for chat. Higher refuses more questions; see `docs/BENCHMARK.md` before changing |
| `HF_HUB_OFFLINE` | `0` | Set to `1` after the models are downloaded |
| `CHROMA_PATH` | `./data/chroma` | Where articles are stored |

The frontend has no settings. In development, `npm start` forwards `/api` and `/health` to `http://127.0.0.1:8000` (see `frontend/proxy.conf.json`). If you serve a built copy of the frontend from a different address than the backend, the backend will need CORS enabled; that is not set up.

## How it works

A follow-up like "How long do I have to return it?" cannot be searched on its own, because the search does not know what "it" is. The backend handles a chat message in these steps:

```
POST /chat {question, history}
  1. VALIDATE   question 1-2,000 chars, history up to 100 plain strings
  2. REWRITE    no history → use the question as it is
                else → Claude turns it into a standalone query (last 12 messages)
                e.g. "Return window for Trailblazer jacket"
                "thanks" / "bye" → short reply, stop
  3. RETRIEVE   vector search: by meaning (bge-small-en-v1.5 + ChromaDB)
                BM25: by exact words, so TB-PRO-22 is not confused with TB-PRO-23
  4. FUSE       merge both lists by rank (Reciprocal Rank Fusion)
  5. RERANK     a cross-encoder (bge-reranker-base) scores each chunk against the query
  6. GATE       drop chunks scoring below 0.05; keep the top 5
  7. ANSWER     Claude gets the kept chunks as [C1]..[C5] with their article and section,
                and must answer only from them, citing a chunk and an exact quote per fact
  8. GROUND     code checks every citation: the chunk exists, the quote is in it, and
                every number in the answer is in a cited chunk; one correction attempt,
                then the answer is withheld
  9. GUARD      contact details, card-like numbers, and links that are not in the cited
                chunks are masked
 10. RESPOND    one answer with its citations, or a fixed "off-topic" / "not found" reply
```

Storing an article splits it into chunks by heading, embeds them into ChromaDB, and rebuilds the keyword index, so a new article is searchable immediately.

| Data | Where | Lifetime |
|---|---|---|
| Articles | ChromaDB on disk (`data/chroma`) | permanent |
| Keyword index | memory | rebuilt at startup and after every article change |
| Conversations | nowhere | sent with each request, then discarded |
| Rewrites and answers | `data/cache/` | caches, so repeated questions cost nothing |

## API

Base path `/api/v1/support`. Every response uses one envelope:

```json
{"success": true, "data": {}, "meta": {"request_id": "...", "processing_time_ms": 12.3}}
{"success": false, "error": {"code": "ARTICLE_NOT_FOUND", "message": "...", "details": null}, "meta": {}}
```

### Chat: `POST /chat`

```json
{"question": "No, I meant the backpack, not the tent.",
 "history": ["What's the warranty on your tents?", "Basecamp tents are covered for five years."]}
```

`question` is 1–2,000 characters. `history` is optional: earlier messages as plain strings, oldest first, up to 100 (only the last 12 are used).

Response `data`: `question`, `answer`, `kind`, `sources[]` of `{article_id, title}`, `citations[]` of `{article_id, title, heading, quote}`, `search_query`, `rewrite_fallback`, `answer_fallback`, `grounding_failed`, `guardrail_actions[]`.

| `kind` | When | `answer` |
|---|---|---|
| `answer` | The retrieved chunks contain the answer | Written by Claude from those chunks only; `citations` gives the section and exact quote behind each fact |
| `not_found` | A real support question the articles do not cover, or an answer that failed the grounding check twice (`grounding_failed: true`) | Fixed message: "I don't have that information", contact support |
| `off_topic` | Not about the store (general knowledge, nonsense, attempts to override instructions) | Fixed message: can only help with store support |
| `small_talk` | Greeting, thanks, goodbye | One short friendly line |

If Claude is unavailable, the request still succeeds: `answer_fallback` is `true` and the answer quotes the best-matching chunk.

Guardrails, in order:

- **Relevance gate.** Chunks with a rerank score below `CHAT_MIN_RERANK_SCORE` (0.05) are not shown to Claude. The value was calibrated, not assumed: see `docs/BENCHMARK.md`. When nothing passes, Claude is given no chunks and can only classify the message.
- **Grounding check.** Done in code after Claude answers. A cited chunk must be one that was provided, each quote must appear in its chunk (ignoring case, punctuation, and spacing, and allowing a small slip in a long quote), and each number in the answer must appear in a cited chunk or in the customer's own messages. A failure is sent back to Claude once with the reason; a second failure withholds the answer.
- **Output guardrail.** Email addresses, phone numbers, and links that are not in the cited chunks are masked, and card-like numbers always are. `guardrail_actions` lists what was changed. This is pattern matching, not a harmful-content classifier.
- **No temperature setting.** The model used rejects a non-default temperature, so consistency comes from the typed output, the grounding check, and the answer cache.

### Articles

| Method | Path | Body | Success |
|---|---|---|---|
| `POST` | `/articles` | `{title, content, category?}` | `201` → the stored article |
| `GET` | `/articles?category=returns` | | `200` → list of summaries |
| `GET` | `/articles/{article_id}` | | `200` → the article with full content |
| `PUT` | `/articles/{article_id}` | `{title, content, category?}` | `200` → the replaced article |
| `DELETE` | `/articles/{article_id}` | | `200` → `{"id": "KB-...", "deleted": true}` |

`title` is 3–200 characters, `content` 20–20,000 characters of text (Markdown is fine), `category` optional: lowercase letters, digits, hyphens, up to 50 characters. Only text is supported; PDF, Word, and Excel files must be converted to text first.

### Search: `POST /search`

The ranked list of articles without a written answer, with a switch for every stage. The benchmark uses it to compare stages.

```json
{
  "question": "How long do I have to return it?",
  "history": ["Do you sell the Trailblazer jacket in waterproof?",
              "Yes, the Trailblazer Pro (TB-PRO-22) is fully waterproof."],
  "top_k": 3,
  "query_mode": "rewrite",
  "retrieval": "hybrid",
  "rerank": true,
  "include_debug": false
}
```

| Field | Values | Default |
|---|---|---|
| `top_k` | 1–10 | `3` |
| `query_mode` | `rewrite` · `question_only` · `full_history` | `rewrite` |
| `retrieval` | `hybrid` · `vector` · `keyword` | `hybrid` |
| `rerank` | `true` · `false` | `true` |
| `include_debug` | `true` · `false` | `false` |

Response `data`: `search_query`, `needs_retrieval`, `rewrite_source`, `rewrite_fallback`, `rerank_fallback`, `score_type`, and `results[]` of `{article_id, title, category, score, snippet, debug?}`. `score_type` says which stage produced `score`: `cosine`, `bm25`, `rrf`, or `rerank`. `top_k`, `rerank`, and `include_debug` are strictly typed: `"3"` or `"false"` is rejected.

### Errors

| HTTP | `code` | When |
|---|---|---|
| 422 | `VALIDATION_ERROR` | Bad body or parameters |
| 404 | `ARTICLE_NOT_FOUND` | Unknown article ID |
| 409 | `KB_EMPTY` | No articles stored |
| 409 | `RERANKER_DISABLED` | `/search` with `rerank: true` while `RERANK_ENABLED=false` |
| 503 | `MODEL_NOT_READY` | Models not loaded yet |
| 503 | `VECTOR_STORE_ERROR` | ChromaDB failure |
| 502 / 504 | `LLM_ERROR` / `LLM_TIMEOUT` | Only when `REWRITE_STRICT=true` |
| 500 | `INTERNAL_ERROR` | Anything else |

A failed rewrite or a failed reranker never fails a request; the response flags it instead (`rewrite_fallback`, `rerank_fallback`).

## Cost and speed

- **Claude calls.** A chat message costs one call for the answer, plus one for the rewrite when there is earlier conversation. Repeated questions are served from the caches. The model is `claude-haiku-5-5`.
- **Hard cap.** Every call is counted in `data/cache/llm_call_budget.json`. At `LLM_CALL_BUDGET` (500 by default) further calls are refused, and the chat falls back to quoting the best article. Raise the setting or delete that file to reset the count.
- **Speed.** A chat answer takes 3 to 5 seconds on a laptop CPU. Most of that is the reranker (about 0.3 s per candidate). `RERANK_ENABLED=false` removes it.
- **Privacy.** Questions, history, and article text are never written to the logs, and conversations are not stored. The caches hold hashed keys and the generated text only.

## Benchmark

Three scripts, all reported in [`docs/BENCHMARK.md`](docs/BENCHMARK.md):

- `uv run python -m scripts.run_benchmark` runs 33 test conversations through six search configurations (the search stages, `/search`).
- `uv run python -m scripts.calibrate_threshold` measures rerank scores for correct answers against off-topic and uncovered questions, which is how the chat gate's 0.05 was chosen.
- `uv run python -m scripts.evaluate_chat` sends all 53 test messages through `/chat` on a running backend: 52 of 53 got the expected kind of reply, with no grounding failures.

Search results:

| Config | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Naive (question only, vector search) | 0.63 | 0.83 | 0.767 |
| + Rewrite | 0.83 | 1.00 | 0.917 |
| + Hybrid (adds BM25) | 0.90 | 0.97 | 0.938 |
| Full pipeline (+ rerank) | 0.80 | 0.93 | 0.874 |

The rewrite fixes follow-up questions and BM25 fixes exact product codes. Reranking lowers the top-1 result on this set and is the slow stage; it is kept because the chat's relevance gate needs its score. `docs/BENCHMARK.md` explains this and the limits of these numbers.

## Tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest                      # default suite: no Claude calls, no model downloads
uv run pytest -m model --no-cov    # uses the real models
uv run pytest -m live --no-cov     # makes real Claude calls (5 to 10)
```

The frontend has no automated tests. `cd frontend; npx ng build` checks that it compiles.

## Docker (backend only)

```powershell
docker build -t support-query-transformer .
docker run --rm -p 8000:8000 --env-file .env.uat `
  -v sqt-chroma:/app/data/chroma `
  -v sqt-hf:/home/app/.cache/huggingface `
  support-query-transformer
# load the starter articles into the container's volume (once)
docker run --rm --env-file .env.uat -v sqt-chroma:/app/data/chroma -v sqt-hf:/home/app/.cache/huggingface `
  support-query-transformer python -m scripts.seed_kb
```

Create `.env.uat` from `.env.example` first. The image is CPU-only and runs as a non-root user; articles and the model cache live in the two volumes. The frontend is not in the image: run it with `npm start` as above.

## Project layout

```
frontend/                        Angular app: chat + article upload, dev proxy to the API
src/
  main.py, api_router.py         app creation, central /api/v1 router
  core/                          startup, dependency providers, error handling, middleware
  domains/support_search/        routers/ schemas/ services/ prompts/ chunking/
  llm/ embeddings/ vectorstore/ keyword_search/ reranking/    one interface + one implementation each
  utils/                         rank fusion, hashing, env helpers
scripts/                         seed_kb.py, run_benchmark.py
data/kb/, data/eval/             starter articles and benchmark conversations
tests/                           unit/, integration/, model/, live/
docs/                            SAMPLE_QUESTIONS.md, BENCHMARK.md, DECISIONS.md
```

Coding conventions and how to extend the project are in [`CLAUDE.md`](CLAUDE.md). The original design is in [`support-query-transformer-workflow.md`](support-query-transformer-workflow.md), and where the build departs from it is listed in [`docs/DECISIONS.md`](docs/DECISIONS.md).
