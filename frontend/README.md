# Support Search frontend

A small Angular 17 app for the Customer Support Query Transformer API: search with every pipeline switch, and upload text files as articles.

## Run (PowerShell)

Start the backend first, from the repository root:

```powershell
uv run python -m scripts.seed_kb                       # once: load the 40 starter articles
$env:APP_ENV = "dev"; uv run uvicorn src.main:app      # http://127.0.0.1:8000
```

Then, in a second terminal:

```powershell
cd frontend
npm install        # first time only
npm start          # http://localhost:4200
```

`npm start` runs `ng serve` with `proxy.conf.json`, which forwards `/api` and `/health` to `http://127.0.0.1:8000`. The backend therefore needs no CORS settings in development.

## What it does

- **Search tab.** Question, optional history (one message per line, oldest first), and the switches `query_mode`, `retrieval`, `rerank`, `top_k`, and debug ranks. Results show the rewritten query, where the rewrite came from, the score type, and each article's score and snippet. The "Try" buttons run ready-made examples.
- **Articles tab.** Upload a `.md`, `.markdown`, or `.txt` file, review the title, category, and content, then store it. Lists stored articles and deletes them.

Uploads are read in the browser and sent to `POST /api/v1/support/articles` as JSON. Only text is supported: PDF, Word, and Excel files must be converted to text first. A file may start with front matter (`title:` and `category:` between `---` lines, as in `data/kb/*.md`); otherwise the title comes from the first `# heading` or the file name. `sample-upload/tent-repair-service.md` is a ready-made file to try.

A search with history in `rewrite` mode asks Claude to rewrite the question. A conversation the backend has not seen before uses one call from its call budget; the app shows a note when that applies.

## Layout

```
src/app/api.service.ts     HTTP calls, response types, error formatting
src/app/article-file.ts    turns an uploaded text file into an article draft
src/app/app.component.*    the two screens
proxy.conf.json            dev proxy to the backend
```
