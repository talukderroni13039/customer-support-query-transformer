"""End-to-end check of POST /chat against a RUNNING backend (real models, real Claude).

Usage (PowerShell):
    $env:APP_ENV = "dev"; uv run uvicorn src.main:app      # terminal 1
    uv run python -m scripts.evaluate_chat                  # terminal 2

Sends the eval conversations and the negatives through the chat endpoint and reports, per
group, how many replies had the expected kind (and, for answers, cited an expected article),
plus grounding failures, fallbacks, and guardrail actions. Every new message costs a Claude
call; repeated runs are served from the caches. Raw results: data/cache/chat_eval.json.
"""

import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = "http://127.0.0.1:8000"
BUDGET = Path("data/cache/llm_call_budget.json")
OUT = Path("data/cache/chat_eval.json")
CASES = Path("data/eval/conversations.json")
NEGATIVES = Path("data/eval/negatives.json")


def budget() -> int:
    return int(json.loads(BUDGET.read_text())["calls"]) if BUDGET.exists() else 0


def wait_for_backend(client: httpx.Client) -> dict[str, Any]:
    for _ in range(60):
        try:
            return dict(client.get(f"{ROOT}/health").json()["data"])
        except httpx.HTTPError:
            time.sleep(3)
    raise SystemExit("backend not reachable on " + ROOT)


def ask(client: httpx.Client, question: str, history: list[str]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = client.post(
        f"{ROOT}/api/v1/support/chat", json={"question": question, "history": history}
    )
    response.raise_for_status()
    return response.json()["data"], (time.perf_counter() - started) * 1000


def main() -> None:
    client = httpx.Client(timeout=180)
    health = wait_for_backend(client)
    start_budget = budget()
    print(
        f"backend {health['status']}, {health['keyword_index_chunks']} chunks, "
        f"Claude calls so far: {start_budget}"
    )

    rows: list[dict[str, Any]] = []
    for case in json.loads(CASES.read_text(encoding="utf-8")):
        data, ms = ask(client, case["question"], case["history"])
        if case["pattern"] == "small_talk":
            expected, ok = "small_talk", data["kind"] == "small_talk"
        else:
            cited = {s["article_id"] for s in data["sources"]}
            expected = "answer"
            ok = data["kind"] == "answer" and bool(cited & set(case["expected_article_ids"]))
        rows.append(
            {"id": case["id"], "group": case["pattern"], "expected": expected, "ok": ok, "ms": ms}
            | data
        )
    for neg in json.loads(NEGATIVES.read_text(encoding="utf-8")):
        data, ms = ask(client, neg["question"], [])
        expected = "off_topic" if neg["type"] == "off_topic" else "not_found"
        rows.append(
            {
                "id": neg["id"],
                "group": neg["type"],
                "expected": expected,
                "ok": data["kind"] == expected,
                "ms": ms,
            }
            | data
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\ngroup                n  correct  grounding_failed  fallback  guardrail")
    for group in dict.fromkeys(r["group"] for r in rows):
        g = [r for r in rows if r["group"] == group]
        correct = sum(r["ok"] for r in g)
        failed = sum(r["grounding_failed"] for r in g)
        fallback = sum(r["answer_fallback"] for r in g)
        guarded = sum(bool(r["guardrail_actions"]) for r in g)
        print(
            f"{group:<18} {len(g):>3}  {correct:>3}/{len(g):<3}  {failed:>10}  "
            f"{fallback:>12}  {guarded:>8}"
        )

    answered = [r for r in rows if r["kind"] == "answer"]
    latencies = [r["ms"] for r in rows]
    if answered:
        per_answer = statistics.fmean(len(r["citations"]) for r in answered)
        print(f"\nanswers written: {len(answered)}, citations per answer: {per_answer:.1f}")
    print(f"latency: median {statistics.median(latencies):.0f} ms, max {max(latencies):.0f} ms")
    print(f"Claude calls this run: {budget() - start_budget}; total {budget()}")

    misses = [r for r in rows if not r["ok"]]
    print(f"\nnot as expected: {len(misses)} of {len(rows)}")
    for r in misses:
        sources = [s["article_id"] for s in r["sources"]]
        print(
            f"  {r['id']} [{r['group']}] expected {r['expected']}, got {r['kind']} "
            f"(grounding_failed={r['grounding_failed']}, sources={sources})"
        )


if __name__ == "__main__":
    main()
