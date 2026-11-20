"""End-to-end check of POST /chat against a RUNNING backend (real models, real Claude).

Usage (PowerShell), either against a server you started:
    $env:APP_ENV = "dev"; uv run uvicorn src.main:app      # terminal 1
    uv run python -m scripts.evaluate_chat                  # terminal 2
or in one process, loading the models itself (no server needed):
    $env:APP_ENV = "dev"; uv run python -m scripts.evaluate_chat --in-process

Four sets are sent through the chat endpoint:
- data/eval/conversations.json  the main set (also used to tune prompts and the threshold)
- data/eval/holdout.json        cases written after the prompts were frozen; never tuned on
- data/eval/negatives.json      off-topic questions and support questions with no article
- data/eval/attacks.json        prompt-injection attempts

A reply is checked on four things where the case defines them: the kind of reply, a cited
source from an expected article, key facts the answer must contain (`expected_facts`: one
phrase from every group), and text it must never contain (`must_not_include`).

Every new message costs a Claude call; repeated runs are served from the caches.
Raw results: data/cache/chat_eval.json. Nothing from the conversations is printed.
"""

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = "http://127.0.0.1:8000"
EVAL_DIR = Path("data/eval")
BUDGET = Path("data/cache/llm_call_budget.json")
OUT = Path("data/cache/chat_eval.json")
BASELINE = Path("data/cache/chat_eval_baseline.json")


def budget() -> int:
    return int(json.loads(BUDGET.read_text())["calls"]) if BUDGET.exists() else 0


def load(name: str) -> list[dict[str, Any]]:
    path = EVAL_DIR / name
    return list(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else []


def wait_for_backend(client: httpx.Client) -> dict[str, Any]:
    for _ in range(60):
        try:
            return dict(client.get(f"{ROOT}/health").json()["data"])
        except httpx.HTTPError:
            time.sleep(3)
    raise SystemExit("backend not reachable on " + ROOT)


def contains_facts(answer: str, groups: list[list[str]]) -> bool:
    """True when the answer has at least one phrase from every group (case-insensitive)."""
    low = answer.lower()
    return all(any(phrase.lower() in low for phrase in group) for group in groups)


def judge(case: dict[str, Any], data: dict[str, Any], set_name: str) -> dict[str, bool]:
    """Each check that applies to this case → passed or not."""
    checks: dict[str, bool] = {}
    kind, answer = data["kind"], data["answer"]
    if set_name == "negatives":
        checks["kind"] = kind == ("off_topic" if case["type"] == "off_topic" else "not_found")
    elif set_name == "attacks":
        checks["kind"] = kind in case["allowed_kinds"]
    elif case.get("pattern") == "small_talk":
        checks["kind"] = kind == "small_talk"
    else:
        checks["kind"] = kind == "answer"
        cited = {s["article_id"] for s in data["sources"]}
        checks["source"] = bool(cited & set(case["expected_article_ids"]))
    if case.get("expected_facts") and (set_name != "attacks" or kind == "answer"):
        checks["facts"] = kind == "answer" and contains_facts(answer, case["expected_facts"])
    if case.get("must_not_include"):
        low = answer.lower()
        checks["clean"] = not any(text.lower() in low for text in case["must_not_include"])
    return checks


def main() -> None:
    if "--in-process" in sys.argv:
        from fastapi.testclient import TestClient

        from src.main import app

        # Same app and code path as the server, started and stopped inside this process.
        with TestClient(app, base_url=ROOT) as client:
            run(client)
    else:
        run(httpx.Client(timeout=180))


def run(client: httpx.Client) -> None:
    health = wait_for_backend(client)
    start_budget = budget()
    print(
        f"backend {health['status']}, {health['keyword_index_chunks']} chunks, "
        f"Claude calls so far: {start_budget}"
    )

    rows: list[dict[str, Any]] = []
    sets = ("conversations", "holdout", "negatives", "attacks")
    for set_name in sets:
        for case in load(f"{set_name}.json"):
            started = time.perf_counter()
            response = client.post(
                f"{ROOT}/api/v1/support/chat",
                json={"question": case["question"], "history": case.get("history", [])},
            )
            response.raise_for_status()
            data = response.json()["data"]
            checks = judge(case, data, set_name)
            rows.append(
                {
                    "id": case["id"],
                    "set": set_name,
                    "group": case.get("pattern") or case.get("type") or case.get("attack"),
                    "checks": checks,
                    "ok": all(checks.values()),
                    "ms": (time.perf_counter() - started) * 1000,
                }
                | data
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print(
        "\nset            n   all checks  kind   source  facts  clean  grounding_failed  fallback"
    )
    for set_name in sets:
        g = [r for r in rows if r["set"] == set_name]
        if not g:
            continue

        def passed(check: str, group: list[dict[str, Any]] = g) -> str:
            having = [r for r in group if check in r["checks"]]
            if not having:
                return "  -  "
            return f"{sum(r['checks'][check] for r in having)}/{len(having)}"

        failed = sum(r["grounding_failed"] for r in g)
        fallback = sum(r["answer_fallback"] for r in g)
        print(
            f"{set_name:<13} {len(g):>3}   {sum(r['ok'] for r in g):>3}/{len(g):<3}     "
            f"{passed('kind'):<6} {passed('source'):<7} {passed('facts'):<6} {passed('clean'):<6} "
            f"{failed:>8}  {fallback:>14}"
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
        failed = [name for name, value in r["checks"].items() if not value]
        print(
            f"  {r['id']} [{r['set']}/{r['group']}] got {r['kind']}; failed: {', '.join(failed)} "
            f"(grounding_failed={r['grounding_failed']})"
        )

    # Regression guard: answers that had the expected fact before must still have it.
    if BASELINE.exists():
        before = set(json.loads(BASELINE.read_text(encoding="utf-8"))["fact_correct"])
        now = {r["id"] for r in rows if r["checks"].get("facts")}
        measured = {r["id"] for r in rows if "facts" in r["checks"]}
        broken = sorted((before & measured) - now)
        gained = sorted(
            (now & measured) - before - {r["id"] for r in rows if r["set"] != "conversations"}
        )
        print(
            f"\ncompared with the saved baseline: {len(broken)} previously correct answers "
            f"now wrong {broken}, {len(gained)} newly correct {gained}"
        )


if __name__ == "__main__":
    main()
