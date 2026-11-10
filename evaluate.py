"""Scores the configured model on data/eval.jsonl and compares with the last run.

GATEWAY_PROVIDER=groq GATEWAY_MODEL=openai/gpt-oss-20b python evaluate.py
python evaluate.py --fail-under 0.8     # exit 1 if accuracy is below 0.8 (for CI)

A case passes when every expected word appears in the answer (case-insensitive).
Results are saved to results/<provider>_<model>.json; the previous file is compared first.
"""

import argparse
import json
import os
import sys
import time

from gateway.core import Budget, Gateway
from gateway.providers import provider_from_config


def load_cases(path="data/eval.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def passes(answer, expected):
    return all(word.lower() in answer.lower() for word in expected)


def run(gw, cases):
    rows, cost = [], 0.0
    t0 = time.time()
    for case in cases:
        c, case_cost = gw.complete("evaluation", case["prompt"])
        cost += case_cost
        rows.append({"id": case["id"], "ok": passes(c.text, case["expected"]), "answer": c.text[:200]})
    return {"accuracy": round(sum(r["ok"] for r in rows) / len(rows), 3), "cases": len(rows),
            "cost_usd": round(cost, 6), "seconds": round(time.time() - t0, 1), "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fail-under", type=float, default=None)
    args = p.parse_args()
    provider = provider_from_config()
    gw = Gateway(provider, Budget(5.0))
    result = run(gw, load_cases())
    path = os.path.join("results", f"{provider.name}_{provider.model}.json".replace("/", "-"))
    if os.path.exists(path):
        previous = json.load(open(path))
        change = result["accuracy"] - previous["accuracy"]
        print(f"previous accuracy {previous['accuracy']:.3f} -> now {result['accuracy']:.3f} ({change:+.3f})")
        newly_failing = [r["id"] for r, old in zip(result["rows"], previous["rows"]) if old["ok"] and not r["ok"]]
        if newly_failing:
            print("now failing:", ", ".join(newly_failing))
    print(f"{provider.name}:{provider.model} accuracy {result['accuracy']:.3f} on {result['cases']} cases, "
          f"cost ${result['cost_usd']:.5f}, {result['seconds']}s")
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(path, "w"), indent=1)
    if args.fail_under is not None and result["accuracy"] < args.fail_under:
        sys.exit(1)


if __name__ == "__main__":
    main()
