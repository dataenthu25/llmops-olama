"""
Eval runner — regression testing for prompt/model changes.

Runs a fixed set of test questions against the running FastAPI service and
checks each answer against simple pass/fail criteria (expected substrings
present, forbidden substrings absent). Exits with a non-zero status code if
any test fails, so this can be wired into a CI pipeline later (Phase 5).

Usage:
    python3 evals/run_evals.py

Requires the FastAPI server to already be running (e.g. via dev-reset.sh
or `uvicorn main:app --reload`).
"""

import json
import sys
from pathlib import Path

import httpx

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
API_URL = "http://localhost:8000/ask"
TIMEOUT_SECONDS = 60


def run_evals():
    test_cases = json.loads(TEST_CASES_PATH.read_text())
    results = []

    for case in test_cases:
        try:
            response = httpx.post(
                API_URL,
                json={"question": case["question"]},
                timeout=TIMEOUT_SECONDS,
            )
            answer = response.json().get("answer", "").lower()
        except Exception as e:
            results.append({
                "id": case["id"],
                "passed": False,
                "failures": [f"request failed: {e}"],
                "answer_preview": "",
            })
            continue

        passed = True
        failures = []

        for expected in case.get("expected_contains", []):
            if expected.lower() not in answer:
                passed = False
                failures.append(f"missing expected: '{expected}'")

        for forbidden in case.get("should_not_contain", []):
            if forbidden.lower() in answer:
                passed = False
                failures.append(f"contains forbidden: '{forbidden}'")

        results.append({
            "id": case["id"],
            "passed": passed,
            "failures": failures,
            "answer_preview": answer[:100],
        })

    return results


def main():
    results = run_evals()
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'=' * 50}")
    print(f"EVAL RESULTS: {passed_count}/{total} passed")
    print(f"{'=' * 50}\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"    {f}")
            print(f"    answer: {r['answer_preview']}...")

    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
