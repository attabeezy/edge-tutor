"""
Evaluate the three-route query classifier: GROUNDED / GENERAL_REASONING / UNRELATED.

Each case declares which route it should exercise and how to verify correctness.
Cases marked ``gap: True`` are expected to fail with the current binary-gate
Python prototype; they are reported as GAP (not FAIL) so the script exits 0 and
documents where Phase 6 routing needs to close the gap.

Usage:
  python tests/eval_phase6_routes.py --doc CalculusMadeEasy
  python tests/eval_phase6_routes.py --doc CalculusMadeEasy --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.query import ask

# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------

ROUTE_CASES = [
    {
        "route": "GROUNDED",
        "question": "What is the method for finding a maximum or minimum of a function?",
        # At least one of these substrings must appear in the LLM's answer.
        "keywords": ["maximum", "minimum", "turning", "derivative"],
        "expect_refusal": False,
        # gap=False: current system should already produce a real answer for this.
        "gap": False,
    },
    {
        "route": "GENERAL_REASONING",
        "question": "How do you solve a quadratic equation?",
        # Keywords expected in the general-reasoning answer once Phase 6 lands.
        "keywords": ["quadratic", "equation", "factor", "formula", "square"],
        "expect_refusal": False,
        # gap=True: today the binary gate returns a hard refusal; Phase 6 will
        # route this to GENERAL_REASONING instead.
        "gap": True,
    },
    {
        "route": "UNRELATED",
        "question": "What is the capital of Ghana?",
        "keywords": [],
        "expect_refusal": True,
        "gap": False,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REFUSAL = "not covered in this document."


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_refusal(text: str) -> bool:
    return normalize(text) == _REFUSAL


def contains_any(text: str, keywords: list[str]) -> bool:
    haystack = normalize(text)
    return any(kw.lower() in haystack for kw in keywords)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(case: dict, doc_name: str, verbose: bool) -> str:
    """
    Run a single route case and return one of: "PASS", "FAIL", "GAP".
    """
    question = case["question"]
    expect_refusal = case["expect_refusal"]
    keywords = case["keywords"]
    is_gap = case.get("gap", False)

    # For GROUNDED, we need the full LLM answer to check keywords.
    # For UNRELATED / GENERAL_REASONING (pre-Phase-6), the gate fires before
    # the LLM, so ask() is cheap (returns immediately with the refusal text).
    answer, _ = ask(question, doc_name, stream=False, verbose=verbose)

    if verbose:
        print(f"  answer: {answer[:120]!r}")

    refusal = is_refusal(answer)

    if expect_refusal:
        passed = refusal
    else:
        if is_gap:
            # Before Phase 6: binary gate returns a refusal — that is the
            # documented gap, not a test failure.
            if refusal:
                return "GAP"
            # If it somehow answered (Phase 6 already landed), check keywords.
            passed = contains_any(answer, keywords)
        else:
            passed = (not refusal) and contains_any(answer, keywords)

    return "PASS" if passed else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate three-route query classification on EdgeTutor."
    )
    parser.add_argument("--doc", required=True, help="Ingested document name.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    total = len(ROUTE_CASES)
    passes = 0
    gaps = 0

    print(f"\nPhase 6 route eval  (doc={args.doc})\n")
    print(f"{'Status':<6}  {'Route':<20}  Question")
    print("-" * 72)

    for case in ROUTE_CASES:
        status = run_case(case, args.doc, args.verbose)
        route = case["route"]
        question_preview = case["question"][:55]
        print(f"{status:<6}  {route:<20}  {question_preview}")
        if status == "PASS":
            passes += 1
        elif status == "GAP":
            gaps += 1

    print()
    if gaps:
        print(
            f"Score: {passes}/{total}  ({gaps} gap{'s' if gaps > 1 else ''} — "
            f"expected pre-Phase-6; will become PASS once three-route logic lands)"
        )
    else:
        print(f"Score: {passes}/{total}")

    # Exit non-zero only on real FAILs (gaps are expected, not failures).
    real_failures = total - passes - gaps
    return 0 if real_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
