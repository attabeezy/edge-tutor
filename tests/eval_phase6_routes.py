"""
Evaluate that the Python RAG path no longer applies a document relevance gate.

Each case should reach the LLM with retrieved context instead of returning the
old hard refusal for weak document matches.

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
        "case": "IN_DOCUMENT",
        "question": "What is the method for finding a maximum or minimum of a function?",
        # At least one of these substrings must appear in the LLM's answer.
        "keywords": ["maximum", "minimum", "turning", "derivative"],
    },
    {
        "case": "ACADEMIC_OUT_OF_DOCUMENT",
        "question": "How do you solve a quadratic equation?",
        # The answer may be imperfect, but it should be generated rather than blocked.
        "keywords": ["quadratic", "equation", "factor", "formula", "square"],
    },
    {
        "case": "NON_ACADEMIC_OUT_OF_DOCUMENT",
        "question": "What is the capital of Ghana?",
        "keywords": [],
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
    Run a single case and return one of: "PASS", "FAIL".
    """
    question = case["question"]
    keywords = case["keywords"]

    answer, _ = ask(question, doc_name, stream=False, verbose=verbose)

    if verbose:
        print(f"  answer: {answer[:120]!r}")

    refusal = is_refusal(answer)

    passed = not refusal and (not keywords or contains_any(answer, keywords))

    return "PASS" if passed else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate that EdgeTutor no longer blocks weak document matches."
    )
    parser.add_argument("--doc", required=True, help="Ingested document name.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    total = len(ROUTE_CASES)
    passes = 0

    print(f"\nNo relevance gate eval  (doc={args.doc})\n")
    print(f"{'Status':<6}  {'Case':<28}  Question")
    print("-" * 72)

    for case in ROUTE_CASES:
        status = run_case(case, args.doc, args.verbose)
        case_name = case["case"]
        question_preview = case["question"][:55]
        print(f"{status:<6}  {case_name:<28}  {question_preview}")
        if status == "PASS":
            passes += 1

    print()
    print(f"Score: {passes}/{total}")

    real_failures = total - passes
    return 0 if real_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
