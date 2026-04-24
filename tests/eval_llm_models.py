"""
Compare multiple Ollama chat models on the existing EdgeTutor RAG pipeline.

This is a pragmatic benchmark for this repo's actual workload:
  - grounded answers on in-document questions
  - refusal on out-of-scope questions
  - basic follow-up continuation behavior
  - latency per turn

Usage:
    python tests/eval_llm_models.py --doc CalculusMadeEasy
    python tests/eval_llm_models.py --doc CalculusMadeEasy --models lfm2.5:350m granite4:350m-h
    python tests/eval_llm_models.py --doc CalculusMadeEasy --report reports/model_benchmark.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.query import ask


COVERED_CASES = [
    {
        "question": "What is a derivative?",
        "keywords": ["slope", "derivative"],
    },
    {
        "question": "What is integration?",
        "keywords": ["integration", "integrat"],
    },
    {
        "question": "What is the relationship between differentiation and integration?",
        "keywords": ["differenti", "integrat"],
    },
    {
        "question": "What is the power rule for differentiation?",
        "keywords": ["power"],
    },
]

OUT_OF_SCOPE_CASES = [
    "Who was Isaac Newton's teacher?",
    "What is the capital of Ghana?",
    "Explain Fourier transforms.",
]

FOLLOWUP_CASES = [
    {
        "first": "What is a derivative?",
        "followup": "continue",
    },
    {
        "first": "How do you find the maximum or minimum of a function?",
        "followup": "give a short example",
    },
]

DEFAULT_MODELS = [
    "lfm2.5:350m",
    "granite4:350m-h",
    "lfm2-math",
]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_any(text: str, keywords: list[str]) -> bool:
    haystack = normalize(text)
    return any(keyword.lower() in haystack for keyword in keywords)


def is_refusal(text: str) -> bool:
    return normalize(text) == "not covered in this document."


def run_turn(question: str, doc_name: str, model: str, history: list[dict] | None = None) -> tuple[str, list[dict], float]:
    started = time.perf_counter()
    answer, new_history = ask(
        question,
        doc_name,
        history=history,
        stream=False,
        llm_model=model,
    )
    elapsed = time.perf_counter() - started
    return answer.strip(), new_history, elapsed


def evaluate_model(model: str, doc_name: str) -> dict:
    covered_results = []
    oos_results = []
    followup_results = []
    latencies = []

    for case in COVERED_CASES:
        answer, _history, latency = run_turn(case["question"], doc_name, model)
        latencies.append(latency)
        passed = (not is_refusal(answer)) and contains_any(answer, case["keywords"])
        covered_results.append(
            {
                "question": case["question"],
                "passed": passed,
                "latency_s": latency,
                "answer": answer,
            }
        )

    for question in OUT_OF_SCOPE_CASES:
        answer, _history, latency = run_turn(question, doc_name, model)
        latencies.append(latency)
        passed = is_refusal(answer)
        oos_results.append(
            {
                "question": question,
                "passed": passed,
                "latency_s": latency,
                "answer": answer,
            }
        )

    for case in FOLLOWUP_CASES:
        first_answer, history, first_latency = run_turn(case["first"], doc_name, model)
        followup_answer, _history, followup_latency = run_turn(case["followup"], doc_name, model, history=history)
        latencies.extend([first_latency, followup_latency])
        passed = (
            not is_refusal(first_answer)
            and not is_refusal(followup_answer)
            and len(followup_answer.split()) >= 8
            and normalize(followup_answer) != normalize(first_answer)
        )
        followup_results.append(
            {
                "question": case["first"],
                "followup": case["followup"],
                "passed": passed,
                "first_latency_s": first_latency,
                "followup_latency_s": followup_latency,
                "first_answer": first_answer,
                "followup_answer": followup_answer,
            }
        )

    covered_passes = sum(item["passed"] for item in covered_results)
    oos_passes = sum(item["passed"] for item in oos_results)
    followup_passes = sum(item["passed"] for item in followup_results)

    score = covered_passes + oos_passes + followup_passes
    total = len(covered_results) + len(oos_results) + len(followup_results)

    return {
        "model": model,
        "score": score,
        "total": total,
        "covered_passes": covered_passes,
        "covered_total": len(covered_results),
        "oos_passes": oos_passes,
        "oos_total": len(oos_results),
        "followup_passes": followup_passes,
        "followup_total": len(followup_results),
        "latency_avg_s": statistics.mean(latencies),
        "latency_p95_s": max(latencies),
        "covered_results": covered_results,
        "oos_results": oos_results,
        "followup_results": followup_results,
    }


def format_report(doc_name: str, embed_model: str, results: list[dict]) -> str:
    lines = [
        "# EdgeTutor Model Benchmark",
        "",
        f"- Document: `{doc_name}`",
        f"- Embedding model: `{embed_model}`",
        "",
        "## Summary",
        "",
        "| Model | Score | Covered | OOS refusal | Follow-up | Avg latency (s) | Max latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for result in sorted(results, key=lambda item: (-item["score"], item["latency_avg_s"])):
        lines.append(
            f"| `{result['model']}` | {result['score']}/{result['total']} | "
            f"{result['covered_passes']}/{result['covered_total']} | "
            f"{result['oos_passes']}/{result['oos_total']} | "
            f"{result['followup_passes']}/{result['followup_total']} | "
            f"{result['latency_avg_s']:.2f} | {result['latency_p95_s']:.2f} |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {result['model']}",
                "",
                "### Covered questions",
            ]
        )
        for item in result["covered_results"]:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"- `{status}` {item['question']} ({item['latency_s']:.2f}s)")
            lines.append(f"  Answer: {item['answer']}")

        lines.append("")
        lines.append("### Out-of-scope questions")
        for item in result["oos_results"]:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"- `{status}` {item['question']} ({item['latency_s']:.2f}s)")
            lines.append(f"  Answer: {item['answer']}")

        lines.append("")
        lines.append("### Follow-ups")
        for item in result["followup_results"]:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(
                f"- `{status}` {item['question']} -> {item['followup']} "
                f"({item['first_latency_s']:.2f}s + {item['followup_latency_s']:.2f}s)"
            )
            lines.append(f"  First: {item['first_answer']}")
            lines.append(f"  Follow-up: {item['followup_answer']}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multiple Ollama models on EdgeTutor RAG behavior.")
    parser.add_argument("--doc", required=True, help="Document name already ingested into data/index.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model names to compare.",
    )
    parser.add_argument(
        "--embed",
        default="all-MiniLM-L6-v2",
        help="Embedding model label for the report header.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional markdown report path to write.",
    )
    args = parser.parse_args()

    results = []
    for model in args.models:
        print(f"[benchmark] evaluating {model}...")
        try:
            result = evaluate_model(model, args.doc)
        except Exception as exc:
            print(f"[benchmark] {model} failed: {exc}")
            continue
        results.append(result)

    if not results:
        print("No model evaluations completed successfully.")
        return 1

    report = format_report(args.doc, args.embed, results)
    print(report)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"[benchmark] wrote report to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
