"""Compare forced RAG, retrieval-free, and automatic routing with Ollama."""

import argparse
import csv
import time
from pathlib import Path

from src.rag.query import ask


CASES = [
    ("g1", "grounded", "What is calculus?", None),
    ("g2", "grounded", "What is a differential?", None),
    ("g3", "grounded", "Explain integration in simple terms.", None),
    ("g4", "grounded", "Give a small worked example of differentiation.", None),
    ("f1", "follow_up", "Show me an example of that.", "What is differentiation?"),
    ("f2", "follow_up", "Can you explain it more simply?", "Explain the power rule."),
    ("f3", "follow_up", "How is it reversed?", "What is differentiation?"),
    ("f4", "follow_up", "Give another example.", "Give an example of integration."),
    ("ua1", "unsupported_academic", "What causes a solar eclipse?", None),
    ("ua2", "unsupported_academic", "Explain photosynthesis.", None),
    ("ua3", "unsupported_academic", "Who wrote Things Fall Apart?", None),
    ("ua4", "unsupported_academic", "What is the capital of Japan?", None),
    ("na1", "non_academic", "How do I bake bread?", None),
    ("na2", "non_academic", "Write a short birthday greeting.", None),
    ("na3", "non_academic", "What should I pack for a picnic?", None),
    ("na4", "non_academic", "Tell me a clean joke.", None),
]

FIELDS = ["mode", "case_id", "category", "question", "answer", "elapsed_s", "blank", "error"]


def write_reports(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "query-mode-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "query-mode-results.md").open("w", encoding="utf-8") as handle:
        handle.write("# Python Query-Mode Evaluation\n\n")
        for row in rows:
            handle.write(
                f"## {row['mode']} / {row['case_id']} / {row['category']}\n\n"
                f"Question: {row['question']}\n\n"
                f"Elapsed: {row['elapsed_s']}s; blank: {row['blank']}; "
                f"error: {row['error'] or 'none'}\n\n"
                f"{row['answer'] or '(no visible answer)'}\n\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--document", default="CalculusMadeEasy")
    parser.add_argument("--modes", nargs="+", choices=["rag", "general", "auto"], default=["general", "auto"])
    parser.add_argument("--output", type=Path, default=Path("reports/python/query-modes"))
    args = parser.parse_args()

    rows: list[dict] = []
    for mode in args.modes:
        for case_id, category, question, setup in CASES:
            started = time.perf_counter()
            answer = ""
            error = ""
            try:
                history = []
                if setup:
                    _, history = ask(
                        setup, args.document, history=history, stream=False,
                        llm_model=args.model, mode=mode,
                    )
                answer, _ = ask(
                    question, args.document, history=history, stream=False,
                    llm_model=args.model, mode=mode,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            row = {
                "mode": mode,
                "case_id": case_id,
                "category": category,
                "question": question,
                "answer": answer.strip(),
                "elapsed_s": f"{time.perf_counter() - started:.2f}",
                "blank": not bool(answer.strip()),
                "error": error,
            }
            rows.append(row)
            write_reports(args.output, rows)
            print(
                f"{mode:7} {case_id:3} {row['elapsed_s']:>7}s "
                f"blank={row['blank']} error={bool(error)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
