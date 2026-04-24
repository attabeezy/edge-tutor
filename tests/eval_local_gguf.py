"""
Evaluate local GGUF models directly with llama-cpp-python.

This benchmark is aimed at the downloaded Hugging Face GGUF files under ./models
and uses the same EdgeTutor-style workload:
  - answer covered questions from retrieved passages
  - refuse out-of-scope questions
  - handle a short follow-up coherently

Requirements:
  pip install llama-cpp-python

Usage:
  python tests/eval_local_gguf.py --doc MyBook
  python tests/eval_local_gguf.py --doc MyBook --report reports/local-gguf-benchmark.md
  python tests/eval_local_gguf.py --doc MyBook --models models\\granite-4.0-h-350m\\granite-4.0-h-350m-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import EMBED_MODEL as DEFAULT_EMBED_MODEL, retrieve

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "llama-cpp-python is required for local GGUF evaluation.\n"
        "Install it in the repo venv, then rerun:\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install llama-cpp-python"
    ) from exc


INDEX_DIR = "data/index"
TOP_K = 3
MAX_RELEVANT_DISTANCE = 1.4
MIN_LEXICAL_OVERLAP = 2
SYSTEM_PROMPT = "Be concise."

DEFAULT_MODEL_PATHS = [
    "models/granite-4.0-h-350m/granite-4.0-h-350m-Q4_K_M.gguf",
    "models/lfm2.5-350m/LFM2.5-350M-Q4_K_M.gguf",
    "models/lfm2-350m-math/LFM2-350M-Math-Q4_K_M.gguf",
]

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i you he she it we they "
    "what how why when where who which this that these those of in on at "
    "to for with by from about into than or and but if not no so".split()
)

_CONTINUATION = re.compile(
    r"^(continue|go on|keep going|more|next|and\??|ok|okay|yes|sure|please)\.?$",
    re.IGNORECASE,
)

COVERED_CASES = [
    {"question": "What is a derivative?", "keywords": ["slope", "derivative"]},
    {"question": "What is integration?", "keywords": ["integration", "integrat"]},
    {
        "question": "What is the relationship between differentiation and integration?",
        "keywords": ["differenti", "integrat"],
    },
    {"question": "What is the power rule for differentiation?", "keywords": ["power"]},
]

OUT_OF_SCOPE_CASES = [
    "Who was Isaac Newton's teacher?",
    "What is the capital of Ghana?",
    "Explain Fourier transforms.",
]

FOLLOWUP_CASES = [
    {"first": "What is a derivative?", "followup": "continue"},
    {"first": "How do you find the maximum or minimum of a function?", "followup": "give a short example"},
]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_any(text: str, keywords: list[str]) -> bool:
    haystack = normalize(text)
    return any(keyword.lower() in haystack for keyword in keywords)


def is_refusal(text: str) -> bool:
    return normalize(text) == "not covered in this document."


def has_lexical_overlap(question: str, chunks: list[str]) -> bool:
    q_tokens = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in _STOPWORDS}
    required = min(MIN_LEXICAL_OVERLAP, max(1, len(q_tokens)))
    for chunk in chunks:
        chunk_tokens = set(re.findall(r"[a-z]+", chunk.lower()))
        if len(q_tokens & chunk_tokens) >= required:
            return True
    return False


def is_followup(text: str) -> bool:
    stripped = text.strip()
    return bool(_CONTINUATION.match(stripped)) or len(stripped.split()) <= 2


def retrieve_chunks(question: str, doc_name: str, top_k: int = TOP_K, embed_model: str = DEFAULT_EMBED_MODEL) -> tuple[list[str], float]:
    results = retrieve(question, INDEX_DIR, doc_name, top_k=top_k, model_name=embed_model)
    chunks = [chunk for chunk, _dist in results]
    min_dist = min(dist for _chunk, dist in results)
    return chunks, min_dist


def build_context_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks))
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"Context passages from the document:\n\n{context}\n\n"
        f"Answer using ONLY the passages above.\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_followup_prompt(history: list[tuple[str, str]], question: str) -> str:
    turns = [f"System: {SYSTEM_PROMPT}", ""]
    for user_text, assistant_text in history:
        turns.append(f"User: {user_text}")
        turns.append(f"Assistant: {assistant_text}")
    turns.append(f"User: {question}")
    turns.append("Assistant:")
    return "\n".join(turns)


def generate(llm: Llama, prompt: str, max_tokens: int, temperature: float) -> str:
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        repeat_penalty=1.05,
        stop=["\nUser:", "\nSystem:"],
    )
    text = output["choices"][0]["text"]
    return text.strip()


def run_turn(
    llm: Llama,
    question: str,
    doc_name: str,
    history: list[tuple[str, str]] | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> tuple[str, list[tuple[str, str]], float]:
    started = time.perf_counter()
    history = list(history or [])

    if is_followup(question) and history:
        prompt = build_followup_prompt(history, question)
        answer = generate(llm, prompt, max_tokens=220, temperature=0.2)
        history.append((question, answer))
        return answer, history, time.perf_counter() - started

    chunks, min_dist = retrieve_chunks(question, doc_name, embed_model=embed_model)
    out_of_scope = min_dist > MAX_RELEVANT_DISTANCE or not has_lexical_overlap(question, chunks)
    if out_of_scope:
        answer = "Not covered in this document."
        history.append((question, answer))
        return answer, history, time.perf_counter() - started

    prompt = build_context_prompt(question, chunks)
    answer = generate(llm, prompt, max_tokens=220, temperature=0.2)
    history.append((question, answer))
    return answer, history, time.perf_counter() - started


def evaluate_model(model_path: Path, doc_name: str, embed_model: str, n_ctx: int, n_threads: int) -> dict:
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False,
    )

    covered_results = []
    oos_results = []
    followup_results = []
    latencies = []

    for case in COVERED_CASES:
        answer, _history, latency = run_turn(llm, case["question"], doc_name, embed_model=embed_model)
        latencies.append(latency)
        passed = (not is_refusal(answer)) and contains_any(answer, case["keywords"])
        covered_results.append({"question": case["question"], "passed": passed, "latency_s": latency, "answer": answer})

    for question in OUT_OF_SCOPE_CASES:
        answer, _history, latency = run_turn(llm, question, doc_name, embed_model=embed_model)
        latencies.append(latency)
        passed = is_refusal(answer)
        oos_results.append({"question": question, "passed": passed, "latency_s": latency, "answer": answer})

    for case in FOLLOWUP_CASES:
        first_answer, history, first_latency = run_turn(llm, case["first"], doc_name, embed_model=embed_model)
        followup_answer, _history, followup_latency = run_turn(llm, case["followup"], doc_name, history=history, embed_model=embed_model)
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
        "model": str(model_path),
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
        "# EdgeTutor Local GGUF Benchmark",
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
            f"| `{Path(result['model']).name}` | {result['score']}/{result['total']} | "
            f"{result['covered_passes']}/{result['covered_total']} | "
            f"{result['oos_passes']}/{result['oos_total']} | "
            f"{result['followup_passes']}/{result['followup_total']} | "
            f"{result['latency_avg_s']:.2f} | {result['latency_p95_s']:.2f} |"
        )

    for result in results:
        lines.extend(["", f"## {result['model']}", "", "### Covered questions"])
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
    parser = argparse.ArgumentParser(description="Benchmark local GGUF models on EdgeTutor RAG behavior.")
    parser.add_argument("--doc", required=True, help="Document name already ingested into data/index.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_PATHS,
        help="Local GGUF model paths to compare.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model used to retrieve context passages.",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window for llama.cpp.")
    parser.add_argument("--n-threads", type=int, default=4, help="CPU threads for llama.cpp.")
    parser.add_argument("--report", default=None, help="Optional markdown report path to write.")
    args = parser.parse_args()

    model_paths = [Path(model).resolve() for model in args.models]
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing model file: {path}")
        return 1

    if not Path(INDEX_DIR).exists():
        print(f"Missing retrieval index directory: {Path(INDEX_DIR).resolve()}")
        return 1

    results = []
    for model_path in model_paths:
        print(f"[benchmark] evaluating {model_path.name}...")
        try:
            result = evaluate_model(model_path, args.doc, args.embed_model, args.n_ctx, args.n_threads)
        except Exception as exc:
            print(f"[benchmark] {model_path.name} failed: {exc}")
            continue
        results.append(result)

    if not results:
        print("No model evaluations completed successfully.")
        return 1

    report = format_report(args.doc, args.embed_model, results)
    print(report)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"[benchmark] wrote report to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
