"""Book-specific routing benchmark using the Android Arctic ONNX pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.arctic_onnx import (  # noqa: E402
    ArcticOnnxEmbedder,
    Chunk,
    cosine_top_k,
    extract_android_style_chunks,
)

GENERAL_PATTERNS = (
    re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))[\s!.?]*$", re.I),
    re.compile(r"\b(write|make|give me)\b.*\b(greeting|wish|poem|joke)\b", re.I),
    re.compile(r"\b(tell me|share)\b.*\b(joke)\b", re.I),
    re.compile(r"\bsay hello\b", re.I),
)
TOKEN_RE = re.compile(r"[a-z]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "what", "how", "why", "who", "of", "in",
    "on", "at", "to", "for", "with", "and", "or", "do", "does", "can", "you",
    "me", "this", "that", "it",
}


def lexical_overlap(question: str, passage: str) -> float:
    query = {token for token in TOKEN_RE.findall(question.lower()) if token not in STOPWORDS}
    context = set(TOKEN_RE.findall(passage.lower()))
    return len(query & context) / len(query) if query else 0.0


def load_or_build_index(
    embedder: ArcticOnnxEmbedder,
    pdf: Path,
    cache: Path,
    batch_size: int,
    rebuild: bool,
) -> tuple[list[Chunk], np.ndarray]:
    if cache.exists() and not rebuild:
        data = np.load(cache, allow_pickle=True)
        chunks = [
            Chunk(int(i), int(start), int(end), str(text))
            for i, start, end, text in zip(
                data["indices"], data["start_pages"], data["end_pages"], data["texts"]
            )
        ]
        return chunks, data["vectors"].astype(np.float32)

    chunks = extract_android_style_chunks(pdf)
    vectors: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.append(embedder.embed_batch([chunk.text for chunk in batch]))
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks", flush=True)
    matrix = np.concatenate(vectors)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        indices=np.array([chunk.index for chunk in chunks]),
        start_pages=np.array([chunk.start_page for chunk in chunks]),
        end_pages=np.array([chunk.end_page for chunk in chunks]),
        texts=np.array([chunk.text for chunk in chunks], dtype=object),
        vectors=matrix,
    )
    return chunks, matrix


def choose_thresholds(rows: list[dict[str, object]]) -> tuple[float, float] | None:
    supported = sorted(float(row["mean_top5"]) for row in rows if row["expected"] == "GROUNDED")
    unsupported = sorted(float(row["mean_top5"]) for row in rows if row["expected"] == "UNSUPPORTED")
    # Protect grounded recall: permit at most one calibration positive below T_high.
    high_index = 1 if len(supported) >= 20 else 0
    high = supported[high_index]
    # Permit no calibration false grounding when defining the lower boundary.
    low = unsupported[-1]
    return (low, high) if low < high else None


def predicted_route(
    question: str,
    mean_top5: float,
    thresholds: tuple[float, float] | None,
) -> str:
    if any(pattern.search(question) for pattern in GENERAL_PATTERNS):
        return "GENERAL_TASK"
    if thresholds is None:
        return "UNRESOLVED"
    low, high = thresholds
    if mean_top5 >= high:
        return "GROUNDED"
    if mean_top5 <= low:
        return "UNSUPPORTED"
    return "CLARIFY"


def evaluate(rows: list[dict[str, object]], thresholds: tuple[float, float] | None) -> dict[str, float]:
    test = [row for row in rows if row["split"] == "test"]
    for row in test:
        row["predicted"] = predicted_route(str(row["question"]), float(row["mean_top5"]), thresholds)
    grounded = [row for row in test if row["expected"] == "GROUNDED"]
    negatives = [row for row in test if row["expected"] == "UNSUPPORTED"]
    general_predictions = [row for row in test if row["predicted"] == "GENERAL_TASK"]
    return {
        "grounded_recall": sum(row["predicted"] == "GROUNDED" for row in grounded) / len(grounded),
        "false_grounding_rate": sum(row["predicted"] == "GROUNDED" for row in negatives) / len(negatives),
        "general_precision": (
            sum(row["expected"] == "GENERAL_TASK" for row in general_predictions) / len(general_predictions)
            if general_predictions else 0.0
        ),
        "clarify_rate": sum(row["predicted"] == "CLARIFY" for row in test) / len(test),
    }


def scalar_diagnostics(rows: list[dict[str, object]]) -> dict[str, float]:
    test = [
        row for row in rows
        if row["split"] == "test" and row["expected"] in {"GROUNDED", "UNSUPPORTED"}
    ]
    grounded = [row for row in test if row["expected"] == "GROUNDED"]
    negatives = [row for row in test if row["expected"] == "UNSUPPORTED"]
    thresholds = sorted({float(row["mean_top5"]) for row in test})
    points = []
    for threshold in thresholds:
        recall = sum(float(row["mean_top5"]) >= threshold for row in grounded) / len(grounded)
        false_grounding = (
            sum(float(row["mean_top5"]) >= threshold for row in negatives) / len(negatives)
        )
        points.append((threshold, recall, false_grounding))
    protect_grounding = min(
        (point for point in points if point[1] >= 0.95),
        key=lambda point: point[2],
    )
    protect_sources = max(
        (point for point in points if point[2] <= 0.05),
        key=lambda point: point[1],
    )
    return {
        "grounding_threshold": protect_grounding[0],
        "grounding_recall": protect_grounding[1],
        "grounding_false_positive": protect_grounding[2],
        "sources_threshold": protect_sources[0],
        "sources_recall": protect_sources[1],
        "sources_false_positive": protect_sources[2],
    }


def write_report(
    output_dir: Path,
    rows: list[dict[str, object]],
    thresholds: tuple[float, float] | None,
    metrics: dict[str, float],
    chunk_count: int,
    index_seconds: float,
    diagnostics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cases.csv"
    fields = [
        "id", "split", "expected", "predicted", "topic", "question",
        "retrieval_question", "top1", "top2", "mean_top5", "margin",
        "lexical_overlap", "top_chunk", "top_pages", "query_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    calibration = [row for row in rows if row["split"] == "calibration"]
    report = [
        "# Calculus Made Easy — Local Arctic ONNX Routing",
        "",
        f"- Chunks: `{chunk_count}`",
        f"- Index build/load time: `{index_seconds:.2f}s`",
        f"- Threshold separation: `{'PASS' if thresholds else 'FAIL'}`",
    ]
    if thresholds:
        report.extend([
            f"- Unsupported threshold: `{thresholds[0]:.6f}`",
            f"- Grounded threshold: `{thresholds[1]:.6f}`",
        ])
    report.extend([
        "",
        "## Held-out metrics",
        "",
        f"- Grounded recall: `{metrics['grounded_recall']:.1%}`",
        f"- False grounding rate: `{metrics['false_grounding_rate']:.1%}`",
        f"- General-task precision: `{metrics['general_precision']:.1%}`",
        f"- Clarification rate: `{metrics['clarify_rate']:.1%}`",
        "",
        "When threshold separation fails, the metrics above use `UNRESOLVED` and "
        "are not release scores. The held-out scalar trade-off is:",
        "",
        f"- Preserve ≥95% grounded recall: threshold `{diagnostics['grounding_threshold']:.6f}`, "
        f"recall `{diagnostics['grounding_recall']:.1%}`, false grounding "
        f"`{diagnostics['grounding_false_positive']:.1%}`.",
        f"- Keep false grounding ≤5%: threshold `{diagnostics['sources_threshold']:.6f}`, "
        f"recall `{diagnostics['sources_recall']:.1%}`, false grounding "
        f"`{diagnostics['sources_false_positive']:.1%}`.",
        "",
        "## Calibration score ranges",
        "",
    ])
    for expected in ("GROUNDED", "UNSUPPORTED"):
        scores = [float(row["mean_top5"]) for row in calibration if row["expected"] == expected]
        report.append(
            f"- {expected}: `{min(scores):.6f}`–`{max(scores):.6f}` "
            f"(median `{statistics.median(scores):.6f}`)"
        )
    report.extend([
        "",
        "## Held-out cases",
        "",
        "| Case | Expected | Predicted | Mean top-5 | Top passage |",
        "|---|---|---|---:|---|",
    ])
    for row in (value for value in rows if value["split"] == "test"):
        excerpt = str(row["top_chunk"]).replace("|", "\\|").replace("\n", " ")[:110]
        report.append(
            f"| {row['id']} | {row['expected']} | {row['predicted']} | "
            f"{float(row['mean_top5']):.4f} | {excerpt} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/arctic.onnx"))
    parser.add_argument("--vocab", type=Path, default=Path("models/vocab.txt"))
    parser.add_argument("--pdf", type=Path, default=Path("data/raw/CalculusMadeEasy.pdf"))
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/calculus_routing_cases.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/index/CalculusMadeEasy_arctic_android_style.npz"))
    parser.add_argument("--output", type=Path, default=Path("reports/python/calculus-routing-onnx"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    embedder = ArcticOnnxEmbedder(args.model, args.vocab)
    started = time.perf_counter()
    chunks, passage_vectors = load_or_build_index(
        embedder, args.pdf, args.cache, args.batch_size, args.rebuild
    )
    index_seconds = time.perf_counter() - started
    full_text = "\n".join(chunk.text for chunk in chunks).lower()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for case in cases:
        absence_term = case.get("absence_term")
        if absence_term and absence_term.lower() in full_text:
            raise SystemExit(f"{case['id']} absence term is present in the book: {absence_term}")
        retrieval_question = " ".join(
            value for value in (case.get("prior_question", ""), case["question"]) if value
        )
        query_started = time.perf_counter()
        query = embedder.embed(retrieval_question, is_query=True)
        indices, scores = cosine_top_k(query, passage_vectors, 5)
        query_ms = (time.perf_counter() - query_started) * 1000
        top = chunks[int(indices[0])]
        rows.append({
            **case,
            "predicted": "",
            "retrieval_question": retrieval_question,
            "top1": float(scores[0]),
            "top2": float(scores[1]),
            "mean_top5": float(scores.mean()),
            "margin": float(scores[0] - scores[1]),
            "lexical_overlap": lexical_overlap(retrieval_question, top.text),
            "top_chunk": top.text,
            "top_pages": f"{top.start_page}-{top.end_page}",
            "query_ms": round(query_ms, 2),
        })

    thresholds = choose_thresholds([row for row in rows if row["split"] == "calibration"])
    metrics = evaluate(rows, thresholds)
    diagnostics = scalar_diagnostics(rows)
    write_report(
        args.output, rows, thresholds, metrics, len(chunks), index_seconds, diagnostics
    )
    confusion = Counter(
        (row["expected"], row["predicted"]) for row in rows if row["split"] == "test"
    )
    print(f"chunks={len(chunks)} threshold_separation={thresholds is not None}")
    print(f"thresholds={thresholds}")
    print(f"metrics={metrics}")
    print(f"scalar_diagnostics={diagnostics}")
    print(f"confusion={dict(confusion)}")
    print(f"report={args.output / 'report.md'}")


if __name__ == "__main__":
    main()
