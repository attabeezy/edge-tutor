"""
Phase 2 evaluation: retrieval precision + response grounding.

Measures whether the right content surfaces in the top-K retrieved chunks
for a hand-labeled question set, then optionally generates responses for
manual review.

Usage:
    # Retrieval precision only (fast, no Ollama needed):
    python tests/eval_rag.py

    # Retrieval + generate responses for manual review:
    python tests/eval_rag.py --generate

    # Use a different embedding model:
    python tests/eval_rag.py --embed bge

Exit criterion (Phase 2): retrieval precision (top-3) > 70%
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import retrieve

# ---------------------------------------------------------------------------
# Hand-labeled test set — CalculusMadeEasy.pdf
# Each entry: question the student might ask + keywords that MUST appear
# in at least one of the top-K retrieved chunks.
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        # Book explains derivative as "slope of a curve"
        "question": "What is a derivative?",
        "must_contain": ["slope"],
    },
    {
        # PDF renders x^n as "x?" but "differentiat" prefix reliably appears
        "question": "How do you differentiate x squared?",
        "must_contain": ["differentiat"],
    },
    {
        # dy/dx is described throughout as the slope of the curve
        "question": "What does dy/dx mean?",
        "must_contain": ["slope"],
    },
    {
        # Book explicitly covers constants vanishing under differentiation
        "question": "What is the derivative of a constant?",
        "must_contain": ["constant"],
    },
    {
        # Chapter on maxima/minima uses both words together
        "question": "How do you find the maximum or minimum of a function?",
        "must_contain": ["maximum", "minimum"],
    },
    {
        # "integral" / "integrating" appear in integration chapters
        "question": "What is integration?",
        "must_contain": ["integrat"],
    },
    {
        # Fundamental theorem section uses both prefixes
        "question": "What is the relationship between differentiation and integration?",
        "must_contain": ["integrat", "differentiat"],
    },
    {
        # Product rule section uses the word "product" explicitly
        "question": "How do you differentiate a product of two functions?",
        "must_contain": ["product"],
    },
    {
        # Book covers limits of functions
        "question": "What is a limit in calculus?",
        "must_contain": ["limit"],
    },
    {
        # Power rule chapter uses "power" directly
        "question": "What is the power rule for differentiation?",
        "must_contain": ["power"],
    },
    {
        # Trig differentiation chapter uses sin/cos
        "question": "How do you differentiate a trigonometric function?",
        "must_contain": ["sin"],
    },
    {
        # Second derivative chapter uses "second"
        "question": "What is the second derivative?",
        "must_contain": ["second"],
    },
]

EMBED_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge":    "TaylorAI/bge-micro-v2",
}

DOC_NAME  = "CalculusMadeEasy"
INDEX_DIR = "data/index"
TOP_K     = 3


def chunk_matches(chunks: list[str], keywords: list[str]) -> bool:
    """True if any chunk contains ALL keywords (case-insensitive)."""
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if all(kw.lower() in chunk_lower for kw in keywords):
            return True
    return False


def run_eval(embed_model: str, generate: bool, top_k: int = TOP_K):
    print(f"\n{'='*60}")
    print(f"EdgeTutor Phase 2 Evaluation")
    print(f"  doc:   {DOC_NAME}")
    print(f"  embed: {embed_model}")
    print(f"  top_k: {top_k}")
    print(f"{'='*60}\n")

    hits = 0
    misses = []

    for i, tc in enumerate(TEST_CASES, 1):
        results = retrieve(tc["question"], INDEX_DIR, DOC_NAME, top_k=top_k, model_name=embed_model)
        chunks  = [chunk for chunk, _ in results]
        passed  = chunk_matches(chunks, tc["must_contain"])

        status = "PASS" if passed else "FAIL"
        print(f"[{i:02d}] {status}  {tc['question']}")

        if passed:
            hits += 1
        else:
            misses.append(tc)
            print(f"       expected keywords: {tc['must_contain']}")
            for j, (chunk, dist) in enumerate(results, 1):
                preview = chunk[:120].replace("\n", " ").encode("ascii", errors="replace").decode("ascii")
                print(f"       chunk {j} (dist={dist:.3f}): {preview!r}")

        if generate:
            from src.rag.query import ask
            print(f"\n  --- Generated response ---")
            _, _ = ask(tc["question"], DOC_NAME, embed_model=embed_model)
            print()

    precision = hits / len(TEST_CASES)
    print(f"\n{'='*60}")
    print(f"Retrieval precision (top-{top_k}): {hits}/{len(TEST_CASES)} = {precision:.0%}")

    target = 0.70
    if precision >= target:
        print(f"PASS: exceeds {target:.0%} exit criterion")
    else:
        gap = round((target - precision) * len(TEST_CASES))
        print(f"FAIL: need {gap} more hit(s) to reach {target:.0%} exit criterion")
        if misses:
            print(f"\nFailing questions:")
            for tc in misses:
                print(f"  - {tc['question']}")
    print(f"{'='*60}\n")

    return precision


def main():
    parser = argparse.ArgumentParser(description="EdgeTutor Phase 2 RAG evaluation")
    parser.add_argument(
        "--embed", choices=["minilm", "bge"], default="minilm",
        help="Embedding model to evaluate (default: minilm)"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Also generate LLM responses for manual review (requires Ollama)"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})"
    )
    args = parser.parse_args()

    embed_model = EMBED_MODELS[args.embed]
    run_eval(embed_model, args.generate, top_k=args.top_k)


if __name__ == "__main__":
    main()
