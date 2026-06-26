"""
RAG query pipeline.
Entry points:
  ask(question, doc_name)  -> streams answer to stdout, returns full text
  retrieve_chunks(question, doc_name, top_k) -> list of chunk strings
"""
import re
import time
import os
import ollama

from src.ingestion.pipeline import retrieve, get_embed_model, EMBED_MODEL as DEFAULT_EMBED_MODEL

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
LLM_MODEL            = os.getenv("EDGE_TUTOR_LLM_MODEL", "qwen2.5:0.5b")
INDEX_DIR            = "data/index"
TOP_K                = 3

SYSTEM_PROMPT = "Be concise."

# ------------------------------------------------------------------
# Retrieve
# ------------------------------------------------------------------
def retrieve_chunks(question: str, doc_name: str, top_k: int = TOP_K, embed_model: str = DEFAULT_EMBED_MODEL, verbose: bool = False):
    """Return (chunks, min_distance) for a question."""
    if verbose:
        t0 = time.perf_counter()
        print(f"\n[embed]     encoding question...", flush=True)
    results = retrieve(question, INDEX_DIR, doc_name, top_k=top_k, model_name=embed_model)
    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"[retrieve]  got {len(results)} chunks  ({elapsed:.3f}s)")
        for i, (chunk, dist) in enumerate(results, 1):
            preview = chunk[:80].replace("\n", " ")
            print(f"[retrieve]  chunk {i} (dist={dist:.3f}): {preview!r}".encode("ascii", "replace").decode())
    chunks = [chunk for chunk, _dist in results]
    min_dist = min(dist for _chunk, dist in results)
    return chunks, min_dist


_CONTINUATION = re.compile(
    r"^(continue|go on|keep going|more|next|and\??|ok|okay|yes|sure|please)\.?$",
    re.IGNORECASE,
)

def _is_followup(text: str) -> bool:
    """True for short inputs that continue the conversation rather than ask something new."""
    stripped = text.strip()
    return bool(_CONTINUATION.match(stripped)) or len(stripped.split()) <= 2


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------
def _build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Context passages from the document:\n\n"
        f"{context}\n\n"
        f"Answer using ONLY the passages above.\n"
        f"Question: {question}"
    )


# ------------------------------------------------------------------
# Generate (streaming)
# ------------------------------------------------------------------
def ask(
    question: str,
    doc_name: str,
    history: list[dict] | None = None,
    stream: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
    verbose: bool = False,
    llm_model: str = LLM_MODEL,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks and generate an answer via Ollama.
    Streams tokens to stdout if stream=True.

    history: list of {"role": ..., "content": ...} dicts from prior turns.
             Pass None or [] to start a fresh conversation.

    Returns (response_text, updated_history).
    """
    history = list(history or [])

    if _is_followup(question) and history:
        # Continuation: don't re-retrieve; just append the bare question
        history.append({"role": "user", "content": question})
    else:
        chunks, _min_dist = retrieve_chunks(question, doc_name, embed_model=embed_model, verbose=verbose)
        if verbose:
            print("[route]     GROUNDED", flush=True)
        prompt = _build_prompt(question, chunks)
        history.append({"role": "user", "content": prompt})

    if verbose:
        print(f"[llm]       sending to {llm_model}...", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response_text = ""
    stream_iter = ollama.chat(
        model=llm_model,
        messages=messages,
        stream=True,
        options={"temperature": 0.3, "num_predict": 450},
    )

    for chunk in stream_iter:
        token = (chunk.get("message", {}).get("content") if isinstance(chunk, dict) else chunk.message.content) or ""
        response_text += token
        if stream:
            print(token, end="", flush=True)

    if stream:
        print()  # newline after streamed output

    history.append({"role": "assistant", "content": response_text})
    return response_text, history
