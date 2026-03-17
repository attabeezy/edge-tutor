"""
RAG query pipeline.
Entry points:
  ask(question, doc_name)  -> streams answer to stdout, returns full text
  retrieve_chunks(question, doc_name, top_k) -> list of chunk strings
"""
import re
import ollama

from src.ingestion.pipeline import retrieve, get_embed_model

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
LLM_MODEL   = "qwen2.5:0.5b"
INDEX_DIR   = "data/index"
TOP_K       = 3

SYSTEM_PROMPT = (
    "You are a helpful tutor for engineering students. "
    "You are given context passages from the student's own textbook. "
    "Use these passages to explain and guide the student — reason from them, "
    "quote relevant parts, and connect ideas. "
    "Do not make up facts not supported by the passages. "
    "If the passages are genuinely irrelevant, say so briefly, then offer a hint. "
    "Be concise (4-6 sentences max)."
)


# ------------------------------------------------------------------
# Retrieve
# ------------------------------------------------------------------
def retrieve_chunks(question: str, doc_name: str, top_k: int = TOP_K):
    """Return top-k chunk strings for a question."""
    results = retrieve(question, INDEX_DIR, doc_name, top_k=top_k)
    return [chunk for chunk, _dist in results]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
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
        f"Student question: {question}"
    )


# ------------------------------------------------------------------
# Generate (streaming)
# ------------------------------------------------------------------
def ask(
    question: str,
    doc_name: str,
    history: list[dict] | None = None,
    stream: bool = True,
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
        chunks = retrieve_chunks(question, doc_name)
        prompt = _build_prompt(question, chunks)
        history.append({"role": "user", "content": prompt})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response_text = ""
    stream_iter = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.3, "num_predict": 512},
    )

    for chunk in stream_iter:
        token = chunk.message.content or ""
        response_text += token
        if stream:
            print(token, end="", flush=True)

    if stream:
        print()  # newline after streamed output

    history.append({"role": "assistant", "content": response_text})
    return response_text, history
