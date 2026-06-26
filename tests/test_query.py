"""
Tests for src/rag/query.py — pure-function coverage.
No Ollama or FAISS required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import query
from src.rag.query import _is_followup, _build_prompt


# ---------------------------------------------------------------------------
# _is_followup
# ---------------------------------------------------------------------------

def test_is_followup_continuation_words():
    for word in ["continue", "go on", "more", "ok", "okay", "yes", "sure", "please"]:
        assert _is_followup(word), f"Expected '{word}' to be a followup"


def test_is_followup_with_period():
    assert _is_followup("continue.")
    assert _is_followup("ok.")


def test_is_followup_short_input():
    assert _is_followup("next")
    assert _is_followup("and?")


def test_not_followup_real_question():
    assert not _is_followup("what is the definition of a limit")
    assert not _is_followup("explain differentiation to me")
    assert not _is_followup("how does integration work in practice")


def test_not_followup_three_words():
    # 3 words, not a continuation keyword — should NOT be a followup
    assert not _is_followup("what is calculus")


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_contains_passage():
    prompt = _build_prompt("what is calculus?", ["Calculus is the study of change."])
    assert "[Passage 1]" in prompt
    assert "Calculus is the study of change." in prompt


def test_build_prompt_instruction_precedes_question():
    prompt = _build_prompt("what is calculus?", ["Some passage."])
    assert prompt.index("Answer using ONLY") < prompt.index("Question:")


def test_build_prompt_question_is_last():
    prompt = _build_prompt("explain limits", ["A passage about limits."])
    assert prompt.strip().endswith("explain limits")


def test_build_prompt_multiple_passages_numbered():
    prompt = _build_prompt("test", ["chunk one", "chunk two", "chunk three"])
    assert "[Passage 1]" in prompt
    assert "[Passage 2]" in prompt
    assert "[Passage 3]" in prompt


def test_build_prompt_passages_separated():
    prompt = _build_prompt("q", ["first", "second"])
    # Passages must be separated by the divider
    assert "---" in prompt


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

def test_ask_always_uses_retrieved_chunks(monkeypatch):
    def fake_retrieve_chunks(question, doc_name, top_k=query.TOP_K, embed_model=query.DEFAULT_EMBED_MODEL, verbose=False):
        return ["A retrieved chunk with no lexical overlap."], 99.0

    def fake_chat(model, messages, stream, options):
        assert "A retrieved chunk with no lexical overlap." in messages[-1]["content"]
        assert "Question: totally unrelated prompt" in messages[-1]["content"]
        yield {"message": {"content": "answer"}}

    monkeypatch.setattr(query, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(query.ollama, "chat", fake_chat)

    response, history = query.ask("totally unrelated prompt", "Doc", stream=False)

    assert response == "answer"
    assert history[-1] == {"role": "assistant", "content": "answer"}
