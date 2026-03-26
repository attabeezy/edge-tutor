"""
Tests for src/rag/query.py — pure-function coverage.
No Ollama or FAISS required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.query import _has_lexical_overlap, _is_followup, _build_prompt

# ---------------------------------------------------------------------------
# _has_lexical_overlap
# ---------------------------------------------------------------------------

def test_lexical_overlap_match():
    chunks = ["The derivative measures the rate of change of a function."]
    assert _has_lexical_overlap("what is a derivative", chunks)


def test_lexical_overlap_no_match():
    chunks = ["The derivative measures the rate of change of a function."]
    assert not _has_lexical_overlap("stochastic differential equations", chunks)


def test_lexical_overlap_stopwords_only_do_not_count():
    # All stopwords — should not satisfy MIN_LEXICAL_OVERLAP=2
    chunks = ["The sky is blue."]
    assert not _has_lexical_overlap("what is the", chunks)


def test_lexical_overlap_matches_second_chunk():
    chunks = [
        "Integration is the reverse of differentiation.",
        "Limits describe the behaviour of functions near a point.",
    ]
    # "limits" is a content word that appears in the second chunk (1 match, required=1 since
    # "how" and "do" are stopwords leaving only "limits" as a content word after removal)
    assert _has_lexical_overlap("what are limits", chunks)


def test_lexical_overlap_single_content_word_fails_threshold():
    # "calculus" matches but only 1 content word — below MIN_LEXICAL_OVERLAP=2
    chunks = ["calculus is hard"]
    assert not _has_lexical_overlap("calculus stochastic equations", chunks)


def test_lexical_overlap_two_content_words_passes():
    chunks = ["calculus studies limits and derivatives"]
    assert _has_lexical_overlap("calculus limits", chunks)


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
