"""
Phase 1 exit criterion: top-3 retrieval precision > 70% on a test set.
Run:  pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import chunk_text, clean_text


def test_clean_strips_lone_page_numbers():
    text = "Some content\n\n42\n\nMore content"
    result = clean_text(text)
    assert "42" not in result.split("\n")


def test_chunk_produces_overlap():
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text, chunk_tokens=200, overlap_tokens=50)
    assert len(chunks) >= 2
    tail = chunks[0][-50:]
    assert tail in chunks[1]


def test_chunk_count_reasonable():
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, chunk_tokens=400, overlap_tokens=50)
    assert 4 <= len(chunks) <= 12