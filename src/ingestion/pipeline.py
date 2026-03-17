"""
Parse -> clean -> chunk -> embed -> FAISS index.
Entry point: ingest(pdf_path, index_dir)
"""
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------------
# Config (tune in Week 1-2 testing)
# ------------------------------------------------------------------
CHUNK_TOKENS            = 400   # target chunk size  (spec: 256-512)
OVERLAP_TOKENS          = 50    # overlap between chunks
APPROX_CHARS_PER_TOKEN  = 4     # rough heuristic for splitting

EMBED_MODEL = "all-MiniLM-L6-v2"   # 22M params, ~80 MB, 384-dim


# ------------------------------------------------------------------
# Step 1: Parse
# ------------------------------------------------------------------
def parse_pdf(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text) tuples."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


# ------------------------------------------------------------------
# Step 2: Clean
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Strip common PDF noise: headers, page numbers, excess whitespace."""
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ------------------------------------------------------------------
# Step 3: Chunk
# ------------------------------------------------------------------
def chunk_text(text: str, chunk_tokens: int = None, overlap_tokens: int = None):
    """
    Sliding window chunker that respects paragraph breaks.
    Returns list of chunk strings.
    """
    chunk_chars   = (chunk_tokens   or CHUNK_TOKENS)   * APPROX_CHARS_PER_TOKEN
    overlap_chars = (overlap_tokens or OVERLAP_TOKENS) * APPROX_CHARS_PER_TOKEN

    # Split on paragraph breaks first; fall back to sentence boundaries
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    # If a paragraph is longer than chunk_chars, break it into smaller pieces
    # so the sliding window below always has manageable units to work with
    units = []
    for para in paragraphs:
        if len(para) <= chunk_chars:
            units.append(para)
        else:
            # Try sentence splits first
            sentences = re.split(r"(?<=[.!?])\s+", para)
            bucket = ""
            for sent in sentences:
                if len(bucket) + len(sent) <= chunk_chars:
                    bucket += (" " if bucket else "") + sent
                else:
                    if bucket:
                        units.append(bucket)
                    # If a single sentence is still too long, hard-split it
                    while len(sent) > chunk_chars:
                        units.append(sent[:chunk_chars])
                        sent = sent[chunk_chars - overlap_chars:]
                    bucket = sent
            if bucket:
                units.append(bucket)

    # Sliding window over units with overlap
    chunks, current = [], ""
    for unit in units:
        if len(current) + len(unit) + 1 <= chunk_chars:
            current += (" " if current else "") + unit
        else:
            if current:
                chunks.append(current)
            overlap = current[-overlap_chars:] if current else ""
            current = (overlap + " " + unit).strip() if overlap else unit
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------
# Step 4: Embed
# ------------------------------------------------------------------
_model_cache = {}

def get_embed_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name} (first run downloads ~80 MB)")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_chunks(chunks: List[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    model = get_embed_model(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


# ------------------------------------------------------------------
# Step 5: FAISS index
# ------------------------------------------------------------------
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks: List[str], index_dir: str, doc_name: str):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(index_dir) / f"{doc_name}.faiss"))
    np.save(
        str(Path(index_dir) / f"{doc_name}_chunks.npy"),
        np.array(chunks, dtype=object),
    )


def load_index(index_dir: str, doc_name: str):
    index  = faiss.read_index(str(Path(index_dir) / f"{doc_name}.faiss"))
    chunks = np.load(
        str(Path(index_dir) / f"{doc_name}_chunks.npy"), allow_pickle=True
    ).tolist()
    return index, chunks


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def ingest(pdf_path: str, index_dir: str = "data/index") -> dict:
    """Full ingestion pipeline. Returns stats dict."""
    doc_name = Path(pdf_path).stem
    print(f"\n--- Ingesting: {pdf_path} ---")

    pages      = parse_pdf(pdf_path)
    full_text  = "\n\n".join(clean_text(text) for _, text in pages)
    chunks     = chunk_text(full_text)
    embeddings = embed_chunks(chunks)
    index      = build_index(embeddings)
    save_index(index, chunks, index_dir, doc_name)

    stats = {
        "doc":       doc_name,
        "pages":     len(pages),
        "chunks":    len(chunks),
        "embed_dim": embeddings.shape[1],
    }
    print(f"Done: {stats}")
    return stats


# ------------------------------------------------------------------
# Quick retrieval (used in Phase 1 validation)
# ------------------------------------------------------------------
def retrieve(query: str, index_dir: str, doc_name: str, top_k: int = 3):
    """Return top-k (chunk, distance) tuples for a query."""
    index, chunks = load_index(index_dir, doc_name)
    model = get_embed_model()
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, idxs = index.search(q_vec, top_k)
    return [(chunks[i], float(distances[0][j])) for j, i in enumerate(idxs[0])]