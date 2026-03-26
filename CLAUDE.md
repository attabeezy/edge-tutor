# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EdgeTutor is an offline RAG (Retrieval-Augmented Generation) tutoring assistant targeting Android devices with ~1 GB RAM. The Python codebase is the MVP prototype; the eventual target is a Kotlin/Android port. Primary use case: first-year engineering students at KNUST querying their textbooks offline.

## Commands

### Setup (Windows, PowerShell 7+)
```bash
# Phase 1: ingestion pipeline only
.\edgetutor_setup.ps1 phase1

# Phase 2: + RAG with Ollama/Qwen2.5
.\edgetutor_setup.ps1 phase2

# Phase 3: + Android Studio scaffold
.\edgetutor_setup.ps1 phase3
```

### Install dependencies manually
```bash
pip install -r requirements-phase1.txt
```

### Run tests
```bash
pytest tests/ -v

# Single test
pytest tests/test_ingestion.py::test_chunk_produces_overlap -v
```

### Run the interactive REPL
```bash
python -m src.rag.repl CalculusMadeEasy                        # default embedding (all-MiniLM-L6-v2)
python -m src.rag.repl CalculusMadeEasy -e arctic              # use Snowflake/snowflake-arctic-embed-xs
python -m src.rag.repl CalculusMadeEasy -m granite4:350m       # override LLM model
python -m src.rag.repl CalculusMadeEasy -v                     # verbose/debug mode
```
REPL commands: type a question to get an answer, `new` to reset history, Ctrl-C to quit. Requires Ollama running with `qwen2.5:0.5b` loaded (or the model specified via `-m`).

### Ingest a new document
```python
from src.ingestion.pipeline import ingest
stats = ingest("data/raw/MyDocument.pdf")
```

## Architecture

### RAG Pipeline

```
INGESTION (one-time):
  PDF/TXT → parse_pdf() → clean_text() → chunk_text() → embed_chunks() → FAISS index

QUERY (per question):
  Question → embed → FAISS search → top-3 chunks → Ollama/Qwen2.5 → streamed answer
```

### Key Design Decision
Retrieval quality is the primary optimization target. The spec states: "A 270M model with excellent retrieval outperforms a 1B model with poor retrieval." All chunking/embedding parameters should be evaluated against retrieval precision, not generation quality.

### Models
| Component | Default | Alt |
|---|---|---|
| LLM | `qwen2.5:0.5b` via Ollama | Gemma 3 270M |
| Embedding (REPL) | `all-MiniLM-L6-v2` | `Snowflake/snowflake-arctic-embed-xs`, `TaylorAI/bge-micro-v2` |
| Vector store | FAISS (CPU) | — |

Target total RAM: ~0.43 GB (fits 1 GB Android devices).

### Module Responsibilities

- **`src/ingestion/pipeline.py`** — Complete ingestion pipeline. Key tunables: `CHUNK_TOKENS=400`, `OVERLAP_TOKENS=50`. Chunking is paragraph-aware with sentence-boundary fallback. Also contains `retrieve()` for querying an existing index.
- **`src/rag/query.py`** — RAG query pipeline. `ask()` handles followup detection, conversation history, and streaming from Ollama. `TOP_K=3`, temperature=0.3. Out-of-scope queries are rejected by a two-gate filter: L2 distance threshold (`MAX_RELEVANT_DISTANCE=1.4`) and lexical overlap (`MIN_LEXICAL_OVERLAP=2` content-word matches).
- **`src/rag/repl.py`** — Interactive test harness; not production code.
- **`tests/test_ingestion.py`** — Phase 1 exit-criteria tests (chunking and cleaning).

### Data Layout
```
data/
  raw/      ← source PDFs/TXTs
  index/    ← per-document FAISS index + numpy chunk arrays
             (filename pattern: <DocName>.faiss, <DocName>_chunks.npy)
```

## MVP Exit Criteria
- Retrieval precision (top-3) > 70% on test set
- Runs on 1 GB RAM without OOM
- First response latency < 30 seconds

## Planned Android Port
The Python prototype will be ported to Kotlin using: Llamatik (llama.cpp wrapper), ONNX Runtime Mobile (embeddings), FAISS JNI (vector search), PdfBox Android (PDF parsing), Jetpack Compose (UI). See `EdgeTutor_MVP_Unified.md` for the full 12-week roadmap and Android architecture.
