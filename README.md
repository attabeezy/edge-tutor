# EdgeTutor

An offline RAG (Retrieval-Augmented Generation) tutoring assistant for Android. Designed for first-year engineering students at KNUST to query their textbooks offline — no internet required.

## Features

- **100% Offline** — Runs entirely on-device with zero network dependency
- **PDF/TXT Ingestion** — Upload textbooks, get AI-powered answers
- **Lightweight Models** — Optimized for ~1 GB free RAM Android devices
- **Qwen3-0.6B** — Quantized SLM for on-device inference via llama.cpp

## Project Structure

```
edge-tutor/
├── src/                    # Python MVP prototype
│   ├── ingestion/          # PDF parsing + chunking + embedding
│   └── rag/                # Query pipeline + REPL
├── scripts/                # Utility scripts (ONNX export)
├── tests/                  # Python unit tests + retrieval eval
├── models/                 # Shared model files (git-ignored — copy manually)
├── data/                   # Python runtime: raw PDFs and FAISS indices
├── android-ltk/            # Android app - Llamatik/llama.cpp backend (primary)
├── android-mlc/            # Android app - MLC-LLM/GPU backend (alternative)
└── PROJECT.md              # Specification, status, and roadmap
```

## Quick Start

### Python Prototype

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest a PDF
python -c "from src.ingestion.pipeline import ingest; ingest('data/raw/MyBook.pdf')"

# Run REPL
python -m src.rag.repl MyBook
```

### Android App

See `android-ltk/README.md` for build instructions (Llamatik/llama.cpp backend, primary).  
See `android-mlc/README.md` for GPU-accelerated variant (MLC-LLM, requires setup).

## Requirements

- Python 3.10+
- Android Studio (for Android app)
- Ollama (optional, for Python REPL testing)

## GitHub prep (Windows + Android build outputs)

If you hit `Filename too long` while staging or pushing, apply these once:

```bash
git config --global core.longpaths true
```

Then keep your clone path short (example: `C:\src\edge-tutor`) and avoid committing generated Android outputs.
This repository intentionally ignores generated/build artifacts for:

- `android-ltk/`
- `android-mlc/`

If build artifacts were already staged, unstage them before committing:

```bash
git restore --staged android-ltk/app/build android-mlc/app/build
```

Run the repo hygiene guardrail before pushing (or wire it into CI):

```bash
pwsh -File scripts/check_repo_hygiene.ps1
```

## License

MIT License — see LICENSE file.
