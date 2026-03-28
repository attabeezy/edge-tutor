# EdgeTutor

An offline RAG (Retrieval-Augmented Generation) tutoring assistant for Android. Designed for first-year engineering students at KNUST to query their textbooks offline — no internet required.

## Features

- **100% Offline** — Runs entirely on-device with zero network dependency
- **PDF/TXT Ingestion** — Upload textbooks, get AI-powered answers
- **Lightweight Models** — Optimized for ~1 GB RAM Android devices
- **Gemma-3-270M** — Efficient quantized SLM for natural responses

## Project Structure

```
edge-tutor/
├── src/                    # Python MVP prototype
│   ├── ingestion/          # PDF parsing + chunking + embedding
│   └── rag/                # Query pipeline + REPL
├── tests/                  # Unit tests
├── scripts/                # Utility scripts (ONNX export)
├── data/                   # Raw PDFs, indices, models
├── edgetutor-android/      # Android app (Kotlin/Jetpack Compose)
└── EdgeTutor_MVP_Unified.md # Full specification
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

See `edgetutor-android/README.md` for building the Android app.

## Requirements

- Python 3.10+
- Android Studio (for Android app)
- Ollama (optional, for Python REPL testing)

## License

MIT License — see LICENSE file.