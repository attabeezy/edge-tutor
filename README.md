# EdgeTutor

An offline RAG (Retrieval-Augmented Generation) tutoring assistant for Android. Designed for first-year engineering students at KNUST to query their textbooks offline — no internet required.

## Features

- **100% Offline** — Runs entirely on-device with zero network dependency
- **PDF/TXT Ingestion** — Upload textbooks, get AI-powered answers
- **Lightweight Models** — Optimized for ~1 GB free RAM Android devices
- **Qwen3.5 0.8B** — Quantized SLM for on-device inference via MNN-LLM

## Project Structure

```
edge-tutor/
├── android-mnn/              # Primary Android app using Qwen3.5 and MNN-LLM
├── android-ltk/              # Historical Llamatik/llama.cpp reference app
├── src/                      # Python MVP prototype
│   ├── ingestion/            # PDF parsing, chunking, and embedding pipeline
│   └── rag/                  # Query pipeline and REPL
├── tests/                    # Python unit tests and RAG/model eval scripts
├── scripts/                  # Setup, export, download, and repo hygiene scripts
├── notebooks/                # Analysis notebooks and experiments
├── data/                     # Runtime data: raw PDFs, processed text, indices
├── models/                   # Local model files (git-ignored; copy/download manually)
├── requirements.txt          # Python dependencies
├── android-ltk/README.md     # Android build and run instructions
└── PROJECT.md                # Specification, status, and roadmap
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

# Override the chat model for an A/B test
python -m src.rag.repl MyBook -m qwen2.5:0.5b
```

### Android App

See `android-mnn/README.md` for build and model-import instructions.

## Requirements

- Python 3.10+
- Android Studio (for Android app)
- Ollama (optional, for Python REPL testing)

## Model Benchmarking

Compare multiple Ollama chat models on the existing RAG pipeline:

```bash
python tests/eval_llm_models.py --doc CalculusMadeEasy
python tests/eval_llm_models.py --doc CalculusMadeEasy --report reports/model-benchmark.md
```

By default this compares:

- `qwen2.5:0.5b`
- `lfm2.5:350m`
- `granite4:350m-h`
- `lfm2-math`

To download the Granite/LFM comparison GGUF variants directly from Hugging Face into `models/`:

```bash
python scripts/download_hf_models.py
```

If you want the Python prototype to default to a different model without editing code, set:

```powershell
$env:EDGE_TUTOR_LLM_MODEL = "qwen2.5:0.5b"
```

To benchmark local GGUF files directly instead of Ollama, install `llama-cpp-python` and run:

```bash
python tests/eval_local_gguf.py --doc CalculusMadeEasy
```

## GitHub prep (Windows + Android build outputs)

If you hit `Filename too long` while staging or pushing, apply these once:

```bash
git config --global core.longpaths true
```

Then keep your clone path short (example: `C:\src\edge-tutor`) and avoid committing generated Android outputs.
This repository intentionally ignores generated/build artifacts for `android-ltk/`.

If build artifacts were already staged, unstage them before committing:

```bash
git restore --staged android-ltk/app/build
```

Run the repo hygiene guardrail before pushing (or wire it into CI):

```bash
pwsh -File scripts/check_repo_hygiene.ps1
```

## License

MIT License — see LICENSE file.
