# EdgeTutor

An offline Android tutoring assistant that retrieves relevant textbook passages
and answers with Qwen3.5-0.8B through MNN-LLM. The application is designed for
low-memory Android devices and does not require internet access at runtime.

## Primary Application

`android-mnn/` is the sole product implementation.

It provides:

- PDF ingestion and local Arctic ONNX embeddings.
- Flat cosine-similarity retrieval.
- On-device Qwen3.5-0.8B MNN generation.
- Bottom navigation for Chat, a multi-textbook Library with independent RAG
  indexes, and Settings/tests.
- One-pass model routing between textbook-grounded and general answers.
- Streaming answers, source attribution, chat sessions, and performance logs.

For build, model-import, and device-validation instructions, see
[`android-mnn/README.md`](android-mnn/README.md).

## Answer Routing

Every question retrieves the best available textbook passage. The model must
begin its answer with one hidden routing marker:

- `[TEXTBOOK]` — answer is supported by the retrieved passage; sources appear.
- `[GENERAL]` — answer uses model knowledge; sources are hidden and the answer
  is visibly labelled as potentially inaccurate.

Missing or malformed markers fail closed to the labelled general route.
Retrieval similarity is retained for diagnostics but does not control routing.

## Tutor Fine-Tuning

The authored 300-row tutoring dataset, validation tools, upstream evaluator, and
free-Colab QLoRA/MNN workflow live in
[`training/tutor/`](training/tutor/README.md). The dataset preserves the
`[TEXTBOOK]`/`[GENERAL]` response contract and covers math, science,
English/language arts, and social studies for beginner learners.
## Python Support Tools

Python is limited to Android-supporting ONNX export and parity evaluation.

```powershell
pip install -r requirements.txt
python scripts/export_onnx.py
.venv\Scripts\python.exe -m pytest tests/test_arctic_onnx.py -q
.venv\Scripts\python.exe tests/eval_calculus_routing_onnx.py
```

The local routing benchmark is diagnostic. Python PDF extraction can differ
from Android PDFBox, so product claims require Android device logs.

## Repository Structure

```text
android-mnn/   Primary Android product
scripts/       Arctic ONNX export
src/rag/       Android-compatible Arctic ONNX parity implementation
tests/         ONNX parity tests and Calculus retrieval benchmark
reports/       Preserved Android device evidence and current parity results
models/        Local model files (ignored)
data/          Local source PDFs and generated indices (ignored)
```

## Verification

```powershell
cd android-mnn
.\gradlew.bat testDebugUnitTest

cd ..
pwsh -File scripts/check_repo_hygiene.ps1
```

## License

MIT License — see `LICENSE`.
