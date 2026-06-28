# EdgeTutor Project Status

**Last reviewed:** June 27, 2026
**Repository state reviewed:** `master` at `8fad200` (`Validate Android MNN harness`)
**Primary product:** Offline Android RAG tutor for textbook question answering
**Current Android app:** `android-mnn/`
**Current default chat model:** Qwen3.5-0.8B-MNN

This document is the current project ground truth. It should describe what the
code actually does, what has been validated, and what remains pending. Do not
record planned behavior as current behavior.

---

## 1. Product Goal

EdgeTutor is an offline Retrieval-Augmented Generation tutoring assistant for
Android. The target user is a student who loads a textbook PDF and asks
questions without relying on internet access.

Current target hardware remains low-end Android devices, with Samsung
SM-A047F-class devices used as the main performance reference.

Product direction:

- EdgeTutor should be a focused offline tutor, not a generic document chat app.
- The first curated content scope is Ghana Basic Education, aligned broadly with
  Ghana Education Service / NaCCA basic education subject areas.
- MVP subject areas are:
  - Mathematics
  - Integrated Science
  - English Language
  - Social Studies
  - Computing
- Built-in subject corpora should become the primary first-run experience.
- User-uploaded PDFs should remain supported as a way to personalize or extend a
  learning area, but they are not the core product identity.
- Future subject expansion candidates include Career Technology, Creative Arts
  and Design, Religious and Moral Education, Ghanaian Language, French, and
  Arabic.

Key product constraints:

- Runs fully offline after model/document assets are present.
- Accepts local PDF documents.
- Extracts, chunks, embeds, indexes, and searches textbook text on device.
- Answers with Qwen3.5 through the on-device MNN-LLM runtime.
- Optimizes for low memory and low time to first visible token.

---

## 2. Current Architecture

### Android Runtime

The Android implementation in `android-mnn/` is the active product surface.

Current stack:

- Kotlin and Jetpack Compose for the app UI.
- Room for document metadata.
- PDFBox Android for PDF text extraction.
- ONNX Runtime Mobile for the embedding model.
- Snowflake Arctic Embed XS exported as `arctic.onnx` for embeddings.
- Custom Kotlin `FlatIndex` for persisted vector storage and in-memory cosine search.
- MNN-LLM through the app JNI bridge for generation.
- Qwen3.5 0.8B MNN as the selected chat model.

Important Android paths:

- `android-mnn/app/src/main/java/com/edgetutor/mnn/MainActivity.kt`
- `android-mnn/app/src/main/java/com/edgetutor/mnn/viewmodel/ChatViewModel.kt`
- `android-mnn/app/src/main/java/com/edgetutor/mnn/llm/MnnEngine.kt`

### Python Prototype

The Python code under `src/` remains useful for prototype testing and model
screening, but it is not the production app.

Current Python stack:

- `src/ingestion/pipeline.py`: PDF parsing, cleaning, chunking, embeddings,
  FAISS indexing, and retrieval.
- `src/rag/query.py`: Ollama-backed RAG answering with retrieved document
  context and basic follow-up handling.
- Default Python model: `qwen2.5:0.5b`, overridable with
  `EDGE_TUTOR_LLM_MODEL`.

The Python and Android retrieval/routing paths are similar but not identical.
Android is the source of truth for shipped behavior.

### Android MNN Runtime

`android-mnn/` is the selected runtime and remains the engineering surface for
prompt, latency, memory, and answer-quality validation.

Current harness direction:

- Keep the user-facing product shape from `android-ltk`: single-document offline
  RAG tutoring, local PDF ingestion, grounded chat, source excerpts, and low-end
  Android performance as the primary constraint.
- Use `MNN/apps/Android/MnnLlmChat` as the reference implementation for native
  model loading and MNN-LLM session behavior.
- Keep model files in app-owned storage instead of requesting broad shared
  storage access.
- Treat model readiness as an explicit app state. Chat is locked until required
  MNN files are present and validated.
- Preserve the same retrieval and ingestion architecture where possible:
  PDF extraction, chunking, ONNX embeddings, `FlatIndex`, Room metadata, and
  Compose UI.
- Instrument the full query path so the harness can answer whether bottlenecks
  are ingestion, retrieval, prompt construction, native prefill, decode, memory,
  or UI flush behavior.

Current MNN harness implementation notes:

- `MnnModelManager` validates and imports model folders into
  `files/mnn_model`.
- Required Qwen MNN files are currently:
  `config.json`, `llm_config.json`, `tokenizer.txt`, `llm.mnn`,
  `llm.mnn.json`, `llm.mnn.weight`, `visual.mnn`, and
  `visual.mnn.weight`.
- The visual graph files are required because the tested Qwen package reports
  `is_visual=1` and `has_deepstack=1`; without them native load fails even for
  text-only RAG.
- Native streaming now buffers UTF-8 boundaries before sending chunks to Kotlin.
- Native decode uses stepped generation so first-token and total-decode timing
  can be measured from the app.
- Qwen thinking is forced off and verified from the effective native config
  before generation. `ThinkingTagFilter` remains as defense in depth, and
  hidden-only output is treated as an explicit generation failure.
- The app no longer requests all-files/shared-storage permissions for model
  setup.

The Qwen3.5/MNN path is the selected product runtime. `android-ltk/` remains a
historical/reference implementation and is no longer a release gate.

---

## 3. Required Local Assets

The Android app requires embedding assets under
`android-mnn/app/src/main/assets/` and an imported MNN model directory under
app-owned storage.

Current expected files:

| File | Purpose | Git status |
|---|---|---|
| Qwen3.5 MNN model directory | Chat model | Ignored/imported |
| `arctic.onnx` | Embedding model | Ignored |
| `vocab.txt` | WordPiece tokenizer vocab | Ignored |

These files are intentionally not tracked in Git. The checked `.gitignore`
ignores Android model assets and root `models/`.

As of this review, the required files are present locally in the workspace.
A fresh clone still needs the assets copied or generated before Android builds
can run end to end.

---

## 4. Implemented Android Behavior

### Ingestion

Implemented in `IngestViewModel.kt`, `PdfExtractor.kt`, `TextChunker.kt`,
`Embedder.kt`, and `FlatIndex.kt`.

Current behavior:

- Only one document is kept at a time; starting a new ingestion deletes prior
  document metadata and index files.
- PDF pages are extracted in windows.
- Chunking is performed on-device.
- Embeddings are generated with ONNX Runtime Mobile.
- Embedding batch size and PDF page window shrink under low-memory conditions.
- Index entries are appended to disk incrementally through `FlatIndex.startAppend`,
  `append`, and `finishAppend`.
- Ingestion progress is exposed through `IngestionProgress`.
- Likely scanned PDFs are flagged and exposed to the UI.

Current ingestion constants:

- Default embedding batch: `32`
- Low-memory embedding batch: `8`
- Default page window: `20`
- Low-memory page window: `5`
- Ingestion low-memory threshold: `50 MB`

### Query Flow

Implemented primarily in `ChatViewModel.kt`.

Current query path:

1. Load the selected document's `.idx` file into `FlatIndex`.
2. Warm the embedder and LLM when a document is loaded.
3. Embed the user query.
4. Search the flat vector index for candidate chunks.
5. Select a small context budget.
6. Use the selected retrieved context directly.
7. Build a short prompt.
8. Stream generation through `LlamaEngine`.
9. Flush the first visible token immediately, then buffer later UI updates.

Current retrieval and prompt-budget constants:

- Candidate retrieval count: `5`
- Maximum kept context chunks: `1`
- Maximum context chars per kept chunk: `800`
- Follow-up context chars: `180`
- Previous answer context chars: `250`

### Query Handling

Android no longer applies a document relevance gate before generation.

Current behavior:

- Retrieve candidate chunks from the loaded document.
- Keep the highest-scoring chunks within the prompt budget.
- Always build the grounded prompt from the kept chunks.
- Always call the LLM for non-follow-up user questions.
- Attach source excerpts from the kept chunks.

### LLM Runtime

Implemented in `MnnEngine.kt` and `edgetutor_jni.cpp`.

Current behavior:

- Model: Qwen3.5-0.8B-MNN with thinking disabled.
- The model is imported into app-owned storage and validated before chat unlocks.
- Native model initialization is started early from `ChatViewModel.init`.
- Generation is serialized with an app-level mutex.
- Prompts use a Qwen-style chat wrapper:
  `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`.
- System prompt is: `Be concise. ASCII only.`
- Prompt text is sanitized before entering MNN.
- Streamed deltas are sanitized before being used by Kotlin/UI code.
- Stop sequences are detected in the streamed text.
- Responses are capped at `3,000` characters.

MNN generation uses `keepHistory=false`; multi-turn context is constructed
explicitly by the Kotlin prompt path.

### Performance Instrumentation

`EdgeTutorPerf.kt` logs structured timing and memory events.

Important query events:

- `query_memory_policy`
- `query_embed_before`
- `query_embed`
- `query_embed_after`
- `query_rewrite`
- `retrieval_search`
- `query_route`
- `prompt_sanitization`
- `prompt_metrics`
- `llm_prompt_sanitization`
- `llm_output_sanitization`
- `llm_decode_first_token`
- `llm_decode_total`
- `query_stage_timing`

Important ingestion events:

- `ingest_start`
- `embed_batch`
- `ingest_total`
- `ingest_pages_per_sec`
- `ingest_end`

Useful Logcat filter:

```bash
adb logcat EdgeTutorPerf:D MnnEngine:D EdgeTutorJNI:D AndroidRuntime:E *:S
```

---

## 5. Validation Status

### Automated Checks

Last local check in this workspace:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Result:

- `14 passed`

Notes:

- Running `pytest -q` with the system Python failed because that interpreter
  did not have `faiss` or `ollama` installed.
- The repository venv is the correct Python environment for current tests.
- Pytest emitted cache-write warnings for `.pytest_cache`, but tests passed.

Android unit test command:

```powershell
cd android-mnn
.\gradlew.bat testDebugUnitTest
```

Result:

- `BUILD SUCCESSFUL`
- `BUILD SUCCESSFUL` on June 27, 2026.

Notes:

- Gradle needs access to the user-level `.gradle` cache.

### Device Validation

Device validation remains the main open gate.

Current required device validation:

- Run the debug prompt-policy matrix and score answer quality.
- Run the fixed 16-case grounded, follow-up, unsupported-academic, and
  non-academic suite.
- Confirm no MNN/JNI UTF-8 failures.
- Confirm no context bleed across turns.
- Confirm first visible token timing with current prompt policy and model.
- Confirm memory behavior during ingestion and generation.
- Confirm user-facing answer quality on real textbook content.

### Android MNN Harness Device Validation

Measured on June 26, 2026 with:

- Device: Samsung SM-A047F-class target, device model `SM_A047F`
- App: `android-mnn/` debug build
- Runtime: MNN-LLM through the app JNI harness
- Model package: Qwen MNN package with `is_visual=1` and `has_deepstack=1`
- Document: `Calculus Made Easy.pdf`
- Query: `what is calculus`

Validated behavior:

- App launches without the previous model-readiness null-state crash.
- Model loads from app-owned `files/mnn_model`.
- Native MNN init succeeds after including `visual.mnn` and
  `visual.mnn.weight`.
- Existing document index survives app reinstall when data is preserved.
- Document screen opens with `Calculus Made Easy.pdf` in ready state.
- Warm-up loads the index and warms both embedder and LLM.
- A grounded query completes end to end with retrieval, prompt build, streaming,
  and total-answer instrumentation.

Measured ingestion result:

- Pages: `292`
- Chunks: `173`
- Total ingestion time: `94,009 ms`
- Throughput: `3.106 pages/sec`
- Index file: about `573 KB`
- Memory at ingestion end: about `903 MB` available, `3,707 MB` total,
  Android low-memory flag `false`

Measured warm-up result after ingestion:

- Index load: `104 ms`
- Embedder warm-up: `1,894 ms`
- LLM warm-up first token: `6,640 ms`
- LLM warm-up total: `6,914 ms`
- Warm-up prompt length: `25`
- Warm-up decode length: `6`

Measured grounded query result:

- Query memory policy: `888 MB` available, no embedder release
- Query embedding: `679 ms`
- Retrieval search: `60 ms`
- Prompt build: `33 ms`
- Route: `GROUNDED`
- Retrieved candidates: `5`
- Kept chunks: `2`
- Prompt chars: `1,729`
- Estimated prompt tokens: about `433`
- Native first token: `25,900 ms`
- UI-visible first token: `26,892 ms`
- Total answer time: `36,558 ms`
- Native decode total: `35,725 ms`
- Prompt length reported by native path: `421`
- Decode length: `71`
- Native prefill: `24,417,180 us`
- Native decode: `9,560,765 us`

Measured memory snapshots:

- After model loaded before ingestion: about `710 MB PSS`, `86 MB RSS`,
  `667 MB swap PSS`
- During ingestion: about `903 MB PSS`, `289 MB RSS`, `663 MB swap PSS`
- After warm-up: about `855 MB PSS`, `515 MB RSS`, `382 MB swap PSS`
- After answer: about `899 MB PSS`, `649 MB RSS`, `295 MB swap PSS`

MNN harness validation conclusion:

- Ingestion and retrieval are not the bottleneck on this device.
- The dominant query bottleneck is MNN native prompt prefill.
- Current first-token latency is usable for engineering validation but not yet a
  polished student experience on low-end hardware.
- The product path worth taking is to continue the MNN harness only if the next
  pass focuses on prompt/prefill reduction, smaller or better-fit MNN model
  packages, and an explicit waiting/thinking UI for the first-token gap.

Prompt-policy benchmark on June 27, 2026:

- `1x800` was selected as the production policy.
- All 12 `1x800` runs produced visible answers.
- Visible TTFT ranged from about `16.5s` to `19.5s`, with an `18.6s` median.
- `1x500` was faster but showed weaker grounding and false unsupported-answer
  behavior.
- Both two-chunk policies had blank/no-visible-answer runs and were rejected.

---

## 6. Known Issues and Risks

### Device TTFT Is Not Fully Revalidated

The app has instrumentation and prompt-budgeting, but the four-policy matrix
still needs to be measured on the target device.

Target:

- Median warm visible time to first token below 20 seconds on Samsung
  SM-A047F-class hardware.

### MNN Harness Prefill Latency

The June 26, 2026 `android-mnn/` device run completed successfully, but native
prefill dominated the grounded-query latency:

- Native first token: `25.9s`
- UI-visible first token: `26.9s`
- Total answer: `36.6s`
- Native prefill: about `24.4s`
- Retrieval plus prompt construction: under `100 ms` after query embedding

This means the next MNN work should not focus on vector search or PDF ingestion
first. The highest-leverage work is smaller prompts, stricter context budgets,
prompt-budget selection and better first-token waiting UX.

### Small-Model Answer Quality Needs Sampling

Qwen3.5 0.8B is selected, but grounded tutoring quality still needs
manual and scripted sampling across subjects.

Important scenarios:

- Definition questions.
- Worked examples.
- Follow-up questions.
- Out-of-document academic questions.
- Non-academic questions that still use retrieved context.

### MNN Context Isolation

Native generation uses `keepHistory=false`, while follow-up context is added
explicitly to the prompt. Consecutive device runs must still confirm that
independent validation cases do not inherit prior context.

### Fresh Clone Requires Local Assets

The Android project is not fully self-contained because large model assets are
ignored. This is intentional, but onboarding must include copying/generating
assets before building.

For the MNN harness, a fresh local setup must include the full MNN model folder,
including `visual.mnn` and `visual.mnn.weight` for the tested Qwen package.
Missing visual files can make the app report a model as incomplete or make
native MNN load fail, depending on the build being tested.

### Python Prototype Is Not Android Parity

Python is useful for quick testing, but it does not exactly match Android
retrieval, context budgeting, or route thresholds. Do not treat Python eval
results as final Android product results without device confirmation.

### FlatIndex Is MVP-Scale

`FlatIndex` is appropriate for the current single-document MVP. Multi-document
libraries or larger corpora may require a different storage/search design after
measurements justify the added complexity.

---

## 7. Current Roadmap

### Phase 1: Python RAG Prototype

Status: Complete for prototype use.

Implemented:

- PDF parsing and cleaning.
- Chunking.
- SentenceTransformer embeddings.
- FAISS indexing and retrieval.
- Ollama-backed answer generation.
- Basic tests and eval scripts.

Current caveat:

- Prototype behavior is not exact Android behavior.

### Phase 2: Android Port

Status: Complete for MVP.

Implemented:

- Android project in `android-ltk`.
- On-device PDF extraction.
- On-device embeddings.
- On-device vector index.
- On-device GGUF generation through Llamatik.
- Room metadata persistence.

### Phase 3: Baseline UX and Warm Runtime

Status: Implemented for MVP.

Implemented:

- Compose chat/document UI.
- Document load and warm-up flow.
- LLM asset copy and early native initialization.
- Embedder warm-up.
- LLM warm-up.
- First-token UI flush.

### Phase 4: TTFT Measurement

Status: Implemented.

Implemented:

- Query-stage timing.
- Prompt metrics.
- Native LLM first-token timing.
- Visible first-token timing.
- Total answer timing.
- Memory policy logging.

### Phase 5: Perceived TTFT Improvements

Status: Implemented, still needs repeated device validation.

Implemented:

- First non-empty assistant token is flushed immediately.
- Later token updates are buffered to reduce Compose churn.
- Native model initialization starts before document selection completes.
- Embedder release is conditional on low memory.

### Phase 6: Prompt and Prefill Optimization

Status: Implemented, still needs repeated device validation.

Implemented:

- Candidate retrieval `k=5`.
- Keep up to `2` context chunks.
- Cap context at `800` chars per kept chunk.
- Short grounded prompt with a small retrieved context budget.
- No document relevance gate before generation.
- Prompt and retrieval metrics.

Pending:

- Run the fixed 16-query validation suite with the selected `1x800` policy.
- Validate answer quality under smaller prompt budgets.

### Phase 7: Llamatik Runtime Tuning

Status: Closed without further work.

The project selected Qwen3.5 with MNN. Llamatik revalidation is no longer on the
critical path.

### Phase 7B: MNN Runtime Harness

Status: Selected runtime; optimization and validation continue.

Implemented:

- `android-mnn/` app harness with MNN-LLM native loading.
- App-owned model import and readiness validation.
- MNN model readiness UI.
- Native first-token and total-decode instrumentation.
- UTF-8-safe streamed chunks from native code.
- Thinking-tag filtering for streamed model output.
- Device-validated ingestion, warm-up, and one grounded query on Samsung
  SM-A047F-class hardware.

Current direction:

- Treat `android-mnn/` as the primary product app.
- Prioritize prefill reduction before broader UI/product expansion in the MNN
  harness.

Pending:

- Run repeated warm and cold grounded queries.
- Compare smaller prompt budgets and kept-chunk counts on device.
- Test follow-up, out-of-document academic, and non-academic query behavior.
- Add UX affordances for the first-token wait window.

### Phase 8: Model Selection

Status: Complete.

Selected model:

- Qwen3.5-0.8B-MNN with thinking disabled.

Alternative model comparison and Llamatik revalidation are closed.

### Phase 9: Python and GGUF Benchmark Cleanup

Status: Pending.

Needed:

- Add TTFT or first streamed chunk timing to Python/Ollama evals.
- Add first-chunk timing to local GGUF evals.
- Add retrieval-only warm-up before measured LLM turns.
- Use Ollama `keep_alive` for hot-model benchmark runs.
- Clearly label Python benchmark results as prototype screening, not Android
  product measurements.

### Phase 10: Retrieval Storage and Scale

Status: Pending.

Potential work:

- Quantize stored vectors.
- Test smaller embedding dimensions if model support permits it.
- Consider disk-backed vector search only if measurements show retrieval or
  index memory is a real bottleneck.

Current rule:

- Do not replace `FlatIndex` solely for theoretical performance gains.

### Phase 11: Product UI

Status: Pending.

Needed:

- Make Ghana Basic Education MVP subjects visible as first-run learning areas:
  Mathematics, Integrated Science, English Language, Social Studies, and
  Computing.
- Treat built-in curated subject corpora as the primary learning entry point.
- Keep document upload as a secondary personalization/import path.
- Library/document list polish.
- Upload and ingestion progress polish.
- Chat-screen polish.
- Source attribution polish.
- Better scanned-PDF handling.
- Better missing-asset, low-memory, and ingestion-error states.

### Phase 12: Device Testing and Pilot

Status: Pending.

Needed:

- Repeated warm and cold start tests.
- Repeated query tests on Samsung SM-A047F-class hardware.
- Memory and crash monitoring.
- Pilot testing with real students and real course PDFs.

---

## 8. Build and Run Notes

### Python

Use the repo virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Quick prototype usage:

```bash
python -c "from src.ingestion.pipeline import ingest; ingest('data/raw/MyBook.pdf')"
python -m src.rag.repl MyBook
```

Override prototype model:

```powershell
$env:EDGE_TUTOR_LLM_MODEL = "qwen2.5:0.5b"
```

### Android

Open `android-mnn/` in Android Studio, or run Gradle from the command line.

Required before running:

1. Ensure `arctic.onnx` and `vocab.txt` are in
   `android-mnn/app/src/main/assets/`.
2. Import the complete Qwen3.5 MNN model folder through the app.
3. Build/install the Android app.
4. Measure on a physical target device for meaningful performance results.

Command-line unit tests:

```powershell
cd android-mnn
.\gradlew.bat testDebugUnitTest
```

Debug builds currently target `arm64-v8a` only to keep APK footprint down.

---

## 9. Maintenance Rules

- Keep `PROJECT.md` aligned with code and measured results.
- If documentation and code disagree, code wins until a validated change lands.
- Do not describe planned runtime behavior as implemented.
- Do not describe historical benchmark results as current release guarantees.
- Every performance claim should name the model, device, prompt policy, and date
  of measurement.
- Android device logs are required before closing performance work.
- Avoid committing generated Android build outputs or model assets.
- Run repo hygiene before pushing:

```powershell
pwsh -File scripts/check_repo_hygiene.ps1
```

---

## 10. Immediate Next Steps

1. Run and manually score the fixed 16-query device validation suite using the
   selected `1x800` policy.
2. Improve the first-token waiting UX for the remaining roughly `18.6s` median
   warm first-token delay.
3. Check for context bleed across consecutive turns in the MNN runtime.
4. Clean up benchmark scripts so Python/Ollama and Android measurements are not
   confused.
