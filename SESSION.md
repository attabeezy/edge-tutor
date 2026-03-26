# EdgeTutor — Project Status

## Phase Progress

| Phase | Status |
|---|---|
| Phase 1 — Python ingestion pipeline | **Complete** ✓ |
| Phase 2 — Python RAG pipeline | **Complete** ✓ |
| Phase 3 — Android port | **Complete** ✓ |
| Phase 4 — UI/UX (Jetpack Compose) | **In progress** |
| Phase 5 — Device testing & tuning | Not started |
| Phase 6 — Soft launch | Not started |

---

## Phase 1 & 2 — Python Prototype (Complete)

- Full ingestion pipeline: PDF → clean → chunk (400 tokens, 50 overlap) → FAISS index
- RAG query pipeline with two-gate out-of-scope filter:
  - L2 distance threshold (`MAX_RELEVANT_DISTANCE = 1.4`)
  - Lexical overlap gate (`MIN_LEXICAL_OVERLAP = 2` content-word matches)
- Three embedding models supported: `all-MiniLM-L6-v2` (default), `Snowflake/snowflake-arctic-embed-xs` (needs query prefix), `TaylorAI/bge-micro-v2`
- LLM: `qwen2.5:0.5b` via Ollama; REPL supports `-m` to override
- Eval: 12/12 retrieval precision (100%), responses reviewed and accepted
- System prompt: `"Be concise."` — factual questions ≤ 3 sentences; procedural/multi-concept questions may use more but limited to one step/idea

## Phase 3 — Android Port (Complete)

Full pipeline running on a physical Android device. Key decisions:

| Component | Implementation |
|---|---|
| PDF parsing | PdfBox Android (`com.tom-roush:pdfbox-android:2.0.27.0`) |
| Tokenizer | `WordPieceTokenizer.kt` (ported from Python) |
| Chunking | `TextChunker.kt` — paragraph-aware, 400 tokens / 50 overlap |
| Embedding | ONNX Runtime Mobile (`onnxruntime-android:1.18.0`) + `minilm.onnx` |
| Vector store | `FlatIndex.kt` — brute-force cosine similarity, pure Kotlin, no JNI |
| LLM | Llamatik `com.llamatik:library-android:0.11.0` + Qwen2.5-0.5B Q4_K_M GGUF |
| UI (scaffold) | Jetpack Compose — single-screen smoke test in `MainActivity.kt` |
| Database | Room 2.7.0 — document metadata (name, status, chunk count) |

**Known Llamatik API notes (version 0.11.0):**
- Class: `com.llamatik.library.platform.LlamaBridge` (singleton object)
- Streaming via `GenStream` interface: `onDelta(text)`, `onComplete()`, `onError(message)`
- Method: `LlamaBridge.generateStreamWithContext(systemPrompt, contextBlock, userPrompt, callback)`
- No lambda overload — must use `object : GenStream { ... }`

**Build fixes applied:**
- Added `gradle.properties` with `android.useAndroidX=true`
- Created `res/values/themes.xml` + added `com.google.android.material:material:1.12.0`
- Upgraded Room from 2.6.1 → 2.7.0 (Kotlin 2.2.x compatibility)
- Fixed `LlamaBridge` import: `com.llamatik.library.platform` (not `com.llamatik`)
- Removed `setTemperature` from MediaPipe options (not available in 0.10.27)

**Assets required (not in git — copy manually before build):**
- `edgetutor-android/app/src/main/assets/qwen2.5-0.5b-instruct-q4_k_m.gguf` (~380 MB)
  - Source: `hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf`
- `edgetutor-android/app/src/main/assets/minilm.onnx` (~23 MB)
  - Source: run `python scripts/export_onnx.py` then copy from `data/models/`
- `edgetutor-android/app/src/main/assets/vocab.txt`
  - Source: same export script

---

## Phase 4 — UI/UX (In Progress)

Replacing the smoke-test `EdgeTutorApp()` composable with proper screens.

**Navigation structure:**
```
MainActivity
  └── NavHost
        ├── Onboarding  (first launch only, SharedPrefs flag)
        ├── Library     (root — document list + FAB)
        ├── Chat        (per-document, navigated from Library)
        └── Settings    (from Library top bar)
```

**New dependency needed:**
```kotlin
implementation("androidx.navigation:navigation-compose:2.8.0")
```

**Screens planned:**
- Library — app bar, document cards with status badges, FAB, empty state
- Chat — message bubbles, thinking dots, source attribution chip, input bar
- Settings — storage usage, clear index, app version
- Onboarding — 3-page pager (Welcome → How it works → Add first book)

**Build order:** Library → Chat → Settings → Onboarding

---

## Running the Python prototype

```bash
# Install deps
pip install -r requirements.txt

# Ingest a PDF
python -c "from src.ingestion.pipeline import ingest; ingest('data/raw/MyDoc.pdf')"

# REPL
python -m src.rag.repl CalculusMadeEasy
python -m src.rag.repl CalculusMadeEasy -e arctic
python -m src.rag.repl CalculusMadeEasy -m granite4:350m

# Eval
python tests/eval_rag.py --generate
```
