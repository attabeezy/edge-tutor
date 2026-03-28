# EdgeTutor Android

## Phase 3 — Complete ✓

Full pipeline running on a physical Android device. All checklist items done:
- Android Studio opened, Gradle synced, SDK licences accepted
- Llamatik `com.llamatik:library-android:0.18.0` wired up
- ONNX Runtime Mobile `onnxruntime-android:1.22.0` (16 KB page-size aligned)
- compileSdk 36 (required by Llamatik 0.18.0; SDK Platform 36 auto-installs via Gradle)
- Gemma-3-270M-IT Q4_K_M GGUF + `minilm.onnx` placed in assets
- App launched successfully; 3 PDFs ingested; questions answered end-to-end

## Model files needed in assets/

| File | Source | Size |
|---|---|---|
| `gemma-3-270m-it-q4_k_m.gguf` | HuggingFace: google/gemma-3-270m-it-GGUF | ~253 MB |
| `minilm.onnx` | Run `python scripts/export_onnx.py` → `data/models/minilm.onnx` | ~23 MB |
| `vocab.txt` | Same export script → `data/models/vocab.txt` | <1 MB |

> These files are not tracked in git — copy them manually before building.

## Vector store

Uses `FlatIndex.kt` — brute-force cosine similarity, pure Kotlin, no JNI required. Sufficient for MVP document sizes.

## Phase 4 — UI/UX (In Progress)

Replacing the smoke-test `EdgeTutorApp()` composable with proper screens.

**Completed:**
- Model switched Qwen2.5-0.5B → Gemma-3-270M for smaller footprint (~253 MB vs ~380 MB)
- Warm-up UX: app shows "Loading model..." while LLM and ONNX embedder initialise; input disabled until ready
- Gemma stop sequences + 3,000-char truncation cap to prevent infinite generation
- Top-K retrieval reduced 3 → 2 chunks (cuts transformer prefill ~33%)

**Remaining screens:**
- Library — document list with ingestion status badges
- Upload — file picker, progress bar, background ingestion
- Chat — streamed output, thinking indicator, source attribution
- Settings — storage usage, clear index
- Onboarding — first-launch walkthrough

## Known Issues

**Llamatik 0.18.0 context accumulation** — after a few prompts in the same session, the model returns short affirmative non-answers instead of real responses. Likely cause: `LlamaBridge.generateStreamWithContext` accumulates internal KV cache state across calls. Workaround under investigation: check for a `resetContext()` method on `LlamaBridge`; if unavailable, consider re-initialising the bridge between calls or reverting to a stateless generation method.
