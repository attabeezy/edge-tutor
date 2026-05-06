# EdgeTutor Android

Kotlin/Jetpack Compose app — offline RAG tutoring assistant for Android.

## Status

- Phase 3 (Android port): **Complete** — full pipeline running on physical device
- Phase 4 (TTFT measurement): **Complete** — device-side instrumentation and measurement logs are in place
- Phase 5 (perceived TTFT): **Partially complete** — first-token UI flush is implemented
- Phase 6 (prompt/prefill): **Implemented** — balanced context budget and query routing are active

## Build

1. Open `android-ltk/` in Android Studio (File -> Open)
2. Copy model files into `android-ltk/app/src/main/assets/` (see table below)
3. Gradle sync -> Run on device

## Model files required in `assets/`

Not tracked in git - store working copies in `models/` at the repo root. `python scripts/export_onnx.py` copies `arctic.onnx` and `vocab.txt`; copy or download the GGUF manually before building.

| File | Source | Size |
|---|---|---|
| `LFM2.5-350M-Q4_K_M.gguf` | HuggingFace: `LiquidAI/LFM2.5-350M-GGUF` | ~267 MB |
| `arctic.onnx` | Run `python scripts/export_onnx.py` | ~23 MB (int8) |
| `vocab.txt` | Same export script | ~226 KB |

```bash
hf download LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf
```

## Key dependencies

| Library | Version | Purpose |
|---|---|---|
| Llamatik | 0.18.0 | llama.cpp wrapper for GGUF inference |
| ONNX Runtime Mobile | 1.22.0 | Embedding model inference (16 KB page-aligned) |
| PdfBox Android | 2.0.27.0 | PDF text extraction |
| Jetpack Compose BOM | 2024.09 | UI |
| Room | 2.7.0 | Document metadata persistence |
| compileSdk | 36 | Required by Llamatik 0.18.0 |

## Llamatik 0.18.0 API notes

- Class: `com.llamatik.library.platform.LlamaBridge` (singleton object)
- Streaming: `LlamaBridge.generateStream(prompt, object : GenStream { ... })`
- Callbacks: `onDelta(text)`, `onComplete()`, `onError(message)` — no lambda overload
- Context reset: `LlamaBridge.sessionReset()` — documented but native JNI implementation is missing in 0.18.0 (`UnsatisfiedLinkError`); do NOT call it

## Known issues

- **Time to first token** — dominated by native prompt prefill on the current Llamatik path. The Phase 4 baseline on Samsung SM-A047F was roughly 105-118 seconds visible TTFT with ~6k-char prompts. The balanced prompt policy reduced tested prompts to roughly 1.7k chars and visible TTFT to roughly 24-27 seconds.
- **Small-model answer quality** — follow-up and worked-example prompts are now more explicit, but answer quality still needs device sampling across subjects.
- **Llamatik JNI UTF-8 crash** — `LlamaBridge.generateStream()` on 0.18.0 can abort in JNI when non-ASCII prompt/context text crosses the stream boundary. The app now sanitizes prompt text to ASCII before generation as a stability workaround.

## Measurement notes

- Use `adb logcat EdgeTutorPerf:D LlamaEngine:D AndroidRuntime:E *:S` for device measurement runs.
- Key events:
  - `query_stage_timing`
  - `prompt_metrics`
  - `prompt_sanitization`
  - `query_route`
  - `query_rewrite`
  - `llm_decode_first_token`
  - `llm_decode_total`
- `prompt_metrics` includes retrieved/kept counts, similarity scores, context cap, final context chars, prompt chars, estimated prompt tokens, and query route.
- Current measurements show retrieval and embedding are cheap; native prompt prefill is the main first-token bottleneck.
