# EdgeTutor Android

Kotlin/Jetpack Compose app — offline RAG tutoring assistant for Android.

## Status

- Phase 3 (Android port): **Complete** — full pipeline running on physical device
- Phase 4 (TTFT measurement): **Complete** — device-side instrumentation and measurement logs are in place
- Phase 5 (perceived TTFT): **Next** — first-token UI flush and low-risk latency polish

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

- **Time to first token** — dominated by prompt prefill on the current Llamatik path. Recent Samsung SM-A047F runs show visible TTFT around 105-118 seconds even with warm-up enabled.
- **Llamatik JNI UTF-8 crash** — `LlamaBridge.generateStream()` on 0.18.0 can abort in JNI when non-ASCII prompt/context text crosses the stream boundary. The app now sanitizes prompt text to ASCII before generation as a stability workaround.

## Measurement notes

- Use `adb logcat EdgeTutorPerf:D LlamaEngine:D AndroidRuntime:E *:S` for device measurement runs.
- Key events:
  - `query_stage_timing`
  - `prompt_metrics`
  - `prompt_sanitization`
  - `llm_decode_first_token`
  - `llm_decode_total`
- Current measurements show retrieval and embedding are cheap; native first-token latency is the main bottleneck.
