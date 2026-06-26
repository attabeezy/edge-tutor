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

Debug APKs are packaged for `arm64-v8a` only to keep device builds small. Add other ABIs only when building emulator or multi-architecture release artifacts.

## Model files required in `assets/`

Not tracked in git - store working copies in `models/` at the repo root. `python scripts/export_onnx.py` copies `arctic.onnx` and `vocab.txt`; copy or download the GGUF manually before building.

| File | Source | Size |
|---|---|---|
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | HuggingFace GGUF for Qwen2.5 0.5B Instruct Q4 | ~469 MB |
| `arctic.onnx` | Run `python scripts/export_onnx.py` | ~23 MB (int8) |
| `vocab.txt` | Same export script | ~226 KB |

```bash
# From the repo root, after placing the GGUF in models/
copy models/qwen2.5-0.5b-instruct-q4_k_m.gguf android-ltk/app/src/main/assets/
```

## Local GGUF swap candidates

`LlamaEngine.MODEL_ASSET` currently points to `qwen2.5-0.5b-instruct-q4_k_m.gguf`. To test another GGUF, copy it from `models/` into `android-ltk/app/src/main/assets/`, update `MODEL_ASSET`, rebuild, and reinstall the debug APK.

| File in `models/` | Size | Test priority |
|---|---:|---|
| `LFM2.5-350M-Q4_K_M.gguf` | 219 MB | Previous default; smallest known baseline. |
| `LFM2-350M-Math-Q4_K_M.gguf` | 219 MB | First math-quality A/B test; same footprint class as current. |
| `granite-4.0-h-350m-Q4_K_M.gguf` | 212 MB | Same footprint class; useful general instruction baseline. |
| `Qwen_Qwen3-0.6B-Q4_K_M.gguf` | 462 MB | Stronger ceiling candidate; may need prompt tuning. |
| `Qwen3-0.6B-Q8_0.gguf` | 610 MB | Last-resort ceiling test for 1 GB devices; highest memory risk. |

For 1 GB-class devices, prefer Q4 models first. The on-disk GGUF size is not the full runtime footprint because native heap, KV cache, Android UI, Room, and the ONNX embedder also consume memory.

## Key dependencies

| Library | Version | Purpose |
|---|---|---|
| Llamatik | 1.7.0 | llama.cpp wrapper for GGUF inference |
| ONNX Runtime Mobile | 1.22.0 | Embedding model inference (16 KB page-aligned) |
| PdfBox Android | 2.0.27.0 | PDF text extraction |
| Jetpack Compose BOM | 2024.09 | UI |
| Room | 2.7.0 | Document metadata persistence |
| compileSdk | 36 | Android build target |

## Llamatik API notes

- Class: `com.llamatik.library.platform.LlamaBridge` (singleton object)
- Streaming: `LlamaBridge.generateStream(prompt, object : GenStream { ... })`
- Callbacks: `onDelta(text)`, `onComplete()`, `onError(message)` — no lambda overload
- Context reset: `LlamaBridge.sessionReset()` is intentionally not called; generation is serialized with the app-level mutex instead.

## Known issues

- **Time to first token** — dominated by native prompt prefill on the current Llamatik path. The Phase 4 baseline on Samsung SM-A047F was roughly 105-118 seconds visible TTFT with ~6k-char prompts. The balanced prompt policy reduced tested prompts to roughly 1.7k chars and visible TTFT to roughly 24-27 seconds.
- **Small-model answer quality** — follow-up and worked-example prompts are now more explicit, but answer quality still needs device sampling across subjects.
- **Llamatik JNI UTF-8 crash** — `LlamaBridge.generateStream()` can abort in JNI when invalid UTF-8 reaches `NewStringUTF`. The app sanitizes prompts in `ChatViewModel` and again at the `LlamaEngine` native boundary, then sanitizes Kotlin-visible streamed deltas before UI use. If native still aborts before `onDelta`, the fix must be in Llamatik itself via an upgrade or fork.

## Measurement notes

- Use `adb logcat EdgeTutorPerf:D LlamaEngine:D AndroidRuntime:E *:S` for device measurement runs.
- Key events:
  - `query_stage_timing`
  - `prompt_metrics`
  - `prompt_sanitization`
  - `llm_prompt_sanitization`
  - `llm_output_sanitization`
  - `query_route`
  - `query_rewrite`
  - `llm_decode_first_token`
  - `llm_decode_total`
- `prompt_metrics` includes retrieved/kept counts, similarity scores, context cap, final context chars, prompt chars, estimated prompt tokens, and query route.
- Current measurements show retrieval and embedding are cheap; native prompt prefill is the main first-token bottleneck.
