# android-mnn — EdgeTutor Android App

Primary EdgeTutor product application using **MNN-LLM** for on-device
inference.

The app combines Arctic XS through ONNX Runtime, a local FlatIndex, PDFBox
ingestion, and Qwen3.5-0.8B-MNN generation.

---

## Architecture

| Layer | Implementation |
|---|---|
| **LLM engine** | MNN-LLM with Qwen3.5-0.8B |
| **JNI bridge** | `MnnNativeBridge` → monolithic `libMNN.so` |
| **Embedding** | ONNX Runtime + `arctic.onnx` |
| **Vector store** | FlatIndex |
| **PDF parsing** | PDFBox Android |
| **UI** | Android views, Room-backed sessions |

---

## Required local assets

All three must be present **before building** — they are git-ignored:

### 1. libMNN.so  ✅ Already present

```
app/src/main/jniLibs/arm64-v8a/libMNN.so
```

Already compiled and copied from `MNN/project/android/build_64/libMNN.so`.
Build flags used: `MNN_BUILD_LLM=ON`, `MNN_ARM82=ON`, `MNN_BUILD_SHARED_LIBS=ON`, Release.

To rebuild (if needed):

```bash
cd MNN/project/android
mkdir build_64 && cd build_64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=android-21 \
  -DMNN_BUILD_LLM=ON \
  -DMNN_ARM82=ON \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) MNN
cp libMNN.so \
    ../../../android-mnn/app/src/main/jniLibs/arm64-v8a/libMNN.so
```

### 2. MNN model files — Qwen3.5-0.8B-MNN ✅ Files ready locally

The model is at `models/Qwen3.5-0.8B-MNN/` in the repository root (~516 MB).
The Android build validates the required files and stages them as generated
assets under `mnn_model/`; no model files are copied into tracked source
directories. A missing or incomplete local model fails the build with the
missing filenames.

On first launch, the app copies the bundled model transactionally into
`context.filesDir/mnn_model/`, validates it, and then initializes MNN. Keep at
least 1.2 GB free for the installed APK, the private model copy, and working
space. Later launches reuse the validated private copy.

The Settings **Import Model** picker remains available if automatic extraction
fails or a development model needs to be supplied manually.

Files bundled:
```
config.json          (MNN-LLM session config — thinking disabled at runtime)
llm.mnn              (~2 MB, model graph)
llm.mnn.weight       (~449 MB, quantised weights)
llm.mnn.json         (~5 MB, graph metadata)
llm_config.json      (architecture config)
tokenizer.txt        (~6 MB, BPE tokenizer)
visual.mnn           (~0.24 MB, vision encoder — unused for text-only RAG)
visual.mnn.weight    (~60 MB, vision weights — unused for text-only RAG)
```

> **Thinking mode:** Qwen3.5-0.8B defaults to `enable_thinking=true`.
> EdgeTutor does not expose that mode for this model package: hidden reasoning
> can consume the complete native token budget without producing a visible
> answer. `MnnEngine` forces `false` at initialization and before every query,
> reads the effective merged native config back, and fails generation if the
> override is not present. `ThinkingTagFilter` remains as defense in depth.

Normal text and vision responses are capped at 192 generated tokens. Warm-up
generation uses a separate 8-token cap.

### 3. ONNX embedding assets

```
app/src/main/assets/arctic.onnx
app/src/main/assets/vocab.txt
```

Generate and copy these with `python scripts/export_onnx.py` from the repository
root.

---

## Build

```powershell
cd android-mnn
.\gradlew.bat assembleDebug
```

Sideload the single APK; a separate model push is no longer required:

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" install -r app\build\outputs\apk\debug\app-debug.apk
```

Unit tests (no device needed):

```powershell
cd android-mnn
.\gradlew.bat testDebugUnitTest
```

---

## Key source files

| File | Purpose |
|---|---|
| `llm/MnnNativeBridge.kt` | JNI `object` that loads `libedgetutor_mnn.so`, linked against monolithic `libMNN.so` |
| `llm/MnnProgressListener.kt` | Per-token callback interface |
| `llm/MnnEngine.kt` | `LlmEngine` implementation backed by MNN |
| `llm/LlmEngine.kt` | Shared engine interface |
| `llm/PromptSanitizer.kt` | Prompt safety filter |
| `viewmodel/ChatViewModel.kt` | RAG query + streaming (MnnEngine plugged in) |
| `viewmodel/IngestViewModel.kt` | PDF ingestion |
| `perf/EdgeTutorPerf.kt` | Structured Logcat perf logger |

---

## Logcat filter

```bash
adb logcat EdgeTutorPerf:D MnnEngine:D EdgeTutorJNI:D AndroidRuntime:E *:S
```

For query validation, confirm each run contains `llm_thinking_config`,
`prompt_metrics`, `llm_decode_first_token`,
`llm_decode_total`, and either `query_complete` or `query_failed`.

Routing telemetry includes `answer_route`, `route_marker_valid`, and
`route_reason`. Textbook sources are attached only when the model begins with
`[TEXTBOOK]`. `[GENERAL]` and malformed markers hide sources and receive the
visible model-knowledge warning.

## Debug device validation

After importing a model and loading a document, debug builds expose:

- **benchmark prompts** — runs the four real-RAG prompt policies (`2x800`,
  `2x500`, `1x800`, and `1x500`) against four grounded questions, three times
  each.
- **validate queries** — runs the fixed 16-case grounded, follow-up,
  unsupported-academic, and non-academic suite.

Start either suite over ADB without adding controls to the production UI:

```powershell
& $adb shell am start -a com.edgetutor.mnn.action.RUN_VALIDATION `
  -n com.edgetutor.mnn/.MainActivity --activity-single-top
```

Reports are written to the app-specific external files directory:

```
Android/data/com.edgetutor.mnn/files/reports/
```

Pull them with:

```powershell
& $adb pull /sdcard/Android/data/com.edgetutor.mnn/files/reports/ reports/android-mnn/
```

The CSV contains native prefill/decode timing, visible TTFT, total time, memory,
answers, sources, model-selected answer route, marker validity, retrieval
similarities, and blank 0-2 rubric columns for manual review.

Similarity values are diagnostic only. Qwen selects `[TEXTBOOK]` or
`[GENERAL]` in the same generation that produces the answer.

---

## Notes

- `libMNN.so` is **not** compressed in the APK (`useLegacyPackaging = true`).
- `.mnn`, `.weight`, `.txt`, `.json`, `.md`, and `.bin` assets are excluded from APK compression via `noCompress`, matching the MNN chat app packaging style.
- The bundled model roughly doubles its device-storage cost during first launch because the native runtime requires normal filesystem paths rather than APK asset paths.
- Updating an installed MVP build preserves an existing valid private model; clear app data or reinstall to force extraction of changed bundled weights.
- The DB name is `edgetutor_mnn.db`.
- Native history is reset between generations. Multi-turn context is rebuilt
  explicitly at the prompt level.
