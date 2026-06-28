# android-mnn — EdgeTutor MNN-LLM Variant

Android implementation of EdgeTutor using **MNN-LLM** as the on-device
inference engine instead of Llamatik/llama.cpp.

The embedding pipeline (ONNX Runtime + Arctic XS), the vector store (FlatIndex),
and the PDF ingestion pipeline are **identical** to `android-ltk` so TTFT and
quality measurements are directly comparable between the two variants.

---

## Architecture

| Layer | android-ltk | android-mnn |
|---|---|---|
| **LLM engine** | Llamatik (llama.cpp, GGUF) | MNN-LLM (MNN, .mnn weights) |
| **JNI bridge** | LlamaBridge (Llamatik AAR) | `MnnNativeBridge` → `libMNN.so` (monolithic) |
| **Embedding** | ONNX Runtime + arctic.onnx | **same** |
| **Vector store** | FlatIndex | **same** |
| **PDF parsing** | PDFBox Android | **same** |
| **UI** | Compose (ChatViewModel) | **same** |

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

The model is at `models/Qwen3.5-0.8B-MNN/` in the repo root (~516 MB total).
Push it to the device in **two steps** — adb cannot write directly to app
`filesDir` without root, so we stage via `/sdcard/` first:

**Step 1 — Authorize USB debugging** on the phone when prompted, then:

```powershell
# Push all model files to external storage (works without root)
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb push models\Qwen3.5-0.8B-MNN\ /sdcard/Download/mnn_model/
```

**Step 2 — Copy from sdcard into app's private filesDir:**

```powershell
& $adb shell run-as com.edgetutor.mnn mkdir -p /data/data/com.edgetutor.mnn/files/mnn_model
& $adb shell run-as com.edgetutor.mnn cp -r /sdcard/Download/mnn_model/. /data/data/com.edgetutor.mnn/files/mnn_model/
```

> **Note:** `run-as` only works on debug builds and debuggable devices.
> For a production workflow, implement a file-copy step at first app launch
> that reads from `/sdcard/Download/mnn_model/` into `context.filesDir`.

Files pushed:
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

### 3. ONNX embedding assets (same as android-ltk)

```
app/src/main/assets/arctic.onnx
app/src/main/assets/vocab.txt
```

Copy from `android-ltk/app/src/main/assets/` — they are the same files.

---

## Build

```powershell
cd android-mnn
.\gradlew.bat assembleDebug
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
| `llm/LlmEngine.kt` | Shared engine interface (mirrors android-ltk) |
| `llm/PromptSanitizer.kt` | ASCII safety filter (verbatim copy from ltk) |
| `viewmodel/ChatViewModel.kt` | RAG query + streaming (MnnEngine plugged in) |
| `viewmodel/IngestViewModel.kt` | PDF ingestion (identical to ltk) |
| `perf/EdgeTutorPerf.kt` | Structured Logcat perf logger |

---

## Logcat filter

```bash
adb logcat EdgeTutorPerf:D MnnEngine:D EdgeTutorJNI:D AndroidRuntime:E *:S
```

For query validation, confirm each run contains `llm_thinking_config`,
`query_route`, `prompt_metrics`, `llm_decode_first_token`,
`llm_decode_total`, and either `query_complete` or `query_failed`.

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
answers, sources, selected query route, top-1 and top-2 cosine similarity, mean
top-5 similarity, the routing threshold, and blank 0-2 rubric columns for
manual review.

Automatic routing uses the mean Arctic Embed XS cosine similarity across the
five highest-ranked chunks. A mean at or above `0.63165` uses textbook
passages; lower means use general generation. Follow-ups are independently
routed from their rewritten retrieval query rather than inheriting the
preceding route. The threshold is experimental and must be validated across
additional textbooks before release.

---

## Notes

- `libMNN.so` is **not** compressed in the APK (`useLegacyPackaging = true`).
- `.mnn`, `.weight`, `.txt`, `.json`, `.md`, and `.bin` assets are excluded from APK compression via `noCompress`, matching the MNN chat app packaging style.
- The DB name is `edgetutor_mnn.db` so it does not conflict with `android-ltk`
  installed on the same device.
- `MnnEngine.keepHistory = false` for single-turn RAG parity with android-ltk.
  Multi-turn context is handled at the prompt level (same as ltk).
