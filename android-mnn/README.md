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
> `MnnEngine` overrides this to `false` via the config merge so the model
> gives direct RAG answers without `<think>...</think>` preambles.

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
adb logcat EdgeTutorPerf:D MnnEngine:D AndroidRuntime:E *:S
```

---

## Notes

- `libMNN.so` is **not** compressed in the APK (`useLegacyPackaging = true`).
- `.mnn`, `.weight`, `.txt`, `.json`, `.md`, and `.bin` assets are excluded from APK compression via `noCompress`, matching the MNN chat app packaging style.
- The DB name is `edgetutor_mnn.db` so it does not conflict with `android-ltk`
  installed on the same device.
- `MnnEngine.keepHistory = false` for single-turn RAG parity with android-ltk.
  Multi-turn context is handled at the prompt level (same as ltk).
