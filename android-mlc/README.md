# EdgeTutor Android — MLC-LLM variant

GPU-accelerated build using [MLC-LLM](https://github.com/mlc-ai/mlc-llm) (TVM/OpenCL backend)
instead of Llamatik/llama.cpp. Targets devices with an OpenCL-capable GPU (Adreno preferred).

| | android-ltk/ (Llamatik) | android-mlc/ (this module) |
|--|------------------------|---------------------------|
| LLM backend | llama.cpp via Llamatik | MLC-LLM (TVM/OpenCL) |
| Model | Qwen2.5-0.5B Q4_K_M GGUF | Qwen2.5-0.5B q4f16_0 MLC |
| LLM RAM | ~350 MB (CPU heap) | ~300-400 MB (GPU vRAM) |
| Embedding RAM | ~23 MB int8 ONNX | ~23 MB int8 ONNX |
| TTFT | baseline | ~2-5× faster (GPU prefill) |
| Initial load | fast | +10-30 s GPU weight transfer |
| Emulator support | yes | **no — physical device only** |

## Device requirements

- ARM64 Android device (arm64-v8a) — API 26 (Android 8.0)+
- **OpenCL-capable GPU required** — Adreno GPU (Snapdragon) is best-tested
- Cannot run in Android emulator

## One-time toolchain setup

MLC-LLM does not publish a Maven artifact. You must run `mlc_llm package` to generate
the native runtime and bundle model weights before the project can be built in Android Studio.

### 1. Install MLC-LLM Python toolchain

```bash
git clone https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm
pip install -e "python[dev]"
```

### 2. Install Android NDK 27

In Android Studio: **SDK Manager → SDK Tools → NDK (Side by side)** → install `27.0.11718014`.
Earlier NDK versions are not supported.

### 3. Set environment variables

```bash
# Linux / macOS
export ANDROID_NDK=$HOME/Android/Sdk/ndk/27.0.11718014
export TVM_NDK_CC=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-clang
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export TVM_SOURCE_DIR=/path/to/mlc-llm/3rdparty/tvm
export MLC_LLM_SOURCE_DIR=/path/to/mlc-llm
```

```powershell
# Windows (PowerShell)
$env:ANDROID_NDK        = "$env:LOCALAPPDATA\Android\Sdk\ndk\27.0.11718014"
$env:TVM_NDK_CC         = "$env:ANDROID_NDK\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android-clang.cmd"
$env:JAVA_HOME          = "C:\Program Files\Microsoft\jdk-17.x.x"
$env:TVM_SOURCE_DIR     = "C:\path\to\mlc-llm\3rdparty\tvm"
$env:MLC_LLM_SOURCE_DIR = "C:\path\to\mlc-llm"
```

### 4. Run `mlc_llm package`

From the `android-mlc/` directory:

```bash
cd edge-tutor/android-mlc
mlc_llm package
```

This will:
- Download `Qwen2.5-0.5B-Instruct-q4f16_0-MLC` weights from HuggingFace (~300-400 MB)
- Compile the TVM OpenCL runtime for Android arm64-v8a
- Generate `dist/lib/mlc4j/` — the Gradle subproject referenced by `settings.gradle.kts`
- Copy bundled weights into `app/src/main/assets/`

Expected output structure after completion:
```
android-mlc/dist/lib/mlc4j/
  build.gradle
  libs/tvm4j_core.jar
  src/main/jniLibs/arm64-v8a/libtvm4j_runtime_packed.so
```

### 5. Open in Android Studio and build

**File → Open → select `edge-tutor/android-mlc/`** (not the repo root).

Gradle sync should resolve both `:app` and `:mlc4j`. Deploy to a physical ARM64 device.

## Embedding model (same as android-ltk/)

`arctic.onnx` (int8 quantized, ~23 MB) + `vocab.txt` are NOT bundled by mlc_llm package.
Generate and copy them separately:

```bash
# From repo root
python scripts/export_onnx.py
cp models/arctic.onnx android-mlc/app/src/main/assets/
cp models/vocab.txt   android-mlc/app/src/main/assets/
```

## Model notes

- **Quantization**: `q4f16_0` — 4-bit weights, float16 activations
- **Do NOT use `q4f16_1` or `q4f32_1`** — the "_1" variants trigger Adreno GPU system-UI
  freezes on Snapdragon 855/865/870/888 devices. Always use "_0".
- **Context window**: 2048 tokens (sufficient for 3 × ~400-token RAG passages)

## MlcEngine.kt — replace reflection after mlc_llm package runs

`MlcEngine.kt` uses reflection so the project compiles before `mlc4j` is generated.
After running `mlc_llm package`, update the direct API calls by checking:

```
dist/lib/mlc4j/src/main/java/ai/mlc/mlcengine/MLCEngine.java
```

and the MLCChat reference app:
`https://github.com/mlc-ai/mlc-llm/tree/main/android/MLCChat`

## Key dependencies

| Library | Version | Purpose |
|---|---|---|
| mlc4j (generated) | — | MLC-LLM TVM/OpenCL runtime |
| ONNX Runtime Mobile | 1.22.0 | Embedding model inference |
| PdfBox Android | 2.0.27.0 | PDF text extraction |
| Jetpack Compose BOM | 2024.09 | UI |
| Room | 2.7.0 | Document metadata persistence |
| minSdk | 26 | Android 8.0 (MLC-LLM requirement) |
