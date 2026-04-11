# EdgeTutor - Project Roadmap

**Date:** April 11, 2026  
**Status:** Phase 3 (RAG Integration & Local LLM)  
**Target Device:** Samsung SM-A047F (3GB/4GB RAM, arm64-v8a)

---

## Current Progress

### Models & Assets
- **Embedding Model:** `arctic.onnx` (~23 MB int8 quantized)
- **LLM Model:** Qwen3-0.6B-Q4_K_M (462 MB) or Gemma-3-270M-Q4_K_M (242 MB)
- **Vocabulary:** `vocab.txt` for wordpiece tokenization

### Android Projects
- **android-ltk/** - Llamatik (llama.cpp) variant - READY TO BUILD
- **android-mlc/** - MLC-LLM (TVM/OpenCL) variant - Requires toolchain setup

---

## Roadmap

### Phase 4 - UI/UX
- [x] Active LLM: Qwen3-0.6B via Llamatik
- [x] Warm-up UX + stop sequences
- [x] Top-K reduced to 2 chunks
- [ ] Library screen (document list)
- [ ] Upload screen (file picker + progress)
- [ ] Chat screen improvements (source attribution)
- [ ] Settings screen

### Phase 5 - Device Testing
- [ ] Test on minimum-spec device (1 GB free RAM)
- [ ] Measure TTFT (target < 30s)
- [ ] Informal user testing with 5-10 students

### Phase 6 - Soft Launch
- [ ] Release APK to 20-50 pilot students
- [ ] Collect feedback

---

## Android Build Options

### Option A: android-ltk (Recommended - READY NOW)

Uses Llamatik (llama.cpp wrapper) - no special toolchain required.

**Requirements:**
- Android Studio installed
- JDK 17 (Temurin) - already installed at `C:/Program Files/Eclipse Adoptium/jdk-17.0.18.8-hotspot`
- NDK (any recent version) - NDK 30 available

**Build Steps:**
1. Open `android-ltk/` in Android Studio
2. Wait for Gradle sync
3. Connect device or start emulator
4. Run the app

**Expected RAM:** ~473 MB (450 MB LLM + 23 MB embedding)

---

### Option B: android-mlc (GPU Variant - Advanced)

Uses MLC-LLM with TVM/OpenCL for GPU-accelerated inference.

**Requirements:**
- Android NDK 27.0.11718014 (NOT installed - need to install via Android Studio SDK Manager)
- JDK 17 (installed)
- MLC-LLM Python package with native build

**Status:** BLOCKED - See "MLC-LLM Build Issues" below

---

## MLC-LLM Build Issues

### What We Tried

1. **JDK 17 Installation** - Completed successfully
2. **MLC-LLM Python Package** - Failed due to native DLL build requirement

### The Problem

MLC-LLM requires building `mlc_llm.dll` from C++ sources using CMake + TVM + LLVM. This is complex on Windows:
- Requires C++ build tools (CMake, Ninja, LLVM)
- Needs TVM compilation from source
- No pre-built wheel available on PyPI

### Prerequisites Still Needed

1. **Android NDK 27** - Must be installed via Android Studio SDK Manager:
   - SDK Manager → SDK Tools → NDK (Side by side) → install `27.0.11718014`

2. **Linux/WSL** - MLC-LLM build is well-supported on Linux. Consider:
   - Using WSL (Windows Subsystem for Linux)
   - Using a Linux machine
   - Using CI/CD pipeline for builds

### Alternative Approaches

1. **Use Pre-built MLC AAR** - Not officially available, experimental
2. **Wait for Official Release** - MLC-LLM may add pre-built Android artifacts later
3. **Build on Linux** - Set up WSL and build there

---

## Immediate Action Items

### For android-ltk (Recommended)

1. ~~Install JDK 17~~ ✅ Done
2. Open `android-ltk/` in Android Studio
3. Build and test on device

### For android-mlc (Blocked)

1. Install Android NDK 27 via Android Studio
2. Set up WSL or Linux machine for MLC-LLM build
3. Run `mlc_llm package` to generate mlc4j

---

## Quick Reference

### Build android-ltk
```bash
cd android-ltk
.\gradlew assembleDebug
# Output: app/build/outputs/apk/debug/app-debug.apk
```

### Install APK
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Smaller Model for Budget Devices
If Qwen3-0.6B (462 MB) causes OOM, switch to Gemma-3-270M (242 MB):
1. Copy `models/gemma-3-270m-it-Q4_K_M.gguf` to `android-ltk/app/src/main/assets/`
2. Update `LlamaEngine.kt`: `private const val MODEL_ASSET = "gemma-3-270m-it-Q4_K_M.gguf"`

### Monitor Logcat
```bash
adb logcat -s LlamaEngine
```

---

## Files to NOT Commit

```
models/                     # Model files
android-ltk/app/src/main/assets/*.gguf
android-ltk/app/src/main/assets/*.onnx
.venv/                      # Python virtual environment
data/                       # Runtime PDFs and FAISS indices
```