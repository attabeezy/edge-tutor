# EdgeTutor - Project Master Document
**Date:** April 29, 2026  
**Status:** Phase 5 (Perceived TTFT Improvements)  
**Target Device:** Samsung SM-A047F (3GB/4GB RAM, arm64-v8a)

---

## 1. Vision & Architecture

### Overview
EdgeTutor is an offline RAG (Retrieval-Augmented Generation) tutoring assistant for Android. It allows students to query their textbooks directly on their devices with zero internet dependency, specifically optimized for budget hardware.

### Implementation Targets
- **RAM:** < 600MB active usage (leaving 400MB buffer for OS).
- **Latency:** < 30s TTFT (Time To First Token).
- **Accuracy:** > 70% retrieval precision on academic technical content.

### Technical Stack
**Current Implementation (Llamatik)**
- **LLM Engine:** Llamatik (llama.cpp wrapper)
- **Active Model:** `LFM2.5-350M-Q4_K_M.gguf` (~267 MB)
- **Embedding:** Snowflake Arctic Embed XS (ONNX Runtime Mobile, ~23 MB int8)
- **Vector Store:** In-memory `FlatIndex` (Kotlin)
- **Database:** SQLite/Room for document metadata

**Target "Nano-Tutor" Stack (Phase 4.1)**
- **Models:** Granite-4.0-350M or Liquid LFM-2.5-350M (lower RAM, higher speed).
- **Vector Store:** `sqlite-vec` (Native SQLite extension) for disk-based retrieval.
- **Orchestration:** Kotlin Coroutines with Zero-Copy data flow.

---

## 2. Phased Roadmap

### Phase 1: Python RAG Prototype - Complete
Goal: prove the offline textbook QA workflow before Android porting.

- [x] Python RAG MVP with FAISS.
- [x] PDF/TXT ingestion pipeline.
- [x] Sentence embedding and retrieval flow.
- [x] Ollama-backed answer generation.
- [x] Basic model evaluation scripts.

### Phase 2: Android Port - Complete
Goal: move the RAG pipeline on-device.

- [x] Android project structure setup (`android-ltk`).
- [x] Llamatik integration for GGUF inference.
- [x] ONNX Runtime Mobile for embeddings.
- [x] In-memory Kotlin `FlatIndex`.
- [x] Room database for document metadata.
- [x] Quantized `arctic.onnx` to int8 (87 MB -> 23 MB).
- [x] Fixed thread-safe embedding model loading.

### Phase 3: Baseline UX + Warm Runtime - Complete
Goal: make the current Android app usable enough to optimize from real traces.

- [x] Warm-up UX state.
- [x] Stop sequence handling.
- [x] Real ONNX embedder warm-up in `Embedder.kt`.
- [x] Real LLM dummy generation warm-up in `LlamaEngine.kt`.
- [x] Background GGUF asset copy in `ChatViewModel.init`.
- [x] Token buffering for Compose updates.
- [x] Basic native TTFT logging at first non-empty Llamatik stream delta.

### Phase 4: TTFT Measurement - Complete
Goal: split user-perceived latency into measurable stages before changing behavior.

Exit gate: one device run can report query embed time, retrieval time, prompt size, native TTFT, visible TTFT, total answer time, and memory state.

- [x] Add prompt metrics before `llm.generate()`:
    - `prompt_chars`
    - estimated `prompt_tokens`
    - `top_k`
    - per-chunk char counts
    - final context char count
- [x] Add query-stage timing:
    - `query_embed_ms`
    - `retrieval_search_ms`
    - `prompt_build_ms`
    - `llm_native_ttft_ms`
    - `ui_visible_ttft_ms`
    - `total_answer_ms`
- [x] Split LLM warm-up metrics:
    - asset copy/check
    - native `initGenerateModel`
    - dummy decode first token
    - dummy decode total
- [x] Log when `genMutex` blocks a user query behind warm-up.
- [x] Log how often `LOW_MEM_THRESHOLD_MB` causes embedder release before generation.
- [x] Keep TTFT benchmarks opt-in. Do not make local LLM benchmarks part of normal `pytest`.

Current findings on Samsung SM-A047F:
- Warm-up and query instrumentation is working end to end in Logcat.
- Retrieval and embedding are cheap relative to TTFT.
- Prompt prefill dominated the Phase 4 baseline; visible TTFT was roughly 105-118 s on `Calculus Made Easy`.
- A Llamatik JNI `NewStringUTF` crash was reproduced and mitigated with prompt sanitization on the app side.
- May 6 first-token flush run confirmed the UI buffer is no longer the main TTFT bottleneck:
    - `prompt_chars=6080`, `estimated_prompt_tokens=1520`, `top_k=3`
    - `chunk_char_counts=1980,1950,2005`, `final_context_chars=5931`
    - `llm_native_ttft_ms=104178`
    - `ui_visible_ttft_ms=104906`, `source=first_token_flush`
    - Native first token to visible first token gap was roughly 728 ms; the remaining work is prompt/prefill reduction.
- May 6 prompt-prefill run confirmed the balanced Android policy materially reduced TTFT:
    - `retrieval_search k=5`, `kept_k=2`, `context_char_cap=800`
    - `prompt_chars` roughly 1,700 and `estimated_prompt_tokens` roughly 425
    - `llm_native_ttft_ms` roughly 23,800-26,000
    - `ui_visible_ttft_ms` roughly 24,300-26,700
    - Follow-up routing now adds a small previous-question context and logs `query_rewrite`.

### Phase 5: Low-Risk Perceived TTFT Improvements - In Progress
Goal: reduce visible latency without changing model choice or retrieval architecture.

Exit gate: visible TTFT improves or stays stable without increasing crashes/OOMs on Samsung SM-A047F.

- [x] Emit the first assistant token immediately, then resume the current 50 ms buffered UI flushing.
- [x] Keep batched flushing after the first token to avoid Compose churn.
- [ ] Move LLM native model load earlier than document selection when memory allows.
- [ ] Keep document-load warm-up for dummy decode, but avoid hiding native load inside it.
- [ ] Benchmark sequential vs concurrent document-load warm-up:
    - current: index load -> embedder warm-up -> LLM warm-up
    - candidate: index load plus embedder/LLM warm-up in parallel under a memory gate
- [ ] Benchmark thresholds below 120 MB for embedder release.
- [ ] If release is frequent, test delayed embedder close after generation rather than immediate close before generation.

### Phase 6: Prompt / Prefill Optimization - Implemented
Goal: reduce native TTFT by shrinking prompt prefill cost while preserving grounded answer quality.

Exit gate: chosen prompt policy passes grounded-answer, general-reasoning, and unrelated-query checks with lower median native TTFT.

Immediate target from May 6 run:
- Current query prompt is too large for the target device: roughly 6k chars / 1.5k estimated prompt tokens from three ~2k-char chunks.
- First implementation should reduce prompt size before changing model, runtime, or answer length.
- Candidate default: retrieve more candidates for filtering, but pass fewer and shorter chunks into the final prompt.

Retrieval and context policy:
- [x] Change retrieval from fixed `top_k=3` final context to candidate retrieval plus filtering:
    - retrieve `RETRIEVAL_CANDIDATE_K=5`
    - filter chunks below a similarity threshold before prompt construction
    - cap each included chunk with `MAX_CONTEXT_CHARS_PER_CHUNK=800`
    - target prompt size around 1,500-2,500 chars before testing smaller caps
- [ ] Benchmark chunk caps of 500, 800, and 1200 chars per retrieved chunk.
- [ ] Benchmark final kept chunk counts of 1, 2, and 3 after similarity filtering.
- [x] Start with balanced default: retrieve 5, keep up to 2 relevant chunks, max 800 chars per kept chunk.
- [x] Log `retrieved_k`, `kept_k`, `max_sim`, `kept_sim_scores`, `dropped_sim_scores`, `context_char_cap`, final context chars, and prompt chars.

Answer routing policy:
- [x] Replace the current binary in-scope/out-of-scope behavior with three routes:
    - `grounded`: one or more chunks pass the context similarity threshold; answer using document passages only.
    - `general_reasoning`: document chunks are weak, but the question is still academic, computational, mathematical, scientific, or tutoring-related; answer from model reasoning and clearly say the document did not provide strong support.
    - `unrelated`: the question is weakly related to the document and not a viable tutoring/reasoning request; refuse before calling the LLM.
- [x] Keep a refusal gate for completely unrelated queries so the app does not waste generation time on irrelevant prompts.
- [x] Add query-route logging: `query_route=GROUNDED|GENERAL_REASONING|UNRELATED`, similarity values, lexical overlap, and route reason.
- [x] Add follow-up query rewrite logging with `query_rewrite` for lightweight previous-question context.
- [ ] Add evaluation questions for all three routes:
    - grounded document question with high-similarity chunks
    - general reasoning question with weak document matches but educational intent
    - unrelated/non-tutoring question that should be rejected before generation

Prompt templates:
- [x] Use a short grounded prompt when `query_route=GROUNDED`; do not include weak chunks.
- [x] Use a short general-reasoning prompt when `query_route=GENERAL_REASONING`; explicitly state that the loaded document did not provide strong support.
- [x] Keep unrelated refusals as direct app responses without an LLM call.
- [x] Remove redundant instruction text between `ChatViewModel` prompt construction and `LlamaEngine.buildChatPrompt()`.
- [x] Keep the Android system prompt short while preserving ASCII-only wording in `LlamaEngine`.
- [x] Preserve the unrelated-query gate; it prevents unnecessary LLM calls and is more valuable than micro-optimizing generation.
- [ ] Do not reduce `num_predict`/`maxTokens` as the first TTFT fix; it mainly reduces full response time.

Remaining Phase 6 risks:
- Answer quality still depends on the small model; worked-example prompts now explicitly ask for derivative -> integrate-back -> check structure, but more device samples are needed.
- The strong semantic match threshold is intentionally general, not subject-specific; tune from logs if unrelated queries begin grounding too often.

### Phase 7: Runtime / Library Experiments
Goal: test whether newer Llamatik/runtime features improve TTFT, memory, or follow-up performance.

Exit gate: only adopt a runtime change if it improves metrics on target hardware and does not regress stability.

- [ ] Upgrade-test Llamatik in a separate branch:
    - current: `com.llamatik:library-android:0.18.0`
    - candidates: `0.18.2`, `0.19.0`, `1.0.0`
- [ ] Re-test `sessionReset()`, `sessionSave()`, `sessionLoad()`, and `generateContinue()` after upgrade.
- [ ] If stable, use KV/session APIs for follow-up questions to avoid rebuilding the full conversation prompt.
- [ ] Add `LlamaBridge.updateGenerateParams` experiments before `initGenerateModel`:
    - `contextLength`
    - `numThreads`
    - `useMmap`
    - `flashAttention`
    - `batchSize`
    - `maxTokens`
- [ ] Treat `maxTokens` as total-latency tuning, not primary TTFT tuning.

### Phase 8: Model Selection
Goal: choose the fastest model that still behaves like a useful grounded tutor.

Exit gate: selected model meets answer quality, refusal behavior, peak memory, and TTFT targets on the Samsung SM-A047F.

Candidate set:
- [ ] `LFM2.5-350M-Q4_K_M`
- [ ] `LFM2.5-350M-Q4_0`
- [ ] `Granite-4.0-350M-H-Q4_K_M`
- [ ] `Granite-4.0-350M-H-Q4_0`
- [ ] `LFM2-350M-Math-Q4_K_M`
- [ ] `Gemma-3-270M` via Llamatik if stable
- [ ] `Gemma-3-270M` via MediaPipe if the `.task` workflow proves faster or more stable

Selection rule:
- Prefer the smallest model that meets answer quality and refusal behavior.
- Prefer lower TTFT over higher long-answer quality for the first pilot, as long as the answer is grounded and useful.
- Compare quantization variants directly; `Q4_K_M` is not guaranteed to beat `Q4_0` on TTFT.

### Phase 9: Python / Ollama Benchmark Cleanup
Goal: make prototype benchmarks useful for model screening without confusing them with Android measurements.

Exit gate: reports include retrieval time, first token time, total response time, tokens/sec, and pass/fail score.

- [ ] Add TTFT timing to `tests/eval_llm_models.py`; current timing is full-turn latency.
- [ ] Add streaming first-chunk timestamp to `tests/eval_local_gguf.py`.
- [ ] Add retrieval-only warm-up before measured LLM turns.
- [ ] Add per-model Ollama warm-up request before measured turns.
- [ ] Use Ollama `keep_alive` for prototype benchmarking so model load does not pollute hot TTFT.

### Phase 10: Retrieval Storage / Scale
Goal: improve memory behavior and large-document scalability after TTFT evidence justifies the migration.

Exit gate: retrieval is proven to be a meaningful latency or memory bottleneck.

- [ ] Migrate from `FlatIndex` to `sqlite-vec` for disk-based vector search.
- [ ] Add `requery` dependency and `.so` binaries to `jniLibs`.
- [ ] Compare retrieval latency and memory against `FlatIndex`.
- [ ] Do not migrate `FlatIndex` solely for TTFT until measurements show retrieval is significant.

### Phase 11: Product UI
Goal: complete the student-facing app experience.

- [ ] Library screen (document list).
- [ ] Upload screen (file picker + progress).
- [ ] Chat screen improvements.
- [ ] Source attribution polish.
- [ ] Scanned PDF handling UX.
- [ ] Error states for missing model/assets and low memory.

### Phase 12: Device Testing + Pilot
Goal: validate the app under real target-device constraints.

- [ ] Device testing on Samsung SM-A047F.
- [ ] Confirm target < 30s visible TTFT.
- [ ] Confirm active RAM target < 600 MB when possible.
- [ ] Run repeated warm/cold start tests.
- [ ] Run 20-50 pilot student soft launch.

---

## 3. TTFT Notes

### Definition
For this project, TTFT means user-submit to first visible assistant token. Track both:
- **Native TTFT:** `llm.generate()` start to first non-empty `LlamaBridge.generateStream` delta.
- **Visible TTFT:** user-submit to first UI text update in Compose.

The Android app is the main performance surface. The active path is:
`MainActivity` -> `ChatViewModel.ask()` -> query embed -> `FlatIndex.searchWithScores()` -> prompt build -> `LlamaEngine.generate()` -> `LlamaBridge.generateStream()`.

### Prompt / Prefill Principle
TTFT is expected to be dominated by model load and prompt prefill. Cut prompt size before cutting answer length.

### Current Device Findings
- Device: Samsung SM-A047F, Android 14
- Current active model path is stable enough for repeated measurement runs after prompt sanitization.
- Measured visible TTFT is far above target, even after warm-up.
- The dominant cost is native first-token latency, not retrieval, embedding, or Compose flushing.

### Things To Cut Later, Not First
- Do not remove source attribution unless it materially affects UI cost.
- Do not migrate `FlatIndex` solely for TTFT until measurements show retrieval is significant.
- Do not reduce `num_predict`/`maxTokens` as the first TTFT fix; it mainly reduces full response time.
- Do not remove warm-up UX; improve its timing and observability instead.

### Research Notes
- Ollama supports preloading and `keep_alive` on `/api/generate` and `/api/chat`; use this for Python prototype benchmarks.
- Llamatik documentation now includes generation parameters and KV/session APIs; upgrade-test before relying on them because this repo notes `sessionReset()` issues on 0.18.0.
- Google MediaPipe LLM Inference remains a viable Plan B for Gemma-family `.task` models, especially if Llamatik upgrades regress on the target device.

---

## 4. Developer Guide

### Build Instructions (Quick Start)
**App: android-ltk (Llamatik CPU)**
1. **Open** `android-ltk/` in Android Studio.
2. **Sync Gradle** and ensure model assets are present in `app/src/main/assets/`. `python scripts/export_onnx.py` copies `arctic.onnx` and `vocab.txt`; copy or download the `.gguf` manually.
3. **Run** on physical device or API 34+ emulator.

### Maintenance & Utility
- **Check Repo Hygiene:** `pwsh -File scripts/check_repo_hygiene.ps1`
- **Re-export Embedding:** `python scripts/export_onnx.py`
- **Build APK (CLI):** `cd android-ltk; .\gradlew assembleDebug`
- **Install via ADB:** `adb install app/build/outputs/apk/debug/app-debug.apk`
- **Monitoring:** Filter Logcat by `LlamaEngine` to monitor initialization and OOM crashes.

### Troubleshooting & "Plan B"
- **Llamatik JNI UTF-8 Crash:** If the app aborts during `generateStream()` with `NewStringUTF`, keep the prompt sanitization path enabled and treat newer Llamatik versions as an upgrade experiment, not an immediate fix.
- **OOM (Out of Memory) Issues:** If LFM2.5 still crashes on low-end hardware:
  1. Swap to Gemma-3-270M: copy the chosen Gemma GGUF from `models/` to assets and update `MODEL_ASSET` in `LlamaEngine.kt`.
  2. Clear Storage: Uninstall the app or "Clear Data" to remove orphaned model copies.
- **Python Tests:** Import failures in `.venv`. Activate environment before running `pytest`.
