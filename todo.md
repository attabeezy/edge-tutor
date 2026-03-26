# EdgeTutor — Phase 3 TODO

## Status

| Phase | Status |
|---|---|
| Phase 1 — Ingestion pipeline (Python) | Complete |
| Phase 2 — RAG pipeline (Python) | **Complete** ✓ |
| Phase 3 — Android port | **Complete** ✓ |
| Phase 4 — UI/UX (Jetpack Compose) | Not started |
| Phase 5 — Device testing & tuning | Not started |
| Phase 6 — Soft launch | Not started |

---

## Before Starting Phase 3

These housekeeping tasks from Phase 2 are still open:

- [x] **Manual response quality review** — 12/12 retrieval precision, responses reviewed and accepted. Q7 can be verbose on broad conceptual questions (known small-LLM behaviour, not a blocker).

- [x] **`requirements.txt`** — now delegates to `requirements-phase1.txt` via `-r`.

- [x] **Update `CLAUDE.md`** — default embedding updated to `minilm`; arctic added to model table.

---

## Phase 3 — Android Port

Exit criterion: full pipeline runs end-to-end on a physical Android device without crashing.

### Step 1 — Export the embedding model to ONNX (PC, one-time)

- [x] Install the Python validation dep (`onnxruntime`)
- [x] Run `python scripts/export_onnx.py` — produced `data/models/minilm.onnx` and `data/models/vocab.txt`

### Housekeeping (while Gradle syncs)

- [x] **`requirements.txt`** — pinned `ollama==0.6.1` and `httpx==0.28.1`; added onnx/onnxruntime deps from export step.
- [x] **`CLAUDE.md`** — added `-m MODEL` flag to REPL docs; documented two-gate out-of-scope filter in `query.py` module description.
- [x] **`query.py` gate review** — lexical overlap gate (`MIN_LEXICAL_OVERLAP=2`) looks safe; short questions with ≤1 content word automatically reduce to requiring 1 match.
- [x] **Project memory saved** — status and key decisions written to `.claude/projects/.../memory/`.

---

### Step 2 — Open the Android project in Android Studio

- [x] Open `edgetutor-android/` in Android Studio (File → Open).
- [x] Gradle sync complete (downloaded ~500 MB tooling + Gradle 9.2.1).
- [x] SDK Build-Tools 36 licence accepted and installed automatically.
- [x] Project compiles clean — `BUILD SUCCESSFUL in 23m 29s`. APK at `app/build/outputs/apk/debug/app-debug.apk`.

### Step 3 — Wire up Llamatik (LLM inference)

- [x] Llamatik `com.llamatik:library-android:0.11.0` wired up via `LlamaEngine.kt`
- [x] GGUF model downloaded and placed in assets
- [x] Correct API confirmed from AAR bytecode: `LlamaBridge` in `com.llamatik.library.platform`, `GenStream` interface with `onDelta`/`onComplete`/`onError`
- [x] Room upgraded to 2.7.0 for Kotlin 2.2.x compatibility

### Step 4 — Smoke test on a physical device

- [x] App launched successfully on physical device
- [x] Ingested 3 PDFs — no OOM or crash
- [x] Questions answered correctly end-to-end

---

## Phase 4 — UI/UX (Weeks 8–9)

Replace the smoke-test `EdgeTutorApp()` composable in `MainActivity.kt` with proper screens:

- [ ] **Library screen** — document list with ingestion status badge (pending / indexing / ready / error)
- [ ] **Upload screen** — file picker (PDF/TXT), progress bar, background ingestion trigger
- [ ] **Chat screen** — streamed token output, "thinking" indicator, source attribution ("Based on: *CalculusMadeEasy*, ~p.42")
- [ ] **Settings screen** — model tier selection (auto / manual), clear index, storage usage display
- [ ] **Onboarding flow** — first-launch walkthrough, target < 2 minutes to complete

---

## Phase 5 — Device Testing (Weeks 10–11)

- [ ] Test on minimum-spec device (1 GB free RAM, ARM Cortex-A55)
- [ ] Measure peak RAM, first-response latency (target < 30 s), crash-free rate (target > 90 %)
- [ ] Run informal sessions with 5–10 first-year engineering students
- [ ] Fix critical bugs; tune chunk size and retrieval prompt based on real usage

---

## Phase 6 — Soft Launch (Week 12)

- [ ] Release APK to 20–50 pilot students
- [ ] Collect structured feedback (target: > 60 % rate responses "helpful")
- [ ] Document known issues and v2 priorities
