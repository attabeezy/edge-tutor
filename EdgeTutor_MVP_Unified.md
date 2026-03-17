# Edge-Tutor: Unified MVP Specification

### Offline RAG Tutoring Assistant for Android
**Kwame Nkrumah University of Science and Technology (KNUST)**
*Unified from all prior project documents — MVP-focused revision, March 2026*

---

## 0. How to Read This Document

This document supersedes all prior files (concept note, proposal, blueprint, slm_options). It
consolidates them into a single, MVP-first specification. Sections marked **[POST-MVP]** describe
valid future directions that are explicitly out of scope for the first release.

---

## 1. Problem Statement

State-of-the-art AI tutoring tools require constant internet access and consume significant mobile
data. At KNUST — and in similar resource-constrained environments across West Africa — connectivity
is inconsistent and data budgets are tight. This makes tools like ChatGPT, Claude, and Khanmigo
effectively unavailable for daily study use.

At the same time, most students already carry Android smartphones. The MVP exploits this: run the
entire tutoring pipeline on-device, with zero data cost, zero cloud dependency, and no ongoing fees.

---

## 2. MVP Scope Decision

The project has accumulated several competing ambitions across its documents. This section locks in
what the MVP is — and explicitly defers everything else.

### IN SCOPE for MVP

| Decision | Choice | Rationale |
|---|---|---|
| Primary platform | Android (standalone on-device) | Where students already are; no shared hardware needed |
| Core technique | RAG over student-uploaded documents | Buildable now; no training infrastructure required |
| Content source | Student uploads their own PDFs / TXT | Curriculum-agnostic; zero faculty coordination needed to launch |
| Generative model | Pretrained quantized SLM (off the shelf) | Ship fast; distillation deferred to v2 |
| Response mode | Full RAG (retrieval + generative) | More useful than retrieval-only; feasible within RAM budget |
| UI | Native Android (Kotlin) or React Native | Single platform for MVP; cross-platform later |

### EXPLICITLY DEFERRED [POST-MVP]

- **Knowledge distillation / custom student model training** — high value, high effort; requires GPU
  compute, training infrastructure, and curated datasets. Target for v2.
- **Server / Wi-Fi hotspot mode** — valid use case, but adds deployment complexity. Revisit after
  standalone mode is validated.
- **Centralized KNUST syllabus dataset** — requires faculty coordination. The architecture supports
  it when ready; the MVP does not depend on it.
- **Gamified retention / spaced repetition** — useful feature, out of scope for v1.
- **DOCX / PPTX ingestion** — PDF + TXT covers the majority of student materials. Extend later.
- **iOS support** — Android first.

---

## 3. Target User & Device Profile

**User:** First-year engineering student at KNUST or similar institution.

**Device baseline (budget Android):**

| Spec | Minimum Target | Recommended |
|---|---|---|
| Free RAM available to app | ~1 GB | ~1.5 GB |
| Storage for app + models | 2 GB | 4 GB |
| Processor | ARM Cortex-A55 or better | Quad-core 1.8 GHz+ |
| Android version | Android 10 (API 29) | Android 12+ |
| Inference speed | ~1 token/sec | ~3–5 tokens/sec |

> **Design constraint:** Every architecture decision must be stress-tested against the minimum
> device spec. If it doesn't run on a ₵500 ($40) Android phone with 3 GB total RAM, it is not
> in scope for MVP.

---

## 4. Target Curriculum

MVP targets first-year engineering courses, which are broadly standardized:

- **Engineering Mathematics:** Calculus, linear algebra, differential equations, numerical methods
- **Basic Sciences:** Engineering physics (mechanics, waves, electromagnetism), chemistry
- **Introductory Engineering:** Statics, thermodynamics, intro electrical systems
- **Technical Skills:** Python programming fundamentals, technical writing

Because content comes from student uploads, no curriculum integration work is required at launch.
KNUST-specific course alignment (syllabi, past papers, course codes) is a post-MVP initiative.

---

## 5. System Architecture

### 5.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       USER DEVICE                           │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  INGESTION (one-time, on upload)                     │   │
│  │                                                      │   │
│  │  PDF / TXT file                                      │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  Parse → Clean → Chunk (256–512 tokens, 50 overlap)  │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  Embedding Model (all-MiniLM-L6-v2, 22M params)      │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  Local Vector Store (FAISS)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  QUERY (per student question)                        │   │
│  │                                                      │   │
│  │  Student question                                    │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  Embed question → FAISS similarity search            │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  Top-3 to 5 relevant chunks                          │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  [chunks + question] → Generative SLM               │   │
│  │       │                                              │   │
│  │       ▼                                              │   │
│  │  Tutoring response (streamed to UI)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Key Design Principle

> **Retrieval quality is the highest-leverage investment in this architecture.**
> The generative model's job is to reason over retrieved text, not recall knowledge from its
> parameters. A 270M model with excellent retrieval will outperform a 1B model with poor retrieval.
> Prioritize chunking strategy and embedding quality above model size.

---

## 6. Model Selection (MVP)

### 6.1 Recommended Configuration (MVP Default)

> **Qwen2.5-0.5B-Instruct + all-MiniLM-L6-v2**

| Component | Model | Parameters | RAM (4-bit) |
|---|---|---|---|
| Generative SLM | Qwen2.5-0.5B-Instruct | 500M | ~0.3 GB |
| Embedding model | all-MiniLM-L6-v2 | 22M | ~0.08 GB |
| Vector store (FAISS) | — | — | ~0.05 GB |
| **Total** | | | **~0.43 GB** |

**Why this configuration:**
- Combined footprint leaves ~0.5+ GB headroom on a 1 GB budget — enough for Android OS overhead.
- Qwen2.5's **128k context window** is a significant advantage for RAG: more retrieved chunks per
  query, longer documents handled gracefully.
- Strong instruction-following and multilingual support (29 languages) — relevant for a multilingual
  student population.
- all-MiniLM-L6-v2 is the most proven lightweight embedding model available; well-documented,
  widely deployed.

### 6.2 Fallback Configuration (Ultra-Low-End Devices)

> **Gemma 3 270M + all-MiniLM-L6-v2**

| Component | Model | RAM (4-bit) |
|---|---|---|
| Generative SLM | Gemma 3 270M | ~0.2 GB |
| Embedding + FAISS | all-MiniLM-L6-v2 | ~0.13 GB |
| **Total** | | **~0.33 GB** |

Use this configuration when the default exceeds available RAM. The app should detect free RAM at
launch and automatically select the appropriate model tier.

### 6.3 Models Removed from Scope (MVP)

The following models appeared in earlier documents but are not suitable for the 1 GB Android target
and are removed from MVP consideration:

| Model | Why Removed |
|---|---|
| Phi-2 (2.7B) | ~1.5 GB at 4-bit — exceeds entire RAM budget |
| TinyLlama (1.1B) | ~0.7 GB — marginal fit; weaker than Qwen2.5-0.5B |
| Gemma 2B as embedder | ~1.5 GB — embedding model alone exceeds budget |
| MobileBERT | Limited to classification/QA; no generative capability |

### 6.4 Deployment Framework

**Primary:** `llama.cpp` via **Llamatik** (Kotlin wrapper for Android). Most mature GGUF inference
runtime; best community support; runs Qwen2.5 and Gemma models.

**Alternative:** MediaPipe LLM Inference API (Google-native, best for Gemma; simpler integration
but less model flexibility).

---

## 7. Document Ingestion Pipeline

### 7.1 Supported Formats (MVP)

- PDF (textbook chapters, lecture slides exported as PDF, past papers)
- Plain text / Markdown (typed notes, study guides)

### 7.2 Processing Steps

1. **Parse** — Extract raw text. Use a lightweight Kotlin/Java PDF library (e.g., PdfBox Android or
   iTextG) for on-device parsing.
2. **Clean** — Strip headers, footers, page numbers, and OCR artifacts.
3. **Chunk** — Split into overlapping segments:
   - Target: **256–512 tokens per chunk**
   - Overlap: **50 tokens**
   - Prefer natural split points: paragraph breaks, section headings, numbered steps.
4. **Embed** — Pass each chunk through all-MiniLM-L6-v2 to produce a 384-dimensional vector.
5. **Index** — Store vectors in a local FAISS flat index. Persist to device storage so ingestion
   only runs once per document.

### 7.3 Chunking Guidance

| Chunk size | Problem |
|---|---|
| Too small (< 100 tokens) | Retriever returns disconnected fragments; poor context |
| Too large (> 600 tokens) | Embedding averages over too much content; relevance diluted |
| No overlap | Information at chunk boundaries is systematically lost |

Start at 400 tokens / 50 overlap and tune based on retrieval quality testing.

---

## 8. Query Pipeline

When a student submits a question:

1. Embed the question using all-MiniLM-L6-v2.
2. Run FAISS similarity search → retrieve top-**3 to 5** most relevant chunks.
3. Build a prompt using the template below.
4. Pass to the generative SLM. Stream tokens to the UI.

### 8.1 System Prompt Template (MVP)

```
You are Edge-Tutor, an offline AI tutor for engineering students.
You only answer questions based on the context passages provided.
If the context does not contain enough information to answer, say so clearly.
Guide the student with hints and step-by-step reasoning rather than just
giving the final answer. Keep responses concise and focused.

Context:
[CHUNK 1]
---
[CHUNK 2]
---
[CHUNK 3]

Student question: [QUESTION]
```

> **Socratic coaching note:** The prompt instructs the model to guide rather than give direct
> answers. This is the MVP implementation of the Socratic coaching feature described in earlier
> documents — no additional engineering required at this stage.

---

## 9. Android App Structure

### 9.1 Screens (MVP)

1. **Home / Library** — List of uploaded documents with ingestion status.
2. **Upload** — File picker for PDF / TXT. Triggers ingestion pipeline. Progress indicator.
3. **Chat** — Conversational interface. Text input, streaming response, source passage attribution
   (show which chunk the answer came from).
4. **Settings** — Model tier selection (auto / manual), clear index, storage usage.

### 9.2 UX Constraints

- Show a **"thinking" indicator** during retrieval + generation. At 1–3 tokens/sec, users need
  feedback that the app is working.
- Display **source attribution** per response: "Based on: [document name], page ~X." This builds
  trust and helps students verify answers.
- Ingestion should run in a **background service** so students can continue using the app.
- First-launch experience should include a short onboarding that explains uploading materials.

### 9.3 Technology Stack

| Layer | Tool |
|---|---|
| Language | Kotlin (native Android) |
| LLM inference | llama.cpp via Llamatik |
| Embedding | all-MiniLM-L6-v2 (ONNX Runtime Mobile) |
| Vector store | FAISS (JNI bindings or Java port) |
| PDF parsing | PdfBox Android / iTextG |
| UI | Jetpack Compose |
| Local persistence | Room database (document metadata) + flat files (FAISS index) |

---

## 10. MVP Implementation Roadmap

The earlier documents proposed a 12-month, 5-phase roadmap that included distillation training.
The MVP roadmap below is scoped to 10–12 weeks for a small team (1–2 developers).

### Week 1–2: Ingestion Pipeline (Python prototype)

- Build parse → clean → chunk → embed → FAISS index pipeline in Python.
- Test chunking strategies on real KNUST lecture notes and textbooks.
- Validate retrieval quality: given a known question, does the right chunk rank in top-3?
- **Exit criterion:** Top-3 retrieval precision > 70% on a hand-labeled test set of 50 questions.

### Week 3–4: RAG Pipeline (Python prototype)

- Wire Qwen2.5-0.5B-Instruct (via llama.cpp / Ollama locally) into the retrieval pipeline.
- Test full question → retrieve → generate flow on engineering subject material.
- Tune chunk size, top-K, and system prompt.
- **Exit criterion:** Responses are factually grounded and coherent for math, physics, and
  chemistry questions sourced from uploaded materials.

### Week 5–7: Android Port

- Port ingestion pipeline to Kotlin (PDF parsing, chunking, embedding via ONNX Runtime Mobile).
- Integrate Llamatik for on-device LLM inference.
- Set up FAISS on Android (JNI bindings).
- **Exit criterion:** Full pipeline runs end-to-end on a physical Android device without crashing.

### Week 8–9: UI & UX

- Build Jetpack Compose screens: Library, Upload, Chat, Settings.
- Implement streaming token output, source attribution, and background ingestion.
- Test on 2–3 device tiers (budget, mid-range, high-end).
- **Exit criterion:** App is usable by a non-technical student without assistance.

### Week 10–11: Internal Testing & Tuning

- Test across a range of Android devices (target the minimum spec).
- Measure RAM usage, inference speed, and crash rate.
- Collect feedback from 5–10 engineering students in informal sessions.
- Fix critical bugs; tune retrieval and prompt.

### Week 12: Soft Launch

- Release APK to a group of 20–50 first-year engineering students.
- Collect structured feedback on response quality and usability.
- Document known issues and v2 priorities.

---

## 11. Success Metrics (MVP)

| Metric | Target |
|---|---|
| Runs on minimum-spec device (1 GB free RAM) | Yes, no OOM crashes |
| First response latency | < 30 seconds end-to-end |
| Retrieval precision (top-3) | > 70% on test set |
| User satisfaction (pilot survey) | > 60% rate responses "helpful" or "very helpful" |
| App stability (crash-free sessions) | > 90% of sessions |
| Average session length | > 3 questions per session |

---

## 12. Budget (MVP Only)

The full proposal budget assumed distillation training and a large team. The MVP requires neither.

| Category | MVP Estimate (USD) |
|---|---|
| Developer time (1–2 developers, ~12 weeks) | $3,000 – $8,000 |
| Test devices (3–4 Android phones, various tiers) | $400 – $800 |
| API credits for prototype testing / evaluation | $100 – $300 |
| Contingency | $500 – $1,000 |
| **Total MVP** | **$4,000 – $10,100** |

> The $30,000–$48,500 budget from the proposal remains valid for the full multi-phase roadmap
> including distillation training, faculty collaboration, and institutional rollout. The MVP
> validates the core product before committing that spend.

---

## 13. Post-MVP Roadmap [POST-MVP]

Once the MVP is validated with real students, the following phases become unblocked:

### v2: Knowledge Distillation

- Use Claude / DeepSeek-R1 as teacher model to generate 10,000+ KNUST-specific educational
  dialogues (question-answer pairs, worked solutions, Socratic exchanges).
- Fine-tune Qwen2.5-0.5B or Gemma 3 270M on this dataset.
- This produces a model that "knows" engineering content from parameters, reducing reliance on
  perfect retrieval for common questions.

### v2: KNUST Syllabus Integration

- Work with faculty to curate a centralized content bundle (syllabi, past papers, reference
  material) that ships with the app.
- Students still upload personal notes on top of this base.

### v3: Server / Hotspot Mode

- Deploy a larger model (Gemma 3 1B, Phi-2) on a laptop or Raspberry Pi.
- Broadcast a local Wi-Fi network; students connect with a thin client.
- Useful for study groups and classroom settings.

### v3: Gamified Retention

- Spaced repetition flashcards auto-generated from uploaded materials.
- Active recall quizzes tied to the RAG knowledge base.

### Future: Multi-language Support

- Qwen2.5-0.5B already supports 29 languages.
- Prioritize Twi and other local Ghanaian languages based on student demand.

---

## 14. Risk Register (MVP-Scoped)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OOM crashes on low-end devices | Medium | High | Implement RAM detection at startup; auto-select model tier; offer retrieval-only fallback |
| Retrieval returns irrelevant chunks | Medium | High | Tune chunk size and overlap; test on real student materials early (Week 1) |
| Slow inference frustrates users | High | Medium | Streaming output + "thinking" indicator; set clear expectations in onboarding |
| PDF parsing fails on scanned documents | Medium | Medium | Warn user if text extraction yields < 100 words/page; advise digitally-typed PDFs for MVP |
| Low adoption among students | Low (if quality is good) | High | Faculty endorsement; keep onboarding < 2 minutes; demo at orientation |
| Academic integrity concerns | Low | Medium | Frame as study aid, not answer machine; Socratic prompt design |

---

## 15. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — retrieve relevant document chunks, then generate a response grounded in that context. |
| **SLM** | Small Language Model — a generative model under ~1B parameters, designed for on-device use. |
| **Embedding** | A dense vector representation of text, enabling semantic similarity comparisons. |
| **FAISS** | Facebook AI Similarity Search — a high-performance library for vector similarity search. |
| **GGUF** | File format for quantized language models; used by llama.cpp for efficient CPU inference. |
| **Quantization** | Reducing model weight precision (32-bit → 4-bit) to shrink memory usage with minimal quality loss. |
| **Chunking** | Splitting documents into fixed-size overlapping segments before embedding. |
| **Top-K Retrieval** | Returning the K most semantically similar document chunks to a given query. |
| **Knowledge Distillation** | [Post-MVP] Training a small model to mimic a large teacher model's reasoning. Deferred to v2. |
| **Llamatik** | A Kotlin-first Android library wrapping llama.cpp for on-device LLM inference. |
| **ONNX Runtime Mobile** | Cross-platform inference runtime; used here for the embedding model on Android. |

---

*Edge-Tutor MVP — Android + RAG, first-year engineering, KNUST*
*Supersedes: Edge-Tutor Concept Note, Edge_Tutor_Proposal.docx, EdgeTutor_Technical_Blueprint.md, slm_options.md*
