# EdgeTutor Project Status

Last updated: June 28, 2026

## Product Direction

The primary and only product application is `android-mnn/`.

- Runtime: MNN-LLM
- Generator: Qwen3.5-0.8B-MNN
- Embedder: quantized Arctic Embed XS ONNX
- Target device: Samsung SM-A047F-class Android hardware
- Default context policy: one retrieved chunk capped at 800 characters

The historical Python/Ollama prototype and Llamatik/GGUF evaluation path are
not part of the active product trajectory.

## Current Architecture

1. Import and ingest a PDF on device.
2. Extract text in adaptive page windows.
3. Create overlapping 400-word chunks.
4. Embed chunks with Arctic and store normalized vectors in `FlatIndex`.
5. Rewrite contextual follow-ups from the most recent source-backed exchange.
6. Retrieve the best passage and include it in every generation prompt.
7. Qwen emits `[TEXTBOOK]` or `[GENERAL]` before its answer.
8. Android removes the marker while streaming and controls source attribution:
   textbook answers show sources; general answers receive a visible warning.
9. Invalid or absent markers fail closed to the labelled general route.

The main UI has three bottom destinations:

- Chat: direct conversation, attachments, and source-backed history.
- Library: multiple retained textbooks, independent RAG indexes, active
  textbook selection, deletion, and live page/chunk/index status.
- Settings: model import and debug validation/benchmark controls.

Retrieval similarity is diagnostic only. Local testing showed that mean top-five
similarity cannot meet both grounded-recall and false-grounding requirements,
even for *Calculus Made Easy*.

## Validated Baseline

Previously validated on Samsung SM-A047F:

- Complete PDF ingestion and local index persistence.
- MNN model initialization and warm-up.
- Grounded retrieval, prompt construction, streaming, and metrics.
- Selected `1x800` policy with approximately 18.6-second median warm visible
  first-token latency in the June 27 prompt benchmark.
- Android unit tests and repository hygiene checks.

The new model-marker routing path is unit tested but still requires a physical
device rerun.

## Required Device Gate

Run the fixed 16-case suite and require:

- 16/16 completed generations.
- 16/16 valid route markers.
- Grounded and supported follow-up cases use `TEXTBOOK`.
- Unsupported-academic and non-academic cases use `GENERAL`.
- Sources appear only on `TEXTBOOK` answers.
- Markers never appear in visible answer text.
- Malformed-marker output receives the general warning and no source.
- No context bleed between independent cases.
- Record visible TTFT, total time, memory, prompt tokens, and answer quality.

## Active Risks

- Qwen3.5-0.8B now owns the evidence decision; its route accuracy is not yet
  device validated.
- General answers can be factually wrong and must remain visibly labelled.
- Native prompt prefill dominates low-end-device latency.
- Python extraction and performance measurements are not Android parity.

## Immediate Next Steps

1. Install the updated Android build on Samsung SM-A047F.
2. Run and score the fixed 16-case suite.
3. Inspect marker compliance and source attribution before answer quality.
4. If marker compliance is poor, tighten the prompt or add constrained decoding;
   do not restore similarity thresholds.
5. Improve the first-token waiting experience after routing is validated.

## Maintenance Rules

- Product behavior belongs in `android-mnn/`.
- Python may support ONNX export and parity evaluation only.
- Performance claims must name model, device, prompt policy, and date.
- Do not promote host-side routing results as Android results.
- Run `scripts/check_repo_hygiene.ps1` before pushing.
