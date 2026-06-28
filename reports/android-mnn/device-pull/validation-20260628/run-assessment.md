# Android MNN Validation Run — June 28, 2026

Device: Samsung SM-A047F, Android 14

Model: Qwen3.5 0.8B MNN

Policy: `1x800`; semantic route threshold `0.35`

## Runtime outcome

- 16/16 validation cases completed.
- 16/16 produced visible output.
- 0 query failures.
- 0 hidden-thinking characters.
- The effective native config verified `enable_thinking=false` 21 times,
  including startup, warm-up, setup queries, and validation queries.
- Median visible TTFT: 12.44s.
- Visible TTFT range: 10.27–16.07s.

The thinking-mode fix passes this device run. The earlier blank-output behavior
did not recur.

## Routing outcome

All 16 evaluated cases routed to textbook RAG.

| Category | Arctic cosine range |
|---|---:|
| Grounded | 0.6502–0.7041 |
| Unsupported academic | 0.5713–0.6504 |
| Non-academic | 0.5497–0.6609 |

The `0.35` threshold is too low. More importantly, top-1 cosine alone does not
separate the classes: unsupported `ua3` scored `0.6504`, while grounded `g2`
scored `0.6502`; non-academic `na2` scored `0.6609`.

Do not solve this by raising the threshold. A higher threshold would still
misroute some unrelated cases while rejecting valid grounded cases.

## Answer-quality outcome

The run fails the answer-quality gate despite passing runtime stability:

- `g4` produced a long, mathematically incoherent answer and hit the 600-token
  native cap.
- Follow-up answers were visible but mostly incorrect or irrelevant.
- Unsupported questions received fabricated textbook-grounded answers.
- `ua3` falsely attributed *Things Fall Apart*.
- `ua4` answered a capital-city question with an unrelated equation.
- `na1` produced unsafe, fabricated bread instructions.
- `na2` returned only `*#*`.

## Performance interpretation

TTFT improved materially from the previous approximately 18.6s median to
12.44s. Three setup/grounded generations reached the 600-token limit, producing
roughly 99–106s total latency. The next performance control should cap normal
answers well below 600 tokens; this is separate from the resolved hidden
thinking issue.

## Next routing experiment

Use a second-stage support check after vector retrieval. Candidate approaches
should be evaluated against this exact captured set:

1. Lightweight entailment/relevance classifier over question plus top passage.
2. Embedding features beyond top-1 score, including score distribution and
   document-level calibration.
3. Conservative fallback to General when support is uncertain.

Raw evidence is preserved in `query-validation.csv`, `query-validation.md`, and
`logcat.txt` in this directory.
