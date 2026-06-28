# Qwen3.5 0.8B / 1x800 Validation Assessment

Run date: June 27, 2026

Source: `query-validation.csv`

## Rubric

Each dimension is scored from 0 to 2:

- Correctness: factually and mathematically correct.
- Grounding: uses the textbook when supported, or clearly identifies that the
  textbook does not support the answer.
- Relevance: directly fulfills the requested task.
- Clarity: understandable, concise, and internally coherent.

Blank outputs score zero in every dimension. For unsupported and non-academic
cases, grounding includes the required general-answer attribution behavior.

## Scores

| Case | Category | C | G | R | Cl | Assessment |
|---|---|---:|---:|---:|---:|---|
| g1 | Grounded | 1 | 2 | 2 | 1 | Broadly grounded, but imprecise and includes meta-output. |
| g2 | Grounded | 1 | 1 | 1 | 1 | Partly correct; the retrieved excerpt does not directly support the definition. |
| g3 | Grounded | 1 | 1 | 1 | 0 | Core idea is present, but the answer is verbose, hedged, and not simple. |
| g4 | Grounded | 0 | 0 | 1 | 0 | States an incorrect derivative before later giving a conflicting result. |
| f1 | Follow-up | 0 | 0 | 0 | 0 | Unrelated and mathematically incoherent. |
| f2 | Follow-up | 0 | 0 | 0 | 1 | A clear refusal that does not answer the follow-up. |
| f3 | Follow-up | 0 | 0 | 0 | 0 | No visible answer. |
| f4 | Follow-up | 0 | 0 | 0 | 0 | No visible answer. |
| ua1 | Unsupported academic | 2 | 2 | 2 | 2 | Correctly states that the textbook does not cover the topic. |
| ua2 | Unsupported academic | 0 | 0 | 1 | 1 | Omits attribution and gives an incorrectly balanced equation. |
| ua3 | Unsupported academic | 0 | 0 | 0 | 0 | No visible answer. |
| ua4 | Unsupported academic | 0 | 0 | 1 | 1 | Omits attribution and gives a false answer. |
| na1 | Non-academic | 0 | 0 | 0 | 0 | Hallucinates unsafe and irrelevant bread instructions. |
| na2 | Non-academic | 0 | 0 | 0 | 0 | No visible answer. |
| na3 | Non-academic | 0 | 0 | 1 | 0 | Mixes irrelevant textbook claims with a late unsupported-topic statement. |
| na4 | Non-academic | 1 | 0 | 1 | 1 | Attempts the task but invents textbook attribution and is barely coherent. |
| **Total** |  | **6/32** | **6/32** | **11/32** | **8/32** | **31/128 (24.2%)** |

## Outcome

This run fails the answer-quality gate.

- Four of 16 cases produced no visible answer.
- Only one of eight unsupported/non-academic cases followed the intended
  unsupported-topic behavior.
- All four follow-up cases failed to provide a useful contextual answer.
- One grounded answer contains a basic derivative error.
- The bread answer is unsafe enough to block release without routing changes.

## Context-isolation audit

No answer contains a clear reference to a preceding validation case. The
observed failures are consistent with irrelevant retrieval, permissive routing,
and weak generation rather than cross-case KV-cache leakage.

The code provides two isolation boundaries:

1. Native `submitMessages` and `submitMessagesWithImage` reset the MNN session
   before every generation.
2. The validation runner explicitly resets the native session and clears the
   Kotlin conversation before every case. Follow-up setup and query pairs retain
   context only by reconstructing it in the prompt.

This confirms the isolation mechanism, but the run is not a controlled
behavioral leakage test. A future device suite should add unique sentinel facts
to adjacent independent cases and assert that the second answer does not repeat
the first sentinel.

## Next engineering gate

Do not spend the next iteration tuning the `1x800` budget. Fix routing first:
common lexical overlap currently marks unrelated questions as document-supported
and attaches irrelevant sources. After routing is deterministic, rerun these
same 16 cases and compare against this baseline.

## Qwen3.5 thinking-budget finding

A June 28 Python/Ollama smoke test reproduced blank visible output when
`qwen3.5:0.8b` had thinking enabled and a limited generation budget. The model
used the entire 128-token budget for hidden reasoning and emitted no answer.
With thinking explicitly disabled, the same prompt returned the expected answer
in two tokens.

The Android path now forces thinking off at initialization and before every
query, verifies the effective merged native configuration, removes the
persistent thinking toggle, and rejects hidden-only output as an explicit
failure. Blank-output device cases must be rerun on target hardware to confirm
the fix against the packaged MNN libraries.
