# Python Query-Mode Assessment

Run date: June 28, 2026

Model: `qwen3.5:0.8b` through Ollama, with thinking explicitly disabled.

Document: `CalculusMadeEasy`

## Runs

| Mode | Completed | Visible answers | Runtime errors | Mean case time |
|---|---:|---:|---:|---:|
| Forced RAG | 16/16 | 16/16 | 0 | 32.22s |
| General | 16/16 | 16/16 | 0 | 14.80s |
| Experimental Auto | 16/16 | 15/16 | 1 | 40.06s |

Follow-up case time includes its setup query. Auto's single blank (`g4`) was an
Ollama HTTP 500 caused by a reset local TCP connection, not a completed
generation with hidden-only output.

Raw reports:

- `query-modes-rag/query-mode-results.csv`
- `query-modes-general/query-mode-results.csv`
- `query-modes/query-mode-results.csv`

## Findings

### Thinking configuration

Disabling thinking removed the hidden-reasoning token-budget failure seen in the
smoke test. Forced RAG and General produced no blank answers. This strengthens
the case for verifying the effective Android native thinking configuration
before treating the device blanks as a prompt-budget failure.

### General mode

Retrieval-free generation is technically viable and materially faster in this
host-side test. It handled lightweight tasks such as a birthday greeting, a
joke, and the capital of Japan without irrelevant textbook excerpts.

It is not factually reliable enough to present as authoritative general
knowledge:

- It attributed *Things Fall Apart* to Charles Dickens instead of Chinua Achebe.
- It incorrectly put Earth between the Sun and Moon for a solar eclipse.
- Several educational explanations contained subtler inaccuracies.
- Follow-up interpretation was unstable; "differentiation" was sometimes
  interpreted biologically rather than mathematically.

### Forced RAG

RAG improved the core calculus answers and usually declined unsupported
academic questions based on absent document evidence. It still produced
irrelevant or nonsensical answers for some non-academic prompts because it was
forced to use unrelated passages.

### Auto routing

Requiring meaningful lexical overlap is better than accepting any shared word,
but it is not sufficient:

- Several grounded calculus questions were handled well.
- Some non-academic tasks correctly reached General.
- *Things Fall Apart* and the capital-of-Japan question still received
  irrelevant document-grounded refusals.
- Route quality remains dependent on accidental words in retrieved passages.

Auto was slower here because it always retrieved before deciding whether to use
the passages. Android latency characteristics will differ, but this policy
cannot provide the speed benefit of an explicit General selection.

## Decision

The experiment supports adding an explicit retrieval-free capability, but not
silently treating its output as trusted factual knowledge.

Recommended Android product shape:

1. Expose explicit `Textbook` and `General` modes.
2. Label General answers as model knowledge and potentially inaccurate.
3. Keep `Auto` experimental until routing is evaluated with a dedicated,
   document-diverse classifier dataset.
4. Disable thinking in the effective MNN configuration and add a regression
   case that fails if the model spends its answer budget on hidden reasoning.
5. Preserve the textbook-only path for grounded tutoring, where the small model
   performed best.
