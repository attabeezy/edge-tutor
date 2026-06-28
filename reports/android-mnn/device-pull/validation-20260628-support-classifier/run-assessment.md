# Android MNN Support-Classifier Validation — June 28, 2026

Device: Samsung SM-A047F, Android 14

Models: Qwen3.5 0.8B MNN; quantized TinyBERT-L2 MS MARCO ONNX

Policy: `1x800`; semantic threshold `0.35`; provisional support threshold `0.5`

## Runtime outcome

- The unchanged 16/16 validation cases completed without query failures.
- Classifier latency was 82–113 ms, averaging 96 ms.
- All 20 generations, including the four follow-up setup queries, remained
  below the 192-token native cap; the maximum was 136 decoded tokens.
- Evaluated-query total time was 5.01–23.09 seconds, averaging 9.99 seconds.

The classifier integration and reduced output cap pass their runtime checks.

## Routing outcome

The provisional `0.5` threshold conservatively routed all cases to General.
The measured support scores cannot satisfy the perfect-routing acceptance gate
at any single threshold:

| Expected class | Score range |
|---|---:|
| Grounded and follow-up | 0.00002281–0.30295175 |
| Unsupported and non-academic | 0.00001056–0.00007984 |

The classes overlap: follow-up `f2` scored `0.00002281`, below unsupported
`ua3` at `0.00007984`. There is therefore no clean positive/negative interval
from which to select the specified midpoint threshold.

## Decision

Do not tune a threshold against individual cases or claim the perfect-routing
gate passed. TinyBERT-L2 over only the top retrieved passage is fast, but this
configuration does not cleanly separate the captured set. Raw evidence is
preserved in `query-validation.csv`, `query-validation.md`, and `logcat.txt`.
