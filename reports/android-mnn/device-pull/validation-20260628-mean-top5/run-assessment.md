# Android MNN Mean-Top-5 Validation — June 28, 2026

Device: Samsung SM-A047F, Android 14

Model: Qwen3.5 0.8B MNN

Policy: `1x800`; mean-top-5 threshold `0.63165`

## Runtime outcome

- The unchanged 16/16 validation cases completed without query failures.
- All 20 generations, including four follow-up setup queries, respected the
  192-token native cap; the observed maximum was exactly 192.
- Evaluated-query visible TTFT was 3.09–15.40 seconds, averaging 8.60 seconds.
- Evaluated-query total time was 3.98–44.83 seconds, averaging 25.11 seconds.

## Routing outcome

The captured-set routing gate passed:

| Expected category | Grounded | General |
|---|---:|---:|
| Grounded | 4 | 0 |
| Follow-up | 4 | 0 |
| Unsupported academic | 0 | 4 |
| Non-academic | 0 | 4 |

The rerun reproduced the offline separation:

| Expected class | Mean top-5 range |
|---|---:|
| Grounded and follow-up | 0.6360262–0.7000912 |
| Unsupported and non-academic | 0.5220391–0.62730134 |

The nearest cases leave a margin of approximately `0.00872` between the
highest negative and lowest positive score. The selected threshold lies inside
that interval.

## Decision

Mean top-5 similarity passes the exact captured routing set and removes the
cross-encoder's runtime and model footprint. This is an in-sample result:
`0.63165` was calibrated from the same cases, so it must remain experimental
until it passes additional textbooks and hard-negative questions.

Raw evidence is preserved in `query-validation.csv`, `query-validation.md`,
and `logcat.txt`.
