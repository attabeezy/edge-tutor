# Android MNN Validation Evidence

This directory contains durable, device-generated evidence used for runtime and
answer-quality decisions.

## Keep

- Raw CSV reports, because they preserve prompts, answers, timing, memory, and
  source data for later analysis.
- Generated Markdown reports when they make a raw run easy to review.
- A scored assessment for each release-candidate validation run.

## Do not add

- ADB screenshots used only to verify that an activity launched.
- Intermediate UI redesign captures.
- Empty or partial device-command output.

Those local artifacts are covered by repository ignore rules. A screenshot
should be committed only when it is deliberately renamed and referenced as
durable product or validation evidence.

The current scored assessment is
[`device-pull/validation/scored-assessment.md`](device-pull/validation/scored-assessment.md).
