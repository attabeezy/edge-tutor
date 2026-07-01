# EdgeTutor Tutor Fine-Tuning

This directory contains the 300-row English tutoring dataset and the
Qwen3.5-0.8B Colab training workflow.

## Dataset

- `train.jsonl`: 240 rows
- `validation.jsonl`: 30 rows
- `test.jsonl`: 30 held-out rows

The dataset covers 60 concepts across math, science, English, and social
studies. Each concept has five tutoring moves: diagnostic feedback, an initial
hint, a concept explanation, corrective feedback, and a complete answer after
the learner requests it.

The JSONL files are the source of truth. Validate them from the repository root:

```powershell
python training/tutor/validate_dataset.py
```

## Train and export

Open `colab_qwen35_tutor.ipynb` in a GPU Colab runtime. It:

1. validates the checked-in dataset;
2. downloads `Qwen/Qwen3.5-0.8B`;
3. trains one epoch of HQQ-aware QLoRA;
4. evaluates the base, adapter, and merged models;
5. exports a standalone 4-bit HQQ MNN model.

Training uses LoRA rank 8, alpha 16, dropout 0.05, 1,024-token sequences,
4-bit HQQ weights, and deterministic evaluation.

## Evaluate locally

In an environment with PyTorch, Transformers, and PEFT:

```powershell
python training/tutor/evaluate_tutor.py `
  --model Qwen/Qwen3.5-0.8B `
  --adapter path/to/adapter `
  --input training/tutor/test.jsonl `
  --output reports/tutor-adapter.csv
```

Automatic checks cover route selection, answer leakage, required answer
content, one guiding question, and response length. Manually compare several
base and adapter responses before deploying the model.

