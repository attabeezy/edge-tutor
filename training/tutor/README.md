# EdgeTutor Tutor Fine-Tuning

This directory contains the authored MVP dataset and the reproducible Qwen3.5-0.8B tutoring workflow. No external model or generation API was used to create the examples.

## Dataset

The dataset has 300 rows:

- `train.jsonl`: 240 rows
- `validation.jsonl`: 30 rows
- `test.jsonl`: 30 held-out rows

It contains 75 examples for each of math, science, English/language arts, and social studies. Each of 60 concepts has five tutoring moves: diagnostic question, initial hint, conceptual explanation, corrective feedback, and a complete answer after an attempt or explicit request. Three moves use `[TEXTBOOK]` and two use `[GENERAL]`, for a 180/120 route split.

Each JSONL row contains `messages` for SFT plus metadata used by validation and evaluation. Concepts never cross data splits.

Regenerate and validate from the repository root:

```powershell
python training/tutor/generate_dataset.py
python training/tutor/validate_dataset.py
```

## Train and export

Open `colab_qwen35_tutor.ipynb` in a free GPU Colab runtime. It:

1. validates the dataset;
2. downloads `Qwen/Qwen3.5-0.8B`;
3. trains an MNN HQQ-aware QLoRA adapter;
4. compares base and adapter on the held-out split;
5. exports a merged 4-bit HQQ MNN model.

The quantization parameters match the currently bundled model: HQQ, 4-bit weights, block size 64, 4-bit LM head, and 16-bit scales. The merged export is a drop-in model directory; Android does not need to load a separate LoRA file.

## Evaluate upstream

The notebook invokes the evaluator automatically. It can also be run directly in a CUDA environment:

```powershell
python training/tutor/evaluate_tutor.py `
  --model Qwen/Qwen3.5-0.8B `
  --adapter path/to/adapter `
  --input training/tutor/test.jsonl `
  --output reports/tutor-adapter.csv
```

Automatic checks cover route markers, premature answer fragments, required answer fragments, one guiding question, and response length. The CSV leaves correctness, helpfulness, adaptation, and feedback columns blank for manual 0-2 review.

## Android validation

The Android device suite retains the original 16 routing cases and adds 24 tutoring cases across the four subjects and six behaviors. After replacing the model, rebuild and reinstall the app or clear app data so the old private model copy is not reused.
