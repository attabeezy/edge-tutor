"""
Export all-MiniLM-L6-v2 to ONNX for ONNX Runtime Mobile (Android).

Outputs
-------
  data/models/minilm.onnx   -- ONNX model (transformer backbone, opset 12)
  data/models/vocab.txt     -- WordPiece vocabulary (30k tokens, one per line)

Then copies both files to edgetutor-android/app/src/main/assets/ so Gradle
picks them up automatically on the next sync / build.

Requirements
------------
  pip install transformers torch onnxruntime numpy
  (onnxruntime is for local validation only; Android uses onnxruntime-android)

Usage
-----
  python scripts/export_onnx.py
  # or from the project root:
  python -m scripts.export_onnx
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
ASSETS_DIR = PROJECT_ROOT / "edgetutor-android" / "app" / "src" / "main" / "assets"
ONNX_PATH = MODELS_DIR / "minilm.onnx"
VOCAB_PATH = MODELS_DIR / "vocab.txt"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

# ---------------------------------------------------------------------------
# 2. Export
# ---------------------------------------------------------------------------
print("Exporting to ONNX (opset 12) ...")
dummy = tokenizer(
    "What is a derivative?",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
)
input_ids = dummy["input_ids"]
attention_mask = dummy["attention_mask"]
token_type_ids = dummy.get("token_type_ids", torch.zeros_like(input_ids))

torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    str(ONNX_PATH),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids":        {0: "batch", 1: "seq"},
        "attention_mask":   {0: "batch", 1: "seq"},
        "token_type_ids":   {0: "batch", 1: "seq"},
        "last_hidden_state":{0: "batch", 1: "seq"},
    },
    opset_version=14,
    do_constant_folding=True,
    dynamo=False,           # force legacy TorchScript exporter (avoids emoji/cp1252 crash on Windows)
)
size_mb = ONNX_PATH.stat().st_size / 1e6
print(f"  -> {ONNX_PATH}  ({size_mb:.1f} MB)")

# ---------------------------------------------------------------------------
# 3. Validate (ONNX vs PyTorch)
# ---------------------------------------------------------------------------
print("Validating ONNX output vs PyTorch ...")
try:
    import onnxruntime as ort
except ImportError:
    print("  WARNING: onnxruntime not installed — skipping validation.")
    print("           pip install onnxruntime  to enable it.")
else:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids":      input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    }
    ort_hidden = sess.run(None, feeds)[0]

    with torch.no_grad():
        pt_hidden = model(**dummy).last_hidden_state.numpy()

    max_diff = float(np.abs(ort_hidden - pt_hidden).max())
    print(f"  Max abs diff: {max_diff:.2e}", end="  ")
    assert max_diff < 1e-4, f"ONNX diverged too much from PyTorch: {max_diff}"
    print("OK")

    # Semantic sanity check via cosine similarity
    def mean_pool_norm(hidden, mask):
        mask_f = mask[:, :, None].astype(np.float32)
        pooled = (hidden * mask_f).sum(1) / np.clip(mask_f.sum(1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, 1e-9, None)

    sentences = [
        "What is a derivative?",
        "Explain the chain rule in calculus.",
        "What is the capital of France?",
    ]
    enc2 = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    ort_h2 = sess.run(None, {
        "input_ids":      enc2["input_ids"].numpy(),
        "attention_mask": enc2["attention_mask"].numpy(),
        "token_type_ids": enc2.get("token_type_ids", torch.zeros_like(enc2["input_ids"])).numpy(),
    })[0]
    embs = mean_pool_norm(ort_h2, enc2["attention_mask"].numpy())
    sims = embs @ embs.T
    print(f"  Cosine sim (calculus-chainrule)={sims[0,1]:.3f}  (calculus-france)={sims[0,2]:.3f}")
    assert sims[0, 1] > sims[0, 2], "Semantic sanity check failed — related sentences should score higher"
    print("  Semantic sanity check OK")

# ---------------------------------------------------------------------------
# 4. Save vocabulary (line N = token ID N)
# ---------------------------------------------------------------------------
print("Saving vocab.txt ...")
sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda kv: kv[1])
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    for token, _ in sorted_vocab:
        f.write(token + "\n")
print(f"  -> {VOCAB_PATH}  ({len(sorted_vocab)} tokens)")

# ---------------------------------------------------------------------------
# 5. Copy to Android assets
# ---------------------------------------------------------------------------
print("Copying to Android assets ...")
for src in [ONNX_PATH, VOCAB_PATH]:
    dst = ASSETS_DIR / src.name
    shutil.copy2(src, dst)
    print(f"  {src.name} -> {dst}")

print("\nAll done.")
print("Next: open edgetutor-android in Android Studio and run Gradle sync.")
