"""
Export Snowflake/snowflake-arctic-embed-xs to ONNX for ONNX Runtime Mobile (Android).

Outputs
-------
  models/arctic.onnx   -- ONNX model (transformer backbone, opset 14)
  models/vocab.txt     -- WordPiece vocabulary (30k tokens, one per line)

Then copies both files to the Android app assets directory so Gradle
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
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIRS = [
    PROJECT_ROOT / "android-ltk" / "app" / "src" / "main" / "assets",
]
ONNX_PATH = MODELS_DIR / "arctic.onnx"
VOCAB_PATH = MODELS_DIR / "vocab.txt"

MODEL_ID = "Snowflake/snowflake-arctic-embed-xs"
MAX_LEN = 128

MODELS_DIR.mkdir(parents=True, exist_ok=True)
for assets_dir in ASSETS_DIRS:
    assets_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_base_model = AutoModel.from_pretrained(MODEL_ID)
_base_model.eval()


class _BertWrapper(torch.nn.Module):
    """Thin wrapper so torch.onnx.export receives explicit keyword args.

    Newer transformers BertModel.forward() raises 'multiple values for
    argument use_cache' when positional args are passed during JIT tracing.
    Passing kwargs fixes it and also makes the ONNX graph output a plain
    tensor rather than a ModelOutput object.
    """

    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state


model = _BertWrapper(_base_model)
model.eval()

# ---------------------------------------------------------------------------
# 2. Export
# ---------------------------------------------------------------------------
print("Exporting to ONNX (opset 14) ...")
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
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=14,
    do_constant_folding=True,
    dynamo=False,  # use legacy exporter — avoids verbose Unicode logging that breaks on Windows cp1252
)

# Force single-file ONNX — inline all weights so Android ORT can load from bytes.
# torch.onnx.export may produce a separate .data sidecar; ORT Android cannot resolve
# relative sidecar paths when the model is loaded via createSession(byteArray, opts).
import onnx as _onnx

print("Inlining weights into single file ...")
_model = _onnx.load(str(ONNX_PATH))  # resolves any sidecar relative to MODELS_DIR
_onnx.save_model(_model, str(ONNX_PATH), save_as_external_data=False)
for _sidecar in MODELS_DIR.glob("*.data"):  # remove any leftover .data files
    _sidecar.unlink()
    print(f"  Removed sidecar: {_sidecar.name}")

size_mb = ONNX_PATH.stat().st_size / 1e6
print(f"  -> {ONNX_PATH}  ({size_mb:.1f} MB, single file)")

# ---------------------------------------------------------------------------
# 2b. Quantize to int8 (dynamic quantization — shrinks ~91 MB → ~23 MB)
# ---------------------------------------------------------------------------
print("Quantizing to int8 ...")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    _quant_path = ONNX_PATH.parent / "arctic_q8.onnx"
    quantize_dynamic(str(ONNX_PATH), str(_quant_path), weight_type=QuantType.QInt8)
    _quant_path.replace(ONNX_PATH)
    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"  -> {ONNX_PATH}  ({size_mb:.1f} MB, int8 quantized)")
    print(f"  File size in bytes: {ONNX_PATH.stat().st_size}")
    print("  NOTE: update ASSET_MODEL_SIZE in Embedder.kt with the bytes value above")
except ImportError:
    print("  WARNING: onnxruntime.quantization not available — skipping int8 step.")
    print("           pip install onnxruntime  to enable it.")

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
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    }
    ort_hidden = sess.run(None, feeds)[0]

    with torch.no_grad():
        pt_hidden = _base_model(**dummy).last_hidden_state.numpy()

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
    ort_h2 = sess.run(
        None,
        {
            "input_ids": enc2["input_ids"].numpy(),
            "attention_mask": enc2["attention_mask"].numpy(),
            "token_type_ids": enc2.get(
                "token_type_ids", torch.zeros_like(enc2["input_ids"])
            ).numpy(),
        },
    )[0]
    embs = mean_pool_norm(ort_h2, enc2["attention_mask"].numpy())
    sims = embs @ embs.T
    print(
        f"  Cosine sim (calculus-chainrule)={sims[0, 1]:.3f}  (calculus-france)={sims[0, 2]:.3f}"
    )
    assert sims[0, 1] > sims[0, 2], (
        "Semantic sanity check failed — related sentences should score higher"
    )
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
for assets_dir in ASSETS_DIRS:
    for src in [ONNX_PATH, VOCAB_PATH]:
        dst = assets_dir / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst}")

print("\nAll done.")
print("Next: open android-ltk/ in Android Studio and run Gradle sync.")
