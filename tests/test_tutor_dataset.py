from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "tutor"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_tutor_dataset_validates():
    validator = load_module("tutor_validator", DATA_DIR / "validate_dataset.py")

    checked_in = validator.read_rows(DATA_DIR)

    assert len(checked_in) == 300
    validator.validate(checked_in)

    assert Counter(row["split"] for row in checked_in) == validator.EXPECTED_SPLITS
    assert Counter(row["tutor_move"] for row in checked_in) == validator.EXPECTED_MOVES


def test_colab_notebook_has_expected_handoff_sections():
    notebook = json.loads((DATA_DIR / "colab_qwen35_tutor.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert heading in markdown

    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert '"--num_train_epochs", "1"' in code
    assert '"--train_data", str(REPO_DIR / "training/tutor/train.jsonl")' in code
    assert '"--validation_data", str(REPO_DIR / "training/tutor/validation.jsonl")' in code
    assert '"--path", str(MERGED_MODEL_DIR)' in code
    assert "--lora_path" not in code
    assert not any(cell.get("outputs") for cell in notebook["cells"])
