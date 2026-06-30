from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "tutor"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutor_dataset_matches_authored_generator_and_validates():
    generator = load_module("tutor_generator", DATA_DIR / "generate_dataset.py")
    validator = load_module("tutor_validator", DATA_DIR / "validate_dataset.py")

    generated = generator.make_rows()
    checked_in = validator.read_rows(DATA_DIR)

    assert len(generated) == 300
    assert [row["id"] for row in checked_in] == [
        row["id"] for split in ("train", "validation", "test")
        for row in generated if row["split"] == split
    ]
    assert [json.dumps(row, sort_keys=True) for row in checked_in] == [
        json.dumps(row, sort_keys=True) for split in ("train", "validation", "test")
        for row in generated if row["split"] == split
    ]
    validator.validate(checked_in)


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
