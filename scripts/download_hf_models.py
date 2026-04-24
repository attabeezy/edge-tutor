"""
Download benchmark candidate GGUF models from Hugging Face into ./models.

Default targets:
  - Granite 4.0 H 350M
  - LFM2.5 350M
  - LFM2 350M Math

The script prefers Q4_K_M GGUF files for a practical size/quality balance,
and falls back to Q4_0 if Q4_K_M is not present.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    filename_hints: tuple[str, ...]
    subdir: str


MODEL_SPECS = [
    ModelSpec(
        alias="granite4:350m-h",
        repo_id="ibm-granite/granite-4.0-h-350m-GGUF",
        filename_hints=("Q4_K_M.gguf", "Q4_0.gguf"),
        subdir="granite-4.0-h-350m",
    ),
    ModelSpec(
        alias="lfm2.5:350m",
        repo_id="LiquidAI/LFM2.5-350M-GGUF",
        filename_hints=("Q4_K_M.gguf", "Q4_0.gguf"),
        subdir="lfm2.5-350m",
    ),
    ModelSpec(
        alias="lfm2-math",
        repo_id="LiquidAI/LFM2-350M-Math-GGUF",
        filename_hints=("Q4_K_M.gguf", "Q4_0.gguf"),
        subdir="lfm2-350m-math",
    ),
]


def select_filename(repo_id: str, hints: tuple[str, ...]) -> str:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    gguf_files = [name for name in files if name.lower().endswith(".gguf")]
    for hint in hints:
        for file_name in gguf_files:
            if file_name.endswith(hint):
                return file_name
    if not gguf_files:
        raise RuntimeError(f"No GGUF files found in {repo_id}")
    raise RuntimeError(
        f"No preferred quantization found in {repo_id}. "
        f"Available GGUF files: {', '.join(sorted(gguf_files))}"
    )


def download_model(spec: ModelSpec, models_dir: Path) -> Path:
    target_dir = models_dir / spec.subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = select_filename(spec.repo_id, spec.filename_hints)
    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        repo_type="model",
        filename=filename,
        local_dir=str(target_dir),
    )
    return Path(downloaded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download EdgeTutor candidate GGUF models from Hugging Face.")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store downloaded models.",
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        default=[spec.alias for spec in MODEL_SPECS],
        help="Subset of model aliases to download.",
    )
    args = parser.parse_args()

    selected = [spec for spec in MODEL_SPECS if spec.alias in set(args.aliases)]
    if not selected:
        print("No matching model aliases selected.")
        return 1

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    for spec in selected:
        print(f"[download] {spec.alias} <- {spec.repo_id}")
        path = download_model(spec, models_dir)
        print(f"[download] saved to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
