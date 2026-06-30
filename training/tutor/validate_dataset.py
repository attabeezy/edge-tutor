#!/usr/bin/env python3
"""Validate tutoring JSONL structure, distribution, and policy invariants."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SPLITS = {"train": 240, "validation": 30, "test": 30}
EXPECTED_SUBJECTS = {"math": 75, "science": 75, "english": 75, "social_studies": 75}
EXPECTED_LEVELS = {"late_grade_school": 100, "middle_school": 100, "high_school": 100}
EXPECTED_ROUTES = {"TEXTBOOK": 180, "GENERAL": 120}
EXPECTED_MOVES = {
    "diagnostic": 60,
    "initial_hint": 60,
    "concept_explanation": 60,
    "corrective_feedback": 60,
    "complete_answer": 60,
}


def read_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for split in EXPECTED_SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing dataset file: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if item.get("split") != split:
                    raise ValueError(f"{path}:{line_number}: split field does not match filename")
                rows.append(item)
    return rows


def assert_counts(name: str, actual: Counter, expected: dict[str, int]) -> None:
    if dict(actual) != expected:
        raise ValueError(f"unexpected {name} distribution: {dict(actual)} != {expected}")


def validate(rows: list[dict]) -> None:
    ids: set[str] = set()
    message_fingerprints: set[str] = set()
    topics_by_split: dict[str, set[str]] = defaultdict(set)

    for item in rows:
        item_id = item.get("id")
        if not item_id or item_id in ids:
            raise ValueError(f"missing or duplicate id: {item_id!r}")
        ids.add(item_id)

        required = {
            "concept_id", "split", "subject", "level", "route", "tutor_move",
            "answer_reveal_allowed", "evaluation", "messages",
        }
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{item_id}: missing fields {sorted(missing)}")
        topics_by_split[item["split"]].add(item["concept_id"])

        messages = item["messages"]
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{item_id}: roles must be system, user, assistant")
        if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
            raise ValueError(f"{item_id}: message content must be non-empty text")
        if any(not message["content"].isascii() for message in messages):
            raise ValueError(f"{item_id}: all messages must be ASCII")

        fingerprint = json.dumps(messages, sort_keys=True)
        if fingerprint in message_fingerprints:
            raise ValueError(f"{item_id}: duplicate message sequence")
        message_fingerprints.add(fingerprint)

        expected_marker = f"[{item['route']}]"
        assistant = messages[-1]["content"]
        if not assistant.startswith(expected_marker + "\n"):
            raise ValueError(f"{item_id}: assistant must start with {expected_marker}")
        if assistant.count("[TEXTBOOK]") + assistant.count("[GENERAL]") != 1:
            raise ValueError(f"{item_id}: assistant must contain exactly one route marker")
        if len(assistant.split()) > 120:
            raise ValueError(f"{item_id}: assistant exceeds 120 words")
        if assistant.count("?") != 1:
            raise ValueError(f"{item_id}: assistant must ask exactly one guiding question")

        evaluation = item["evaluation"]
        for fragment in evaluation.get("must_not_include", []):
            if fragment.lower() in assistant.lower():
                raise ValueError(f"{item_id}: prematurely reveals forbidden fragment {fragment!r}")
        for fragment in evaluation.get("must_include", []):
            if fragment.lower() not in assistant.lower():
                raise ValueError(f"{item_id}: missing required fragment {fragment!r}")
        if item["tutor_move"] == "complete_answer" and not item["answer_reveal_allowed"]:
            raise ValueError(f"{item_id}: complete answers must explicitly allow revelation")

    split_names = list(EXPECTED_SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = topics_by_split[left] & topics_by_split[right]
            if overlap:
                raise ValueError(f"concept leakage between {left} and {right}: {sorted(overlap)}")

    assert_counts("split", Counter(item["split"] for item in rows), EXPECTED_SPLITS)
    assert_counts("subject", Counter(item["subject"] for item in rows), EXPECTED_SUBJECTS)
    assert_counts("level", Counter(item["level"] for item in rows), EXPECTED_LEVELS)
    assert_counts("route", Counter(item["route"] for item in rows), EXPECTED_ROUTES)
    assert_counts("tutor move", Counter(item["tutor_move"] for item in rows), EXPECTED_MOVES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows = read_rows(args.data_dir)
    validate(rows)
    print(f"validated {len(rows)} tutoring rows")
    print(f"splits: {dict(Counter(item['split'] for item in rows))}")
    print(f"subjects: {dict(Counter(item['subject'] for item in rows))}")
    print(f"routes: {dict(Counter(item['route'] for item in rows))}")


if __name__ == "__main__":
    main()
