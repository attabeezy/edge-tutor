#!/usr/bin/env python3
"""Validate the static six-batch tutoring dataset and release policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_SPLITS = {"train": 1000, "validation": 250, "test": 250}
EXPECTED_MODES = {"grounded": 750, "general": 750}
EXPECTED_MOVES = {
    "diagnostic": 250, "hint": 250, "explanation": 250,
    "correction": 250, "understanding_check": 250, "complete_answer": 250,
}
ROUTE_MARKERS = ("[TEXTBOOK]", "[GENERAL]")
EXPECTED_BATCHES = {f"batch-{number}": 250 for number in range(1, 7)}
EXPECTED_SUBJECT_CONCEPTS = {
    "math": 63, "science": 63, "english": 62, "social_studies": 62,
}
LEGACY_EXPECTED_SPLITS = {"train": 240, "validation": 30, "test": 30}
LEGACY_EXPECTED_SUBJECTS = {
    "math": 75, "science": 75, "english": 75, "social_studies": 75,
}
LEGACY_EXPECTED_LEVELS = {
    "late_grade_school": 100, "middle_school": 100, "high_school": 100,
}
LEGACY_EXPECTED_ROUTES = {"TEXTBOOK": 180, "GENERAL": 120}
LEGACY_EXPECTED_MOVES = {
    "diagnostic": 60,
    "initial_hint": 60,
    "concept_explanation": 60,
    "corrective_feedback": 60,
    "complete_answer": 60,
}
VALID_LEVELS = {f"JHS {number}" for number in range(1, 4)} | {
    f"SHS {number}" for number in range(1, 4)
}


def read_rows(directory: Path) -> list[dict]:
    rows = []
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
                    raise ValueError(f"{path}:{line_number}: split does not match filename")
                rows.append(item)
    return rows


def assert_counts(name: str, actual: Counter, expected: dict[str, int]) -> None:
    if dict(actual) != expected:
        raise ValueError(f"unexpected {name} distribution: {dict(actual)} != {expected}")


def validate_legacy(rows: list[dict]) -> None:
    """Validate the checked-in 300-row dataset used by the demo notebook."""
    ids, fingerprints = set(), set()
    topics_by_split: dict[str, set[str]] = defaultdict(set)
    required = {
        "id", "concept_id", "split", "subject", "level", "route",
        "tutor_move", "answer_reveal_allowed", "evaluation", "messages",
    }
    for item in rows:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{item.get('id')}: missing fields {sorted(missing)}")
        if item["id"] in ids:
            raise ValueError(f"duplicate id: {item['id']}")
        ids.add(item["id"])
        topics_by_split[item["split"]].add(item["concept_id"])

        messages = item["messages"]
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{item['id']}: invalid message roles")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError(f"{item['id']}: empty message")
        if any(not m["content"].isascii() for m in messages):
            raise ValueError(f"{item['id']}: messages must be ASCII")

        assistant = messages[2]["content"]
        expected_marker = f"[{item['route']}]"
        if not assistant.startswith(expected_marker + "\n"):
            raise ValueError(f"{item['id']}: assistant must start with {expected_marker}")
        if assistant.count("[TEXTBOOK]") + assistant.count("[GENERAL]") != 1:
            raise ValueError(f"{item['id']}: assistant must contain one route marker")
        if assistant.count("?") != 1:
            raise ValueError(f"{item['id']}: target must contain exactly one question")
        if not 5 <= len(assistant.split()) <= 120:
            raise ValueError(f"{item['id']}: target length outside 5..120 words")

        evaluation = item["evaluation"]
        for fragment in evaluation.get("must_not_include", []):
            if fragment.lower() in assistant.lower():
                raise ValueError(f"{item['id']}: forbidden answer fragment {fragment!r}")
        for fragment in evaluation.get("must_include", []):
            if fragment.lower() not in assistant.lower():
                raise ValueError(f"{item['id']}: missing required fragment {fragment!r}")
        if item["tutor_move"] == "complete_answer" and not item["answer_reveal_allowed"]:
            raise ValueError(f"{item['id']}: complete answer must allow revelation")

        fingerprint = json.dumps(messages, sort_keys=True)
        if fingerprint in fingerprints:
            raise ValueError(f"{item['id']}: duplicate message sequence")
        fingerprints.add(fingerprint)

    split_names = list(LEGACY_EXPECTED_SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = topics_by_split[left] & topics_by_split[right]
            if overlap:
                raise ValueError(f"concept leakage between {left} and {right}")

    assert_counts("split", Counter(r["split"] for r in rows), LEGACY_EXPECTED_SPLITS)
    assert_counts("subject", Counter(r["subject"] for r in rows), LEGACY_EXPECTED_SUBJECTS)
    assert_counts("level", Counter(r["level"] for r in rows), LEGACY_EXPECTED_LEVELS)
    assert_counts("route", Counter(r["route"] for r in rows), LEGACY_EXPECTED_ROUTES)
    assert_counts("tutor move", Counter(r["tutor_move"] for r in rows), LEGACY_EXPECTED_MOVES)


def validate(rows: list[dict], require_reviewed: bool = False) -> None:
    if rows and "batch_id" not in rows[0]:
        validate_legacy(rows)
        return

    ids, fingerprints = set(), set()
    rows_by_concept: dict[str, list[dict]] = defaultdict(list)
    required = {
        "id", "batch_id", "concept_id", "split", "subject", "level", "knowledge_mode",
        "tutor_move", "answer_permission", "answer_reveal_allowed", "evaluation",
        "review", "messages",
    }
    for item in rows:
        if required - item.keys():
            raise ValueError(f"{item.get('id')}: missing fields {sorted(required - item.keys())}")
        if "route" in item:
            raise ValueError(f"{item['id']}: route metadata is obsolete")
        if item["id"] in ids:
            raise ValueError(f"duplicate id: {item['id']}")
        ids.add(item["id"])
        rows_by_concept[item["concept_id"]].append(item)
        if item["level"] not in VALID_LEVELS:
            raise ValueError(f"{item['id']}: invalid Ghanaian school level")
        batch_number = int(item["batch_id"].removeprefix("batch-"))
        expected_split = "train" if batch_number <= 4 else (
            "validation" if batch_number == 5 else "test"
        )
        if item["split"] != expected_split:
            raise ValueError(f"{item['id']}: batch does not match split")
        messages = item["messages"]
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{item['id']}: invalid message roles")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError(f"{item['id']}: empty message")
        if any(not m["content"].isascii() for m in messages):
            raise ValueError(f"{item['id']}: messages must be ASCII")
        combined = "\n".join(m["content"] for m in messages)
        if any(marker in combined for marker in ROUTE_MARKERS):
            raise ValueError(f"{item['id']}: route marker present")
        user, assistant = messages[1]["content"], messages[2]["content"]
        has_context = user.startswith("Textbook passage:\n")
        if has_context != (item["knowledge_mode"] == "grounded"):
            raise ValueError(f"{item['id']}: context does not match knowledge_mode")
        if assistant.count("?") != 1:
            raise ValueError(f"{item['id']}: target must contain exactly one question")
        if not 5 <= len(assistant.split()) <= 120:
            raise ValueError(f"{item['id']}: target length outside 5..120 words")
        evaluation = item["evaluation"]
        if not evaluation.get("required_facts"):
            raise ValueError(f"{item['id']}: at least one required fact is mandatory")
        for fragment in evaluation.get("forbidden_answer_fragments", []):
            if fragment.lower() in assistant.lower():
                raise ValueError(f"{item['id']}: forbidden answer fragment {fragment!r}")
        for fragment in evaluation.get("required_facts", []):
            if fragment.lower() not in assistant.lower():
                raise ValueError(f"{item['id']}: missing required fragment {fragment!r}")
        explicit = "complete worked answer" in user.lower()
        if item["answer_reveal_allowed"] != explicit:
            raise ValueError(f"{item['id']}: answer permission does not match explicit request")
        expected_permission = "explicit" if explicit else "withhold"
        if item["answer_permission"] != expected_permission:
            raise ValueError(f"{item['id']}: invalid answer_permission")
        if item["tutor_move"] == "complete_answer" and not explicit:
            raise ValueError(f"{item['id']}: complete answer lacks explicit request")
        review = item["review"]
        if review.get("status") not in {"pending", "approved", "rejected"}:
            raise ValueError(f"{item['id']}: invalid review status")
        if require_reviewed and (review.get("status") != "approved" or not review.get("reviewer")):
            raise ValueError(f"{item['id']}: target has not been manually approved")
        fingerprint = json.dumps(messages, sort_keys=True)
        if fingerprint in fingerprints:
            raise ValueError(f"{item['id']}: duplicate message sequence")
        fingerprints.add(fingerprint)

    if len(rows_by_concept) != 250:
        raise ValueError("dataset must contain exactly 250 concepts")
    for concept_id, concept_rows in rows_by_concept.items():
        if len(concept_rows) != 6:
            raise ValueError(f"{concept_id}: expected six interactions")
        if {row["tutor_move"] for row in concept_rows} != set(EXPECTED_MOVES):
            raise ValueError(f"{concept_id}: must contain every tutor move exactly once")
        if Counter(row["knowledge_mode"] for row in concept_rows) != {
            "grounded": 3, "general": 3
        }:
            raise ValueError(f"{concept_id}: expected three grounded and three general rows")
        if len({row["batch_id"] for row in concept_rows}) != 6:
            raise ValueError(f"{concept_id}: interactions must span all six batches")
    assert_counts("split", Counter(r["split"] for r in rows), EXPECTED_SPLITS)
    assert_counts("batch", Counter(r["batch_id"] for r in rows), EXPECTED_BATCHES)
    assert_counts("knowledge mode", Counter(r["knowledge_mode"] for r in rows), EXPECTED_MODES)
    assert_counts("tutor move", Counter(r["tutor_move"] for r in rows), EXPECTED_MOVES)
    concept_subjects = Counter(
        concept_rows[0]["subject"] for concept_rows in rows_by_concept.values()
    )
    assert_counts("subject concepts", concept_subjects, EXPECTED_SUBJECT_CONCEPTS)
    subjects = Counter(r["subject"] for r in rows)
    levels = Counter(r["level"] for r in rows)
    if max(subjects.values()) - min(subjects.values()) > 6:
        raise ValueError(f"subjects are not balanced: {dict(subjects)}")
    if max(levels.values()) - min(levels.values()) > 6:
        raise ValueError(f"levels are not balanced: {dict(levels)}")
    reports = Path(__file__).resolve().parent / "reviews"
    for number in range(1, 7):
        report = reports / f"batch-{number}.md"
        if not report.is_file():
            raise ValueError(f"missing batch review report: {report}")
        if require_reviewed and "Status: approved" not in report.read_text(encoding="utf-8"):
            raise ValueError(f"batch-{number}: review report is not approved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.data_dir)
    validate(rows, require_reviewed=args.require_reviewed)
    schema = "six-batch" if rows and "batch_id" in rows[0] else "legacy demo"
    print(f"validated {len(rows)} rows ({schema} schema)")
    print(f"splits: {dict(Counter(r['split'] for r in rows))}")
    if schema == "six-batch":
        print(f"knowledge modes: {dict(Counter(r['knowledge_mode'] for r in rows))}")
    else:
        print(f"routes: {dict(Counter(r['route'] for r in rows))}")


if __name__ == "__main__":
    main()
