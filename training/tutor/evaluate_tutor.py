#!/usr/bin/env python3
"""Generate and score upstream Qwen tutor responses on a held-out JSONL split."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def automatic_scores(item: dict, answer: str) -> dict[str, bool]:
    evaluation = item["evaluation"]
    normalized = answer.lower()
    legacy = "route" in item
    forbidden = evaluation.get(
        "forbidden_answer_fragments", evaluation.get("must_not_include", [])
    )
    required = evaluation.get("required_facts", evaluation.get("must_include", []))
    no_forbidden_answer = all(
        fragment.lower() not in normalized
        for fragment in forbidden
    )
    required_content = all(
        fragment.lower() in normalized
        for fragment in required
    )
    user = item["messages"][1]["content"]
    if legacy:
        marker = f"[{item['route']}]"
        route_marker_valid = answer.startswith(marker + "\n") and sum(
            answer.count(candidate) for candidate in ("[TEXTBOOK]", "[GENERAL]")
        ) == 1
        passage = user.partition("TEXTBOOK PASSAGE:\n")[2].split("\n\n", 1)[0]
    else:
        route_marker_valid = not any(
            marker in answer for marker in ("[TEXTBOOK]", "[GENERAL]")
        )
        passage = user.split("\n\n", 1)[0].removeprefix("Textbook passage:\n")
    stopwords = {"about", "because", "which", "their", "there", "these", "those", "would", "could"}
    passage_terms = {
        word.lower().strip(".,:;!?()") for word in passage.split()
        if len(word) >= 5 and word.lower().strip(".,:;!?()") not in stopwords
    }
    answer_terms = {word.strip(".,:;!?()") for word in normalized.split()}
    knowledge_mode = item.get("knowledge_mode", item.get("route", "").lower())
    grounded = route_marker_valid if legacy else (
        knowledge_mode != "grounded" or len(passage_terms & answer_terms) >= 2
    )
    meaningful = len(answer.split()) >= 5
    malformed = not answer.strip() or not route_marker_valid
    return {
        "route_marker_valid": route_marker_valid,
        "no_premature_answer": no_forbidden_answer,
        "required_content_present": required_content,
        "correctness_fragments": required_content,
        "no_forbidden_leakage": no_forbidden_answer,
        "groundedness": grounded,
        "instruction_adaptation": required_content and no_forbidden_answer,
        "response_complete": meaningful and not malformed and answer.rstrip().endswith(("?", ".", "!")),
        "one_guiding_question": answer.count("?") == 1,
        "within_120_words": len(answer.split()) <= 120,
    }


def load_model(model_name: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], device: str, max_new_tokens: int) -> str:
    import torch

    prompt_messages = messages[:-1]
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[-1]:]
    return clean_output(tokenizer.decode(generated, skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    model, tokenizer = load_model(args.model, args.adapter, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, item in enumerate(rows, 1):
        answer = generate(model, tokenizer, item["messages"], args.device, args.max_new_tokens)
        scores = automatic_scores(item, answer)
        result = {
            "id": item["id"],
            "subject": item["subject"],
            "level": item["level"],
            "knowledge_mode": item.get("knowledge_mode", item.get("route", "").lower()),
            "tutor_move": item["tutor_move"],
            **scores,
            "answer": answer,
            "correctness_0_2": "",
            "helpfulness_0_2": "",
            "adaptation_0_2": "",
            "feedback_0_2": "",
        }
        results.append(result)
        print(json.dumps({"progress": f"{index}/{len(rows)}", **result}, ensure_ascii=True))

    fieldnames = list(results[0]) if results else []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote {len(results)} results to {args.output}")
    for key in (
        "route_marker_valid", "no_premature_answer", "required_content_present",
        "correctness_fragments", "no_forbidden_leakage", "groundedness",
        "instruction_adaptation", "response_complete", "one_guiding_question",
        "within_120_words",
    ):
        passed = sum(bool(result[key]) for result in results)
        print(f"{key}: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
