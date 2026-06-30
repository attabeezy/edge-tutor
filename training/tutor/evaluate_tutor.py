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
    route_marker = f"[{item['route']}]"
    evaluation = item["evaluation"]
    marker_valid = answer.startswith(route_marker) and (
        answer.count("[TEXTBOOK]") + answer.count("[GENERAL]") == 1
    )
    no_forbidden_answer = all(
        fragment.lower() not in answer.lower()
        for fragment in evaluation.get("must_not_include", [])
    )
    required_content = all(
        fragment.lower() in answer.lower()
        for fragment in evaluation.get("must_include", [])
    )
    return {
        "route_marker_valid": marker_valid,
        "no_premature_answer": no_forbidden_answer,
        "required_content_present": required_content,
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
            "route": item["route"],
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
        "one_guiding_question", "within_120_words",
    ):
        passed = sum(bool(result[key]) for result in results)
        print(f"{key}: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
