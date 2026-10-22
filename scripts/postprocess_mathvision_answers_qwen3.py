#!/usr/bin/env python3
"""Ground-truth-blind answer extraction for saved MathVision generations."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = "/data/share/Qwen3-VL-8B-Instruct"
INPUT_GLOB = str(
    ROOT
    / "results/qwen3vl_thinking_baseline/mathvision_greedy32768_shard*/0shot.json"
)


def load_records() -> list[dict]:
    records = []
    for path in sorted(glob.glob(INPUT_GLOB)):
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f)["records"])
    return records


def response_excerpt(response: str) -> str:
    response = str(response or "")
    if "</think>" in response:
        return response.rsplit("</think>", 1)[-1].strip()
    return response[-6000:].strip()


def build_prompt(record: dict) -> str:
    options = record.get("options") or []
    option_text = ""
    if options and "".join(str(x) for x in options) != "ABCDE":
        option_text = "\nChoices:\n" + "\n".join(
            f"{chr(ord('A') + i)}. {value}" for i, value in enumerate(options)
        )
    return (
        "You are an answer extractor, not a problem solver. Extract the answer that "
        "the model ultimately committed to in its response. Do not correct its reasoning "
        "and do not solve the problem yourself. For multiple choice, return the option "
        "letter when it is stated or can be mapped unambiguously from the listed choices. "
        "Return only <answer>SHORT_ANSWER</answer>.\n\n"
        f"Question:\n{record.get('question', '')}{option_text}\n\n"
        f"Model response:\n{response_excerpt(record.get('response', ''))}"
    )


def clean_extraction(text: str) -> str:
    tagged = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged[-1].strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(1234)
    records = load_records()
    records = records[args.shard_index :: args.shard_count]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype="auto",
        device_map="cuda",
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_prompt(record)}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for record in batch
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=96,
                    repetition_penalty=1.0,
                )
            new_tokens = generated[:, inputs.input_ids.shape[1] :]
            outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for record, output in zip(batch, outputs):
                row = {
                    "id": str(record["id"]),
                    "extracted_answer": clean_extraction(output),
                    "extractor_output": output,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            print(
                f"shard {args.shard_index}: {min(start + len(batch), len(records))}/"
                f"{len(records)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
