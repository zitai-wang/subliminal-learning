#!/usr/bin/env python3
"""Deterministically repair MathVision scoring in saved result JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().replace("−", "-")
    text = text.replace("$", "").replace("\\,", "").replace(",", "")
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(".()[]")
    return text.upper()


def process_file(input_path: Path, output_path: Path) -> tuple[int, int, list[str]]:
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("records", [])
    original_correct = sum(bool(record.get("is_correct", False)) for record in records)
    rescued_ids: list[str] = []

    for record in records:
        original = bool(record.get("is_correct", False))
        exact_final_match = bool(normalize_answer(record.get("raw_extraction"))) and (
            normalize_answer(record.get("raw_extraction"))
            == normalize_answer(record.get("answer"))
        )
        repaired = original or exact_final_match
        record["original_is_correct"] = original
        record["post_is_correct"] = repaired
        record["postprocess_exact_final_match"] = exact_final_match
        if repaired and not original:
            rescued_ids.append(str(record.get("id")))

    post_correct = sum(bool(record["post_is_correct"]) for record in records)
    payload["postprocess"] = {
        "method": "original_score_or_normalized_exact_final_answer_match",
        "ground_truth_used_for_scoring_only": True,
        "original_correct": original_correct,
        "post_correct": post_correct,
        "total": len(records),
        "original_accuracy": original_correct / len(records) if records else 0.0,
        "post_accuracy": post_correct / len(records) if records else 0.0,
        "rescued_ids": rescued_ids,
    }
    payload["eval_result_postprocessed"] = {
        "accuracy": post_correct / len(records) if records else 0.0
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return original_correct, post_correct, rescued_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    total_original = 0
    total_post = 0
    total_records = 0
    rescued: list[str] = []
    for input_path in args.inputs:
        output_path = args.output_dir / input_path.parent.name / input_path.name
        original, post, file_rescued = process_file(input_path, output_path)
        with input_path.open(encoding="utf-8") as handle:
            count = len(json.load(handle).get("records", []))
        total_original += original
        total_post += post
        total_records += count
        rescued.extend(file_rescued)
        print(f"{input_path}: {original}/{count} -> {post}/{count}; rescued={file_rescued}")

    summary = {
        "method": "original_score_or_normalized_exact_final_answer_match",
        "original_correct": total_original,
        "post_correct": total_post,
        "total": total_records,
        "original_accuracy": total_original / total_records if total_records else 0.0,
        "post_accuracy": total_post / total_records if total_records else 0.0,
        "rescued_ids": rescued,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
