#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SOURCES = [
    (
        "llm_derived",
        "qwen2.5-7b-instruct",
        "commonsenseqa",
        ROOT / "output/commonsenseqa/self_cot_qwen2.5-7b-instruct_train500_20260417_163423.json",
    ),
    (
        "llm_derived",
        "qwen2.5-7b-instruct",
        "strategyqa",
        ROOT / "output/strategyqa/self_cot_qwen2.5-7b-instruct_train500_20260417_171124.json",
    ),
    (
        "vlm_derived",
        "qwen2.5-vl-7b-instruct",
        "commonsenseqa",
        ROOT / "results/self_cot/commonsenseqa/qwen2.5-vl-7b-instruct_commonsenseqa_self_cot_500.json",
    ),
    (
        "vlm_derived",
        "qwen2.5-vl-7b-instruct",
        "strategyqa",
        ROOT / "results/self_cot/strategyqa/qwen2.5-vl-7b-instruct_strategyqa_self_cot_500.json",
    ),
    (
        "llm_derived",
        "internlm2_5-7b-chat",
        "commonsenseqa",
        ROOT / "results/self_cot/commonsenseqa/internlm2_5-7b-chat_commonsenseqa_self_cot_500.json",
    ),
    (
        "llm_derived",
        "internlm2_5-7b-chat",
        "strategyqa",
        ROOT / "results/self_cot/strategyqa/internlm2_5-7b-chat_strategyqa_self_cot_500.json",
    ),
    (
        "vlm_derived",
        "internvl2_5-8b",
        "commonsenseqa",
        ROOT / "results/self_cot/commonsenseqa/internvl2_5-8b_commonsenseqa_self_cot_500.json",
    ),
    (
        "vlm_derived",
        "internvl2_5-8b",
        "strategyqa",
        ROOT / "results/self_cot/strategyqa/internvl2_5-8b_strategyqa_self_cot_500.json",
    ),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice support-N self-CoT files from existing support500 files.")
    parser.add_argument("--support-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = ROOT / "results" / "rebuttal_support_size" / f"support{args.support_size}_from_500"
    manifests = []
    for derived, model, dataset, source_path in SOURCES:
        rows = read_jsonl(source_path)
        support_rows = rows[: args.support_size]
        correct = [row for row in support_rows if bool(row.get("is_correct"))]

        stem = f"{model}_{dataset}_support{args.support_size}_{derived}"
        out_dir = out_root / derived / model / dataset
        full_path = out_dir / f"{stem}_self_cot.json"
        correct_path = out_dir / f"{stem}_self_cot_correct_only.json"
        manifest_path = out_dir / "manifest.json"

        write_jsonl(full_path, support_rows)
        write_jsonl(correct_path, correct)

        manifest = {
            "derived": derived,
            "model": model,
            "support_source": dataset,
            "source_500_path": str(source_path),
            f"support{args.support_size}_full_path": str(full_path),
            f"support{args.support_size}_correct_only_path": str(correct_path),
            "nominal_support_size": args.support_size,
            "effective_correct_count": len(correct),
            "source_500_count": len(rows),
            "sample_then_filter": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifests.append({**manifest, "manifest_path": str(manifest_path)})
        print(json.dumps(manifests[-1], ensure_ascii=False), flush=True)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
