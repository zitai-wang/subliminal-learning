#!/usr/bin/env python3
"""Continue only the Qwen3-VL MathVision responses truncated at 32K tokens."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_utils.mathvision import DatasetState, MathVisionDataset  # noqa: E402
from qwen_vl_model_wrapper import QwenVLModelWrapper  # noqa: E402


MODEL_PATH = "/data/share/Qwen3-VL-8B-Thinking"
INPUT_GLOB = str(
    ROOT
    / "results/qwen3vl_thinking_baseline/mathvision_greedy32768_shard*/0shot.json"
)


def load_truncated_records() -> list[dict]:
    records = []
    for path in sorted(glob.glob(INPUT_GLOB)):
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f)["records"])
    return [record for record in records if "</think>" not in record.get("response", "")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--extra-tokens", type=int, default=8192)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    records = load_truncated_records()
    records = records[args.shard_index :: args.shard_count]

    data_cfg = OmegaConf.create(
        {
            "name": "mathvision",
            "source": "/data/share/MathVision",
            "query_split": "testmini",
            "support_split": "testmini",
            "num_query_samples": 304,
            "seed": 42,
        }
    )
    dataset = MathVisionDataset(data_cfg, model_name="qwen3-vl-8b-thinking")

    model = QwenVLModelWrapper(
        model_root=MODEL_PATH,
        processor_class=AutoProcessor,
        model_class=Qwen3VLForConditionalGeneration,
        support_models=["Qwen/Qwen3-VL-8B-Thinking"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        processor_args={"padding_side": "left"},
        model_args={"attn_implementation": "flash_attention_2"},
        model_name="qwen3-vl-8b-thinking",
    ).to("cuda")
    model.eval()
    dataset.model_processor = model.processor

    items_by_id = {str(item["id"]): item for item in dataset._query_set}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for position, record in enumerate(records, start=1):
            item = items_by_id[str(record["id"])]
            query = dataset.create_query(item)
            messages = dataset._create_vlm_messages(
                query,
                DatasetState.EVAL_BASELINE,
                item=item,
                include_image=True,
            )
            prompt = dataset._apply_chat_template(messages, add_generation_prompt=True)
            prefix = prompt if args.from_scratch else prompt + record["response"]
            image = dataset._load_image(item)

            generated = model.generate(
                [prefix],
                images=[image],
                num_beams=1,
                do_sample=False,
                max_new_tokens=args.extra_tokens,
                repetition_penalty=1.05,
            )[0]
            combined = generated if args.from_scratch else record["response"] + generated
            extracted = dataset.extract_answer(combined)
            is_correct = dataset._is_correct(item, extracted, combined)
            row = {
                "id": str(record["id"]),
                "answer": record["answer"],
                "original_response": record["response"],
                "from_scratch": args.from_scratch,
                "continuation": generated,
                "combined_response": combined,
                "raw_extraction": extracted,
                "is_correct": is_correct,
                "closed_think": "</think>" in combined,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"shard {args.shard_index}: {position}/{len(records)} "
                f"id={record['id']} closed={row['closed_think']} correct={is_correct}",
                flush=True,
            )


if __name__ == "__main__":
    main()
