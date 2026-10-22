#!/usr/bin/env python3
"""Matched-information LLM vs VLM ablation on MathVista/MathVision.

Runs qwen2.5-7b-instruct (text LLM) and qwen2.5-vl-7b-instruct (VLM, but given
no raw image -- only the generated image caption + question, same as the LLM)
on the full testmini set of both datasets, all on cuda:0. This isolates
multimodal-alignment effects from task mismatch: both models see identical
text, so any gap is attributable to how accessible reasoning is post-alignment
rather than to the VLM never having been trained on text-only inputs.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/data1/wzy/sub-learn")
PYTHON = "/home/wzy/anaconda3/envs/licv/bin/python"
RECORD_ROOT = ROOT / "results" / "record" / "caption_text_only"

MODELS = ["qwen2.5-7b-instruct", "qwen2.5-vl-7b-instruct"]
DATASETS = ["mathvista", "mathvision"]


def record_dir_for(dataset: str, model: str) -> Path:
    return RECORD_ROOT / f"{dataset}_{model}"


def record_path_for(dataset: str, model: str) -> Path:
    return record_dir_for(dataset, model) / "0shot.json"


def build_cmd(dataset: str, model: str, num_query_samples: int, batch_size: int) -> list[str]:
    record_dir = record_dir_for(dataset, model)
    return [
        PYTHON,
        str(ROOT / "src" / "eval_baseline.py"),
        "--config-name",
        "eval_baseline",
        f"model_name={model}",
        f"data.name={dataset}",
        f"data.num_query_samples={num_query_samples}",
        "data.num_shot=0",
        "data.text_only_caption_eval=True",
        "ckpt_path=null",
        "use_extracted_cot_vector=False",
        "eval_mode=EVAL_BASELINE",
        "mode=eval",
        "resume=True",
        "dtype=torch.float16",
        f"batch_size={batch_size}",
        f"+record_root={record_dir}",
    ]


def run_one(dataset: str, model: str, num_query_samples: int, batch_size: int) -> None:
    record_dir = record_dir_for(dataset, model)
    record_dir.mkdir(parents=True, exist_ok=True)
    log_path = record_dir / "run.log"
    cmd = build_cmd(dataset, model, num_query_samples, batch_size)
    print(f"[{dataset}/{model}] COMMAND: {' '.join(cmd)}")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    env["HF_HOME"] = str(ROOT / ".hf_cache")
    env["HF_DATASETS_CACHE"] = str(ROOT / ".hf_cache" / "datasets")
    # hf-mirror.com's redirect chain breaks huggingface_hub's HEAD resolution
    # for MathLLMs/MathVision; huggingface.co is directly reachable here.
    env.pop("HF_ENDPOINT", None)

    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    print(f"[{dataset}/{model}] done, log at {log_path}")


def read_accuracy(dataset: str, model: str) -> dict | None:
    path = record_path_for(dataset, model)
    if not path.exists() or path.stat().st_size == 0:
        return None
    payload = json.loads(path.read_text())
    accuracy = payload.get("eval_result", {}).get("accuracy")
    records = payload.get("records", [])
    return {"accuracy": accuracy, "num_records": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-query-samples", type=int, default=0, help="0 = full testmini set")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--skip-run", action="store_true", help="Only (re)build the summary from existing records")
    args = parser.parse_args()

    if not args.skip_run:
        for dataset in args.datasets:
            for model in args.models:
                run_one(dataset, model, args.num_query_samples, args.batch_size)

    summary = []
    for dataset in args.datasets:
        for model in args.models:
            result = read_accuracy(dataset, model)
            summary.append({"dataset": dataset, "model": model, **(result or {"accuracy": None, "num_records": 0})})
            print(f"[{dataset}/{model}] {result}")

    summary_path = RECORD_ROOT / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
