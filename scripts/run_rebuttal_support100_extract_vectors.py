#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/wzy/anaconda3/envs/licv/bin/python")
RUN_PYTHON = str(PYTHON if PYTHON.exists() else sys.executable)
SELF_COT_ROOT = ROOT / "results" / "rebuttal_support_size" / "support100_from_500"

DATA_SOURCES = {
    "commonsenseqa": "/data/share/commonsenceqa/",
    "strategyqa": "/data/share/strategy_qa/",
}


@dataclass(frozen=True)
class ExtractSpec:
    derived: str
    model: str
    dataset: str
    script: str
    config: str


SPECS = [
    ExtractSpec("llm_derived", "qwen2.5-7b-instruct", "commonsenseqa", "src/extract_cot_vector.py", "extract_cot_vector"),
    ExtractSpec("llm_derived", "qwen2.5-7b-instruct", "strategyqa", "src/extract_cot_vector.py", "extract_cot_vector"),
    ExtractSpec("vlm_derived", "qwen2.5-vl-7b-instruct", "commonsenseqa", "src/extract_cot_vector.py", "extract_cot_vector"),
    ExtractSpec("vlm_derived", "qwen2.5-vl-7b-instruct", "strategyqa", "src/extract_cot_vector.py", "extract_cot_vector"),
    ExtractSpec("llm_derived", "internlm2_5-7b-chat", "commonsenseqa", "src/extract_cot_vector_internlm.py", "extract_cot_vector_internlm"),
    ExtractSpec("llm_derived", "internlm2_5-7b-chat", "strategyqa", "src/extract_cot_vector_internlm.py", "extract_cot_vector_internlm"),
    ExtractSpec("vlm_derived", "internvl2_5-8b", "commonsenseqa", "src/extract_cot_vector_internvl.py", "extract_cot_vector_internvl"),
    ExtractSpec("vlm_derived", "internvl2_5-8b", "strategyqa", "src/extract_cot_vector_internvl.py", "extract_cot_vector_internvl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract vectors from support-N correct-only self-CoT files.")
    parser.add_argument("--support-size", type=int, default=100)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def task_dir(spec: ExtractSpec, support_size: int) -> Path:
    root = ROOT / "results" / "rebuttal_support_size" / f"support{support_size}_from_500"
    return root / spec.derived / spec.model / spec.dataset


def paths(spec: ExtractSpec, support_size: int) -> dict[str, Path]:
    stem = f"{spec.model}_{spec.dataset}_support{support_size}_{spec.derived}"
    root = task_dir(spec, support_size)
    return {
        "correct": root / f"{stem}_self_cot_correct_only.json",
        "vector": root / f"{stem}_avg_cot_vector.pt",
        "log": root / "logs" / "03_extract_vector.log",
        "manifest": root / "manifest.json",
    }


def run_logged(command: list[str], gpu: str, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf_cache" / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache" / "transformers"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND:\n")
        handle.write(" ".join(command) + "\n\n")
        handle.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
        handle.flush()
        if dry_run:
            return 0
        process = subprocess.run(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        return process.returncode


def update_manifest(spec: ExtractSpec, payload: dict) -> None:
    manifest_path = paths(spec, payload["support_size"])["manifest"]
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(spec: ExtractSpec, gpu: str, support_size: int, dry_run: bool) -> dict:
    p = paths(spec, support_size)
    if not p["correct"].exists():
        return {"model": spec.model, "dataset": spec.dataset, "exit_code": 2, "error": f"missing {p['correct']}"}

    device_override = "+devices=0" if spec.config == "extract_cot_vector" else "devices=0"
    cmd = [
        RUN_PYTHON,
        str(ROOT / spec.script),
        "--config-name",
        spec.config,
        f"model_name={spec.model}",
        f"data.name={spec.dataset}",
        f"data.num_query_samples={support_size}",
        "data.seed=42",
        f"++data.source={DATA_SOURCES[spec.dataset]}",
        device_override,
        "data.use_self_cot=True",
        f"data.self_cot_path={p['correct']}",
        f"output_path={p['vector']}",
        "batch_size=1",
    ]
    code = run_logged(cmd, gpu, p["log"], dry_run)
    result = {
        "derived": spec.derived,
        "model": spec.model,
        "support_source": spec.dataset,
        "support_size": support_size,
        "exit_code": code,
        "vector_path": str(p["vector"]),
        "extract_log": str(p["log"]),
    }
    if code == 0 and not dry_run:
        update_manifest(spec, {"support_size": support_size, "vector_path": str(p["vector"]), "extract_log": str(p["log"])})
    return result


def partition(items: list[ExtractSpec], count: int) -> list[list[ExtractSpec]]:
    groups = [[] for _ in range(count)]
    for index, item in enumerate(items):
        groups[index % count].append(item)
    return groups


def run_group(gpu: str, group: list[ExtractSpec], support_size: int, dry_run: bool) -> list[dict]:
    return [run_one(spec, gpu, support_size, dry_run) for spec in group]


def main() -> None:
    args = parse_args()
    self_cot_root = ROOT / "results" / "rebuttal_support_size" / f"support{args.support_size}_from_500"
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    groups = partition(SPECS, min(len(devices), len(SPECS)))
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {pool.submit(run_group, devices[index], group, args.support_size, args.dry_run): devices[index] for index, group in enumerate(groups)}
        for future in as_completed(futures):
            for result in future.result():
                print(json.dumps(result, ensure_ascii=False), flush=True)
                results.append(result)

    summary = {
        "elapsed_seconds": time.time() - started,
        "devices": devices,
        "results": results,
        "failures": [result for result in results if result.get("exit_code") != 0],
    }
    summary_path = self_cot_root / "vector_extract_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}")
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
