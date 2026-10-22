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


@dataclass(frozen=True)
class Family:
    name: str
    target_model: str
    llm_source_model: str
    vlm_source_model: str
    num_layers: int


@dataclass(frozen=True)
class EvalDataset:
    name: str
    vector_source: str
    source: str
    num_samples: int
    batch_size_qwen: int
    batch_size_intern: int
    extra: tuple[str, ...] = ()


FAMILIES = (
    Family(
        name="qwen",
        target_model="qwen2.5-vl-7b-instruct",
        llm_source_model="qwen2.5-7b-instruct",
        vlm_source_model="qwen2.5-vl-7b-instruct",
        num_layers=28,
    ),
    Family(
        name="intern",
        target_model="internvl2_5-8b",
        llm_source_model="internlm2_5-7b-chat",
        vlm_source_model="internvl2_5-8b",
        num_layers=32,
    ),
)


EVAL_DATASETS = (
    EvalDataset(
        name="commonsenseqa",
        vector_source="commonsenseqa",
        source="/data/share/commonsenceqa/",
        num_samples=1221,
        batch_size_qwen=4,
        batch_size_intern=4,
    ),
    EvalDataset(
        name="strategyqa",
        vector_source="strategyqa",
        source="/data/share/strategy_qa/",
        num_samples=687,
        batch_size_qwen=4,
        batch_size_intern=4,
    ),
    EvalDataset(
        name="scienceqa",
        vector_source="commonsenseqa",
        source="/data/share/ScienceQA",
        num_samples=4241,
        batch_size_qwen=4,
        batch_size_intern=1,
        extra=(
            "++data.query_split=test",
            "++data.support_split=train",
            "++data.num_eval_samples=4241",
            "++data.num_support_samples=0",
        ),
    ),
)


DERIVED_TYPES = ("llm_derived", "vlm_derived")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-sizes", default="100,200,300")
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument(
        "--families",
        default="qwen,intern",
        help="Comma-separated family names to run: qwen,intern",
    )
    parser.add_argument("--derived-types", default="llm_derived,vlm_derived")
    parser.add_argument("--datasets", default="commonsenseqa,strategyqa,scienceqa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def vector_path(support_size: int, family: Family, derived_type: str, eval_dataset: EvalDataset) -> Path:
    source_model = family.llm_source_model if derived_type == "llm_derived" else family.vlm_source_model
    return (
        ROOT
        / "results"
        / "rebuttal_support_size"
        / f"support{support_size}_from_500"
        / derived_type
        / source_model
        / eval_dataset.vector_source
        / f"{source_model}_{eval_dataset.vector_source}_support{support_size}_{derived_type}_avg_cot_vector.pt"
    )


def record_root(support_size: int, family: Family, derived_type: str, eval_dataset: EvalDataset) -> Path:
    source_model = family.llm_source_model if derived_type == "llm_derived" else family.vlm_source_model
    return (
        ROOT
        / "results"
        / "rebuttal_support_size"
        / f"support{support_size}_from_500"
        / "evals"
        / "base_layerwise"
        / family.target_model
        / derived_type
        / f"from_{source_model}"
        / eval_dataset.name
        / f"vector_{eval_dataset.vector_source}"
    )


def expected_record_files(out_dir: Path, num_layers: int) -> list[Path]:
    return [out_dir / f"use_base_vector_layers_{layer}_direct_q_1.0.json" for layer in range(num_layers)]


def build_command(
    support_size: int,
    family: Family,
    derived_type: str,
    eval_dataset: EvalDataset,
    gpu: str,
) -> tuple[list[str], Path, Path]:
    vec = vector_path(support_size, family, derived_type, eval_dataset)
    out_dir = record_root(support_size, family, derived_type, eval_dataset)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    source_model = family.llm_source_model if derived_type == "llm_derived" else family.vlm_source_model
    log_path = log_dir / f"eval_support{support_size}_{family.target_model}_{derived_type}_from_{source_model}_{eval_dataset.name}.log"
    batch_size = eval_dataset.batch_size_intern if family.name == "intern" else eval_dataset.batch_size_qwen

    cmd = [
        str(PYTHON if PYTHON.exists() else sys.executable),
        str(ROOT / "scripts" / "run_eval_layers_base.py"),
        f"model_name={family.target_model}",
        f"data.name={eval_dataset.name}",
        f"++data.source={eval_dataset.source}",
        f"data.num_query_samples={eval_dataset.num_samples}",
        "data.num_shot=0",
        f"batch_size={batch_size}",
        "resume=True",
        "eval_mode=EVAL_WITH_COT_VECTOR_DIRECT_Q",
        "use_extracted_cot_vector=True",
        "use_extracted_cot_vector_type=licv",
        f"extracted_cot_vector_path={vec}",
        "static_shift_scale=1.0",
        "use_static_mu=True",
        "static_mu_value=1.0",
        "use_base_vector=True",
        "layer_eval_mode=single_layer",
        "start_layer=0",
        f"end_layer={family.num_layers}",
        f"model_layers={family.num_layers}",
        f"devices={gpu}",
        f"++record_root={out_dir}",
    ]
    if family.name == "intern" and eval_dataset.name == "scienceqa":
        cmd.extend(["++image_size=448", "++max_num_tiles=4"])
    cmd.extend(eval_dataset.extra)
    return cmd, vec, log_path


def run_one(
    support_size: int,
    family: Family,
    derived_type: str,
    eval_dataset: EvalDataset,
    gpu: str,
    dry_run: bool,
    force: bool,
) -> dict:
    vec = vector_path(support_size, family, derived_type, eval_dataset)
    out_dir = record_root(support_size, family, derived_type, eval_dataset)
    expected = expected_record_files(out_dir, family.num_layers)
    if not vec.exists():
        return {
            "status": "missing_vector",
            "support_size": support_size,
            "family": family.name,
            "derived_type": derived_type,
            "eval_dataset": eval_dataset.name,
            "vector_path": str(vec),
        }
    if not force and all(path.exists() for path in expected):
        return {
            "status": "skipped_complete",
            "support_size": support_size,
            "family": family.name,
            "derived_type": derived_type,
            "eval_dataset": eval_dataset.name,
            "record_root": str(out_dir),
        }

    cmd, _, log_path = build_command(support_size, family, derived_type, eval_dataset, gpu)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "support_size": support_size,
        "family": family.name,
        "target_model": family.target_model,
        "derived_type": derived_type,
        "eval_dataset": eval_dataset.name,
        "vector_source": eval_dataset.vector_source,
        "num_eval_samples": eval_dataset.num_samples,
        "num_layers": family.num_layers,
        "record_root": str(out_dir),
        "vector_path": str(vec),
        "command": cmd,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        return {**manifest, "status": "dry_run", "log_path": str(log_path)}

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(ROOT / ".hf_cache")
    env["HF_DATASETS_CACHE"] = str(ROOT / ".hf_cache" / "datasets")

    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n===== COMMAND =====\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    status = "ok" if proc.returncode == 0 else "failed"
    return {
        **manifest,
        "status": status,
        "exit_code": proc.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
    }


def main() -> None:
    args = parse_args()
    support_sizes = [int(x) for x in split_csv(args.support_sizes)]
    devices = split_csv(args.devices)
    requested_families = set(split_csv(args.families))
    families = [family for family in FAMILIES if family.name in requested_families]
    requested_derived_types = set(split_csv(args.derived_types))
    derived_types = [item for item in DERIVED_TYPES if item in requested_derived_types]
    requested_datasets = set(split_csv(args.datasets))
    eval_datasets = [item for item in EVAL_DATASETS if item.name in requested_datasets]
    unknown_families = requested_families - {family.name for family in FAMILIES}
    if unknown_families:
        raise ValueError(f"Unknown families: {sorted(unknown_families)}")
    if not families:
        raise ValueError("No families selected.")
    if not derived_types:
        raise ValueError("No derived types selected.")
    if not eval_datasets:
        raise ValueError("No datasets selected.")
    max_workers = args.max_workers or len(devices)
    tasks: list[tuple[int, Family, str, EvalDataset, str]] = []
    idx = 0
    for support_size in support_sizes:
        for family in families:
            for derived_type in derived_types:
                for eval_dataset in eval_datasets:
                    tasks.append((support_size, family, derived_type, eval_dataset, devices[idx % len(devices)]))
                    idx += 1

    root = ROOT / "results" / "rebuttal_support_size" / "eval_launches"
    root.mkdir(parents=True, exist_ok=True)
    family_tag = "_".join(family.name for family in families)
    summary_path = root / f"base_layerwise_{family_tag}_support_{'_'.join(map(str, support_sizes))}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"[launcher] tasks={len(tasks)} devices={devices} max_workers={max_workers}")
    print(f"[launcher] summary={summary_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool, summary_path.open("w", encoding="utf-8") as summary:
        futures = [
            pool.submit(run_one, support_size, family, derived_type, eval_dataset, gpu, args.dry_run, args.force)
            for support_size, family, derived_type, eval_dataset, gpu in tasks
        ]
        for fut in as_completed(futures):
            result = fut.result()
            print(json.dumps(result, ensure_ascii=False))
            summary.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary.flush()
            if result.get("status") == "failed":
                raise SystemExit(1)


if __name__ == "__main__":
    main()
