#!/usr/bin/env python3
"""Train three full-CoT Qwen2.5-VL LoRA adapters and evaluate six benchmarks."""

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
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/wzy/anaconda3/envs/licv/bin/python")
MODEL = "qwen2.5-vl-7b-instruct"
DEFAULT_RUNNAME = "paper-lora-full-cot-sft-r16-s42"


@dataclass(frozen=True)
class SupportSpec:
    dataset: str
    source: str
    self_cot_path: str


@dataclass(frozen=True)
class EvalSpec:
    dataset: str
    support_dataset: str
    source: str
    eval_samples: int
    max_new_tokens: int
    batch_size: int
    extra_overrides: tuple[str, ...] = ()
    use_cot: bool = True
    direct_answer_max_new_tokens: int | None = None


SUPPORTS = (
    SupportSpec(
        "gsm8k",
        "/data/share/datasets/gsm8k",
        str(ROOT / "output/gsm8k/self_cot_qwen2.5-7b-instruct_train500_correct_only_aligned458.json"),
    ),
    SupportSpec(
        "commonsenseqa",
        "/data/share/commonsenceqa/",
        str(ROOT / "output/commonsenseqa/self_cot_qwen2.5-7b-instruct_train500_20260417_163423_correct_only.json"),
    ),
    SupportSpec(
        "strategyqa",
        "/data/share/strategy_qa/",
        str(ROOT / "output/strategyqa/self_cot_qwen2.5-7b-instruct_train500_20260417_171124_correct_only.json"),
    ),
)


EVALS = (
    EvalSpec("gsm8k", "gsm8k", "/data/share/datasets/gsm8k", 1319, 512, 4),
    EvalSpec(
        "mathvista",
        "gsm8k",
        "/data/share/datasets/mathvista",
        1000,
        4096,
        1,
        ("++data.query_split=testmini", "++data.support_split=testmini"),
    ),
    EvalSpec(
        "mathvision",
        "gsm8k",
        "/data/share/MathVision",
        304,
        4096,
        1,
        ("++data.query_split=testmini", "++data.support_split=testmini"),
    ),
    EvalSpec(
        "commonsenseqa",
        "commonsenseqa",
        "/data/share/commonsenceqa/",
        1221,
        512,
        4,
    ),
    EvalSpec(
        "strategyqa",
        "strategyqa",
        "/data/share/strategy_qa/",
        687,
        512,
        4,
    ),
    EvalSpec(
        "scienceqa",
        "commonsenseqa",
        "/data/share/ScienceQA",
        4241,
        512,
        1,
        (
            "++data.query_split=validation",
            "++data.support_split=train",
            "++data.num_eval_samples=4241",
            "++data.num_support_samples=0",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "train", "eval", "all"), default="all")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--runname", default=DEFAULT_RUNNAME)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def devices(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("At least one CUDA device is required.")
    return parsed


def output_root(args: argparse.Namespace) -> Path:
    return ROOT / "results" / "lora_baselines" / args.runname


def environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(ROOT / ".hf_cache")
    env["HF_DATASETS_CACHE"] = str(ROOT / ".hf_cache" / "datasets")
    return env


def expanded_runname(args: argparse.Namespace, support_dataset: str, train_samples: int | None = None) -> str:
    samples = args.train_samples if train_samples is None else train_samples
    return f"{args.runname}-{MODEL}-{support_dataset}-{samples}"


def checkpoint_path(args: argparse.Namespace, support_dataset: str) -> Path:
    return ROOT / "results" / "ckpt" / expanded_runname(args, support_dataset) / f"epoch-{args.epochs - 1}"


def train_command(
    args: argparse.Namespace,
    spec: SupportSpec,
    train_samples: int | None = None,
    epochs: int | None = None,
    runname: str | None = None,
) -> list[str]:
    samples = args.train_samples if train_samples is None else train_samples
    num_epochs = args.epochs if epochs is None else epochs
    effective_runname = args.runname if runname is None else runname
    return [
        str(PYTHON),
        "src/train.py",
        f"runname={effective_runname}",
        f"model_name={MODEL}",
        f"data.name={spec.dataset}",
        f"data.num_query_samples={samples}",
        "data.num_shot=0",
        f"data.seed={args.seed}",
        f"++data.source={spec.source}",
        "data.use_self_cot=true",
        f"data.self_cot_path={spec.self_cot_path}",
        "++data.student_target=full_cot",
        "training-mode=TRAIN_STUDENT_DIRECT_Q_SELF_COT",
        "data.training_mode=TRAIN_STUDENT_DIRECT_Q_SELF_COT",
        "++data.train_use_images=false",
        "++data.max_seq_len=1536",
        "dtype=torch.float16",
        "precision=16-mixed",
        "batch_size=1",
        "accumulate_grad_batches=4",
        f"epochs={num_epochs}",
        f"lr={args.learning_rate}",
        "weight_decay=0.001",
        "warmup_step=0.1",
        "grad_clip_val=1.0",
        "method_name=lora_full_cot_sft",
        "loss_mode=ce_only",
        "only_alignment_loss=false",
        "only_shift_at_layer=null",
        "++enable_gradient_checkpointing=true",
        "++enable_low_memory_forward=true",
        "+encoder=lora",
        "+peft=lora",
        f"peft.r={args.rank}",
    ]


def eval_command(args: argparse.Namespace, spec: EvalSpec) -> list[str]:
    ckpt = checkpoint_path(args, spec.support_dataset)
    record_root = output_root(args) / "records" / spec.dataset
    return [
        str(PYTHON),
        "src/eval_baseline.py",
        "--config-name",
        "eval_baseline",
        f"model_name={MODEL}",
        f"data.name={spec.dataset}",
        f"data.num_query_samples={spec.eval_samples}",
        f"++data.source={spec.source}",
        "data.seed=42",
        "data.num_shot=0",
        f"ckpt_path={ckpt}",
        f"++record_root={record_root}",
        "mode=eval",
        "resume=False",
        "dtype=torch.float16",
        f"batch_size={spec.batch_size}",
        f"use_cot={str(spec.use_cot)}",
        "eval_mode=EVAL_WITH_COT_VECTOR_DIRECT_Q",
        "use_extracted_cot_vector=False",
        "only_shift_at_layer=null",
        "generation_args.num_beams=1",
        f"generation_args.max_new_tokens={spec.max_new_tokens}",
        *(
            [f"+direct_answer_max_new_tokens={spec.direct_answer_max_new_tokens}"]
            if spec.direct_answer_max_new_tokens is not None
            else []
        ),
        "generation_args.length_penalty=0.0",
        "generation_args.do_sample=False",
        "generation_args.temperature=0.0",
        "generation_args.top_p=1.0",
        "+encoder=lora",
        "+peft=lora",
        *spec.extra_overrides,
    ]


def run_logged(command: list[str], gpu: str, log_path: Path, dry_run: bool) -> int:
    rendered = " ".join(command)
    print(f"[GPU {gpu}] {rendered}", flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"GPU={gpu}\nCOMMAND={rendered}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def verify_support_files() -> None:
    missing = [spec.self_cot_path for spec in SUPPORTS if not Path(spec.self_cot_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing support files: {missing}")


def run_smoke(args: argparse.Namespace, gpu: str) -> None:
    spec = SUPPORTS[0]
    smoke_runname = f"{args.runname}-smoke"
    command = train_command(args, spec, train_samples=8, epochs=1, runname=smoke_runname)
    log_path = output_root(args) / "logs" / "smoke_gsm8k.log"
    code = run_logged(command, gpu, log_path, args.dry_run)
    if code:
        raise RuntimeError(f"Smoke test failed with exit code {code}; see {log_path}")
    if not args.dry_run:
        final_dir = ROOT / "results" / "ckpt" / f"{smoke_runname}-{MODEL}-gsm8k-8" / "final"
        required = (final_dir / "adapter_config.json", final_dir / "adapter_model.safetensors")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Smoke test did not produce a loadable LoRA adapter: {missing}")
        print(f"Smoke test completed: {final_dir}")


def run_train(args: argparse.Namespace, gpu_ids: list[str]) -> None:
    logs = output_root(args) / "logs"
    jobs = []
    for index, spec in enumerate(SUPPORTS):
        gpu = gpu_ids[index % len(gpu_ids)]
        command = train_command(args, spec)
        log_path = logs / f"train_{spec.dataset}.log"
        jobs.append((spec.dataset, gpu, command, log_path))

    failures = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(run_logged, command, gpu, log_path, args.dry_run): (name, log_path)
            for name, gpu, command, log_path in jobs
        }
        for future in as_completed(futures):
            name, log_path = futures[future]
            code = future.result()
            print(f"[Train finished] {name}: exit_code={code}", flush=True)
            if code:
                failures.append((name, code, str(log_path)))

    if failures:
        raise RuntimeError(f"LoRA training failures: {failures}")
    if not args.dry_run:
        missing = [
            str(checkpoint_path(args, spec.dataset))
            for spec in SUPPORTS
            if not (checkpoint_path(args, spec.dataset) / "adapter_config.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"Missing trained LoRA checkpoints: {missing}")


def partition(items: Iterable[EvalSpec], count: int) -> list[list[EvalSpec]]:
    groups = [[] for _ in range(count)]
    for index, item in enumerate(items):
        groups[index % count].append(item)
    return groups


def run_eval_group(args: argparse.Namespace, gpu: str, specs: list[EvalSpec]) -> list[tuple[str, int, str]]:
    failures = []
    for spec in specs:
        command = eval_command(args, spec)
        log_path = output_root(args) / "logs" / f"eval_{spec.dataset}.log"
        code = run_logged(command, gpu, log_path, args.dry_run)
        print(f"[Eval finished] {spec.dataset}: exit_code={code}", flush=True)
        if code:
            failures.append((spec.dataset, code, str(log_path)))
    return failures


def run_eval(args: argparse.Namespace, gpu_ids: list[str]) -> None:
    if not args.dry_run:
        for spec in SUPPORTS:
            ckpt = checkpoint_path(args, spec.dataset)
            if not (ckpt / "adapter_config.json").is_file():
                raise FileNotFoundError(f"LoRA checkpoint is missing: {ckpt}")

    groups = partition(EVALS, min(len(gpu_ids), len(EVALS)))
    failures = []
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = {
            pool.submit(run_eval_group, args, gpu_ids[index], group): gpu_ids[index]
            for index, group in enumerate(groups)
        }
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        raise RuntimeError(f"LoRA evaluation failures: {failures}")


def write_manifest(args: argparse.Namespace, gpu_ids: list[str], started_at: float) -> None:
    root = output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL,
        "method": "LoRA answer-only SFT",
        "runname": args.runname,
        "seed": args.seed,
        "rank": args.rank,
        "lora_alpha": args.rank * 2,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "nominal_support_samples": args.train_samples,
        "devices": gpu_ids,
        "support_mapping": {
            spec.dataset: spec.support_dataset for spec in EVALS
        },
        "elapsed_seconds": time.time() - started_at,
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    gpu_ids = devices(args.devices)
    verify_support_files()
    started_at = time.time()

    if args.stage == "smoke":
        run_smoke(args, gpu_ids[0])
    elif args.stage == "train":
        run_train(args, gpu_ids)
    elif args.stage == "eval":
        run_eval(args, gpu_ids)
    else:
        run_train(args, gpu_ids)
        run_eval(args, gpu_ids)

    if not args.dry_run:
        write_manifest(args, gpu_ids, started_at)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_qwenvl_lora_baselines] ERROR: {exc}", file=sys.stderr)
        raise
