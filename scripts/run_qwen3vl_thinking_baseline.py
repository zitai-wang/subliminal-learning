#!/usr/bin/env python3
"""Run the Qwen3-VL-8B-Thinking ICL baseline on 6 datasets, GPU 3 only.

Add-only companion to run_qwenvl_lora_baselines.py: reuses the same EvalSpec /
run_logged pattern but targets src/eval_qwen3vl_baseline.py (a new, standalone
script) instead of eval_baseline.py, and hardcodes CUDA_VISIBLE_DEVICES=3 so it
never touches the other GPUs other experiments are running on.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/wzy/anaconda3/envs/licv/bin/python")
MODEL = "qwen3-vl-8b-thinking"
GPU = "3"


@dataclass(frozen=True)
class EvalSpec:
    dataset: str
    source: str
    eval_samples: int
    max_new_tokens: int
    extra_overrides: tuple[str, ...] = field(default_factory=tuple)
    direct_answer_max_new_tokens: int | None = None


EVALS = (
    EvalSpec("gsm8k", "/data/share/datasets/gsm8k", 1319, 512),
    EvalSpec(
        "mathvista",
        "/data/share/datasets/mathvista",
        1000,
        2048,
        ("++data.query_split=testmini", "++data.support_split=testmini"),
    ),
    EvalSpec(
        "mathvision",
        "/data/share/MathVision",
        304,
        4096,
        ("++data.query_split=testmini", "++data.support_split=testmini"),
    ),
    EvalSpec(
        "commonsenseqa",
        "/data/share/commonsenceqa/",
        1221,
        512,
        direct_answer_max_new_tokens=512,
    ),
    EvalSpec(
        "strategyqa",
        "/data/share/strategy_qa/",
        687,
        512,
        direct_answer_max_new_tokens=512,
    ),
    EvalSpec(
        "scienceqa",
        "/data/share/ScienceQA",
        4241,
        512,
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
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[spec.dataset for spec in EVALS],
        help="Subset of dataset names to run (default: all 6).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=(
            "If set, cap thinking to this many tokens per sample and force a "
            "final-answer continuation with the remaining max_new_tokens "
            "budget instead of letting the model think unconstrained. "
            "Speeds up Qwen3-VL-Thinking generation; null/unset keeps the "
            "original unconstrained-thinking behavior."
        ),
    )
    return parser.parse_args()


def output_root() -> Path:
    return ROOT / "results" / "qwen3vl_thinking_baseline"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(ROOT / ".hf_cache")
    env["HF_DATASETS_CACHE"] = str(ROOT / ".hf_cache" / "datasets")
    return env


def eval_command(spec: EvalSpec, thinking_budget: int | None = None) -> list[str]:
    return [
        str(PYTHON),
        "src/eval_qwen3vl_baseline.py",
        "--config-name",
        "eval_qwen3vl_baseline",
        f"model_name={MODEL}",
        f"data.name={spec.dataset}",
        f"data.num_query_samples={spec.eval_samples}",
        f"++data.source={spec.source}",
        "data.seed=42",
        "data.num_shot=0",
        "resume=True",
        "dtype=torch.bfloat16",
        "batch_size=1",
        "use_cot=False",
        "eval_mode=EVAL_BASELINE",
        "use_extracted_cot_vector=False",
        "generation_args.num_beams=1",
        f"generation_args.max_new_tokens={spec.max_new_tokens}",
        "generation_args.do_sample=False",
        "generation_args.temperature=0.0",
        "generation_args.top_p=1.0",
        "generation_args.repetition_penalty=1.05",
        *(
            [f"+direct_answer_max_new_tokens={spec.direct_answer_max_new_tokens}"]
            if spec.direct_answer_max_new_tokens is not None
            else []
        ),
        *(
            [f"generation_args.thinking_budget={thinking_budget}"]
            if thinking_budget is not None
            else []
        ),
        *spec.extra_overrides,
    ]


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> int:
    rendered = " ".join(command)
    print(f"[GPU {GPU}] {rendered}", flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"GPU={GPU}\nCOMMAND={rendered}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def read_accuracy(dataset: str) -> float | None:
    record_path = output_root() / dataset / "0shot.json"
    if not record_path.exists():
        return None
    try:
        with record_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("eval_result", {}).get("accuracy")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    args = parse_args()
    specs = [spec for spec in EVALS if spec.dataset in args.datasets]
    if not specs:
        raise ValueError(f"No matching datasets for {args.datasets}")

    logs = output_root() / "logs"
    failures: list[tuple[str, int, str]] = []
    for spec in specs:
        command = eval_command(spec, args.thinking_budget)
        log_path = logs / f"eval_{spec.dataset}.log"
        code = run_logged(command, log_path, args.dry_run)
        print(f"[Eval finished] {spec.dataset}: exit_code={code}", flush=True)
        if code:
            failures.append((spec.dataset, code, str(log_path)))

    print("\n=== Qwen3-VL-8B-Thinking baseline summary ===")
    for spec in specs:
        acc = read_accuracy(spec.dataset)
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
        print(f"{spec.dataset}: accuracy={acc_str}")

    if failures:
        raise RuntimeError(f"Baseline evaluation failures: {failures}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_qwen3vl_thinking_baseline] ERROR: {exc}", file=sys.stderr)
        raise
