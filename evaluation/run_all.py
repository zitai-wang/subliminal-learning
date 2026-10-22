import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

import csv
import importlib
import json
import os
import re
import shlex
import sys
import time

from types import SimpleNamespace

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from evaluation.utils import (
    DEFAULT_SYSTEM_PROMPT,
    format_judge_summary,
    init_judge_client_or_raise,
    load_system_prompt,
    new_judge_stats,
)

SUITES = {
    "mathvista": {
        "eval_module": "evaluation.mathvista.evaluate_mathvista",
        "extract_module": "evaluation.mathvista.extract_calculate",
    },
    "mathvision": {
        "eval_module": "evaluation.mathvision.evaluate_mathvision",
        "extract_module": "evaluation.mathvision.extract_calculate",
    },
    "mathverse": {
        "eval_module": "evaluation.mathverse.evaluate_mathverse",
        "extract_module": "evaluation.mathverse.extract_calculate",
    },
    "mmk12": {
        "eval_module": "evaluation.mmk12.evaluate_mmk12",
        "extract_module": "evaluation.mmk12.extract_calculate",
    },
    "olympiadbench": {
        "eval_module": "evaluation.olympiadbench.evaluate_olympiadbench",
        "extract_module": "evaluation.olympiadbench.extract_calculate",
    },
}

_POST_CHECKS = {}


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def print_header(title):
    line = "=" * 88
    print(f"\n{line}\n[{now_text()}] {title}\n{line}", flush=True)


def print_subheader(title):
    line = "-" * 88
    print(f"\n{line}\n[{now_text()}] {title}\n{line}", flush=True)


def print_kv(key, value):
    print(f"  - {key:<18}: {value}", flush=True)


def parse_dataset_name(result_file):
    match = re.match(r"(.+)_\d{12}\.json$", result_file)
    if match:
        return match.group(1)
    return result_file.replace(".json", "")


def normalize_model_name(model_name):
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", model_name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "model"


def parse_stop_token_ids(value):
    if not value:
        return None
    token_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    return token_ids or None


def to_suite_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def to_dataset_specs(value):
    if value is None:
        return {}
    normalized = {}
    for suite_name, suite_cfg in dict(value).items():
        suite_name = str(suite_name).strip()
        if not suite_name:
            continue
        suite_cfg = dict(suite_cfg or {})
        datasets = {}
        for ds_name, ds_cfg in dict(suite_cfg.get("datasets", {}) or {}).items():
            ds_name = str(ds_name).strip()
            if not ds_name:
                continue
            ds_cfg = dict(ds_cfg or {})
            datasets[ds_name] = {
                "split": ds_cfg.get("split"),
                "name": ds_cfg.get("name"),
                "source": ds_cfg.get("source"),
            }
        normalized[suite_name] = {
            "source": str(suite_cfg.get("source", "")).strip(),
            "datasets": datasets,
        }
    return normalized


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_accuracy_0_100(accuracy_value, correct_value, total_value):
    if correct_value is not None and total_value not in (None, 0):
        return round(correct_value / total_value * 100, 4)
    if accuracy_value is None:
        return None
    if 0 <= accuracy_value <= 1:
        return round(accuracy_value * 100, 4)
    return round(accuracy_value, 4)


def infer_correctness(suite_name, sample):
    if "score" in sample:
        return bool(sample["score"])
    try:
        if suite_name not in _POST_CHECKS:
            if suite_name == "mathvista":
                from evaluation.mathvista.extract_calculate import post_check

                _POST_CHECKS[suite_name] = post_check
            elif suite_name == "mathvision":
                from evaluation.mathvision.extract_calculate import post_check

                _POST_CHECKS[suite_name] = post_check
            else:
                _POST_CHECKS[suite_name] = None
        if suite_name == "mathvista":
            return bool(_POST_CHECKS[suite_name](sample, prefetch=False))
        if suite_name == "mathvision":
            return bool(_POST_CHECKS[suite_name](sample, prefetch=False))
    except Exception:
        return None
    return None


def add_correctness(suite_name, dataset_name, result_file, samples):
    _ = dataset_name, result_file
    updated_count = 0
    if isinstance(samples, dict):
        updated_samples = {}
        iterator = samples.items()
    else:
        updated_samples = []
        iterator = enumerate(samples)

    for sample_id, sample in iterator:
        if not isinstance(sample, dict):
            continue
        sample_copy = dict(sample)
        is_correct = infer_correctness(suite_name, sample_copy)
        sample_copy["is_correct"] = is_correct
        if is_correct is not None and "score" not in sample_copy:
            sample_copy["score"] = is_correct
        updated_count += 1
        if isinstance(updated_samples, dict):
            updated_samples[sample_id] = sample_copy
        else:
            updated_samples.append(sample_copy)
    return updated_samples, updated_count


def inject_correctness_to_raw(raw_samples, scored_samples):
    if not isinstance(raw_samples, dict) or not isinstance(scored_samples, dict):
        return raw_samples
    for sample_id, sample in raw_samples.items():
        scored = scored_samples.get(sample_id)
        if not isinstance(sample, dict) or not isinstance(scored, dict):
            continue
        if "is_correct" in scored:
            sample["is_correct"] = scored["is_correct"]
        if "score" in scored and "score" not in sample:
            sample["score"] = scored["score"]
    return raw_samples


def append_metric_row(metric_rows, suite_name, dataset_name, group, name, accuracy, correct, total, score_file):
    accuracy_value = to_float(accuracy)
    correct_value = to_int(correct)
    total_value = to_int(total)
    accuracy_0_100 = normalize_accuracy_0_100(accuracy_value, correct_value, total_value)
    level = "detail"
    if str(name).lower() in {"overall", "average"}:
        level = "overall"
    metric_rows.append(
        {
            "suite": suite_name,
            "dataset": dataset_name,
            "level": level,
            "metric_group": group,
            "metric_name": name,
            "accuracy": accuracy_0_100,
            "correct": correct_value,
            "total": total_value,
            "score_file": score_file,
        }
    )


def flatten_score(metric_rows, suite_name, dataset_name, score_data, score_file, path=None):
    if path is None:
        path = []
    if isinstance(score_data, list):
        for idx, item in enumerate(score_data):
            if isinstance(item, dict):
                name_key = None
                for key in ("Task&Skill", "Subject", "name", "metric"):
                    if key in item:
                        name_key = key
                        break
                metric_name = item.get(name_key, f"item_{idx}")
                append_metric_row(
                    metric_rows,
                    suite_name,
                    dataset_name,
                    path[-1] if path else "table",
                    metric_name,
                    item.get("acc", item.get("accuracy")),
                    item.get("hit", item.get("correct")),
                    item.get("tot", item.get("total")),
                    score_file,
                )
        return

    if isinstance(score_data, dict):
        has_metric = any(key in score_data for key in ("acc", "accuracy", "hit", "correct", "tot", "total"))
        if has_metric:
            metric_name = path[-1] if path else "overall"
            metric_group = "/".join(path[:-1]) if len(path) > 1 else "overall"
            append_metric_row(
                metric_rows,
                suite_name,
                dataset_name,
                metric_group,
                metric_name,
                score_data.get("acc", score_data.get("accuracy")),
                score_data.get("hit", score_data.get("correct")),
                score_data.get("tot", score_data.get("total")),
                score_file,
            )
            return
        for key, value in score_data.items():
            flatten_score(metric_rows, suite_name, dataset_name, value, score_file, path + [str(key)])


def write_metrics_csv(path, metric_rows):
    fieldnames = [
        "suite",
        "dataset",
        "level",
        "metric_group",
        "metric_name",
        "accuracy",
        "correct",
        "total",
        "score_file",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


def build_shared_runtime(args, suite_names):
    from transformers import AutoProcessor
    from vllm import LLM

    llm_kwargs = {
        "model": args.checkpoint,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": 16384,
    }
    if "olympiadbench" in suite_names:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 8}
    llm = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = parse_stop_token_ids(args.gen_stop_token_ids)
    return llm, processor, stop_token_ids


def run_generation_stage(args, suite_names, run_out_dir, llm, processor, stop_token_ids, dataset_specs_by_suite):
    generated = []
    total_suites = len(suite_names)
    for idx, suite_name in enumerate(suite_names, start=1):
        config = SUITES[suite_name]
        suite_spec = dataset_specs_by_suite[suite_name]
        suite_datasets = list(suite_spec["datasets"].keys())
        suite_out_dir = os.path.join(run_out_dir, suite_name)
        os.makedirs(suite_out_dir, exist_ok=True)
        print_subheader(f"Generation {idx}/{total_suites}: {suite_name}")
        print_kv("datasets", ",".join(suite_datasets))
        print_kv("out_dir", suite_out_dir)
        print_kv("dataset_source", suite_spec.get("source") or "default_root")
        suite_start = time.time()

        eval_module = importlib.import_module(config["eval_module"])
        eval_args = SimpleNamespace(
            datasets=suite_datasets,
            out_dir=suite_out_dir,
            seed=args.seed,
            limit=args.limit,
            gen_temperature=args.gen_temperature,
            gen_max_tokens=args.gen_max_tokens,
            system_prompt=getattr(args, "system_prompt", ""),
            system_prompt_text=getattr(args, "system_prompt_text", ""),
            dataset_source=suite_spec.get("source", ""),
            dataset_entries=suite_spec.get("datasets", {}),
        )
        result_files = eval_module.evaluate_chat_model(eval_args, llm, processor, stop_token_ids)
        if not result_files:
            raise RuntimeError(f"No raw result files generated for suite '{suite_name}' in {suite_out_dir}")
        print_kv("generated_files", len(result_files))
        print_kv("elapsed_sec", f"{time.time() - suite_start:.1f}")
        for result_file in result_files:
            print_kv("result_file", result_file)
            generated.append(
                {
                    "suite_name": suite_name,
                    "suite_out_dir": suite_out_dir,
                    "result_file": result_file,
                }
            )
    return generated


@hydra.main(config_path="conf", config_name="run_all", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    args = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    if not args.checkpoint:
        raise ValueError("checkpoint must be set. Example: checkpoint=dataset/models/Qwen2.5-VL-7B-Instruct")
    system_prompt_text = load_system_prompt(
        getattr(args, "system_prompt", ""),
        default_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    args.system_prompt_text = system_prompt_text

    run_start_time = time.time()
    run_start_text = now_text()

    base_out_dir = args.out_dir
    if not os.path.isabs(base_out_dir):
        base_out_dir = os.path.join(PROJECT_ROOT, base_out_dir)
    model_name = args.model_name or os.path.basename(args.checkpoint.rstrip("/"))
    model_name = normalize_model_name(model_name)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_out_dir = os.path.join(base_out_dir, model_name, run_timestamp)
    os.makedirs(run_out_dir, exist_ok=True)

    suite_names = to_suite_list(args.suites)
    unknown = [name for name in suite_names if name not in SUITES]
    if unknown:
        raise ValueError(f"Unknown suites: {unknown}. Supported: {list(SUITES.keys())}")
    dataset_specs = to_dataset_specs(args.dataset_specs)
    unknown_map_keys = [name for name in dataset_specs if name not in SUITES]
    if unknown_map_keys:
        raise ValueError(f"Unknown suites in dataset_specs: {unknown_map_keys}. Supported: {list(SUITES.keys())}")
    dataset_specs_by_suite = {}
    missing_specs = []
    for suite_name in suite_names:
        suite_spec = dataset_specs.get(suite_name)
        if not suite_spec or not suite_spec.get("datasets"):
            missing_specs.append(suite_name)
            continue
        dataset_specs_by_suite[suite_name] = suite_spec
    if missing_specs:
        raise ValueError(f"Missing dataset_specs for suites: {missing_specs}")

    print_header("Run Configuration")
    print_kv("checkpoint", args.checkpoint)
    print_kv("out_dir", base_out_dir)
    print_kv("model_name", model_name)
    print_kv("run_timestamp", run_timestamp)
    print_kv("run_out_dir", run_out_dir)
    print_kv("suites", ",".join(suite_names))
    print_kv(
        "dataset_specs",
        {
            k: {"source": v.get("source"), "datasets": list(v.get("datasets", {}).keys())}
            for k, v in dataset_specs_by_suite.items()
        }
        or "None",
    )
    print_kv("seed", args.seed)
    print_kv("tensor_parallel", args.tensor_parallel_size)
    print_kv("limit", args.limit)
    print_kv("gen_temperature", args.gen_temperature)
    print_kv("gen_max_tokens", args.gen_max_tokens)
    print_kv("gen_stop_token_ids", args.gen_stop_token_ids or "None")
    print_kv("system_prompt_input", getattr(args, "system_prompt", ""))
    prompt_preview = system_prompt_text.replace("\n", "\\n")
    if len(prompt_preview) > 120:
        prompt_preview = f"{prompt_preview[:117]}..."
    print_kv("system_prompt", prompt_preview)
    print_kv("judge_model_env", os.environ.get("JUDGE_MODEL", ""))
    print_kv("judge_base_url_env", os.environ.get("JUDGE_MODEL_BASE_URL", ""))
    print_kv("all_metrics_policy", "overall/average only")

    print_header("Judge Preflight")
    judge_client, judge_model, judge_base_url = init_judge_client_or_raise()
    judge_stats = new_judge_stats()
    print_kv("status", "ok")
    print_kv("model", judge_model)
    print_kv("base_url", judge_base_url)


    print_header("Stage 1/2: Generation (All Suites)")
    stage1_start = time.time()
    llm, processor, stop_token_ids = build_shared_runtime(args, suite_names)
    generated_jobs = run_generation_stage(
        args,
        suite_names,
        run_out_dir,
        llm,
        processor,
        stop_token_ids,
        dataset_specs_by_suite,
    )
    stage1_elapsed = time.time() - stage1_start
    print(f"[{now_text()}] Stage 1 complete ({stage1_elapsed:.1f}s), jobs={len(generated_jobs)}", flush=True)

    print_header("Stage 2/2: Extraction + Scoring + Aggregation")
    stage2_start = time.time()
    all_metric_rows = []
    total_jobs = len(generated_jobs)
    total_correctness_updates = 0
    for idx, job in enumerate(generated_jobs, start=1):
        suite_name = job["suite_name"]
        suite_out_dir = job["suite_out_dir"]
        result_file = job["result_file"]
        config = SUITES[suite_name]
        print_subheader(f"Scoring {idx}/{total_jobs}: {suite_name} | {result_file}")
        print_kv("suite_out_dir", suite_out_dir)
        extract_module = importlib.import_module(config["extract_module"])
        if not hasattr(extract_module, "run_extract"):
            raise RuntimeError(f"Missing run_extract() in {config['extract_module']}")
        extract_start = time.time()
        extract_result = extract_module.run_extract(
            output_dir=suite_out_dir,
            output_file=result_file,
            response_label="response",
            number=-1,
            output_label="extract",
            init_judge=False,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_stats=judge_stats,
        )
        print_kv("extract_elapsed_sec", f"{time.time() - extract_start:.1f}")

        dataset_name = parse_dataset_name(result_file)
        raw_path = os.path.join(suite_out_dir, result_file)
        extract_path = raw_path.replace(".json", "_extract.json")
        score_path = raw_path.replace(".json", "_score.json")
        if isinstance(extract_result, dict):
            extract_path = extract_result.get("extract_file", extract_path)
            score_path = extract_result.get("score_file", score_path)

        prediction_source_path = extract_path if os.path.exists(extract_path) else raw_path
        samples = read_json(prediction_source_path)
        samples_with_correctness, updated_count = add_correctness(
            suite_name=suite_name,
            dataset_name=dataset_name,
            result_file=result_file,
            samples=samples,
        )
        total_correctness_updates += updated_count
        write_json(prediction_source_path, samples_with_correctness)
        if os.path.exists(raw_path) and prediction_source_path != raw_path:
            raw_samples = read_json(raw_path)
            raw_samples = inject_correctness_to_raw(raw_samples, samples_with_correctness)
            write_json(raw_path, raw_samples)

        if os.path.exists(score_path):
            score_data = read_json(score_path)
            flatten_score(all_metric_rows, suite_name, dataset_name, score_data, os.path.basename(score_path))
        print_kv("metric_rows_so_far", len(all_metric_rows))
        print_kv("correctness_updates_so_far", total_correctness_updates)
    judge_summary_text = format_judge_summary(judge_stats)
    print(judge_summary_text, flush=True)

    key_metric_rows = [row for row in all_metric_rows if row["level"] == "overall"]
    if not key_metric_rows:
        key_metric_rows = all_metric_rows
    metrics_path = os.path.join(run_out_dir, "all_metrics.csv")
    write_metrics_csv(metrics_path, key_metric_rows)

    meta_info = {
        "run": {
            "started_at": run_start_text,
            "finished_at": now_text(),
            "elapsed_sec": round(time.time() - run_start_time, 2),
            "timestamp": run_timestamp,
            "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        },
        "config": {
            "hydra": OmegaConf.to_container(cfg, resolve=True),
            "checkpoint": args.checkpoint,
            "model_name": model_name,
            "out_dir": base_out_dir,
            "run_out_dir": run_out_dir,
            "suites": suite_names,
            "suite_datasets": {suite: list(dataset_specs_by_suite[suite]["datasets"].keys()) for suite in suite_names},
            "dataset_specs": dataset_specs_by_suite,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "limit_per_dataset": args.limit,
            "generation": {
                "temperature": args.gen_temperature,
                "max_tokens": args.gen_max_tokens,
                "stop_token_ids": parse_stop_token_ids(args.gen_stop_token_ids),
                "system_prompt": {
                    "input": getattr(args, "system_prompt", ""),
                    "text": system_prompt_text,
                },
            },
            "judge": {
                "model_env": judge_model,
                "base_url_env": judge_base_url,
                "api_key_set": bool(os.environ.get("JUDGE_MODEL_API_KEY", "")),
                "preflight_model": judge_model,
                "summary": judge_summary_text,
            },
            "metrics_policy": "overall/average only",
        },
        "artifacts": {
            "all_metrics": metrics_path,
            "generated_jobs": generated_jobs,
        },
        "summary": {
            "all_metric_rows_full": len(all_metric_rows),
            "all_metric_rows_key_only": len(key_metric_rows),
            "correctness_updates": total_correctness_updates,
            "stage1_elapsed_sec": round(stage1_elapsed, 2),
            "stage2_elapsed_sec": round(time.time() - stage2_start, 2),
        },
    }
    meta_info_path = os.path.join(run_out_dir, "meta_info.json")
    write_json(meta_info_path, meta_info)

    print_header("Run Finished")
    print_kv("stage2_elapsed_sec", f"{time.time() - stage2_start:.1f}")
    print_kv("all_metric_rows_full", len(all_metric_rows))
    print_kv("all_metric_rows_key", len(key_metric_rows))
    print_kv("correctness_updates", total_correctness_updates)
    print_kv("metrics_table", metrics_path)
    print_kv("meta_info", meta_info_path)


if __name__ == "__main__":
    main()
