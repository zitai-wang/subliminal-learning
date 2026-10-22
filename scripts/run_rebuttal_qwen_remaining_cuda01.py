#!/usr/bin/env python3
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_rebuttal_support_size_base_layerwise_eval import (
    DERIVED_TYPES,
    EVAL_DATASETS,
    FAMILIES,
    expected_record_files,
    record_root,
    run_one,
)


def main() -> None:
    family = next(f for f in FAMILIES if f.name == "qwen")
    devices = ["0", "1"]
    support_sizes = [100, 200, 300]
    # This exact job is already alive in the original tmux on cuda2.
    skip_running = {(100, "llm_derived", "scienceqa")}

    tasks = []
    for support_size in support_sizes:
        for derived_type in DERIVED_TYPES:
            for eval_dataset in EVAL_DATASETS:
                key = (support_size, derived_type, eval_dataset.name)
                if key in skip_running:
                    continue
                out_dir = record_root(support_size, family, derived_type, eval_dataset)
                if all(path.exists() for path in expected_record_files(out_dir, family.num_layers)):
                    continue
                tasks.append((support_size, derived_type, eval_dataset))

    summary = Path("results/rebuttal_support_size/eval_launches/rebuttal_qwen_remaining_cuda01_summary.jsonl")
    summary.parent.mkdir(parents=True, exist_ok=True)
    print(f"[remaining] tasks={len(tasks)} devices={devices} summary={summary}", flush=True)

    def job(index: int, task):
        support_size, derived_type, eval_dataset = task
        gpu = devices[index % len(devices)]
        return run_one(
            support_size,
            family,
            derived_type,
            eval_dataset,
            gpu,
            dry_run=False,
            force=False,
        )

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(job, index, task) for index, task in enumerate(tasks)]
        for future in as_completed(futures):
            result = future.result()
            line = json.dumps(result, ensure_ascii=False)
            print(line, flush=True)
            with summary.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    print("[remaining] done", flush=True)


if __name__ == "__main__":
    main()
