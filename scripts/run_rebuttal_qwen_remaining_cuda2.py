#!/usr/bin/env python3
from __future__ import annotations

import json
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
    # Already alive in rebuttal_qwen_remaining_cuda01.
    skip_running = {
        (100, "vlm_derived", "scienceqa"),
        (200, "vlm_derived", "scienceqa"),
    }
    tasks = []
    for support_size in [100, 200, 300]:
        for derived_type in DERIVED_TYPES:
            for eval_dataset in EVAL_DATASETS:
                key = (support_size, derived_type, eval_dataset.name)
                if key in skip_running:
                    continue
                out_dir = record_root(support_size, family, derived_type, eval_dataset)
                if all(path.exists() for path in expected_record_files(out_dir, family.num_layers)):
                    continue
                tasks.append((support_size, derived_type, eval_dataset))

    summary = Path("results/rebuttal_support_size/eval_launches/rebuttal_qwen_remaining_cuda2_summary.jsonl")
    summary.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cuda2-remaining] tasks={len(tasks)} device=2 summary={summary}", flush=True)
    for support_size, derived_type, eval_dataset in tasks:
        result = run_one(
            support_size,
            family,
            derived_type,
            eval_dataset,
            "2",
            dry_run=False,
            force=False,
        )
        line = json.dumps(result, ensure_ascii=False)
        print(line, flush=True)
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print("[cuda2-remaining] done", flush=True)


if __name__ == "__main__":
    main()
