#!/usr/bin/env python3
"""Paired post-processing for the caption-matched LLM-vs-VLM ablation."""
import argparse
import json
import math
from pathlib import Path
import random


ROOT = Path("/data1/wzy/sub-learn")
RECORD_ROOT = ROOT / "results" / "record" / "caption_text_only"
LLM = "qwen2.5-7b-instruct"
VLM = "qwen2.5-vl-7b-instruct"


def load_records(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    keyed = {str(row.get("id", row.get("pid"))): row for row in records}
    if None in keyed or len(keyed) != len(records):
        raise ValueError(f"Records in {path} lack a unique id/pid key")
    return keyed


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for McNemar's test."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def breakdown(records_a: dict, records_b: dict, field: str) -> dict:
    groups = {}
    for pid, record in records_a.items():
        groups.setdefault(str(record.get(field, "")), []).append(pid)
    rows = {}
    for label, pids in sorted(groups.items()):
        n = len(pids)
        a_correct = sum(bool(records_a[pid]["is_correct"]) for pid in pids)
        b_correct = sum(bool(records_b[pid]["is_correct"]) for pid in pids)
        rows[label] = {
            "n": n,
            "llm_accuracy": a_correct / n,
            "vlm_accuracy": b_correct / n,
            "delta_vlm_minus_llm": (b_correct - a_correct) / n,
            "llm_only_correct": sum(bool(records_a[p]["is_correct"]) and not bool(records_b[p]["is_correct"]) for p in pids),
            "vlm_only_correct": sum(not bool(records_a[p]["is_correct"]) and bool(records_b[p]["is_correct"]) for p in pids),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mathvision", choices=("mathvista", "mathvision"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = {
        LLM: RECORD_ROOT / f"{args.dataset}_{LLM}" / "0shot.json",
        VLM: RECORD_ROOT / f"{args.dataset}_{VLM}" / "0shot.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation record(s): " + ", ".join(missing))

    llm, vlm = load_records(paths[LLM]), load_records(paths[VLM])
    if set(llm) != set(vlm):
        raise ValueError(f"Record IDs differ: LLM={len(llm)}, VLM={len(vlm)}, overlap={len(set(llm) & set(vlm))}")

    ids = sorted(llm, key=lambda x: (len(x), x))
    pairs = [(bool(llm[i]["is_correct"]), bool(vlm[i]["is_correct"])) for i in ids]
    n = len(pairs)
    llm_correct = sum(a for a, _ in pairs)
    vlm_correct = sum(b for _, b in pairs)
    llm_only = sum(a and not b for a, b in pairs)
    vlm_only = sum(b and not a for a, b in pairs)
    both_correct = sum(a and b for a, b in pairs)
    both_wrong = sum(not a and not b for a, b in pairs)

    # Bootstrap the paired accuracy difference (VLM minus LLM).
    rng = random.Random(args.seed)
    deltas = [int(b) - int(a) for a, b in pairs]
    bootstrap = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(args.bootstrap_samples))
    ci = [bootstrap[int((args.bootstrap_samples - 1) * q)] for q in (0.025, 0.975)]

    result = {
        "dataset": args.dataset,
        "condition": "caption-matched text-only; VLM receives no raw image",
        "n": n,
        "accuracy": {"llm": llm_correct / n, "vlm": vlm_correct / n, "delta_vlm_minus_llm": (vlm_correct - llm_correct) / n},
        "paired_outcomes": {"both_correct": both_correct, "llm_only_correct": llm_only, "vlm_only_correct": vlm_only, "both_wrong": both_wrong},
        "delta_bootstrap_95ci": ci,
        "mcnemar_exact_two_sided_p": binom_two_sided(min(llm_only, vlm_only), llm_only + vlm_only),
        "breakdown": {
            "question_type": breakdown(llm, vlm, "question_type"),
            "answer_type": breakdown(llm, vlm, "answer_type"),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "source_records": {model: str(path) for model, path in paths.items()},
    }
    output_dir = RECORD_ROOT / "postprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.dataset}_paired_summary.json"
    md_path = output_dir / f"{args.dataset}_paired_summary.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pp = lambda x: f"{100 * x:.2f}%"
    lines = [
        f"# {args.dataset.title()} caption-matched LLM vs VLM\n\n"
        f"- Condition: {result['condition']}\n"
        f"- Samples: {n}\n"
        f"- LLM accuracy: {pp(result['accuracy']['llm'])} ({llm_correct}/{n})\n"
        f"- VLM accuracy: {pp(result['accuracy']['vlm'])} ({vlm_correct}/{n})\n"
        f"- VLM − LLM: {pp(result['accuracy']['delta_vlm_minus_llm'])} "
        f"(paired bootstrap 95% CI: {pp(ci[0])}, {pp(ci[1])})\n"
        f"- Paired outcomes: both correct {both_correct}; LLM-only correct {llm_only}; "
        f"VLM-only correct {vlm_only}; both wrong {both_wrong}\n"
        f"- Exact two-sided McNemar p: {result['mcnemar_exact_two_sided_p']:.6g}\n",
        "\n## JSON-derived breakdown\n",
    ]
    for field, rows in result["breakdown"].items():
        lines.extend([f"\n### {field}\n", "\n| Group | n | LLM | VLM | VLM − LLM | LLM-only | VLM-only |\n", "|---|---:|---:|---:|---:|---:|---:|\n"])
        for label, row in rows.items():
            lines.append(
                f"| {label} | {row['n']} | {pp(row['llm_accuracy'])} | {pp(row['vlm_accuracy'])} | "
                f"{pp(row['delta_vlm_minus_llm'])} | {row['llm_only_correct']} | {row['vlm_only_correct']} |\n"
            )
    md_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
