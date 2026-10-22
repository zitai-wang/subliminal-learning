#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data1/wzy/sub-learn")
OUT_DIR = ROOT / "results" / "static_pipeline" / "commonsenseqa_all" / "qwen2.5-7b-instruct_20260527"

BASE_RECORD_DIR = ROOT / "results" / "static_pipeline" / "qwen2.5-7b-instruct_commonsenseqa_base_20260527_162921" / "records"
FFN_RECORD_DIR = ROOT / "results" / "static_pipeline" / "qwen2.5-7b-instruct_commonsenseqa_ffn_20260527_141517" / "records"
MIMIC_RECORD_DIR = ROOT / "results" / "static_pipeline" / "qwen2.5-7b-instruct_commonsenseqa_mimic_20260527_154105" / "records"
BASELINE_PATH = ROOT / "results" / "static_pipeline" / "qwen2.5-7b-instruct_commonsenseqa_mimic_20260527_154105" / "baseline_records" / "0shot.json"


def load_layerwise(record_dir: Path, prefix: str) -> list[float]:
    pattern = re.compile(rf"{prefix}_layers_(\d+)_direct_q_1\.0\.json$")
    scores: dict[int, float] = {}
    for path in record_dir.glob("*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        layer = int(match.group(1))
        payload = json.load(path.open())
        scores[layer] = float(payload["eval_result"]["accuracy"]) * 100.0
    return [scores[idx] for idx in range(28)]


def pct(value: float) -> str:
    return f"{value:.2f}"


def build_rows(base: list[float], ffn: list[float], mimic: list[float]) -> list[list[str]]:
    rows = []
    for idx in range(28):
        rows.append(
            [
                f"layer{idx}",
                pct(base[idx]),
                pct(ffn[idx]),
                pct(mimic[idx]),
            ]
        )
    return rows


def save_csv(rows: list[list[str]]) -> Path:
    csv_path = OUT_DIR / "layerwise_accuracy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "base", "ffn", "attn"])
        writer.writerows(rows)
    return csv_path


def plot_table(rows: list[list[str]], best_layers: dict[str, int]) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 11.2))
    ax.axis("off")

    col_labels = ["", "base", "ffn", "attn"]
    col_widths = [0.22, 0.20, 0.20, 0.20]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.02, 0.02, 0.96, 0.96],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    header_color = "#f1f3f5"
    edge_color = "#cfd4da"
    best_color = "#fff3bf"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.75)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_weight("bold")
            cell.set_height(0.035)
        else:
            cell.set_height(0.0315)

    table[(best_layers["base"] + 1, 1)].set_facecolor(best_color)
    table[(best_layers["ffn"] + 1, 2)].set_facecolor(best_color)
    table[(best_layers["mimic"] + 1, 3)].set_facecolor(best_color)

    png_path = OUT_DIR / "layerwise_accuracy_table.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_lines(
    base: list[float],
    ffn: list[float],
    mimic: list[float],
    baseline: float,
    best_points: dict[str, tuple[int, float]],
) -> tuple[Path, Path]:
    layers = np.arange(28)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.axhline(
        y=baseline,
        color="#7a7a7a",
        linestyle="--",
        linewidth=2.5,
        label=f"Baseline ({baseline:.2f})",
        zorder=1,
    )
    ax.plot(layers, base, color="#1f77b4", marker="o", markersize=4.8, linewidth=2.5, label="Base", zorder=4)
    ax.plot(layers, ffn, color="#2ca02c", marker="s", markersize=4.6, linewidth=2.5, label="FFN", zorder=3)
    ax.plot(layers, mimic, color="#e41a1c", marker="^", markersize=5.0, linewidth=2.5, label="Attn", zorder=5)

    for name, color in [("base", "#1f77b4"), ("ffn", "#2ca02c"), ("mimic", "#e41a1c")]:
        layer, score = best_points[name]
        ax.scatter([layer], [score], color=color, s=55, zorder=6, edgecolors="white", linewidths=0.8)

    summary = "\n".join(
        [
            f"Base best: L{best_points['base'][0]} ({best_points['base'][1]:.2f})",
            f"FFN best: L{best_points['ffn'][0]} ({best_points['ffn'][1]:.2f})",
            f"Attn best: L{best_points['mimic'][0]} ({best_points['mimic'][1]:.2f})",
        ]
    )
    ax.text(
        0.985,
        0.06,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#b0b0b0", alpha=0.92),
    )

    ax.set_title("Qwen2.5-7B-Instruct", fontsize=16, pad=8)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_xlim(-0.5, 27.5)
    ymin = min(min(base), min(ffn), min(mimic), baseline)
    ymax = max(max(base), max(ffn), max(mimic), baseline)
    ax.set_ylim(np.floor((ymin - 0.6) * 2) / 2, np.ceil((ymax + 0.4) * 2) / 2)
    ax.set_xticks(np.arange(0, 28, 2))
    ax.grid(True, axis="both", color="#c8c8c8", linewidth=0.8, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#8a8a8a")

    legend = ax.legend(loc="lower right", frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#909090")
    legend.get_frame().set_alpha(0.95)

    png_path = OUT_DIR / "layerwise_accuracy_plot.png"
    pdf_path = OUT_DIR / "layerwise_accuracy_plot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def save_best_summary(best_points: dict[str, tuple[int, float]], baseline: float) -> Path:
    md_path = OUT_DIR / "best_summary.md"
    lines = [
        "# CommonsenseQA Qwen2.5-7B-Instruct Layerwise Summary",
        "",
        f"- Baseline: {baseline:.2f}",
        f"- Base best: layer {best_points['base'][0]}, {best_points['base'][1]:.2f}",
        f"- FFN best: layer {best_points['ffn'][0]}, {best_points['ffn'][1]:.2f}",
        f"- Attn best: layer {best_points['mimic'][0]}, {best_points['mimic'][1]:.2f}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_layerwise(BASE_RECORD_DIR, "use_base_vector")
    ffn = load_layerwise(FFN_RECORD_DIR, "licv")
    mimic = load_layerwise(MIMIC_RECORD_DIR, "mimic")
    baseline = float(json.load(BASELINE_PATH.open())["eval_result"]["accuracy"]) * 100.0

    rows = build_rows(base, ffn, mimic)
    best_points = {
        "base": max(enumerate(base), key=lambda x: x[1]),
        "ffn": max(enumerate(ffn), key=lambda x: x[1]),
        "mimic": max(enumerate(mimic), key=lambda x: x[1]),
    }
    best_layers = {k: v[0] for k, v in best_points.items()}

    csv_path = save_csv(rows)
    table_path = plot_table(rows, best_layers)
    png_path, pdf_path = plot_lines(base, ffn, mimic, baseline, best_points)
    md_path = save_best_summary(best_points, baseline)

    print(csv_path)
    print(table_path)
    print(png_path)
    print(pdf_path)
    print(md_path)


if __name__ == "__main__":
    main()
