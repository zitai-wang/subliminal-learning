#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data1/wzy/sub-learn")
OUT_DIR = ROOT / "picture_code"


# =========================
# Data
# =========================
QWEN_LAYERS = np.arange(28)
QWEN_BASELINE = 66.6

QWEN_LLM_DERIVED = np.array([
    67.3, 67.2, 66.9, 66.5, 66.4, 66.2, 65.1, 66.1,
    66.2, 66.1, 66.1, 65.5, 64.3, 66.1, 66.3, 66.4,
    66.7, 66.1, 66.3, 66.8, 67.0, 67.2, 67.0, 66.2,
    66.4, 66.6, 67.1, 67.3,
], dtype=np.float32)

QWEN_VLM_DERIVED = np.array([
    67.0, 66.8, 65.6, 66.2, 65.6, 65.8, 65.5, 65.3,
    64.8, 64.9, 64.9, 65.3, 64.5, 65.5, 64.8, 64.4,
    65.0, 64.7, 65.4, 66.0, 66.5, 66.7, 66.3, 66.0,
    66.1, 66.3, 67.3, 66.8,
], dtype=np.float32)


INTERN_LAYERS = np.arange(32)
INTERN_BASELINE = 66.5

INTERN_LLM_DERIVED = np.array([
    66.7, 66.6, 66.4, 66.8, 66.4, 66.1, 66.5, 66.3,
    66.3, 66.0, 65.8, 66.3, 66.3, 65.7, 66.5, 66.3,
    67.0, 66.6, 66.9, 66.9, 66.6, 66.5, 67.4, 66.8,
    66.8, 67.0, 67.2, 66.9, 66.9, 66.5, 66.9, 66.9,
], dtype=np.float32)

INTERN_VLM_DERIVED = np.array([
    66.4, 66.3, 66.5, 66.5, 66.3, 66.2, 66.1, 65.8,
    66.1, 65.3, 66.0, 66.1, 65.8, 66.2, 66.4, 66.5,
    66.0, 66.3, 66.6, 66.2, 66.2, 66.5, 66.7, 67.1,
    66.8, 66.2, 66.3, 66.5, 66.4, 66.3, 66.6, 66.5,
], dtype=np.float32)


def plot_panel(
    ax: plt.Axes,
    layers: np.ndarray,
    baseline: float,
    llm_derived: np.ndarray,
    vlm_derived: np.ndarray,
    title: str,
    xticks: np.ndarray,
    ylim: tuple[float, float],
    yticks: np.ndarray,
) -> None:
    color_baseline = "#7a7a7a"
    color_llm = "#e41a1c"
    color_vlm = "#1b8e2a"

    # Baseline: horizontal dashed line
    ax.axhline(
        y=baseline,
        color=color_baseline,
        linestyle="--",
        linewidth=3.0,
        label="Baseline",
        zorder=2,
    )

    # LLM-derived
    ax.plot(
        layers,
        llm_derived,
        color=color_llm,
        marker="o",
        markersize=8.0,
        markeredgewidth=0.0,
        linewidth=4.4,
        label="LLM-derived",
        zorder=4,
    )

    # VLM-derived
    ax.plot(
        layers,
        vlm_derived,
        color=color_vlm,
        marker="s",
        markersize=8.0,
        markeredgewidth=0.0,
        linewidth=4.4,
        label="VLM-derived",
        zorder=3,
    )

    ax.set_title(title, fontsize=22, pad=6)
    ax.set_xlabel("Layer", fontsize=18, labelpad=4)
    ax.set_ylabel("Accuracy (%)", fontsize=18, labelpad=10)

    ax.set_xlim(layers.min() - 0.25, layers.max() + 0.25)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.grid(
        True,
        axis="both",
        color="#bdbdbd",
        linewidth=0.85,
        alpha=0.65,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#8a8a8a")

    ax.tick_params(
        axis="both",
        labelsize=13,
        width=0.9,
        length=4,
    )

    # Legend: only three method labels, no accuracy values
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=11,
        handlelength=2.2,
        borderpad=0.55,
        labelspacing=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#808080")
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_alpha(0.95)


def plot() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.82,
        bottom=0.18,
        wspace=0.20,
    )

    plot_panel(
        axes[0],
        QWEN_LAYERS,
        QWEN_BASELINE,
        QWEN_LLM_DERIVED,
        QWEN_VLM_DERIVED,
        "Qwen2.5-VL-7B-Instruct",
        xticks=np.arange(0, 28, 5),
        ylim=(64.0, 67.55),
        yticks=np.arange(64.0, 67.6, 0.5),
    )

    plot_panel(
        axes[1],
        INTERN_LAYERS,
        INTERN_BASELINE,
        INTERN_LLM_DERIVED,
        INTERN_VLM_DERIVED,
        "InternVL2.5-8B",
        xticks=np.arange(0, 32, 5),
        ylim=(65.0, 67.55),
        yticks=np.arange(65.0, 67.6, 0.5),
    )

    png_path = OUT_DIR / "layerwise_two_vlms_with_baseline.png"
    pdf_path = OUT_DIR / "layerwise_two_vlms_with_baseline.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = plot()
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()