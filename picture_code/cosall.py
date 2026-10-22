#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
OUT_DIR = Path("/data1/wzy/sub-learn/picture_code/57")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_KEY = "ffn_hs_vector"


PAIRS = [
    {
        "group": "Qwen2.5-VL-7B-Instruct",
        "task": "GSM8K",
        "llm": "/data1/wzy/sub-learn/results/average_cot_vectors/qwen2.5-7b-instruct_gsm8k_selfcot458_aligned_avg_cot_vector_mimic_20260417_161916.pt",
        "vlm": "/data1/wzy/sub-learn/results/average_cot_vectors/qwen2.5-vl-7b-instruct_gsm8k_correct_only_avg_cot_vector_mimic_20260402_162907.pt",
        "tag": "qwenvl_gsm8k",
    },
    {
        "group": "InternVL2.5-8B",
        "task": "GSM8K",
        "llm": "/data1/wzy/sub-learn/results/average_cot_vectors/internlm2_5-7b-chat_gsm8k_correct_only_500_avg_cot_vector_mimic_20260409_cuda2.pt",
        "vlm": "/data1/wzy/sub-learn/results/average_cot_vectors/internvl2_5-8b_gsm8k_correct_only_500_avg_cot_vector_mimic_20260408_cuda2.pt",
        "tag": "internvl_gsm8k",
    },
]


# =========================================================
# Utils
# =========================================================
def load_vector(path: str | Path, key: str = VECTOR_KEY) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if key not in obj:
        raise KeyError(f"{key} not found in {path}. Available keys: {list(obj.keys())}")

    vec = obj[key].float()

    # ffn_hs_vector / ffn_cot_vector: [layers, hidden_dim]
    # attn_cot_vector: [layers, heads, head_dim], flatten if needed.
    if vec.ndim == 3:
        vec = vec.reshape(vec.shape[0], -1)
    elif vec.ndim != 2:
        raise ValueError(f"Unexpected vector shape for {key}: {tuple(vec.shape)}")

    return vec


def cross_layer_cosine(llm_vec: torch.Tensor, vlm_vec: torch.Tensor) -> torch.Tensor:
    if llm_vec.shape[1] != vlm_vec.shape[1]:
        raise ValueError(
            f"Hidden dim mismatch: LLM {tuple(llm_vec.shape)}, VLM {tuple(vlm_vec.shape)}"
        )

    llm_vec = llm_vec / llm_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    vlm_vec = vlm_vec / vlm_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)

    return llm_vec @ vlm_vec.T


def split_regions(n_layers: int) -> tuple[list[int], list[int], list[int]]:
    shallow = list(range(0, n_layers // 3))
    middle = list(range(n_layers // 3, 2 * n_layers // 3))
    deep = list(range(2 * n_layers // 3, n_layers))
    return shallow, middle, deep


def region_mean(diag: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return float("nan")
    return float(np.mean(diag[indices]))


# =========================================================
# Main computation
# =========================================================
def compute_all():
    sim_mats = []
    records = []

    for item in PAIRS:
        llm_vec = load_vector(item["llm"], VECTOR_KEY)
        vlm_vec = load_vector(item["vlm"], VECTOR_KEY)

        sim = cross_layer_cosine(llm_vec, vlm_vec)
        sim_np = sim.cpu().numpy()

        n_llm, n_vlm = sim_np.shape
        n_diag = min(n_llm, n_vlm)
        diag = np.diag(sim_np[:n_diag, :n_diag])

        shallow, middle, deep = split_regions(n_diag)

        record = {
            "group": item["group"],
            "task": item["task"],
            "label": f'{item["group"]}-{item["task"]}',
            "n_llm_layers": n_llm,
            "n_vlm_layers": n_vlm,
            "same_layer_mean": float(np.mean(diag)),
            "same_layer_min": float(np.min(diag)),
            "same_layer_max": float(np.max(diag)),
            "shallow_mean": region_mean(diag, shallow),
            "middle_mean": region_mean(diag, middle),
            "deep_mean": region_mean(diag, deep),
        }

        records.append(record)
        sim_mats.append(
            {
                "group": item["group"],
                "task": item["task"],
                "tag": item["tag"],
                "sim": sim_np,
                "diag_mean": float(np.mean(diag)),
            }
        )

        print(
            f'{item["group"]:8s} / {item["task"]:13s} | '
            f'diag mean={record["same_layer_mean"]:.3f}, '
            f'shallow={record["shallow_mean"]:.3f}, '
            f'middle={record["middle_mean"]:.3f}, '
            f'deep={record["deep_mean"]:.3f}'
        )

    summary_df = pd.DataFrame(records)
    summary_path = OUT_DIR / "ffn_hs_alignment_summary_2heatmaps.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV: {summary_path}")

    return sim_mats, summary_df


# =========================================================
# Figure: 2 heatmaps in one panel
# =========================================================
def plot_heatmap_panel(sim_mats) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.4, 3.35))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1, 1, 0.045],
        wspace=0.24,
        left=0.075,
        right=0.92,
        bottom=0.18,
        top=0.82,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]
    cax = fig.add_subplot(gs[0, 2])

    vmin, vmax = 0.0, 1.0
    im = None
    panel_labels = ["(a)", "(b)"]

    for ax, item, panel_label in zip(axes, sim_mats, panel_labels):
        sim = item["sim"]
        n_llm, n_vlm = sim.shape
        n_diag = min(n_llm, n_vlm)

        im = ax.imshow(
            sim,
            origin="lower",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        ax.set_box_aspect(1)

        # diagonal guide
        ax.plot(
            range(n_diag),
            range(n_diag),
            linestyle="--",
            linewidth=1.2,
            color="white",
            alpha=0.9,
        )

        ax.set_title(
            f'{panel_label} {item["group"]} / {item["task"]}\n'
            f'Diagonal mean = {item["diag_mean"]:.3f}',
            fontsize=12,
            pad=7,
        )

        ax.set_xlabel("VLM Layer", fontsize=11)
        ax.set_ylabel("LLM Layer", fontsize=11)

        xticks = list(range(0, n_vlm, 4))
        if xticks[-1] != n_vlm - 1:
            xticks.append(n_vlm - 1)

        yticks = list(range(0, n_llm, 4))
        if yticks[-1] != n_llm - 1:
            yticks.append(n_llm - 1)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#555555")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Cosine Similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    out_png = OUT_DIR / "cos_2heatmaps.png"
    out_pdf = OUT_DIR / "cos_2heatmaps.pdf"

    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap panel PNG: {out_png}")
    print(f"Saved heatmap panel PDF: {out_pdf}")


def main() -> None:
    sim_mats, _ = compute_all()
    plot_heatmap_panel(sim_mats)


if __name__ == "__main__":
    main()