#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
LLM_VECTOR_PATH = Path(
    "/data1/wzy/sub-learn/results/average_cot_vectors/"
    "qwen2.5-7b-instruct_gsm8k_selfcot458_aligned_avg_cot_vector_mimic_20260417_161916.pt"
)

VLM_VECTOR_PATH = Path(
    "/data1/wzy/sub-learn/results/average_cot_vectors/"
    "qwen2.5-vl-7b-instruct_gsm8k_correct_only_avg_cot_vector_mimic_20260402_162907.pt"
)

OUT_DIR = Path("/data1/wzy/sub-learn/picture_code/vector_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_KEY = "ffn_hs_vector"

OUT_NAME = "ffn_hs_cross_layer_cosine_heatmap"


# =========================
# Utils
# =========================
def load_vector(path: Path, key: str) -> torch.Tensor:
    """
    Load vector from .pt file.

    Expected shape:
        ffn_hs_vector: [num_layers, hidden_dim]
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if key not in obj:
        raise KeyError(f"{key} not found in {path}. Available keys: {list(obj.keys())}")

    vec = obj[key].float()

    if vec.ndim != 2:
        raise ValueError(f"{key} should be 2D [layers, dim], but got shape {tuple(vec.shape)}")

    return vec


def compute_cross_layer_cosine(llm_vec: torch.Tensor, vlm_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-layer cosine similarity.

    Output:
        sim[i, j] = cosine(llm_vec[i], vlm_vec[j])
    """
    if llm_vec.shape[1] != vlm_vec.shape[1]:
        raise ValueError(
            f"Hidden dim mismatch: LLM {llm_vec.shape}, VLM {vlm_vec.shape}"
        )

    llm_norm = llm_vec / llm_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    vlm_norm = vlm_vec / vlm_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)

    sim = llm_norm @ vlm_norm.T
    return sim


def plot_heatmap(
    sim: torch.Tensor,
    out_png: Path,
    out_pdf: Path,
    title: str = "Cross-layer Cosine Similarity",
) -> None:
    sim_np = sim.cpu().numpy()
    n_llm, n_vlm = sim_np.shape

    diag_len = min(n_llm, n_vlm)
    diag = sim.diag()[:diag_len]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.4))

    im = ax.imshow(
        sim_np,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("VLM Layer", fontsize=13)
    ax.set_ylabel("LLM Layer", fontsize=13)

    ax.set_title(
        title + "\n"
        + f"Diagonal mean={diag.mean().item():.3f}, "
        + f"min={diag.min().item():.3f}, "
        + f"max={diag.max().item():.3f}",
        fontsize=14,
        pad=10,
    )

    # diagonal guide
    ax.plot(
        range(diag_len),
        range(diag_len),
        linestyle="--",
        linewidth=1.2,
        color="white",
        alpha=0.85,
    )

    # ticks
    xticks = list(range(0, n_vlm, 2))
    yticks = list(range(0, n_llm, 2))
    if xticks[-1] != n_vlm - 1:
        xticks.append(n_vlm - 1)
    if yticks[-1] != n_llm - 1:
        yticks.append(n_llm - 1)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.tick_params(axis="both", labelsize=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#666666")

    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    llm_vec = load_vector(LLM_VECTOR_PATH, VECTOR_KEY)
    vlm_vec = load_vector(VLM_VECTOR_PATH, VECTOR_KEY)

    print(f"Loaded LLM vector: {LLM_VECTOR_PATH}")
    print(f"  shape = {tuple(llm_vec.shape)}")
    print(f"Loaded VLM vector: {VLM_VECTOR_PATH}")
    print(f"  shape = {tuple(vlm_vec.shape)}")

    sim = compute_cross_layer_cosine(llm_vec, vlm_vec)

    diag = sim.diag()
    print("\nSame-layer cosine statistics:")
    print(f"  mean = {diag.mean().item():.4f}")
    print(f"  min  = {diag.min().item():.4f}")
    print(f"  max  = {diag.max().item():.4f}")

    best_match = sim.argmax(dim=1)
    same_layer_ratio = (best_match == torch.arange(sim.shape[0])).float().mean().item()
    print(f"  same-layer best-match ratio = {same_layer_ratio:.4f}")

    out_png = OUT_DIR / f"{OUT_NAME}.png"
    out_pdf = OUT_DIR / f"{OUT_NAME}.pdf"

    plot_heatmap(
        sim,
        out_png=out_png,
        out_pdf=out_pdf,
        title="FFN HS Vector: LLM-derived vs VLM-derived",
    )

    print(f"\nSaved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()