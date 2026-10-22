# -*- coding: utf-8 -*-
"""Standalone, add-only ICL baseline eval for Qwen3-VL-8B-Thinking.

Mirrors the qwen2.5-vl branch of eval_baseline.py's main() (build dataset ->
build QwenVLModelWrapper -> dataset.eval -> evaluate.save), reusing the
existing, unmodified QwenVLModelWrapper and dataset_utils classes. This file
does not import or alter eval_baseline.py itself so that experiments already
running against the existing code are unaffected.
"""

import json
import os
import random

import evaluate
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paths
from utils import get_expand_runname, get_runtime_device
from dataset_utils import dataset_mapping, DatasetBase
from qwen_vl_model_wrapper import QwenVLModelWrapper
from shift_encoder import AttnFFNShift

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

QWEN3_VL_MODELS = {
    "qwen3-vl-8b-thinking": (
        "Qwen/Qwen3-VL-8B-Thinking",
        "/data/share/Qwen3-VL-8B-Thinking",
    ),
}


def configure_official_mathvision_prompt(cfg: DictConfig, dataset: DatasetBase) -> None:
    """Match the Qwen3-VL report's MathVision prompt: image + raw question only."""
    if cfg.data.name != "mathvision" or not bool(
        getattr(cfg, "official_mathvision_prompt", False)
    ):
        return

    def create_query(problem, shot_type="solution", use_caption=False, use_ocr=False):
        del shot_type, use_caption, use_ocr
        return dataset._clean_question_text(problem.get("question", ""))

    def create_messages(question, state, item=None, include_image=True):
        del state
        image = dataset._load_image(item) if (item is not None and include_image) else None
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    dataset.create_query = create_query
    dataset._create_vlm_messages = create_messages
    print("Using official Qwen3-VL MathVision prompt: image + raw question only.")


def build_qwen3_vl_model_info(model_name: str):
    key = model_name.lower()
    if key not in QWEN3_VL_MODELS:
        raise ValueError(
            f"Unsupported Qwen3-VL model: {model_name}. "
            f"Add it to QWEN3_VL_MODELS in {__file__}."
        )
    hf_id, local_path = QWEN3_VL_MODELS[key]
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local weights for {model_name} not found at {local_path}")
    return hf_id, local_path


def install_base_vector_hooks(cfg: DictConfig, model: QwenVLModelWrapper, device):
    """Load an extracted base vector and install FFN-output shift hooks.

    In this repository, "base" means the averaged teacher hidden state stored
    as ``ffn_hs_vector``.  The source and target text backbones must have the
    same layer count and hidden size.
    """
    if not bool(getattr(cfg, "use_extracted_cot_vector", False)):
        return None, {}

    if not bool(getattr(cfg, "use_base_vector", False)):
        raise ValueError(
            "eval_qwen3vl_baseline currently supports Qwen3-VL vector injection "
            "only in base mode; set use_base_vector=True."
        )

    vector_path = getattr(cfg, "extracted_cot_vector_path", None)
    if not vector_path or not os.path.exists(vector_path):
        raise FileNotFoundError(f"Extracted CoT vector file not found: {vector_path}")

    payload = torch.load(vector_path, map_location="cpu", weights_only=False)
    base_vector = payload.get("ffn_hs_vector")
    if not isinstance(base_vector, torch.Tensor):
        raise ValueError(f"{vector_path} does not contain a tensor ffn_hs_vector.")
    if not torch.isfinite(base_vector).all():
        raise ValueError(f"{vector_path} contains NaN/Inf in ffn_hs_vector.")

    text_cfg = model.model.config.text_config
    expected_shape = (text_cfg.num_hidden_layers, text_cfg.hidden_size)
    if tuple(base_vector.shape) != expected_shape:
        raise ValueError(
            "Base vector shape does not match Qwen3-VL text backbone: "
            f"loaded={tuple(base_vector.shape)}, expected={expected_shape}."
        )

    only_shift_at_layer = getattr(cfg, "only_shift_at_layer", None)
    if only_shift_at_layer is not None:
        if isinstance(only_shift_at_layer, int):
            selected_layers = [only_shift_at_layer]
        else:
            selected_layers = [int(layer) for layer in only_shift_at_layer]
        invalid_layers = [
            layer for layer in selected_layers
            if layer < 0 or layer >= text_cfg.num_hidden_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"Invalid only_shift_at_layer entries {invalid_layers}; "
                f"valid range is [0, {text_cfg.num_hidden_layers - 1}]."
            )

    shift_scale = float(getattr(cfg, "static_shift_scale", 1.0))
    encoder = AttnFFNShift(
        lmm=model,
        attn_strategy="ShiftStrategy(0)",
        ffn_strategy="ShiftStrategy.VECTOR_SHIFT",
        static_mu_value=shift_scale,
        only_shift_at_layer=only_shift_at_layer,
    ).to(device)
    with torch.no_grad():
        encoder.ffn_shift.copy_(
            base_vector.to(device=device, dtype=encoder.ffn_shift.dtype)
        )
    encoder.eval()
    hooks = encoder.register_shift_hooks()
    ffn_hook_count = len(hooks.get("ffn_hook", []))
    if ffn_hook_count != text_cfg.num_hidden_layers:
        encoder.remove_hooks(hooks)
        raise RuntimeError(
            "Qwen3-VL base-vector hooks did not match every text layer: "
            f"matched={ffn_hook_count}, expected={text_cfg.num_hidden_layers}."
        )

    selected = "all" if only_shift_at_layer is None else str(only_shift_at_layer)
    print(
        "Installed Qwen3-VL base-vector hooks: "
        f"path={vector_path}, shape={tuple(base_vector.shape)}, "
        f"layers={selected}, scale={shift_scale}, hooks={ffn_hook_count}."
    )
    return encoder, hooks


@hydra.main(config_path="config", config_name="eval_qwen3vl_baseline.yaml", version_base=None)
def main(cfg: DictConfig):
    generation_seed = int(getattr(cfg, "generation_seed", 1234))
    random.seed(generation_seed)
    np.random.seed(generation_seed)
    torch.manual_seed(generation_seed)
    torch.cuda.manual_seed_all(generation_seed)
    print(f"Generation seed: {generation_seed}")

    runname = get_expand_runname(cfg)
    run_tag = getattr(cfg, "run_tag", None)
    dataset_dir_name = f"{cfg.data.name}_{run_tag}" if run_tag else cfg.data.name
    result_family = (
        "qwen3vl_thinking_base_vector"
        if bool(getattr(cfg, "use_extracted_cot_vector", False))
        else "qwen3vl_thinking_baseline"
    )
    record_dir = os.path.join(paths.result_dir, result_family, dataset_dir_name)
    os.makedirs(record_dir, exist_ok=True)
    record_path = os.path.join(record_dir, f"{cfg.data.num_shot}shot.json")

    if cfg.resume and os.path.exists(record_path):
        print(f"Found existing record {record_path}, skipping ({runname}).")
        return

    device = get_runtime_device()

    dataset: DatasetBase = dataset_mapping[cfg.data.name](cfg.data, model_name=cfg.model_name)
    configure_official_mathvision_prompt(cfg, dataset)

    shard_count = int(getattr(cfg, "shard_count", 1) or 1)
    shard_index = int(getattr(cfg, "shard_index", 0) or 0)
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(f"Invalid shard_index={shard_index} for shard_count={shard_count}.")
        shard_indices = list(range(shard_index, len(dataset._query_set), shard_count))
        dataset._query_set = dataset._query_set.select(shard_indices)
        print(f"Evaluating shard {shard_index + 1}/{shard_count} with {len(dataset._query_set)} query samples.")

    concise_suffix = getattr(cfg, "concise_system_prompt_suffix", None)
    if concise_suffix and hasattr(dataset, "direct_answer_system_prompt"):
        dataset.direct_answer_system_prompt = (
            dataset.direct_answer_system_prompt.rstrip() + " " + concise_suffix
        )
        print(f"Patched direct_answer_system_prompt -> {dataset.direct_answer_system_prompt!r}")

    hf_id, model_path = build_qwen3_vl_model_info(cfg.model_name)
    print(f"Instantiating QwenVLModelWrapper for {cfg.model_name} from {model_path}...")
    attn_implementation = getattr(cfg, "attn_implementation", None)
    model_args = {"attn_implementation": attn_implementation} if attn_implementation else None
    if attn_implementation:
        print(f"Using attn_implementation={attn_implementation!r}")
    model = QwenVLModelWrapper(
        model_root=model_path,
        processor_class=AutoProcessor,
        model_class=Qwen3VLForConditionalGeneration,
        support_models=[hf_id],
        model_args=model_args,
        local_files_only=True,
        torch_dtype=eval(cfg.dtype),
        processor_args={"padding_side": "left"},
        model_name=cfg.model_name,
    ).to(device)
    dataset.model_processor = model.processor
    model.eval()

    encoder = None
    hooks = {}
    try:
        encoder, hooks = install_base_vector_hooks(cfg, model, device)
        result, eval_result = dataset.eval(cfg, model)
        print(f"Evaluation result for {runname}: {eval_result}")

        config = {"eval_args": OmegaConf.to_container(cfg, resolve=True)}
        evaluate.save(
            record_path,
            **config,
            eval_result=eval_result,
            records=result,
        )
    finally:
        if encoder is not None:
            encoder.remove_hooks(hooks)
        if os.path.exists(record_path) and os.path.getsize(record_path) == 0:
            os.remove(record_path)


if __name__ == "__main__":
    main()
