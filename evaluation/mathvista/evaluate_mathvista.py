import argparse
import json
import os
import random

from evaluation.utils import DEFAULT_SYSTEM_PROMPT, load_data_split, load_system_prompt
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

def evaluate_chat_model(args, llm, processor, stop_token_ids):
    random.seed(args.seed)
    results_files = []
    dataset_entries = getattr(args, "dataset_entries", {}) or {}
    default_source = getattr(args, "dataset_source", "") or ""
    cache_dir = getattr(args, "dataset_cache_dir", None)
    system_prompt = getattr(args, "system_prompt_text", "") or load_system_prompt(
        getattr(args, "system_prompt", ""),
        default_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    for ds_name in args.datasets:
        ds_cfg = dataset_entries.get(ds_name, {}) if isinstance(dataset_entries, dict) else {}
        source = ds_cfg.get("source") or default_source
        split = ds_cfg.get("split")
        name = ds_cfg.get("name")
        if not source or not split:
            raise ValueError(f"Missing dataset config for {ds_name}: source/split are required.")
        data = load_data_split(
            source=source,
            split=split,
            cache_dir=cache_dir,
            name=name,
        )
        if args.limit > 0:
            data = data.select(range(min(args.limit, len(data))))

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            image = data_item["decoded_image"]
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": data_item["query"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            )

        sampling_params = SamplingParams(
            temperature=args.gen_temperature,
            max_tokens=args.gen_max_tokens,
            stop_token_ids=stop_token_ids,
        )
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item["decoded_image"]
            data_item["response"] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            pid = data_item["pid"]
            temp[pid] = data_item

        print(f"Evaluating {ds_name} ...")
        results_file = f"{ds_name}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))
        results_files.append(results_file)

    return results_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="MathVista_test")
    parser.add_argument("--out-dir", type=str, default="ouputs/evaluation/mathvista")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1, help="Max number of samples per dataset for quick debug.")
    parser.add_argument("--gen-temperature", type=float, default=0.0)
    parser.add_argument("--gen-max-tokens", type=int, default=4096)
    parser.add_argument("--gen-stop-token-ids", type=str, default="", help="Comma-separated stop token ids, empty for none.")
    parser.add_argument("--dataset-source", type=str, default="", help="Dataset source (HF repo id or local path).")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt text or prompt file path.")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    args.dataset_entries = {}
    print("datasets:", args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = [int(x.strip()) for x in args.gen_stop_token_ids.split(",") if x.strip()] or None

    evaluate_chat_model(args, llm, processor, stop_token_ids)
