import argparse
import json
import os
import random

from evaluation.utils import DEFAULT_SYSTEM_PROMPT, load_data_split, load_system_prompt
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

english_answer_type_dict = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}
def make_input(prompt, question_content):
    input = prompt + "\n" + question_content
    return input


def build_prompt(data_item):
    is_math = "maths" in data_item["source"]
    subject_content = "Math" if is_math else "Physics"
    if data_item["is_multiple_answer"]:
        multiple_answer_text = "\\boxed{multiple answers connected with commas}"
    else:
        multiple_answer_text = "\\boxed{answer}"
    unit_text = ""
    if data_item["unit"]:
        multiple_answer_text += "(unit)"
        unit_text = ", note that the unit of the answer should not be included in \\boxed{}"
    answer_type_text = get_answer_type_text(data_item["answer_type"], multiple_answer=data_item["is_multiple_answer"])
    prompt = (
        f"The following is an open-ended problem from an International {subject_content} competition. "
        f"{answer_type_text}Please calculate the answer according to the given requirements and "
        "the information provided. Please use LaTeX format to represent the variables and formulas "
        'used in the solution process and results. Please end your solution with "So the final answer '
        f'is {multiple_answer_text}." and give the result explicitly{unit_text}.'
    )
    if is_math:
        input = make_input(prompt, data_item["question"])
    else:
        if "context" in data_item.keys() and str(data_item["context"]) != "nan":
            input = make_input(prompt, data_item["context"] + "\n" + data_item["question"])
        else:
            input = make_input(prompt, data_item["question"])
    return input


def get_answer_type_text(answer_type, multiple_answer):
    if ("Need_human_evaluate" in answer_type) or ("Tuple" in answer_type):
        full_answer_text = ""
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type)
            full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if "," not in answer_type:
                answer_text = get_single_answer_type_text(answer_type)
                full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
            else:
                answer_types = answer_type.split(",")
                answer_types = [get_single_answer_type_text(t) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
                else:
                    answer_text = ", ".join(answer_types)
                    full_answer_text = (
                        f"The problem has multiple answers, with the answers in order being {answer_text}. "
                    )
    return full_answer_text


def get_single_answer_type_text(answer_type):
    if "-" in answer_type:
        answer_type = answer_type[: answer_type.find("-")]
    for t in ["Numerical", "Expression", "Equation", "Interval"]:
        if t in answer_type:
            return english_answer_type_dict[t]
    raise ValueError(f"Error parsing answer type {answer_type}!")


def evaluate_chat_model(args, llm, processor, stop_token_ids):
    random.seed(args.seed)
    dataset_entries = getattr(args, "dataset_entries", {}) or {}
    default_source = getattr(args, "dataset_source", "") or ""
    cache_dir = getattr(args, "dataset_cache_dir", None)
    system_prompt = getattr(args, "system_prompt_text", "") or load_system_prompt(
        getattr(args, "system_prompt", ""),
        default_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    data = []
    for ds_name in args.datasets:
        ds_cfg = dataset_entries.get(ds_name, {}) if isinstance(dataset_entries, dict) else {}
        source = ds_cfg.get("source") or default_source
        split = ds_cfg.get("split")
        name = ds_cfg.get("name")
        if not source or not split:
            raise ValueError(f"Missing dataset config for {ds_name}: source/split are required.")
        split = load_data_split(
            source=source,
            split=split,
            cache_dir=cache_dir,
            name=name,
        )
        if args.limit > 0:
            split = split.select(range(min(args.limit, len(split))))
        source_label = name or ds_name
        for data_item in split:
            data_item["source"] = source_label
            data.append(data_item)

    inputs = []
    for idx, data_item in tqdm(enumerate(data)):
        images_content = []
        if "image_1" in data_item:
            for i in range(1, 6):
                if data_item.get(f"image_{i}"):
                    images_content.append({"type": "image", "image": data_item[f"image_{i}"]})

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
                    *images_content,
                    {"type": "text", "text": build_prompt(data_item)},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        if image_data:
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            )
        else:
            inputs.append(
                {
                    "prompt": prompt,
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
        for i in range(1, 10):
            if data_item.get(f"image_{i}"):
                del data_item[f"image_{i}"]
        data_item["response"] = model_output.outputs[0].text
        outputs.append(data_item)

    temp = {}
    for data_item in outputs:
        id = data_item["id"]
        temp[id] = data_item

    print("Evaluating OlympiadBench ...")
    results_file = "olympiadbench.json"
    output_path = os.path.join(args.out_dir, results_file)
    json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print("Results saved to {}".format(output_path))

    return [results_file]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="OE_MM_maths_en_COMP,OE_MM_physics_en_COMP,OE_TO_maths_en_COMP,OE_TO_physics_en_COMP",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="results")
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
        limit_mm_per_prompt={"image": 8},
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = [int(x.strip()) for x in args.gen_stop_token_ids.split(",") if x.strip()] or None

    evaluate_chat_model(args, llm, processor, stop_token_ids)
