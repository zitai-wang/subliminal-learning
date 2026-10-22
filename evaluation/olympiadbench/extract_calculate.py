import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import regex

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_script_dir))
if _root not in sys.path:
    sys.path.insert(0, _root)
from evaluation.utils import get_chat_response, init_judge_client_or_raise


def build_olympiadbench_gpt4_prompt(question_data):
    prompt = """You are given a question, a solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "Yes" if the solution is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{solution}
--------------------------------
"""
    question = question_data["question"]
    answer = "\n".join(question_data["final_answer"])
    response = str(question_data["response"]).strip()
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        completion_match = regex.findall(
            r"\\boxed\{((?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*)\}", response, re.DOTALL
        )
        response = completion_match[-1][0].strip() if completion_match else response
    prompt = prompt.format(question=question, answer=answer, solution=response)
    return prompt


def score_answer(problem, judge_client, judge_model, judge_stats=None):
    prompt = build_olympiadbench_gpt4_prompt(problem)
    completion = get_chat_response(judge_client, judge_model, prompt, judge_stats=judge_stats)
    if completion.lower() == "yes":
        return True, problem["id"]
    elif completion.lower() == "no":
        return False, problem["id"]
    return False, problem["id"]


def OlympiadBench_acc(results):
    correct = 0
    total = len(results)
    for result in results:
        if result["score"]:
            correct += 1
    return {"correct": correct, "total": total, "accuracy": correct / total}


def run_extract(
    output_dir,
    output_file,
    response_label="response",
    number=-1,
    output_label="extract",
    init_judge=True,
    judge_client=None,
    judge_model=None,
    judge_stats=None,
):
    if judge_client is None or judge_model is None:
        if not init_judge:
            raise RuntimeError("judge_client/judge_model must be provided when init_judge=False")
        judge_client, judge_model, _ = init_judge_client_or_raise()

    _ = response_label
    result_file = os.path.join(output_dir, output_file)
    if output_label != "":
        extract_file = result_file.replace(".json", f"_{output_label}.json")
    else:
        extract_file = result_file
    score_file = result_file.replace(".json", "_score.json")

    print(f"Reading {result_file}...")
    results = json.load(open(result_file))

    test_ids = list(results.keys())
    if number > 0:
        test_ids = test_ids[: min(number, len(test_ids))]
    print("Number of testing problems:", len(test_ids))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(score_answer, results[sample_id], judge_client, judge_model, judge_stats)
            for sample_id in test_ids
        ]

    for future in as_completed(futures):
        score, sample_id = future.result()
        results[str(sample_id)]["score"] = score

    print(f"Saving results to {extract_file}...")
    json.dump(results, open(extract_file, "w"), indent=4, ensure_ascii=False)
    print("Results saved.")

    scores = OlympiadBench_acc([v for _, v in results.items()])
    print(scores)
    print(f"Saving scores to {score_file}...")
    with open(score_file, "w") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)
    print("Scores saved.")
    return {"extract_file": extract_file, "score_file": score_file, "num_problems": len(test_ids)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--response_label", type=str, default="response", help="response label for the input file")
    parser.add_argument("--number", type=int, default=-1, help="number of problems to run")
    parser.add_argument("--output_label", type=str, default="extract", help="label for the output file")
    args = parser.parse_args()
    run_extract(
        output_dir=args.output_dir,
        output_file=args.output_file,
        response_label=args.response_label,
        number=args.number,
        output_label=args.output_label,
        init_judge=True,
    )
