"""Cross-dataset MCQ evaluation for a fixed subliminal-learning Student."""

import argparse
import ast
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subliminal_route1 import (
    EVALUATION_CONDITIONS,
    append_row,
    dense_lora_dispatch,
    extract_answer,
    load_model,
    read_rows,
    seed_all,
    write_json,
)


def load_selected_rows(cfg):
    import pyarrow.parquet as pq

    if cfg['dataset_format'] == 'mmmu':
        rows = []
        for path in sorted(glob.glob(cfg['dataset_glob'])):
            subject = Path(path).parent.name
            for item in pq.read_table(path).to_pylist():
                item['_source_subject'] = subject
                rows.append(item)
    else:
        rows = pq.read_table(cfg['dataset_path']).to_pylist()
    assert len(rows) == cfg['evaluation']['source_samples'], (len(rows), cfg['evaluation']['source_samples'])
    sample_size = cfg['evaluation']['expected_samples']
    assert 0 < sample_size <= len(rows)
    indices = (list(range(len(rows))) if sample_size == len(rows) else
               sorted(random.Random(cfg['seed']).sample(range(len(rows)), sample_size)))
    selected = []
    for source_index in indices:
        row = dict(rows[source_index])
        row['_source_index'] = source_index
        row['_id'] = f"{cfg['evaluation']['split']}:{source_index}"
        dataset_format = cfg['dataset_format']
        if dataset_format == 'scienceqa':
            row['_options'] = list(row['choices'])
            row['_answer'] = chr(ord('A') + int(row['answer']))
            row['_gold_answers'] = [row['_answer']]
            row['_question_type'] = 'multiple-choice'
            row['_subject'] = str(row.get('subject') or 'unknown')
        elif dataset_format == 'mmbench':
            row['_options'] = [str(row[key]) for key in ('A', 'B', 'C', 'D')
                               if str(row.get(key) or '').strip().casefold() not in {'', 'nan', 'none'}]
            row['_answer'] = str(row['answer']).strip().upper()
            row['_gold_answers'] = [row['_answer']]
            row['_question_type'] = 'multiple-choice'
            row['_subject'] = str(row.get('category') or 'unknown')
        elif dataset_format == 'mmstar':
            row['_options'] = parse_embedded_choices(row['question']) or ['A', 'B', 'C', 'D']
            row['_answer'] = str(row['answer']).strip().upper()
            row['_gold_answers'] = [row['_answer']]
            row['_question_type'] = 'multiple-choice'
            row['_subject'] = str(row.get('category') or 'unknown')
        elif dataset_format == 'mmmu':
            options = parse_list(row.get('options'))
            question_type = ('multiple-choice' if 'multiple' in str(row.get('question_type')).lower()
                             else 'open')
            row['_options'] = options
            row['_answer'] = str(row['answer']).strip()
            row['_gold_answers'] = parse_list(row['answer']) or [row['_answer']]
            row['_question_type'] = question_type
            row['_subject'] = row['_source_subject']
        elif dataset_format == 'chartqa':
            row['_options'] = []
            row['_gold_answers'] = parse_list(row['label']) or [str(row['label']).strip()]
            row['_answer'] = row['_gold_answers'][0]
            row['_question_type'] = 'open'
            row['_subject'] = 'human' if int(row.get('human_or_machine') or 0) == 0 else 'machine'
        elif dataset_format == 'mathverse':
            raw_answer = str(row['answer']).strip()
            options = parse_embedded_choices(row.get('question_for_eval') or row['question'])
            letter_match = re.fullmatch(r'\(?\s*([A-J])\s*\)?', raw_answer.upper())
            mapped = letter_match.group(1) if letter_match else None
            if mapped is None and options:
                answer_key = canonical_open(raw_answer)
                matches = [index for index, option in enumerate(options)
                           if canonical_open(option) == answer_key]
                mapped = chr(ord('A') + matches[0]) if len(matches) == 1 else None
            if mapped and len(options) <= ord(mapped) - ord('A'):
                options = [chr(ord('A') + index)
                           for index in range(max(4, ord(mapped) - ord('A') + 1))]
            question_type = ('multiple-choice' if 'choice' in row['question_type'] and mapped
                             else 'open')
            row['_options'] = options if question_type == 'multiple-choice' else []
            row['_answer'] = mapped if question_type == 'multiple-choice' else raw_answer
            row['_gold_answers'] = [row['_answer']]
            row['_question_type'] = question_type
            metadata = row.get('metadata') or {}
            row['_subject'] = str(metadata.get('subject') or 'unknown')
        else:
            raise ValueError(f"Unsupported dataset_format: {dataset_format}")
        if row['_question_type'] == 'multiple-choice':
            assert row['_answer'] in [chr(ord('A') + i) for i in range(len(row['_options']))]
        selected.append(row)
    assert len({row['_id'] for row in selected}) == sample_size
    return selected


def parse_list(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or '').strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def parse_embedded_choices(question):
    text = str(question or '')
    text = re.sub(r'(?i)(?:choices?|options?)\s*:\s*', '\n', text)
    matches = re.findall(
        r'(?:^|\n|,\s)([A-J])\s*[:.)]\s*(.*?)(?=(?:\n|,\s)[A-J]\s*[:.)]|\Z)',
        text, flags=re.DOTALL)
    choices = {}
    for letter, value in matches:
        choices.setdefault(letter, value.strip().rstrip(','))
    if not choices or sorted(choices) != [chr(ord('A') + i) for i in range(len(choices))]:
        return []
    return [choices[chr(ord('A') + i)] for i in range(len(choices))]


def normalize_mcq_prediction(extraction, options):
    text = str(extraction or '').strip()
    if not text:
        return None
    wrappers = ('text', 'mathrm', 'mathbf', 'textbf')
    for wrapper in wrappers:
        match = re.fullmatch(rf'\\{wrapper}\{{\s*(.*?)\s*\}}', text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
            break
    match = re.fullmatch(r'\(?\s*([A-J])\s*\)?[.:]?', text.upper())
    if match and ord(match.group(1)) - ord('A') < len(options):
        return match.group(1)
    canonical = re.sub(r'\s+', ' ', text).strip().rstrip('.').casefold()
    matches = [index for index, option in enumerate(options)
               if canonical == re.sub(r'\s+', ' ', str(option)).strip().rstrip('.').casefold()]
    return chr(ord('A') + matches[0]) if len(matches) == 1 else None


def canonical_open(value):
    text = str(value or '').strip()
    text = re.sub(r'^\\(?:text|mathrm|mathbf|textbf)\{(.*)\}$', r'\1', text, flags=re.DOTALL)
    text = text.strip('$').strip().strip('"\'').rstrip('.').strip().casefold()
    return re.sub(r'\s+', ' ', text)


def open_equal(prediction, answer, dataset_format):
    pred, gold = canonical_open(prediction), canonical_open(answer)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    try:
        pred_num = float(pred.replace(',', '').rstrip('%'))
        gold_num = float(gold.replace(',', '').rstrip('%'))
        tolerance = 0.05 * abs(gold_num) if dataset_format == 'chartqa' else 1e-6
        return abs(pred_num - gold_num) <= max(tolerance, 1e-6)
    except ValueError:
        pass
    if dataset_format in {'mathverse', 'mmmu'}:
        from src.subliminal_mathvision import mathvision_equal
        return mathvision_equal(pred, gold)
    return False


def score_mcq_response(response, row, dataset_format=None):
    extraction = extract_answer(str(response or ''))
    answer = row['_answer']
    if row['_question_type'] == 'multiple-choice':
        prediction = normalize_mcq_prediction(extraction, row['_options'])
        return extraction, prediction, answer, prediction == answer if prediction is not None else False
    prediction = canonical_open(extraction) or None
    correct = any(open_equal(extraction, gold, dataset_format or '') for gold in row['_gold_answers'])
    return extraction, prediction, answer, correct


def row_images(row, cfg):
    from PIL import Image

    values = []
    if row.get('image'):
        values.append(row['image'])
    values.extend(row.get(f'image_{index}') for index in range(1, 8) if row.get(f'image_{index}'))
    images = []
    for image_data in values:
        if hasattr(image_data, 'convert'):
            images.append(image_data.convert('RGB'))
        elif isinstance(image_data, (bytes, bytearray)):
            images.append(Image.open(io.BytesIO(image_data)).convert('RGB'))
        elif isinstance(image_data, dict) and image_data.get('bytes') is not None:
            images.append(Image.open(io.BytesIO(image_data['bytes'])).convert('RGB'))
        elif isinstance(image_data, dict) and image_data.get('path'):
            images.append(Image.open(Path(cfg.get('dataset_path', '.')).parents[1]
                                     / image_data['path']).convert('RGB'))
    return images


def build_query(row, instruction, dataset_format, cot=False):
    parts = []
    if dataset_format in {'scienceqa', 'mmbench'} and str(row.get('hint') or '').strip():
        parts.append('Context: ' + str(row['hint']).strip())
    if dataset_format == 'chartqa':
        parts.append(str(row['query']).strip())
    elif dataset_format == 'mathverse':
        parts.append(str(row['query_cot' if cot else 'query_wo']).strip())
    else:
        question = re.sub(r'<image\s*\d+>', '', str(row['question']), flags=re.I).strip()
        parts.append(question)
    if row['_options'] and dataset_format not in {'mmstar', 'mathverse'}:
        parts.append('Choices:\n' + '\n'.join(
            f"({chr(ord('A') + index)}) {option}" for index, option in enumerate(row['_options'])))
    parts.append(instruction)
    return '\n'.join(parts)


def evaluate(cfg, run, conditions=None, batch_size=4, incorrect_ids_from=None):
    from peft import PeftModel

    rows = load_selected_rows(cfg)
    all_expected_ids = {row['_id'] for row in rows}
    processor = AutoProcessor.from_pretrained(
        cfg['model_path'], local_files_only=True,
        min_pixels=cfg['evaluation']['min_pixels'], max_pixels=cfg['evaluation']['max_pixels'])
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    model = load_model(cfg).eval()
    names = list(conditions or EVALUATION_CONDITIONS)
    assert names == sorted(set(names), key=list(EVALUATION_CONDITIONS).index)
    write_json(run / 'evaluation/runtime.json', {
        'dataset': cfg['dataset_name'], 'split': cfg['evaluation']['split'],
        'source_samples': cfg['evaluation']['source_samples'], 'samples': len(rows),
        'sampling': ('all rows in source split' if len(rows) == cfg['evaluation']['source_samples']
                     else f"random without replacement; seed={cfg['seed']}; selected indices sorted"),
        'batch_size': batch_size, 'padding_side': 'left', 'decoding': 'greedy',
        'direct_max_new_tokens': cfg['evaluation']['direct_max_new_tokens'],
        'cot_max_new_tokens': cfg['evaluation']['cot_max_new_tokens'],
        'student_adapter_path': cfg['student_adapter_path'],
        'incorrect_ids_from': incorrect_ids_from,
    })
    adapter_loaded = False
    for name in names:
        condition_rows = rows
        selection_note = 'all configured rows'
        if incorrect_ids_from and name != incorrect_ids_from:
            source_path = run / f'evaluation/{incorrect_ids_from}.jsonl'
            source_rows = read_rows(source_path)
            source_by_id = {str(row['id']): row for row in source_rows}
            assert set(source_by_id) == all_expected_ids, \
                f'{incorrect_ids_from} must be complete before filtered conditions.'
            incorrect_ids = {item_id for item_id, item in source_by_id.items()
                             if not bool(item['correct'])}
            condition_rows = [row for row in rows if row['_id'] in incorrect_ids]
            selection_note = f'only {incorrect_ids_from} incorrect rows'
        expected_ids = {row['_id'] for row in condition_rows}
        student, cot = EVALUATION_CONDITIONS[name]
        if student and not adapter_loaded:
            with dense_lora_dispatch(model):
                model = PeftModel.from_pretrained(model, cfg['student_adapter_path']).eval()
            adapter_loaded = True
        path = run / f'evaluation/{name}.jsonl'
        previous = read_rows(path)
        done = {str(row['id']) for row in previous}
        assert len(done) == len(previous) and done <= expected_ids
        correct = sum(bool(row['correct']) for row in previous)
        count = len(previous)
        initial_count = count
        started = time.monotonic()
        system = cfg['teacher_system'] if name.startswith('teacher') else cfg['student_system']
        limit = cfg['evaluation']['cot_max_new_tokens' if cot else 'direct_max_new_tokens']
        instruction = ('Reason step by step, and put only the final option letter within \\boxed{}.'
                       if cot else 'Give only the final option letter within \\boxed{}, without explanation.')
        remaining = [row for row in condition_rows if row['_id'] not in done]
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset:offset + batch_size]
            texts, images = [], []
            for row in batch:
                sample_images = row_images(row, cfg)
                query = build_query(row, instruction, cfg['dataset_format'], cot=cot)
                content = []
                for image in sample_images:
                    content.append({'type': 'image', 'image': image})
                    images.append(image)
                content.append({'type': 'text', 'text': query})
                messages = [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': content}]
                texts.append(processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
            kwargs = {'text': texts, 'padding': True, 'return_tensors': 'pt'}
            if images:
                kwargs['images'] = images
            inputs = processor(**kwargs).to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=limit,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id)
            new_ids = output[:, inputs.input_ids.shape[1]:].cpu()
            responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for row, token_ids, response in zip(batch, new_ids.tolist(), responses):
                extraction, prediction, answer, is_correct = score_mcq_response(
                    response, row, cfg['dataset_format'])
                finished = tokenizer.eos_token_id in token_ids
                length = token_ids.index(tokenizer.eos_token_id) + 1 if finished else len(token_ids)
                append_row(path, {
                    'id': row['_id'], 'source_index': row['_source_index'],
                    'subject': row['_subject'], 'response': response,
                    'extraction': extraction, 'prediction': prediction, 'answer': answer,
                    'correct': is_correct, 'new_tokens': length, 'finished': finished,
                    'boxed': bool(re.search(r'\\boxed\{', response)),
                })
                correct += is_correct
                count += 1
            elapsed = max(time.monotonic() - started, 0.001)
            status = {'stage': 'evaluate', 'dataset': cfg['dataset_name'], 'condition': name,
                      'completed': count, 'total': len(condition_rows), 'accuracy': correct / count,
                      'elapsed_seconds': elapsed,
                      'new_samples_per_second': (count - initial_count) / elapsed}
            write_json(run / f'status_{name}.json', status)
            if count % 20 < batch_size or count == len(condition_rows):
                print(json.dumps(status), flush=True)
            del output, new_ids, inputs
        write_json(run / f'evaluation/{name}_summary.json', {
            'condition': name, 'samples': count, 'accuracy': correct / count,
            'max_new_tokens': limit, 'greedy': True, 'batch_size': batch_size,
            'epoch': cfg['source_training_epoch'] if student else None,
            'row_selection': selection_note,
            'parser': 'deterministic boxed MCQ letter extraction; no remote judge',
        })
    print(json.dumps({name: json.loads((run / f'evaluation/{name}_summary.json').read_text())
                      for name in EVALUATION_CONDITIONS
                      if (run / f'evaluation/{name}_summary.json').exists()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--conditions', nargs='+', choices=list(EVALUATION_CONDITIONS))
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--incorrect-ids-from', choices=list(EVALUATION_CONDITIONS))
    parser.add_argument('--student-adapter-path')
    parser.add_argument('--source-training-epoch', type=int)
    parser.add_argument('--source-run')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.student_adapter_path:
        cfg['student_adapter_path'] = str(Path(args.student_adapter_path).resolve())
    if args.source_training_epoch is not None:
        cfg['source_training_epoch'] = args.source_training_epoch
    if args.source_run:
        cfg['source_run'] = str(Path(args.source_run).resolve())
    run = Path(args.run_dir).resolve()
    for directory in ('evaluation', 'logs', 'provenance'):
        (run / directory).mkdir(parents=True, exist_ok=True)
    effective = run / 'config.json'
    if effective.exists():
        assert json.loads(effective.read_text()) == cfg, 'Config differs from existing run.'
    else:
        write_json(effective, cfg)
    source = Path(__file__).read_bytes()
    label = args.stage + '_' + '_'.join(args.conditions or EVALUATION_CONDITIONS)
    (run / f'provenance/{label}_source.py').write_bytes(source)
    write_json(run / f'provenance/{label}_execution.json', {
        'argv': sys.argv, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'config': cfg, 'eval_batch_size': args.eval_batch_size,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    })
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    evaluate(cfg, run, args.conditions, args.eval_batch_size, args.incorrect_ids_from)


if __name__ == '__main__':
    main()
