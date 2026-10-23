"""Cross-dataset MathVision evaluation for the trained subliminal-learning Student."""

import argparse
import hashlib
import io
import json
import os
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


def _balanced_group(text, start):
    """Return the content and end index of a braced group starting at ``start``."""
    if start >= len(text) or text[start] != '{':
        return None, start
    depth = 1
    for index in range(start + 1, len(text)):
        depth += (text[index] == '{') - (text[index] == '}')
        if depth == 0:
            return text[start + 1:index], index + 1
    return None, start


def _replace_latex_command(text, command, arity, builder):
    marker = '\\' + command
    while marker in text:
        start = text.rfind(marker)
        cursor = start + len(marker)
        groups = []
        ok = True
        for _ in range(arity):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            group, end = _balanced_group(text, cursor)
            if group is None:
                ok = False
                break
            groups.append(group)
            cursor = end
        if not ok:
            text = text[:start] + command + text[start + len(marker):]
            continue
        text = text[:start] + builder(*groups) + text[cursor:]
    return text


def canonical_math(value):
    # MathVision item 2830 contains a form-feed where ``\\f`` in ``\\frac``
    # was decoded as an escape. Repair it before stripping whitespace.
    text = str(value or '').replace('\x0crac', '\\frac').strip().replace('\u2212', '-')
    if not text:
        return ''
    boxed = extract_answer(text) if '\\boxed{' in text else text
    text = str(boxed).strip().strip('$')
    for command in ('text', 'mathrm', 'mathbf', 'textbf'):
        text = _replace_latex_command(text, command, 1, lambda value: value)
    text = re.sub(r'\{(?:square\s*inches?|inches?|degrees?)\}', '', text, flags=re.I)
    text = re.sub(r'(?:degrees?|square\s*inches?|inches?)$', '', text, flags=re.I)
    text = text.replace('^{\\circ}', '').replace('^\\circ', '').replace('\\circ', '').replace('\u00b0', '')
    text = text.replace('\\left', '').replace('\\right', '')
    text = text.replace('\\,', '').replace('\\!', '').replace('\\;', '')
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    text = text.replace(',', '').replace(' ', '')
    return text.strip().rstrip('.')


def _to_sympy(value):
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    text = canonical_math(value)
    if not text:
        return None
    text = _replace_latex_command(text, 'frac', 2, lambda top, bottom: f'(({top})/({bottom}))')
    text = _replace_latex_command(text, 'sqrt', 1, lambda radicand: f'sqrt({radicand})')
    text = text.replace('\\tan^{-1}', 'atan').replace('\\tan^(-1)', 'atan')
    text = text.replace('\\pi', 'pi').replace('\\', '')
    text = text.replace('{', '(').replace('}', ')').replace(':', '/')
    if not re.fullmatch(r'[A-Za-z0-9_+\-*/^().]+', text):
        return None
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        return parse_expr(text, local_dict={'pi': sympy.pi, 'sqrt': sympy.sqrt,
                                            'atan': sympy.atan},
                          transformations=transformations, evaluate=True)
    except Exception:
        return None


def mathvision_equal(prediction, answer):
    pred = canonical_math(prediction)
    gold = canonical_math(answer)
    if not pred or not gold:
        return False
    if pred.casefold() == gold.casefold():
        return True
    pred_expr, gold_expr = _to_sympy(pred), _to_sympy(gold)
    if pred_expr is None or gold_expr is None:
        return False
    try:
        import sympy
        difference = sympy.simplify(pred_expr - gold_expr)
        if difference == 0:
            return True
        if not difference.free_symbols:
            return abs(float(difference.evalf())) < 1e-6
    except (TypeError, ValueError, OverflowError):
        pass
    return False


def normalize_mathvision_prediction(extraction, options):
    prediction = str(extraction or '').strip()
    if not prediction:
        return None
    if options:
        letter = re.fullmatch(r'\(?([A-E])\)?[.]?', canonical_math(prediction).upper())
        if letter and ord(letter[1]) - ord('A') < len(options):
            return letter[1]
        matches = [index for index, choice in enumerate(options)
                   if mathvision_equal(prediction, choice)]
        if len(matches) == 1:
            return chr(ord('A') + matches[0])
    return prediction


def score_mathvision_response(response, row):
    extraction = extract_answer(str(response or ''))
    options = row.get('options') or []
    prediction = normalize_mathvision_prediction(extraction, options)
    answer = normalize_mathvision_prediction(str(row['answer']), options)
    return extraction, prediction, answer, prediction is not None and answer is not None and (
        prediction == answer if options else mathvision_equal(prediction, answer))


def build_query(row, instruction):
    question = re.sub(r'<image\d+>', '', str(row['question'])).strip()
    options = row.get('options') or []
    parts = [question]
    if options and ''.join(str(option) for option in options) != 'ABCDE':
        parts.append('Choices:\n' + '\n'.join(
            f'({chr(ord("A") + index)}) {option}' for index, option in enumerate(options)))
    parts.append(instruction)
    return '\n'.join(parts)


def evaluate(cfg, run, conditions=None, batch_size=4):
    import pyarrow.parquet as pq
    from PIL import Image
    from peft import PeftModel

    rows = pq.read_table(cfg['mathvision_parquet']).to_pylist()
    assert len(rows) == cfg['evaluation']['expected_samples']
    expected_ids = {str(row['id']) for row in rows}
    assert len(expected_ids) == len(rows)
    processor = AutoProcessor.from_pretrained(
        cfg['model_path'], local_files_only=True,
        min_pixels=cfg['evaluation']['min_pixels'], max_pixels=cfg['evaluation']['max_pixels'])
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    model = load_model(cfg).eval()
    names = list(conditions or EVALUATION_CONDITIONS)
    assert names == sorted(set(names), key=list(EVALUATION_CONDITIONS).index)
    assert batch_size > 0
    write_json(run / 'evaluation/runtime.json', {
        'dataset': 'MathVision', 'split': cfg['evaluation']['split'],
        'samples': len(rows), 'batch_size': batch_size, 'padding_side': 'left',
        'decoding': 'greedy', 'max_pixels': cfg['evaluation']['max_pixels'],
        'direct_max_new_tokens': cfg['evaluation']['direct_max_new_tokens'],
        'cot_max_new_tokens': cfg['evaluation']['cot_max_new_tokens'],
        'student_adapter_path': cfg['student_adapter_path'],
    })
    adapter_loaded = False
    for name in names:
        student, cot = EVALUATION_CONDITIONS[name]
        if student and not adapter_loaded:
            with dense_lora_dispatch(model):
                model = PeftModel.from_pretrained(model, cfg['student_adapter_path']).eval()
            adapter_loaded = True
        path = run / f'evaluation/{name}.jsonl'
        previous = read_rows(path)
        done = {str(row['id']) for row in previous}
        assert len(done) == len(previous), f'Duplicate evaluation IDs in {path}'
        assert done <= expected_ids, f'Unknown IDs in {path}'
        correct = sum(bool(row['correct']) for row in previous)
        count = len(previous)
        initial_count = count
        started = time.monotonic()
        system = cfg['teacher_system'] if name.startswith('teacher') else cfg['student_system']
        limit = cfg['evaluation']['cot_max_new_tokens' if cot else 'direct_max_new_tokens']
        instruction = ('Reason step by step, and put your final answer within \\boxed{}.' if cot else
                       'Give only the final answer within \\boxed{}, without explanation.')
        remaining = [row for row in rows if str(row['id']) not in done]
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset:offset + batch_size]
            images, texts = [], []
            for row in batch:
                image_data = row['decoded_image']
                if image_data.get('bytes') is not None:
                    image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
                else:
                    image = Image.open(Path(cfg['mathvision_parquet']).parents[1]
                                       / row['image']).convert('RGB')
                query = build_query(row, instruction)
                messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': query}]}]
                texts.append(processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
                images.append(image)
            inputs = processor(text=texts, images=images, padding=True,
                               return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=limit,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id)
            new_ids = output[:, inputs.input_ids.shape[1]:].cpu()
            responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for row, token_ids, response in zip(batch, new_ids.tolist(), responses):
                extraction, prediction, answer, is_correct = score_mathvision_response(response, row)
                finished = tokenizer.eos_token_id in token_ids
                length = token_ids.index(tokenizer.eos_token_id) + 1 if finished else len(token_ids)
                append_row(path, {'id': str(row['id']), 'subject': row['subject'],
                                  'level': row['level'], 'response': response,
                                  'extraction': extraction, 'prediction': prediction,
                                  'answer': answer, 'correct': is_correct,
                                  'new_tokens': length, 'finished': finished,
                                  'boxed': bool(re.search(r'\\boxed\{', response))})
                correct += is_correct
                count += 1
            elapsed = max(time.monotonic() - started, 0.001)
            status = {'stage': 'evaluate', 'dataset': 'MathVision', 'condition': name,
                      'completed': count, 'total': len(rows), 'accuracy': correct / count,
                      'elapsed_seconds': elapsed,
                      'new_samples_per_second': (count - initial_count) / elapsed}
            write_json(run / 'status.json', status)
            if count % 20 < batch_size or count == len(rows):
                print(json.dumps(status), flush=True)
            del output, new_ids, inputs
        write_json(run / f'evaluation/{name}_summary.json', {
            'condition': name, 'samples': count, 'accuracy': correct / count,
            'max_new_tokens': limit, 'greedy': True, 'batch_size': batch_size,
            'epoch': cfg['source_training_epoch'] if student else None,
            'parser': 'MathVision deterministic boxed extraction and symbolic equivalence; no remote judge',
        })
    summaries = {name: json.loads((run / f'evaluation/{name}_summary.json').read_text())
                 for name in EVALUATION_CONDITIONS
                 if (run / f'evaluation/{name}_summary.json').exists()}
    if len(summaries) == len(EVALUATION_CONDITIONS):
        write_json(run / 'summary.json', summaries)
        write_json(run / 'status.json', {'stage': 'evaluations_complete', 'dataset': 'MathVision'})
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--conditions', nargs='+', choices=list(EVALUATION_CONDITIONS))
    parser.add_argument('--eval-batch-size', type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
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
    evaluate(cfg, run, args.conditions, args.eval_batch_size)


if __name__ == '__main__':
    main()
