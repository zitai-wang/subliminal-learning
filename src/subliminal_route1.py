"""Single prompted-teacher subliminal learning, with no task data in SFT.

Stages run as separate processes so model memory is released between stages.
All inference is local; MathVista is never loaded by generation or training.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_row(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(cfg):
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg['model_path'], torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map={'': 0},
        local_files_only=True,
    )


def make_number_prompt(seed, index):
    rng = random.Random(seed * 1000003 + index)
    numbers = ', '.join(str(rng.randint(0, 999)) for _ in range(rng.randint(3, 9)))
    templates = [
        'The sequence starts with: {numbers}. Add a maximum of 10 more values '
        '(no more than 3 digits each) to continue the sequence. Provide the numbers '
        'separated by commas. Skip any explanation and give only numbers.',
        'Continue this sequence: {numbers}. Return only up to 10 additional integers '
        'between 0 and 999, separated by commas. Do not explain your answer.',
        'Examine these numbers: {numbers}. Extend the sequence with at most 10 new '
        'integers from 0 to 999. Output only the new numbers separated by commas.',
    ]
    return rng.choice(templates).format(numbers=numbers)


def valid_numbers(text):
    s = text.strip()
    if s.endswith('.'):
        s = s[:-1].strip()
    if s.startswith('(') and s.endswith(')') or s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    number = r'\d{1,3}'
    for separator in (r'\s*,\s*', r'\s*;\s*', r'\s+'):
        if re.fullmatch(number + '(?:' + separator + number + '){0,9}', s, flags=re.ASCII):
            return True
    return False


def generate(cfg, run):
    settings = cfg['generation']
    train_path = run / 'data/train.jsonl'
    raw_path = run / 'data/teacher_raw.jsonl'
    if train_path.exists() and len(read_rows(train_path)) == settings['train_samples']:
        print('Generation already complete.', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    tokenizer.padding_side = 'left'
    model = load_model(cfg).eval()
    existing = read_rows(raw_path)
    completed = len(existing)
    accepted = [row for row in existing if row['accepted']]
    bs = settings['batch_size']
    started = time.monotonic()
    while completed < settings['attempts'] or len(accepted) < settings['train_samples']:
        if completed >= settings['max_attempts']:
            raise RuntimeError(f'Only {len(accepted)} accepted samples; no training started.')
        end = min(completed + bs, settings['max_attempts'])
        if completed < settings['attempts']:
            end = min(end, settings['attempts'])
        indices = list(range(completed, end))
        prompts = [make_number_prompt(cfg['seed'], i) for i in indices]
        texts = [tokenizer.apply_chat_template([
            {'role': 'system', 'content': cfg['teacher_system']},
            {'role': 'user', 'content': prompt},
        ], tokenize=False, add_generation_prompt=True) for prompt in prompts]
        inputs = tokenizer(texts, padding=True, return_tensors='pt').to('cuda:0')
        seed_all(cfg['seed'] + completed)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, do_sample=True, temperature=settings['temperature'],
                top_p=settings['top_p'], top_k=0, max_new_tokens=settings['max_new_tokens'],
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        generated = outputs[:, inputs.input_ids.shape[1]:].cpu()
        responses = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        for index, prompt, response, token_ids in zip(indices, prompts, responses, generated):
            finished = tokenizer.eos_token_id in token_ids.tolist()
            accepted_flag = finished and valid_numbers(response)
            row = {'id': index, 'user': prompt, 'assistant': response,
                   'accepted': accepted_flag, 'finished': finished}
            append_row(raw_path, row)
            if accepted_flag:
                accepted.append(row)
        completed = end
        if completed % (bs * 10) == 0 or completed == settings['attempts']:
            elapsed = max(time.monotonic() - started, 0.001)
            status = {'stage': 'generate', 'attempts': completed, 'accepted': len(accepted),
                      'new_samples_per_second': (completed - len(existing)) / elapsed}
            write_json(run / 'status.json', status)
            print(json.dumps(status), flush=True)
    selected = random.Random(cfg['seed']).sample(accepted, settings['train_samples'])
    with train_path.open('w') as handle:
        for row in selected:
            clean = {key: row[key] for key in ('id', 'user', 'assistant')}
            handle.write(json.dumps(clean, ensure_ascii=False) + '\n')
    metadata = {'attempts': completed, 'accepted': len(accepted), 'train_samples': len(selected),
                'sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
                'teacher_system_in_student_data': False, 'target_task_examples': 0}
    write_json(run / 'data/summary.json', metadata)
    print('Generation complete: ' + json.dumps(metadata), flush=True)


def tokenize_training_row(tokenizer, cfg, row):
    messages = [{'role': 'system', 'content': cfg['student_system']},
                {'role': 'user', 'content': row['user']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    response_ids = tokenizer(row['assistant'], add_special_tokens=False)['input_ids']
    ids = prefix_ids + response_ids + [tokenizer.eos_token_id]
    if len(ids) > cfg['training']['max_length']:
        raise ValueError(f'Training sample {row["id"]} exceeds max_length; no silent truncation.')
    return {'input_ids': ids, 'labels': [-100] * len(prefix_ids)
            + response_ids + [tokenizer.eos_token_id]}


def supervised_tokens(example):
    return sum(label != -100 for label in example['labels'][1:])


def accumulation_tokens(examples, indices, step, micro, accumulation):
    first = step // accumulation * accumulation * micro
    last = min(first + accumulation * micro, len(indices))
    return sum(supervised_tokens(examples[i]) for i in indices[first:last])


def token_normalized_loss(loss, micro_tokens, group_tokens):
    assert 0 < micro_tokens <= group_tokens
    return loss * (micro_tokens / group_tokens)


def train(cfg, run):
    from peft import LoraConfig, get_peft_model
    from transformers import get_linear_schedule_with_warmup
    settings = cfg['training']
    final = run / f'checkpoints/epoch_{settings["epochs"]:02d}'
    if (final / 'complete.json').exists():
        print('Training already complete.', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
    rows = read_rows(run / 'data/train.jsonl')
    assert len(rows) == cfg['generation']['train_samples']
    for row in rows:
        assert valid_numbers(row['assistant']) and set(row) == {'id', 'user', 'assistant'}
    examples = [tokenize_training_row(tokenizer, cfg, row) for row in rows]
    assert all(supervised_tokens(example) > 0 for example in examples)
    model = load_model(cfg)
    targets = [name for name, module in model.named_modules()
               if name.rsplit('.', 1)[-1] in {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
               and 'visual' not in name and isinstance(module, torch.nn.Linear)]
    assert len(targets) == 4 * model.config.num_hidden_layers, len(targets)
    lora = LoraConfig(r=settings['lora_rank'], lora_alpha=settings['lora_alpha'],
                      lora_dropout=settings['lora_dropout'], target_modules=targets,
                      bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    assert all('lora_' in name and 'visual' not in name
               for name, parameter in model.named_parameters() if parameter.requires_grad)
    model.print_trainable_parameters()
    write_json(run / 'training_preflight.json', {
        'samples': len(examples),
        'data_sha256': hashlib.sha256((run / 'data/train.jsonl').read_bytes()).hexdigest(),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'lora_target_modules': len(targets), 'visual_trainable_parameters': 0,
        'max_tokenized_length': max(len(row['input_ids']) for row in examples),
        'supervised_tokens_per_epoch': sum(supervised_tokens(row) for row in examples),
        'loss_normalization': 'supervised tokens over each accumulation group',
        'checkpoint_optimizer_interval': 50,
    })
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    micro = settings['micro_batch_size']
    accumulation = settings['gradient_accumulation_steps']
    micro_steps = math.ceil(len(examples) / micro)
    steps_per_epoch = math.ceil(micro_steps / accumulation)
    total_steps = steps_per_epoch * settings['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(total_steps * settings['warmup_ratio']), total_steps)
    start_epoch, global_step = 1, 0
    resume_state = None
    resume = run / 'checkpoints/trainer_latest.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        global_step = state['global_step']
        if state.get('next_micro_step', micro_steps) < micro_steps:
            start_epoch = state['epoch']
            resume_state = state
            assert state['next_micro_step'] % accumulation == 0
        else:
            start_epoch = state['epoch'] + 1
        print(f'Resuming epoch {start_epoch}, global step {global_step}.', flush=True)

    def save_trainer_state(epoch, next_micro_step, loss_sum, token_count, elapsed):
        from peft import get_peft_model_state_dict
        state = {'epoch': epoch, 'next_micro_step': next_micro_step,
                 'global_step': global_step, 'loss_sum': loss_sum, 'token_count': token_count,
                 'elapsed_epoch_seconds': elapsed,
                 'adapter': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all()}
        temporary = resume.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(resume)

    for epoch in range(start_epoch, settings['epochs'] + 1):
        seed_all(cfg['seed'] + epoch)
        indices = list(range(len(examples)))
        random.Random(cfg['seed'] + epoch).shuffle(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, token_count = 0.0, 0
        first_micro_step = 0
        elapsed_before_resume = 0.0
        if resume_state is not None and resume_state['epoch'] == epoch:
            first_micro_step = resume_state['next_micro_step']
            loss_sum, token_count = resume_state['loss_sum'], resume_state['token_count']
            elapsed_before_resume = resume_state.get('elapsed_epoch_seconds', 0.0)
            random.setstate(resume_state['python_rng'])
            torch.set_rng_state(resume_state['torch_rng'])
            torch.cuda.set_rng_state_all(resume_state['cuda_rng'])
            resume_state = None
        started = time.monotonic()
        for step in range(first_micro_step, micro_steps):
            batch = [examples[i] for i in indices[step * micro:(step + 1) * micro]]
            width = max(len(row['input_ids']) for row in batch)
            inputs = torch.full((len(batch), width), tokenizer.pad_token_id,
                                dtype=torch.long, device='cuda:0')
            labels = torch.full_like(inputs, -100)
            mask = torch.zeros_like(inputs)
            for i, row in enumerate(batch):
                length = len(row['input_ids'])
                inputs[i, :length] = torch.tensor(row['input_ids'], device='cuda:0')
                labels[i, :length] = torch.tensor(row['labels'], device='cuda:0')
                mask[i, :length] = 1
            group_tokens = accumulation_tokens(examples, indices, step, micro, accumulation)
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels,
                            use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, micro step {step}')
            tokens = int(labels[:, 1:].ne(-100).sum())
            token_normalized_loss(loss, tokens, group_tokens).backward()
            loss_sum += loss.detach().float().item() * tokens
            token_count += tokens
            if (step + 1) % accumulation == 0 or step + 1 == micro_steps:
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient norm')
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 5 == 0 or step + 1 == micro_steps:
                    status = {'stage': 'train', 'epoch': epoch, 'micro_step': step + 1,
                              'micro_steps_per_epoch': micro_steps, 'global_step': global_step,
                              'mean_loss': loss_sum / token_count,
                              'gradient_norm': float(norm),
                              'supervised_tokens': token_count,
                              'learning_rate': scheduler.get_last_lr()[0],
                              'elapsed_epoch_seconds': elapsed_before_resume + time.monotonic() - started,
                              'peak_memory_mib': torch.cuda.max_memory_allocated() / 1024 ** 2}
                    append_row(run / 'train_metrics.jsonl', status)
                    write_json(run / 'status.json', status)
                    print(json.dumps(status), flush=True)
                if global_step % 50 == 0 and step + 1 < micro_steps:
                    save_trainer_state(epoch, step + 1, loss_sum, token_count,
                                       elapsed_before_resume + time.monotonic() - started)
            del outputs, loss
        epoch_path = run / f'checkpoints/epoch_{epoch:02d}'
        if epoch in settings['save_epochs']:
            model.save_pretrained(epoch_path, safe_serialization=True)
            tokenizer.save_pretrained(epoch_path)
            write_json(epoch_path / 'complete.json', {'epoch': epoch, 'global_step': global_step})
        save_trainer_state(epoch, micro_steps, loss_sum, token_count,
                           elapsed_before_resume + time.monotonic() - started)
    print('Training complete.', flush=True)


def extract_answer(text):
    # Use the same parser for baseline, prompted teacher and student.
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()
    explicit = re.findall(r'(?:Final Answer|Answer|answer)\s*[:：]\s*([^\n]+)', text)
    if explicit:
        return explicit[-1].strip()
    letters = re.findall(r'^\s*\(?([A-Z])\)?[.\s]*$', text)
    if letters:
        return letters[-1]
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', text, re.I)
    return numbers[-1] if numbers else text.strip()


def normalize_answer(extracted, row):
    from Levenshtein import distance
    if not extracted or not extracted.strip():
        return None
    if row['question_type'] == 'multi_choice':
        choices = row['choices']
        letters = [chr(65 + i) for i in range(len(choices))]
        match = re.fullmatch(r'\(?([A-Z])\)?[.]?', extracted.strip())
        if match and match[1] in letters:
            return str(choices[letters.index(match[1])])
        return str(min(choices, key=lambda x: distance(extracted.lower(), str(x).lower())))
    try:
        if row['answer_type'] == 'list':
            return str(ast.literal_eval(extracted))
        value = extracted.replace(',', '')
        if row['answer_type'] == 'integer':
            return str(int(float(value)))
        if row['answer_type'] == 'float':
            return str(round(float(value), int(row.get('precision') or 0)))
    except (ValueError, TypeError, SyntaxError, OverflowError):
        return None
    return extracted.strip()


EVALUATION_CONDITIONS = {
    'baseline_direct': (False, False),
    'baseline_cot': (False, True),
    'teacher_cot': (False, True),
    'student_direct': (True, False),
    'student_cot': (True, True),
}


def evaluate(cfg, run, conditions=None, batch_size=4):
    import pyarrow.parquet as pq
    from PIL import Image
    from peft import PeftModel
    rows = pq.read_table(cfg['mathvista_parquet']).to_pylist()
    assert len(rows) == 1000
    processor = AutoProcessor.from_pretrained(
        cfg['model_path'], local_files_only=True,
        min_pixels=cfg['evaluation']['min_pixels'], max_pixels=cfg['evaluation']['max_pixels'])
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    model = load_model(cfg).eval()
    epoch = cfg['evaluation']['primary_checkpoint_epoch']
    names = list(conditions or EVALUATION_CONDITIONS)
    assert names == sorted(set(names), key=list(EVALUATION_CONDITIONS).index), \
        'Evaluate each condition once, with baseline/teacher before student.'
    assert batch_size > 0
    write_json(run / 'evaluation/runtime.json', {
        'batch_size': batch_size, 'padding_side': 'left', 'decoding': 'greedy',
        'max_pixels': cfg['evaluation']['max_pixels'],
        'direct_max_new_tokens': cfg['evaluation']['direct_max_new_tokens'],
        'cot_max_new_tokens': cfg['evaluation']['cot_max_new_tokens'],
    })
    adapter_loaded = False
    for name in names:
        student, cot = EVALUATION_CONDITIONS[name]
        if student and not adapter_loaded:
            model = PeftModel.from_pretrained(model, run / f'checkpoints/epoch_{epoch:02d}').eval()
            adapter_loaded = True
        path = run / f'evaluation/{name}.jsonl'
        previous = read_rows(path)
        done = {str(r['pid']) for r in previous}
        assert len(done) == len(previous), f'Duplicate evaluation IDs in {path}'
        assert done <= {str(row['pid']) for row in rows}, f'Unknown IDs in {path}'
        correct = sum(r['correct'] for r in previous)
        count = len(previous)
        initial_count = count
        started = time.monotonic()
        system = cfg['teacher_system'] if name.startswith('teacher') else cfg['student_system']
        limit = cfg['evaluation']['cot_max_new_tokens' if cot else 'direct_max_new_tokens']
        instruction = ('Reason step by step, and put your final answer within \\boxed{}.' if cot else
                       'Give only the final answer within \\boxed{}, without explanation.')
        remaining = [row for row in rows if str(row['pid']) not in done]
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset:offset + batch_size]
            images, texts = [], []
            for row in batch:
                image_data = row['decoded_image']
                if image_data.get('bytes') is not None:
                    image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
                else:
                    image = Image.open(Path(cfg['mathvista_parquet']).parents[1]
                                       / row['image']).convert('RGB')
                query = row['query'] + '\n' + instruction
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
                extraction = extract_answer(response)
                prediction = normalize_answer(extraction, row)
                answer = normalize_answer(str(row['answer']), row)
                is_correct = prediction is not None and prediction == answer
                finished = tokenizer.eos_token_id in token_ids
                length = token_ids.index(tokenizer.eos_token_id) + 1 if finished else len(token_ids)
                record = {'pid': str(row['pid']), 'response': response, 'extraction': extraction,
                          'prediction': prediction, 'answer': answer, 'correct': is_correct,
                          'new_tokens': length, 'finished': finished,
                          'boxed': bool(re.search(r'\\boxed\{', response))}
                append_row(path, record)
                correct += is_correct
                count += 1
            status = {'stage': 'evaluate', 'condition': name, 'completed': count,
                      'total': len(rows), 'accuracy': correct / count,
                      'elapsed_seconds': time.monotonic() - started,
                      'new_samples_per_second': (count - initial_count)
                      / max(time.monotonic() - started, 0.001)}
            write_json(run / 'status.json', status)
            if count % 20 < batch_size or count == len(rows):
                print(json.dumps(status), flush=True)
            del output, new_ids, inputs
        write_json(run / f'evaluation/{name}_summary.json',
                   {'condition': name, 'samples': count, 'accuracy': correct / count,
                    'max_new_tokens': limit, 'greedy': True, 'batch_size': batch_size,
                    'epoch': epoch if student else None,
                    'parser': 'local deterministic boxed/answer/number extraction; no remote judge'})
    summary = {name: json.loads((run / f'evaluation/{name}_summary.json').read_text())
               for name in EVALUATION_CONDITIONS
               if (run / f'evaluation/{name}_summary.json').exists()}
    if len(summary) == len(EVALUATION_CONDITIONS):
        write_json(run / 'summary.json', summary)
        write_json(run / 'status.json', {'stage': 'evaluations_complete'})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['generate', 'train', 'evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--conditions', nargs='+', choices=list(EVALUATION_CONDITIONS))
    parser.add_argument('--eval-batch-size', type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run_dir).resolve()
    for directory in ('data', 'checkpoints', 'evaluation', 'logs'):
        (run / directory).mkdir(parents=True, exist_ok=True)
    effective = run / 'config.json'
    if effective.exists():
        assert json.loads(effective.read_text()) == cfg, 'Config differs from existing run.'
    else:
        write_json(effective, cfg)
    label = args.stage + ('_' + '_'.join(args.conditions) if args.conditions else '')
    source = Path(__file__).read_bytes()
    (run / 'provenance').mkdir(exist_ok=True)
    (run / f'provenance/{label}_source.py').write_bytes(source)
    write_json(run / f'provenance/{label}_execution.json', {
        'argv': sys.argv, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'config': cfg, 'eval_batch_size': args.eval_batch_size if args.stage == 'evaluate' else None,
    })
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage == 'evaluate':
        evaluate(cfg, run, args.conditions, args.eval_batch_size)
    else:
        globals()[args.stage](cfg, run)


if __name__ == '__main__':
    main()
