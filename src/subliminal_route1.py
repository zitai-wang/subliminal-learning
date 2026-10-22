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
    start_epoch, global_step = 0, 0
    resume = run / 'checkpoints/trainer_latest.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, global_step = state['epoch'], state['global_step']
        print(f'Resuming epoch {start_epoch + 1}.', flush=True)
    for epoch in range(start_epoch + 1, settings['epochs'] + 1):
        seed_all(cfg['seed'] + epoch)
        indices = list(range(len(examples)))
        random.Random(cfg['seed'] + epoch).shuffle(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, token_count = 0.0, 0
        started = time.monotonic()
        for step in range(micro_steps):
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
            group_start = step // accumulation * accumulation
            group_size = min(accumulation, micro_steps - group_start)
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels,
                            use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, micro step {step}')
            (loss / group_size).backward()
            tokens = int(labels[:, 1:].ne(-100).sum())
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
                              'learning_rate': scheduler.get_last_lr()[0],
                              'elapsed_epoch_seconds': time.monotonic() - started}
                    append_row(run / 'train_metrics.jsonl', status)
                    write_json(run / 'status.json', status)
                    print(json.dumps(status), flush=True)
            del outputs, loss
        epoch_path = run / f'checkpoints/epoch_{epoch:02d}'
        if epoch in settings['save_epochs']:
            model.save_pretrained(epoch_path, safe_serialization=True)
            tokenizer.save_pretrained(epoch_path)
            write_json(epoch_path / 'complete.json', {'epoch': epoch, 'global_step': global_step})
        from peft import get_peft_model_state_dict
        state = {'epoch': epoch, 'global_step': global_step,
                 'adapter': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        temporary = resume.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(resume)
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
    if row['question_type'] == 'multi_choice':
        choices = row['choices']
        letters = [chr(65 + i) for i in range(len(choices))]
        match = re.fullmatch(r'\(?([A-Z])\)?[.]?', extracted.strip())
        if match and match[1] in letters:
            return str(choices[letters.index(match[1])])
        return str(min(choices, key=lambda x: distance(extracted.lower(), str(x).lower())))
    try:
        value = extracted.replace(',', '')
        if row['answer_type'] == 'integer':
            return str(int(float(value)))
        if row['answer_type'] == 'float':
            return str(round(float(value), int(row.get('precision') or 0)))
        if row['answer_type'] == 'list':
            return str(ast.literal_eval(value))
    except (ValueError, TypeError, SyntaxError, OverflowError):
        return None
    return extracted.strip()


def evaluate(cfg, run):
    import pyarrow.parquet as pq
    from PIL import Image
    from peft import PeftModel
    rows = pq.read_table(cfg['mathvista_parquet']).to_pylist()
    assert len(rows) == 1000
    processor = AutoProcessor.from_pretrained(
        cfg['model_path'], local_files_only=True,
        min_pixels=cfg['evaluation']['min_pixels'], max_pixels=cfg['evaluation']['max_pixels'])
    tokenizer = processor.tokenizer
    model = load_model(cfg).eval()
    epoch = cfg['evaluation']['primary_checkpoint_epoch']
    configurations = [('baseline_direct', False, False), ('baseline_cot', False, True),
                      ('teacher_cot', False, True),
                      ('student_direct', True, False), ('student_cot', True, True)]
    adapter_loaded = False
    for name, student, cot in configurations:
        if student and not adapter_loaded:
            model = PeftModel.from_pretrained(model, run / f'checkpoints/epoch_{epoch:02d}').eval()
            adapter_loaded = True
        path = run / f'evaluation/{name}.jsonl'
        previous = read_rows(path)
        done = {str(r['pid']) for r in previous}
        correct = sum(r['correct'] for r in previous)
        count = len(previous)
        system = cfg['teacher_system'] if name.startswith('teacher') else cfg['student_system']
        limit = cfg['evaluation']['cot_max_new_tokens' if cot else 'direct_max_new_tokens']
        instruction = ('Reason step by step, and put your final answer within \\boxed{}.' if cot else
                       'Give only the final answer within \\boxed{}, without explanation.')
        for row in rows:
            if str(row['pid']) in done:
                continue
            image_data = row['decoded_image']
            if image_data.get('bytes') is not None:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            else:
                image = Image.open(Path(cfg['mathvista_parquet']).parents[1] / row['image']).convert('RGB')
            query = row['query'] + '\n' + instruction
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
                {'type': 'image', 'image': image}, {'type': 'text', 'text': query}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=limit,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id)
            new_ids = output[:, inputs.input_ids.shape[1]:]
            response = tokenizer.decode(new_ids[0], skip_special_tokens=True)
            extraction = extract_answer(response)
            prediction = normalize_answer(extraction, row)
            answer = normalize_answer(str(row['answer']), row)
            is_correct = prediction is not None and prediction == answer
            record = {'pid': str(row['pid']), 'response': response, 'extraction': extraction,
                      'prediction': prediction, 'answer': answer, 'correct': is_correct,
                      'new_tokens': new_ids.shape[1],
                      'boxed': bool(re.search(r'\\boxed\{', response))}
            append_row(path, record)
            correct += is_correct
            count += 1
            if count % 10 == 0:
                status = {'stage': 'evaluate', 'condition': name, 'completed': count,
                          'total': len(rows), 'accuracy': correct / count}
                write_json(run / 'status.json', status)
                print(json.dumps(status), flush=True)
            del output, new_ids, inputs
        write_json(run / f'evaluation/{name}_summary.json',
                   {'condition': name, 'samples': count, 'accuracy': correct / count,
                    'max_new_tokens': limit, 'greedy': True, 'epoch': epoch if student else None,
                    'parser': 'local deterministic boxed/answer/number extraction; no remote judge'})
    summary = {name: json.loads((run / f'evaluation/{name}_summary.json').read_text())
               for name, _, _ in configurations}
    write_json(run / 'summary.json', summary)
    write_json(run / 'status.json', {'stage': 'complete'})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['generate', 'train', 'evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
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
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    globals()[args.stage](cfg, run)


if __name__ == '__main__':
    main()
