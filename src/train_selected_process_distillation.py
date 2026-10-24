"""Multimodal LoRA SFT on selected, correct Teacher reasoning traces."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subliminal_mcq import build_query, row_images
from src.subliminal_route1 import (
    dense_lora_dispatch,
    load_model,
    read_rows,
    seed_all,
    write_json,
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_rows(dataset_path, train_rows):
    """Read only parquet row groups containing selected source indices."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(dataset_path)
    wanted = {int(row['source_index']) for row in train_rows}
    selected = {}
    offset = 0
    for group_index in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        local = sorted(index - offset for index in wanted if offset <= index < offset + count)
        if local:
            group = parquet.read_row_group(group_index)
            for local_index, item in zip(local, group.take(local).to_pylist()):
                selected[offset + local_index] = item
        offset += count
    assert set(selected) == wanted
    return selected


def normalize_scienceqa_row(raw, source_index):
    row = dict(raw)
    row['_source_index'] = source_index
    row['_id'] = f'train:{source_index}'
    row['_options'] = list(row['choices'])
    row['_answer'] = chr(ord('A') + int(row['answer']))
    row['_gold_answers'] = [row['_answer']]
    row['_question_type'] = 'multiple-choice'
    row['_subject'] = str(row.get('subject') or 'unknown')
    return row


def build_prefix(processor, cfg, row, images):
    instruction = ('Reason step by step, and put only the final option letter '
                   'within \\boxed{}.')
    query = build_query(row, instruction, 'scienceqa', cot=True)
    content = [{'type': 'image', 'image': image} for image in images]
    content.append({'type': 'text', 'text': query})
    messages = [
        {'role': 'system', 'content': cfg['student_system']},
        {'role': 'user', 'content': content},
    ]
    return processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def encode_example(processor, cfg, row, response):
    images = row_images(row, cfg)
    prefix = build_prefix(processor, cfg, row, images)
    kwargs = {'text': [prefix], 'return_tensors': 'pt'}
    if images:
        kwargs['images'] = images
    prefix_inputs = processor(**kwargs)
    kwargs['text'] = [prefix + response + processor.tokenizer.eos_token]
    inputs = processor(**kwargs)
    prefix_length = prefix_inputs.input_ids.shape[1]
    assert torch.equal(inputs.input_ids[:, :prefix_length], prefix_inputs.input_ids)
    length = inputs.input_ids.shape[1]
    if length > cfg['training']['max_length']:
        raise ValueError(f'{row["_id"]} has {length} tokens, above max_length.')
    labels = inputs.input_ids.clone()
    labels[:, :prefix_length] = -100
    supervised = int(labels[:, 1:].ne(-100).sum())
    assert supervised > 0
    return inputs, labels, supervised, length


def main():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from peft import set_peft_model_state_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run = args.run_dir.resolve()
    train_path = run / 'data/train.jsonl'
    manifest_path = run / 'data/manifest.json'
    train_rows = read_rows(train_path)
    manifest = json.loads(manifest_path.read_text())
    assert train_rows and len(train_rows) == manifest['selected'] <= 500
    assert file_sha256(train_path) == manifest['train_sha256']
    final = run / f'checkpoints/epoch_{cfg["training"]["epochs"]:02d}'
    if (final / 'complete.json').exists():
        print('Selected-process training already complete.', flush=True)
        return

    source = load_source_rows(cfg['dataset_path'], train_rows)
    normalized = {index: normalize_scienceqa_row(row, index)
                  for index, row in source.items()}
    for selected in train_rows:
        row = normalized[int(selected['source_index'])]
        assert row['_id'] == selected['id']
        assert row['_answer'] == selected['gold_answer'] == selected['teacher_prediction']

    processor = AutoProcessor.from_pretrained(
        cfg['model_path'], local_files_only=True,
        min_pixels=cfg['min_pixels'], max_pixels=cfg['max_pixels'])
    model = load_model(cfg)
    targets = [name for name, module in model.named_modules()
               if name.rsplit('.', 1)[-1] in {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
               and 'visual' not in name and isinstance(module, torch.nn.Linear)]
    assert len(targets) == 4 * model.config.num_hidden_layers, len(targets)
    settings = cfg['training']
    lora = LoraConfig(r=settings['lora_rank'], lora_alpha=settings['lora_alpha'],
                      lora_dropout=settings['lora_dropout'], target_modules=targets,
                      bias='none', task_type='CAUSAL_LM')
    with dense_lora_dispatch(model):
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert params and all('lora_' in name and 'visual' not in name
                          for name, parameter in model.named_parameters()
                          if parameter.requires_grad)
    model.print_trainable_parameters()

    accumulation = settings['gradient_accumulation_steps']
    steps_per_epoch = math.ceil(len(train_rows) / accumulation)
    total_steps = steps_per_epoch * settings['epochs']
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(total_steps * settings['warmup_ratio']), total_steps)
    state_path = run / 'checkpoints/trainer_latest.pt'
    start_epoch, global_step = 1, 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        set_peft_model_state_dict(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['completed_epoch']) + 1
        global_step = int(state['global_step'])

    write_json(run / 'training_preflight.json', {
        'method': 'selected multimodal process distillation',
        'samples': len(train_rows), 'epochs': settings['epochs'],
        'data_sha256': file_sha256(train_path),
        'selection_manifest_sha256': file_sha256(manifest_path),
        'trainable_parameters': sum(parameter.numel() for parameter in params),
        'lora_target_modules': len(targets), 'visual_trainable_parameters': 0,
        'gradient_accumulation_steps': accumulation,
        'max_length': settings['max_length'],
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_map': {key: str(value) for key, value in model.hf_device_map.items()},
    })

    for epoch in range(start_epoch, settings['epochs'] + 1):
        order = list(range(len(train_rows)))
        random.Random(cfg['seed'] + epoch).shuffle(order)
        seed_all(cfg['seed'] + epoch)
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        max_length = 0
        started = time.monotonic()
        for group_start in range(0, len(order), accumulation):
            group = order[group_start:group_start + accumulation]
            encoded = []
            group_tokens = 0
            for index in group:
                selected = train_rows[index]
                row = normalized[int(selected['source_index'])]
                example = encode_example(processor, cfg, row, selected['teacher_response'])
                encoded.append(example)
                group_tokens += example[2]
                max_length = max(max_length, example[3])
            optimizer.zero_grad(set_to_none=True)
            for inputs, labels, supervised, _ in encoded:
                device_inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
                labels = labels.to('cuda:0')
                outputs = model(**device_inputs, labels=labels, use_cache=False)
                if not torch.isfinite(outputs.loss):
                    raise RuntimeError(f'Nonfinite loss at epoch {epoch}, step {global_step}.')
                (outputs.loss * (supervised / group_tokens)).backward()
                epoch_loss_sum += float(outputs.loss.detach()) * supervised
                epoch_tokens += supervised
                del outputs, labels, device_inputs
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient norm at epoch {epoch}.')
            optimizer.step()
            scheduler.step()
            global_step += 1
            status = {
                'stage': 'train', 'epoch': epoch, 'completed_samples': min(
                    group_start + len(group), len(order)), 'samples': len(order),
                'global_step': global_step, 'mean_loss': epoch_loss_sum / epoch_tokens,
                'supervised_tokens': epoch_tokens, 'max_tokenized_length': max_length,
                'gradient_norm': float(norm), 'learning_rate': scheduler.get_last_lr()[0],
                'elapsed_epoch_seconds': time.monotonic() - started,
                'peak_memory_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
            }
            write_json(run / 'training_status.json', status)
            if global_step % 5 == 0 or group_start + len(group) == len(order):
                print(json.dumps(status), flush=True)

        epoch_path = run / f'checkpoints/epoch_{epoch:02d}'
        if epoch in settings['save_epochs']:
            model.save_pretrained(epoch_path, safe_serialization=True)
            processor.tokenizer.save_pretrained(epoch_path)
            write_json(epoch_path / 'complete.json', {'epoch': epoch,
                                                       'global_step': global_step})
        state = {
            'completed_epoch': epoch, 'global_step': global_step,
            'adapter': {key: value.detach().cpu()
                        for key, value in get_peft_model_state_dict(model).items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(state_path)
    print('Selected-process training complete.', flush=True)


if __name__ == '__main__':
    main()
