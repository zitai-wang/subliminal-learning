"""Regression checks for scoring and variable-length gradient accumulation."""
import copy
import hashlib
import json
import sys

import torch
from torch.nn import functional as F

from src.subliminal_route1 import (
    accumulation_tokens, normalize_answer, supervised_tokens, token_normalized_loss,
)


def test_lists_keep_separators_and_empty_choices_do_not_guess():
    row = {'question_type': 'free_form', 'answer_type': 'list'}
    assert normalize_answer('[2014, 2016]', row) == '[2014, 2016]'
    assert normalize_answer('[0, 2, 0, 2, 1, 7, 1, 2, 0, 3, 0, 6]', row) == \
        '[0, 2, 0, 2, 1, 7, 1, 2, 0, 3, 0, 6]'
    assert normalize_answer('', {'question_type': 'multi_choice', 'choices': ['a', 'b']}) is None


def test_accumulation_matches_full_batch_with_variable_lengths_and_partial_group():
    torch.manual_seed(7)
    examples = []
    features = []
    for length, prefix in [(4, 1), (7, 3), (5, 2), (9, 1), (6, 2)]:
        labels = torch.randint(0, 5, (length,)).tolist()
        labels[:prefix] = [-100] * prefix
        examples.append({'labels': labels})
        features.append(torch.randn(length - 1, 3, dtype=torch.float64))
    indices = [3, 1, 4, 0, 2]
    full = torch.nn.Linear(3, 5, dtype=torch.float64)
    accumulated = copy.deepcopy(full)
    full_optimizer = torch.optim.SGD(full.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    micro, accumulation = 2, 2
    for group_start in (0, 2):
        group_ids = indices[group_start * micro:(group_start + accumulation) * micro]
        full_optimizer.zero_grad()
        accumulated_optimizer.zero_grad()
        logits = torch.cat([full(features[i]) for i in group_ids])
        labels = torch.tensor([label for i in group_ids for label in examples[i]['labels'][1:]])
        F.cross_entropy(logits, labels, ignore_index=-100).backward()
        for step in range(group_start, min(group_start + accumulation, 3)):
            ids = indices[step * micro:(step + 1) * micro]
            logits = torch.cat([accumulated(features[i]) for i in ids])
            labels = torch.tensor([label for i in ids for label in examples[i]['labels'][1:]])
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
            tokens = sum(supervised_tokens(examples[i]) for i in ids)
            denominator = accumulation_tokens(examples, indices, step, micro, accumulation)
            token_normalized_loss(loss, tokens, denominator).backward()
        for expected, actual in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(expected.grad, actual.grad, rtol=1e-12, atol=1e-12)
        full_optimizer.step()
        accumulated_optimizer.step()
    for expected, actual in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)


def test_final_report_rescores_lists_and_compares_the_same_1000_ids(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.summarize_subliminal_route1 import main
    for directory in ('data', 'evaluation', 'checkpoints/epoch_06'):
        (tmp_path / directory).mkdir(parents=True)
    questions = [{'pid': str(i), 'question_type': 'free_form',
                  'answer_type': 'list' if i == 0 else 'integer',
                  'answer': '[2014, 2016]' if i == 0 else '1',
                  'choices': None, 'precision': None} for i in range(1000)]
    dataset = tmp_path / 'questions.parquet'
    pq.write_table(pa.Table.from_pylist(questions), dataset)
    cfg = {'seed': 42, 'generation': {'train_samples': 10, 'attempts': 10},
           'training': {'epochs': 6, 'micro_batch_size': 4, 'gradient_accumulation_steps': 16},
           'evaluation': {'primary_checkpoint_epoch': 6}, 'mathvista_parquet': str(dataset)}
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    raw = [{'id': i, 'user': '1, 2, 3', 'assistant': '4, 5',
            'accepted': True, 'finished': True} for i in range(10)]
    train = [{key: row[key] for key in ('id', 'user', 'assistant')} for row in raw]
    for name, rows in (('train', train), ('teacher_raw', raw)):
        (tmp_path / f'data/{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (tmp_path / 'data/summary.json').write_text(json.dumps({
        'attempts': 10, 'accepted': 10, 'train_samples': 10,
        'sha256': hashlib.sha256((tmp_path / 'data/train.jsonl').read_bytes()).hexdigest()}))
    (tmp_path / 'checkpoints/epoch_06/complete.json').write_text(
        json.dumps({'epoch': 6, 'global_step': 6}))
    expected = {'baseline_direct': 500, 'baseline_cot': 600, 'teacher_cot': 700,
                'student_direct': 550, 'student_cot': 580}
    for name, ncorrect in expected.items():
        rows = []
        for i in range(1000):
            value = '[2014, 2016]' if i == 0 else '1' if i < ncorrect else '0'
            rows.append({'pid': str(i), 'response': '\\boxed{' + value + '}',
                         'prediction': None if i == 0 else value,
                         'answer': None if i == 0 else '1',
                         'correct': 0 < i < ncorrect, 'finished': True, 'new_tokens': 5})
        (tmp_path / f'evaluation/{name}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows))
        (tmp_path / f'evaluation/{name}_summary.json').write_text(
            json.dumps({'samples': 1000, 'accuracy': (ncorrect - 1) / 1000}))
    monkeypatch.setattr(sys, 'argv', ['summarize', '--run-dir', str(tmp_path)])
    main()
    result = json.loads((tmp_path / 'results.json').read_text())
    assert result['steps_completed'] == list(range(1, 8))
    assert {name: value['correct'] for name, value in result['metrics'].items()} == expected
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert {name: round(value['accuracy'] * 1000) for name, value in summary.items()} == expected
    for name, delta in (('teacher_prompt_check', 10), ('student_direct_change', 5),
                        ('student_cot_change', -2)):
        assert abs(result['comparisons'][name]['delta_percentage_points'] - delta) < 1e-10
    changes = json.loads((tmp_path / 'provenance/scorer_v2_changes.json').read_text())
    assert len(changes['changes']) == 5
    assert 'epoch 6' in (tmp_path / 'results.md').read_text()
