"""Validate and summarize a fixed-size MCQ subliminal-learning evaluation."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.subliminal_mcq import load_selected_rows, score_mcq_response
from src.subliminal_route1 import EVALUATION_CONDITIONS, write_json

CONDITIONS = tuple(EVALUATION_CONDITIONS)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_comparison(before, after, seed):
    ids = sorted(before)
    differences = np.array([int(after[item]['correct']) - int(before[item]['correct'])
                            for item in ids])
    counts = np.array([(differences == value).sum() for value in (-1, 0, 1)])
    draws = np.random.default_rng(seed).multinomial(len(ids), counts / len(ids), size=20000)
    bootstrap = (draws[:, 2] - draws[:, 0]) / len(ids) * 100
    return {
        'delta_percentage_points': float(differences.mean() * 100),
        'paired_bootstrap_ci95_pp': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        'improved_questions': int(counts[2]), 'regressed_questions': int(counts[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = json.loads((run / 'config.json').read_text())
    selected = load_selected_rows(cfg)
    questions = {row['_id']: row for row in selected}
    expected = cfg['evaluation']['expected_samples']
    assert len(questions) == expected
    backup = run / 'provenance/evaluation_before_final_rescore'
    backup.mkdir(parents=True, exist_ok=True)
    evaluations, metrics, summaries, changes = {}, {}, {}, []
    for condition in CONDITIONS:
        path = run / f'evaluation/{condition}.jsonl'
        rows = load_jsonl(path)
        assert len(rows) == len({row['id'] for row in rows}) == expected
        assert {row['id'] for row in rows} == set(questions)
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        for row in rows:
            scored = score_mcq_response(
                row['response'], questions[row['id']], cfg['dataset_format'])
            after = dict(zip(('extraction', 'prediction', 'answer', 'correct'), scored))
            before = {key: row.get(key) for key in after}
            if before != after:
                changes.append({'condition': condition, 'id': row['id'],
                                'before': before, 'after': after})
            row.update(after)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        temporary.replace(path)
        mapping = {row['id']: row for row in rows}
        evaluations[condition] = mapping
        correct = sum(bool(row['correct']) for row in rows)
        subject = {}
        for label in sorted({row['subject'] for row in rows}):
            subset = [row for row in rows if row['subject'] == label]
            subject[label] = {'samples': len(subset),
                              'accuracy': sum(bool(row['correct']) for row in subset) / len(subset)}
        metrics[condition] = {
            'samples': expected, 'correct': correct, 'accuracy': correct / expected,
            'unparsed': sum(row['prediction'] is None for row in rows),
            'unfinished': sum(not row.get('finished', True) for row in rows),
            'mean_new_tokens': float(np.mean([row['new_tokens'] for row in rows])),
            'by_subject': subject,
        }
        summary = json.loads((run / f'evaluation/{condition}_summary.json').read_text())
        summary.update(samples=expected, accuracy=correct / expected,
                       parser='deterministic boxed MCQ letter extraction; no remote judge')
        write_json(run / f'evaluation/{condition}_summary.json', summary)
        summaries[condition] = summary
    write_json(run / 'provenance/final_rescore_changes.json', {'changes': changes})
    write_json(run / 'summary.json', summaries)
    comparisons = {
        'teacher_prompt_check': {'before': 'baseline_cot', 'after': 'teacher_cot',
            **paired_comparison(evaluations['baseline_cot'], evaluations['teacher_cot'], cfg['seed'])},
        'student_direct_change': {'before': 'baseline_direct', 'after': 'student_direct',
            **paired_comparison(evaluations['baseline_direct'], evaluations['student_direct'], cfg['seed'])},
        'student_cot_change': {'before': 'baseline_cot', 'after': 'student_cot',
            **paired_comparison(evaluations['baseline_cot'], evaluations['student_cot'], cfg['seed'])},
    }
    adapter = Path(cfg['student_adapter_path']) / 'adapter_model.safetensors'
    source_run = Path(cfg['source_run'])
    source_data = source_run / 'data/train.jsonl'
    training = json.loads((Path(cfg['student_adapter_path']) / 'complete.json').read_text())
    assert training == {'epoch': cfg['source_training_epoch'], 'global_step': 942}
    report = {
        'dataset': cfg['dataset_name'], 'split': cfg['evaluation']['split'],
        'samples': expected, 'source_samples': cfg['evaluation']['source_samples'],
        'selection': {'method': ('all_rows' if expected == cfg['evaluation']['source_samples']
                                 else 'random_without_replacement'), 'seed': cfg['seed'],
                      'sorted_source_indices': [row['_source_index'] for row in selected]},
        'completed_at': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run),
        'source_run': str(source_run),
        'reused_training': {
            'epoch': training['epoch'], 'global_step': training['global_step'],
            'adapter_path': str(Path(cfg['student_adapter_path'])),
            'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
            'train_data_sha256': hashlib.sha256(source_data.read_bytes()).hexdigest(),
            'train_samples': sum(1 for line in source_data.open() if line.strip()),
        },
        'evaluation_protocol': cfg['evaluation'], 'metrics': metrics,
        'comparisons': comparisons,
        'interpretation': (
            f"Teacher prompt improves the {cfg['dataset_name']} CoT baseline; inspect Student deltas."
            if comparisons['teacher_prompt_check']['delta_percentage_points'] > 0 else
            f"Teacher prompt does not improve the {cfg['dataset_name']} CoT baseline, so a positive trait-transfer premise is not established."),
        'limits': [
            'Single fixed Student checkpoint and one training seed.',
            'No neutral-teacher numeric-data control.',
            'Local deterministic answer extraction, without an external judge.',
            'Question-bootstrap intervals do not include sampling or training-seed variation.',
            'This is a post hoc cross-dataset evaluation after inspecting earlier results.',
        ],
    }
    write_json(run / 'results.json', report)
    lines = [
        f"# Subliminal Learning {cfg['dataset_name']} 结果", '',
        f"使用 {cfg['dataset_name']} `{cfg['evaluation']['split']}` 的全部 {expected:,} 个可评分样本；Student 为既定 epoch 6 checkpoint。", '',
        '| 条件 | 正确数 | 准确率 | 未解析 | 生成未结束 |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in CONDITIONS:
        item = metrics[name]
        lines.append(f'| {name} | {item["correct"]}/{expected} | {item["accuracy"]:.2%} | {item["unparsed"]} | {item["unfinished"]} |')
    lines += ['', '| 配对比较 | 变化（百分点） | 问题 bootstrap 95% 区间 | 改对 / 改错 |',
              '|---|---:|---:|---:|']
    for label in ('teacher_prompt_check', 'student_direct_change', 'student_cot_change'):
        item = comparisons[label]
        low, high = item['paired_bootstrap_ci95_pp']
        lines.append(f'| {item["after"]} − {item["before"]} | {item["delta_percentage_points"]:+.2f} | [{low:+.2f}, {high:+.2f}] | {item["improved_questions"]} / {item["regressed_questions"]} |')
    lines += ['', report['interpretation'], '',
              '本轮复用同一份纯数字训练数据和同一个 epoch 6 Student；direct 使用 512 tokens，CoT 使用 2048 tokens，均为 greedy decoding。', '']
    (run / 'results.md').write_text('\n'.join(lines))
    write_json(run / 'status.json', {'stage': 'complete', 'dataset': cfg['dataset_name'],
                                     'samples': expected})
    print((run / 'results.md').read_text())


if __name__ == '__main__':
    main()
