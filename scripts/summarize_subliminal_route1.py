"""Validate all seven experiment steps and report matched MathVista comparisons."""
import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONDITIONS = ('baseline_direct', 'baseline_cot', 'teacher_cot',
              'student_direct', 'student_cot')


def load_json(path):
    return json.loads(path.read_text())


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rescore_evaluations(run, cfg):
    """Apply the same corrected scorer to every saved response, without inference."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.subliminal_route1 import extract_answer, normalize_answer
    columns = ['pid', 'question_type', 'answer_type', 'answer', 'choices', 'precision']
    questions = {str(row['pid']): row for row in
                 pq.read_table(cfg['mathvista_parquet'], columns=columns).to_pylist()}
    assert len(questions) == 1000
    assert all(normalize_answer(str(row['answer']), row) is not None
               for row in questions.values()), 'Unparseable dataset answer'
    changes = []
    rescored_summaries = {}
    backup = run / 'provenance/evaluation_before_scorer_v2'
    backup.mkdir(parents=True, exist_ok=True)
    for name in CONDITIONS:
        path = run / f'evaluation/{name}.jsonl'
        rows = load_jsonl(path)
        assert len(rows) == len({r['pid'] for r in rows}) == 1000
        assert {r['pid'] for r in rows} == set(questions)
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        for row in rows:
            question = questions[row['pid']]
            extraction = extract_answer(row['response'])
            prediction = normalize_answer(extraction, question)
            answer = normalize_answer(str(question['answer']), question)
            correct = prediction is not None and prediction == answer
            if (row['prediction'], row['answer'], row['correct']) != (prediction, answer, correct):
                changes.append({'condition': name, 'pid': row['pid'],
                                'before': {k: row[k] for k in ('prediction', 'answer', 'correct')},
                                'after': {'prediction': prediction, 'answer': answer, 'correct': correct}})
            row.update(extraction=extraction, prediction=prediction, answer=answer, correct=correct)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        temporary.replace(path)
        summary_path = run / f'evaluation/{name}_summary.json'
        summary = load_json(summary_path)
        summary.update(accuracy=sum(r['correct'] for r in rows) / len(rows),
                       parser='local deterministic extraction, scorer v2; list commas preserved')
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        rescored_summaries[name] = summary
    (run / 'summary.json').write_text(
        json.dumps(rescored_summaries, ensure_ascii=False, indent=2) + '\n')
    (run / 'provenance/scorer_v2_changes.json').write_text(
        json.dumps({'reason': 'Preserve list-answer commas; empty predictions remain unparsed.',
                    'changes': changes}, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = load_json(run / 'config.json')
    rescore_evaluations(run, cfg)
    generation = load_json(run / 'data/summary.json')
    training = load_json(run / f'checkpoints/epoch_{cfg["evaluation"]["primary_checkpoint_epoch"]:02d}/complete.json')
    assert training['epoch'] == cfg['training']['epochs'] == cfg['evaluation']['primary_checkpoint_epoch']
    effective_batch = cfg['training']['micro_batch_size'] * cfg['training']['gradient_accumulation_steps']
    expected_updates = math.ceil(cfg['generation']['train_samples'] / effective_batch) * cfg['training']['epochs']
    assert training['global_step'] == expected_updates, 'Incomplete optimizer updates'
    train_path = run / 'data/train.jsonl'
    train = load_jsonl(train_path)
    raw = load_jsonl(run / 'data/teacher_raw.jsonl')
    assert len(train) == cfg['generation']['train_samples'] == generation['train_samples']
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == generation['sha256']
    assert len(raw) == generation['attempts'] >= cfg['generation']['attempts']
    assert len({row['id'] for row in raw}) == len(raw)
    accepted = {row['id']: row for row in raw if row['accepted'] and row['finished']}
    assert len(accepted) == generation['accepted']
    assert len({row['id'] for row in train}) == len(train)
    assert all(set(row) == {'id', 'user', 'assistant'} and row['id'] in accepted
               and row['user'] == accepted[row['id']]['user']
               and row['assistant'] == accepted[row['id']]['assistant'] for row in train)
    evaluations, metrics = {}, {}
    common_ids = None
    for condition in CONDITIONS:
        rows = load_jsonl(run / f'evaluation/{condition}.jsonl')
        assert len(rows) == len({row['pid'] for row in rows}) == 1000, condition
        mapping = {row['pid']: row for row in rows}
        assert common_ids is None or set(mapping) == common_ids, condition
        common_ids = set(mapping)
        evaluations[condition] = mapping
        correct = sum(row['correct'] for row in rows)
        summary = load_json(run / f'evaluation/{condition}_summary.json')
        assert summary['samples'] == 1000 and abs(summary['accuracy'] - correct / 1000) < 1e-12
        metrics[condition] = {'samples': 1000, 'correct': correct,
                              'accuracy': correct / 1000,
                              'unparsed': sum(row['prediction'] is None for row in rows),
                              'unfinished': sum(not row.get('finished', True) for row in rows),
                              'mean_new_tokens': float(np.mean([r['new_tokens'] for r in rows]))}
    comparisons = {}
    for label, before, after in (
        ('teacher_prompt_check', 'baseline_cot', 'teacher_cot'),
        ('student_direct_change', 'baseline_direct', 'student_direct'),
        ('student_cot_change', 'baseline_cot', 'student_cot'),
    ):
        diffs = np.array([int(evaluations[after][pid]['correct'])
                          - int(evaluations[before][pid]['correct']) for pid in sorted(common_ids)])
        counts = np.array([(diffs == value).sum() for value in (-1, 0, 1)])
        draws = np.random.default_rng(cfg['seed']).multinomial(len(diffs), counts / len(diffs), size=20000)
        bootstrap = (draws[:, 2] - draws[:, 0]) / len(diffs) * 100
        comparisons[label] = {'before': before, 'after': after,
                              'delta_percentage_points': float(diffs.mean() * 100),
                              'paired_bootstrap_ci95_pp': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                              'improved_questions': int(counts[2]),
                              'regressed_questions': int(counts[0])}
    teacher_gain = comparisons['teacher_prompt_check']['delta_percentage_points']
    interpretation = (
        'Teacher 的 CoT 准确率高于同提示预算下的 baseline CoT；Student 的变化见配对比较。'
        if teacher_gain > 0 else
        'Teacher 未建立相对 baseline CoT 的准确率优势，因此本轮不能检验已验证的 Teacher 优势是否传递。'
    )
    report = {'run_dir': str(run), 'completed_at': datetime.now(timezone.utc).isoformat(),
              'steps_completed': list(range(1, 8)), 'generation': generation,
              'training': training, 'metrics': metrics, 'comparisons': comparisons,
              'interpretation': interpretation,
              'limits': ['Single training seed; no neutral-teacher numeric-data control.',
                         'Custom deterministic local answer extraction; no external judge.',
                         'Question bootstrap intervals describe this evaluation set, not variation across training runs.',
                         'Accuracy changes alone do not establish causal subliminal transfer.']}
    (run / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
             '- Origin Mode: run', '- Verification Status: UNVERIFIED (no independent rerun)',
             '- Version Label: route1_result_v1', '', '# Subliminal Learning 路线一实验结果', '',
             f'7 个步骤已完成；以下均为完整 MathVista testmini 1,000 题，主检查点固定 epoch {cfg["evaluation"]["primary_checkpoint_epoch"]}。', '',
             '| 条件 | 正确数 / 1000 | 准确率 | 未解析 | 生成未结束 |',
             '|---|---:|---:|---:|---:|']
    for name, value in metrics.items():
        lines.append(f'| {name} | {value["correct"]} | {value["accuracy"]:.2%} | {value["unparsed"]} | {value["unfinished"]} |')
    lines += ['', '| 配对比较 | 变化（百分点） | 问题 bootstrap 95% 区间 | 改对 / 改错 |',
              '|---|---:|---:|---:|']
    for value in comparisons.values():
        lo, hi = value['paired_bootstrap_ci95_pp']
        lines.append(f'| {value["after"]} − {value["before"]} | {value["delta_percentage_points"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {value["improved_questions"]} / {value["regressed_questions"]} |')
    lines += ['', interpretation, '',
              f'数字生成 {generation["attempts"]:,} 条，通过过滤 {generation["accepted"]:,} 条，训练采用 {len(train):,} 条。',
              f'Student 完成 epoch {training["epoch"]}，共 {training["global_step"]} 次优化器更新。', '',
              '这是单 seed 的流程验证，没有中性 Teacher 数字训练对照；分数变化不足以单独证明因果迁移。',
              '所有条件使用相同的本地确定性答案提取与归一化；逐题原始回答保存在 evaluation/。',
              '区间仅反映本评测集的问题抽样差异，不能表示训练 seed 之间的变化，也未作多重比较校正。', '']
    (run / 'results.md').write_text('\n'.join(lines))
    (run / 'status.json').write_text(json.dumps({'stage': 'complete', 'steps_completed': list(range(1, 8))}, indent=2) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
