"""Build auditable, high-signal process-distillation data from candidate outputs.

The input run must contain paired ``baseline_direct`` and ``teacher_cot``
outputs on a labelled *training* split.  We retain only examples for which the
teacher is correct, direct answering is incorrect, and the teacher response is
complete and bounded in length.  The held-out evaluation splits are never read.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subliminal_mcq import load_selected_rows
from src.subliminal_route1 import read_rows, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--per-subject-cap', type=int, default=250)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.max_samples > 0 and args.per_subject_cap > 0 and args.max_new_tokens > 0

    candidate = args.candidate_run.resolve()
    output = args.output_dir.resolve()
    config_path = candidate / 'config.json'
    config = json.loads(config_path.read_text())
    assert config['evaluation']['split'] == 'train', 'Candidate data must come from a training split.'
    assert config['dataset_format'] == 'scienceqa', 'This first implementation targets ScienceQA.'
    rows = load_selected_rows(config)
    by_id = {row['_id']: row for row in rows}
    direct = {row['id']: row for row in read_rows(candidate / 'evaluation/baseline_direct.jsonl')}
    teacher = {row['id']: row for row in read_rows(candidate / 'evaluation/teacher_cot.jsonl')}
    assert set(direct) == set(teacher) == set(by_id), 'Candidate outputs must be complete and aligned.'

    rejected = Counter()
    eligible = []
    for item_id in sorted(by_id):
        base, cot = direct[item_id], teacher[item_id]
        if bool(base['correct']):
            rejected['direct_correct'] += 1
            continue
        if not bool(cot['correct']):
            rejected['teacher_incorrect'] += 1
            continue
        if not bool(cot.get('finished')):
            rejected['teacher_unfinished'] += 1
            continue
        if not bool(cot.get('boxed')):
            rejected['teacher_unboxed'] += 1
            continue
        if int(cot.get('new_tokens', 0)) > args.max_new_tokens:
            rejected['teacher_too_long'] += 1
            continue
        row = by_id[item_id]
        eligible.append({
            'id': item_id,
            'source_index': row['_source_index'],
            'subject': row['_subject'],
            'teacher_response': cot['response'],
            'teacher_prediction': cot['prediction'],
            'gold_answer': row['_answer'],
            'teacher_new_tokens': cot['new_tokens'],
        })

    # Deterministic round-robin subject balancing: no dominant subject can
    # consume the whole train set, while smaller subjects remain represented.
    buckets = defaultdict(list)
    for item in eligible:
        buckets[item['subject']].append(item)
    for subject in buckets:
        random.Random(f'{args.seed}:{subject}').shuffle(buckets[subject])
    selected, counts = [], Counter()
    subjects = sorted(buckets)
    while len(selected) < args.max_samples:
        added = False
        for subject in subjects:
            if len(selected) >= args.max_samples:
                break
            if buckets[subject] and counts[subject] < args.per_subject_cap:
                selected.append(buckets[subject].pop())
                counts[subject] += 1
                added = True
        if not added:
            break
    assert selected and len({item['id'] for item in selected}) == len(selected)

    output.mkdir(parents=True, exist_ok=True)
    train_path = output / 'train.jsonl'
    with train_path.open('w') as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    manifest = {
        'method': 'selected_process_distillation',
        'selection_rule': ('teacher_cot correct AND baseline_direct incorrect AND '
                           'teacher finished/boxed/<=max_new_tokens'),
        'candidate_run': str(candidate),
        'candidate_config_sha256': sha256(config_path),
        'candidate_split': config['evaluation']['split'],
        'heldout_eval_split_excluded': 'ScienceQA test',
        'candidate_total': len(rows),
        'eligible_before_subject_balancing': len(eligible),
        'selected': len(selected),
        'rejected': dict(sorted(rejected.items())),
        'subject_counts': dict(sorted(counts.items())),
        'max_new_tokens': args.max_new_tokens,
        'per_subject_cap': args.per_subject_cap,
        'seed': args.seed,
        'train_sha256': sha256(train_path),
    }
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
