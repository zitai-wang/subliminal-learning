import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_selected_distillation_rejects_non_signal_rows(tmp_path, monkeypatch):
    # Exercise the selection predicate directly without model inference.
    from scripts import prepare_selected_distillation as module

    rows = [
        {'_id': 'train:0', '_source_index': 0, '_subject': 'a', '_answer': 'A'},
        {'_id': 'train:1', '_source_index': 1, '_subject': 'b', '_answer': 'B'},
        {'_id': 'train:2', '_source_index': 2, '_subject': 'a', '_answer': 'C'},
    ]
    candidate = tmp_path / 'candidate'
    (candidate / 'evaluation').mkdir(parents=True)
    (candidate / 'config.json').write_text(json.dumps({
        'dataset_format': 'scienceqa', 'evaluation': {'split': 'train'},
    }))
    (candidate / 'evaluation/baseline_direct.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'id': 'train:0', 'correct': False}, {'id': 'train:1', 'correct': True},
        {'id': 'train:2', 'correct': False},
    ]) + '\n')
    (candidate / 'evaluation/teacher_cot.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'id': 'train:0', 'correct': True, 'finished': True, 'boxed': True,
         'new_tokens': 10, 'response': 'work \\boxed{A}', 'prediction': 'A'},
        {'id': 'train:1', 'correct': True, 'finished': True, 'boxed': True,
         'new_tokens': 10, 'response': 'work \\boxed{B}', 'prediction': 'B'},
        {'id': 'train:2', 'correct': False, 'finished': True, 'boxed': True,
         'new_tokens': 10, 'response': 'work \\boxed{C}', 'prediction': 'C'},
    ]) + '\n')
    monkeypatch.setattr(module, 'load_selected_rows', lambda cfg: rows)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--candidate-run', str(candidate),
                                      '--output-dir', str(tmp_path / 'out')])
    module.main()
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    selected = [json.loads(line) for line in (tmp_path / 'out/train.jsonl').read_text().splitlines()]
    assert manifest['selected'] == 1
    assert selected[0]['id'] == 'train:0'
