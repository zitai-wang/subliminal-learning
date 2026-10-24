"""Run the selected-process candidate pool after the primary benchmark queue.

The primary held-out evaluation has priority.  Once it reports completion this
worker claims the first idle GPU, produces paired train-split outputs, and
materializes the auditable high-signal training set.  It never reads a held-out
test split for selecting training examples.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('SUBLEARN_PYTHON', sys.executable)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def gpu_snapshot():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in output.strip().splitlines():
        index, uuid, used, utilization = [part.strip() for part in line.split(',')]
        cards[int(index)] = {'uuid': uuid, 'used_mib': int(used),
                             'utilization': int(utilization), 'pids': []}
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    by_uuid = {card['uuid']: card for card in cards.values()}
    for line in output.splitlines():
        if line.strip():
            uuid, pid = [part.strip() for part in line.split(',')]
            if uuid in by_uuid:
                by_uuid[uuid]['pids'].append(int(pid))
    return cards


def available(card):
    return not card['pids'] and card['used_mib'] < 2048 and card['utilization'] < 15


def primary_complete(run_root):
    status = Path(run_root) / 'queue_status.json'
    return status.exists() and json.loads(status.read_text()).get('stage') == 'complete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-run-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'src/config/subliminal_scienceqa_selected_train.yaml')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--poll-seconds', type=float, default=0.25)
    parser.add_argument('--max-samples', type=int, default=500)
    args = parser.parse_args()
    assert args.poll_seconds > 0
    run_dir, config = args.run_dir.resolve(), args.config.resolve()
    gpus = [int(value) for value in args.gpus.split(',')]
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / 'queue_status.json'

    while not primary_complete(args.primary_run_root):
        write_json(status_path, {'stage': 'waiting_for_primary_benchmark', 'updated_at': now(),
                                 'primary_run_root': str(args.primary_run_root)})
        time.sleep(args.poll_seconds)
    while True:
        cards = gpu_snapshot()
        free = next((gpu for gpu in gpus if gpu in cards and available(cards[gpu])), None)
        write_json(status_path, {'stage': 'waiting_for_gpu' if free is None else 'launching',
                                 'updated_at': now(), 'gpus': cards})
        if free is not None:
            break
        time.sleep(args.poll_seconds)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(free), TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    evaluate_log = log_dir / f'candidate_evaluate_gpu{free}.log'
    command = [PYTHON, '-u', str(ROOT / 'src/subliminal_mcq.py'), 'evaluate',
               '--config', str(config), '--run-dir', str(run_dir), '--conditions',
               'baseline_direct', 'teacher_cot', '--eval-batch-size', '4']
    write_json(status_path, {'stage': 'candidate_evaluation', 'updated_at': now(),
                             'gpu': free, 'command': command})
    with evaluate_log.open('a') as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    if code:
        write_json(status_path, {'stage': 'failed', 'updated_at': now(), 'gpu': free,
                                 'step': 'candidate_evaluation', 'exit_code': code})
        raise SystemExit(code)

    prepare_log = log_dir / 'prepare_selected_data.log'
    command = [PYTHON, '-u', str(ROOT / 'scripts/prepare_selected_distillation.py'),
               '--candidate-run', str(run_dir), '--output-dir', str(run_dir / 'data'),
               '--max-samples', str(args.max_samples)]
    with prepare_log.open('a') as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        write_json(status_path, {'stage': 'failed', 'updated_at': now(), 'gpu': free,
                                 'step': 'prepare_selected_data', 'exit_code': code})
        raise SystemExit(code)
    manifest = json.loads((run_dir / 'data/manifest.json').read_text())
    write_json(status_path, {'stage': 'data_ready', 'updated_at': now(), 'gpu': free,
                             'manifest': manifest})


if __name__ == '__main__':
    main()
