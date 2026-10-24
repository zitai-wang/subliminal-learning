"""Run the selected-process pipeline once all primary work has been assigned.

The primary held-out evaluation has priority. Once it has no unassigned dataset,
this worker claims the first idle GPU even if an already assigned primary job is
still running. It produces paired train-split outputs and materializes the
auditable high-signal training set. It never reads a held-out test split for
selecting training examples.
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


def primary_work_assigned(run_root):
    status = Path(run_root) / 'queue_status.json'
    if not status.exists():
        return False
    state = json.loads(status.read_text())
    return state.get('stage') in {'running', 'complete'} and not state.get('pending')


def wait_for_gpu(gpus, poll_seconds, status_path, stage):
    while True:
        cards = gpu_snapshot()
        free = next((gpu for gpu in gpus if gpu in cards and available(cards[gpu])), None)
        write_json(status_path, {
            'stage': f'waiting_for_gpu_before_{stage}' if free is None else f'launching_{stage}',
            'updated_at': now(), 'gpus': cards,
        })
        if free is not None:
            return free
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-run-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'src/config/subliminal_scienceqa_selected_train.yaml')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--poll-seconds', type=float, default=0.25)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--eval-run-root', type=Path, required=True)
    args = parser.parse_args()
    assert args.poll_seconds > 0
    run_dir, config = args.run_dir.resolve(), args.config.resolve()
    gpus = [int(value) for value in args.gpus.split(',')]
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / 'queue_status.json'

    while not primary_work_assigned(args.primary_run_root):
        write_json(status_path, {'stage': 'waiting_for_primary_assignment', 'updated_at': now(),
                                 'primary_run_root': str(args.primary_run_root)})
        time.sleep(args.poll_seconds)
    code = 1
    for attempt, batch_size in enumerate((4, 2), start=1):
        free = wait_for_gpu(gpus, args.poll_seconds, status_path, 'candidate_evaluation')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(free), TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        evaluate_log = log_dir / f'candidate_evaluate_attempt{attempt}_gpu{free}.log'
        command = [PYTHON, '-u', str(ROOT / 'src/subliminal_mcq.py'), 'evaluate',
                   '--config', str(config), '--run-dir', str(run_dir), '--conditions',
                   'baseline_direct', 'teacher_cot', '--eval-batch-size', str(batch_size),
                   '--incorrect-ids-from', 'baseline_direct']
        write_json(status_path, {'stage': 'candidate_evaluation', 'updated_at': now(),
                                 'gpu': free, 'attempt': attempt, 'command': command})
        with evaluate_log.open('a') as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code == 0:
            break
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
    write_json(status_path, {'stage': 'data_ready', 'updated_at': now(),
                             'manifest': manifest})

    training_config = ROOT / 'src/config/selected_process_training.yaml'
    code = 1
    for attempt in (1, 2):
        free = wait_for_gpu(gpus, args.poll_seconds, status_path, 'student_training')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(free), TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        train_log = log_dir / f'train_attempt{attempt}_gpu{free}.log'
        command = [PYTHON, '-u', str(ROOT / 'src/train_selected_process_distillation.py'),
                   '--config', str(training_config), '--run-dir', str(run_dir)]
        write_json(status_path, {'stage': 'student_training', 'updated_at': now(),
                                 'gpu': free, 'attempt': attempt, 'command': command,
                                 'manifest': manifest})
        with train_log.open('a') as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code == 0:
            break
    if code:
        write_json(status_path, {'stage': 'failed', 'updated_at': now(), 'gpu': free,
                                 'step': 'student_training', 'exit_code': code})
        raise SystemExit(code)

    adapter = run_dir / 'checkpoints/epoch_06'
    assert (adapter / 'complete.json').exists()
    eval_root = args.eval_run_root.resolve()
    eval_log = log_dir / 'cross_dataset_evaluation_queue.log'
    command = [PYTHON, '-u', str(ROOT / 'scripts/run_subliminal_benchmark_queue.py'),
               '--run-root', str(eval_root), '--gpus', args.gpus,
               '--poll-seconds', str(args.poll_seconds),
               '--student-adapter-path', str(adapter), '--source-training-epoch', '6',
               '--source-run', str(run_dir)]
    write_json(status_path, {'stage': 'cross_dataset_evaluation', 'updated_at': now(),
                             'adapter': str(adapter), 'eval_run_root': str(eval_root),
                             'command': command})
    with eval_log.open('a') as log:
        code = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                              stderr=subprocess.STDOUT).returncode
    write_json(status_path, {'stage': 'complete' if code == 0 else 'failed',
                             'updated_at': now(), 'step': 'cross_dataset_evaluation',
                             'adapter': str(adapter), 'eval_run_root': str(eval_root),
                             'exit_code': code})
    raise SystemExit(code)


if __name__ == '__main__':
    main()
