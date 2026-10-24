"""Run six full labeled-split cross-dataset evaluations as GPUs become free."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('SUBLEARN_PYTHON', '/home/wzy/anaconda3/envs/licv/bin/python')
DATASETS = ('scienceqa', 'mmmu', 'chartqa', 'mmbench', 'mathverse', 'mmstar')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def gpu_snapshot():
    query = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in query.strip().splitlines():
        index, uuid, used, utilization = [part.strip() for part in line.split(',')]
        cards[int(index)] = {'uuid': uuid, 'used_mib': int(used),
                             'utilization': int(utilization), 'pids': []}
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    by_uuid = {card['uuid']: card for card in cards.values()}
    for line in processes.splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(',')]
        if uuid in by_uuid:
            by_uuid[uuid]['pids'].append(int(pid))
    return cards


def available(card):
    return not card['pids'] and card['used_mib'] < 2048 and card['utilization'] < 15


def completed(run_dir):
    status = run_dir / 'status.json'
    return status.exists() and json.loads(status.read_text()).get('stage') == 'complete'


def launch(dataset, gpu, run_root, attempt):
    run_dir = run_root / dataset
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    config = ROOT / 'src/config' / f'subliminal_{dataset}.yaml'
    batch_size = 4 if attempt == 1 else 2
    log_path = run_dir / 'logs' / f'evaluate_attempt{attempt}_gpu{gpu}.log'
    log = log_path.open('a')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [PYTHON, '-u', str(ROOT / 'src/subliminal_mcq.py'), 'evaluate',
               '--config', str(config), '--run-dir', str(run_dir),
               '--eval-batch-size', str(batch_size)]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return {'dataset': dataset, 'gpu': gpu, 'attempt': attempt, 'pid': process.pid,
            'process': process, 'log': log, 'log_path': str(log_path), 'started_at': now()}


def summarize(dataset, run_root):
    run_dir = run_root / dataset
    log_path = run_dir / 'logs/summarize.log'
    with log_path.open('a') as log:
        return subprocess.run([
            PYTHON, '-u', str(ROOT / 'scripts/summarize_subliminal_mcq.py'),
            '--run-dir', str(run_dir)], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT).returncode


def public_state(pending, active, done, failures, cards):
    return {
        'stage': 'complete' if not pending and not active else 'running',
        'updated_at': now(), 'pending': pending, 'completed': done,
        'failures': failures,
        'active': [{key: value for key, value in worker.items()
                    if key not in {'process', 'log'}} for worker in active.values()],
        'gpus': cards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--poll-seconds', type=float, default=0.25)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    gpus = [int(value) for value in args.gpus.split(',')]
    done = [dataset for dataset in DATASETS if completed(run_root / dataset)]
    pending = [dataset for dataset in DATASETS if dataset not in done]
    attempts = {dataset: 0 for dataset in DATASETS}
    failures, active, free_streak = [], {}, {gpu: 0 for gpu in gpus}
    exit_code = 0
    try:
        while pending or active:
            cards = gpu_snapshot()
            for gpu in gpus:
                free_streak[gpu] = free_streak[gpu] + 1 if available(cards[gpu]) else 0

            for gpu, worker in list(active.items()):
                code = worker['process'].poll()
                if code is None:
                    continue
                worker['log'].close()
                del active[gpu]
                if code == 0 and summarize(worker['dataset'], run_root) == 0:
                    done.append(worker['dataset'])
                    print(f"{now()} completed {worker['dataset']} on GPU {gpu}", flush=True)
                elif worker['attempt'] < 2:
                    pending.insert(0, worker['dataset'])
                    failures.append({'dataset': worker['dataset'], 'gpu': gpu,
                                     'attempt': worker['attempt'], 'exit_code': code,
                                     'retrying': True, 'at': now()})
                    print(f"{now()} retrying {worker['dataset']} with batch size 2", flush=True)
                else:
                    failures.append({'dataset': worker['dataset'], 'gpu': gpu,
                                     'attempt': worker['attempt'], 'exit_code': code,
                                     'retrying': False, 'at': now()})
                    exit_code = 1

            for gpu in gpus:
                if not pending or gpu in active or free_streak[gpu] < 1:
                    continue
                dataset = pending.pop(0)
                attempts[dataset] += 1
                worker = launch(dataset, gpu, run_root, attempts[dataset])
                active[gpu] = worker
                free_streak[gpu] = 0
                print(f"{now()} started {dataset} on GPU {gpu} pid={worker['pid']}", flush=True)

            write_json(run_root / 'queue_status.json',
                       public_state(pending, active, done, failures, cards))
            if pending or active:
                time.sleep(args.poll_seconds)
    finally:
        cards = gpu_snapshot()
        write_json(run_root / 'queue_status.json',
                   public_state(pending, active, done, failures, cards))
        write_json(run_root / 'queue_exit.json', {'exit_code': exit_code,
                                                  'finished_at': now()})
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
