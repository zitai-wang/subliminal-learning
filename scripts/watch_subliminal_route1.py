"""Monitor a run and recover known CUDA OOM failures using recorded amendments."""
import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from wait_for_gpu import available, gpu_snapshot


def write_json(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def gpu_state():
    return list(gpu_snapshot().values())


def run_processes(run):
    pids = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        if any(Path(arg).name in {'subliminal_route1.py', 'wait_for_gpu.py'}
               for arg in argv if arg) and str(run) in argv:
            pids.append(int(path.name))
    return pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    project = Path(__file__).resolve().parents[1]
    config_path = project / 'src/config/subliminal_route1.yaml'
    events_path = run / 'provenance/oom_recovery_events.jsonl'
    attempts = len(events_path.read_text().splitlines()) if events_path.exists() else 0
    batch_size = 4
    no_process_checks = 0
    while True:
        status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
        exit_path = run / 'pipeline_exit.json'
        now = datetime.now(timezone.utc).isoformat()
        monitor = {'checked_at': now, 'pipeline_status': status, 'oom_recoveries': attempts}
        monitor['process_pids'] = run_processes(run)
        no_process_checks = 0 if monitor['process_pids'] else no_process_checks + 1
        monitor['seconds_since_status_update'] = (
            time.time() - (run / 'status.json').stat().st_mtime
            if (run / 'status.json').exists() else None)
        training_status = run / 'training_status.json'
        if training_status.exists():
            monitor['parallel_training'] = json.loads(training_status.read_text())
            monitor['seconds_since_status_update'] = min(
                monitor['seconds_since_status_update'] or float('inf'),
                time.time() - training_status.stat().st_mtime)
        training_exit = run / 'training_worker_exit.json'
        if training_exit.exists():
            monitor['training_worker_exit'] = json.loads(training_exit.read_text())
        if exit_path.exists():
            exit_info = json.loads(exit_path.read_text())
            monitor['pipeline_exit'] = exit_info
            if exit_info['exit_code'] == 0 and status.get('stage') == 'complete':
                assert (run / 'results.md').exists() and (run / 'results.json').exists()
                monitor['state'] = 'complete'
                write_json(run / 'supervisor_status.json', monitor)
                print(json.dumps(monitor), flush=True)
                return
            logs = sorted((run / 'logs').glob('step*.log'), key=lambda p: p.stat().st_mtime)
            last_log = logs[-1] if logs else None
            tail = last_log.read_text(errors='replace')[-16000:] if last_log else ''
            oom = 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            if not oom or attempts >= 4:
                monitor.update(state='needs_code_diagnosis', last_log=str(last_log))
            else:
                cards = gpu_state()
                idle = sorted([card for card in cards if available(card)],
                              key=lambda card: (card['index'] != 2, card['index']))
                if not idle:
                    monitor.update(state='waiting_for_idle_gpu', gpus=cards)
                else:
                    cfg = yaml.safe_load(config_path.read_text())
                    assert cfg == json.loads((run / 'config.json').read_text())
                    before = json.loads(json.dumps(cfg))
                    label = last_log.stem
                    devices = str(idle[0]['index'])
                    parallel = False
                    if 'train' in label and len(idle) > 1:
                        devices = ','.join(str(card['index']) for card in idle[:2])
                        parallel = True
                    elif 'train' in label and cfg['training']['micro_batch_size'] > 1:
                        old_micro = cfg['training']['micro_batch_size']
                        cfg['training']['micro_batch_size'] = old_micro // 2
                        cfg['training']['gradient_accumulation_steps'] *= 2
                    elif 'generate' in label and cfg['generation']['batch_size'] > 1:
                        cfg['generation']['batch_size'] //= 2
                    elif 'baseline' in label or 'teacher' in label or 'student' in label:
                        batch_size = max(1, batch_size // 2)
                    else:
                        monitor.update(state='needs_code_diagnosis', last_log=str(last_log))
                        write_json(run / 'supervisor_status.json', monitor)
                        time.sleep(args.poll_seconds)
                        continue
                    assert cfg['training']['micro_batch_size'] * cfg['training']['gradient_accumulation_steps'] == 64
                    assert cfg['training']['epochs'] == cfg['evaluation']['primary_checkpoint_epoch'] == 6
                    attempts += 1
                    event = {'attempt': attempts, 'time': now, 'failed_log': str(last_log),
                             'gpus': cards, 'selected_devices': devices, 'model_parallel': parallel,
                             'before': before, 'after': cfg, 'evaluation_batch_size': batch_size}
                    with events_path.open('a') as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + '\n')
                    if cfg != before:
                        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
                        write_json(run / 'config.json', cfg)
                    exit_path.replace(run / f'provenance/pipeline_exit_before_oom_recovery_{attempts}.json')
                    session = f'sublearn_oom_recovery_{attempts}_{int(time.time())}'
                    command = ['env', f'SUBLEARN_GPU={devices}',
                               f'SUBLEARN_MODEL_PARALLEL={int(parallel)}',
                               f'SUBLEARN_EVAL_BATCH_SIZE={batch_size}',
                               f'SUBLEARN_PYTHON={os.sys.executable}',
                               'bash', str(project / 'scripts/run_subliminal_route1_cuda2.sh'), str(run)]
                    subprocess.run(['tmux', 'new-session', '-d', '-s', session,
                                    '-c', str(project), shlex.join(command)], check=True)
                    latest = project / 'runs/latest_route1.json'
                    info = json.loads(latest.read_text()) if latest.exists() else {}
                    info.update(session=session, run_dir=str(run), physical_cuda=devices)
                    write_json(latest, info)
                    monitor.update(state='restarted_after_oom', session=session, recovery=event)
        else:
            monitor['state'] = ('waiting_for_gpu' if status.get('stage') == 'waiting_for_gpu'
                                else 'running')
            if no_process_checks >= 3 or (monitor['seconds_since_status_update'] or 0) > 900:
                monitor['state'] = 'needs_process_diagnosis'
            if monitor.get('training_worker_exit', {}).get('exit_code', 0) != 0:
                offset = monitor['training_worker_exit'].get('log_start_bytes', 0)
                training_log = run / 'logs/step5_train.log'
                tail = (training_log.read_bytes()[offset:][-16000:].decode(errors='replace')
                        if training_log.exists() else '')
                oom = 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail
                if not oom or attempts >= 4:
                    monitor['state'] = 'needs_training_diagnosis'
                else:
                    cards = gpu_state()
                    idle = sorted([card for card in cards if available(card)],
                                  key=lambda card: (card['index'] != 1, card['index']))
                    cfg = yaml.safe_load(config_path.read_text())
                    assert cfg == json.loads((run / 'config.json').read_text())
                    before = json.loads(json.dumps(cfg))
                    devices = None
                    parallel = False
                    if idle and cfg['training']['micro_batch_size'] > 1:
                        devices = str(idle[0]['index'])
                        cfg['training']['micro_batch_size'] //= 2
                        cfg['training']['gradient_accumulation_steps'] *= 2
                    elif len(idle) > 1:
                        devices = ','.join(str(card['index']) for card in idle[:2])
                        parallel = True
                    if devices is None:
                        monitor.update(state='waiting_for_idle_gpu_after_training_oom', gpus=cards)
                    else:
                        assert cfg['training']['micro_batch_size'] * cfg['training']['gradient_accumulation_steps'] == 64
                        assert cfg['training']['epochs'] == cfg['evaluation']['primary_checkpoint_epoch'] == 6
                        attempts += 1
                        event = {'attempt': attempts, 'time': now, 'failed_log':
                                 str(run / 'logs/step5_train.log'), 'worker': True,
                                 'gpus': cards, 'selected_devices': devices,
                                 'model_parallel': parallel, 'before': before, 'after': cfg}
                        with events_path.open('a') as handle:
                            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
                        if cfg != before:
                            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
                            write_json(run / 'config.json', cfg)
                        training_exit.replace(
                            run / f'provenance/training_worker_exit_before_oom_recovery_{attempts}.json')
                        session = f'sublearn_training_oom_recovery_{attempts}_{int(time.time())}'
                        command = ['env', f'SUBLEARN_GPU={devices}',
                                   f'SUBLEARN_MODEL_PARALLEL={int(parallel)}',
                                   f'SUBLEARN_PYTHON={os.sys.executable}',
                                   'bash', str(project / 'scripts/run_subliminal_training_worker.sh'), str(run)]
                        subprocess.run(['tmux', 'new-session', '-d', '-s', session,
                                        '-c', str(project), shlex.join(command)], check=True)
                        latest = project / 'runs/latest_route1.json'
                        info = json.loads(latest.read_text()) if latest.exists() else {}
                        info.update(training_session=session, training_cuda=devices)
                        write_json(latest, info)
                        monitor.update(state='restarted_training_after_oom', session=session,
                                       recovery=event)
        write_json(run / 'supervisor_status.json', monitor)
        print(json.dumps(monitor, ensure_ascii=False), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
