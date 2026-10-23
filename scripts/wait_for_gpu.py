"""Wait for the requested physical GPUs without disturbing existing jobs."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def gpu_snapshot():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in output.strip().splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
        cards[int(index)] = {'index': int(index), 'uuid': uuid, 'used_mib': int(memory),
                             'utilization': int(utilization), 'compute_pids': []}
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    by_uuid = {card['uuid']: card for card in cards.values()}
    for line in processes.strip().splitlines():
        uuid, pid = [part.strip() for part in line.split(',')]
        by_uuid[uuid]['compute_pids'].append(int(pid))
    return cards


def available(card):
    return not card['compute_pids'] and card['used_mib'] < 2048 and card['utilization'] < 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--status-name', default='status.json')
    args = parser.parse_args()
    assert Path(args.status_name).name == args.status_name
    devices = [int(value) for value in args.devices.split(',')]
    while True:
        cards = gpu_snapshot()
        selected = [cards[index] for index in devices]
        if all(available(card) for card in selected):
            print(f'Physical GPUs {devices} are free; starting the pipeline.', flush=True)
            return
        status = {'stage': 'waiting_for_gpu', 'requested_devices': devices,
                  'checked_at': datetime.now(timezone.utc).isoformat(), 'gpus': selected}
        path = args.run_dir / args.status_name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
        print(json.dumps(status), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
