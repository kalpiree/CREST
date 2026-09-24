import argparse
import json
import os
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'src'
if SOURCE.is_dir():
    sys.path.insert(0, str(SOURCE))

from crest.artifacts import digest, file_digest, read_json, write_json
from crest.rq1_rewriting_export import prepare_rewriting, run_rewriting


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare and resume the explicitly frozen multiple-record TextSimu variant')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for name in ('sequence-manifest', 'database', 'manifest', 'attack-settings', 'runtime-settings', 'output'):
        prepare.add_argument('--' + name, required=True)
    run = commands.add_parser('run')
    for name in ('prepared', 'prepared-sha256', 'model', 'cache', 'device'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--invocation-budget-seconds', required=True, type=float)
    run.add_argument('--queue-receipt')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare_rewriting(args.sequence_manifest, args.database, args.manifest, args.output, read_json(args.attack_settings), read_json(args.runtime_settings))
    else:
        if args.queue_receipt:
            job = os.environ.get('CREST_QUEUE_JOB_ID')
            fingerprint = os.environ.get('CREST_QUEUE_INPUT_FINGERPRINT')
            if not job or not fingerprint:
                raise ValueError('Queue receipt requires the immutable queue job and input fingerprint environment')
        result = run_rewriting(args.prepared, args.model, args.cache, args.device, args.invocation_budget_seconds, prepared_sha256=args.prepared_sha256)
        if args.queue_receipt:
            prepared = read_json(args.prepared)
            directory = Path(args.prepared).resolve().parent
            state = read_json(directory / 'status.json')
            complete = result['status'] == 'complete'
            files = [directory / 'receipt.json', directory / 'descriptors.json', directory / 'observed-sequences.json'] if complete else [directory / 'status.json']
            body = {'status': 'complete' if complete else 'partial', 'job_id': job, 'input_fingerprint': fingerprint,
                    'expected_items': len(prepared['planned_attempts']), 'completed_items': sum(row['status'] == 'complete' for row in state['attempts'].values()),
                    'artifacts': [{'path': str(path), 'sha256': file_digest(path)} for path in files]}
            write_json(args.queue_receipt, {**body, 'receipt_sha256': digest(body)})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 2 if result['status'] in ('partial', 'budget_exhausted') else 0


if __name__ == '__main__':
    raise SystemExit(main())
