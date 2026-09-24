import argparse
import json
import statistics
import time
from pathlib import Path

import run_rq1 as cli
from crest.ablation import _stage
from crest.artifacts import digest, read_json, write_json
from crest.rq1_execution import observed_view
from crest.rq1_providers import build_rq1_providers
from crest.selection import select


def synchronize(device):
    if device.startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Time final ranking generation and CREST selection from prepared inputs.')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test-index', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.test_index < 0:
        parser.error('Repetitions must be positive and test-index nonnegative')
    plan = cli._read_plan(args.plan)
    config, runtime, units = cli._verified_inputs(plan)
    if len(units) != 1:
        raise ValueError('Supply one dataset/attack plan')
    unit = units[0]
    if args.test_index >= len(unit['test']):
        raise ValueError('Test index is outside the fixed cohort')
    item = unit['test'][args.test_index]
    cell = Path(args.run_dir) / 'cells' / unit['id']
    manifest = read_json(cell / 'manifest.json')
    immutable = manifest['immutable']
    if digest(immutable) != manifest['fingerprint'] or immutable['config'] != unit['config'] or immutable['validated_plan_sha256'] != plan['dependency_plan']['plan_sha256']:
        raise ValueError('The saved cell differs from this plan')
    if immutable['test'] != [{'id': x['id'], 'sha256': digest(x)} for x in unit['test']]:
        raise ValueError('The test cohort differs from the saved cell')
    calibration = _stage(cell, ['calibration'], manifest['fingerprint'])
    if calibration['status'] != 'ok':
        raise ValueError('Valid saved calibration is required')
    methods = unit['config']['methods']
    prepared = {}
    for method in methods:
        name = ['method', method, item['id']] if method in ('CREST', 'Random-Matched') else ['baseline', method, item['id']]
        result = _stage(cell, name, manifest['fingerprint'])
        if result['status'] != 'ok':
            raise ValueError('Every method needs valid prepared inputs for timing')
        prepared[method] = result['omitted_ids']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    values = []
    decisions = observed_view(item['pair']['observed'])
    for repetition in range(args.repetitions):
        order = methods[repetition % len(methods):] + methods[:repetition % len(methods)]
        for method in order:
            directory = output / 'cache' / str(repetition) / method
            bundle = cli._providers(build_rq1_providers, unit, config, runtime, directory)
            try:
                bundle.ranker.load()
                synchronize(runtime['device'])
                start = time.perf_counter()
                rankings = [bundle.ranker.rank(decision, unit['config']['K'], frozenset(prepared[method]), method == 'Spotlighting') for decision in decisions]
                synchronize(runtime['device'])
                generation = time.perf_counter() - start
                selection = 0.0
                if method == 'CREST':
                    start = time.perf_counter()
                    selected = select(rankings, [decision['candidates'] for decision in decisions],
                                      unit['config']['K'], unit['config']['eta'], calibration['h_alpha'])
                    selection = time.perf_counter() - start
                    if selected['status'] != 'ok':
                        raise ValueError('Timed final selection returned no exact-K sequence')
                values.append({'method': method, 'repetition': repetition, 'generation_seconds': generation,
                               'selection_seconds': selection, 'total_seconds': generation + selection,
                               'inference': bundle.ranker.statistics()})
                write_json(output / 'timings.json', {'scope': 'Final-stage generation from prepared inputs and CREST final selection; excludes screening, detector execution, calibration and model loading.',
                    'cache_policy': 'Initially empty, separate exact-query cache per method and repetition; within-sequence reuse allowed.',
                    'sequence_id': item['id'], 'decisions': len(decisions), 'plan_sha256': plan['sha256'],
                    'model_runtime': bundle.ranker.metadata, 'rows': values})
            finally:
                bundle.close()
    summary = {method: {metric: statistics.mean(row[metric] for row in values if row['method'] == method)
                        for metric in ('generation_seconds', 'selection_seconds', 'total_seconds')} for method in methods}
    write_json(output / 'summary.json', {'scope': 'final_stage_only', 'repetitions': args.repetitions,
                                       'sequence_count': 1, 'seconds_per_sequence': summary})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
