import argparse
import copy
import json
from pathlib import Path

import run_rq1 as cli
from crest.artifacts import digest, file_digest, read_json, write_json
from crest.rq1_execution import execute_cell, verified_fixed_calibration
from crest.rq1_protocol import load_pair_artifact
from crest.rq1_providers import build_rq1_providers
from crest.robustness import intensity_conditions, original_attack_settings, recurrence_conditions, timing_conditions
from crest.reporting import summarize_results


def once(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError('Existing inputs differ; use a new output directory')
    if not path.exists():
        write_json(path, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description='RQ2 with fixed primary calibration and paired test sequences.')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--primary-cell', required=True)
    parser.add_argument('--mode', choices=['intensity', 'recurrence', 'timing'], required=True)
    parser.add_argument('--source-descriptors')
    parser.add_argument('--source-attacks')
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget-seconds', type=float, default=86400, help='Maximum execution seconds per condition and invocation')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    plan = cli._read_plan(args.plan)
    config, runtime, units = cli._verified_inputs(plan)
    if len(units) != 1 or units[0]['dataset'] != 'amazon-2018-all-beauty' or units[0]['attack_family'] != 'instruction_injection':
        raise ValueError('Use one Beauty instruction-injection plan')
    unit = units[0]
    primary = Path(args.primary_cell)
    primary_manifest = read_json(primary / 'manifest.json')
    if digest(primary_manifest['immutable']) != primary_manifest['fingerprint']:
        raise ValueError('Primary cell manifest is corrupt')
    immutable = primary_manifest['immutable']
    if immutable['validated_plan_sha256'] != plan['dependency_plan']['plan_sha256'] or immutable['config'] != unit['config']:
        raise ValueError('The primary cell does not match this plan')
    for role in ('calibration', 'test'):
        if immutable[role] != [{'id': item['id'], 'sha256': digest(item)} for item in unit[role]]:
            raise ValueError('The primary cell calibration or test cohort differs from this plan')
    primary_report = read_json(primary / 'report.json')
    if primary_report['fingerprint'] != primary_manifest['fingerprint'] or primary_report['calibration']['status'] != 'ok':
        raise ValueError('A completed matching primary calibration is required')
    primary_calibration = verified_fixed_calibration(primary, unit['config'], unit['calibration'])['result']
    if primary_report['calibration'] != primary_calibration:
        raise ValueError('The primary report calibration differs from its verified stages')
    attack = config['frozen_settings']['attack:instruction_injection']
    if args.source_attacks:
        attack = read_json(args.source_attacks)['instruction_injection']
    attack = original_attack_settings(attack)
    source_rows = read_json(args.source_descriptors) if args.source_descriptors else []
    sources = {(row['seed_group'], row['sequence_seed']): row for row in source_rows
               if row['attack_family'] == 'instruction_injection' and row['split'] == 'test'}
    descriptors = {row['id']: row for row in plan['dependency_plan']['inputs']}
    output = Path(args.output).resolve()
    specification = {'mode': args.mode, 'plan_sha256': file_digest(args.plan),
                     'primary_manifest': file_digest(primary / 'manifest.json'),
                     'primary_report': file_digest(primary / 'report.json'), 'attack': attack,
                     'source_descriptors_sha256': file_digest(args.source_descriptors) if args.source_descriptors else None}
    spec_hash = digest(specification)
    once(output / 'specification.json', specification)
    conditions = {}
    for item in unit['test']:
        prior = load_pair_artifact(descriptors[item['id']])
        reference = prior['reference']
        if args.mode == 'intensity':
            source = sources.get((descriptors[item['id']]['seed_group'], descriptors[item['id']]['sequence_seed']))
            original = load_pair_artifact(source)['reference'] if source else None
            variants = intensity_conditions(reference, attack, spec_hash, source_reference=original, T=unit['config']['T'])
        elif args.mode == 'recurrence':
            variants = recurrence_conditions(reference, attack, spec_hash, T=unit['config']['T'])
        else:
            variants = timing_conditions(reference, attack, spec_hash)
        for variant in variants:
            pair = variant['bundle']['pair']
            if args.mode == 'intensity' and variant['level'] == 0.05 and pair['observed'] != item['pair']['observed']:
                raise ValueError('The default intensity differs from RQ1. Supply the original T100 descriptors and attack settings.')
            level = str(variant['level'])
            conditions.setdefault(level, []).append({'id': item['id'], 'seed_group': item['seed_group'], 'pair': pair})
    once(output / 'conditions.json', conditions)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'conditions': len(conditions), 'model_calls': 0}))
        return 0
    combined = copy.deepcopy(unit)
    combined['test'] = [dict(item, id=level + '/' + item['id']) for level, items in conditions.items() for item in items]
    bundle = cli._providers(build_rq1_providers, combined, config, runtime, output)
    summaries = {}
    try:
        for level, items in conditions.items():
            result = execute_cell(unit['config'], unit['calibration'], items, bundle.ranker, bundle.detectors,
                                  output / 'conditions' / level, validated_plan_sha256=digest([spec_hash, level]),
                                  budget_seconds=args.budget_seconds, fixed_calibration_source=primary)
            if 'rows' not in result:
                print(json.dumps({'status': 'partial', 'level': level}))
                return 2
            if result['calibration'] != primary_calibration:
                raise ValueError('Calibration changed across attack conditions')
            summaries[level] = summarize_results(result['rows'], k=unit['config']['K'])
            write_json(output / 'plot-values.json', {'mode': args.mode, 'eta': unit['config']['eta'], 'conditions': summaries})
    finally:
        bundle.close()
    print(json.dumps({'status': 'complete', 'conditions': len(summaries)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
