import argparse
import json
import math
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Create the final-stage timing table from measured summaries.')
    parser.add_argument('--beauty', type=Path, required=True)
    parser.add_argument('--steam', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    summaries = [json.loads(path.read_text()) for path in (args.beauty, args.steam)]
    if any(value['scope'] != 'final_stage_only' for value in summaries):
        raise ValueError('Both inputs must measure the same final-stage scope')
    if any(value['sequence_count'] != 1 for value in summaries):
        raise ValueError('This caption requires one fixed sequence per dataset')
    if summaries[0]['repetitions'] != summaries[1]['repetitions']:
        raise ValueError('Timing repetition counts differ')
    methods = list(summaries[0]['seconds_per_sequence'])
    if set(methods) != set(summaries[1]['seconds_per_sequence']):
        raise ValueError('Both timing summaries must include the same methods')
    caption = ('Final-stage recommendation time from prepared inputs. Values are mean seconds per sequence over '
               f"{summaries[0]['repetitions']} repetitions of one fixed sequence per dataset. "
               'Timing includes fresh ranking generation and CREST final selection; it excludes screening, detector execution, calibration, and model loading.')
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
             r'\begin{tabular}{lrr}', r'\toprule', r'Method & Beauty & Steam \\', r'\midrule']
    for method in methods:
        label = method.replace('PromptGuard2', 'Prompt Guard 2')
        if method == 'CREST':
            label = r'\textbf{CREST}'
        numbers = [value['seconds_per_sequence'][method]['total_seconds'] for value in summaries]
        if any(isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0 for number in numbers):
            raise ValueError('Timing means must be nonnegative numbers')
        lines.append(label + ' & ' + ' & '.join(f'{number:.3f}' for number in numbers) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
