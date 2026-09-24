import argparse
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'src'
if SOURCE.is_dir():
    sys.path.insert(0, str(SOURCE))

from crest.artifacts import read_json
from crest.rq1_sequence_export import export_rq1_sequences


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare immutable chronological RQ1 development pairs without model calls')
    parser.add_argument('--database', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sequence-settings', required=True)
    parser.add_argument('--attack-settings', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = export_rq1_sequences(args.database, args.manifest, args.output, read_json(args.sequence_settings), read_json(args.attack_settings),
                                  progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
