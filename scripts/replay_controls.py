import argparse
from pathlib import Path

from crest.artifacts import read_json, write_json
from crest.controls import export_native_blocks, plotting_values, replay_controls


def main():
    parser = argparse.ArgumentParser(description="Replay RQ3 calibration and selection from saved independent blocks.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path)
    source.add_argument("--units", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=3141592653)
    args = parser.parse_args()
    if (args.units is None) != (args.runs is None):
        parser.error("--units and --runs must be supplied together")
    traces = read_json(args.traces) if args.traces else export_native_blocks(args.units, args.runs)
    report = replay_controls(traces, bootstrap_replicates=args.bootstrap_replicates,
                             bootstrap_seed=args.bootstrap_seed)
    existing = args.output / "traces.json"
    if existing.exists() and read_json(existing).get("sha256") != traces["sha256"]:
        raise ValueError("Output directory already belongs to a different saved experiment")
    write_json(existing, traces)
    write_json(args.output / "controls.json", report)
    write_json(args.output / "plot_values.json", plotting_values(report))
    print(args.output / "controls.json")


if __name__ == "__main__":
    main()
