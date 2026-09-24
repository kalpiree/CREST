import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crest.rq1_assets import export_frequency_cohort, export_return_graph


def parser():
    result = argparse.ArgumentParser(description="Export verified real train/development RQ1 baseline assets")
    commands = result.add_subparsers(dest="kind", required=True)
    for kind in ("graph", "cohort"):
        command = commands.add_parser(kind)
        for name in ("database", "manifest", "observed-path", "observed-sha256", "output"):
            command.add_argument("--" + name, required=True)
        command.add_argument("--source-split", choices=("train", "development"), required=True)
        if kind == "graph":
            command.add_argument("--history-scope", choices=("split_only", "prefix_through_split"), required=True)
            command.add_argument("--max-hops", type=int, required=True)
            command.add_argument("--history-cap", type=int, required=True, help="Latest events per user; 0 retains all source history")
            command.add_argument("--unknown-item-policy", choices=("error", "zero_support"), required=True)
        else:
            for name in ("cohort-size", "candidate-count", "recommendation-k", "history-count", "selection-seed", "text-chars"):
                command.add_argument("--" + name, type=int, required=True)
            command.add_argument("--candidate-policy", choices=("cover_observed_identities",), required=True)
    return result


def main(argv=None):
    args = vars(parser().parse_args(argv))
    kind = args.pop("kind")
    shared = [args.pop(name) for name in ("database", "manifest", "observed_path", "observed_sha256", "output")]
    function = export_return_graph if kind == "graph" else export_frequency_cohort
    result = function(*shared, **args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
