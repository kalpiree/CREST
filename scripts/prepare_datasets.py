import argparse
import json
from pathlib import Path

from crest.artifacts import read_json
from crest.datasets import prepare_dataset, resolve_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["amazon-2018-all-beauty", "steam", "movielens-1m"])
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    catalog = read_json(args.config)
    overrides = dict(value.split("=", 1) for value in args.source)
    for dataset in args.datasets:
        sources = resolve_sources(catalog, dataset, args.raw_dir, overrides)
        definition = catalog["datasets"][dataset]
        for role, path in sources.items():
            expected = definition["files"][role].get("expected_bytes")
            if not path.is_file() or (expected is not None and path.stat().st_size != expected):
                raise ValueError(f"Missing or wrong-size complete raw archive for {dataset}:{role}: {path}")
        result = prepare_dataset(dataset, sources, Path(args.output_dir) / dataset, catalog["preprocessing"], definition)
        print(json.dumps({"dataset": dataset, "status": result["status"], "split_examples": result["split_examples"]}), flush=True)


if __name__ == "__main__":
    main()
