import argparse
import json
from pathlib import Path

from crest.artifacts import read_json, write_json
from crest.datasets import download_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["amazon-2018-all-beauty", "steam", "movielens-1m"])
    args = parser.parse_args()
    catalog = read_json(args.config)
    for dataset in args.datasets:
        definition = catalog["datasets"][dataset]
        root = Path(args.raw_dir) / definition["raw_subdir"]
        manifest = {"dataset": dataset, "edition": definition["edition"], "source_page": definition["source_page"], "usage": definition["usage"], "files": {}}
        for role, spec in definition["files"].items():
            result = download_file(spec, root / spec["filename"])
            manifest["files"][role] = result
            write_json(root / "download_manifest.json", manifest)
            print(json.dumps({"dataset": dataset, "role": role, **result}), flush=True)


if __name__ == "__main__":
    main()
