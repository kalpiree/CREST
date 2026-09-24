import argparse
import fcntl
import json
from pathlib import Path

import run_rq1
from crest.ablation import analyze_ablation
from crest.artifacts import write_json
from crest.reporting import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run component ablations from completed RQ1 checkpoints without new model calls.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unit-id")
    args = parser.parse_args(argv)
    artifact = run_rq1._read_plan(args.plan)
    _, _, units = run_rq1._verified_inputs(artifact)
    if args.unit_id is not None:
        units = [unit for unit in units if unit["id"] == args.unit_id]
    if len(units) != 1:
        raise ValueError("Select exactly one dataset--attack cell using --unit-id")
    unit = units[0]
    cell = args.run_dir.resolve() / "cells" / unit["id"]
    output = args.output.resolve()
    if output == cell or cell in output.parents:
        raise ValueError("Place ablation outputs outside the original cell")
    with (cell / ".worker.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        report = analyze_ablation(unit, cell, plan_sha256=artifact["dependency_plan"]["plan_sha256"])
    write_json(output / "report.json", report)
    write_report(report["summary"], output / "tables")
    print(json.dumps({"output": str(output), "new_model_calls": 0, "common_returned_sequences": report["summary"]["common_returned_sequences"]}))


if __name__ == "__main__":
    main()
