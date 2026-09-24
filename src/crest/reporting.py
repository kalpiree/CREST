import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("recall", "ndcg", "gmax", "fpr")
PERCENT_METRICS = {"recall", "ndcg", "fpr"}
REMOVAL_METHODS = {"CREST", "Random-Matched", "PromptGuard2-Record", "RewriteDetection-Record", "RETURN-Del", "Screening-Only"}
NO_REMOVAL_METHODS = {"Backbone", "Spotlighting", "Without Screening", "Without-Screening"}
DISPLAY_NAMES = {"PromptGuard2-Record": "Prompt Guard 2--Record"}


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite fraction on [0, 1]")
    return float(value)


def _quantile(values, probability):
    index = (len(values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (index - lower) * (values[upper] - values[lower])


def exact_seed_interval(values):
    if not 2 <= len(values) <= 6:
        raise ValueError("Exact seed resampling requires two to six seed groups")
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Seed values must be finite")
    samples = sorted(_mean(sample) for sample in itertools.product(values, repeat=len(values)))
    return [_quantile(samples, 0.025), _quantile(samples, 0.975)]


def summarize_results(rows, methods=None, *, k=5, bootstrap=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty list of per-sequence results is required")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    methods = list(methods) if methods is not None else list(rows[0]["methods"])
    if not methods or len(set(methods)) != len(methods) or any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("Distinct method names are required")
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in identities:
            raise ValueError("Every sequence must have a unique nonempty ID")
        identities.add(row["id"])
        if "seed_group" not in row or any(method not in row.get("methods", {}) for method in methods):
            raise ValueError("Every sequence must identify its seed and every requested method")
        if "clean_reference_status" not in row:
            raise ValueError("Clean-reference status is required for the common comparison cohort")
        for method in methods:
            result = row["methods"][method]
            if not isinstance(result.get("status"), str):
                raise ValueError("Every attempted method must retain its execution status")
            if result["status"] == "ok" and "rankings" in result:
                if not result["rankings"] or any(len(ranking) != k or len(set(ranking)) != k for ranking in result["rankings"]):
                    raise ValueError("Returned rankings do not match the declared list size")
        groups[str(row["seed_group"])].append(row)
    seed_groups = {}
    for seed, subset in sorted(groups.items()):
        common = [row for row in subset if row["clean_reference_status"] == "ok" and all(row["methods"][method]["status"] == "ok" for method in methods)]
        values = {}
        for method in methods:
            metric_values = {metric: [] for metric in METRICS}
            returned = [row for row in subset if row["methods"][method]["status"] == "ok"]
            for row in common:
                metrics = row["methods"][method].get("metrics", {})
                for metric in ("recall", "ndcg", "gmax"):
                    metric_values[metric].append(_fraction(metrics.get(metric), metric))
            for row in returned:
                if method in NO_REMOVAL_METHODS:
                    continue
                result = row["methods"][method]
                record_metrics = result.get("record_metrics")
                if record_metrics is None:
                    if method in REMOVAL_METHODS:
                        raise ValueError(f"Missing per-sequence omission counts for {method}; recompute them from the recorded omission IDs")
                    continue
                fp, denominator = record_metrics.get("fp"), record_metrics.get("unmodified")
                if any(isinstance(value, bool) or not isinstance(value, int) for value in (fp, denominator)) or not 0 <= fp <= denominator:
                    raise ValueError("Invalid sequence-local false-positive counts")
                if denominator:
                    value = fp / denominator
                    if record_metrics.get("fpr") is not None and not math.isclose(_fraction(record_metrics["fpr"], "fpr"), value, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Stored FPR disagrees with sequence-local counts")
                    metric_values["fpr"].append(value)
            values[method] = {"returned": len(returned), "fpr_sequences": len(metric_values["fpr"]),
                              "metrics": {metric: _mean(items) for metric, items in metric_values.items()}}
        seed_groups[seed] = {"attempted": len(subset), "common_returned": len(common), "methods": values}
    result = {
        "schema_version": 1,
        "k": k,
        "attempted_sequences": len(rows),
        "common_returned_sequences": sum(group["common_returned"] for group in seed_groups.values()),
        "seed_count": len(seed_groups),
        "units": {"recall": "percent", "ndcg": "percent", "gmax": "fraction", "fpr": "percent"},
        "seed_group_units": "All per-seed metrics are fractions on [0,1].",
        "aggregation": "Sequence means within each seed, followed by an equally weighted mean of seed means. Utility and promotion share the all-method returned cohort with a valid clean reference. FPR averages sequence-local ratios on each method's returned sequences.",
        "uncertainty": "Exact seed-group percentile resampling; descriptive stability intervals conditional on fixed calibration. No significance test or significance stars." if bootstrap else "No uncertainty or significance claim.",
        "seed_groups": seed_groups,
        "methods": {},
    }
    for method in methods:
        entry = {"returned_sequences": sum(group["methods"][method]["returned"] for group in seed_groups.values()),
                 "metrics": {}, "metric_seed_counts": {}, "intervals_95": {}}
        entry["return_rate_percent"] = 100 * entry["returned_sequences"] / len(rows)
        for metric in METRICS:
            values = [group["methods"][method]["metrics"][metric] for group in seed_groups.values()
                      if group["methods"][method]["metrics"][metric] is not None]
            scale = 100 if metric in PERCENT_METRICS else 1
            entry["metrics"][metric] = scale * _mean(values) if values else None
            entry["metric_seed_counts"][metric] = len(values)
            entry["intervals_95"][metric] = [scale * bound for bound in exact_seed_interval(values)] if bootstrap and len(values) > 1 else None
        result["methods"][method] = entry
    return result


def _latex_text(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in value)


def _number(value, metric):
    return "--" if value is None else f"{value:.3f}" if metric in ("gmax", "fpr") else f"{value:.2f}"


def latex_table(summary):
    k = summary["k"]
    caption = f"Recommendation quality and promotion control. Recall@{k}, NDCG@{k}, and FPR are in percent; $G_{{\\max}}$ is on $[0,1]$. Values average sequence means within each seed and then across seeds."
    lines = [r"\begin{table}[t]", r"\centering", "\\caption{" + caption + "}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             f"Method & R@{k}$\\uparrow$ & N@{k}$\\uparrow$ & $G_{{\\max}}\\downarrow$ & FPR$\\downarrow$ & Returned " + r"\\", r"\midrule"]
    for method, entry in summary["methods"].items():
        label = _latex_text(DISPLAY_NAMES.get(method, method))
        if method == "CREST":
            label = r"\textbf{" + label + "}"
        fields = [_number(entry["metrics"][metric], metric) for metric in METRICS]
        fields.append(f"{entry['returned_sequences']}/{summary['attempted_sequences']}")
        lines.append(label + " & " + " & ".join(fields) + " " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_report(summary, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (directory / "table.tex").write_text(latex_table(summary))
    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", f"recall_at_{summary['k']}_percent", f"ndcg_at_{summary['k']}_percent", "gmax", "fpr_percent", "returned_sequences", "attempted_sequences"])
        for method, entry in summary["methods"].items():
            writer.writerow([method, *(entry["metrics"][metric] for metric in METRICS), entry["returned_sequences"], summary["attempted_sequences"]])


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate tables from per-sequence CREST evaluation outputs.")
    parser.add_argument("--input", required=True, type=Path, help="A per-sequence report.json or an RQ1 run directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args(argv)
    if args.input.is_dir():
        reports = [args.input / "report.json"] if (args.input / "report.json").is_file() else sorted((args.input / "cells").glob("*/report.json"))
        if not reports:
            raise ValueError("The run directory has no completed per-sequence report files")
    else:
        reports = [args.input]
    index = []
    for path in reports:
        source = json.loads(path.read_text())
        rows = source.get("rows") if isinstance(source, dict) else source
        summary = summarize_results(rows, args.methods, k=args.k, bootstrap=args.bootstrap)
        target = args.output_dir / path.parent.name if args.input.is_dir() and path.parent != args.input else args.output_dir
        write_report(summary, target)
        index.append({"source": str(path.resolve()), "output_dir": str(target.resolve()), "attempted_sequences": summary["attempted_sequences"], "common_returned_sequences": summary["common_returned_sequences"]})
    if args.input.is_dir():
        (args.output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"reports": index}))


if __name__ == "__main__":
    main()
