import math
import random
from pathlib import Path

from .artifacts import digest, file_digest, read_json
from .calibration import calibrate, promotion_budget
from .metrics import _validate_rankings, gmax, utility
from .selection import select


ALPHAS = (0.01, 0.025, 0.05, 0.10, 0.20)
ETAS = (0.02, 0.05, 0.10, 0.15, 0.20)
PAPER_PROTOCOL = {"blocks": 20, "N": 19, "tests_per_block": 5, "T": 20, "C": 50, "K": 5}


def seal(body):
    body = {key: value for key, value in body.items() if key != "sha256"}
    return {**body, "sha256": digest(body)}


def _verify_seal(value, label):
    if not isinstance(value, dict) or value.get("sha256") != seal(value)["sha256"]:
        raise ValueError(f"Checksum mismatch: {label}")


def _native_stage(directory, name, fingerprint):
    path = Path(directory) / "stages" / (digest(name) + ".json")
    stage = read_json(path)
    if (
        stage.get("name") != name
        or stage.get("fingerprint") != fingerprint
        or stage.get("result_sha256") != digest(stage.get("result"))
        or stage.get("status") != "complete"
        or stage.get("result", {}).get("status") != "ok"
    ):
        raise ValueError(f"Incomplete, changed or invalid stage: {path}")
    return stage["result"], {"name": name, "sha256": file_digest(path)}


def export_native_blocks(units, runs, expected_protocol=PAPER_PROTOCOL):
    units, runs = Path(units), Path(runs)
    blocks, provenance, ids, seeds, configurations = [], [], set(), set(), set()
    paths = sorted(units.glob("block-*.json"))
    if len(paths) != expected_protocol["blocks"]:
        raise ValueError("The complete prespecified calibration-block allocation is required")
    for number, path in enumerate(paths):
        if path.name != f"block-{number:02d}.json":
            raise ValueError("Block positions must be contiguous and retain their original order")
        unit = read_json(path)
        _verify_seal(unit, str(path))
        if unit.get("block") != number or isinstance(unit.get("block"), bool):
            raise ValueError("Unit block identity changed")
        config = unit["config"]
        for key in ("N", "T", "C", "K"):
            if config.get(key) != expected_protocol[key] or type(config[key]) is not int:
                raise ValueError(f"Block differs from requested {key}")
        if (
            config.get("dataset") != "amazon-2018-all-beauty"
            or config.get("attack_family") != "instruction_injection"
        ):
            raise ValueError("RQ3 uses Amazon Beauty under instruction injection")
        configurations.add(digest(config))
        cell = runs / f"block-{number:02d}"
        manifest = read_json(cell / "manifest.json")
        immutable = manifest.get("immutable", {})
        fingerprint = manifest.get("fingerprint")
        if fingerprint != digest(immutable) or immutable.get("config") != config:
            raise ValueError("Run fingerprint or unit configuration changed")
        scores, tests, pins = [], [], []
        for role, count in (("calibration", expected_protocol["N"]), ("test", expected_protocol["tests_per_block"])):
            entries = unit[role]
            if len(entries) != count or immutable.get(role) != [
                {"id": item["id"], "sha256": digest(item)} for item in entries
            ]:
                raise ValueError("Native run and block allocation do not match")
            for item in entries:
                pair = item["pair"]
                if item["id"] in ids or pair["seed"] in seeds:
                    raise ValueError("Independent blocks must not reuse sequence identifiers or seeds")
                ids.add(item["id"])
                seeds.add(pair["seed"])
                if len(pair["clean"]) != config["T"] or len(pair["observed"]) != config["T"]:
                    raise ValueError("Sequence length differs from the declared protocol")
                candidates = [decision["candidates"] for decision in pair["observed"]]
                if candidates != [decision["candidates"] for decision in pair["clean"]]:
                    raise ValueError("Clean and observed candidates differ")
                clean, pin = _native_stage(cell, ["clean", item["id"]], fingerprint)
                pins.append(pin)
                screened, pin = _native_stage(cell, ["screen", item["id"]], fingerprint)
                pins.append(pin)
                row = {
                    "id": item["id"],
                    "candidates": candidates,
                    "clean_rankings": clean["rankings"],
                    "screened_rankings": screened["rankings"],
                    "positives": pair["evaluation"]["positives"],
                }
                _validate_test(row, expected_protocol)
                if role == "calibration":
                    scores.append(gmax(row["screened_rankings"], row["clean_rankings"]))
                else:
                    tests.append(row)
        blocks.append({"block": number, "calibration_scores": scores, "tests": tests})
        provenance.append({
            "block": number,
            "unit_sha256": file_digest(path),
            "manifest_sha256": file_digest(cell / "manifest.json"),
            "fingerprint": fingerprint,
            "stages": pins,
        })
    if len(configurations) != 1:
        raise ValueError("All independent blocks must share one fixed method configuration")
    result = seal({"kind": "crest-control-traces-v1", "protocol": dict(expected_protocol),
                   "blocks": blocks, "provenance": provenance})
    validate_traces(result, expected_protocol)
    return result


def _validate_test(row, protocol):
    t, k, c = protocol["T"], protocol["K"], protocol["C"]
    for name in ("screened_rankings", "clean_rankings"):
        _validate_rankings(row[name], name, t, k)
    if len(row["candidates"]) != t or len(row["positives"]) != t:
        raise ValueError("Each decision needs candidates and one held-out positive")
    for candidates, clean, screened, positive in zip(
        row["candidates"], row["clean_rankings"], row["screened_rankings"], row["positives"]
    ):
        if (
            not isinstance(candidates, list)
            or len(candidates) != c
            or any(not isinstance(item, str) or not item for item in candidates)
            or len(set(candidates)) != c
            or not set(clean) <= set(candidates)
            or not set(screened) <= set(candidates)
            or positive not in candidates
        ):
            raise ValueError("Invalid candidate set, ranking or held-out positive")


def validate_traces(traces, expected_protocol=PAPER_PROTOCOL):
    _verify_seal(traces, "control traces")
    if traces.get("kind") != "crest-control-traces-v1" or traces.get("protocol") != expected_protocol:
        raise ValueError("Control traces do not match the requested protocol")
    blocks, identities = traces["blocks"], set()
    if len(blocks) != expected_protocol["blocks"]:
        raise ValueError("Missing independent calibration blocks")
    for number, block in enumerate(blocks):
        if type(block.get("block")) is not int or block["block"] != number:
            raise ValueError("Block identity or ordering changed")
        if len(block["calibration_scores"]) != expected_protocol["N"] or len(block["tests"]) != expected_protocol["tests_per_block"]:
            raise ValueError("Incomplete calibration or test allocation")
        calibrate(block["calibration_scores"], 0.10, expected_protocol["T"])
        for row in block["tests"]:
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in identities:
                raise ValueError("Test sequence identities must be distinct across blocks")
            identities.add(row["id"])
            _validate_test(row, expected_protocol)


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _estimate(block_values, draws):
    numerators = [math.fsum(values) for values in block_values]
    denominators = [len(values) for values in block_values]
    total = sum(denominators)
    values = []
    for indices in draws:
        denominator = sum(denominators[i] for i in indices)
        if denominator:
            values.append(math.fsum(numerators[i] for i in indices) / denominator)
    values.sort()
    return {
        "mean": math.fsum(numerators) / total if total else None,
        "interval": [_quantile(values, 0.025), _quantile(values, 0.975)] if values else None,
        "denominator": total,
        "defined_bootstrap_replicates": len(values),
    }


def replay_controls(traces, *, bootstrap_replicates=10000, bootstrap_seed=3141592653,
                    expected_protocol=PAPER_PROTOCOL):
    validate_traces(traces, expected_protocol)
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise ValueError("Bootstrap seed must be an integer")
    protocol = traces["protocol"]
    generator = random.Random(bootstrap_seed)
    draws = [tuple(generator.randrange(protocol["blocks"]) for _ in range(protocol["blocks"]))
             for _ in range(bootstrap_replicates)]
    parameters = [(0.10, 0.10)] + [(a, 0.10) for a in ALPHAS if a != 0.10] + [(0.10, e) for e in ETAS if e != 0.10]
    settings = []
    for alpha, eta in parameters:
        blocks = []
        for block in traces["blocks"]:
            calibration = calibrate(block["calibration_scores"], alpha, protocol["T"])
            rows = []
            for test in block["tests"]:
                selected = select(test["screened_rankings"], test["candidates"], protocol["K"], eta, calibration["h_alpha"])
                returned = selected["status"] == "ok"
                screen_count = round(gmax(test["screened_rankings"], test["clean_rankings"]) * protocol["T"])
                metrics = utility(selected["rankings"], test["positives"]) if returned else None
                promotion = gmax(selected["rankings"], test["clean_rankings"]) if returned else None
                violation = returned and round(promotion * protocol["T"]) > promotion_budget(protocol["T"], eta)
                rows.append({
                    "id": test["id"], "returned": returned,
                    "clean_family_miscoverage": screen_count > calibration["h_alpha"],
                    "feasible_and_violating_return": violation,
                    "conditional_violation": violation if returned else None,
                    "recall_percent": 100 * metrics["recall"] if returned else None,
                    "ndcg_percent": 100 * metrics["ndcg"] if returned else None,
                    "gmax": promotion, "regime": selected["regime"],
                })
            blocks.append({"block": block["block"], "calibration": calibration, "rows": rows})
        keys = {
            "miscoverage_percent": ("clean_family_miscoverage", 100),
            "return_percent": ("returned", 100),
            "violating_return_percent": ("feasible_and_violating_return", 100),
            "conditional_violation_percent": ("conditional_violation", 100),
            "recall_percent": ("recall_percent", 1),
            "ndcg_percent": ("ndcg_percent", 1),
            "mean_gmax": ("gmax", 1),
        }
        summary = {
            output: _estimate([[row[key] * factor for row in block["rows"] if row[key] is not None]
                               for block in blocks], draws)
            for output, (key, factor) in keys.items()
        }
        settings.append({"alpha": alpha, "eta": eta, "summary": summary, "blocks": blocks})
    return seal({
        "kind": "crest-controls-replay-v1", "trace_sha256": traces["sha256"], "protocol": protocol,
        "uncertainty": {"method": "whole-block percentile bootstrap", "confidence": 0.95,
                        "replicates": bootstrap_replicates, "seed": bootstrap_seed,
                        "zero_return_policy": "Utility and conditional violation are undefined without returns; undefined bootstrap replicates are excluded."},
        "settings": settings,
    })


def plotting_values(report):
    _verify_seal(report, "controls report")
    default = next(row for row in report["settings"] if row["alpha"] == row["eta"] == 0.10)
    return {
        "source_sha256": report["sha256"], "units": {"miscoverage": "percent", "ndcg": "percent", "gmax": "fraction"},
        "alpha": [{"x": alpha, "nominal_percent": 100 * alpha,
                   **next(row for row in report["settings"] if row["alpha"] == alpha and row["eta"] == 0.10)["summary"]["miscoverage_percent"]}
                  for alpha in ALPHAS],
        "eta": [{"x": eta, **next(row for row in report["settings"] if row["eta"] == eta and row["alpha"] == 0.10)["summary"]}
                for eta in ETAS],
        "shared_default": {"alpha": 0.10, "eta": 0.10, "summary": default["summary"]},
    }
