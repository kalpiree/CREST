import copy
import json
from pathlib import Path

from .artifacts import digest, read_json
from .calibration import calibrate
from .metrics import gmax, identity_metrics, utility
from .records import parse_ranking
from .reporting import summarize_results
from .selection import select


METHODS = ["Without Screening", "Screening-Only", "Random-Matched", "CREST"]


def _stage(cell, name, fingerprint):
    path = Path(cell) / "stages" / (digest(name) + ".json")
    if not path.is_file():
        raise ValueError(f"Missing completed stage: {name}")
    value = read_json(path)
    if value.get("fingerprint") != fingerprint or value.get("name") != name or value.get("result_sha256") != digest(value.get("result")):
        raise ValueError(f"Corrupt stage: {name}")
    if value.get("status") not in ("complete", "failed"):
        raise ValueError(f"Unfinished stage: {name}")
    return copy.deepcopy(value["result"])


def _rankings(rankings, decisions, k):
    if not isinstance(rankings, list) or len(rankings) != len(decisions):
        raise ValueError("Saved rankings must cover every decision")
    for ranking, decision in zip(rankings, decisions):
        parse_ranking(json.dumps(ranking), decision["candidates"], k)
    return rankings


def _evaluated(result, item, clean, k, include_records):
    value = copy.deepcopy(result)
    if value["status"] != "ok":
        return value
    pair = item["pair"]
    output = _rankings(value["rankings"], pair["observed"], k)
    value["metrics"] = {**utility(output, pair["evaluation"]["positives"]), "gmax": gmax(output, clean)}
    if include_records:
        value["record_metrics"] = identity_metrics(set(value["omitted_ids"]), set(pair["evaluation"]["attacked_ids"]),
                                                   {record["id"] for decision in pair["observed"] for record in decision["records"]})
    return value


def analyze_ablation(unit, cell, *, plan_sha256=None):
    cell = Path(cell)
    config = unit["config"]
    calibration_items, test_items = unit["calibration"], unit["test"]
    manifest = read_json(cell / "manifest.json")
    immutable, fingerprint = manifest["immutable"], manifest["fingerprint"]
    if digest(immutable) != fingerprint or immutable["config"] != config:
        raise ValueError("The cell manifest does not match the requested configuration")
    if plan_sha256 is not None and immutable.get("validated_plan_sha256") != plan_sha256:
        raise ValueError("The cell belongs to a different dependency plan")
    if len(calibration_items) != config["N"] or not test_items:
        raise ValueError("A complete calibration cohort and nonempty test cohort are required")
    for role, items in (("calibration", calibration_items), ("test", test_items)):
        if immutable[role] != [{"id": item["id"], "sha256": digest(item)} for item in items]:
            raise ValueError("The calibration or test cohort differs from the completed cell")
    clean, screened = {}, {}
    for item in calibration_items + test_items:
        identity = item["id"]
        clean_value = _stage(cell, ["clean", identity], fingerprint)
        screen_value = _stage(cell, ["screen", identity], fingerprint)
        if clean_value.get("status") != "ok" or screen_value.get("status") != "ok":
            raise ValueError("Complete valid clean and screened rankings are required for ablation")
        if "observed_rankings" not in screen_value:
            raise ValueError("The saved screen predates observed_rankings retention; use a release-generated RQ1 cell or recover exact empty-omission rankings separately")
        pair = item["pair"]
        clean[identity] = _rankings(clean_value["rankings"], pair["clean"], config["K"])
        _rankings(screen_value["rankings"], pair["observed"], config["K"])
        _rankings(screen_value["observed_rankings"], pair["observed"], config["K"])
        screened[identity] = screen_value
    unfiltered_scores = [gmax(screened[item["id"]]["observed_rankings"], clean[item["id"]]) for item in calibration_items]
    screened_scores = [gmax(screened[item["id"]]["rankings"], clean[item["id"]]) for item in calibration_items]
    own_calibration = {"scores": unfiltered_scores, **calibrate(unfiltered_scores, config["alpha"], config["T"])}
    full_calibration = _stage(cell, ["calibration"], fingerprint)
    expected_full = {"status": "ok", "scores": screened_scores, **calibrate(screened_scores, config["alpha"], config["T"])}
    if full_calibration != expected_full:
        raise ValueError("Full CREST calibration differs from the saved screened map")
    rows = []
    for item in test_items:
        identity, pair = item["id"], item["pair"]
        source = screened[identity]
        no_screening = {**select(source["observed_rankings"], [decision["candidates"] for decision in pair["observed"]],
                                 config["K"], config["eta"], own_calibration["h_alpha"]), "omitted_ids": []}
        screening_only = {"status": "ok", "rankings": source["rankings"], "omitted_ids": source["omitted_ids"]}
        matched = _stage(cell, ["method", "Random-Matched", identity], fingerprint)
        full = _stage(cell, ["method", "CREST", identity], fingerprint)
        if full.get("status") == "ok" and full.get("omitted_ids") != source["omitted_ids"]:
            raise ValueError("CREST and Screening-Only have different omissions")
        if matched.get("status") == "ok":
            from .baselines import random_matched

            expected = random_matched(pair["observed"], source["omitted_ids"], pair["seed"] + config["random_seed_offset"])
            if matched.get("omitted_ids") != expected["omitted_ids"]:
                raise ValueError("Random-Matched omissions differ from the frozen matching rule")
        methods = {method: _evaluated(value, item, clean[identity], config["K"], method != "Without Screening")
                   for method, value in zip(METHODS, (no_screening, screening_only, matched, full))}
        rows.append({"id": identity, "seed_group": item["seed_group"], "methods": methods,
                     "clean_reference_status": "ok", "common_returned": all(value["status"] == "ok" for value in methods.values())})
    return {"kind": "crest_component_ablation", "source_fingerprint": fingerprint, "config": config,
            "calibration": {"Without Screening": own_calibration, "CREST": full_calibration},
            "rows": rows, "summary": summarize_results(rows, METHODS, k=config["K"]), "new_model_calls": 0,
            "calibration_policy": "Without Screening calibrates its identity map separately on the same calibration pairs. Screening-Only retains CREST screening and omits final selection. Random-Matched uses the saved matched omissions without final selection."}
