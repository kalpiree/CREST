import random
import math
from collections import defaultdict
import copy

from .records import parse_ranking, validate_decision
import json


IMPLEMENTED_PILOT_METHODS = {"Backbone", "Spotlighting", "Random-Matched", "CREST"}
RECORD_METHOD_FAMILIES = {
    "PromptGuard2-Record": {"instruction_injection", "deceptive_text_rewriting", "interaction_history_manipulation"},
    "RewriteDetection-Record": {"deceptive_text_rewriting"},
    "RETURN-Del": {"interaction_history_manipulation"},
}


def require_methods(methods, detectors=None):
    detectors = detectors or {}
    unknown = set(detectors) - set(RECORD_METHOD_FAMILIES)
    if unknown:
        raise ValueError("Unknown detector adapters: " + ", ".join(sorted(unknown)))
    for name, detector in detectors.items():
        if not callable(getattr(detector, "detect", None)) or not getattr(detector, "metadata", None):
            raise ValueError(f"Detector {name} requires detect() and frozen metadata")
    missing = sorted(set(methods) - IMPLEMENTED_PILOT_METHODS - set(detectors))
    if missing:
        raise NotImplementedError("Baseline adapters are not implemented: " + ", ".join(missing))
    if "Random-Matched" in methods and "CREST" not in methods:
        raise ValueError("Random-Matched requires CREST's realized screening omissions")
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate methods")


def run_record_baseline(name, observed, k, ranker, detector, *, attack_family):
    if name not in RECORD_METHOD_FAMILIES or attack_family not in RECORD_METHOD_FAMILIES[name]:
        raise ValueError(f"{name} is not applicable to {attack_family}")
    if not isinstance(observed, (list, tuple)) or not observed:
        raise ValueError("Pass a nonempty observed decision sequence")
    decisions = []
    for decision in observed:
        validate_decision(decision)
        decisions.append({"user_id": decision["user_id"], "candidates": list(decision["candidates"]), "records": [{"id": record["id"], "type": record["type"], "fields": dict(record["fields"])} for record in decision["records"]]})
    report = detector.detect(copy.deepcopy(decisions))
    omitted = report["omitted_ids"]
    known = {record["id"] for decision in decisions for record in decision["records"]}
    if not isinstance(omitted, list) or any(not isinstance(identity, str) for identity in omitted) or len(set(omitted)) != len(omitted) or not set(omitted).issubset(known):
        raise ValueError("Detector must return distinct observed record identities")
    rankings = [ranker.rank(decision, k, frozenset(omitted)) for decision in decisions]
    for ranking, decision in zip(rankings, decisions):
        parse_ranking(json.dumps(ranking), decision["candidates"], k)
    return {"status": "ok", "rankings": rankings, "omitted_ids": sorted(omitted), "detection": report}


def random_matched(decisions, omitted_ids, seed):
    occurrences = defaultdict(int)
    types = {}
    for decision in decisions:
        for record in decision["records"]:
            occurrences[record["id"]] += 1
            if record["id"] in types and types[record["id"]] != record["type"]:
                raise ValueError("Record type changed for one identity")
            types[record["id"]] = record["type"]
    buckets = defaultdict(list)
    for record_id in sorted(types):
        pattern = tuple(len(decision["records"]) if any(record["id"] == record_id for record in decision["records"]) else 0 for decision in decisions)
        key = (types[record_id], occurrences[record_id], tuple(sorted(value for value in pattern if value)))
        buckets[key].append(record_id)
    chosen = []
    audit = []
    possibilities = 1
    rng = random.Random(seed)
    omitted = set(omitted_ids)
    for key in sorted(buckets):
        bucket = buckets[key]
        count = len(omitted.intersection(bucket))
        chosen.extend(rng.sample(bucket, count))
        possibilities *= math.comb(len(bucket), count)
        if count:
            audit.append({"type": key[0], "appearances": key[1], "record_denominators": list(key[2]), "pool_size": len(bucket), "selected_count": count})
    if len(chosen) != len(omitted):
        raise ValueError("Unknown omitted identity")
    return {"omitted_ids": sorted(chosen), "bucket_audit": audit, "possible_subsets": possibilities, "degenerate": possibilities == 1, "overlap_with_crest": len(set(chosen) & omitted)}
