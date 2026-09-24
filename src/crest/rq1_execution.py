import copy
import contextlib
import fcntl
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .baselines import random_matched, require_methods, run_record_baseline
from .calibration import calibrate
from .experiments import implementation_digest
from .metrics import gmax, identity_metrics, utility
from .records import parse_ranking, validate_decision
from .screening import screen
from .selection import select


class RQ1Deadline(RuntimeError):
    pass


def observed_view(decisions):

    result = []
    for decision in decisions:
        validate_decision(decision)
        result.append({"user_id": decision["user_id"],
                       "candidates": list(decision["candidates"]),
                       "records": [{"id": row["id"], "type": row["type"],
                                    "fields": dict(row["fields"])} for row in decision["records"]]})
    return result


class DeadlineRanker:
    def __init__(self, ranker, check):
        self.base, self.check = ranker, check

    def __getattr__(self, key):
        return getattr(self.base, key)

    def rank(self, *args, **kwargs):
        self.check()
        value = self.base.rank(*args, **kwargs)
        self.check()
        return value


class DeadlineCallable:
    def __init__(self, provider, check):
        self.base, self.check = provider, check

    def __getattr__(self, key):
        return getattr(self.base, key)

    def __call__(self, *args, **kwargs):
        self.check()
        value = self.base(*args, **kwargs)
        self.check()
        return value

    def generate(self, *args, **kwargs):
        self.check()
        value = self.base.generate(*args, **kwargs)


        return value


def bounded_detector(detector, check):
    result = copy.copy(detector)
    if hasattr(result, "scorer"):
        result.scorer = copy.copy(detector.scorer)
        if getattr(result.scorer, "model", None) is not None:
            result.scorer.model = DeadlineCallable(result.scorer.model, check)
        result.scorer.score = DeadlineCallable(result.scorer.score, check)
    if hasattr(result, "rank_provider"):
        result.rank_provider = DeadlineCallable(result.rank_provider, check)
    if hasattr(result, "continuation_provider"):
        provider = copy.copy(result.continuation_provider)
        if hasattr(provider, "ranker"):
            provider.ranker = copy.copy(provider.ranker)
            if getattr(provider.ranker, "model", None) is not None:
                provider.ranker.model = DeadlineCallable(provider.ranker.model, check)
        result.continuation_provider = DeadlineCallable(provider, check)
    result.detect = DeadlineCallable(result.detect, check)
    return result


def _validate(config, calibration_items, test_items, ranker, detectors):
    required = {"T", "K", "C", "N", "alpha", "eta", "beta", "penalty",
                "beam_width", "max_depth", "methods", "attack_family", "random_seed_offset"}
    if not required <= config.keys():
        raise ValueError("Unresolved RQ1 runtime settings: " + str(sorted(required - config.keys())))
    for name in ("T", "K", "C", "N", "beam_width", "max_depth"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["K"] > config["C"] or config["N"] != len(calibration_items) or not test_items:
        raise ValueError("Incorrect calibration/test allocation or ranking shape")
    if isinstance(config["random_seed_offset"], bool) or not isinstance(config["random_seed_offset"], int):
        raise ValueError("An explicit fixed Random-Matched seed offset is required")
    if not getattr(ranker, "metadata", None):
        raise ValueError("Ranker metadata is required")
    require_methods(config["methods"], detectors)
    from .rq1_protocol import rq1_matrix
    matrix = rq1_matrix([config["dataset"]], [config["attack_family"]])
    expected = matrix[0]["methods"]
    if set(config["methods"]) != set(expected):
        raise ValueError("All and only applicable RQ1 methods must be present")
    calibrate([0.0] * config["N"], config["alpha"], config["T"])
    seen = set()
    for item in [*calibration_items, *test_items]:
        if not isinstance(item.get("id"), str) or item["id"] in seen:
            raise ValueError("Sequence IDs must be distinct across calibration and test")
        seen.add(item["id"])
        pair = item["pair"]
        if len(pair["clean"]) != config["T"] or len(pair["observed"]) != config["T"]:
            raise ValueError("Sequence shape differs from frozen T")
        if len(pair["evaluation"]["positives"]) != config["T"]:
            raise ValueError("A positive is required per decision")
        if pair["attack"]["name"] != config["attack_family"]:
            raise ValueError("Attack family differs from frozen cell")
        for clean, observed in zip(pair["clean"], pair["observed"]):
            validate_decision(clean)
            validate_decision(observed)
            if clean["candidates"] != observed["candidates"] or len(clean["candidates"]) != config["C"]:
                raise ValueError("Pair candidates differ from frozen C/order")
        if isinstance(pair.get("seed"), bool) or not isinstance(pair.get("seed"), int):
            raise ValueError("An immutable sequence seed is required")
    if any("seed_group" not in item for item in test_items):
        raise ValueError("Every test sequence needs a seed group")


def verified_fixed_calibration(source, config, calibration_items, inference=None):
    source = Path(source)
    with (source / ".worker.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = read_json(source / "manifest.json")
        immutable, fingerprint = manifest["immutable"], manifest["fingerprint"]
        if digest(immutable) != fingerprint or immutable["config"] != config:
            raise ValueError("Fixed calibration source configuration or manifest differs")
        cohort = [{"id": item["id"], "sha256": digest(item)} for item in calibration_items]
        if immutable["calibration"] != cohort or len(calibration_items) != config["N"]:
            raise ValueError("Fixed calibration source cohort differs")
        if immutable["implementation_sha256"] != implementation_digest():
            raise ValueError("Fixed calibration source implementation differs")
        if inference is not None and digest(immutable["inference"]) != digest(inference):
            raise ValueError("Fixed calibration source ranking runtime differs")
        proofs = []
        def saved(name):
            path = source / "stages" / (digest(name) + ".json")
            value = read_json(path)
            if value.get("fingerprint") != fingerprint or value.get("name") != name or value.get("result_sha256") != digest(value.get("result")):
                raise ValueError("Corrupt fixed calibration source stage")
            if value.get("status") != "complete" or value["result"].get("status") != "ok":
                raise ValueError("Fixed calibration source requires completed successful stages")
            proofs.append({"name": name, "sha256": file_digest(path)})
            return value["result"]
        scores = []
        for item in calibration_items:
            clean = saved(["clean", item["id"]])["rankings"]
            screened = saved(["screen", item["id"]])["rankings"]
            for rankings, key in ((clean, "clean"), (screened, "observed")):
                if len(rankings) != config["T"]:
                    raise ValueError("Fixed calibration source sequence length differs")
                for ranking, decision in zip(rankings, item["pair"][key]):
                    parse_ranking(json.dumps(ranking), decision["candidates"], config["K"])
            scores.append(gmax(screened, clean))
        result = saved(["calibration"])
        expected = {"status": "ok", "scores": scores, **calibrate(scores, config["alpha"], config["T"])}
        if result != expected:
            raise ValueError("Fixed calibration threshold does not match its original calibration scores")
        return {"result": copy.deepcopy(result), "source_fingerprint": fingerprint,
                "manifest_sha256": file_digest(source / "manifest.json"), "stage_files": proofs}


def execute_cell(config, calibration_items, test_items, ranker, detectors, output,
                 *, validated_plan_sha256, budget_seconds, max_phase="complete", clock=time.monotonic,
                 fixed_calibration_source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)


    with (output / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_cell_locked(config, calibration_items, test_items, ranker, detectors, output,
            validated_plan_sha256=validated_plan_sha256, budget_seconds=budget_seconds,
            max_phase=max_phase, clock=clock, fixed_calibration_source=fixed_calibration_source)


def _execute_cell_locked(config, calibration_items, test_items, ranker, detectors, output,
                         *, validated_plan_sha256, budget_seconds, max_phase, clock, fixed_calibration_source=None):


    if not isinstance(validated_plan_sha256, str) or len(validated_plan_sha256) != 64 or any(c not in "0123456789abcdef" for c in validated_plan_sha256):
        raise ValueError("A validated plan SHA256 is required")
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (float, int)) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("A finite positive invocation budget is required")
    if max_phase not in ("baselines", "random_matched", "complete"):
        raise ValueError("Unknown stopping phase")
    started = clock()
    _validate(config, calibration_items, test_items, ranker, detectors)
    config = copy.deepcopy(config)


    if clock() - started >= budget_seconds:
        return {"status": "partial", "phase": "provider_loading", "reason": "Budget reached before loading"}
    if callable(getattr(ranker, "load", None)):
        ranker.load()
    for detector in detectors.values():
        if clock() - started >= budget_seconds:
            return {"status": "partial", "phase": "provider_loading", "reason": "Budget reached; no next provider loaded"}
        if callable(getattr(detector, "load", None)):
            detector.load()
    if clock() - started >= budget_seconds:
        return {"status": "partial", "phase": "provider_loading", "reason": "Loading used invocation budget"}
    immutable = json.loads(json.dumps({"protocol": "rq1-cell-execution-v1",
        "validated_plan_sha256": validated_plan_sha256, "config": config,
        "calibration": [{"id": x["id"], "sha256": digest(x)} for x in calibration_items],
        "test": [{"id": x["id"], "sha256": digest(x)} for x in test_items],
        "inference": ranker.metadata, "detectors": {k: v.metadata for k, v in detectors.items()},
        "implementation_sha256": implementation_digest()}, allow_nan=False))
    fixed_calibration = None
    if fixed_calibration_source is not None:
        if Path(fixed_calibration_source).resolve() == Path(output).resolve():
            raise ValueError("Fixed calibration must come from a separate primary cell")
        fixed_calibration = verified_fixed_calibration(fixed_calibration_source, config, calibration_items, ranker.metadata)
        immutable["fixed_calibration_source"] = fixed_calibration
    fingerprint = digest(immutable)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completed, active, failed_stages = [], [None], []

    def status(state, error=None):
        write_json(output / "status.json", {"status": state, "fingerprint": fingerprint,
            "updated_unix": time.time(), "active_stage": active[0], "completed_stages": completed,
            "invocation_seconds": clock() - started, "error": error,
            "failed_stages": failed_stages,
            "inference": ranker.statistics() if hasattr(ranker, "statistics") else None})

    def check():
        if clock() - started >= budget_seconds:
            raise RQ1Deadline("Invocation budget reached; resume identical inputs and query caches")

    bounded = DeadlineRanker(ranker, check)
    bounded_detectors = {name: bounded_detector(detector, check) for name, detector in detectors.items()}
    with contextlib.nullcontext():
        path = output / "manifest.json"
        if path.exists():
            prior = read_json(path)
            if prior.get("fingerprint") != fingerprint or digest(prior.get("immutable")) != fingerprint:
                raise ValueError("Different or corrupted immutable RQ1 inputs; use another output directory")
        else:
            write_json(path, {"fingerprint": fingerprint, "immutable": immutable})

        def stage(name, compute):
            active[0] = name
            target = output / "stages" / (digest(name) + ".json")
            if target.exists():
                old = read_json(target)
                if old.get("fingerprint") != fingerprint or old.get("name") != name or digest(old.get("result")) != old.get("result_sha256"):
                    raise ValueError(f"Corrupted RQ1 checkpoint: {name}")
                if old.get("status") in ("complete", "failed"):
                    completed.append(name)
                    if old["result"].get("status") != "ok":
                        failed_stages.append({"stage": name, "status": old["result"].get("status")})
                    return old["result"]
                if old.get("status") != "partial":
                    raise ValueError("Unknown stage checkpoint status")
            check()
            status("running")
            tick = clock()
            previous_attempts = old.get("attempts", []) if target.exists() else []
            try:
                result = compute()
                state = "complete"
            except RQ1Deadline as error:
                result = {"status": "partial", "error": str(error)}
                state = "partial"
            except Exception as error:
                result = {"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}}
                state = "failed"
            attempt = {"status": state, "seconds": clock() - tick, "finished_unix": time.time()}
            write_json(target, {"name": name, "fingerprint": fingerprint, "status": state,
                "result": result, "result_sha256": digest(result), "attempts": [*previous_attempts, attempt]})
            if state == "partial":
                raise RQ1Deadline(result["error"])
            completed.append(name)
            if result.get("status") != "ok":
                failed_stages.append({"stage": name, "status": result.get("status")})
            return result

        def ranks(decisions, omitted=frozenset(), datamark=False):
            result = [bounded.rank(d, config["K"], omitted, datamark) for d in observed_view(decisions)]
            for row, decision in zip(result, decisions):
                parse_ranking(json.dumps(row), decision["candidates"], config["K"])
            return {"status": "ok", "rankings": result, "omitted_ids": sorted(omitted)}

        def screening(pair):
            decisions = observed_view(pair["observed"])
            ids = [[r["id"] for r in d["records"]] for d in decisions]
            result = screen(ids, lambda omitted: ranks(decisions, omitted)["rankings"],
                            config["K"], config["beta"], config["penalty"],
                            config["beam_width"], config["max_depth"])
            return {"status": "ok", **result}

        try:
            clean = {x["id"]: stage(["clean", x["id"]], lambda x=x: ranks(x["pair"]["clean"]))
                     for x in test_items}
            methods = {x["id"]: {} for x in test_items}
            for method in config["methods"]:
                if method in ("CREST", "Random-Matched"):
                    continue
                for item in test_items:
                    def baseline(item=item, method=method):
                        decisions = observed_view(item["pair"]["observed"])
                        if method in detectors:
                            return run_record_baseline(method, decisions, config["K"], bounded,
                                bounded_detectors[method], attack_family=config["attack_family"])
                        return ranks(decisions, datamark=method == "Spotlighting")
                    methods[item["id"]][method] = stage(["baseline", method, item["id"]], baseline)
            if max_phase == "baselines":
                status("paused_after_baselines")
                return {"status": "paused_after_baselines", "fingerprint": fingerprint,
                        "contains_failed_results": bool(failed_stages), "failed_stages": failed_stages}
            if fixed_calibration is None:
                clean.update({x["id"]: stage(["clean", x["id"]], lambda x=x: ranks(x["pair"]["clean"]))
                              for x in calibration_items})
            screening_items = [*calibration_items, *test_items] if fixed_calibration is None else test_items
            screened = {x["id"]: stage(["screen", x["id"]], lambda x=x: screening(x["pair"]))
                        for x in screening_items}

            def calibration_result():
                if fixed_calibration is not None:
                    return copy.deepcopy(fixed_calibration["result"])
                eligible = all(screened[x["id"]]["status"] == clean[x["id"]]["status"] == "ok" for x in calibration_items)
                if not eligible:
                    return {"status": "blocked", "reason": "Incomplete frozen calibration; N cannot be reduced"}
                scores = [gmax(screened[x["id"]]["rankings"], clean[x["id"]]["rankings"]) for x in calibration_items]
                return {"status": "ok", "scores": scores, **calibrate(scores, config["alpha"], config["T"])}
            calibration = stage(["calibration"], calibration_result)
            for method in ("Random-Matched", "CREST"):
                if method == "CREST" and max_phase == "random_matched":
                    status("paused_after_random_matched")
                    return {"status": "paused_after_random_matched", "fingerprint": fingerprint,
                            "contains_failed_results": bool(failed_stages), "failed_stages": failed_stages}
                for item in test_items:
                    def dependent(item=item, method=method):
                        screened_result = screened[item["id"]]
                        if screened_result["status"] != "ok":
                            return {"status": "blocked", "reason": "Screening failed"}
                        pair = item["pair"]
                        if method == "Random-Matched":
                            match = random_matched(observed_view(pair["observed"]), screened_result["omitted_ids"], pair["seed"] + config["random_seed_offset"])
                            return {**ranks(pair["observed"], frozenset(match["omitted_ids"])), "matching": match}
                        if calibration["status"] != "ok":
                            return {"status": "blocked", "reason": "Calibration unavailable"}
                        return {**select(screened_result["rankings"], [d["candidates"] for d in pair["observed"]],
                                        config["K"], config["eta"], calibration["h_alpha"]),
                                "omitted_ids": screened_result["omitted_ids"]}
                    methods[item["id"]][method] = stage(["method", method, item["id"]], dependent)
            rows = []
            for item in test_items:
                pair, identity = item["pair"], item["id"]
                outputs = copy.deepcopy(methods[identity])
                for method, result in outputs.items():
                    if result["status"] == "ok" and clean[identity]["status"] == "ok":
                        result["metrics"] = {**utility(result["rankings"], pair["evaluation"]["positives"]),
                                             "gmax": gmax(result["rankings"], clean[identity]["rankings"])}
                    if result["status"] == "ok" and method in {*detectors, "Random-Matched", "CREST"}:
                        result["record_metrics"] = identity_metrics(set(result["omitted_ids"]),
                            set(pair["evaluation"]["attacked_ids"]), {r["id"] for d in pair["observed"] for r in d["records"]})
                rows.append({"id": identity, "seed_group": item["seed_group"], "methods": outputs,
                             "clean_reference_status": clean[identity]["status"],
                             "common_returned": all(x["status"] == "ok" for x in outputs.values())})
            report = {"fingerprint": fingerprint, "calibration": calibration,
                      "rows": rows, "summary": aggregate_rows(rows, config["methods"])}
            write_json(output / "report.json", report)
            status("complete_with_failures" if any(x["status"] != "ok" for r in rows for x in r["methods"].values()) or any(x["status"] != "ok" for x in clean.values()) or calibration["status"] != "ok" else "complete")
            return report
        except RQ1Deadline:
            status("partial")
            return {"status": "partial", "fingerprint": fingerprint, "completed_stages": completed}
        except Exception as error:
            status("failed", {"type": type(error).__name__, "message": str(error)})
            raise


def aggregate_rows(rows, methods):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["seed_group"])].append(row)
    result = {"attempted_sequences": len(rows), "methods": {}, "seed_groups": {}}
    for group, subset in groups.items():
        common = [r for r in subset if r["common_returned"] and r["clean_reference_status"] == "ok"]
        group_result = {"attempted_sequences": len(subset), "common_returned_with_reference": len(common), "methods": {}}
        for method in methods:
            returned = [r["methods"][method] for r in subset if r["methods"][method]["status"] == "ok"]
            counts = [x["record_metrics"] for x in returned if "record_metrics" in x]
            fprs = [x["fp"] / x["unmodified"] for x in counts if x["unmodified"]]
            pooled = {k: sum(x[k] for x in counts) for k in ("tp", "fp", "fn", "unmodified")} if counts else None
            if pooled is not None:
                tp, fp, fn, unmodified = (pooled[k] for k in ("tp", "fp", "fn", "unmodified"))
                pooled.update(f1=2*tp/(2*tp+fp+fn) if tp+fn else None,
                              fpr=fp/unmodified if unmodified else None,
                              completed_identification_sequences=len(counts))
            group_result["methods"][method] = {"returned_sequences": len(returned), "return_rate": len(returned)/len(subset),
                "common_sequence_metrics": {k: math.fsum(r["methods"][method]["metrics"][k] for r in common)/len(common)
                                            for k in ("recall", "ndcg", "gmax")} if common else None,
                "identity_counts_pooled_within_seed": pooled,
                "sequence_mean_fpr": math.fsum(fprs) / len(fprs) if fprs else None,
                "fpr_sequences": len(fprs),
                "statuses": {state: sum(r["methods"][method]["status"] == state for r in subset)
                             for state in sorted({r["methods"][method]["status"] for r in subset})}}
        result["seed_groups"][group] = group_result
    for method in methods:
        returned = sum(r["methods"][method]["status"] == "ok" for r in rows)
        result["methods"][method] = {"returned_sequences": returned, "return_rate": returned/len(rows) if rows else None}
        fprs = [g["methods"][method]["sequence_mean_fpr"] for g in result["seed_groups"].values()
                if g["methods"][method]["sequence_mean_fpr"] is not None]
        result["methods"][method]["fpr"] = math.fsum(fprs) / len(fprs) if fprs else None
    result["fpr_aggregation"] = "Mean of sequence-level false-positive rates within each seed, then mean across seeds; pooled identity counts are diagnostics only."
    result["uncertainty"] = "No intervals computed; a prespecified resampling policy is required for publication."
    return result
