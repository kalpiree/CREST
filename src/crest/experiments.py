import fcntl
import time
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .baselines import random_matched, require_methods, run_record_baseline
from .calibration import calibrate
from .data import make_sequence
from .metrics import gmax, identity_metrics, utility
from .screening import screen
from .selection import select


def implementation_digest():
    root = Path(__file__).parent
    return digest({str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*.py"))})


def run_pilot(config, pool, ranker, output_dir, detectors=None):
    if config.get("status") != "pilot_only" or config.get("is_full_reproduction") is not False:
        raise ValueError("This launcher is restricted to explicitly labeled infrastructure pilots")
    detectors = detectors or {}
    require_methods(config["methods"], detectors)
    if set(config["methods"]) & {"RETURN-Del", "RewriteDetection-Record"}:
        raise ValueError("This infrastructure pilot generates instruction injection only; use the matching attack family for these adapters")
    if min(config["N"], config["test_sequences"]) < 1:
        raise ValueError("Calibration and test counts must be positive")
    if config["K"] > config["candidates_per_decision"]:
        raise ValueError("K exceeds the candidate set")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker already owns this output directory") from error
        ranker.load()
        for detector in detectors.values():
            if callable(getattr(detector, "load", None)):
                detector.load()
        detector_metadata = {name: detector.metadata for name, detector in detectors.items()}
        fingerprint = digest({"config": config, "pool": pool, "inference": ranker.metadata, "detectors": detector_metadata, "implementation": implementation_digest()})
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to different inputs/code; choose a new output directory")
        write_json(manifest_path, {"fingerprint": fingerprint, "status": "pilot_only", "config": config, "inference": ranker.metadata, "detectors": detector_metadata, "dataset_provenance": pool["provenance"], "implementation_hash": implementation_digest()})
        completed = []
        active_stage = [None]
        scores = []
        started = time.time()

        def update_status(state, current=None, error=None):
            value = {"status": state, "fingerprint": fingerprint, "completed_stages": completed, "current_stage": current, "process_started": started, "updated": time.time(), "inference_this_process": ranker.statistics()}
            if error is not None:
                value["error"] = {"type": type(error).__name__, "message": str(error)}
            write_json(output_dir / "status.json", value)

        def sequence(seed):
            return make_sequence(pool, seed, config["T"], config["candidates_per_decision"], config["history_records"], config["recurrence_fraction"])

        def screened(observed):
            ids = [[record["id"] for record in decision["records"]] for decision in observed]
            return screen(ids, lambda omitted: [ranker.rank(decision, config["K"], omitted) for decision in observed], config["K"], config["beta"], config["penalty"], config["beam_width"], config["max_depth"])

        def stage(name, compute):
            active_stage[0] = name
            path = output_dir / "stages" / f"{name}.json"
            if path.exists():
                record = read_json(path)
                if record.get("fingerprint") != fingerprint or record.get("status") != "complete":
                    raise ValueError(f"Invalid stage checkpoint: {name}")
                if digest(record["result"]) != record.get("result_hash"):
                    raise ValueError(f"Corrupted stage checkpoint: {name}")
                completed.append(name)
                update_status("running", name)
                return record["result"]
            update_status("running", name)
            clock = time.monotonic()
            result = compute()
            write_json(path, {"fingerprint": fingerprint, "status": "complete", "result": result, "result_hash": digest(result), "seconds": time.monotonic() - clock})
            completed.append(name)
            update_status("running", name)
            return result

        try:
            for index in range(config["N"]):
                pair = sequence(config["seed"] + index)

                def calibration_result(pair=pair):
                    result = screened(pair["observed"])
                    clean = [ranker.rank(decision, config["K"]) for decision in pair["clean"]]
                    return {"seed": pair["seed"], "screening": result, "clean_rankings": clean, "score": gmax(result["rankings"], clean)}

                result = stage(f"calibration-{index:04d}", calibration_result)
                scores.append(result["score"])
            calibration = calibrate(scores, config["alpha"], config["T"])
            write_json(output_dir / "calibration.json", calibration)
            test_results = []
            for index in range(config["test_sequences"]):
                pair = sequence(config["seed"] + 1000000 + index)

                def test_result(pair=pair):
                    observed = pair["observed"]
                    candidates = [decision["candidates"] for decision in observed]
                    clean = [ranker.rank(decision, config["K"]) for decision in pair["clean"]]
                    screening = screened(observed)
                    outputs = {}
                    for method in config["methods"]:
                        omissions = []
                        extra = {}
                        if method == "CREST":
                            omissions = screening["omitted_ids"]
                            result = select(screening["rankings"], candidates, config["K"], config["eta"], calibration["h_alpha"])
                        elif method in detectors:
                            result = run_record_baseline(method, observed, config["K"], ranker, detectors[method], attack_family=pair["attack"]["name"])
                            omissions = result["omitted_ids"]
                        else:
                            if method == "Random-Matched":
                                extra["matching"] = random_matched(observed, screening["omitted_ids"], pair["seed"] + 71)
                                omissions = extra["matching"]["omitted_ids"]
                            rankings = [ranker.rank(decision, config["K"], frozenset(omissions), method == "Spotlighting") for decision in observed]
                            result = {"status": "ok", "rankings": rankings}
                        record = {**result, **extra, "omitted_ids": omissions}
                        if result["status"] == "ok":
                            record["metrics"] = {**utility(result["rankings"], pair["evaluation"]["positives"]), "gmax": gmax(result["rankings"], clean)}
                        if method == "CREST" or method in detectors:
                            record["record_metrics"] = identity_metrics(set(omissions), set(pair["evaluation"]["attacked_ids"]), {record["id"] for decision in observed for record in decision["records"]})
                        outputs[method] = record
                    return {"seed": pair["seed"], "evaluation": pair["evaluation"], "attack": pair["attack"], "clean_rankings": clean, "screening": screening, "methods": outputs, "common_returned": all(result["status"] == "ok" for result in outputs.values())}

                test_results.append(stage(f"test-{index:04d}", test_result))
            common = [result for result in test_results if result["common_returned"]]
            summary = {"status": "pilot_only", "is_full_reproduction": False, "fingerprint": fingerprint, "calibration": calibration, "attempted_sequences": len(test_results), "common_returned_sequences": len(common), "methods": {}, "missing_baselines": config.get("missing_main_table_baselines", []), "notes": config["notes"]}
            for method in config["methods"]:
                returned = sum(result["methods"][method]["status"] == "ok" for result in test_results)
                summary["methods"][method] = {"returned_sequences": returned, "return_rate": returned / len(test_results), "common_sequence_metrics": {key: sum(result["methods"][method]["metrics"][key] for result in common) / len(common) for key in ("recall", "ndcg", "gmax")} if common else None}
            write_json(output_dir / "summary.json", summary)
            update_status("complete")
            return summary
        except Exception as error:
            update_status("failed", active_stage[0], error)
            raise
