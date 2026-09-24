import argparse
import copy
import fcntl
import math
import time
from pathlib import Path

from crest.artifacts import digest, file_digest, read_json, write_json
from crest.experiments import implementation_digest
from crest.rq1_protocol import build_dependency_plan, load_pair_artifact, provider_decisions


PHASES = ("baselines", "random_matched", "complete")


class InvocationBudgetReached(RuntimeError):
    pass


def code_identity():
    return {"crest_implementation_sha256": implementation_digest(), "entry_point_sha256": file_digest(__file__)}


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _pinned(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def _read_pin(pin):
    if file_digest(pin["path"]) != pin["sha256"]:
        raise ValueError(f"Pinned input changed: {pin['path']}")
    return read_json(pin["path"])


def _check_file(path, expected):
    if file_digest(path) != expected:
        raise ValueError(f"Input checksum mismatch: {path}")


def _runtime(config, runtime):
    required = {"model_path", "guard_model_path", "device", "guard_device", "dataset_files", "provider_inputs"}
    continuation = config["frozen_settings"].get("baseline:RewriteDetection-Record", {}).get("parameters", {}).get("continuation", {})
    if continuation.get("provider") == "separate_qwen2_5_raw":
        required.add("continuation_model")
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise ValueError("Runtime requires explicit model/guard paths and devices, dataset_files, and provider_inputs")
    if "continuation_model" in required:
        auxiliary = runtime["continuation_model"]
        if (not isinstance(auxiliary, dict) or set(auxiliary) != {"model_path", "revision", "model_identity_sha256", "device"}
                or any(not isinstance(auxiliary[k], str) or not auxiliary[k] for k in auxiliary)
                or any(auxiliary[k] != continuation.get(k) for k in ("revision", "model_identity_sha256"))
                or auxiliary["device"] != runtime["device"]):
            raise ValueError("Separate continuation model requires exact frozen identity and the declared device")
        _check_file(Path(auxiliary["model_path"]) / "model_identity.json", auxiliary["model_identity_sha256"])
    for key in ("model_path", "guard_model_path", "device", "guard_device"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"Runtime {key} must be explicit")
    settings = config["frozen_settings"]
    for key, setting in (("model_path", "backbone"), ("guard_model_path", "baseline:PromptGuard2-Record")):
        parameters = settings[setting]["parameters"]
        for name in ("revision", "model_identity_sha256"):
            if not isinstance(parameters.get(name), str) or not parameters[name]:
                raise ValueError(f"Frozen {setting}.{name} is required")
        _check_file(Path(runtime[key]) / "model_identity.json", parameters["model_identity_sha256"])
    if not isinstance(runtime["dataset_files"], dict) or set(runtime["dataset_files"]) != set(config["datasets"]):
        raise ValueError("Runtime dataset paths must cover exactly the planned datasets")
    if not isinstance(runtime["provider_inputs"], dict) or set(runtime["provider_inputs"]) != set(config["datasets"]):
        raise ValueError("Provider artifact maps must cover exactly the planned datasets")
    required_artifacts = set()
    if "deceptive_text_rewriting" in config["attack_families"]:
        required_artifacts.add("rewrite_cohort")
    if "interaction_history_manipulation" in config["attack_families"]:
        required_artifacts.add("return_graph")
    for dataset in config["datasets"]:
        files = runtime["dataset_files"][dataset]
        if not isinstance(files, dict) or set(files) != {"database_path", "manifest_path", "pool_path"}:
            raise ValueError("Each dataset needs database_path, manifest_path and pool_path")
        for name in ("database", "manifest", "pool"):
            _check_file(files[name + "_path"], config["dataset_inputs"][dataset][name + "_sha256"])
        inputs = runtime["provider_inputs"][dataset]
        if not isinstance(inputs, dict) or set(inputs) != required_artifacts:
            raise ValueError("Provider inputs must supply exactly the applicable real artifacts")
        for artifact in inputs.values():
            if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "manifest_path", "manifest_sha256"}:
                raise ValueError("Provider artifacts require content and source-manifest file pins")
            _check_file(artifact["path"], artifact["sha256"])
            _check_file(artifact["manifest_path"], artifact["manifest_sha256"])


def fixed_baseline_parameters(random_seed_offset):
    from crest import baselines, records
    if isinstance(random_seed_offset, bool) or not isinstance(random_seed_offset, int):
        raise ValueError("random_seed_offset must be an explicit integer")
    record_source = {"module": "crest.records", "functions": ["messages_for", "serialize_records"], "sha256": file_digest(records.__file__)}
    return {
        "Backbone": {"variant": "observed_frozen_ranker", "datamark": False, "source": copy.deepcopy(record_source)},
        "Spotlighting": {"variant": "datamarking", "marker": "^", "whitespace_transform": "str.split", "join_rule": "marker.join(tokens)", "scope": "record_field_values", "system_instruction": "messages_for(datamark=True)", "source": copy.deepcopy(record_source)},
        "Random-Matched": {"algorithm": "bucketed_without_replacement", "matching_keys": ["record_type", "appearance_count", "sorted_record_count_denominators"], "rng": "python.random.Random", "random_seed_offset": random_seed_offset, "empty_omissions_policy": "empty_set", "overlap_policy": "allow", "degeneracy_policy": "report", "source": {"module": "crest.baselines", "function": "random_matched", "sha256": file_digest(baselines.__file__)}},
    }


def _execution_config(config, cell):
    frozen = config["frozen_settings"]
    screening = frozen["screening"]["parameters"]
    selection = frozen["selection"]["parameters"]
    random = frozen["baseline:Random-Matched"]["parameters"]
    for key in ("beta", "penalty", "beam_width", "max_depth"):
        if key not in screening:
            raise ValueError(f"Unresolved screening parameter: {key}")
    for key in ("alpha", "eta"):
        if key not in selection:
            raise ValueError(f"Unresolved selection parameter: {key}")
    if "random_seed_offset" not in random:
        raise ValueError("An explicit Random-Matched random_seed_offset is required")
    expected = fixed_baseline_parameters(random["random_seed_offset"])
    for method, supported in expected.items():
        actual = frozen["baseline:" + method]["parameters"]
        if digest(actual) != digest(supported):
            raise ValueError(f"Frozen {method} parameters contradict the supported fixed adapter or its source hash; use fixed_baseline_parameters with an explicit random_seed_offset")
    result = {"T": config["T"], "K": config["K"], "C": config["candidate_count"], "N": config["N"], "dataset": cell["dataset"], "attack_family": cell["attack_family"], "methods": cell["methods"], **{key: screening[key] for key in ("beta", "penalty", "beam_width", "max_depth")}, **{key: selection[key] for key in ("alpha", "eta")}, "random_seed_offset": random["random_seed_offset"]}
    from crest.calibration import calibrate, promotion_budget
    calibrate([0.0] * result["N"], result["alpha"], result["T"])
    promotion_budget(result["T"], result["eta"])
    for key in ("beta", "penalty"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid frozen {key}")
    for key in ("beam_width", "max_depth"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"Invalid frozen {key}")
    if isinstance(result["random_seed_offset"], bool) or not isinstance(result["random_seed_offset"], int):
        raise ValueError("random_seed_offset must be an integer")
    return result


def _units(plan):
    config = plan["config"]
    units = []
    for cell in plan["matrix"]:
        rows = [row for row in plan["inputs"] if row["dataset"] == cell["dataset"] and row["attack_family"] == cell["attack_family"]]
        groups = config["selection"]["test"]["seed_ids"] if config["calibration_sharing"]["mode"] == "per_seed_group" else [None]
        for group in groups:
            identity = {"dataset": cell["dataset"], "attack_family": cell["attack_family"], "seed_group": group}
            selected = [row for row in rows if group is None or row["seed_group"] == group]
            items = {role: [{"id": row["id"], "seed_group": row["seed_group"], "pair": load_pair_artifact(row)["pair"]} for row in selected if row["split"] == role] for role in ("calibration", "test")}
            if len(items["calibration"]) != config["N"] or not items["test"]:
                raise ValueError("Execution unit violates its explicit calibration sharing allocation")
            units.append({"id": digest(identity), **identity, "config": _execution_config(config, cell), **items})
    return units


def create_plan(config_path, descriptors_path, runtime_path, output_path):
    pins = {name: _pinned(path) for name, path in (("configuration", config_path), ("descriptors", descriptors_path), ("runtime", runtime_path))}
    config, descriptors, runtime = (_read_pin(pins[key]) for key in ("configuration", "descriptors", "runtime"))
    if not isinstance(descriptors, list):
        raise ValueError("Descriptors must be a JSON list")
    plan = build_dependency_plan(config, descriptors)
    _runtime(config, runtime)
    for cell in plan["matrix"]:
        _execution_config(config, cell)
    artifact = {"schema_version": 1, "kind": "rq1_cli_plan", "status": "plan_only", "pins": pins, "code_identity": code_identity(), "dependency_plan": plan, "source_files_rehashed": True, "scope": "Validated dependency plan; execution pending"}
    artifact["sha256"] = digest(artifact)
    output = Path(output_path)
    if output.exists():
        if read_json(output) != artifact:
            raise ValueError("Plan output already belongs to different immutable inputs")
    else:
        write_json(output, artifact)
    return artifact


def _read_plan(path):
    artifact = read_json(path)
    if artifact.get("kind") != "rq1_cli_plan" or artifact.get("sha256") != digest({key: value for key, value in artifact.items() if key != "sha256"}):
        raise ValueError("RQ1 CLI plan checksum mismatch")
    if artifact["code_identity"] != code_identity():
        raise ValueError("Implementation changed after planning; preserve old outputs and create a new plan")
    plan = artifact["dependency_plan"]
    if plan.get("plan_sha256") != digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("Dependency plan checksum mismatch")
    return artifact


def _verified_inputs(artifact):
    config, descriptors, runtime = (_read_pin(artifact["pins"][key]) for key in ("configuration", "descriptors", "runtime"))
    rebuilt = build_dependency_plan(config, descriptors)
    if rebuilt != artifact["dependency_plan"]:
        raise ValueError("Revalidated dependency plan differs from the frozen plan")
    _runtime(config, runtime)
    return config, runtime, _units(rebuilt)


def _providers(factory, unit, config, runtime, output):
    family = unit["attack_family"]
    needed = {"deceptive_text_rewriting": ["rewrite_cohort"], "interaction_history_manipulation": ["return_graph"]}.get(family, [])
    kwargs = {"dataset": unit["dataset"], "attack_family": family, "model_path": runtime["model_path"], "revision": config["frozen_settings"]["backbone"]["parameters"]["revision"], "rank_cache_path": str(output / "caches" / (unit["id"] + "-rank.sqlite")), "device": runtime["device"], "guard_model_path": runtime["guard_model_path"], "guard_revision": config["frozen_settings"]["baseline:PromptGuard2-Record"]["parameters"]["revision"], "guard_cache_path": str(output / "caches" / (unit["id"] + "-guard.sqlite")), "guard_device": runtime["guard_device"], "provider_inputs": {name: runtime["provider_inputs"][unit["dataset"]][name] for name in needed}, "observed_sequences": [provider_decisions(item["pair"]) for item in unit["calibration"] + unit["test"]]}
    if "continuation_model" in runtime:
        kwargs["continuation_model"] = runtime["continuation_model"]
    bundle = factory(copy.deepcopy(config["frozen_settings"]), **copy.deepcopy(kwargs))
    if not callable(getattr(bundle, "close", None)) or not hasattr(bundle, "ranker") or not isinstance(getattr(bundle, "detectors", None), dict):
        raise ValueError("Provider factory must return ranker, detectors, and close() without loading models")
    return bundle


def _check_saved_outputs(state, output):
    for phase in PHASES:
        for unit, pin in state["phase_results"][phase].items():
            value = _read_pin(pin)
            if value.get("phase") != phase or value.get("unit_id") != unit or value.get("run_fingerprint") != state["fingerprint"] or value.get("result_sha256") != digest(value.get("result")):
                raise ValueError("Corrupt saved CLI phase result")
    for manifest_path in (output / "cells").glob("*/manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") != digest(manifest.get("immutable")):
            raise ValueError("Corrupt executor cell manifest")
        for stage_path in (manifest_path.parent / "stages").glob("*.json"):
            stage = read_json(stage_path)
            if stage.get("fingerprint") != manifest["fingerprint"] or stage.get("result_sha256") != digest(stage.get("result")):
                raise ValueError("Corrupt executor stage checkpoint")


def _write_state(path, state, wall_clock):
    state["updated_unix"] = wall_clock()
    state["state_sha256"] = digest({key: value for key, value in state.items() if key != "state_sha256"})
    write_json(path, state)


def run_plan(plan_path, output_dir, *, invocation_budget_seconds, unit_id=None, max_phase="complete", factory=None, executor=None, clock=time.monotonic, wall_clock=time.time):
    if max_phase not in PHASES:
        raise ValueError("Unknown requested RQ1 phase")
    if unit_id is not None and (not isinstance(unit_id, str) or not unit_id):
        raise ValueError("Queue execution requires an explicit unit identity")
    requested_phases = PHASES[:PHASES.index(max_phase) + 1]
    invocation_limit = _positive_seconds(invocation_budget_seconds, "invocation_budget_seconds")
    started = clock()
    artifact = _read_plan(plan_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"plan_sha256": artifact["sha256"], "output": str(output), "code_identity": artifact["code_identity"]}
    if unit_id is not None:
        identity["execution_scope"] = {"unit_id": unit_id, "cross_unit_barrier_owner": "external_durable_queue"}
    fingerprint = digest(identity)
    total_budget = artifact["dependency_plan"]["config"]["time_budget_seconds"]
    status_path = output / "status.json"
    with (output / ".rq1.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another RQ1 invocation owns this output directory") from error
        if status_path.exists():
            state = read_json(status_path)
            if state.get("fingerprint") != fingerprint or state.get("state_sha256") != digest({key: value for key, value in state.items() if key != "state_sha256"}):
                raise ValueError("Run status belongs to different inputs or has been changed")
            _check_saved_outputs(state, output)
            if state["status"] == "failed":
                raise ValueError("A failed invocation is held for diagnosis; failed work is not automatically retried")
            for attempt in state["attempts"]:
                if attempt["status"] == "running":
                    attempt.update(status="unclean_termination", elapsed_seconds=max(attempt["reserved_seconds"], attempt.get("elapsed_seconds", 0.0)), charging_policy="Entire reserved budget charged after unclean termination")
        else:
            state = {"schema_version": 1, "kind": "rq1_cli_run", "fingerprint": fingerprint, "plan_sha256": artifact["sha256"], "status": "pending", "phase_results": {phase: {} for phase in PHASES}, "attempts": [], "time_budget_seconds": total_budget, "hard_timeout_contract": "Budgets are cooperative between calls; caller must enforce an external hard timeout for an in-flight GPU call"}
        settled = sum(attempt["elapsed_seconds"] for attempt in state["attempts"])
        remaining = total_budget - settled
        if remaining <= 0:
            if state["status"] not in ("complete", "complete_with_failures"):
                state["status"] = "budget_exhausted"
            state["cumulative_seconds"] = settled
            _write_state(status_path, state, wall_clock)
            return state
        reservation = min(invocation_limit, remaining)
        attempt = {"number": len(state["attempts"]) + 1, "status": "running", "started_unix": wall_clock(), "reserved_seconds": reservation, "elapsed_seconds": 0.0, "active_phase": None, "active_unit": None}
        attempt["requested_max_phase"] = max_phase
        if unit_id is not None:
            state["execution_scope"] = identity["execution_scope"]
        state["attempts"].append(attempt)
        state["status"] = "running"
        def persist():
            attempt["elapsed_seconds"] = max(0.0, clock() - started)
            state["cumulative_seconds"] = settled + attempt["elapsed_seconds"]
            _write_state(status_path, state, wall_clock)
        def check():
            elapsed = clock() - started
            if elapsed >= reservation:
                raise InvocationBudgetReached("Invocation or frozen cumulative time budget reached")
            return reservation - elapsed
        persist()
        try:
            config, runtime, units = _verified_inputs(artifact)
            if unit_id is not None:
                units = [unit for unit in units if unit["id"] == unit_id]
                if len(units) != 1:
                    raise ValueError("Requested execution unit is absent from the immutable plan")
            check()
            if factory is None:
                from crest.rq1_providers import build_rq1_providers
                factory = build_rq1_providers
            if executor is None:
                from crest.rq1_execution import execute_cell
                executor = execute_cell
            unit_ids = {unit["id"] for unit in units}
            if any(not set(state["phase_results"][phase]) <= unit_ids for phase in PHASES):
                raise ValueError("Run state contains an unknown execution unit")
            for index, phase in enumerate(PHASES[1:], 1):
                if state["phase_results"][phase] and set(state["phase_results"][PHASES[index - 1]]) != unit_ids:
                    raise ValueError("Run state violates the global phase barrier")
            def completed_status():
                if all(set(state["phase_results"][phase]) == unit_ids for phase in PHASES):
                    return "complete_with_failures" if state.get("contains_failed_results") else "complete"
                return "paused_after_" + max_phase
            if all(set(state["phase_results"][phase]) == unit_ids for phase in requested_phases):
                state["status"] = completed_status()
                attempt["status"] = "verified_complete_replay"
                persist()
                return state
            (output / "caches").mkdir(parents=True, exist_ok=True)
            for unit in units:
                check()
                bundle = _providers(factory, unit, config, runtime, output)
                bundle.close()
            for phase in requested_phases:
                for unit in units:
                    if unit["id"] in state["phase_results"][phase]:
                        continue
                    check()
                    if code_identity() != artifact["code_identity"]:
                        raise ValueError("Implementation changed during execution")
                    attempt.update(active_phase=phase, active_unit=unit["id"])
                    persist()
                    bundle = _providers(factory, unit, config, runtime, output)
                    try:
                        budget = check()
                        result = executor(copy.deepcopy(unit["config"]), copy.deepcopy(unit["calibration"]), copy.deepcopy(unit["test"]), bundle.ranker, bundle.detectors, output / "cells" / unit["id"], validated_plan_sha256=artifact["dependency_plan"]["plan_sha256"], budget_seconds=budget, max_phase=phase, clock=clock)
                    finally:
                        bundle.close()
                    result_state = result.get("status")
                    if result_state == "partial":
                        raise InvocationBudgetReached("Cell returned a resumable partial result")
                    expected = {"baselines": "paused_after_baselines", "random_matched": "paused_after_random_matched"}
                    if phase in expected and result_state != expected[phase]:
                        raise ValueError("Executor did not complete the requested phase boundary")
                    state["contains_failed_results"] = state.get("contains_failed_results", False) or bool(result.get("contains_failed_results"))
                    if phase == "complete":
                        if not isinstance(result.get("rows"), list) or len(result["rows"]) != len(unit["test"]):
                            raise ValueError("Executor did not return every planned test sequence")
                        if any(row.get("id") not in {item["id"] for item in unit["test"]} for row in result["rows"]) or len({row["id"] for row in result["rows"]}) != len(result["rows"]):
                            raise ValueError("Executor returned incorrect or duplicate sequence identities")
                        if any(not isinstance(row.get("methods"), dict) or set(row["methods"]) != set(unit["config"]["methods"]) for row in result["rows"]):
                            raise ValueError("Executor result omits an applicable method")
                        state["contains_failed_results"] = state.get("contains_failed_results", False) or any(value.get("status") != "ok" for row in result["rows"] for value in row["methods"].values()) or any(row.get("clean_reference_status") != "ok" for row in result["rows"]) or result.get("calibration", {}).get("status") != "ok"
                    value = {"phase": phase, "unit_id": unit["id"], "run_fingerprint": fingerprint, "result": result, "result_sha256": digest(result)}
                    path = output / "phase_results" / phase / (unit["id"] + ".json")
                    if path.exists() and read_json(path) != value:
                        raise ValueError("Existing phase result differs; do not overwrite completed work")
                    write_json(path, value)
                    state["phase_results"][phase][unit["id"]] = _pinned(path)
                    persist()
            attempt["status"] = "complete"
            state["status"] = completed_status()
            persist()
            return state
        except InvocationBudgetReached as error:
            attempt.update(status="partial", error=str(error))
            state["status"] = "budget_exhausted" if settled + clock() - started >= total_budget else "partial"
            persist()
            return state
        except Exception as error:
            attempt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            state["status"] = "failed"
            persist()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan or execute RQ1 experiments.", epilog="Run budgets are cooperative. Use an external hard timeout for an in-flight GPU call. Inputs and frozen settings must already be complete.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate inputs and write an execution plan")
    for name in ("config", "descriptors", "runtime", "output"):
        plan.add_argument("--" + name, required=True)
    run = commands.add_parser("run", help="Verify the plan and resume globally ordered execution within both budgets")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--invocation-budget-seconds", required=True, type=float)
    run.add_argument("--unit-id", help="Execute only this immutable unit in its own output directory; the external durable queue must enforce cross-unit phase barriers")
    run.add_argument("--max-phase", choices=PHASES, default="complete", help="Stop successfully after this phase; subsequent invocations preserve the same output and cache identity")
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = create_plan(args.config, args.descriptors, args.runtime, args.output)
        print(f"Plan ready: {value['sha256']}")
        return 0
    value = run_plan(args.plan, args.output, invocation_budget_seconds=args.invocation_budget_seconds, unit_id=args.unit_id, max_phase=args.max_phase)
    print(f"RQ1 status: {value['status']}; cumulative seconds: {value['cumulative_seconds']:.3f}")
    return 2 if value["status"] in ("partial", "budget_exhausted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
