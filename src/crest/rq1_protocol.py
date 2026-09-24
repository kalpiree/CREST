import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path

from .artifacts import digest, file_digest
from .records import validate_decision


DATASETS = ("amazon-2018-all-beauty", "steam", "movielens-1m")
ATTACK_FAMILIES = (
    "instruction_injection", "deceptive_text_rewriting", "interaction_history_manipulation"
)
COMMON_METHODS = ("Backbone", "Spotlighting", "PromptGuard2-Record")
CONTENT_FIELDS = frozenset(("title", "description", "text", "content", "review", "review_text", "body", "summary"))
SOURCE_KEYS = ("database_sha256", "manifest_sha256", "pool_sha256", "source_split")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an explicit integer >= {minimum}")
    return value


def _sha(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name} must be a lowercase SHA256")
    return value


def _nonempty(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty")
    return value


def _resolved(value, name):
    if value is None or (isinstance(value, str) and value.strip().lower() in {
        "unresolved", "todo", "tbd", "pending", "unknown"
    }):
        raise ValueError(f"{name} is unresolved")
    if isinstance(value, dict):
        for key, child in value.items():
            _resolved(child, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for child in value:
            _resolved(child, name)
    digest(value)


def _unique_strings(values, name, allowed=None):
    _require(isinstance(values, list) and bool(values), f"{name} must be a nonempty list")
    for value in values:
        _nonempty(value, name)
    _require(len(values) == len(set(values)), f"{name} contains duplicates")
    if allowed is not None:
        _require(set(values) <= set(allowed), f"Unsupported {name}")
    return values


def rq1_matrix(datasets, attack_families):
    _unique_strings(datasets, "datasets", DATASETS)
    _unique_strings(attack_families, "attack_families", ATTACK_FAMILIES)
    cells = []
    for dataset in datasets:
        for family in attack_families:
            extra = {"deceptive_text_rewriting": ["RewriteDetection-Record"],
                     "interaction_history_manipulation": ["RETURN-Del"]}.get(family, [])
            cells.append({"dataset": dataset, "attack_family": family,
                          "methods": list(COMMON_METHODS) + extra + ["Random-Matched", "CREST"]})
    return cells


def provider_decisions(pair):
    result = []
    for decision in pair["observed"]:
        validate_decision(decision)
        result.append({"user_id": decision["user_id"], "candidates": list(decision["candidates"]),
                       "records": [{"id": r["id"], "type": r["type"], "fields": copy.deepcopy(r["fields"])}
                                   for r in decision["records"]]})
    return result


def _timestamp(value):
    _require(not isinstance(value, bool), "Invalid timestamp")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _require(isinstance(value, str), "Timestamp must be numeric or timezone-aware ISO8601")
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require(stamp.tzinfo is not None, "Timestamp must include timezone")
        parsed = stamp.timestamp()
    _require(math.isfinite(parsed), "Timestamp must be finite")
    return parsed


def _operation(index, record, previous=None):
    fields = list(record["fields"]) if previous is None else [
        key for key in previous["fields"] if previous["fields"][key] != record["fields"][key]
    ]
    return {"decision_index": index, "record_id": record["id"],
            "kind": "insert" if previous is None else "edit",
            "before_sha256": None if previous is None else digest(previous),
            "after_sha256": digest(record), "changed_fields": fields}


def validate_attack_pair(pair, reference, audit, *, dataset, attack_family, ingestion=None):
    _require(dataset in DATASETS and attack_family in ATTACK_FAMILIES, "Unsupported RQ1 cell")
    _require(pair.get("attack", {}).get("name") == attack_family, "Pair attack name differs from audited family")
    _integer(reference["seed"], "reference.seed")
    _require(type(pair["seed"]) is int and pair["seed"] == reference["seed"], "Sequence seed changed")
    original = reference["clean"]
    _require(isinstance(original, list) and bool(original), "Empty original reference")
    _require(pair["clean"] == original, "Clean reference must restore edits and exclude insertions")
    observed = pair["observed"]
    _require(isinstance(observed, list) and len(observed) == len(original), "Sequence length changed")
    positives = reference["evaluation"]["positives"]
    _require(isinstance(positives, list) and len(positives) == len(original), "Positive count differs from sequence length")
    target = _nonempty(reference["evaluation"]["target"], "target")
    for name in ("positives", "target"):
        _require(pair["evaluation"][name] == reference["evaluation"][name], f"Evaluation {name} changed")
    _require(audit.get("status") == "validated", "Attack audit is not validated")
    _require(audit.get("dataset") == dataset and audit.get("attack_family") == attack_family, "Attack audit cell mismatch")
    _require(audit.get("reference_digest") == digest(reference) and audit.get("pair_digest") == digest(pair),
             "Attack audit digest mismatch")
    _require(audit.get("selection_frozen_before_feedback") is True and audit.get("include_unsuccessful_attempts") is True,
             "Attack audit must retain preselected unsuccessful attempts")
    generator = audit.get("generator", {})
    _nonempty(generator.get("name"), "generator.name")
    for name in ("implementation_sha256", "settings_sha256"):
        _sha(generator.get(name), "generator." + name)
    permitted = audit.get("permitted_edit_fields")
    _require(isinstance(permitted, dict), "Missing explicit permitted_edit_fields")
    for record_type, fields in permitted.items():
        _nonempty(record_type, "permitted record type")
        _require(isinstance(fields, list) and len(fields) == len(set(fields)) and set(fields) <= CONTENT_FIELDS,
                 "Only explicitly permitted content fields may be edited")
    original_ids = {r["id"] for d in original for r in d["records"]}
    inserted_records = {}
    if ingestion is not None:
        _require(ingestion.get("status") == "validated" and ingestion.get("assigned_by") == "trusted_ingestion",
                 "Inserted identities need trusted ingestion provenance")
        _require(isinstance(ingestion.get("records"), dict), "Missing trusted ingestion records")
        inserted_records = ingestion["records"]
    actual, seen_insertions = [], set()
    for index, (clean, current) in enumerate(zip(original, observed)):
        validate_decision(clean)
        validate_decision(current)
        _nonempty(clean.get("user_id"), "user_id")
        _require(positives[index] in clean["candidates"] and target in clean["candidates"],
                 "Positive and frozen target must occur in fixed candidates")
        _require({k: v for k, v in clean.items() if k != "records"} ==
                 {k: v for k, v in current.items() if k != "records"},
                 "Candidates, task metadata, or original references changed")
        originals = {r["id"]: r for r in clean["records"]}
        _require([r["id"] for r in current["records"] if r["id"] in originals] == list(originals),
                 "Original records were removed or reordered")
        for original_record, pair_clean_record in zip(clean["records"], pair["clean"][index]["records"]):
            _require(list(original_record["fields"]) == list(pair_clean_record["fields"]), "Clean field ordering changed")
        for record in current["records"]:
            previous = originals.get(record["id"])
            if previous is None:
                _require(attack_family == "interaction_history_manipulation", "Insertions require history manipulation")
                _require(record["id"] not in original_ids, "Inserted ID collides with an original identity")
                _require(record["type"] == "interaction", "Only interaction insertions are permitted")
                _require(inserted_records.get(record["id"]) == record, "Inserted record differs from trusted ingestion receipt")
                fields = record["fields"]
                _require(fields.get("user_id") == clean["user_id"], "Inserted history belongs to a different user")
                _nonempty(fields.get("item_id"), "inserted item_id")
                cutoff = clean.get("cutoff_unix", clean.get("cutoff"))
                _require(cutoff is not None and "timestamp" in fields, "Insertion needs trusted cutoff and timestamp")
                _require(_timestamp(fields["timestamp"]) < _timestamp(cutoff), "Inserted event is not strictly before cutoff")
                actual.append(_operation(index, record))
                seen_insertions.add(record["id"])
                continue
            _require({k: v for k, v in record.items() if k != "fields"} ==
                     {k: v for k, v in previous.items() if k != "fields"}, "Original record identity/type metadata changed")
            _require(list(record["fields"]) == list(previous["fields"]), "Original field names/order changed")
            changed = [key for key in previous["fields"] if previous["fields"][key] != record["fields"][key]]
            if changed:
                _require(attack_family != "interaction_history_manipulation", "History attack cannot edit original events")
                _require(set(changed) <= set(permitted.get(record["type"], [])), "Changed field is not permitted content")
                actual.append(_operation(index, record, previous))
    _require(set(inserted_records) == seen_insertions, "Ingestion receipt has unused or missing inserted identities")
    declared = audit.get("operations")
    _require(isinstance(declared, list), "Missing audited operations")
    actual_map = {(row["decision_index"], row["record_id"]): row for row in actual}
    declared_map = {}
    for row in declared:
        _integer(row["decision_index"], "operation decision index")
        key = (row["decision_index"], row["record_id"])
        _require(key not in declared_map, "Duplicate audited operation")
        declared_map[key] = row
    _require(declared_map == actual_map, "Audited operations do not match actual record changes")
    for key, values in (("attacked_ids", {r["record_id"] for r in actual}),
                        ("affected_decisions", {r["decision_index"] for r in actual})):
        declared_values = pair["evaluation"][key]
        expected_type = str if key == "attacked_ids" else int
        _require(isinstance(declared_values, list) and all(type(value) is expected_type for value in declared_values),
                 f"Evaluation {key} has invalid label types")
        _require(isinstance(declared_values, list) and len(declared_values) == len(set(declared_values)) and
                 set(declared_values) == values, f"Evaluation {key} does not match actual changes")
    attempt = audit.get("attempt_status")
    _require(attempt in ("applied", "no_change", "generation_failed"), "Unknown attack attempt status")
    _require((attempt == "applied") == bool(actual), "Attempt status contradicts actual changes")
    return {"status": "validated", "dataset": dataset, "attack_family": attack_family,
            "reference_digest": digest(reference), "pair_digest": digest(pair), "audit_digest": digest(audit),
            "ingestion_digest": None if ingestion is None else digest(ingestion),
            "operations": len(actual), "manipulated_identities": len({r["record_id"] for r in actual}),
            "attempt_status": attempt, "T": len(original)}


def _read_pinned(path, expected):
    _sha(expected, "input SHA256")
    raw = Path(path).read_bytes()
    import hashlib
    _require(hashlib.sha256(raw).hexdigest() == expected, f"Input checksum mismatch: {path}")
    return json.loads(raw)


def load_pair_artifact(descriptor):
    _validate_descriptor(descriptor)
    values = {name: _read_pinned(descriptor[f"{name}_path"], descriptor[f"{name}_sha256"])
              for name in ("pair", "reference", "audit")}
    if "ingestion_path" in descriptor or "ingestion_sha256" in descriptor:
        values["ingestion"] = _read_pinned(descriptor["ingestion_path"], descriptor["ingestion_sha256"])
    validation = validate_attack_pair(**values, dataset=descriptor["dataset"], attack_family=descriptor["attack_family"])
    _require(values["pair"]["seed"] == descriptor["sequence_seed"], "Descriptor sequence seed differs from pair")
    return dict(values, validation=validation, descriptor=copy.deepcopy(descriptor))


def _validate_descriptor(row):
    for name in ("id", "source_sequence_id"):
        _nonempty(row.get(name), name)
    _integer(row.get("sequence_seed"), "sequence_seed")
    _integer(row.get("seed_group"), "seed_group")
    _require(row.get("dataset") in DATASETS and row.get("attack_family") in ATTACK_FAMILIES, "Unsupported artifact cell")
    _require(row.get("split") in ("calibration", "test"), "Artifact split must be calibration or test")
    for name in ("pair", "reference", "audit"):
        _nonempty(row.get(name + "_path"), name + "_path")
        _sha(row.get(name + "_sha256"), name + "_sha256")
    source = row.get("source", {})
    for key in SOURCE_KEYS[:-1]:
        _sha(source.get(key), "source." + key)
    _nonempty(source.get("source_split"), "source.source_split")


def select_artifacts(descriptors, *, dataset, attack_family, split, seed_ids, per_seed, subset_seed):
    _require(dataset in DATASETS and attack_family in ATTACK_FAMILIES, "Unsupported selection cell")
    _require(split in ("calibration", "test"), "Invalid selection split")
    _require(isinstance(seed_ids, list) and bool(seed_ids), "Explicit seed_ids required")
    for seed in seed_ids:
        _integer(seed, "seed_id")
    _require(len(seed_ids) == len(set(seed_ids)), "Duplicate seed IDs")
    _integer(per_seed, "per_seed", 1)
    _integer(subset_seed, "subset_seed")
    ids, sources, groups = set(), set(), {seed: [] for seed in seed_ids}
    for row in descriptors:
        _validate_descriptor(row)
        _require(row["id"] not in ids, "Duplicate artifact ID")
        ids.add(row["id"])
        if (row["dataset"], row["attack_family"], row["split"]) != (dataset, attack_family, split):
            continue
        key = (row["seed_group"], row["source_sequence_id"])
        _require(key not in sources, "Duplicate source sequence could select between attack attempts")
        sources.add(key)
        if row["seed_group"] in groups:
            groups[row["seed_group"]].append(row)
    selected = []
    for seed in seed_ids:
        _require(len(groups[seed]) >= per_seed, f"Insufficient {split} artifacts for {dataset}/{attack_family}/seed {seed}")
        ordered = sorted(groups[seed], key=lambda row: (digest({"subset_seed": subset_seed,
            "source_sequence_id": row["source_sequence_id"], "sequence_seed": row["sequence_seed"]}), row["source_sequence_id"]))
        selected.extend(copy.deepcopy(ordered[:per_seed]))
    return selected


def _validate_config(config):
    _resolved(config, "config")
    cells = rq1_matrix(config["datasets"], config["attack_families"])
    for name in ("T", "K", "candidate_count", "N", "time_budget_seconds"):
        _integer(config.get(name), name, 1)
    _require(config["K"] <= config["candidate_count"], "K exceeds candidate_count")
    selection = config["selection"]
    _integer(selection.get("subset_seed"), "selection.subset_seed")
    for role in ("calibration", "test"):
        value = selection[role]
        _integer(value.get("per_seed"), role + ".per_seed", 1)
        _require(isinstance(value.get("seed_ids"), list) and bool(value["seed_ids"]), "Explicit seed IDs required")
        for seed in value["seed_ids"]:
            _integer(seed, "seed_id")
        _require(len(set(value["seed_ids"])) == len(value["seed_ids"]), "Duplicate seed IDs")
    sharing = config["calibration_sharing"]
    mode = sharing.get("mode")
    if mode == "per_seed_group":
        _require(set(selection["calibration"]["seed_ids"]) == set(selection["test"]["seed_ids"]),
                 "Per-seed calibration requires matching seed groups")
        _require(selection["calibration"]["per_seed"] == config["N"], "N must equal each calibration seed-group count")
    elif mode == "shared_across_seed_groups":
        provenance = sharing.get("provenance")
        _require(isinstance(provenance, dict) and bool(provenance) and sharing.get("sha256") == digest(provenance),
                 "Shared calibration needs explicit hashed provenance")
        _require(selection["calibration"]["per_seed"] * len(selection["calibration"]["seed_ids"]) == config["N"],
                 "N must equal selected shared calibration count per cell")
    else:
        raise ValueError("Explicit calibration_sharing mode required")
    for dataset in config["datasets"]:
        source = config["dataset_inputs"][dataset]
        _require(set(source) == set(SOURCE_KEYS), "Dataset input pins must explicitly identify database, manifest, pool and split")
        for key in SOURCE_KEYS[:-1]:
            _sha(source[key], key)
        _nonempty(source["source_split"], "source_split")
    required = {"backbone", "sequence_protocol", "screening", "selection", "aggregation"}
    required.update("attack:" + family for family in config["attack_families"])
    required.update("baseline:" + method for cell in cells for method in cell["methods"] if method != "CREST")
    settings = config["frozen_settings"]
    for key in sorted(required):
        value = settings.get(key, {})
        _require(value.get("status") == "frozen", f"Unfrozen or missing setting: {key}")
        content = {name: value.get(name) for name in ("parameters", "provenance")}
        _require(all(isinstance(item, dict) and bool(item) for item in content.values()), f"Missing explicit parameters/provenance: {key}")
        _require(value.get("sha256") == digest(content), f"Frozen setting checksum mismatch: {key}")
    return cells


def build_dependency_plan(config, descriptors):
    cells = _validate_config(config)
    for pin in config.get("verify_input_files", []):
        _sha(pin["sha256"], "additional input SHA256")
        _require(file_digest(pin["path"]) == pin["sha256"], f"Source input checksum mismatch: {pin['path']}")
    selected, loaded = {}, {}
    for cell in cells:
        cell_key = (cell["dataset"], cell["attack_family"])
        selected[cell_key] = {}
        for role in ("calibration", "test"):
            rows = select_artifacts(descriptors, **{k: cell[k] for k in ("dataset", "attack_family")},
                                    split=role, subset_seed=config["selection"]["subset_seed"], **config["selection"][role])
            selected[cell_key][role] = rows
            for row in rows:
                _require(row["source"] == config["dataset_inputs"][row["dataset"]], "Artifact source pins differ from frozen dataset inputs")
                bundle = load_pair_artifact(row)
                _require(bundle["validation"]["T"] == config["T"], "Artifact T differs from frozen setting")
                _require(all(len(d["candidates"]) == config["candidate_count"] for d in bundle["pair"]["clean"]),
                         "Artifact candidate count differs from frozen setting")
                expected_generator = config["frozen_settings"]["attack:" + row["attack_family"]]["sha256"]
                _require(bundle["audit"]["generator"]["settings_sha256"] == expected_generator,
                         "Attack generator settings differ from frozen protocol")
                loaded[row["id"]] = bundle
        calibration_sources = {row["source_sequence_id"] for row in selected[cell_key]["calibration"]}
        _require(not calibration_sources.intersection(row["source_sequence_id"] for row in selected[cell_key]["test"]),
                 "Calibration and test reuse the same source sequence")
        calibration_seeds = {row["sequence_seed"] for row in selected[cell_key]["calibration"]}
        _require(not calibration_seeds.intersection(row["sequence_seed"] for row in selected[cell_key]["test"]),
                 "Calibration and test reuse a sequence seed")
    nodes = []
    def add(phase, action, dependencies=(), rows=(), cell=None, seed_group=None, method=None, access=()):
        identity = {"phase": phase, "action": action, "artifact_ids": [r["id"] for r in rows],
                    "dataset": None if cell is None else cell["dataset"],
                    "attack_family": None if cell is None else cell["attack_family"],
                    "seed_group": seed_group, "method": method}
        node_id = phase + ":" + digest(identity)[:24]
        nodes.append(dict(identity, id=node_id, depends_on=list(dict.fromkeys(dependencies)),
                          input_access=list(access), status="pending"))
        return node_id
    clean_nodes, baseline_nodes, screen_nodes, calibration_nodes, random_nodes, final_nodes = {}, [], {}, {}, [], []
    for cell in cells:
        for role in ("calibration", "test"):
            for row in selected[(cell["dataset"], cell["attack_family"])][role]:
                clean_nodes[row["id"]] = add("clean_reference", "rank_clean_for_evaluator", rows=[row], cell=cell,
                                            seed_group=row["seed_group"], access=["clean_decisions_evaluator_only"])
    test_clean_nodes = [clean_nodes[row["id"]] for cell in cells
                        for row in selected[(cell["dataset"], cell["attack_family"])]["test"]]
    clean_barrier = add("barrier", "all_test_clean_references_complete", test_clean_nodes)
    for cell in cells:
        for row in selected[(cell["dataset"], cell["attack_family"])]["test"]:
            for method in cell["methods"]:
                if method in ("Random-Matched", "CREST"):
                    continue
                baseline_nodes.append(add("independent_baselines", "run_frozen_baseline", [clean_barrier], [row],
                    cell, row["seed_group"], method, ["observed_provider_decisions", "frozen_method_settings"]))
    independent_barrier = add("barrier", "all_independent_baselines_complete", baseline_nodes)
    for cell in cells:
        cell_rows = selected[(cell["dataset"], cell["attack_family"])]
        for role in ("calibration", "test"):
            for row in cell_rows[role]:
                screen_nodes[row["id"]] = add("crest_screening", "save_frozen_screening_without_final_selector",
                    [independent_barrier], [row], cell, row["seed_group"], "CREST-screening",
                    ["observed_provider_decisions", "frozen_screening_settings"])
        groups = config["selection"]["test"]["seed_ids"] if config["calibration_sharing"]["mode"] == "per_seed_group" else [None]
        for group in groups:
            rows = [r for r in cell_rows["calibration"] if group is None or r["seed_group"] == group]
            deps = [screen_nodes[r["id"]] for r in rows] + [clean_nodes[r["id"]] for r in rows]
            calibration_nodes[(cell["dataset"], cell["attack_family"], group)] = add("crest_calibration", "calibrate_frozen_cell",
                deps, rows, cell, group, "CREST-calibration", ["saved_calibration_screening", "clean_rankings_evaluator_only"])
        for row in cell_rows["test"]:
            random_nodes.append(add("random_matched", "run_random_matched_from_saved_omissions",
                [independent_barrier, screen_nodes[row["id"]]], [row], cell, row["seed_group"], "Random-Matched",
                ["observed_provider_decisions", "saved_screening_omissions", "frozen_method_settings"]))
    all_baselines = add("barrier", "all_baselines_complete", [independent_barrier] + random_nodes)
    aggregates = []
    for cell in cells:
        cell_rows = selected[(cell["dataset"], cell["attack_family"])]
        for group in config["selection"]["test"]["seed_ids"]:
            rows = [r for r in cell_rows["test"] if r["seed_group"] == group]
            current_finals = []
            for row in rows:
                cal_group = group if config["calibration_sharing"]["mode"] == "per_seed_group" else None
                cal = calibration_nodes[(cell["dataset"], cell["attack_family"], cal_group)]
                node = add("crest_final", "select_from_frozen_screening_and_calibration",
                    [all_baselines, screen_nodes[row["id"]], cal], [row], cell, group, "CREST",
                    ["saved_test_screening", "frozen_calibration_threshold", "frozen_selection_settings"])
                current_finals.append(node)
                final_nodes.append(node)
            aggregates.append(add("aggregate", "evaluate_cell_seed_common_returned_and_all_attempted",
                [all_baselines] + current_finals + [clean_nodes[r["id"]] for r in rows], rows, cell, group,
                access=["all_method_results", "evaluation_labels", "clean_rankings_evaluator_only", "frozen_aggregation_settings"]))
    add("aggregate", "aggregate_rq1_matrix", aggregates, access=["cell_seed_aggregates", "frozen_aggregation_settings"])
    pins = [dict(bundle["descriptor"], validation=bundle["validation"]) for _, bundle in sorted(loaded.items())]
    plan = {"schema_version": 1, "kind": "rq1_dependency_plan", "status": "plan_only",
            "protocol_implementation_sha256": file_digest(__file__), "config": copy.deepcopy(config),
            "config_sha256": digest(config), "matrix": cells, "inputs": pins, "nodes": nodes,
            "requested_test_sequences": sum(len(r["test"]) for r in selected.values()),
            "requested_calibration_sequences": sum(len(r["calibration"]) for r in selected.values()),
            "source_database_bytes_reverified": False,
            "time_budget_contract": "Executor must enforce this explicit total budget; planning launches no tasks."}
    plan["plan_sha256"] = digest(plan)
    return plan


def ready_nodes(plan, statuses):
    _require(plan.get("kind") == "rq1_dependency_plan", "Not an RQ1 dependency plan")
    _require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "Plan checksum mismatch")
    nodes = {node["id"]: node for node in plan["nodes"]}
    _require(len(nodes) == len(plan["nodes"]), "Duplicate phase node IDs")
    _require(set(statuses) <= set(nodes), "Unknown node status")
    _require(set(statuses.values()) <= {"pending", "running", "complete", "failed"}, "Unknown phase status")
    _require("failed" not in statuses.values(), "A failed phase holds the plan; failed work is never automatically retried")
    return [copy.deepcopy(node) for node in plan["nodes"]
            if statuses.get(node["id"], "pending") == "pending" and
            all(statuses.get(dependency) == "complete" for dependency in node["depends_on"])]
