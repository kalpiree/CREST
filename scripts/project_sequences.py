import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "rq1-reduced-budget-projection-v1"
FAMILIES = ("instruction_injection", "deceptive_text_rewriting", "interaction_history_manipulation")
SEEDS = (11, 23, 37, 53, 71)
POLICY = {"schema": SCHEMA, "N": 50, "T": 20, "C": 50, "K": 5,
          "source_T": 100, "source_K": 10, "source_calibration_N": 199, "subset_seed": 8675309,
          "clean_source_indices": list(range(50)), "attacked_source_indices": list(range(50, 100)),
          "decisions_per_stratum": 10, "test_sequences": 10, "test_seed_groups": list(SEEDS),
          "selection": "Native SHA256 cohort order; source-only repeated-user anchor in attacked half, SHA256 remaining indices, then chronological order",
          "outcome_selection": False, "rankings_read": False, "model_calls": 0}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def read_pin(value):
    if set(value) != {"path", "sha256"} or file_hash(value["path"]) != value["sha256"]:
        raise ValueError("Original input pin changed")
    return json.loads(Path(value["path"]).read_text())


def once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Immutable projection differs: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return pin(path)


def source_key(row):
    return (row["split"], row["seed_group"], row["sequence_seed"], row["source_sequence_id"])


def select_cohort(rows, dataset):

    from crest.rq1_protocol import _validate_descriptor, select_artifacts
    ids = set()
    for row in rows:
        _validate_descriptor(row)
        if row["id"] in ids:
            raise ValueError("Duplicate source artifact ID")
        ids.add(row["id"])
    base = [r for r in rows if r["dataset"] == dataset and r["attack_family"] == "instruction_injection"]
    calibration = [r for r in base if r["split"] == "calibration"]
    tests = [r for r in base if r["split"] == "test"]
    if len(calibration) != 199 or {r["seed_group"] for r in calibration} != {314159} or Counter(r["seed_group"] for r in tests) != Counter({s: 2 for s in SEEDS}):
        raise ValueError("Freeze from the full original N199 plus ten-test injection cohort, not completed-generation availability")
    selected = select_artifacts(rows, dataset=dataset, attack_family="instruction_injection", split="calibration",
                                seed_ids=[314159], per_seed=50, subset_seed=POLICY["subset_seed"])
    selected += select_artifacts(rows, dataset=dataset, attack_family="instruction_injection", split="test",
                                 seed_ids=list(SEEDS), per_seed=2, subset_seed=POLICY["subset_seed"])
    if len({r["sequence_seed"] for r in base}) != len(base) or len({r["source_sequence_id"] for r in base}) != len(base):
        raise ValueError("Original calibration/test sequence identities overlap")
    return selected


def chronology(decisions):
    from crest.rq1_protocol import _timestamp
    cutoffs = [_timestamp(d.get("cutoff_unix", d.get("cutoff"))) for d in decisions]
    if cutoffs != sorted(cutoffs) or len({(d["user_id"], c) for d, c in zip(decisions, cutoffs)}) != len(decisions):
        raise ValueError("Source chronology or distinct user/cutoffs differ")
    for decision, cutoff in zip(decisions, cutoffs):
        history = [_timestamp(r["fields"]["timestamp"]) for r in decision["records"] if r["type"] == "interaction"]
        if history != sorted(history) or any(t >= cutoff for t in history):
            raise ValueError("History is not chronological and strictly prior to its cutoff")


def decision_map(reference, original_source_id):
    clean = reference["clean"]
    if len(clean) != 100 or any(len(d["candidates"]) != 50 or len(d["records"]) != 53 for d in clean):
        raise ValueError("Expected the original T100/C50/H3 references")
    chronology(clean)
    identity = {"policy_sha256": digest(POLICY), "reference_sha256": digest(reference),
                "source_sequence_id": original_source_id, "sequence_seed": reference["seed"]}
    users = defaultdict(list)
    for index in range(50, 100):
        users[clean[index]["user_id"]].append(index)
    eligible = [u for u, indices in users.items() if len(indices) >= 2]
    if not eligible:
        raise ValueError("Source lacks a repeated user in the attacked half; retain this failure without choosing another sequence")
    user = min(eligible, key=lambda u: (digest({**identity, "anchor_user": u}), u))
    order = lambda i: (digest({**identity, "decision_index": i}), i)
    anchors = sorted(users[user], key=order)[:2]
    selected = sorted(sorted(range(50), key=order)[:10] + anchors +
                      sorted(set(range(50, 100)) - set(anchors), key=order)[:8])
    value = {**identity, "old_indices": selected, "old_to_new": {str(old): new for new, old in enumerate(selected)},
             "anchor_user": user, "anchor_old_indices": sorted(anchors), "selection_reads": ["reference", "descriptor_identity"],
             "scheduled_affected_decisions": list(range(10, 20)), "T": 20, "C": 50, "K": 5,
             "adaptation": "Chronological source subtrajectory with balanced original clean/attack halves; temporal distances now count retained decisions"}
    value["sha256"] = digest(value)
    return value


def family_settings(source_attack, family):
    original = source_attack["settings"]
    if original.get("status") != "frozen" or original.get("sha256") != digest({k: original[k] for k in ("parameters", "provenance")}):
        raise ValueError("Original frozen attack setting differs")
    if original["parameters"].get("affected_decisions") != list(range(50, 100)):
        raise ValueError("Projection only supports the declared original 50:100 attack schedule")
    content = {"parameters": {"projection": SCHEMA, "policy_sha256": digest(POLICY), "attack_family": family,
                              "N": 50, "T": 20, "C": 50, "K": 5, "affected_decisions": list(range(10, 20)),
                              "source_attack_settings_sha256": original["sha256"]},
               "provenance": {"original_attack_settings": copy.deepcopy(original),
                              "projection_implementation_sha256": file_hash(__file__),
                              "adaptation": "Reuse exactly selected original observed decisions; no attack regeneration or selection by effectiveness",
                              "generation_scope": "Original T100 attack generation evidence retained; projected T20 evaluation is a disclosed adaptation"}}
    return {"status": "frozen", **content, "sha256": digest(content)}


def project_bundle(bundle, mapping, settings):
    from crest.rq1_protocol import validate_attack_pair
    source = bundle["descriptor"]
    family, dataset = source["attack_family"], source["dataset"]
    pair, reference, audit = (bundle[k] for k in ("pair", "reference", "audit"))
    validate_attack_pair(pair, reference, audit, dataset=dataset, attack_family=family, ingestion=bundle.get("ingestion"))
    if pair["seed"] != source["sequence_seed"] or audit["generator"] != pair["attack"]["generator"] or audit["generator"]["settings_sha256"] != pair["attack"]["settings"]["sha256"]:
        raise ValueError("Original descriptor, generator or settings identity differs")
    if audit["attempt_status"] not in ("applied", "no_change"):
        raise ValueError("Failed or partial generation cannot be projected as clean")
    if family == "deceptive_text_rewriting" and any(r.get("attempt_status") not in ("applied", "no_change") for r in pair["attack"].get("record_runs", [])):
        raise ValueError("Every original rewriting record must be a completed valid attempt")
    if mapping != decision_map(reference, source["source_sequence_id"]) or settings != family_settings(pair["attack"], family):
        raise ValueError("Projection mapping/settings differ from source-only deterministic policy")
    indices = mapping["old_indices"]
    remap = {old: new for new, old in enumerate(indices)}
    source_pins = {k: {"path": source[k + "_path"], "sha256": source[k + "_sha256"]} for k in ("pair", "reference", "audit")}
    if "ingestion_path" in source:
        source_pins["ingestion"] = {"path": source["ingestion_path"], "sha256": source["ingestion_sha256"]}
    provenance = {"schema": SCHEMA, "source_descriptor": copy.deepcopy(source), "source_descriptor_sha256": digest(source),
                  "source_artifacts": source_pins, "index_map_sha256": mapping["sha256"], "old_indices": indices,
                  "source_attempt_status": audit["attempt_status"], "evaluation_launched": False, "model_calls": 0,
                  "cache_policy": "Only exact native per-query keys including K may be reused; K10 rankings must not be truncated or relabeled as K5. No source stages, omissions, thresholds, metrics or phase receipts are reused"}
    source_id = source["source_sequence_id"] + "/" + SCHEMA + "/" + mapping["sha256"]
    projected_reference = {"seed": reference["seed"], "clean": [copy.deepcopy(reference["clean"][i]) for i in indices],
                           "evaluation": {"target": reference["evaluation"]["target"], "positives": [reference["evaluation"]["positives"][i] for i in indices]},
                           "protocol": {"name": SCHEMA, "source_sequence_id": source_id, "source_protocol": copy.deepcopy(reference.get("protocol", {})),
                                        "index_map": copy.deepcopy(mapping), "selection_frozen_before_model_feedback": True}}
    operations = [{**copy.deepcopy(op), "decision_index": remap[op["decision_index"]]} for op in audit["operations"] if op["decision_index"] in remap]
    counts = Counter(op["record_id"] for op in operations)
    generator = {"name": SCHEMA, "implementation_sha256": file_hash(__file__), "settings_sha256": settings["sha256"]}
    projected = copy.deepcopy(projected_reference)
    projected["observed"] = [copy.deepcopy(pair["observed"][i]) for i in indices]
    chronology(projected["observed"])
    projected["evaluation"].update(attacked_ids=sorted(counts), affected_decisions=sorted({op["decision_index"] for op in operations}))
    projected["attack"] = {"name": family, "generator": generator, "settings": settings, "source_attack": copy.deepcopy(pair["attack"]),
                           "projection_provenance": provenance, "scheduled_affected_decisions": list(range(10, 20)),
                           "include_unsuccessful_attempts": True, "is_full_reproduction": False,
                           "achieved": {"identity_appearance_counts": dict(sorted(counts.items())), "affected_decision_fraction": len(projected["evaluation"]["affected_decisions"]) / 20,
                                        "manipulated_record_appearances": len(operations), "nominal_recurrence_fraction": 0.5}}
    ingestion = None
    if family == "interaction_history_manipulation":
        if not counts or max(counts.values()) < 2:
            raise ValueError("Projection loses required repeated manipulated history identity")
        original = bundle["ingestion"]
        assignments = []
        for row in original.get("assignments", []):
            retained = [remap[i] for i in row["decision_indices"] if i in remap]
            if retained:
                assignments.append({**copy.deepcopy(row), "source_first_decision_index": row["first_decision_index"],
                                    "first_decision_index": min(retained), "decision_indices": retained})
        ingestion = {"status": "validated", "assigned_by": "trusted_ingestion", "records": {key: copy.deepcopy(original["records"][key]) for key in counts},
                     "assignments": assignments, "source_ingestion": source_pins["ingestion"], "index_map_sha256": mapping["sha256"],
                     "identity_rule": "Unchanged original trusted ingestion IDs and record bytes; indices remapped only"}
        for key in counts:
            appearances = [i for i, d in enumerate(projected["observed"]) if any(r["id"] == key for r in d["records"])]
            user = ingestion["records"][key]["fields"]["user_id"]
            if appearances != [i for i, d in enumerate(projected["observed"]) if i >= min(appearances) and d["user_id"] == user]:
                raise ValueError("Inserted history does not recur in every later retained user window")
    projected_audit = {"status": "validated", "dataset": dataset, "attack_family": family,
                       "reference_digest": digest(projected_reference), "pair_digest": digest(projected), "generator": generator,
                       "permitted_edit_fields": copy.deepcopy(audit["permitted_edit_fields"]), "operations": operations,
                       "attempt_status": "applied" if operations else "no_change", "selection_frozen_before_feedback": True,
                       "include_unsuccessful_attempts": True, "projection_provenance": provenance}
    validation = validate_attack_pair(projected, projected_reference, projected_audit, dataset=dataset, attack_family=family, ingestion=ingestion)
    return {"pair": projected, "reference": projected_reference, "audit": projected_audit, "ingestion": ingestion,
            "validation": validation, "source_sequence_id": source_id}


def prepare(descriptor_pins, output, dataset, *, families=FAMILIES, source_identity=None):
    from crest.rq1_protocol import load_pair_artifact
    if not families or len(set(families)) != len(families) or not set(families) <= set(FAMILIES):
        raise ValueError("Explicit supported family subset required")
    output = Path(output).resolve()
    rows = []
    for value in descriptor_pins:
        loaded = read_pin(value)
        if not isinstance(loaded, list):
            raise ValueError("Descriptor inputs must be JSON lists")
        rows.extend(loaded)
    originals = [p["path"] for p in descriptor_pins]
    originals += [r[k] for r in rows for k in ("pair_path", "reference_path", "audit_path", "ingestion_path") if k in r]
    if any(Path(p).resolve().is_relative_to(output) for p in originals):
        raise ValueError("Projection output must be separate from every original input")
    selected = select_cohort(rows, dataset)
    lookup = {(source_key(r), r["attack_family"]): r for r in rows if r["dataset"] == dataset}
    if len(lookup) != len([r for r in rows if r["dataset"] == dataset]):
        raise ValueError("Ambiguous family/source allocation")
    intent = {"schema": SCHEMA, "policy": POLICY, "dataset": dataset, "families": list(families),
              "source_descriptor_files": descriptor_pins, "source_identity": source_identity,
              "implementation_sha256": file_hash(__file__), "selected_sources": [{k: r[k] for k in ("split", "seed_group", "sequence_seed", "source_sequence_id")} for r in selected],
              "selection_uses_model_outputs": False}
    intent["sha256"] = digest(intent)
    once(output / "cohort.json", intent)
    descriptors, pending, settings_pins, maps = [], [], {}, {}
    for base in selected:
        base_bundle = load_pair_artifact(base)
        mapping = decision_map(base_bundle["reference"], base["source_sequence_id"])
        key = digest(source_key(base))
        maps[key] = once(output / "index-maps" / (key + ".json"), mapping)
        for family in families:
            row = lookup.get((source_key(base), family))
            if row is None:
                pending.append({"source": list(source_key(base)), "attack_family": family, "status": "missing_original_completed_pair"})
                continue
            bundle = load_pair_artifact(row)
            if bundle["reference"] != base_bundle["reference"]:
                raise ValueError("Attack families disagree on the unchanged original reference")
            settings = family_settings(bundle["pair"]["attack"], family)
            settings_pins[family] = once(output / "settings" / ("attack-" + family + ".json"), settings)
            projected = project_bundle(bundle, mapping, settings)
            descriptor = {k: copy.deepcopy(row[k]) for k in ("dataset", "attack_family", "split", "seed_group", "sequence_seed", "source")}
            descriptor.update(id=row["id"] + "/" + SCHEMA, source_sequence_id=projected["source_sequence_id"],
                              projection={"schema": SCHEMA, "cohort": pin(output / "cohort.json"), "index_map": maps[key],
                                          "source_descriptor": copy.deepcopy(row), "source_descriptor_sha256": digest(row)})
            directory = output / "projected" / family / key
            for name in ("pair", "reference", "audit", "ingestion"):
                if projected[name] is not None:
                    value = once(directory / (name + ".json"), projected[name])
                    descriptor[name + "_path"], descriptor[name + "_sha256"] = value["path"], value["sha256"]
            load_pair_artifact(descriptor)
            descriptors.append(descriptor)
    descriptor_pin = once(output / "descriptors.json", descriptors)
    receipt = {"schema": SCHEMA, "status": "projected_inputs_only" if not pending else "projected_inputs_with_missing_families",
               "cohort": pin(output / "cohort.json"), "descriptors": descriptor_pin, "attack_settings": settings_pins,
               "projected_pairs": len(descriptors), "planned_pairs": 60 * len(families), "missing": pending,
               "index_maps": maps, "model_calls": 0, "evaluation_launched": False, "native_plan_created": False,
               "thresholds_or_results_reused": False, "old_outputs_modified": False}
    receipt["sha256"] = digest(receipt)
    once(output / "preparation-receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--dataset", choices=("amazon-2018-all-beauty", "steam"), required=True)
    parser.add_argument("--descriptor-pin", type=json.loads, action="append", required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tree = args.source.resolve() / "src/crest"
    files = {str(p.relative_to(tree)): file_hash(p) for p in sorted(tree.rglob("*.py"))}
    if not files or digest(files) != args.source_tree_sha256:
        raise ValueError("Frozen scientific source differs before import")
    sys.path.insert(0, str(tree.parent))
    value = prepare(args.descriptor_pin, args.output, args.dataset, families=tuple(args.families),
                    source_identity={"path": str(tree), "sha256": args.source_tree_sha256})
    print(json.dumps({k: value[k] for k in ("status", "projected_pairs", "planned_pairs", "model_calls", "evaluation_launched", "native_plan_created")}))
    return 0 if not value["missing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
