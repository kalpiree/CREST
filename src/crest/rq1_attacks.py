import copy
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from string import Formatter

from .artifacts import digest, file_digest
from .records import validate_decision


VERSION = "rq1-explicit-development-attacks-v1"
CONTENT_FIELDS = {"title", "description", "text", "content", "review", "review_text", "body", "summary"}
ROUNDINGS = {"floor": ROUND_FLOOR, "ceil": ROUND_CEILING, "half_up": ROUND_HALF_UP}
COMMON = {"schema_version", "status", "frozen", "protocol_id", "dataset", "seed", "attack_family", "intensity", "intensity_rounding", "intensity_denominator", "recurrence_fraction", "recurrence_rounding", "recurrence_scope", "affected_decisions", "zero_budget_policy", "feedback_mode", "feedback_query_budget", "minimum_reused_identities", "minimum_identity_appearances"}
INJECTION = {"editable_record_types", "editable_field", "record_selection", "payload_variant", "payload_bank", "template_selection", "append_separator"}
HISTORY = {"recurrence_policy", "target_events_per_user", "max_events_per_user", "filler_policy", "timestamp_rule", "timestamp_step_seconds", "inserted_content"}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _fraction(value, name, upper_inclusive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric fraction")
    result = Decimal(str(value))
    if result < 0 or result > 1 or not upper_inclusive and result == 1:
        raise ValueError(f"{name} is outside its valid range")
    return result


def _timestamp(value):
    if isinstance(value, bool):
        raise ValueError("Invalid timestamp")
    try:
        result = float(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Timestamp must be numeric seconds or timezone-aware ISO8601") from error
        if parsed.tzinfo is None:
            raise ValueError("Timestamp must include a timezone")
        result = parsed.timestamp()
    if not math.isfinite(result):
        raise ValueError("Timestamp must be finite")
    return result


def _rounded(value, rule):
    if rule not in ROUNDINGS:
        raise ValueError("Rounding must explicitly be floor, ceil, or half_up")
    return int(value.to_integral_value(rounding=ROUNDINGS[rule]))


def freeze_attack_config(parameters, provenance):
    if not isinstance(parameters, dict) or not isinstance(provenance, dict) or not provenance:
        raise ValueError("Explicit parameters and nonempty setting-selection provenance are required")
    content = {"parameters": copy.deepcopy(parameters), "provenance": copy.deepcopy(provenance)}
    frozen = {"status": "frozen", **content, "sha256": digest(content)}
    _config(frozen, parameters.get("attack_family"))
    return frozen


def _config(frozen, family):
    if family not in ("instruction_injection", "interaction_history_manipulation"):
        raise ValueError("Only injection and trusted history insertion are implemented here")
    if not isinstance(frozen, dict) or set(frozen) != {"status", "parameters", "provenance", "sha256"} or frozen["status"] != "frozen" or not isinstance(frozen["provenance"], dict) or not frozen["provenance"]:
        raise ValueError("A frozen parameter/provenance envelope is required")
    if digest({key: frozen[key] for key in ("parameters", "provenance")}) != frozen["sha256"]:
        raise ValueError("Attack settings fingerprint differs")
    config = frozen["parameters"]
    required = COMMON | (INJECTION if family == "instruction_injection" else HISTORY)
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Attack configuration has missing or unknown fields")
    if config["schema_version"] != 1 or isinstance(config["schema_version"], bool) or config["status"] != "development_preparation" or config["frozen"] is not True or config["attack_family"] != family:
        raise ValueError("Use an explicitly frozen development-preparation configuration")
    for key in ("protocol_id", "dataset"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string")
    _integer(config["seed"], "seed")
    _fraction(config["intensity"], "intensity", family != "interaction_history_manipulation")
    _fraction(config["recurrence_fraction"], "recurrence_fraction")
    _rounded(Decimal(0), config["intensity_rounding"])
    _rounded(Decimal(0), config["recurrence_rounding"])
    if config["intensity_denominator"] != "observed_records" or config["recurrence_scope"] not in ("affected_decisions", "each_manipulated_identity") or config["zero_budget_policy"] not in ("error", "allow_zero"):
        raise ValueError("Resolve denominator, recurrence scope, and zero-budget policy explicitly")
    if config["feedback_mode"] != "none" or config["feedback_query_budget"] != 0 or isinstance(config["feedback_query_budget"], bool):
        raise ValueError("These builders perform zero feedback queries; optimized attacks require a separate generator")
    _integer(config["minimum_reused_identities"], "minimum_reused_identities")
    _integer(config["minimum_identity_appearances"], "minimum_identity_appearances", 2)
    if not isinstance(config["affected_decisions"], list):
        raise ValueError("A frozen affected-decision schedule is required")
    for index in config["affected_decisions"]:
        _integer(index, "affected decision")
    if config["affected_decisions"] != sorted(set(config["affected_decisions"])):
        raise ValueError("Affected decisions must be sorted and distinct")
    return {**copy.deepcopy(config), "settings_sha256": frozen["sha256"], "settings_provenance": copy.deepcopy(frozen["provenance"])}


def _reference(reference, config, validate_schedule=True):
    if not isinstance(reference, dict) or not isinstance(reference.get("clean"), list) or not reference["clean"]:
        raise ValueError("A nonempty independently retained clean reference is required")
    digest(reference)
    _integer(reference.get("seed"), "reference.seed")
    if "protocol" in reference and not isinstance(reference["protocol"], dict):
        raise ValueError("Reference protocol metadata must be a mapping")
    clean = reference["clean"]
    if "observed" in reference and reference["observed"] != clean:
        raise ValueError("Input reference already contains modified observations")
    evaluation = reference.get("evaluation", {})
    positives, target = evaluation.get("positives"), evaluation.get("target")
    if not isinstance(positives, list) or len(positives) != len(clean) or not isinstance(target, str) or not target or target in positives:
        raise ValueError("Reference requires one positive per decision and a separate fixed target")
    if evaluation.get("attacked_ids", []) or evaluation.get("affected_decisions", []):
        raise ValueError("Reference contains manipulation labels")
    seen, users, cutoffs = {}, {}, []
    for index, decision in enumerate(clean):
        validate_decision(decision)
        user = decision.get("user_id")
        if not isinstance(user, str) or not user or not decision["records"] or positives[index] not in decision["candidates"] or target not in decision["candidates"]:
            raise ValueError("Clean decisions require records, users, the positive, and the fixed target")
        if "cutoff_unix" not in decision or isinstance(decision["cutoff_unix"], bool) or not isinstance(decision["cutoff_unix"], (int, float)):
            raise ValueError("A numeric decision cutoff_unix is required")
        cutoff = _timestamp(decision["cutoff_unix"])
        if "cutoff" in decision and _timestamp(decision["cutoff"]) != cutoff:
            raise ValueError("Decision cutoff representations disagree")
        if user in users and cutoff <= users[user]:
            raise ValueError("Repeated user decisions must advance to strictly later cutoffs")
        if cutoffs and cutoff < cutoffs[-1]:
            raise ValueError("Decision sequence must be chronological")
        users[user] = cutoff
        cutoffs.append(cutoff)
        history_times = []
        for record in decision["records"]:
            fields = record["fields"]
            trusted = {key: value for key, value in fields.items() if key.endswith("_id") or key in ("id", "timestamp", "event")}
            stable = (record["type"], tuple(fields), trusted)
            if record["id"] in seen and seen[record["id"]] != stable:
                raise ValueError("A clean identity changed its type, field schema, or trusted references")
            seen[record["id"]] = stable
            if record["type"] == "interaction":
                if fields.get("user_id") != user or not fields.get("item_id") or "timestamp" not in fields:
                    raise ValueError("History records must have trusted user/item/timestamp references")
                timestamp = _timestamp(fields["timestamp"])
                if timestamp >= cutoff:
                    raise ValueError("Clean history must strictly precede its decision cutoff")
                history_times.append(timestamp)
        if history_times != sorted(history_times):
            raise ValueError("Clean history records must retain chronological order")
    if validate_schedule:
        expected = _rounded(Decimal(len(clean)) * _fraction(config["recurrence_fraction"], "recurrence_fraction"), config["recurrence_rounding"])
        schedule = config["affected_decisions"]
        if len(schedule) != expected or schedule and schedule[-1] >= len(clean):
            raise ValueError("Affected schedule does not match the explicit rounded recurrence budget")
    return copy.deepcopy(reference), target


def _count(size, config, insertion=False):
    fraction = _fraction(config["intensity"], "intensity", not insertion)
    desired = Decimal(size) * fraction / (1 - fraction) if insertion else Decimal(size) * fraction
    count = _rounded(desired, config["intensity_rounding"])
    if count == 0 and config["zero_budget_policy"] == "error":
        raise ValueError("Rounded intensity budget is zero; no hidden minimum-one adjustment is allowed")
    return count


def _operation(index, before, after):
    changed = list(after["fields"]) if before is None else [field for field in before["fields"] if before["fields"][field] != after["fields"][field]]
    return {"decision_index": index, "record_id": after["id"], "kind": "insert" if before is None else "edit", "before_sha256": None if before is None else digest(before), "after_sha256": digest(after), "changed_fields": changed}


def _finish(reference, observed, config, operations, permitted, ingestion, details):
    counts = Counter(row["record_id"] for row in operations)
    affected = sorted({row["decision_index"] for row in operations})
    if not operations and config["zero_budget_policy"] == "error":
        raise ValueError("Attack changes no records under the declared policy")
    if operations and affected != config["affected_decisions"]:
        raise ValueError("The achieved attack schedule differs from the declared affected decisions")
    reused = sum(count >= config["minimum_identity_appearances"] for count in counts.values())
    if operations and reused < config["minimum_reused_identities"]:
        raise ValueError("The achieved identity reuse does not satisfy the explicit constraint")
    if config["recurrence_scope"] == "each_manipulated_identity" and any(count != len(config["affected_decisions"]) for count in counts.values()):
        raise ValueError("Manipulated identities do not each satisfy the declared recurrence fraction")
    t = len(observed)
    generator = {"name": VERSION, "implementation_sha256": file_digest(__file__), "settings_sha256": config["settings_sha256"]}
    by_decision = Counter(row["decision_index"] for row in operations)
    all_ids = {record["id"] for decision in observed for record in decision["records"]}
    achieved = {"nominal_intensity": config["intensity"], "intensity_denominator": "observed_records", "integer_policy": config["intensity_rounding"], "nominal_recurrence_fraction": config["recurrence_fraction"], "recurrence_scope": config["recurrence_scope"], "affected_decisions": affected, "affected_decision_fraction": len(affected) / t, "manipulated_distinct_identities": len(counts), "observed_distinct_identities": len(all_ids), "manipulated_record_appearances": len(operations), "identity_appearance_counts": dict(sorted(counts.items())), "identity_recurrence_fractions": {identity: count / t for identity, count in sorted(counts.items())}, "reused_identities_meeting_constraint": reused, "per_decision": [{"decision_index": index, "clean_record_count": len(reference["clean"][index]["records"]), "observed_record_count": len(decision["records"]), "manipulated_records": by_decision[index], "intensity": by_decision[index] / len(decision["records"])} for index, decision in enumerate(observed)]}
    pair = copy.deepcopy(reference)
    pair["observed"] = observed
    pair["evaluation"] = {**pair["evaluation"], "attacked_ids": sorted(counts), "affected_decisions": affected}
    frozen = {"status": "frozen", "parameters": {key: value for key, value in config.items() if key not in ("settings_sha256", "settings_provenance")}, "provenance": config["settings_provenance"], "sha256": config["settings_sha256"]}
    pair["attack"] = {"name": config["attack_family"], "generator": generator, "protocol_id": config["protocol_id"], "settings": frozen, "query_budget": 0, "feedback_queries_used": 0, "feedback_protocol": "fixed construction without ranking-feedback optimization", "is_full_reproduction": False, "scope": "Fixed target-promotion construction without ranking feedback", "include_unsuccessful_attempts": True, "achieved": achieved, **details}
    pair.setdefault("protocol", {})["rq1_attack_preparation"] = {"generator": generator, "reference_digest": digest(reference), "is_full_reproduction": False}
    audit = {"status": "validated", "dataset": config["dataset"], "attack_family": config["attack_family"], "reference_digest": digest(reference), "pair_digest": digest(pair), "generator": generator, "permitted_edit_fields": permitted, "operations": operations, "attempt_status": "applied" if operations else "no_change", "selection_frozen_before_feedback": True, "include_unsuccessful_attempts": True}
    return {"pair": pair, "reference": copy.deepcopy(reference), "audit": audit, "ingestion": ingestion}


def inject_instructions(reference, frozen_config):
    config = _config(frozen_config, "instruction_injection")
    reference, target = _reference(reference, config)
    field = config["editable_field"]
    types = config["editable_record_types"]
    if field not in CONTENT_FIELDS or not isinstance(types, list) or not types or any(kind not in ("item_text", "review_text", "review") for kind in types) or len(types) != len(set(types)) or "item_text" not in types:
        raise ValueError("Declare editable text record types and one permitted content field")
    if config["record_selection"] not in ("shared_identities", "target_then_seeded_per_decision") or config["template_selection"] != "seeded_per_identity" or config["payload_variant"] not in ("direct", "obfuscated") or not isinstance(config["append_separator"], str):
        raise ValueError("Resolve record and payload selection explicitly")
    bank = config["payload_bank"]
    if not isinstance(bank, list) or not bank:
        raise ValueError("An explicit payload bank is required")
    templates = {}
    for entry in bank:
        if not isinstance(entry, dict) or set(entry) != {"id", "variant", "template"} or not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in templates or entry["variant"] not in ("direct", "obfuscated") or not isinstance(entry["template"], str):
            raise ValueError("Invalid or duplicate payload bank entry")
        parsed = list(Formatter().parse(entry["template"]))
        substitutions = [name for _, name, _, _ in parsed if name is not None]
        if not substitutions or any(name != "target" or spec or conversion for _, name, spec, conversion in parsed if name is not None):
            raise ValueError("Payload templates require only plain {target} substitutions")
        templates[entry["id"]] = entry
    eligible_templates = sorted(identity for identity, entry in templates.items() if entry["variant"] == config["payload_variant"])
    if not eligible_templates:
        raise ValueError("Payload bank has no entries for the chosen variant")
    observed = copy.deepcopy(reference["clean"])
    eligible, targets, budgets = {}, {}, {}
    for index in config["affected_decisions"]:
        decision = observed[index]
        eligible[index] = {record["id"]: record for record in decision["records"] if record["type"] in types and field in record["fields"]}
        target_ids = [record["id"] for record in eligible[index].values() if record["type"] == "item_text" and record["fields"].get("item_id") == target]
        if len(target_ids) != 1:
            raise ValueError("Exactly one editable target item-text identity is required per affected decision")
        targets[index] = target_ids[0]
        budgets[index] = _count(len(decision["records"]), config)
        if budgets[index] > len(eligible[index]):
            raise ValueError("Rounded manipulation budget exceeds the eligible content records")
    rng = random.Random(config["seed"])
    shared = None
    if eligible and config["record_selection"] == "shared_identities":
        common = set.intersection(*(set(value) for value in eligible.values()))
        if len(set(budgets.values())) != 1 or len(set(targets.values())) != 1:
            raise ValueError("Shared-identity edits require equal budgets and a stable target identity")
        count, target_id = next(iter(budgets.values())), next(iter(targets.values()))
        if len(common) < count or target_id not in common:
            raise ValueError("Insufficient shared identities for the declared manipulation budget")
        shared = [target_id, *rng.sample(sorted(common - {target_id}), count - 1)] if count else []
    operations, payloads = [], {}
    for index in config["affected_decisions"]:
        count, target_id = budgets[index], targets[index]
        chosen = shared if shared is not None else ([target_id, *rng.sample(sorted(set(eligible[index]) - {target_id}), count - 1)] if count else [])
        for identity in chosen:
            record = eligible[index][identity]
            if identity not in payloads:
                selected = random.Random(digest({"seed": config["seed"], "identity": identity})).choice(eligible_templates)
                payloads[identity] = {"template_id": selected, "variant": config["payload_variant"], "payload": templates[selected]["template"].format(target=target)}
            before = copy.deepcopy(record)
            record["fields"][field] += config["append_separator"] + payloads[identity]["payload"]
            operations.append(_operation(index, before, record))
    return _finish(reference, observed, config, operations, {kind: [field] for kind in types}, None, {"payloads_by_identity": dict(sorted(payloads.items())), "edit_mode": "append original content", "target_identity_required_in_affected_decisions": True})


def _merge_history(records, additions):
    pending = sorted(additions, key=lambda record: (_timestamp(record["fields"]["timestamp"]), record["id"]))
    last = max(index for index, record in enumerate(records) if record["type"] == "interaction")
    result = []
    for index, record in enumerate(records):
        if record["type"] == "interaction":
            key = (_timestamp(record["fields"]["timestamp"]), record["id"])
            while pending and (_timestamp(pending[0]["fields"]["timestamp"]), pending[0]["id"]) < key:
                result.append(pending.pop(0))
        result.append(record)
        if index == last:
            result.extend(pending)
            pending.clear()
    return result


def insert_history(reference, frozen_config, catalog):
    config = _config(frozen_config, "interaction_history_manipulation")
    reference, target = _reference(reference, config)
    if config["recurrence_policy"] != "all_later_windows_of_same_user" or config["filler_policy"] not in {"recent_distinct_history_excluding_target_and_sequence_positives", "recent_history_with_replacement_excluding_target"} or config["timestamp_rule"] != "after_latest_history_and_previous_user_cutoff":
        raise ValueError("Resolve the explicit chronological insertion and filler policies")
    _integer(config["target_events_per_user"], "target_events_per_user", 1)
    _integer(config["max_events_per_user"], "max_events_per_user", 1)
    _integer(config["timestamp_step_seconds"], "timestamp_step_seconds", 1)
    content = config["inserted_content"]
    if not isinstance(content, dict) or not content or "event" not in content or any(key not in CONTENT_FIELDS | {"event", "rating"} or not isinstance(value, str) for key, value in content.items()):
        raise ValueError("Explicit inserted event/content fields must not supply identities or timestamps")
    if "title" in content or not content["event"]:
        raise ValueError("Inserted titles come from the trusted catalog and event must be nonempty")
    if not isinstance(catalog, dict):
        raise ValueError("A trusted catalog mapping is required")
    observed = copy.deepcopy(reference["clean"])
    users = defaultdict(list)
    for index, decision in enumerate(observed):
        users[decision["user_id"]].append(index)
    affected = set(config["affected_decisions"])
    existing = {record["id"] for decision in observed for record in decision["records"]}
    operations, assigned, receipts = [], {}, []
    replacement = config['filler_policy'] == 'recent_history_with_replacement_excluding_target'
    forbidden_fillers = {target} if replacement else {target, *reference["evaluation"]["positives"]}
    for user, indices in sorted(users.items()):
        selected = [index for index in indices if index in affected]
        if not selected:
            continue
        if selected != indices[indices.index(selected[0]):]:
            raise ValueError("Inserted histories must recur at every subsequent selected user window")
        budgets = {_count(len(observed[index]["records"]), config, True) for index in selected}
        if len(budgets) != 1:
            raise ValueError("Stable inserted identities require equal rounded budgets across this user's affected windows")
        count = budgets.pop()
        if count == 0:
            continue
        if not config["target_events_per_user"] <= count <= config["max_events_per_user"]:
            raise ValueError("Rounded insertion budget violates the explicit target/event limits")
        first = observed[selected[0]]
        history = [record for record in first["records"] if record["type"] == "interaction"]
        if not history:
            raise ValueError("Context-compatible history insertion requires original interaction history")
        fillers, filler_sources = [], {}
        for record in reversed(history):
            item = record["fields"]["item_id"]
            if item not in forbidden_fillers and item not in fillers:
                fillers.append(item)
                filler_sources[item] = record['id']
        needed = count - config["target_events_per_user"]
        if needed and not fillers or not replacement and len(fillers) < needed:
            raise ValueError("Insufficient distinct context-compatible filler items for the declared event budget")
        chosen_fillers = [fillers[i % len(fillers)] for i in range(needed)] if replacement else fillers[:needed]
        items = [target] * config["target_events_per_user"] + chosen_fillers
        latest = max(_timestamp(record["fields"]["timestamp"]) for record in history)
        previous = indices.index(selected[0]) - 1
        if previous >= 0:
            latest = max(latest, _timestamp(observed[indices[previous]]["cutoff_unix"]))
        base = math.floor(latest)
        additions = []
        for slot, item in enumerate(items):
            if item not in catalog or not isinstance(catalog[item], dict) or not isinstance(catalog[item].get("title"), str) or catalog[item].get("id", item) != item:
                raise ValueError("An inserted item is missing from the trusted canonical catalog")
            timestamp = base + (slot + 1) * config["timestamp_step_seconds"]
            if timestamp <= latest or timestamp >= first["cutoff_unix"]:
                raise ValueError("Insufficient chronological space before the first affected cutoff")
            identity = "crest-inserted/" + digest({"reference_digest": digest(reference), "settings_sha256": config["settings_sha256"], "user_id": user, "slot": slot})
            if identity in existing or identity in assigned:
                raise ValueError("Trusted ingestion identity collision")
            fields = {"user_id": user, "item_id": item, "title": catalog[item]["title"], "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), **content}
            record = {"id": identity, "type": "interaction", "fields": fields}
            additions.append(record)
            assigned[identity] = copy.deepcopy(record)
            receipt = {"record_id": identity, "user_id": user, "first_decision_index": selected[0], "decision_indices": selected, "role": "target_support" if slot < config["target_events_per_user"] else "context_filler", "timestamp_unix": timestamp}
            if replacement and slot >= config['target_events_per_user']:
                receipt.update(source_history_record_id=filler_sources[item], source_item_id=item, filler_policy=config['filler_policy'],
                               distinct_eligible_history_items=len(fillers), replacement_cycle= (slot-config['target_events_per_user']) // len(fillers))
            receipts.append(receipt)
        for index in selected:
            observed[index]["records"] = _merge_history(observed[index]["records"], copy.deepcopy(additions))
            validate_decision(observed[index])
            for record in additions:
                operations.append(_operation(index, None, record))
    ingestion = {"status": "validated", "assigned_by": "trusted_ingestion", "records": assigned, "assignments": receipts, "catalog_sha256": digest(catalog), "identity_rule": "sha256(reference, frozen settings, trusted user, ingestion slot)"}
    return _finish(reference, observed, config, operations, {}, ingestion, {"insertion_count_rule": "round(clean_record_count * intensity / (1 - intensity))", "original_history_retained": True, "chronological_insertion_policy": config["timestamp_rule"],
        "filler_policy": config['filler_policy'], "filler_provenance": 'Most-recent distinct original non-target history items, cycled when needed; sequence-positive membership is not consulted' if replacement else 'Most-recent distinct original history items excluding target and every sequence positive'})


def bundle_rewriting_result(reference, result, frozen_settings, *, dataset):
    from .rq1_protocol import validate_attack_pair
    reference, target = _reference(reference, {}, validate_schedule=False)
    if not isinstance(frozen_settings, dict) or set(frozen_settings) != {"status", "parameters", "provenance", "sha256"} or frozen_settings["status"] != "frozen":
        raise ValueError("A frozen TextSimu parameter/provenance envelope is required")
    content = {key: frozen_settings[key] for key in ("parameters", "provenance")}
    if any(not isinstance(value, dict) or not value for value in content.values()) or digest(content) != frozen_settings["sha256"]:
        raise ValueError("TextSimu settings fingerprint differs")
    parameters = content["parameters"]
    edited_item = result.get('edited_item_id', target)
    if edited_item != target and frozen_settings['provenance'].get('multi_record_same_target_user_approved') is not True:
        raise ValueError('Non-target source rewriting requires explicit multiple-record same-target approval')
    generator = result.get("generator", {})
    if generator.get("name") != "TextSimu-RecAtom" or generator.get("settings_sha256") != frozen_settings["sha256"] or result.get("attack_family") != "deceptive_text_rewriting":
        raise ValueError("A real TextSimu-RecAtom result bound to the frozen settings is required")
    if result.get("target_item_id") != target or result.get("text_field") not in CONTENT_FIELDS or result["text_field"] != parameters.get("text_field"):
        raise ValueError("Rewriting target or permitted content field differs from the frozen input")
    for key in ("record_id", "original_text", "rewritten_text"):
        if not isinstance(result.get(key), str) or key == "record_id" and not result[key]:
            raise ValueError("Rewriting result requires a record identity and original/revised text")
    schedule = parameters.get("affected_decisions")
    if not isinstance(schedule, list) or result.get("affected_decisions") != schedule:
        raise ValueError("Rewriting result schedule differs from frozen settings")
    for index in schedule:
        _integer(index, "rewriting affected decision")
    if schedule != sorted(set(schedule)) or schedule and schedule[-1] >= len(reference["clean"]):
        raise ValueError("Invalid rewriting affected-decision schedule")
    used = _integer(result.get("feedback_queries_used"), "feedback_queries_used")
    budget = _integer(parameters.get("feedback_query_budget"), "feedback_query_budget")
    if used > budget or not isinstance(result.get("trace"), (dict, list)) or not result["trace"] or not isinstance(result.get("generation_metadata"), dict) or not result["generation_metadata"]:
        raise ValueError("Rewriting requires retained trace/model provenance within its feedback budget")
    expected = copy.deepcopy(reference["clean"])
    operations = []
    for index, decision in enumerate(expected):
        matches = [record for record in decision["records"] if record["id"] == result["record_id"]]
        if index in schedule and len(matches) != 1:
            raise ValueError("Affected rewriting identity is absent from the independently retained reference")
        for record in matches:
            field = result["text_field"]
            if record["type"] != "item_text" or record["fields"].get("item_id") != edited_item or record["fields"].get(field) != result["original_text"]:
                raise ValueError("Rewriting source text/type/target differs from its independent reference")
            if index in schedule:
                before = copy.deepcopy(record)
                record["fields"][field] = result["rewritten_text"]
                if before != record:
                    operations.append(_operation(index, before, record))
    if result.get("observed") != expected:
        raise ValueError("Rewriting result changed content outside the frozen target-field schedule")
    attempt = result.get("attempt_status")
    if attempt not in ("applied", "no_change", "generation_failed") or (attempt == "applied") != bool(operations):
        raise ValueError("Rewriting attempt status contradicts its actual edits")
    pair = copy.deepcopy(reference)
    pair["observed"] = copy.deepcopy(result["observed"])
    pair["evaluation"] = {**pair["evaluation"], "attacked_ids": sorted({row["record_id"] for row in operations}), "affected_decisions": sorted({row["decision_index"] for row in operations})}
    pair["attack"] = {"name": "deceptive_text_rewriting", "generator": copy.deepcopy(generator), "settings": copy.deepcopy(frozen_settings), "query_budget": budget, "feedback_queries_used": used, "is_full_reproduction": False, "scope": "TextSimu adaptation to frozen inference and one editable RecAtom; model substitutions remain in generation metadata", "include_unsuccessful_attempts": True, "construction_result_sha256": digest(result), "trace": copy.deepcopy(result["trace"]), "generation_metadata": copy.deepcopy(result["generation_metadata"]), "scheduled_affected_decisions": list(schedule)}
    pair.setdefault("protocol", {})["rq1_attack_preparation"] = {"generator": copy.deepcopy(generator), "reference_digest": digest(reference), "bundler_implementation_sha256": file_digest(__file__), "is_full_reproduction": False}
    audit = {"status": "validated", "dataset": dataset, "attack_family": "deceptive_text_rewriting", "reference_digest": digest(reference), "pair_digest": digest(pair), "generator": copy.deepcopy(generator), "permitted_edit_fields": {"item_text": [result["text_field"]]}, "operations": operations, "attempt_status": attempt, "selection_frozen_before_feedback": True, "include_unsuccessful_attempts": True}
    validate_attack_pair(pair, reference, audit, dataset=dataset, attack_family="deceptive_text_rewriting")
    return {"pair": pair, "reference": reference, "audit": audit, "ingestion": None}


def rewriting_subsettings(frozen_settings, sequence_seed, record_id):
    if not isinstance(frozen_settings, dict) or set(frozen_settings) != {'status', 'parameters', 'provenance', 'sha256'} or frozen_settings['status'] != 'frozen' or digest({k: frozen_settings[k] for k in ('parameters', 'provenance')}) != frozen_settings['sha256']:
        raise ValueError('A pinned multiple-record TextSimu envelope is required')
    if frozen_settings['provenance'].get('multi_record_same_target_user_approved') is not True:
        raise ValueError('Multiple-record same-target adaptation requires explicit user approval')
    multi = frozen_settings['parameters'].get('multi_record', {})
    required = {'record_selection', 'record_count', 'nominal_intensity', 'intensity_rounding', 'coordination', 'feedback_scope', 'seed_policy'}
    if not isinstance(multi, dict) or set(multi) != required or multi['record_selection'] != 'target_then_shared_distractors' or multi['coordination'] != 'independent_from_same_original' or multi['feedback_scope'] != 'per_record_alone_same_target' or multi['seed_policy'] != 'sha256_base_seed_sequence_seed_record_id':
        raise ValueError('Resolve all supported multiple-record selection/composition policies explicitly')
    _integer(multi['record_count'], 'record_count', 1)
    _fraction(multi['nominal_intensity'], 'nominal_intensity')
    _rounded(Decimal(0), multi['intensity_rounding'])
    _integer(sequence_seed, 'sequence_seed')
    if not isinstance(record_id, str) or not record_id:
        raise ValueError('Sub-run requires a stable record identity')
    parameters = copy.deepcopy(frozen_settings['parameters'])
    parameters.pop('multi_record')
    _integer(parameters.get('seed'), 'base seed')
    parameters['seed'] = int(digest({'base_seed': parameters['seed'], 'sequence_seed': sequence_seed, 'record_id': record_id})[:15], 16)
    provenance = {**copy.deepcopy(frozen_settings['provenance']), 'base_settings_sha256': frozen_settings['sha256'],
                  'record_id': record_id, 'sequence_seed': sequence_seed, 'seed_derivation': multi['seed_policy']}
    content = {'parameters': parameters, 'provenance': provenance}
    return {'status': 'frozen', **content, 'sha256': digest(content)}


def bundle_rewriting_results(reference, results, frozen_settings, *, dataset):
    from .rq1_protocol import validate_attack_pair
    reference, target = _reference(reference, {}, validate_schedule=False)
    rewriting_subsettings(frozen_settings, reference['seed'], 'validation')
    parameters, multi = frozen_settings['parameters'], frozen_settings['parameters']['multi_record']
    schedule = parameters['affected_decisions']
    shared = reference.get('protocol', {}).get('shared_distractor_ids', [])
    selected_items = [target, *shared[:multi['record_count'] - 1]]
    if len(selected_items) != multi['record_count'] or len(set(selected_items)) != len(selected_items):
        raise ValueError('Reference lacks the prespecified distinct shared editable records')
    selected_ids = ['item/' + item for item in selected_items]
    if not isinstance(results, list) or [r.get('record_id') for r in results] != selected_ids:
        raise ValueError('Results must include every frozen selected record in its original order')
    expected_counts = [_rounded(Decimal(len(reference['clean'][i]['records'])) * _fraction(multi['nominal_intensity'], 'nominal_intensity'), multi['intensity_rounding']) for i in schedule]
    if not expected_counts or any(count != len(selected_ids) for count in expected_counts):
        raise ValueError('Edited-record count does not equal the explicitly rounded nominal intensity')
    observed, operations, record_runs = copy.deepcopy(reference['clean']), [], []
    used = budget = 0
    implementations = set()
    for item, result in zip(selected_items, results):
        if result.get('edited_item_id', target) != item or result.get('attempt_status') not in ('applied', 'no_change'):
            raise ValueError('A valid completed result is required for every selected source item; failures are not clean-input fallbacks')
        settings = rewriting_subsettings(frozen_settings, reference['seed'], result['record_id'])
        single = bundle_rewriting_result(reference, result, settings, dataset=dataset)
        for operation in single['audit']['operations']:
            matches = [r for r in observed[operation['decision_index']]['records'] if r['id'] == result['record_id']]
            matches[0]['fields'][parameters['text_field']] = result['rewritten_text']
            operations.append(operation)
        used += result['feedback_queries_used']
        budget += settings['parameters']['feedback_query_budget']
        implementations.add(result['generator']['implementation_sha256'])
        record_runs.append({'record_id': result['record_id'], 'edited_item_id': item, 'subsettings': settings, 'result_sha256': digest(result),
                            'attempt_status': result['attempt_status'], 'feedback_queries_used': result['feedback_queries_used'],
                            'trace': copy.deepcopy(result['trace']), 'generation_metadata': copy.deepcopy(result['generation_metadata'])})
    if len(implementations) != 1:
        raise ValueError('Per-record generation implementations differ')
    counts = Counter(row['record_id'] for row in operations)
    per_decision = [{'decision': i, 'total_records': len(d['records']), 'manipulated_records': sum(row['decision_index'] == i for row in operations),
                     'intensity': sum(row['decision_index'] == i for row in operations) / len(d['records'])} for i, d in enumerate(observed)]
    generator = {'name': 'TextSimu-RecAtom', 'implementation_sha256': digest({'core': next(iter(implementations)), 'bundler': file_digest(__file__)}), 'settings_sha256': frozen_settings['sha256']}
    pair = copy.deepcopy(reference)
    pair['observed'] = observed
    pair['evaluation'] = {**pair['evaluation'], 'attacked_ids': sorted(counts), 'affected_decisions': sorted({row['decision_index'] for row in operations})}
    pair['attack'] = {'name': 'deceptive_text_rewriting', 'generator': generator, 'settings': copy.deepcopy(frozen_settings),
                      'query_budget': budget, 'feedback_queries_used': used, 'selected_record_ids': selected_ids, 'record_runs': record_runs,
                      'composition': multi['coordination'], 'feedback_scope': multi['feedback_scope'],
                      'achieved': {'nominal_intensity': multi['nominal_intensity'], 'per_decision': per_decision,
                                   'distinct_manipulated_records': len(counts), 'manipulated_record_appearances': sum(counts.values()),
                                   'identity_manipulated_appearances': dict(counts), 'identity_recurrence_fractions': {k: v / len(observed) for k, v in counts.items()}},
                      'include_unsuccessful_attempts': True, 'is_full_reproduction': False,
                      'scope': 'Multiple existing records with one fixed target; records are optimized independently against the same original input and then combined'}
    audit = {'status': 'validated', 'dataset': dataset, 'attack_family': 'deceptive_text_rewriting', 'reference_digest': digest(reference),
             'pair_digest': digest(pair), 'generator': generator, 'permitted_edit_fields': {'item_text': [parameters['text_field']]},
             'operations': operations, 'attempt_status': 'applied' if operations else 'no_change',
             'selection_frozen_before_feedback': True, 'include_unsuccessful_attempts': True}
    validate_attack_pair(pair, reference, audit, dataset=dataset, attack_family='deceptive_text_rewriting')
    return {'pair': pair, 'reference': reference, 'audit': audit, 'ingestion': None}
