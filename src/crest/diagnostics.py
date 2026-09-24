import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .metrics import _validate_rankings, gmax, identity_metrics, utility


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _at(mapping, index):
    return mapping.get(index, mapping.get(str(index), {}))


def _records(decision):
    records = decision["records"]
    result = {record["id"]: record for record in records}
    if not records or len(result) != len(records):
        raise ValueError("Each decision needs nonempty, distinct record identities")
    return result


def _cutoff(decision):
    value = decision.get("cutoff_unix", decision.get("cutoff"))
    if value is None:
        return None, None
    if isinstance(value, bool):
        raise ValueError("Invalid decision cutoff")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Invalid decision cutoff")
        return value, float(value)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return str(value), parsed.timestamp()


def mechanism_diagnostics(pair, screening, calibration=None, selection=None, k=None):
    observed = pair["observed"]
    clean = pair["clean"]
    if not observed or len(observed) != len(clean):
        raise ValueError("Clean and observed decisions must have matching nonempty lengths")
    t_count = len(observed)
    k = _validate_rankings(screening["rankings"], "screened rankings", t_count, k)
    observed_records = [_records(decision) for decision in observed]
    clean_records = [_records(decision) for decision in clean]
    presence = [set(records) for records in observed_records]
    all_ids = set().union(*presence)
    omitted = set(screening["omitted_ids"])
    if not omitted <= all_ids:
        raise ValueError("Omitted identities must occur in the observed sequence")
    for decision, ranking in zip(observed, screening["rankings"]):
        candidates = decision["candidates"]
        if len(candidates) != len(set(candidates)) or len(candidates) < k or not set(ranking) <= set(candidates):
            raise ValueError("Screened rankings must respect distinct fixed candidates")
    appearances = Counter(record for records in presence for record in records)
    changed_by_decision = [
        {record for record in records if record not in reference or records[record] != reference[record]}
        for records, reference in zip(observed_records, clean_records)
    ]
    actual_changed = set().union(*changed_by_decision)
    evaluation = pair.get("evaluation", {})
    labels_present = "attacked_ids" in evaluation
    attacked = set(evaluation.get("attacked_ids", actual_changed))
    if not attacked <= all_ids:
        raise ValueError("Attacked identities must occur in the observed sequence")
    changed_appearances = Counter(record for records in changed_by_decision for record in records)

    def cost(ids):
        return sum(len(ids & records) / len(records) for records in presence) / t_count

    identity_recurrence = {
        record: {
            "observed_appearances": appearances[record],
            "identity_presence_fraction": appearances[record] / t_count,
            "changed_appearances": changed_appearances[record],
            "changed_decision_fraction": changed_appearances[record] / t_count,
            "changed_fraction_given_presence": changed_appearances[record] / appearances[record],
            "unchanged_appearances": appearances[record] - changed_appearances[record],
            "identity_wide_omission_cost": cost({record}),
        }
        for record in sorted(attacked | actual_changed)
    }
    recurrence = {
        "T": t_count,
        "labeled_attacked_ids": sorted(attacked) if labels_present else None,
        "actual_changed_ids": sorted(actual_changed),
        "label_identity_agreement": attacked == actual_changed if labels_present else None,
        "labeled_but_unchanged_ids": sorted(attacked - actual_changed) if labels_present else None,
        "changed_but_unlabeled_ids": sorted(actual_changed - attacked) if labels_present else None,
        "changed_decisions": sum(bool(records) for records in changed_by_decision),
        "changed_decision_fraction": sum(bool(records) for records in changed_by_decision) / t_count,
        "changed_record_appearances": sum(changed_appearances.values()),
        "deleted_clean_record_appearances": sum(len(set(reference) - set(records)) for records, reference in zip(observed_records, clean_records)),
        "per_identity": identity_recurrence,
    }
    users = defaultdict(list)
    interaction_users = defaultdict(set)
    interaction_counts = Counter()
    for t, decision in enumerate(observed):
        users[decision["user_id"]].append(t)
        for record_id, record in observed_records[t].items():
            if record["type"] == "interaction":
                interaction_counts[record_id] += 1
                interaction_users[record_id].add(decision["user_id"])
    per_user = {}
    for user, indices in sorted(users.items()):
        cutoffs = [_cutoff(observed[t]) for t in indices]
        known = [numeric for _, numeric in cutoffs if numeric is not None]
        transitions = Counter()
        for (_, earlier), (_, later) in zip(cutoffs, cutoffs[1:]):
            transitions["unknown" if earlier is None or later is None else "backward" if later < earlier else "equal" if later == earlier else "increasing"] += 1
        signatures = [tuple(record_id for record_id, record in clean_records[t].items() if record["type"] == "interaction") for t in indices]
        per_user[user] = {
            "decision_indices": indices,
            "cutoffs": [value for value, _ in cutoffs],
            "known_cutoffs": len(known),
            "distinct_known_cutoffs": len(set(known)),
            "increasing_transitions": transitions["increasing"],
            "equal_transitions": transitions["equal"],
            "backward_transitions": transitions["backward"],
            "unknown_transitions": transitions["unknown"],
            "repeated_history_signatures": len(signatures) - len(set(signatures)),
        }
    chronology = {
        "unique_users": len(users),
        "repeated_users": sum(len(indices) > 1 for indices in users.values()),
        "duplicate_user_decisions": t_count - len(users),
        "decisions_with_known_cutoff": sum(value["known_cutoffs"] for value in per_user.values()),
        "backward_transitions": sum(value["backward_transitions"] for value in per_user.values()),
        "equal_transitions": sum(value["equal_transitions"] for value in per_user.values()),
        "unknown_transitions": sum(value["unknown_transitions"] for value in per_user.values()),
        "per_user": per_user,
        "unique_record_ids": len(all_ids),
        "record_appearances": sum(appearances.values()),
        "repeated_record_ids": sum(count > 1 for count in appearances.values()),
        "reused_record_appearances": sum(count - 1 for count in appearances.values()),
        "record_presence_histogram": dict(sorted(Counter(appearances.values()).items())),
        "repeated_interaction_ids": sum(count > 1 for count in interaction_counts.values()),
        "cross_user_interaction_ids": sorted(record for record, owners in interaction_users.items() if len(owners) > 1),
    }
    baselines = screening.get("singleton_baselines", {})
    responses = screening.get("omission_responses", {})
    observed_rankings = pair.get("observed_rankings")
    if observed_rankings is not None:
        _validate_rankings(observed_rankings, "observed rankings", t_count, k)
    previous = Counter()
    by_decision = []
    by_record = defaultdict(Counter)
    for t, records in enumerate(presence):
        available = [set(_at(baselines.get(record, {}), t)) or set(_at(responses.get(record, {}), t)) for record in sorted(records)]
        available = [items for items in available if items]
        items = set(observed_rankings[t]) if observed_rankings is not None else available[0] if available else None
        if items is None or len(items) != k or any(value != items for value in available):
            raise ValueError("Observed Top-K membership is missing or inconsistent in screening diagnostics")
        counts = Counter()
        for record in sorted(records):
            for item in sorted(items):
                prior = previous[(record, item)]
                baseline = _at(baselines.get(record, {}), t).get(item)
                if baseline is not None and (not isinstance(baseline, (int, float)) or not math.isfinite(baseline) or not 0 <= baseline <= 1):
                    raise ValueError("Temporal baselines must be finite probabilities")
                values = {"slots": 1, "eligible_slots": int(prior > 0), "prior_eligible_appearances": prior, "baseline_values_available": int(baseline is not None), "nonzero_baselines": int(baseline is not None and baseline > 0), "nonzero_without_prior": int(prior == 0 and baseline is not None and baseline > 0)}
                counts.update(values)
                by_record[record].update(values)
                previous[(record, item)] += 1
        by_decision.append({"decision": t, **dict(counts)})
    totals = Counter()
    for counts in by_record.values():
        totals.update(counts)
    temporal = {
        **dict(totals),
        "eligible_fraction": _fraction(totals["eligible_slots"], totals["slots"]),
        "nonzero_baseline_fraction": _fraction(totals["nonzero_baselines"], totals["baseline_values_available"]),
        "nonzero_fraction_of_eligible_slots": _fraction(totals["nonzero_baselines"], totals["eligible_slots"]),
        "by_decision": by_decision,
        "repeated_or_selected_or_attacked_records": {record: dict(by_record[record]) for record in sorted(all_ids) if appearances[record] > 1 or record in omitted | attacked},
    }
    selected_cost = cost(omitted)
    stored_cost = screening.get("omission_cost")
    omissions = {
        "selected_ids": sorted(omitted),
        **identity_metrics(omitted, attacked, all_ids),
        "label_source": "evaluation.attacked_ids" if labels_present else "clean_observed_record_comparison",
        "selected_cost": selected_cost,
        "stored_cost": stored_cost,
        "stored_cost_matches": math.isclose(selected_cost, stored_cost, rel_tol=1e-12, abs_tol=1e-12) if stored_cost is not None else None,
        "selected_clean_cost": cost(omitted - attacked),
        "selected_attacked_cost": cost(omitted & attacked),
        "all_attacked_identity_cost": cost(attacked),
        "per_attacked_identity_cost": {record: cost({record}) for record in sorted(attacked)},
        "unchanged_appearances_removed_of_attacked_ids": sum(appearances[record] - changed_appearances[record] for record in omitted & attacked),
    }
    n_i = Counter(item for decision, ranking in zip(observed, screening["rankings"]) if len(decision["candidates"]) > k for item in ranking)
    h = calibration.get("h_alpha") if calibration is not None else None
    if h is not None and (isinstance(h, bool) or not isinstance(h, int) or not 0 <= h <= t_count):
        raise ValueError("h_alpha must be an integer in [0,T]")
    max_n = max(n_i.values(), default=0)
    family = {"max_i_n_i": max_n, "n_i": dict(sorted(n_i.items())), "h_alpha": h, "H_eta": selection.get("H_eta") if selection is not None else None, "q_alpha": calibration.get("q_alpha") if calibration is not None else None, "informative": max_n > h if h is not None else None, "completely_uninformative": max_n <= h if h is not None else None, "flexible_decisions": sum(len(decision["candidates"]) > k for decision in observed)}
    final = None
    if selection is not None:
        final = {"status": selection["status"], "regime": selection.get("regime"), "changed_decisions": None, "membership_changed_decisions": None, "order_only_changed_decisions": None, "replacement_item_appearances": None}
        if selection["status"] == "ok":
            _validate_rankings(selection["rankings"], "selected rankings", t_count, k)
            for decision, ranking in zip(observed, selection["rankings"]):
                if not set(ranking) <= set(decision["candidates"]):
                    raise ValueError("Selected rankings must respect fixed candidates")
            pairs = list(zip(screening["rankings"], selection["rankings"]))
            final.update(changed_decisions=sum(before != after for before, after in pairs), membership_changed_decisions=sum(set(before) != set(after) for before, after in pairs), order_only_changed_decisions=sum(before != after and set(before) == set(after) for before, after in pairs), replacement_item_appearances=sum(len(set(after) - set(before)) for before, after in pairs))
    effects = None
    if "clean_rankings" in pair:
        references = pair["clean_rankings"]
        _validate_rankings(references, "clean rankings", t_count, k)
        outputs = {"screening": screening["rankings"]}
        if observed_rankings is not None:
            outputs["observed"] = observed_rankings
        if selection is not None and selection["status"] == "ok":
            outputs["final_selection"] = selection["rankings"]
        effects = {}
        for name, ranking in outputs.items():
            effects[name] = {"gmax": gmax(ranking, references)}
            if "positives" in evaluation:
                effects[name].update(utility(ranking, evaluation["positives"]))
    return {"scope": "evaluator_only", "recurrence": recurrence, "chronology_and_reuse": chronology, "temporal": temporal, "omissions": omissions, "clean_family": family, "final_selection": final, "clean_relative_effects": effects}
