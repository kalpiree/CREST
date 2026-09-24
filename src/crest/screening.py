import math
from collections import Counter
from collections.abc import Callable

from .calibration import _positive_integer
from .metrics import _validate_rankings


def screen(
    record_ids: list[list[str]],
    rank_fn: Callable[[frozenset[str]], list[list[str]]],
    k: int,
    beta: float,
    penalty: float,
    beam_width: int,
    max_depth: int,
    temporal_baseline: bool = True,
    singleton_only: bool = False,
) -> dict:
    for value, name in ((k, "k"), (beam_width, "beam_width"), (max_depth, "max_depth")):
        _positive_integer(value, name)
    for value, name in ((beta, "beta"), (penalty, "penalty")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(temporal_baseline, bool) or not isinstance(singleton_only, bool):
        raise ValueError("ablation flags must be booleans")
    if not isinstance(record_ids, list) or not record_ids:
        raise ValueError("record_ids must be a nonempty decision sequence")
    for records in record_ids:
        if not isinstance(records, list) or not records:
            raise ValueError("each decision must contain at least one record")
        if any(not isinstance(record, str) or not record for record in records):
            raise ValueError("record identifiers must be nonempty strings")
        if len(records) != len(set(records)):
            raise ValueError("record identifiers must be distinct within each decision")
    if not callable(rank_fn):
        raise ValueError("rank_fn must be callable")
    t_count = len(record_ids)
    presence = [set(records) for records in record_ids]
    identities = sorted(set().union(*presence))
    rankings_cache = {}
    response_cache = {}
    objectives = {}

    def query(omissions):
        if omissions not in rankings_cache:
            output = rank_fn(omissions)
            _validate_rankings(output, "rank_fn output", t_count, k)
            rankings_cache[omissions] = [list(row) for row in output]
        return rankings_cache[omissions]

    observed = query(frozenset())

    def responses(omissions):
        if omissions not in response_cache:
            result = []
            for original, changed in zip(observed, query(omissions)):
                new_ranks = {item: rank for rank, item in enumerate(changed, 1)}
                result.append({
                    item: max(new_ranks.get(item, k + 1) - rank, 0) / (k + 1 - rank)
                    for rank, item in enumerate(original, 1)
                })
            response_cache[omissions] = result
        return response_cache[omissions]

    singleton_responses = {record: responses(frozenset([record])) for record in identities}
    baselines = {record: {} for record in identities}
    history = {}
    for t, records in enumerate(record_ids):
        for record in records:
            baselines[record][t] = {}
            for item in observed[t]:
                previous = history.get((record, item))
                if previous is None:
                    numerator, denominator = 0.0, 0.0
                    mean = 0.0
                else:
                    old_numerator, old_denominator, last_t = previous
                    mean = old_numerator / old_denominator
                    decay = math.exp(-beta * (t - last_t))
                    numerator = old_numerator * decay
                    denominator = old_denominator * decay
                baselines[record][t][item] = mean if temporal_baseline else 0.0
                history[(record, item)] = (numerator + singleton_responses[record][t][item], denominator + 1.0, t)

    def evaluate(omissions):
        if omissions not in objectives:
            totals = {}
            for t, delta in enumerate(responses(omissions)):
                relevant = sorted(omissions & presence[t])
                for item, response in delta.items():
                    baseline = min(1.0, sum(baselines[record][t][item] for record in relevant))
                    totals[item] = totals.get(item, 0.0) + max(response - baseline, 0.0)
            influence = max(totals.values(), default=0.0) / t_count
            cost = sum(len(omissions & records) / len(records) for records in presence) / t_count
            objectives[omissions] = (influence - penalty * cost, influence, cost)
        return objectives[omissions]

    def ordering(omissions):
        return (-evaluate(omissions)[0], len(omissions), tuple(sorted(omissions)))

    empty = frozenset()
    evaluate(empty)
    beam = sorted((frozenset([record]) for record in identities), key=ordering)[:beam_width]
    if not singleton_only:
        for depth in range(2, min(max_depth, len(identities)) + 1):
            expanded = {subset | {record} for subset in beam for record in identities if record not in subset}
            if not expanded:
                break
            beam = sorted(expanded, key=ordering)[:beam_width]
    best = min(objectives, key=ordering)
    objective, influence, cost = objectives[best]
    return {
        "omitted_ids": sorted(best),
        "observed_rankings": [list(row) for row in observed],
        "rankings": [list(row) for row in rankings_cache[best]],
        "omission_cost": cost,
        "objective": objective,
        "influence": influence,
        "query_sets": len(rankings_cache),
        "query_sets_by_depth": dict(sorted(Counter(map(len, rankings_cache)).items())),
        "singleton_baselines": baselines,
        "omission_responses": {
            record: {t: singleton_responses[record][t] for t, records in enumerate(presence) if record in records}
            for record in identities
        },
    }
