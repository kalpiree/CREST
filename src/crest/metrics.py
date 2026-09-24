import math


def _validate_rankings(rankings, name, expected_t=None, expected_k=None):
    if not isinstance(rankings, list) or not rankings:
        raise ValueError(f"{name} must be a nonempty list of rankings")
    if expected_t is not None and len(rankings) != expected_t:
        raise ValueError(f"{name} has the wrong number of decisions")
    if not isinstance(rankings[0], list):
        raise ValueError(f"{name} must contain lists")
    k = expected_k if expected_k is not None else len(rankings[0])
    if k < 1:
        raise ValueError("rankings must contain at least one item")
    for ranking in rankings:
        if not isinstance(ranking, list) or len(ranking) != k:
            raise ValueError(f"{name} must contain exactly {k} items per decision")
        if any(not isinstance(item, str) or not item for item in ranking):
            raise ValueError("item identifiers must be nonempty strings")
        if len(set(ranking)) != len(ranking):
            raise ValueError("item identifiers must be distinct within each ranking")
    return k


def gmax(output: list[list[str]], clean: list[list[str]]) -> float:
    k = _validate_rankings(output, "output")
    _validate_rankings(clean, "clean", len(output), k)
    counts = {}
    for ranking, reference in zip(output, clean):
        for item in set(ranking) - set(reference):
            counts[item] = counts.get(item, 0) + 1
    return max(counts.values(), default=0) / len(output)


def utility(output: list[list[str]], positives: list[str]) -> dict:
    _validate_rankings(output, "output")
    if not isinstance(positives, list) or len(positives) != len(output):
        raise ValueError("positives must contain one item per decision")
    if any(not isinstance(item, str) or not item for item in positives):
        raise ValueError("positive item identifiers must be nonempty strings")
    hits = 0
    discounted = 0.0
    for ranking, positive in zip(output, positives):
        if positive in ranking:
            hits += 1
            discounted += 1.0 / math.log2(ranking.index(positive) + 2)
    return {"recall": hits / len(output), "ndcg": discounted / len(output)}


def identity_metrics(omitted: set[str], attacked: set[str], all_ids: set[str]) -> dict:
    for ids in (omitted, attacked, all_ids):
        if not isinstance(ids, (set, frozenset)):
            raise ValueError("record identities must be sets")
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("record identifiers must be nonempty strings")
    if not omitted <= all_ids or not attacked <= all_ids:
        raise ValueError("omitted and attacked identities must belong to all_ids")
    tp = len(omitted & attacked)
    fp = len(omitted - attacked)
    fn = len(attacked - omitted)
    unmodified = len(all_ids - attacked)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unmodified": unmodified,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if attacked else None,
        "fpr": fp / unmodified if unmodified else None,
    }
