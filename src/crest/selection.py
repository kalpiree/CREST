import networkx as nx

from .calibration import _positive_integer, promotion_budget
from .metrics import _validate_rankings


def select(rankings: list[list[str]], candidates: list[list[str]], k: int, eta: float, h_alpha: int) -> dict:
    _positive_integer(k, "k")
    _validate_rankings(rankings, "rankings", expected_k=k)
    t_count = len(rankings)
    if not isinstance(candidates, list) or len(candidates) != t_count:
        raise ValueError("candidates must contain one candidate list per decision")
    if isinstance(h_alpha, bool) or not isinstance(h_alpha, int) or not 0 <= h_alpha <= t_count:
        raise ValueError("h_alpha must be an integer between 0 and T")
    for ranking, pool in zip(rankings, candidates):
        if not isinstance(pool, list) or len(pool) < k:
            raise ValueError("each candidate list must contain at least k items")
        if any(not isinstance(item, str) or not item for item in pool):
            raise ValueError("candidate identifiers must be nonempty strings")
        if len(pool) != len(set(pool)):
            raise ValueError("candidate identifiers must be distinct within each decision")
        if not set(ranking) <= set(pool):
            raise ValueError("rankings must be subsets of the corresponding candidates")
    budget = promotion_budget(t_count, eta)
    if budget >= h_alpha:
        return {"status": "ok", "rankings": [list(row) for row in rankings], "regime": "passthrough", "H_eta": budget}
    flexible = [t for t, pool in enumerate(candidates) if len(pool) > k]
    result = [list(row) for row in rankings]
    if not flexible:
        return {"status": "ok", "rankings": result, "regime": "occurrence_cap", "H_eta": budget}
    if budget == 0:
        return {"status": "infeasible", "rankings": [], "regime": "occurrence_cap", "H_eta": budget}
    graph = nx.DiGraph()
    source, sink = ("source",), ("sink",)
    required = len(flexible) * k
    graph.add_node(source, demand=-required)
    graph.add_node(sink, demand=required)
    tie_scale = k * sum(len(candidates[t]) - 1 for t in flexible) + 1
    for t in flexible:
        decision = ("decision", t)
        graph.add_node(decision, demand=0)
        graph.add_edge(source, decision, capacity=k, weight=0)
        ranks = {item: rank for rank, item in enumerate(rankings[t], 1)}
        for position, item in enumerate(candidates[t]):
            item_node = ("item", item)
            if item_node not in graph:
                graph.add_node(item_node, demand=0)
                graph.add_edge(item_node, sink, capacity=budget, weight=0)
            graph.add_edge(decision, item_node, capacity=1, weight=ranks.get(item, k + 1) * tie_scale + position)
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible:
        return {"status": "infeasible", "rankings": [], "regime": "occurrence_cap", "H_eta": budget}
    for t in flexible:
        ranks = {item: rank for rank, item in enumerate(rankings[t], 1)}
        positions = {item: position for position, item in enumerate(candidates[t])}
        chosen = [item for item in candidates[t] if flow[("decision", t)].get(("item", item), 0)]
        result[t] = sorted(chosen, key=lambda item: (ranks.get(item, k + 1), positions[item]))
    return {"status": "ok", "rankings": result, "regime": "occurrence_cap", "H_eta": budget}
