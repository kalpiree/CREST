import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from ..records import validate_decision


UPSTREAM_REVISION = "f56bc959890a39a8eae83097d041e33026c2495e"
UPSTREAM_URL = "https://github.com/Biglemon-Ning/RETURN"
ADAPTER_VERSION = "return-del-hopgraph-v1"


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _manifest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("A SHA-256 source manifest is required")
    return value


def _settings(value):
    value = copy.deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError("Explicit settings provenance is required")
    if value.get("mode") == "development_selected":
        if value.get("source_split") != "development":
            raise ValueError("Tuned settings must come from development data")
        _manifest(value.get("manifest_sha256"))
    elif value.get("mode") == "explicit_unvalidated":
        if not isinstance(value.get("description"), str) or not value["description"].strip():
            raise ValueError("Unvalidated settings require an explicit description")
    else:
        raise ValueError("Settings must be development_selected or explicit_unvalidated")
    _digest(value)
    return value


def observed_views(decisions):
    if not isinstance(decisions, (list, tuple)) or not decisions:
        raise ValueError("Pass a nonempty observed decision list, not a clean/evaluation bundle")
    views = []
    identities = {}
    for decision in decisions:
        if not isinstance(decision.get("user_id"), str) or not decision["user_id"]:
            raise ValueError("A nonempty user ID is required")
        view = {
            "user_id": decision["user_id"],
            "candidates": list(decision["candidates"]),
            "records": [{"id": record["id"], "type": record["type"], "fields": dict(record["fields"])} for record in decision["records"]],
        }
        validate_decision(view)
        for record in view["records"]:
            signature = (record["type"], tuple(record["fields"]), *(record["fields"].get(key) for key in ("item_id", "user_id", "timestamp")))
            previous = identities.setdefault(record["id"], signature)
            if previous != signature:
                raise ValueError("Trusted identity, reference, timestamp, or field schema changed across appearances")
        views.append(view)
    return views


def aggregate_scores(values, mode):
    if mode not in ("min", "max", "mean"):
        raise ValueError("identity_aggregation must explicitly be min, max, or mean")
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Finite nonempty appearance scores are required")
    return min(values) if mode == "min" else max(values) if mode == "max" else math.fsum(values) / len(values)


@dataclass(frozen=True)
class HopGraph:
    counts: object
    row_totals: object
    item_ids: frozenset
    max_hops: int
    source_split: str
    source_manifest_sha256: str
    corpus_sha256: str
    graph_sha256: str

    def provenance(self):
        return {
            "upstream_url": UPSTREAM_URL,
            "upstream_revision": UPSTREAM_REVISION,
            "source_split": self.source_split,
            "source_manifest_sha256": self.source_manifest_sha256,
            "corpus_sha256": self.corpus_sha256,
            "graph_sha256": self.graph_sha256,
            "max_hops": self.max_hops,
            "item_count": len(self.item_ids),
            "representation": "sparse integer counts; Python floating-point score accumulation",
        }


def build_hop_graph(histories, *, source_split, source_manifest_sha256, max_hops=300, catalog_item_ids=None):
    if source_split not in ("train", "development"):
        raise ValueError("Collaborative graphs may use only train/development histories")
    _manifest(source_manifest_sha256)
    if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
        raise ValueError("max_hops must be a positive integer")
    sequences = []
    counts = defaultdict(int)
    vocabulary = set(catalog_item_ids or ())
    for history in histories:
        if not isinstance(history, (list, tuple)) or any(not isinstance(item, str) or not item for item in history):
            raise ValueError("Corpus entries must be chronological lists of item ID strings only")
        history = tuple(history)
        sequences.append(history)
        vocabulary.update(history)
        for gap in range(1, min(len(history), max_hops + 1)):
            for left in range(len(history) - gap):
                first, second = history[left], history[left + gap]
                counts[(gap, first, second)] += 1
                counts[(gap, second, first)] += 1
    if not sequences or not any(sequences) or not vocabulary or any(not isinstance(item, str) or not item for item in vocabulary):
        raise ValueError("A nonempty corpus/catalog of valid item IDs is required")
    totals = defaultdict(int)
    for (gap, first, _), count in counts.items():
        totals[(gap, first)] += count
    serialized = [[gap, first, second, count] for (gap, first, second), count in sorted(counts.items())]
    return HopGraph(
        MappingProxyType(dict(counts)), MappingProxyType(dict(totals)), frozenset(vocabulary), max_hops,
        source_split, source_manifest_sha256,
        _digest({"histories": sequences, "catalog_item_ids": sorted(vocabulary)}),
        _digest({"max_hops": max_hops, "counts": serialized, "catalog_item_ids": sorted(vocabulary)}),
    )


def upstream_support_scores(graph, item_ids, *, unknown_item_policy):
    if unknown_item_policy not in ("error", "zero_support"):
        raise ValueError("unknown_item_policy must explicitly be error or zero_support")
    if not isinstance(item_ids, (list, tuple)) or any(not isinstance(item, str) or not item for item in item_ids):
        raise ValueError("History must contain item ID strings")
    if len(item_ids) > graph.max_hops + 1:
        raise ValueError("History exceeds the graph's hop coverage; do not silently truncate it")
    unknown = set(item_ids) - graph.item_ids
    if unknown and unknown_item_policy == "error":
        raise ValueError("History contains items absent from the declared collaborative catalog")
    scores = [0.0] * len(item_ids)
    for gap in range(1, len(item_ids)):
        for left in range(len(item_ids) - gap):
            first, second = item_ids[left], item_ids[left + gap]
            denominator = graph.row_totals.get((gap, first), 0)
            if denominator:
                value = graph.counts.get((gap, first, second), 0) / denominator
                scores[left] += value
                scores[left + gap] += value
    return scores


class ReturnDel:
    def __init__(self, graph, *, threshold, identity_aggregation, unknown_item_policy, settings_provenance):
        if not isinstance(graph, HopGraph) or graph.source_split not in ("train", "development"):
            raise ValueError("A train/development HopGraph is required")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("A finite nonnegative development/configuration threshold is required")
        aggregate_scores([0.0], identity_aggregation)
        if unknown_item_policy not in ("error", "zero_support"):
            raise ValueError("Resolve the unknown-item policy explicitly")
        self.graph = graph
        self.threshold = threshold
        self.identity_aggregation = identity_aggregation
        self.unknown_item_policy = unknown_item_policy
        self.settings_provenance = _settings(settings_provenance)

    @property
    def metadata(self):
        return {
            "method": "RETURN-Del", "adapter_version": ADAPTER_VERSION,
            "is_full_upstream_reproduction": False, "graph": self.graph.provenance(),
            "threshold": self.threshold, "threshold_comparison": "strictly_less_than",
            "identity_aggregation": self.identity_aggregation, "unknown_item_policy": self.unknown_item_policy,
            "settings": copy.deepcopy(self.settings_provenance),
            "changes": ["sparse graph storage", "identity-wide deletion", "explicit support threshold instead of sampled weak-position budget", "no replacement", "no ensemble", "fixed CREST candidates/backbone"],
        }

    def detect(self, decisions):
        views = observed_views(decisions)
        identity_values = defaultdict(list)
        appearance_scores = []
        for index, decision in enumerate(views):
            records = [record for record in decision["records"] if record["type"] == "interaction"]
            times = []
            for record in records:
                fields = record["fields"]
                if fields.get("user_id") != decision["user_id"]:
                    raise ValueError("Interaction user reference differs from the decision user")
                if not fields.get("item_id") or not fields.get("timestamp"):
                    raise ValueError("Interaction item and timestamp references are required")
                try:
                    times.append(datetime.fromisoformat(fields["timestamp"]))
                except ValueError as error:
                    raise ValueError("Interaction timestamps must use ISO format") from error
            try:
                chronological = all(first <= second for first, second in zip(times, times[1:]))
            except TypeError as error:
                raise ValueError("Do not mix timezone-aware and naive interaction times") from error
            if not chronological:
                raise ValueError("Interaction records must already be in chronological order")
            scores = upstream_support_scores(self.graph, [record["fields"]["item_id"] for record in records], unknown_item_policy=self.unknown_item_policy)
            for record, score in zip(records, scores):
                identity_values[record["id"]].append(score)
                appearance_scores.append({"decision_index": index, "record_id": record["id"], "score": score})
        scores = {identity: aggregate_scores(values, self.identity_aggregation) for identity, values in sorted(identity_values.items())}
        return {
            "omitted_ids": [identity for identity, score in scores.items() if score < self.threshold],
            "identity_scores": scores,
            "appearance_scores": appearance_scores,
            "provenance": {**self.metadata, "observed_sha256": _digest(views)},
        }
