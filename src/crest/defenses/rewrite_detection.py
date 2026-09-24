import copy
import json
import math
from collections import defaultdict

from ..records import parse_ranking
from .return_del import _digest, _manifest, _settings, aggregate_scores, observed_views


PAPER_URL = "https://arxiv.org/html/2409.11690v3"
ADAPTER_VERSION = "rewrite-detection-record-formulas-v1"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tokens(text, tokenize):
    tokens = tokenize(text)
    if not isinstance(tokens, (list, tuple)) or any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("The explicit tokenizer must return a list/tuple of nonempty strings")
    return tuple(tokens)


def continuation_overlap(original_tail, generated_tails, *, tokenize, ngram_min, ngram_max, empty_ngram_policy):
    _positive_integer(ngram_min, "ngram_min")
    _positive_integer(ngram_max, "ngram_max")
    if ngram_max < ngram_min or empty_ngram_policy not in ("error", "skip"):
        raise ValueError("Resolve a valid n-gram range and explicit empty_ngram_policy")
    if not isinstance(original_tail, str) or not isinstance(generated_tails, (list, tuple)) or not generated_tails or any(not isinstance(tail, str) for tail in generated_tails):
        raise ValueError("Original and generated continuations must be strings")
    original = _tokens(original_tail, tokenize)
    reference = {n: {original[index:index + n] for index in range(len(original) - n + 1)} for n in range(ngram_min, ngram_max + 1)}
    if empty_ngram_policy == "error" and any(not values for values in reference.values()):
        raise ValueError("Original continuation is too short for the configured n-gram orders")
    sample_scores = []
    for tail in generated_tails:
        generated = _tokens(tail, tokenize)
        score = 0.0
        for n, expected in reference.items():
            if expected:
                actual = {generated[index:index + n] for index in range(len(generated) - n + 1)}
                score += len(actual & expected) / len(expected)
        sample_scores.append(score)
    return math.fsum(sample_scores) / len(sample_scores)


def recommendation_frequency_change(original_count, replacement_counts, *, denominator):
    if isinstance(original_count, bool) or not isinstance(original_count, int) or original_count < 0:
        raise ValueError("The original recommendation count must be a nonnegative integer")
    if not isinstance(replacement_counts, (list, tuple)) or not replacement_counts or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in replacement_counts):
        raise ValueError("Replacement recommendation counts must be nonnegative integers")
    if isinstance(denominator, bool) or not isinstance(denominator, (int, float)) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("A finite positive recommendation-frequency denominator is required")
    if max([original_count, *replacement_counts]) > denominator:
        raise ValueError("Recommendation counts exceed the declared denominator")
    return math.fsum(abs(original_count - value) / denominator for value in replacement_counts) / len(replacement_counts)


class RewriteDetectionRecord:
    def __init__(self, *, continuation_provider, rank_provider, split_text, tokenize, text_field, continuation_count, ngram_min, ngram_max, score_weight_beta, threshold, identity_aggregation, empty_ngram_policy, recommendation_k, frequency_denominator, frequency_cohort, cohort_source_split, cohort_manifest_sha256, settings_provenance, component_provenance):
        if not all(callable(value) for value in (continuation_provider, rank_provider, split_text, tokenize)):
            raise ValueError("Explicit continuation, ranking, split, and tokenization providers are required")
        if not isinstance(text_field, str) or not text_field or text_field in ("item_id", "user_id", "timestamp"):
            raise ValueError("text_field must select editable content, not a trusted reference")
        _positive_integer(continuation_count, "continuation_count")
        _positive_integer(ngram_min, "ngram_min")
        _positive_integer(ngram_max, "ngram_max")
        _positive_integer(recommendation_k, "recommendation_k")
        if ngram_max < ngram_min or empty_ngram_policy not in ("error", "skip"):
            raise ValueError("Resolve the n-gram range and empty-continuation behavior")
        if isinstance(score_weight_beta, bool) or not isinstance(score_weight_beta, (int, float)) or not math.isfinite(score_weight_beta) or score_weight_beta <= 0:
            raise ValueError("A finite positive beta is required to retain the frequency-change component")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("An explicit finite decision threshold is required")
        aggregate_scores([0.0], identity_aggregation)
        if frequency_denominator not in ("lists", "slots"):
            raise ValueError("Resolve reclistnum explicitly as lists or slots")
        if cohort_source_split not in ("train", "development"):
            raise ValueError("The fixed frequency-query cohort must come from train/development data")
        _manifest(cohort_manifest_sha256)
        cohort = observed_views(frequency_cohort)
        if any(len(decision["candidates"]) < recommendation_k for decision in cohort):
            raise ValueError("Every cohort candidate set must fit recommendation_k")
        if not isinstance(component_provenance, dict) or any(not component_provenance.get(key) for key in ("continuation", "ranker", "split_text", "tokenize")):
            raise ValueError("Record model, ranker, splitter, and tokenizer provenance")
        continuation_meta = component_provenance["continuation"]
        if not isinstance(continuation_meta, dict) or any(not continuation_meta.get(key) for key in ("model_id", "revision")) or "generation_parameters" not in continuation_meta:
            raise ValueError("Continuation model/revision and generation parameters must be explicit")
        self.continuation_provider = continuation_provider
        self.rank_provider = rank_provider
        self.split_text = split_text
        self.tokenize = tokenize
        self.text_field = text_field
        self.continuation_count = continuation_count
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.score_weight_beta = score_weight_beta
        self.threshold = threshold
        self.identity_aggregation = identity_aggregation
        self.empty_ngram_policy = empty_ngram_policy
        self.recommendation_k = recommendation_k
        self.frequency_denominator = frequency_denominator
        self._cohort = cohort
        self._cohort_sha256 = _digest(cohort)
        self.cohort_source_split = cohort_source_split
        self.cohort_manifest_sha256 = cohort_manifest_sha256
        self.settings_provenance = _settings(settings_provenance)
        self.component_provenance = copy.deepcopy(component_provenance)
        _digest(self.component_provenance)

    @property
    def metadata(self):
        return {
            "method": "RewriteDetection-Record", "adapter_version": ADAPTER_VERSION,
            "paper": PAPER_URL, "upstream_code_revision": None, "is_full_upstream_reproduction": False,
            "cohort_sha256": self._cohort_sha256, "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "cohort_source_split": self.cohort_source_split,
            "settings": copy.deepcopy(self.settings_provenance), "components": copy.deepcopy(self.component_provenance),
            "resolved_parameters": {"text_field": self.text_field, "continuation_count": self.continuation_count, "ngram_min": self.ngram_min, "ngram_max": self.ngram_max, "empty_ngram_policy": self.empty_ngram_policy, "score_weight_beta": self.score_weight_beta, "threshold": self.threshold, "threshold_comparison": "strictly_greater_than", "identity_aggregation": self.identity_aggregation, "recommendation_k": self.recommendation_k, "frequency_denominator": self.frequency_denominator},
            "changes": ["identity-wide record deletion instead of item filtering", "fixed-candidate frozen-ranker query adapter", "explicit repeated-appearance aggregation", "explicit continuation/cohort/normalization settings"],
        }

    def _frequency_counts(self, record, texts):
        item_id = record["fields"].get("item_id")
        if not item_id:
            raise ValueError("Text records require their trusted item_id reference")
        eligible = []
        for index, decision in enumerate(self._cohort):
            if item_id not in decision["candidates"]:
                continue
            matches = [entry for entry in decision["records"] if entry["id"] == record["id"]]
            if len(matches) != 1 or matches[0]["type"] != "item_text" or matches[0]["fields"].get("item_id") != item_id or self.text_field not in matches[0]["fields"]:
                raise ValueError("The fixed cohort must represent the scored identity whenever its item is a candidate")
            eligible.append(index)
        if not eligible:
            raise ValueError("No eligible item-text query in the fixed cohort; cannot fabricate a zero frequency score")
        counts = []
        query_count = 0
        for text in texts:
            count = 0
            for index in eligible:
                decision = copy.deepcopy(self._cohort[index])
                for entry in decision["records"]:
                    if entry["id"] == record["id"]:
                        entry["fields"][self.text_field] = text
                candidates = list(decision["candidates"])
                ranking = self.rank_provider(decision, self.recommendation_k)
                ranking = parse_ranking(json.dumps(ranking), candidates, self.recommendation_k)
                count += item_id in ranking
                query_count += 1
            counts.append(count)
        denominator = len(self._cohort) * (self.recommendation_k if self.frequency_denominator == "slots" else 1)
        return counts, denominator, eligible, query_count

    def _score_record(self, record):
        text = record["fields"][self.text_field]
        pieces = self.split_text(text)
        if not isinstance(pieces, (list, tuple)) or len(pieces) != 2 or any(not isinstance(piece, str) for piece in pieces) or "".join(pieces) != text:
            raise ValueError("split_text must return an exact lossless (prefix, continuation) split")
        prefix, tail = pieces
        generated = self.continuation_provider(prefix, self.continuation_count)
        if not isinstance(generated, (list, tuple)) or len(generated) != self.continuation_count or any(not isinstance(value, str) for value in generated):
            raise ValueError("Continuation provider must return exactly the configured number of tail strings")
        overlap = continuation_overlap(tail, generated, tokenize=self.tokenize, ngram_min=self.ngram_min, ngram_max=self.ngram_max, empty_ngram_policy=self.empty_ngram_policy)
        counts, denominator, eligible, query_count = self._frequency_counts(record, [text, *(prefix + continuation for continuation in generated)])
        frequency = recommendation_frequency_change(counts[0], counts[1:], denominator=denominator)
        return {
            "score": overlap + self.score_weight_beta * frequency,
            "S_A": overlap, "S_R": frequency,
            "original_recommendation_count": counts[0], "replacement_recommendation_counts": counts[1:],
            "frequency_denominator": denominator, "eligible_cohort_indices": eligible,
            "cohort_size": len(self._cohort), "rank_queries": query_count,
            "text_sha256": _digest(text), "continuations_sha256": _digest(list(generated)),
        }

    def detect(self, decisions):
        views = observed_views(decisions)
        if _digest(self._cohort) != self._cohort_sha256:
            raise ValueError("The declared frequency-query cohort was mutated")
        identity_values = defaultdict(list)
        appearance_scores = []
        evaluated = {}
        for index, decision in enumerate(views):
            for record in decision["records"]:
                if record["type"] != "item_text":
                    continue
                if self.text_field not in record["fields"]:
                    raise ValueError("A scored item-text record lacks the configured text field")
                key = (record["id"], record["fields"][self.text_field])
                if key not in evaluated:
                    evaluated[key] = self._score_record(record)
                result = evaluated[key]
                identity_values[record["id"]].append(result["score"])
                appearance_scores.append({"decision_index": index, "record_id": record["id"], **copy.deepcopy(result)})
        scores = {identity: aggregate_scores(values, self.identity_aggregation) for identity, values in sorted(identity_values.items())}
        return {
            "omitted_ids": [identity for identity, score in scores.items() if score > self.threshold],
            "identity_scores": scores, "appearance_scores": appearance_scores,
            "rank_queries": sum(result["rank_queries"] for result in evaluated.values()),
            "continuation_requests": len(evaluated),
            "provenance": {**self.metadata, "observed_sha256": _digest(views)},
        }
