import copy
import gc
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .artifacts import digest, file_digest
from .continuations import LocalContinuations
from .defenses.prompt_guard import LocalPromptGuard, PromptGuardRecord, MODEL_ID, MODEL_REVISION
from .defenses.return_del import HopGraph, ReturnDel, _settings, observed_views
from .defenses.rewrite_detection import RewriteDetectionRecord
from .rq1_protocol import rq1_matrix
from .rq1_inference import ScalableSDPARanker
from .scalable_inference import ScalableLocalRanker, OUTPUT_PROTOCOL


BACKBONE_KEYS = {"repo_id", "revision", "model_identity_sha256", "dtype", "attention_implementation",
                 "output_protocol", "do_sample", "num_beams", "max_input_tokens", "max_new_tokens"}
GUARD_KEYS = {"repo_id", "revision", "model_identity_sha256", "threshold", "overlap", "max_input_tokens", "dtype"}
REWRITE_KEYS = {"text_field", "continuation_count", "ngram_min", "ngram_max", "score_weight_beta", "threshold",
                "identity_aggregation", "empty_ngram_policy", "recommendation_k", "frequency_denominator",
                "cohort_sha256", "cohort_manifest_sha256", "cohort_source_split", "splitter", "tokenizer", "continuation"}
RETURN_KEYS = {"threshold", "identity_aggregation", "unknown_item_policy", "max_hops", "graph_sha256",
               "graph_manifest_sha256", "graph_source_split"}
QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact(value, keys, name):
    _require(isinstance(value, dict) and set(value) == set(keys), f"{name} requires exactly these settings: {sorted(keys)}")


def _positive(value, name, minimum=1):
    _require(type(value) is int and value >= minimum, f"{name} must be an explicit integer >= {minimum}")


def _finite(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), f"{name} must be an explicit finite number")


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "An explicit lowercase SHA256 is required")
    return value


def _resolved(value):
    _require(value is not None, "Null/unresolved provider setting")
    if isinstance(value, dict):
        for child in value.values():
            _resolved(child)
    elif isinstance(value, list):
        for child in value:
            _resolved(child)
    elif isinstance(value, str):
        _require(value.strip().lower() not in ("unresolved", "tbd", "todo", "pending", "unknown"), "Unresolved provider setting")
    digest(value)


def _setting(settings, name, keys=None, detector=False):
    value = settings.get(name)
    _require(isinstance(value, dict), f"Missing frozen setting {name}")
    _resolved(value)
    _require(value.get("status") == "frozen", f"Unfrozen setting {name}")
    content = {key: value.get(key) for key in ("parameters", "provenance")}
    _require(all(isinstance(v, dict) and bool(v) for v in content.values()), f"Missing parameters/provenance for {name}")
    _require(value.get("sha256") == digest(content), f"Frozen setting checksum mismatch: {name}")
    if keys is not None:
        _exact(content["parameters"], keys, name)
    if detector:
        _settings(content["provenance"])
    return content["parameters"], content["provenance"], value["sha256"]


def _read_pinned(path, checksum):
    _sha(checksum)
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == checksum, f"Provider input checksum mismatch: {path}")
    return json.loads(raw)


def _snapshot(path, repo_id, revision, checksum):
    _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "An immutable revision is required")
    identity = _read_pinned(Path(path) / "model_identity.json", checksum)
    _require(identity.get("status") == "verified" and identity.get("repo_id") == repo_id and identity.get("revision") == revision,
             "Model identity differs from frozen provider setting")
    return identity


def _strict_views(decisions):
    _require(isinstance(decisions, list) and bool(decisions), "A nonempty observed-only decision list is required")
    for decision in decisions:
        _exact(decision, {"user_id", "candidates", "records"}, "Provider decision (no evaluator/protocol data)")
        for record in decision["records"]:
            _exact(record, {"id", "type", "fields"}, "Provider record")
    return observed_views(decisions)


def lossless_whitespace_split(text, *, prefix_fraction, rounding):
    _require(isinstance(text, str), "Rewrite text must be a string")
    _finite(prefix_fraction, "prefix_fraction")
    _require(0 < prefix_fraction < 1 and rounding == "floor", "Explicit fractional floor split is required")
    spans = list(re.finditer(r"\S+\s*", text))
    count = math.floor(len(spans) * prefix_fraction)
    _require(1 <= count < len(spans), "Configured split needs nonempty prefix and continuation tokens")
    boundary = spans[count - 1].end()
    return text[:boundary], text[boundary:]


def _provider_artifact(descriptor, parameters, prefix, dataset, kind):
    _exact(descriptor, {"path", "sha256", "manifest_path", "manifest_sha256"}, prefix + " input")
    _require(descriptor["sha256"] == parameters[prefix + "_sha256"] and
             descriptor["manifest_sha256"] == parameters[prefix + "_manifest_sha256"], "Provider input pins differ from frozen settings")
    source = _read_pinned(descriptor["manifest_path"], descriptor["manifest_sha256"])
    _require(source.get("dataset") == dataset and source.get("status") == "complete" and source.get("is_synthetic") is not True,
             "Provider source manifest must identify the completed real prepared dataset")
    value = _read_pinned(descriptor["path"], descriptor["sha256"])
    _require(value.get("schema_version") == 1 and value.get("kind") == kind and value.get("dataset") == dataset,
             "Provider artifact schema, kind or dataset mismatch")
    _require(value.get("source_split") in ("train", "development") and value.get("source_split") == parameters[prefix + "_source_split"],
             "Provider artifact requires explicitly frozen train/development split")
    _require(value.get("source_manifest_sha256") == descriptor["manifest_sha256"] and value.get("is_synthetic") is False,
             "Provider artifact needs pinned real source provenance")
    return value


def _load_graph(value, max_hops):
    _positive(max_hops, "max_hops")
    _require(type(value.get("max_hops")) is int and value["max_hops"] == max_hops, "Graph hop limit differs from frozen setting")
    items = value.get("catalog_item_ids")
    _require(isinstance(items, list) and bool(items) and all(isinstance(i, str) and i for i in items) and len(set(items)) == len(items),
             "Graph requires a unique explicit collaborative catalog")
    item_set = frozenset(items)
    counts, totals = {}, defaultdict(int)
    _require(isinstance(value.get("counts"), list), "Graph requires explicit count rows")
    for row in value["counts"]:
        _require(isinstance(row, list) and len(row) == 4, "Invalid graph count row")
        gap, first, second, count = row
        _positive(gap, "hop distance")
        _positive(count, "graph count")
        _require(gap <= max_hops and first in item_set and second in item_set, "Graph row exceeds declared coverage")
        key = (gap, first, second)
        _require(key not in counts, "Duplicate graph row")
        counts[key] = count
        totals[(gap, first)] += count
    _require(all(counts.get((gap, second, first)) == count for (gap, first, second), count in counts.items()),
             "RETURN graph must preserve symmetric upstream hop counts")
    serialized = [[gap, first, second, count] for (gap, first, second), count in sorted(counts.items())]
    _require(value.get("graph_sha256") == digest({"max_hops": max_hops, "counts": serialized, "catalog_item_ids": sorted(items)}),
             "Graph semantic checksum mismatch")
    _sha(value.get("corpus_sha256"))
    return HopGraph(MappingProxyType(counts), MappingProxyType(dict(totals)), item_set, max_hops,
                    value["source_split"], value["source_manifest_sha256"], value["corpus_sha256"], value["graph_sha256"])


def _validate_return_coverage(graph, sequences, unknown_item_policy="error"):
    _require(unknown_item_policy in ("error", "zero_support"), "Explicit RETURN unknown-item policy required")
    for decisions in sequences:
        for decision in decisions:
            history = [r for r in decision["records"] if r["type"] == "interaction"]
            _require(len(history) <= graph.max_hops + 1, "Observed history exceeds frozen graph hop coverage")
            if unknown_item_policy == "error":
                _require(all(r["fields"].get("item_id") in graph.item_ids for r in history), "Observed history item is missing from graph catalog coverage")


def _validate_rewrite_coverage(parameters, cohort, sequences, splitter):
    k, text_field = parameters["recommendation_k"], parameters["text_field"]
    _positive(k, "recommendation_k")
    _require(all(len(d["candidates"]) >= k for d in cohort), "Fixed cohort cannot fit recommendation_k")
    candidates = defaultdict(list)
    for index, decision in enumerate(cohort):
        for item in decision["candidates"]:
            candidates[item].append(index)
    for decisions in sequences:
        for decision in decisions:
            for record in decision["records"]:
                if record["type"] != "item_text":
                    continue
                _require(text_field in record["fields"], "Scored item text lacks the frozen field")
                prefix, tail = splitter(record["fields"][text_field])
                _require(prefix + tail == record["fields"][text_field], "Text split changed source bytes")
                if parameters["empty_ngram_policy"] == "error":
                    _require(len(tail.split()) >= parameters["ngram_max"], "Original continuation is too short for frozen n-gram orders")
                item = record["fields"].get("item_id")
                _require(bool(candidates.get(item)), "Fixed cohort has no candidate coverage for a scored item")
                for index in candidates[item]:
                    rows = [r for r in cohort[index]["records"] if r["id"] == record["id"]]
                    _require(len(rows) == 1 and rows[0]["type"] == "item_text" and
                             rows[0]["fields"].get("item_id") == item and text_field in rows[0]["fields"],
                             "Fixed cohort lacks scored identity coverage where the item is a candidate")


class RuntimeRewriteDetectionRecord(RewriteDetectionRecord):
    def __init__(self, *, runtime_ranker, runtime_continuation, factory_metadata, **kwargs):
        super().__init__(**kwargs)
        self._runtime_ranker = runtime_ranker
        self._runtime_continuation = runtime_continuation
        self._factory_metadata = factory_metadata

    def load(self):


        if self._runtime_continuation.ranker is not self._runtime_ranker:
            self._runtime_continuation.ranker.load()

    @property
    def metadata(self):
        result = super().metadata
        result["components"]["ranker"] = copy.deepcopy(self._runtime_ranker.metadata)
        result["components"]["continuation"] = copy.deepcopy(self._runtime_continuation.metadata)
        result["provider_factory"] = copy.deepcopy(self._factory_metadata)
        return result


@dataclass
class RQ1Providers:
    ranker: object
    detectors: dict
    provenance: dict
    continuation: object = None
    continuation_ranker: object = None
    closed: bool = False

    @property
    def metadata(self):
        if self.closed:
            return copy.deepcopy(self._closed_metadata)
        return {"factory": copy.deepcopy(self.provenance), "ranker": copy.deepcopy(self.ranker.metadata),
                "detectors": {name: copy.deepcopy(detector.metadata) for name, detector in self.detectors.items()}}

    def close(self):
        if self.closed:
            return
        self._closed_metadata = self.metadata
        cuda_loaded = self.ranker.model is not None and self.ranker.device.startswith("cuda")
        guard = self.detectors.get("PromptGuard2-Record")
        if guard is not None:
            scorer = guard.scorer
            cuda_loaded = cuda_loaded or (scorer.model is not None and scorer.device.startswith("cuda"))
            scorer.model = None
            scorer.tokenizer = None
            scorer.cache.close()
        self.ranker.model = None
        self.ranker.tokenizer = None
        self.ranker.cache.connection.close()
        if self.continuation_ranker is not None:
            auxiliary = self.continuation_ranker
            cuda_loaded = cuda_loaded or (auxiliary.model is not None and auxiliary.device.startswith("cuda"))
            auxiliary.model = None
            auxiliary.tokenizer = None
            auxiliary.cache.connection.close()
            self.continuation_ranker = None
        self.detectors.clear()
        self.continuation = None
        self.closed = True
        gc.collect()
        if cuda_loaded and "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()


def build_rq1_providers(frozen_settings, *, dataset, attack_family, model_path, revision, rank_cache_path,
                        device, guard_model_path, guard_revision, guard_cache_path, guard_device,
                        provider_inputs, observed_sequences, continuation_model=None):
    methods = rq1_matrix([dataset], [attack_family])[0]["methods"]
    _require(isinstance(provider_inputs, dict), "Explicit provider_inputs mapping required")
    expected_inputs = {"rewrite_cohort"} if "RewriteDetection-Record" in methods else ({"return_graph"} if "RETURN-Del" in methods else set())
    _require(set(provider_inputs) == expected_inputs, "Provide exactly the applicable frozen graph/cohort inputs")
    _require(isinstance(observed_sequences, list) and bool(observed_sequences), "Explicit observed sequence coverage inputs required")
    sequences = [_strict_views(value) for value in observed_sequences]
    for name in ("baseline:" + method for method in methods if method != "CREST"):
        _setting(frozen_settings, name)
    is_qwen35 = frozen_settings.get("backbone", {}).get("parameters", {}).get("repo_id") == "Qwen/Qwen3.5-9B"
    backbone_keys = BACKBONE_KEYS | {"enable_thinking", "linear_attention_backend"} if is_qwen35 else BACKBONE_KEYS
    backbone, _, backbone_hash = _setting(frozen_settings, "backbone", backbone_keys)
    _require(backbone["revision"] == revision, "Backbone revision override differs from frozen setting")
    allowed_attention = ("qwen35_text_sdpa_nonthinking",) if is_qwen35 else ("eager", "sdpa", "sdpa_last_token")
    _require(backbone["dtype"] == "bfloat16" and backbone["attention_implementation"] in allowed_attention and
             backbone["output_protocol"] == OUTPUT_PROTOCOL and backbone["do_sample"] is False and type(backbone["num_beams"]) is int and
             backbone["num_beams"] == 1, "The common BF16 eager/SDPA greedy scalable ranking protocol must be explicit")
    if is_qwen35:
        _require(backbone["enable_thinking"] is False and backbone["linear_attention_backend"] == "torch_reference",
                 "Qwen3.5 requires frozen non-thinking mode and torch reference linear attention")
    for name in ("max_input_tokens", "max_new_tokens"):
        _positive(backbone[name], name)
    _require(isinstance(device, str) and bool(device) and isinstance(guard_device, str) and bool(guard_device), "Explicit runtime devices required")
    _require(Path(rank_cache_path).resolve() != Path(guard_cache_path).resolve(), "Ranker and guard need separate explicit cache paths")
    _snapshot(model_path, backbone["repo_id"], revision, backbone["model_identity_sha256"])
    pg, pg_provenance, pg_hash = _setting(frozen_settings, "baseline:PromptGuard2-Record", GUARD_KEYS, detector=True)
    _require(pg["repo_id"] == MODEL_ID and pg["revision"] == guard_revision == MODEL_REVISION and
             type(pg["max_input_tokens"]) is int and pg["max_input_tokens"] == 512 and pg["dtype"] == "float32", "Exact native Prompt Guard 2 86M protocol is required")
    _finite(pg["threshold"], "PromptGuard threshold")
    _require(0 <= pg["threshold"] <= 1, "PromptGuard threshold must be a probability")
    _positive(pg["overlap"], "PromptGuard overlap", minimum=0)
    _require(pg["overlap"] < 510, "PromptGuard overlap exceeds native window")
    _snapshot(guard_model_path, MODEL_ID, guard_revision, pg["model_identity_sha256"])
    graph, cohort, rewrite, return_parameters = None, None, None, None
    _require(continuation_model is None or "RewriteDetection-Record" in methods,
             "A separate continuation model is only applicable to RewriteDetection")
    if "RETURN-Del" in methods:
        return_parameters, return_provenance, _ = _setting(frozen_settings, "baseline:RETURN-Del", RETURN_KEYS, detector=True)
        graph_json = _provider_artifact(provider_inputs["return_graph"], return_parameters, "graph", dataset, "rq1_return_hop_graph")
        graph = _load_graph(graph_json, return_parameters["max_hops"])
        _validate_return_coverage(graph, sequences, return_parameters["unknown_item_policy"])
        return_detector = ReturnDel(graph, **{key: return_parameters[key] for key in ("threshold", "identity_aggregation", "unknown_item_policy")},
                                    settings_provenance=return_provenance)
    if "RewriteDetection-Record" in methods:
        rewrite, rewrite_provenance, _ = _setting(frozen_settings, "baseline:RewriteDetection-Record", REWRITE_KEYS, detector=True)
        continuation = rewrite["continuation"]
        continuation_keys = {"provider", "acknowledge_unspecified_original_checkpoint", "seed", "temperature", "top_p", "max_new_tokens", "max_input_tokens"}
        separate = continuation.get("provider") == "separate_qwen2_5_raw"
        if separate:
            _exact(continuation, continuation_keys | {"repo_id", "revision", "model_identity_sha256", "attention_implementation"}, "continuation")
            _exact(continuation_model, {"model_path", "revision", "model_identity_sha256", "device"}, "continuation_model")
            _require(continuation["repo_id"] == QWEN_REPO and continuation["attention_implementation"] in ("sdpa_last_token", "sdpa_native_last_token"),
                     "Separate continuation must preserve the frozen Qwen2.5 model and explicit runtime")
            _require(all(continuation_model[k] == continuation[k] for k in ("revision", "model_identity_sha256")) and
                     continuation_model["device"] == device, "Separate continuation identity/device differs; one declared GPU is required")
            _snapshot(continuation_model["model_path"], QWEN_REPO, continuation["revision"], continuation["model_identity_sha256"])
        else:
            _exact(continuation, continuation_keys, "continuation")
            _require(backbone["repo_id"] == QWEN_REPO and continuation_model is None,
                     "The explicit raw-Qwen continuation provider requires the same Qwen backbone")
        _require(continuation["provider"] in ("shared_backbone_raw_qwen", "separate_qwen2_5_raw") and continuation["acknowledge_unspecified_original_checkpoint"] is True,
                 "Explicit acknowledgement of the unspecified original continuation checkpoint is required")
        for name in ("seed", "max_new_tokens", "max_input_tokens"):
            _positive(continuation[name], "continuation." + name, minimum=0 if name == "seed" else 1)
        for name in ("temperature", "top_p"):
            _finite(continuation[name], "continuation." + name)
        _require(0 < continuation["temperature"] <= 2 and 0 < continuation["top_p"] <= 1, "Invalid continuation sampling parameters")
        _exact(rewrite["splitter"], {"kind", "prefix_fraction", "rounding"}, "splitter")
        _require(rewrite["splitter"]["kind"] == "whitespace_boundary" and rewrite["tokenizer"] == {"kind": "python_str_split"},
                 "Only the explicitly declared lossless whitespace split and tokenization are implemented")
        split_settings = {key: rewrite["splitter"][key] for key in ("prefix_fraction", "rounding")}
        splitter = lambda text: lossless_whitespace_split(text, **split_settings)
        cohort_json = _provider_artifact(provider_inputs["rewrite_cohort"], rewrite, "cohort", dataset, "rq1_frequency_cohort")
        cohort = _strict_views(cohort_json["decisions"])
        _validate_rewrite_coverage(rewrite, cohort, sequences, splitter)
    provenance = {"implementation_sha256": file_digest(__file__), "dataset": dataset, "attack_family": attack_family,
                  "frozen_setting_hashes": {name: frozen_settings[name]["sha256"] for name in ["backbone"] +
                                            ["baseline:" + method for method in methods if method != "CREST"]},
                  "provider_inputs": copy.deepcopy(provider_inputs), "coverage_observed_sha256": digest(sequences),
                  "coverage_uses_evaluation_labels": False, "models_loaded_by_factory": False}
    if continuation_model is not None:
        provenance["continuation_model"] = copy.deepcopy(continuation_model)
    ranker_class = ScalableLocalRanker if backbone["attention_implementation"] == "eager" else ScalableSDPARanker
    if backbone["attention_implementation"] == "sdpa_last_token":
        from .rq1_last_token_inference import LastTokenSDPARanker
        ranker_class = LastTokenSDPARanker
    elif is_qwen35:
        from .qwen35_inference import Qwen35TextRanker
        ranker_class = Qwen35TextRanker
    ranker = ranker_class(model_path, revision, rank_cache_path, device=device, dtype=backbone["dtype"],
                                 max_input_tokens=backbone["max_input_tokens"], max_new_tokens=backbone["max_new_tokens"])
    bundle = RQ1Providers(ranker, {}, provenance)
    try:
        scorer = LocalPromptGuard(guard_model_path, guard_cache_path, revision=guard_revision, device=guard_device, overlap=pg["overlap"])
        bundle.detectors["PromptGuard2-Record"] = PromptGuardRecord(scorer, threshold=pg["threshold"], settings_provenance=pg_provenance)
        if graph is not None:
            bundle.detectors["RETURN-Del"] = return_detector
        if rewrite is not None:
            params = {key: rewrite["continuation"][key] for key in ("seed", "temperature", "top_p", "max_new_tokens", "max_input_tokens")}
            generator = ranker
            if continuation_model is not None:
                if rewrite["continuation"]["attention_implementation"] == "sdpa_last_token":
                    from .rq1_last_token_inference import LastTokenSDPARanker
                    generator_class = LastTokenSDPARanker
                else:
                    from .qwen35_inference import NativeQwen2ContinuationRanker
                    generator_class = NativeQwen2ContinuationRanker
                generator = generator_class(continuation_model["model_path"], continuation_model["revision"],
                    str(Path(rank_cache_path).with_name(Path(rank_cache_path).name + ".continuation.sqlite")),
                    device=continuation_model["device"], dtype="bfloat16", max_input_tokens=params["max_input_tokens"],
                    max_new_tokens=max(256, params["max_new_tokens"]))
                bundle.continuation_ranker = generator
            continuation = LocalContinuations(generator, **params)
            bundle.continuation = continuation
            constructor = {key: rewrite[key] for key in REWRITE_KEYS - {"splitter", "tokenizer", "continuation", "cohort_sha256", "cohort_manifest_sha256", "cohort_source_split"}}
            bundle.detectors["RewriteDetection-Record"] = RuntimeRewriteDetectionRecord(
                runtime_ranker=ranker, runtime_continuation=continuation, factory_metadata=provenance,
                continuation_provider=continuation, rank_provider=ranker.rank, split_text=splitter, tokenize=str.split,
                **constructor, frequency_cohort=cohort, cohort_source_split=rewrite["cohort_source_split"],
                cohort_manifest_sha256=rewrite["cohort_manifest_sha256"], settings_provenance=rewrite_provenance,
                component_provenance={"continuation": continuation.metadata, "ranker": ranker.metadata,
                                      "split_text": copy.deepcopy(rewrite["splitter"]), "tokenize": copy.deepcopy(rewrite["tokenizer"])})
        return bundle
    except BaseException:
        bundle.close()
        raise
