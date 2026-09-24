import copy
import json
import math
import re
import time

from .artifacts import digest, file_digest


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_TABLE = "qwen_masked_mean_embeddings_v1"
FAILURE_TABLE = "qwen_embedding_failures_v1"
PARAMETER_KEYS = {"pooling", "max_input_tokens", "truncation", "normalize", "add_special_tokens", "batch_size"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _config(value):
    _require(callable(getattr(value, "to_dict", None)), "Loaded model must expose its complete configuration")
    return _copy(value.to_dict())


def _validate_vector(vector, dimension):
    _require(isinstance(vector, list) and len(vector) == dimension and all(type(value) in (int, float) and math.isfinite(value) for value in vector),
             "Embedding vector has invalid dimension or nonfinite values")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    _require(abs(norm - 1.0) <= 1e-5, "Embedding vector is not normalized")


class QwenEmbeddings:
    def __init__(self, loaded_ranker, *, embedding_parameters, check=None):
        _require(isinstance(embedding_parameters, dict) and set(embedding_parameters) == PARAMETER_KEYS,
                 "Every embedding parameter must be explicit; missing/unknown parameters are rejected")
        parameters = _copy(embedding_parameters)
        _require(parameters["pooling"] == "masked_mean" and parameters["truncation"] is False and parameters["normalize"] is True and
                 type(parameters["add_special_tokens"]) is bool and type(parameters["batch_size"]) is int and parameters["batch_size"] == 1,
                 "Only explicitly masked-mean, normalized, untruncated, batch-one Qwen embeddings are supported")
        _require(type(parameters["max_input_tokens"]) is int and parameters["max_input_tokens"] > 0, "Explicit positive embedding input-token budget required")
        _require(check is None or callable(check), "Cooperative check must be callable")
        self.ranker = loaded_ranker
        self.check = check if check is not None else lambda: None
        self._model, self._tokenizer = getattr(loaded_ranker, "model", None), getattr(loaded_ranker, "tokenizer", None)
        _require(self._model is not None and self._tokenizer is not None, "Load the pinned ranker and tokenizer before constructing embeddings")
        self._decoder = getattr(self._model, "model", None)
        _require(callable(self._decoder) and getattr(self._model, "training", None) is False and getattr(self._decoder, "training", None) is False,
                 "Loaded Qwen causal model and base decoder must already be in eval mode")
        identity = getattr(loaded_ranker, "identity", {})
        revision = identity.get("revision")
        _require(identity.get("status") == "verified" and identity.get("repo_id") == MODEL_ID and isinstance(revision, str) and
                 re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "The verified immutable Qwen2.5-7B-Instruct checkpoint is required")
        inference = getattr(loaded_ranker, "metadata", {})
        _require(inference.get("model_id") == MODEL_ID and inference.get("revision") == revision and
                 inference.get("model_identity_hash") == digest(identity) and inference.get("dtype") == "bfloat16" and
                 isinstance(inference.get("runtime"), dict), "Finish loading the pinned BF16 runtime before freezing embeddings")
        _require(isinstance(getattr(loaded_ranker, "device", None), str) and bool(loaded_ranker.device), "Explicit loaded device required")
        self._forward = self._decoder.forward
        self._parameters = parameters
        self._device = loaded_ranker.device
        self._live = self._live_identity()
        self._dimension = self._live["model_config"].get("hidden_size")
        self._capacity = self._live["model_config"].get("max_position_embeddings")
        _require(type(self._dimension) is int and self._dimension > 0 and type(self._capacity) is int and self._capacity > 0,
                 "Loaded Qwen hidden size and context capacity must be explicit positive integers")
        self._metadata = {"model_id": MODEL_ID, "revision": revision, "implementation_sha256": file_digest(__file__),
            "provider_version": "qwen-base-decoder-masked-mean-v1", "inference": _copy(inference),
            "model_identity_sha256": digest(identity), "native_model_config": self._live["model_config"],
            "native_tokenizer": self._live["tokenizer"], "embedding_parameters": copy.deepcopy(parameters),
            "dimension": self._dimension, "forward": "Pinned Qwen base decoder last_hidden_state; no causal LM head or model.generate",
            "forward_parameters": {"use_cache": False, "return_dict": True, "output_hidden_states": False, "output_attentions": False},
            "pooling_precision": "float32 attention-mask-weighted sum divided by nonpadding token count, followed by float32 L2 normalization",
            "special_token_pooling": "Every unmasked encoded token participates; add_special_tokens is explicitly configured",
            "cache_namespace": CACHE_TABLE, "reproduction_status": "User-approved Qwen embedding substitution for original Gemma-1.1-2b-it; numerical or retrieval equivalence is not claimed"}
        self._metadata_hash, self._live_hash, self._parameter_hash = digest(self._metadata), digest(self._live), digest(parameters)
        self._connection = loaded_ranker.cache.connection
        self._connection.execute(f"CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self._connection.execute(f"CREATE TABLE IF NOT EXISTS {FAILURE_TABLE} (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self._connection.commit()
        self.calls = self.cache_hits = self.input_tokens = 0

    def _live_identity(self):
        return {"identity": _copy(self.ranker.identity), "inference": _copy(self.ranker.metadata), "device": self.ranker.device,
                "model_config": _config(self._model.config), "base_decoder_class": type(self._decoder).__name__,
                "tokenizer": {"class": type(self._tokenizer).__name__, "eos_token_id": self._tokenizer.eos_token_id,
                              "bos_token_id": self._tokenizer.bos_token_id, "pad_token_id": self._tokenizer.pad_token_id,
                              "all_special_ids": list(self._tokenizer.all_special_ids)}}

    def _unchanged(self):
        _require(self.ranker.model is self._model and self.ranker.tokenizer is self._tokenizer and self._model.model is self._decoder and
                 self._decoder.forward == self._forward and self._model.training is False and self._decoder.training is False,
                 "Frozen embedding model/tokenizer/base decoder changed")
        _require(digest(self._live_identity()) == self._live_hash and digest(self._metadata) == self._metadata_hash and
                 digest(self._parameters) == self._parameter_hash, "Frozen embedding inference metadata or settings changed")

    @property
    def metadata(self):
        self._unchanged()
        return copy.deepcopy(self._metadata)

    def _cached(self, table, key):
        row = self._connection.execute(f"SELECT payload FROM {table} WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        _require(payload.get("sha256") == digest({name: value for name, value in payload.items() if name != "sha256"}) and
                 payload.get("key") == key and payload.get("provider_sha256") == self._metadata_hash, "Corrupt or incompatible embedding cache entry")
        return payload

    def _persist(self, table, key, values):
        payload = dict(values, key=key, provider_sha256=self._metadata_hash)
        payload["sha256"] = digest(payload)
        with self._connection:
            existing = self._cached(table, key)
            _require(existing is None or table != CACHE_TABLE or existing["vector"] == payload["vector"],
                     "Conflicting embedding vectors for identical frozen inputs")
            self._connection.execute(f"INSERT OR IGNORE INTO {table} VALUES (?,?)", (key, json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def __call__(self, texts):
        _require(isinstance(texts, list) and bool(texts) and all(isinstance(text, str) and bool(text.strip()) for text in texts),
                 "Embedding requests require a nonempty list of nonblank strings")
        self._unchanged()
        result = []
        for text in texts:
            self.check()
            self._unchanged()
            inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=self._parameters["add_special_tokens"], padding=False, truncation=False)
            ids, mask = inputs["input_ids"], inputs["attention_mask"]
            _require(len(ids.shape) == 2 and ids.shape[0] == 1 and ids.shape[1] > 0 and mask.shape == ids.shape,
                     "Embedding tokenizer must return one nonempty prompt and matching attention mask")
            input_count = ids.shape[1]
            _require(input_count <= self._parameters["max_input_tokens"] and input_count <= self._capacity,
                     "Embedding input exceeds explicit/native context budget; no truncation")
            encoded = {name: tensor.tolist() for name, tensor in inputs.items()}
            _require(all(value in (0, 1) for row in encoded["attention_mask"] for value in row) and sum(encoded["attention_mask"][0]) > 0,
                     "Embedding attention mask must be binary with nonzero token coverage")
            query = {"provider": self._metadata, "text": text, "encoded_inputs": encoded}
            key = digest(query)
            failed = self._cached(FAILURE_TABLE, key)
            if failed is not None:
                raise RuntimeError("Preserved embedding failure; no automatic retry: " + failed["error"])
            cached = self._cached(CACHE_TABLE, key)
            if cached is not None:
                _validate_vector(cached.get("vector"), self._dimension)
                self.cache_hits += 1
                result.append(list(cached["vector"]))
                continue
            self.check()
            self._unchanged()
            import torch

            inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
            started = time.monotonic()
            self.calls += 1
            self.input_tokens += input_count
            try:
                with torch.inference_mode():
                    output = self._decoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=False, output_attentions=False)
                    hidden = output.last_hidden_state
                    _require(tuple(hidden.shape) == (1, input_count, self._dimension), "Base decoder returned unexpected hidden-state shape")
                    weights = inputs["attention_mask"].to(dtype=torch.float32).unsqueeze(-1)
                    pooled = (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1)
                    norm = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True)
                    _require(bool(torch.isfinite(pooled).all()) and bool(torch.isfinite(norm).all()) and bool((norm > 0).all()),
                             "Base decoder produced zero or nonfinite pooled embeddings")
                    vector = (pooled / norm)[0].cpu().tolist()
                _validate_vector(vector, self._dimension)
                self._persist(CACHE_TABLE, key, {"vector": vector, "query": query, "input_tokens": input_count,
                    "unmasked_tokens": int(sum(encoded["attention_mask"][0])), "seconds": time.monotonic() - started})
                self._unchanged()
                result.append(vector)
            except BaseException as error:
                self._persist(FAILURE_TABLE, key, {"error": str(error), "error_type": type(error).__name__, "query": query,
                                                  "seconds": time.monotonic() - started})
                raise
        return result

    def statistics(self):
        return {"model_calls": self.calls, "cache_hits": self.cache_hits, "input_tokens": self.input_tokens,
                "provider_sha256": self._metadata_hash}
