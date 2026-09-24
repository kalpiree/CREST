import copy
import json
import math
import re
import time

from .artifacts import digest, file_digest
from . import constrained_inference, keyword_inference, scalable_inference


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_TABLE = "qwen_attack_raw_responses_v2_keyword_format"
FAILURE_TABLE = "qwen_attack_generation_failures_v2_keyword_format"
PARAMETER_KEYS = {"do_sample", "num_beams", "temperature", "top_p", "top_k",
                  "max_input_tokens", "max_new_tokens", "use_cache"}


class SourcePrefixConstraint:


    def __init__(self, tokenizer, prefix, allow_no, maximum):
        self.prefix = prefix
        self.tokens = tokenizer.encode(prefix, add_special_tokens=False)
        self.no = tokenizer.encode('No', add_special_tokens=False) if allow_no else None
        self.eos = tokenizer.eos_token_id
        self.vocabulary = list(range(len(tokenizer)))
        _require(self.tokens and len(self.tokens) + 2 <= maximum, 'Source prefix leaves no room for rewriting')
        _require(tokenizer.decode(self.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False) == prefix,
                 'Source prefix must round-trip through native tokenizer exactly')
        self.metadata = {'kind': 'qwen-source-prefix-v1', 'prefix': prefix, 'token_ids': self.tokens,
                         'allow_discussion_no': allow_no, 'post_hoc_repair': False}

    def allowed(self, generated):
        if generated[:len(self.tokens)] == self.tokens:
            return self.vocabulary
        choices = set()
        for sequence in [self.tokens] + ([self.no] if self.no is not None else []):
            if generated == sequence:
                choices.add(self.eos)
            elif generated == sequence[:len(generated)] and len(generated) < len(sequence):
                choices.add(sequence[len(generated)])
        _require(bool(choices), 'Generated tokens departed from the source prefix constraint')
        return sorted(choices)

    def prefix_allowed_tokens_fn(self, input_count):
        return lambda batch_id, tokens: self.allowed(tokens.tolist()[input_count:])

    def validate_completed(self, tokens, response):
        if self.no is not None and response.strip().lower() == 'no':
            return
        _require(tokens[:len(self.tokens)] == self.tokens and response.startswith(self.prefix),
                 'Native generation failed the frozen source prefix constraint')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an explicit integer >= {minimum}")


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _object_config(value, name):
    _require(callable(getattr(value, "to_dict", None)), f"Loaded {name} must expose its complete configuration")
    return _json_copy(value.to_dict())


def _request(request):
    _require(isinstance(request, dict), "Attacker request must be a mapping")
    formats = {"response_format", "output_schema"} & set(request)
    _require(len(formats) == 1 and set(request) == {"stage", "system", "inputs", "seed"} | formats,
             "Attacker request requires stage, system, inputs, seed and exactly one response format/schema")
    _require(isinstance(request["system"], str) and bool(request["system"].strip()), "A nonempty explicit system instruction is required")
    _require(isinstance(request["inputs"], dict), "Attacker inputs must be a JSON mapping")
    stage = request["stage"]
    _require(isinstance(stage, (str, list)) and bool(stage), "A nonempty stage identity is required")
    _integer(request["seed"], "request seed")
    _require(request["seed"] < 2**63, "Request seed exceeds the declared 63-bit range")
    name = next(iter(formats))
    _require(isinstance(request[name], (str, dict)) and bool(request[name]), "An explicit response format/schema is required")
    return _json_copy(request), name


class QwenAttackProvider:
    def __init__(self, loaded_ranker, *, generation_parameters, check=None):
        _require(isinstance(generation_parameters, dict) and set(generation_parameters) == PARAMETER_KEYS,
                 "Every attacker generation parameter must be explicit; missing/unknown parameters are rejected")
        parameters = _json_copy(generation_parameters)
        _require(type(parameters["do_sample"]) is bool and type(parameters["num_beams"]) is int and
                 parameters["num_beams"] == 1 and parameters["use_cache"] is True,
                 "Attacker generation requires explicit sampling choice, one beam and use_cache=True")
        for name in ("max_input_tokens", "max_new_tokens"):
            _integer(parameters[name], name, 1)
        _integer(parameters["top_k"], "top_k")
        for name in ("temperature", "top_p"):
            _require(type(parameters[name]) in (int, float) and math.isfinite(parameters[name]), f"{name} must be finite and explicit")
        _require(parameters["temperature"] > 0 and 0 < parameters["top_p"] <= 1, "Invalid explicit temperature/top_p")
        _require(check is None or callable(check), "Cooperative check must be callable")
        self.ranker = loaded_ranker
        self.check = check if check is not None else lambda: None
        self._model = getattr(loaded_ranker, "model", None)
        self._tokenizer = getattr(loaded_ranker, "tokenizer", None)
        _require(self._model is not None and self._tokenizer is not None, "Load the ranker and native tokenizer before freezing the attacker")
        _require(getattr(self._model, "training", None) is False, "Attacker model must already be in eval mode")
        identity = getattr(loaded_ranker, "identity", {})
        revision = identity.get("revision")
        _require(identity.get("status") == "verified" and identity.get("repo_id") == MODEL_ID and
                 isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                 "The verified immutable Qwen2.5-7B-Instruct identity is required")
        inference = getattr(loaded_ranker, "metadata", {})
        _require(inference.get("model_id") == MODEL_ID and inference.get("revision") == revision and
                 inference.get("model_identity_hash") == digest(identity) and inference.get("dtype") == "bfloat16" and
                 isinstance(inference.get("runtime"), dict), "Finish loading the pinned BF16 ranker before freezing its inference metadata")
        _require(isinstance(getattr(loaded_ranker, "device", None), str) and bool(loaded_ranker.device), "Loaded ranker device is required")
        self._generate = self._model.generate
        self._template = self._tokenizer.apply_chat_template
        _require(callable(self._generate) and callable(self._template), "Native chat template and model.generate are required")
        self._parameters = parameters
        self._device = loaded_ranker.device
        self._live = self._live_identity()
        maximum = self._live["model_config"].get("max_position_embeddings")
        _integer(maximum, "native model context capacity", 1)
        self._context_capacity = maximum
        eos = self._live["tokenizer"]["eos_token_id"]
        _integer(eos, "native EOS token ID")
        _require(bool(self._live["tokenizer"]["chat_template"]), "A native chat template must already be configured")
        self._metadata = {
            "model_id": MODEL_ID, "revision": revision, "implementation_sha256": file_digest(__file__),
            "provider_version": "qwen-local-chat-attacker-v2-keyword-format", "inference": _json_copy(inference),
            "model_identity_sha256": digest(identity), "native_model_config": self._live["model_config"],
            "native_generation_config": self._live["generation_config"], "native_tokenizer": self._live["tokenizer"],
            "generation_parameters": dict(parameters, seed_policy="Each request supplies its frozen integer seed in [0,2**63)",
                num_return_sequences=1, pad_token_id=eos, return_dict_in_generate=False, output_scores=False),
            "prompt": {"system": "exact request.system", "user": "sorted compact JSON of stage, inputs and exact response_format/output_schema field",
                       "native_chat_template": True, "add_generation_prompt": True, "add_special_tokens": False,
                       "truncation": False, "ranking_prefix_constraint": False},
            "raw_response": "Native decode with skip_special_tokens=True and clean_up_tokenization_spaces=False; no parsing or repair; token IDs and special-token-preserving decode are retained",
            "cache_namespace": CACHE_TABLE, "reproduction_status": "User-approved Qwen attacker substitution; original TextSimu attacker was GPT-4o mini",
            "keyword_format_adapter": {"output_protocol": keyword_inference.OUTPUT_PROTOCOL,
                "implementation_sha256": file_digest(keyword_inference.__file__),
                "inventory_implementation_sha256": file_digest(scalable_inference.__file__),
                "generation_auditor_sha256": file_digest(constrained_inference.__file__),
                "scope": "stage=keyword_selection and response_format=json_array_of_keyword_strings only; exactly20 distinct strings from the unchanged40 supplied candidates",
                "native_generation_parameters_unchanged": True, "repair": False, "retry": False,
                "adaptation": "Additional native legal-format mask for the Qwen substitution; not the unpublished upstream response handling"},
            "source_prefix_adapter": {"version": "qwen-source-prefix-v1", "scope": "Declared preservation requests only",
                "change": "Force exact source title and numeric whitespace chunks before a freely generated rewriting body; retain discussion No",
                "post_hoc_repair": False, "validator_unchanged": True, "original_textsimu": False},
        }
        self._metadata_hash = digest(self._metadata)
        self._live_hash = digest(self._live)
        self._parameter_hash = digest(self._parameters)
        self._connection = loaded_ranker.cache.connection
        self._connection.execute(f"CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self._connection.execute(f"CREATE TABLE IF NOT EXISTS {FAILURE_TABLE} (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self._connection.commit()
        self.calls = self.cache_hits = self.input_tokens = self.output_tokens = 0
        self._keyword_inventory = None

    def _live_identity(self):
        return {"identity": _json_copy(self.ranker.identity), "inference": _json_copy(self.ranker.metadata),
                "device": self.ranker.device, "model_config": _object_config(self._model.config, "model"),
                "generation_config": _object_config(self._model.generation_config, "generation"),
                "tokenizer": {"class": type(self._tokenizer).__name__, "chat_template": _json_copy(self._tokenizer.chat_template),
                              "eos_token_id": self._tokenizer.eos_token_id, "bos_token_id": self._tokenizer.bos_token_id,
                              "pad_token_id": self._tokenizer.pad_token_id, "all_special_ids": list(self._tokenizer.all_special_ids)}}

    def _unchanged(self):
        _require(self.ranker.model is self._model and self.ranker.tokenizer is self._tokenizer and
                 self._model.generate == self._generate and self._tokenizer.apply_chat_template == self._template and
                 self._model.training is False, "Frozen attacker model/tokenizer/generation callable changed")
        _require(digest(self._live_identity()) == self._live_hash and digest(self._parameters) == self._parameter_hash and
                 digest(self._metadata) == self._metadata_hash, "Frozen attacker inference metadata or generation settings changed")

    @property
    def metadata(self):
        self._unchanged()
        return copy.deepcopy(self._metadata)

    def _cached(self, table, key):
        row = self._connection.execute(f"SELECT payload FROM {table} WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        _require(value.get("sha256") == digest({name: child for name, child in value.items() if name != "sha256"}) and
                 value.get("key") == key and value.get("provider_sha256") == self._metadata_hash,
                 "Corrupted or incompatible attacker cache entry")
        return value

    def _persist(self, table, key, content):
        payload = dict(content, key=key, provider_sha256=self._metadata_hash)
        payload["sha256"] = digest(payload)
        with self._connection:
            prior = self._cached(table, key)
            if prior is not None and table == CACHE_TABLE:
                _require(prior["response"] == payload["response"] and prior["generated_token_ids"] == payload["generated_token_ids"],
                         "Conflicting raw attacker outputs for identical frozen inputs")
            self._connection.execute(f"INSERT OR IGNORE INTO {table} VALUES (?,?)",
                                     (key, json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def __call__(self, request):
        self.check()
        self._unchanged()
        request, format_name = _request(request)
        grammar = None
        source_constraint = None
        if request['stage'] == 'keyword_selection' and format_name == 'response_format' and request[format_name] == 'json_array_of_keyword_strings':
            values = request['inputs']
            _require(set(values) == {'count', 'ranked_keywords'} and type(values['count']) is int and values['count'] == 20 and isinstance(values['ranked_keywords'], list) and len(values['ranked_keywords']) == 40,
                     'The frozen keyword adapter requires count20 and exactly40 supplied keyword candidates')
            audit = constrained_inference.audit_generation_config({**self._live['generation_config'], **self._parameters, 'num_return_sequences': 1, 'return_dict_in_generate': False, 'output_scores': False}, self._tokenizer.eos_token_id)
            if self._keyword_inventory is None:
                self._keyword_inventory = scalable_inference.ByteLevelTokenInventory(self._tokenizer, audit['native_eos_token_ids'], alphabet=keyword_inference.ASCII_ALPHABET)
            grammar = keyword_inference.build_keyword_grammar(self._tokenizer, values['ranked_keywords'], values['count'], audit['selected_eos_token_id'], self._parameters['max_new_tokens'], inventory=self._keyword_inventory)
        if 'preserved_source_prefix' in request['inputs']:
            _require(grammar is None and request['inputs'].get('preservation_rule', {}).get('kind') == 'retain_original_numeric_and_title_tokens',
                     'Source prefix requires the declared preservation rule')
            source_constraint = SourcePrefixConstraint(self._tokenizer, request['inputs']['preserved_source_prefix'],
                isinstance(request['stage'], list) and request['stage'][0] == 'discussion', self._parameters['max_new_tokens'])
        content = {"stage": request["stage"], "inputs": request["inputs"], format_name: request[format_name]}
        messages = [{"role": "system", "content": request["system"]},
                    {"role": "user", "content": json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)}]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _require(isinstance(prompt, str) and bool(prompt), "Native template did not return a nonempty prompt")
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        ids = inputs["input_ids"]
        _require(len(ids.shape) == 2 and ids.shape[0] == 1 and ids.shape[1] > 0, "Attacker requires one nonempty native tokenized prompt")
        input_count = ids.shape[1]
        _require(input_count <= self._parameters["max_input_tokens"], "Attacker prompt exceeds explicit input budget; no truncation")
        _require(input_count + self._parameters["max_new_tokens"] <= self._context_capacity,
                 "Attacker prompt and generation exceed native model context; no truncation")
        encoded = {name: tensor.tolist() for name, tensor in inputs.items()}
        query = {"provider": self._metadata, "request": request, "messages": messages,
                 "rendered_prompt_sha256": digest(prompt), "encoded_inputs": encoded,
                 "keyword_grammar": _json_copy(grammar.metadata) if grammar is not None else None}
        query['source_prefix_constraint'] = _json_copy(source_constraint.metadata) if source_constraint is not None else None
        key = digest(query)
        failed = self._cached(FAILURE_TABLE, key)
        if failed is not None:
            raise RuntimeError("Preserved attacker generation failure; no automatic retry: " + failed["error"])
        cached = self._cached(CACHE_TABLE, key)
        if cached is not None:
            _require(isinstance(cached.get("response"), str) and digest(cached["response"]) == cached.get("response_sha256"),
                     "Corrupted raw attacker response")
            if grammar is not None:
                _require(grammar.validate_completed(cached['generated_token_ids']) == cached['response'], 'Cached keyword response differs from its verified grammar')
            if source_constraint is not None:
                source_constraint.validate_completed(cached['generated_token_ids'], cached['response'])
            self.cache_hits += 1
            return cached["response"]
        self.check()
        self._unchanged()
        import torch

        torch_device = torch.device(self._device)
        devices = []
        if torch_device.type == "cuda":
            devices = [torch.cuda.current_device() if torch_device.index is None else torch_device.index]
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
        started = time.monotonic()
        self.calls += 1
        self.input_tokens += input_count
        response = raw_with_special = token_ids = None
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.default_generator.manual_seed(request["seed"])
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(request["seed"])
                output = self._model.generate(
                    **inputs, **{name: self._parameters[name] for name in ("do_sample", "num_beams", "temperature", "top_p", "top_k", "max_new_tokens", "use_cache")},
                    num_return_sequences=1, pad_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=False, output_scores=False,
                    **({'prefix_allowed_tokens_fn': (grammar or source_constraint).prefix_allowed_tokens_fn(input_count)} if grammar is not None or source_constraint is not None else {}),
                )
            _require(len(output.shape) == 2 and output.shape[0] == 1 and output.shape[1] >= input_count,
                     "Attacker model returned an invalid token sequence shape")
            generated = output[0, input_count:]
            token_ids = generated.tolist()
            response = self._tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            raw_with_special = self._tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            _require(isinstance(response, str) and isinstance(raw_with_special, str), "Native decoder must return text")
            self.output_tokens += len(token_ids)
            if grammar is not None:
                _require(grammar.validate_completed(token_ids) == response, 'Native decoded keyword response differs from the verified canonical grammar')
            if source_constraint is not None:
                source_constraint.validate_completed(token_ids, response)
            self._persist(CACHE_TABLE, key, {"response": response, "response_sha256": digest(response),
                "raw_response_with_special_tokens": raw_with_special, "generated_token_ids": token_ids,
                "query": query, "input_tokens": input_count, "output_tokens": len(token_ids), "seconds": time.monotonic()-started})
            self._unchanged()
            return response
        except BaseException as error:
            self._persist(FAILURE_TABLE, key, {"error": str(error), "error_type": type(error).__name__, "query": query,
                                             "raw_response": response, "raw_response_with_special_tokens": raw_with_special, "generated_token_ids": token_ids,
                                             "seconds": time.monotonic()-started})
            raise

    def statistics(self):
        return {"model_calls": self.calls, "cache_hits": self.cache_hits, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "provider_sha256": self._metadata_hash}
