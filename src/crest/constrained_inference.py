import itertools
import json
import time

from .artifacts import digest, file_digest
from .inference import InvalidRanking, LocalRanker
from .records import messages_for, parse_ranking

OUTPUT_PROTOCOL = "crest-json-trie-v3"
MAX_VALID_SEQUENCES = 10000
MAX_DIAGNOSTIC_CANDIDATES = 8
MAX_DIAGNOSTIC_K = 2


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class TokenTrieGrammar:
    def __init__(self, token_paths, eos_token_id, labels, k):
        self.token_paths = tuple(tuple(path) for path in token_paths)
        self.eos_token_id = eos_token_id
        self.root = {}
        self.terminals = set(self.token_paths)
        self.labels = tuple(labels)
        self.k = k
        for path in self.token_paths:
            node = self.root
            for token in path:
                node = node.setdefault(token, {})
        self.metadata = {
            "output_protocol": OUTPUT_PROTOCOL,
            "candidate_count": len(labels),
            "k": k,
            "valid_sequence_count": len(self.token_paths),
            "token_paths_sha256": digest(self.token_paths),
            "eos_token_id": eos_token_id,
            "maximum_json_tokens": max(map(len, self.token_paths)),
            "maximum_new_tokens_including_eos": max(map(len, self.token_paths)) + 1,
            "serialization": "json.dumps(list(labels), ensure_ascii=True, separators=(', ', ': '))",
        }

    def allowed_tokens(self, suffix_ids):
        prefix = tuple(suffix_ids)
        node = self.root
        for token in prefix:
            if token not in node:
                raise ValueError("Generated token prefix is outside the frozen ranking grammar")
            node = node[token]
        allowed = list(node)
        if prefix in self.terminals:
            allowed.append(self.eos_token_id)
        if not allowed:
            raise ValueError("Ranking grammar has no legal next token")
        return sorted(set(allowed))

    def prefix_allowed_tokens_fn(self, prompt_token_count):
        _positive_integer(prompt_token_count, "prompt_token_count")
        def allowed(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError("The diagnostic decoder supports one input sequence at a time")
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            if len(ids) < prompt_token_count:
                raise ValueError("Generation prefix is shorter than the native prompt")
            return self.allowed_tokens(ids[prompt_token_count:])
        return allowed


def build_ranking_grammar(tokenizer, candidate_count, k, eos_token_id, max_new_tokens, max_sequences=MAX_VALID_SEQUENCES):
    for value, name in ((candidate_count, "candidate_count"), (k, "k"), (max_new_tokens, "max_new_tokens"), (max_sequences, "max_sequences")):
        _positive_integer(value, name)
    if k > candidate_count:
        raise ValueError("K exceeds the candidate count")
    if max_sequences > MAX_VALID_SEQUENCES:
        raise ValueError("Grammar enumeration cannot exceed the diagnostic safety cap")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("A native integer EOS token is required")
    count = 1
    for factor in range(candidate_count, candidate_count - k, -1):
        count *= factor
        if count > max_sequences:
            raise ValueError("Ranking grammar exceeds the maximum enumerated sequence count")
    labels = [f"C{index}" for index in range(candidate_count)]
    paths = []
    special_ids = set(getattr(tokenizer, "all_special_ids", [])) | {eos_token_id}
    for ordering in itertools.permutations(labels, k):
        text = json.dumps(list(ordering), ensure_ascii=True, separators=(", ", ": "))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(ids, list) or not ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError("Tokenizer must produce a nonempty list of integer token IDs")
        if special_ids.intersection(ids):
            raise ValueError("Ranking content tokenization contains EOS or another special token")
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text or parse_ranking(decoded, labels, k) != list(ordering):
            raise ValueError("Tokenizer does not round-trip the exact canonical ranking output")
        paths.append(tuple(ids))
    if len(set(paths)) != count:
        raise ValueError("Distinct rankings have colliding tokenizations")
    grammar = TokenTrieGrammar(paths, eos_token_id, labels, k)
    if grammar.metadata["maximum_new_tokens_including_eos"] > max_new_tokens:
        raise ValueError("max_new_tokens is insufficient for every legal ranking plus EOS")
    return grammar


def audit_generation_config(config, tokenizer_eos_token_id):
    for name in ("forced_bos_token_id", "forced_eos_token_id", "forced_decoder_ids", "force_words_ids", "constraints", "bad_words_ids", "suppress_tokens", "begin_suppress_tokens", "sequence_bias", "stop_strings", "dola_layers"):
        if config.get(name) not in (None, [], {}):
            raise ValueError(f"Native generation setting conflicts with the diagnostic grammar: {name}")
    for name in ("min_length", "min_new_tokens", "no_repeat_ngram_size", "encoder_no_repeat_ngram_size"):
        if config.get(name) not in (None, 0):
            raise ValueError(f"Native generation setting may suppress every legal token: {name}")
    if config.get("max_time") is not None or config.get("penalty_alpha") not in (None, 0) or config.get("return_dict_in_generate", False):
        raise ValueError("Native generation defaults do not support this complete greedy tensor-output protocol")
    if config.get("remove_invalid_values", False):
        raise ValueError("remove_invalid_values may undo the grammar mask")
    if config.get("num_return_sequences", 1) != 1 or config.get("num_beam_groups", 1) != 1:
        raise ValueError("The diagnostic decoder supports one greedy sequence")
    eos = config.get("eos_token_id")
    native = [eos] if isinstance(eos, int) and not isinstance(eos, bool) else eos
    if not isinstance(native, (list, tuple)) or not native or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in native):
        raise ValueError("Native generation must recognize an integer EOS token")
    selected = tokenizer_eos_token_id if tokenizer_eos_token_id in native else native[0]
    return {"native_eos_token_ids": list(native), "selected_eos_token_id": selected, "generation_defaults_sha256": digest(config), "native_repetition_penalty": config.get("repetition_penalty", 1.0)}


class ConstrainedLocalRanker(LocalRanker):
    def __init__(self, model_path, revision, cache_path, device="cuda:0", dtype="bfloat16", max_input_tokens=4096, max_new_tokens=128):
        super().__init__(model_path, revision, cache_path, device, dtype, max_input_tokens, max_new_tokens)
        self.metadata.update({
            "output_protocol": OUTPUT_PROTOCOL,
            "cache_namespace": "crest-rankings-v3-json-trie",
            "constrained_inference_implementation_hash": file_digest(__file__),
            "output_constraint": {
                "kind": "complete canonical JSON token trie over all ordered distinct candidate-label permutations",
                "maximum_valid_sequences": MAX_VALID_SEQUENCES,
                "maximum_candidates": MAX_DIAGNOSTIC_CANDIDATES,
                "maximum_k": MAX_DIAGNOSTIC_K,
                "ranking_choice": "native greedy generation after masking tokens outside the trie; every legal child remains available",
                "candidate_order": "unchanged input candidate order mapped to C0, C1, ...",
                "scope": "Small development diagnostic",
                "repair": False,
            },
        })
        self._grammars = {}
        self._generation_audited = False

    def load(self):
        super().load()
        if not self._generation_audited:
            defaults = self.model.generation_config.to_dict()
            audit = audit_generation_config(defaults, self.tokenizer.eos_token_id)
            self.metadata["constrained_generation"] = {
                **audit,
                "native_generation_defaults": defaults,
                "explicit_generation_overrides": {"do_sample": False, "num_beams": 1, "max_new_tokens": self.metadata["max_new_tokens"], "pad_token_id": self.tokenizer.eos_token_id, "use_cache": True},
                "additional_override": "prefix_allowed_tokens_fn bound to the exact candidate-label trie",
            }
            self._generation_audited = True

    def rank(self, decision, k, omitted=frozenset(), datamark=False):
        messages = messages_for(decision, k, omitted, datamark)
        _positive_integer(k, "k")
        if len(decision["candidates"]) > MAX_DIAGNOSTIC_CANDIDATES or k > MAX_DIAGNOSTIC_K:
            raise ValueError("Constrained v3 is restricted to at most eight candidates and Top-2 diagnostics")
        labels = {f"C{index}": item for index, item in enumerate(decision["candidates"])}
        self.load()
        grammar_key = (len(labels), k)
        if grammar_key not in self._grammars:
            eos = self.metadata["constrained_generation"]["selected_eos_token_id"]
            self._grammars[grammar_key] = build_ranking_grammar(self.tokenizer, len(labels), k, eos, self.metadata["max_new_tokens"])
        grammar = self._grammars[grammar_key]
        key = digest({"inference": self.metadata, "grammar": grammar.metadata, "messages": messages, "k": k, "candidates": decision["candidates"]})
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return parse_ranking(json.dumps(cached["ranking"]), decision["candidates"], k)
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_count = inputs["input_ids"].shape[1]
        if input_count > self.metadata["max_input_tokens"]:
            raise ValueError("Prompt exceeds the configured context budget; shorten whole records during preparation")
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.metadata["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                prefix_allowed_tokens_fn=grammar.prefix_allowed_tokens_fn(input_count),
            )
        generated = output[0, input_count:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.calls += 1
        self.input_tokens += input_count
        self.output_tokens += len(generated)
        try:
            generated_ids = generated.tolist()
            if not generated_ids or generated_ids[-1] != grammar.eos_token_id or tuple(generated_ids[:-1]) not in grammar.terminals:
                raise ValueError("Generated output is not a complete canonical ranking followed by native EOS")
            output_labels = parse_ranking(response, list(labels), k)
            ranking = [labels[label] for label in output_labels]
        except (ValueError, TypeError) as error:
            self.cache.failure(key, error, response, {"messages": messages, "k": k, "labels": labels, "inference": self.metadata, "grammar": grammar.metadata})
            raise InvalidRanking(f"Invalid constrained model ranking recorded in cache: {error}") from error
        self.cache.put(key, {"ranking": ranking, "response": response, "input_tokens": input_count, "output_tokens": len(generated), "seconds": time.monotonic() - started, "grammar": grammar.metadata, "output_protocol": OUTPUT_PROTOCOL})
        return ranking
