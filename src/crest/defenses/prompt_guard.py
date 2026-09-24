import json
import math
import sqlite3
from pathlib import Path

from ..artifacts import digest, file_digest
from ..records import validate_decision


MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MODEL_REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"


LABEL_MAPPING_SOURCE_URL = "https://github.com/meta-llama/llama-cookbook/blob/3c106f3e6ee79d6df51ae706bc7ef2d734ec3ded/getting-started/responsible_ai/prompt_guard/inference.py#L109"
LABEL_MAPPING_SOURCE_COMMIT = "3c106f3e6ee79d6df51ae706bc7ef2d734ec3ded"
LABEL_MAPPING_SOURCE_SHA256 = "29afa64415b07811c22b3c2e27330ee40f9e8edff0a7faf4940b8cfc1c74d424"


def resolve_label_mapping(raw_id2label, num_labels, *, repo_id, revision):
    if repo_id != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError("Label resolution requires the exact pinned Prompt Guard 2 86M snapshot")
    if not isinstance(num_labels, int) or isinstance(num_labels, bool) or num_labels != 2:
        raise ValueError("Expected a binary Prompt Guard 2 classifier")
    if not isinstance(raw_id2label, dict) or len(raw_id2label) != 2:
        raise ValueError("Expected two native classifier labels")
    normalized = {}
    for index, label in raw_id2label.items():
        if isinstance(index, bool) or not isinstance(index, (int, str)) or index not in (0, 1, "0", "1") or not isinstance(label, str):
            raise ValueError("Invalid native classifier label mapping")
        numeric_index = int(index)
        if numeric_index in normalized:
            raise ValueError("Duplicate native classifier label index")
        normalized[numeric_index] = label.upper()
    if set(normalized) != {0, 1}:
        raise ValueError("Expected native classifier indices zero and one")
    if set(normalized.values()) == {"BENIGN", "MALICIOUS"}:
        labels = {label: index for index, label in normalized.items()}
        provenance = {"mode": "native_config_id2label"}
    elif normalized == {0: "LABEL_0", 1: "LABEL_1"}:
        labels = {"BENIGN": 0, "MALICIOUS": 1}
        provenance = {
            "mode": "official_numeric_mapping_for_exact_pinned_generic_labels",
            "source_url": LABEL_MAPPING_SOURCE_URL,
            "source_commit": LABEL_MAPPING_SOURCE_COMMIT,
            "source_sha256": LABEL_MAPPING_SOURCE_SHA256,
            "source_model_line": 22,
            "source_score_line": 109,
        }
    else:
        raise ValueError("Unrecognized Prompt Guard 2 labels; no mapping is inferred")
    return {
        "raw_id2label": dict(raw_id2label),
        "label_mapping": labels,
        "label_mapping_provenance": provenance,
    }


def token_windows(token_ids, capacity, overlap):
    if capacity < 1 or not 0 <= overlap < capacity:
        raise ValueError("Chunk overlap must be smaller than the content-token capacity")
    if not token_ids:
        return [[]]
    result = []
    for start in range(0, len(token_ids), capacity - overlap):
        result.append(token_ids[start:start + capacity])
        if start + capacity >= len(token_ids):
            break
    return result


class PromptGuardRecord:
    def __init__(self, scorer, *, threshold, settings_provenance):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Prompt Guard threshold must be a probability")
        if not settings_provenance or not getattr(scorer, "metadata", None):
            raise ValueError("Detector settings and scorer provenance are required")
        self.scorer = scorer
        self.threshold = threshold
        self.metadata = {
            "method": "PromptGuard2-Record",
            "adapter_version": "prompt-guard-record-v1",
            "threshold": threshold,
            "comparison": "score >= threshold",
            "chunk_aggregation": "max",
            "identity_aggregation": "max across all observed appearances",
            "content_serialization": "ordered field-name colon value lines, no wrapper or identity",
            "settings_provenance": settings_provenance,
            "scorer": scorer.metadata,
        }

    def load(self):
        if callable(getattr(self.scorer, "load", None)):
            self.scorer.load()

    def detect(self, observed_decisions):
        self.load()
        scores = {}
        appearances = []
        types = {}
        cached = {}
        for decision_index, decision in enumerate(observed_decisions):
            validate_decision(decision)
            for record in decision["records"]:
                identity = record["id"]
                if identity in types and types[identity] != record["type"]:
                    raise ValueError("Record type changed for an identity")
                types[identity] = record["type"]
                text = "\n".join(f"{name}: {value}" for name, value in record["fields"].items())
                key = digest(text)
                if key not in cached:
                    cached[key] = self.scorer.score(text)
                output = cached[key]
                score = output["score"]
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Classifier returned an invalid malicious probability")
                scores[identity] = max(scores.get(identity, 0), score)
                appearances.append({"decision_index": decision_index, "record_id": identity, **output})
        return {
            "omitted_ids": sorted(identity for identity, score in scores.items() if score >= self.threshold),
            "identity_scores": scores,
            "appearance_scores": appearances,
            "provenance": self.metadata,
        }


def _require_native_chunk(value, message):
    if not value:
        raise ValueError(message)


class NativeIdChunkTokenizer:
    def __init__(self, tokenizer, tokenizer_file):
        import transformers
        import tokenizers
        from tokenizers import Tokenizer
        _require_native_chunk(transformers.__version__ == '5.3.0' and tokenizers.__version__ == '0.22.2',
                'Prompt Guard ID-chunk bridge requires its pinned native 5.3 runtime')
        _require_native_chunk(type(tokenizer).__name__ == 'DebertaV2Tokenizer' and not hasattr(tokenizer, 'prepare_for_model'),
                'ID-chunk bridge is only for the native DeBERTa tokenizer with the removed API')
        self.native = tokenizer


        serialized = Path(tokenizer_file).read_text()
        native_state = json.loads(serialized)
        self.processor = Tokenizer.from_str(serialized)
        self.processor.no_padding()
        self.processor.no_truncation()
        _require_native_chunk(self.processor.num_special_tokens_to_add(False) == tokenizer.num_special_tokens_to_add(False) == 2,
                'Pinned Prompt Guard special-token count changed')
        _require_native_chunk(set(tokenizer.model_input_names) == {'input_ids', 'attention_mask', 'token_type_ids'},
                'Pinned Prompt Guard model input fields changed')
        self.metadata = {'kind': 'prompt-guard-checkpoint-id-chunk-bridge-v2',
            'implementation_sha256': file_digest(__file__),
            'tokenizer_asset_sha256': file_digest(tokenizer_file),
            'checkpoint_normalizer_sha256': digest(native_state['normalizer']),
            'native_postprocessor_sha256': digest(native_state['post_processor']),
            'text_encoding': 'Verified tokenizer.json pipeline with checkpoint normalization and added tokens',
            'content_ids': 'Preserved exactly; no decoding, retokenization, padding or truncation',
            'special_tokens': 'Unchanged native checkpoint postprocessor',
            'tensors': 'Single batch, int64 input_ids/attention_mask/token_type_ids',
            'runtime': {'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__}}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def encode(self, text, *, add_special_tokens):
        _require_native_chunk(isinstance(text, str) and add_special_tokens is False,
                              'Only raw record text without special tokens is supported')
        return self.processor.encode(text, add_special_tokens=False).ids

    def num_special_tokens_to_add(self, pair=False):
        _require_native_chunk(pair is False, 'Only single-record tokenization is supported')
        return self.processor.num_special_tokens_to_add(False)

    def prepare_for_model(self, ids, *, add_special_tokens, truncation, return_attention_mask,
                          return_tensors, prepend_batch_axis):
        _require_native_chunk(add_special_tokens is True and truncation is False and return_attention_mask is True
                and return_tensors == 'pt' and prepend_batch_axis is True,
                'Only the frozen single-chunk Prompt Guard preparation contract is supported')
        _require_native_chunk(isinstance(ids, list) and len(ids) <= 510
                and all(type(value) is int and value >= 0 for value in ids), 'Invalid content-ID chunk')
        import torch
        from tokenizers import Encoding
        tokens = [self.processor.id_to_token(value) for value in ids]
        _require_native_chunk(len(tokens) == len(ids) and all(isinstance(value, str) for value in tokens),
                'Content chunk contains an unknown token ID')
        encoding = Encoding()
        state = json.loads(encoding.__getstate__())
        _require_native_chunk(set(state) == {'ids', 'type_ids', 'tokens', 'words', 'offsets', 'special_tokens_mask',
                              'attention_mask', 'overflowing', 'sequence_ranges'}, 'Native Encoding schema changed')
        size = len(ids)
        state.update(ids=list(ids), type_ids=[0]*size, tokens=tokens, words=[None]*size,
                     offsets=[[0, 0] for _ in ids], special_tokens_mask=[0]*size,
                     attention_mask=[1]*size, overflowing=[], sequence_ranges={})
        encoding.__setstate__(json.dumps(state, ensure_ascii=False).encode())
        processed = self.processor.post_process(encoding, add_special_tokens=True)
        _require_native_chunk(processed.ids == [self.native.cls_token_id, *ids, self.native.sep_token_id]
                and processed.attention_mask == [1]*(size+2) and processed.type_ids == [0]*(size+2)
                and not processed.overflowing, 'Native postprocessor changed content or chunk semantics')
        return {key: torch.tensor([values], dtype=torch.long) for key, values in
                [('input_ids', processed.ids), ('attention_mask', processed.attention_mask),
                 ('token_type_ids', processed.type_ids)]}


class LocalPromptGuard:
    def __init__(self, model_path, cache_path, *, revision=MODEL_REVISION, device="cpu", overlap=64):
        self.path = Path(model_path).resolve()
        self.identity = json.loads((self.path / "model_identity.json").read_text())
        if revision != MODEL_REVISION or self.identity.get("revision") != revision or self.identity.get("repo_id") != MODEL_ID or self.identity.get("status") != "verified":
            raise ValueError("The verified pinned Prompt Guard 2 86M snapshot is required")
        if not 0 <= overlap < 510:
            raise ValueError("Invalid Prompt Guard window overlap")
        self.device = device
        self.overlap = overlap
        self.model = None
        self.tokenizer = None
        self.calls = 0
        self.cache_hits = 0
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(cache_path, timeout=60)
        self.cache.execute("CREATE TABLE IF NOT EXISTS prompt_guard_scores (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.cache.commit()
        self.metadata = {
            "model_id": MODEL_ID,
            "revision": revision,
            "model_identity_hash": digest(self.identity),
            "implementation_hash": file_digest(__file__),
            "dtype": "float32",
            "device": device,
            "max_input_tokens": 512,
            "overlap_content_tokens": overlap,
            "score": "softmax probability at native MALICIOUS label",
            "tokenizer": "unaltered checkpoint tokenizer; special tokens included in 512-token budget",
        }

    def load(self):
        if self.model is not None:
            return
        expected = self.identity.get("files_sha256", {})
        if "config.json" not in expected or "tokenizer_config.json" not in expected or not any(name.endswith(".safetensors") for name in expected):
            raise ValueError("Incomplete Prompt Guard snapshot manifest")
        for name, checksum in expected.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or file_digest(self.path / name) != checksum:
                raise ValueError(f"Prompt Guard snapshot changed: {name}")
        import torch
        import transformers
        import tokenizers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for Prompt Guard but unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        if not hasattr(self.tokenizer, "prepare_for_model"):
            if "tokenizer.json" not in expected:
                raise ValueError("Native Prompt Guard bridge requires the verified tokenizer.json asset")
            self.tokenizer = NativeIdChunkTokenizer(self.tokenizer, self.path / "tokenizer.json")
            self.metadata["tokenizer_api_compatibility"] = self.tokenizer.metadata
        model = AutoModelForSequenceClassification.from_pretrained(self.path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32).to(self.device).eval()
        resolution = resolve_label_mapping(model.config.id2label, model.config.num_labels, repo_id=self.identity["repo_id"], revision=self.identity["revision"])
        labels = resolution["label_mapping"]
        self.metadata.update(resolution)
        self.malicious_index = labels["MALICIOUS"]
        self.capacity = 512 - self.tokenizer.num_special_tokens_to_add(pair=False)
        token_windows([], self.capacity, self.overlap)
        self.metadata.update({"torch_version": torch.__version__, "transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__, "label_mapping": labels, "content_token_capacity": self.capacity, "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "tf32": False, "cudnn_benchmark": False})
        if self.device.startswith("cuda"):
            self.metadata["gpu_name"] = torch.cuda.get_device_properties(self.device).name
        self.model = model

    def score(self, text):
        if not isinstance(text, str):
            raise TypeError("Prompt Guard expects record text")
        self.load()
        key = digest({"model": self.metadata, "text": text})
        row = self.cache.execute("SELECT payload FROM prompt_guard_scores WHERE key=?", (key,)).fetchone()
        if row:
            self.cache_hits += 1
            return json.loads(row[0])
        import torch

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = token_windows(ids, self.capacity, self.overlap)
        probabilities = []
        with torch.inference_mode():
            for chunk in chunks:
                inputs = self.tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False, return_attention_mask=True, return_tensors="pt", prepend_batch_axis=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                if inputs["input_ids"].shape[-1] > 512:
                    raise ValueError("Prompt Guard chunk exceeded its context budget")
                logits = self.model(**inputs).logits
                probabilities.append(float(torch.softmax(logits.float(), dim=-1)[0, self.malicious_index].item()))
                self.calls += 1
        result = {"score": max(probabilities), "chunk_scores": probabilities, "chunk_count": len(chunks), "content_token_count": len(ids)}
        with self.cache:
            self.cache.execute("INSERT OR IGNORE INTO prompt_guard_scores VALUES (?, ?)", (key, json.dumps(result, allow_nan=False)))
        return result
