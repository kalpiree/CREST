import json
import sqlite3
import time
from pathlib import Path

from .artifacts import digest, file_digest
from .records import messages_for, parse_ranking


class InvalidRanking(RuntimeError):
    pass


class QueryCache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute("CREATE TABLE IF NOT EXISTS rankings (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS failures (key TEXT, error TEXT, response TEXT, created REAL, request TEXT)")
        if "request" not in {row[1] for row in self.connection.execute("PRAGMA table_info(failures)")}:
            self.connection.execute("ALTER TABLE failures ADD COLUMN request TEXT")
        self.connection.commit()

    def get(self, key):
        row = self.connection.execute("SELECT payload FROM rankings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.connection:
            existing = self.get(key)
            if existing is not None and existing["ranking"] != payload["ranking"]:
                raise RuntimeError("Conflicting rankings for identical inference inputs")
            self.connection.execute("INSERT OR IGNORE INTO rankings VALUES (?, ?)", (key, encoded))

    def failure(self, key, error, response, request=None):
        with self.connection:
            self.connection.execute("INSERT INTO failures (key,error,response,created,request) VALUES (?, ?, ?, ?, ?)", (key, str(error), response, time.time(), json.dumps(request)))


class LocalRanker:
    def __init__(self, model_path, revision, cache_path, device="cuda:0", dtype="bfloat16", max_input_tokens=4096, max_new_tokens=128):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("An immutable 40-character model revision is required")
        self.path = Path(model_path).resolve()
        identity_path = self.path / "model_identity.json"
        if not identity_path.is_file():
            raise ValueError("Missing verified model_identity.json; verify the snapshot before inference")
        identity = json.loads(identity_path.read_text())
        if identity.get("status") != "verified" or identity.get("revision") != revision:
            raise ValueError("Model revision does not match the verified snapshot identity")
        self.identity = identity
        asset_hashes = {}
        for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model"):
            if (self.path / name).is_file():
                asset_hashes[name] = file_digest(self.path / name)
                expected = identity.get("files_sha256", {}).get(name)
                if expected is not None and expected != asset_hashes[name]:
                    raise ValueError(f"Model asset changed after verification: {name}")
        self.device = device
        self.dtype = dtype
        self.cache = QueryCache(cache_path)
        self.model = None
        self.tokenizer = None
        self.metadata = {
            "revision": revision,
            "model_id": identity["repo_id"],
            "model_identity_hash": digest(identity),
            "inference_implementation_hash": file_digest(__file__),
            "asset_hashes": asset_hashes,
            "model_config_hash": digest(json.loads((self.path / "config.json").read_text())),
            "dtype": dtype,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "seed": 0,
            "prompt_version": "crest-rrm-labels-v2",
            "attention_implementation": "eager",
        }
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def verify_weights(self):
        expected = self.identity.get("files_sha256", {})
        weights = [name for name in expected if name.endswith((".safetensors", ".safetensors.index.json"))]
        if not any(name.endswith(".safetensors") for name in weights):
            raise ValueError("Snapshot identity has no verified safetensor weights")
        for name in sorted(weights):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid weight path in snapshot identity")
            if not (self.path / name).is_file() or file_digest(self.path / name) != expected[name]:
                raise ValueError(f"Weight file changed after verification: {name}")

    def load(self):
        if self.model is not None:
            return
        self.verify_weights()
        import torch
        import transformers
        import tokenizers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=getattr(torch, self.dtype),
            device_map={"": self.device},
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
        ).eval()
        self.metadata["torch_version"] = torch.__version__
        self.metadata["transformers_version"] = transformers.__version__
        self.metadata["tokenizers_version"] = tokenizers.__version__
        self.metadata["runtime"] = {"device": self.device, "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "tf32": False}
        if self.device.startswith("cuda"):
            properties = torch.cuda.get_device_properties(self.device)
            self.metadata["runtime"]["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor]}

    def rank(self, decision, k, omitted=frozenset(), datamark=False):
        messages = messages_for(decision, k, omitted, datamark)
        labels = {f"C{index}": item for index, item in enumerate(decision["candidates"])}
        self.load()
        key = digest({"inference": self.metadata, "messages": messages, "k": k, "candidates": decision["candidates"]})
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
            )
        generated = output[0, input_count:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.calls += 1
        self.input_tokens += input_count
        self.output_tokens += len(generated)
        try:
            output_labels = parse_ranking(response, list(labels), k)
            ranking = [labels[label] for label in output_labels]
        except (ValueError, TypeError) as error:
            self.cache.failure(key, error, response, {"messages": messages, "k": k, "labels": labels, "inference": self.metadata})
            raise InvalidRanking(f"Invalid model ranking recorded in cache: {error}") from error
        self.cache.put(key, {"ranking": ranking, "response": response, "input_tokens": input_count, "output_tokens": len(generated), "seconds": time.monotonic() - started})
        return ranking

    def statistics(self):
        return {"model_calls": self.calls, "cache_hits": self.cache_hits, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "inference": self.metadata}
