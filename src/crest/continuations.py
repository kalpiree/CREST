from .artifacts import digest, file_digest


class LocalContinuations:
    def __init__(self, ranker, *, seed, temperature, top_p, max_new_tokens, max_input_tokens):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("A nonnegative continuation seed is required")
        if not 0 < temperature <= 2 or not 0 < top_p <= 1 or min(max_new_tokens, max_input_tokens) < 1:
            raise ValueError("Invalid explicit continuation generation settings")
        self.ranker = ranker
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.calls = 0
        self.cache_hits = 0
        self.ranker.cache.connection.execute("CREATE TABLE IF NOT EXISTS continuations (key TEXT PRIMARY KEY, text TEXT NOT NULL)")
        self.ranker.cache.connection.commit()

    @property
    def metadata(self):
        return {
            "model_id": self.ranker.identity["repo_id"],
            "revision": self.ranker.identity["revision"],
            "inference": self.ranker.metadata,
            "implementation_hash": file_digest(__file__),
            "generation_parameters": {"seed": self.seed, "do_sample": True, "temperature": self.temperature, "top_p": self.top_p, "top_k": 0, "max_new_tokens": self.max_new_tokens, "max_input_tokens": self.max_input_tokens},
            "prompt": "native raw-text continuation from supplied prefix, no instruction wrapper",
            "reproduction_status": "Explicit local provider choice; original RewriteDetection continuation checkpoint is unspecified",
        }

    def __call__(self, prefix, count):
        if not isinstance(prefix, str) or not prefix or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("A nonempty text prefix and positive continuation count are required")
        self.ranker.load()
        import torch

        inputs = self.ranker.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(self.ranker.device)
        input_count = inputs["input_ids"].shape[1]
        if input_count > self.max_input_tokens:
            raise ValueError("Continuation prefix exceeds the declared input-token budget")
        if input_count + self.max_new_tokens > self.ranker.model.config.max_position_embeddings:
            raise ValueError("Continuation exceeds model context capacity")
        devices = [torch.device(self.ranker.device).index or 0] if self.ranker.device.startswith("cuda") else []
        result = []
        for index in range(count):
            generation_seed = int(digest({"prefix": prefix, "seed": self.seed, "sample_index": index})[:15], 16)
            key = digest({"provider": self.metadata, "prefix": prefix, "generation_seed": generation_seed})
            row = self.ranker.cache.connection.execute("SELECT text FROM continuations WHERE key=?", (key,)).fetchone()
            if row:
                self.cache_hits += 1
                result.append(row[0])
                continue
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(generation_seed)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(generation_seed)
                output = self.ranker.model.generate(**inputs, do_sample=True, num_beams=1, temperature=self.temperature, top_p=self.top_p, top_k=0, max_new_tokens=self.max_new_tokens, pad_token_id=self.ranker.tokenizer.eos_token_id, use_cache=True)
            text = self.ranker.tokenizer.decode(output[0, input_count:], skip_special_tokens=True)
            with self.ranker.cache.connection:
                self.ranker.cache.connection.execute("INSERT INTO continuations VALUES (?,?)", (key, text))
            self.calls += 1
            result.append(text)
        return result
