from .artifacts import file_digest
from .scalable_inference import ScalableLocalRanker


class ScalableSDPARanker(ScalableLocalRanker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata.update({"attention_implementation": "sdpa",
            "cache_namespace": "crest-rankings-v4-json-byte-dfa-native-sdpa",
            "sdpa_runtime_implementation_hash": file_digest(__file__),
            "runtime_change": "Native Transformers/PyTorch SDPA; same weights and greedy grammar, separate caches/calibration; numerical equality with eager is not assumed"})

    def load(self):
        if self.model is None:
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
            self.model = AutoModelForCausalLM.from_pretrained(self.path,
                torch_dtype=getattr(torch, self.dtype), device_map={"": self.device},
                local_files_only=True, trust_remote_code=False, attn_implementation="sdpa").eval()
            if self.model.config._attn_implementation != "sdpa":
                raise RuntimeError("Requested native SDPA was not selected")
            self.metadata.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                                 tokenizers_version=tokenizers.__version__)
            self.metadata["runtime"] = {"device": self.device, "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(), "tf32": False,
                "sdp_backends": {"flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                                 "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 "math_enabled": torch.backends.cuda.math_sdp_enabled()}}
            if self.device.startswith("cuda"):
                properties = torch.cuda.get_device_properties(self.device)
                self.metadata["runtime"]["gpu"] = {"name": properties.name,
                                                    "compute_capability": [properties.major, properties.minor]}
        super().load()
