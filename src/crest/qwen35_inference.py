import copy
import inspect
from collections import Counter

from .artifacts import digest, file_digest
from .rq1_inference import ScalableSDPARanker
from .scalable_inference import ScalableLocalRanker

MODEL_ID = 'Qwen/Qwen3.5-9B'
RUNTIME_PROTOCOL = 'qwen35-native-text-sdpa-nonthinking-json-v1'
SUPPORTED_RUNTIME = {'torch': '2.6.0+cu124', 'transformers': '5.3.0', 'tokenizers': '0.22.2'}
OFFICIAL_SOURCE = 'https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_runtime(versions):
    require(versions == SUPPORTED_RUNTIME, 'Qwen3.5 requires its separately pinned torch2.6.0+cu124/transformers5.3.0/tokenizers0.22.2 runtime')


class NonThinkingTokenizer:

    def __init__(self, tokenizer):
        self.native = tokenizer

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, *args, **kwargs):
        return self.native(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        require(kwargs.get('enable_thinking', False) is False, 'Thinking cannot override the frozen non-thinking mode')
        require(isinstance(messages, list) and all(isinstance(m, dict) and isinstance(m.get('content'), str) for m in messages),
                'The Qwen3.5 comparison accepts text messages only')
        prompt = self.native.apply_chat_template(messages, **{**kwargs, 'enable_thinking': False})
        if kwargs.get('add_generation_prompt') and kwargs.get('tokenize', True) is False:
            require(prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'),
                    'Native Qwen3.5 template did not render the frozen non-thinking assistant prefix')
        return prompt


def verify_loading_info(info):
    require(not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),
            'Text-only Qwen3.5 checkpoint has missing, mismatched or failed language weights')
    unexpected = info.get('unexpected_keys', [])
    require(all(k.startswith(('model.visual.', 'mtp.')) for k in unexpected),
            'Text-only loading omitted an unexpected language weight')


def loading_info_digest(info):

    verify_loading_info(info)
    def normalize(value):
        if isinstance(value, dict):
            return {name: normalize(child) for name, child in value.items()}
        if isinstance(value, (set, frozenset)):
            require(all(isinstance(key, str) for key in value), 'Unexpected native loading-info set contents')
            return sorted(value)
        if isinstance(value, (list, tuple)):
            return [normalize(child) for child in value]
        return value
    return digest(normalize(info))


def resolve_native_generation_defaults(model):

    original = copy.deepcopy(model.generation_config.to_dict())
    resolved, model_kwargs = model._prepare_generation_config(None)
    require(not model_kwargs, 'Unexpected native generation model kwargs')
    effective = resolved.to_dict()
    require(all(effective.get(key) == value for key, value in original.items() if value is not None),
            'Native default resolution changed an explicit checkpoint generation setting')
    model.generation_config = resolved
    return {'native_checkpoint_defaults': original, 'resolved_native_defaults': copy.deepcopy(effective),
            'resolver': 'transformers.GenerationMixin._prepare_generation_config(None)'}


class Qwen35TextRanker(ScalableLocalRanker):
    def __init__(self, *args, enable_thinking=False, linear_attention_backend='torch_reference', **kwargs):
        require(enable_thinking is False and linear_attention_backend == 'torch_reference',
                'Qwen3.5 requires explicit non-thinking mode and the frozen torch reference linear-attention backend')
        super().__init__(*args, **kwargs)
        require(self.identity['repo_id'] == MODEL_ID and self.dtype == 'bfloat16', 'Only the official unquantized Qwen3.5-9B BF16 snapshot is supported')
        template = self.path/'chat_template.jinja'
        require(template.is_file() and self.identity.get('files_sha256', {}).get(template.name) == file_digest(template),
                'Qwen3.5 needs its verified native chat_template.jinja')
        self.metadata.update(attention_implementation='qwen35_text_sdpa_nonthinking',
            cache_namespace=RUNTIME_PROTOCOL, qwen35_implementation_sha256=file_digest(__file__),
            native_model_class='transformers.Qwen3_5ForCausalLM', text_only=True,
            chat_template_kwargs={'enable_thinking':False}, chat_template_sha256=file_digest(template),
            linear_attention_backend=linear_attention_backend, required_runtime=copy.deepcopy(SUPPORTED_RUNTIME),
            native_logits_to_keep=1, runtime_source=OFFICIAL_SOURCE)

    def load(self):
        if self.model is None:
            import torch
            import transformers
            import tokenizers
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM
            from transformers.models.qwen3_5 import modeling_qwen3_5 as native
            versions={'torch':str(torch.__version__), 'transformers':transformers.__version__, 'tokenizers':tokenizers.__version__}
            validate_runtime(versions)
            require(self.device.startswith('cuda') and torch.cuda.is_available(), 'Qwen3.5 production runtime requires CUDA')
            optional=('causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule','FusedRMSNormGated')
            require(all(getattr(native,name,None) is None for name in optional), 'Optional linear-attention kernels differ from the frozen torch reference backend')
            self.verify_weights()
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
            model,info=Qwen3_5ForCausalLM.from_pretrained(self.path,dtype=torch.bfloat16,
                device_map={'':self.device},local_files_only=True,trust_remote_code=False,
                attn_implementation='sdpa',output_loading_info=True)
            verify_loading_info(info)
            model.eval()
            require(type(model) is Qwen3_5ForCausalLM and model.config.model_type=='qwen3_5_text' and model.config._attn_implementation=='sdpa',
                    'Wrong native text architecture or attention implementation')
            require(model._supports_logits_to_keep() and 'logits_to_keep' in inspect.signature(model.forward).parameters,
                    'Native generation must support memory-bounded last-token logits')
            require(not getattr(model,'is_quantized',False) and not getattr(model,'is_loaded_in_4bit',False) and not getattr(model,'is_loaded_in_8bit',False),
                    'Quantization is outside the frozen BF16 comparison')
            properties=torch.cuda.get_device_properties(self.device)
            parameter_dtypes=Counter(str(p.dtype) for p in model.parameters())
            require(all(p.device.type=='cuda' and p.device.index==(torch.device(self.device).index or 0) for p in model.parameters()),
                    'Qwen3.5 must use one declared GPU without offloading')
            self.model=model;self.tokenizer=NonThinkingTokenizer(tokenizer)
            self.metadata['native_generation_resolution'] = resolve_native_generation_defaults(model)
            self.metadata.update(torch_version=versions['torch'],transformers_version=versions['transformers'],tokenizers_version=versions['tokenizers'])
            self.metadata['runtime']={'device':self.device,'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
                'tf32':False,'gpu':{'name':properties.name,'compute_capability':[properties.major,properties.minor]},
                'sdp_backends':{'flash_enabled':torch.backends.cuda.flash_sdp_enabled(),'memory_efficient_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),'math_enabled':torch.backends.cuda.math_sdp_enabled()},
                'linear_attention_backend':'torch_reference','parameter_dtype_counts':dict(parameter_dtypes),
                'native_model_source_sha256':file_digest(native.__file__),
                'loading_info_sha256':loading_info_digest(info)}

        super().load()


class NativeQwen2ContinuationRanker(ScalableSDPARanker):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        require(self.identity['repo_id']=='Qwen/Qwen2.5-7B-Instruct' and self.dtype=='bfloat16', 'Separate generator must be Qwen2.5 BF16')
        self.metadata.update(cache_namespace='qwen25-native-tf53-continuation-v1',
            continuation_runtime_implementation_sha256=file_digest(__file__), required_runtime=copy.deepcopy(SUPPORTED_RUNTIME),
            native_logits_to_keep=1, continuation_runtime_change='Same frozen Qwen2.5 weights and sampling parameters; native Transformers5.3 generation, separately recorded cache identity')

    def load(self):
        if self.model is None:
            import torch,transformers,tokenizers
            from transformers import AutoModelForCausalLM,AutoTokenizer
            validate_runtime({'torch':str(torch.__version__),'transformers':transformers.__version__,'tokenizers':tokenizers.__version__})
            require(self.device.startswith('cuda') and torch.cuda.is_available(), 'Separate Qwen2.5 requires CUDA')
            self.verify_weights()
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
            self.model=AutoModelForCausalLM.from_pretrained(self.path,dtype=torch.bfloat16,
                device_map={'':self.device},local_files_only=True,trust_remote_code=False,
                attn_implementation='sdpa').eval()
            require(self.model.config._attn_implementation=='sdpa', 'Separate Qwen2.5 must use native SDPA')
            require(self.model.config.model_type=='qwen2' and self.model._supports_logits_to_keep(),
                    'Separate Qwen2.5 native generation lacks last-token logits support')
            self.metadata['native_generation_resolution']=resolve_native_generation_defaults(self.model)
            self.metadata.update(torch_version=str(torch.__version__),transformers_version=transformers.__version__,
                tokenizers_version=tokenizers.__version__)
            properties=torch.cuda.get_device_properties(self.device)
            require(all(p.device.type=='cuda' and p.device.index==(torch.device(self.device).index or 0)
                        for p in self.model.parameters()), 'Separate Qwen2.5 must use one declared GPU without offloading')
            self.metadata['runtime']={'device':self.device,'cuda_version':torch.version.cuda,
                'cudnn_version':torch.backends.cudnn.version(),'tf32':False,
                'gpu':{'name':properties.name,'compute_capability':[properties.major,properties.minor]},
                'sdp_backends':{'flash_enabled':torch.backends.cuda.flash_sdp_enabled(),
                    'memory_efficient_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),
                    'math_enabled':torch.backends.cuda.math_sdp_enabled()}}
        ScalableLocalRanker.load(self)
