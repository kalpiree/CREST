import ast
import copy
import functools
import hashlib
import inspect
import textwrap
import threading

from .artifacts import digest, file_digest
from .rq1_inference import ScalableSDPARanker

RUNTIME_PROTOCOL = 'qwen2-native-sdpa-generation-last-token-logits-v1'
SUPPORTED_RUNTIME = {'torch': '2.4.1+cu121', 'transformers': '4.44.2', 'tokenizers': '0.19.1'}
UPSTREAM_SOURCE = 'https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py'


def validate_runtime(versions, device_type, dtype):
    if versions != SUPPORTED_RUNTIME or device_type != 'cuda' or dtype != 'torch.bfloat16':
        raise ValueError('Last-token logits require the audited torch2.4.1+cu121/transformers4.44.2/tokenizers0.19.1 CUDA BF16 runtime')


def audit_forward_source(source):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError('Native Qwen forward source could not be verified')
    function = tree.body[0]
    arguments = [arg.arg for arg in function.args.args]
    expected = ['self', 'input_ids', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict', 'cache_position']
    if arguments != expected or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise ValueError('Qwen forward signature differs from the audited Transformers4.44.2 implementation')
    assignments = []
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            assignments.append((statement.targets[0].id, ast.unparse(statement.value)))
    required = [('hidden_states', 'outputs[0]'), ('logits', 'self.lm_head(hidden_states)'), ('logits', 'logits.float()'), ('loss', 'None')]
    if not any(assignments[index:index+4] == required for index in range(len(assignments)-3)):
        raise ValueError('Qwen forward no longer uses the audited hidden-state/Linear/float32-logits sequence')
    head_calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self' and node.func.attr == 'lm_head']
    if len(head_calls) != 1:
        raise ValueError('Exactly one native lm_head invocation is required')
    return {'forward_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'forward_ast_sha256': digest(ast.dump(function, include_attributes=False)), 'upstream_source': UPSTREAM_SOURCE}


def _audit_qwen_model(model):
    import torch
    import transformers
    import tokenizers
    from transformers.generation.utils import GenerationMixin
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

    versions = {'torch': str(torch.__version__), 'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__}
    if type(model) is not Qwen2ForCausalLM or type(model.model) is not Qwen2Model or type(model.lm_head) is not torch.nn.Linear:
        raise ValueError('Only the original native Qwen2ForCausalLM/Qwen2Model/Linear architecture is supported')
    if getattr(model.forward, '__func__', None) is not Qwen2ForCausalLM.forward or getattr(model.generate, '__func__', None) is not GenerationMixin.generate or getattr(model.lm_head.forward, '__func__', None) is not torch.nn.Linear.forward:
        raise ValueError('Wrapped or replaced native forward/generation/head methods are unsupported')
    if 'generate' in model.__dict__ or getattr(model, '_compiled_call_impl', None) is not None or getattr(model, 'is_quantized', False) or getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
        raise ValueError('Patched, compiled or quantized Qwen models are unsupported')
    if model.config.model_type != 'qwen2' or model.config._attn_implementation != 'sdpa' or getattr(model.config, 'is_encoder_decoder', False):
        raise ValueError('The audited native decoder-only Qwen SDPA configuration is required')
    head = model.lm_head
    if head.bias is not None or head.in_features != model.config.hidden_size or head.out_features != model.config.vocab_size:
        raise ValueError('Native Qwen output-head dimensions/bias changed')
    validate_runtime(versions, head.weight.device.type, str(head.weight.dtype))
    parameters = list(model.parameters())
    if not parameters or any(parameter.device != head.weight.device or parameter.dtype != torch.bfloat16 for parameter in parameters):
        raise ValueError('Only a single-device BF16 model without offloading or mixed parameter dtypes is supported')
    if any(module.training or hasattr(module, '_hf_hook') for module in model.modules()):
        raise ValueError('Load an eval-only native model without offload/dispatch hooks')
    source = inspect.getsource(Qwen2ForCausalLM.forward)
    return {**audit_forward_source(source), 'versions': versions, 'model_class': 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM',
            'device': str(head.weight.device), 'dtype': str(head.weight.dtype),
            'native_generate_source_sha256': hashlib.sha256(inspect.getsource(GenerationMixin.generate).encode()).hexdigest(),
            'native_linear_source_sha256': hashlib.sha256(inspect.getsource(torch.nn.Linear.forward).encode()).hexdigest()}


class GenerationLastTokenLogits:
    def __init__(self, model):
        self.audit = _audit_qwen_model(model)
        self.model = model
        self.head = model.lm_head
        self.native_generate = model.generate
        self.native_forward = model.forward
        self.signature = inspect.signature(model.forward)
        self.lock = threading.Lock()
        self.generation_calls = 0
        self.forward_calls = 0
        self.last_generation = None
        self.metadata = {'runtime_protocol': RUNTIME_PROTOCOL, 'implementation_sha256': file_digest(__file__), 'native_audit': self.audit,
            'scope': 'model.generate only; one sequence with native cached greedy or sampling generation; direct forwards remain unchanged',
            'transformation': 'Temporary lm_head prehook selects hidden_states[:, -1:, :] before the original Linear and subsequent native float32 conversion',
            'unchanged': ['model weights', 'all transformer hidden-state computation', 'full attention inputs and history', 'KV-cache construction and positions', 'native vocabulary logits at the last position'],
            'prohibited': ['training', 'labels/loss computation during generation', 'grad-enabled generation', 'batching', 'beam/assisted/contrastive generation', 'compilation', 'quantization', 'offloading'],
            'numerical_equivalence': 'No bitwise equality claim; changed output-head GEMM shape requires separate cache identity and calibration'}

        @functools.wraps(self.native_generate)
        def generate(*args, **kwargs):
            return self._generate(args, kwargs)

        self.wrapper = generate
        model.generate = generate

    def assert_installed(self):
        if self.model.generate is not self.wrapper or self.model.lm_head is not self.head or self.model.forward != self.native_forward:
            raise ValueError('The audited generation wrapper or native model/head was replaced')

    def _inference_only(self):
        import torch
        if self.model.training or self.head.training or torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            raise ValueError('Last-token logits generation requires model.eval() inside torch.inference_mode(); training and gradients are forbidden')
        if torch.jit.is_tracing() or torch.jit.is_scripting() or getattr(getattr(torch, 'compiler', None), 'is_compiling', lambda: False)():
            raise ValueError('Compiled or traced generation is unsupported')

    def _validate_generate(self, args, kwargs):
        import torch
        self._inference_only()
        if any(module.training for module in self.model.modules()):
            raise ValueError('All model components must remain in evaluation mode during generation')
        if len(args) > 1 or args and ('input_ids' in kwargs or 'inputs' in kwargs):
            raise ValueError('Only one unambiguous input tensor and named generation options are supported')
        if 'labels' in kwargs or kwargs.get('inputs_embeds') is not None or kwargs.get('generation_config') is not None:
            raise ValueError('Labels, embedding-only inputs and alternate generation-config objects are unsupported')
        inputs = args[0] if args else kwargs.get('input_ids', kwargs.get('inputs'))
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] < 1 or inputs.dtype != torch.long:
            raise ValueError('One nonempty integer-token input sequence is required')
        config = self.model.generation_config
        value = lambda name, default=None: kwargs.get(name, getattr(config, name, default))
        if value('use_cache') is not True or value('num_beams', 1) != 1 or value('num_beam_groups', 1) != 1 or value('num_return_sequences', 1) != 1:
            raise ValueError('Last-token logits support only one sequence with native KV caching and no beams')
        if value('penalty_alpha') not in (None, 0) or value('constraints') or value('force_words_ids') or value('cache_implementation') is not None:
            raise ValueError('Contrastive/constrained-beam or nondefault cache modes are unsupported')
        if kwargs.get('assistant_model') is not None or kwargs.get('synced_gpus', False) or kwargs.get('past_key_values') is not None:
            raise ValueError('Assisted/distributed generation and externally supplied caches are unsupported')

    def _generate(self, args, kwargs):
        self.assert_installed()
        self._validate_generate(args, kwargs)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Concurrent/reentrant generation cannot share temporary output-head hooks')
        handles = []
        owner = threading.get_ident()
        active = [False]
        positions = []
        failed = True

        def forward_before(module, inputs, options):
            self._inference_only()
            if threading.get_ident() != owner or active[0]:
                raise RuntimeError('Concurrent or nested native forward during last-token generation')
            bound = self.signature.bind_partial(*inputs, **options).arguments
            if bound.get('labels') is not None or bound.get('use_cache', self.model.config.use_cache) is not True:
                raise ValueError('Generation forwards must be label-free and retain the native KV cache')
            active[0] = True

        def forward_after(module, inputs, options, output):
            active[0] = False

        def head_before(module, inputs, options):
            import torch
            self._inference_only()
            if threading.get_ident() != owner or not active[0] or len(inputs) != 1 or options:
                raise ValueError('Output-head slicing is permitted only within the native generation forward')
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1 or hidden.shape[2] != self.head.in_features or hidden.requires_grad or hidden.device != self.head.weight.device or hidden.dtype != self.head.weight.dtype:
                raise ValueError('Unexpected native output-head hidden-state shape, dtype, device or gradient state')
            positions.append(int(hidden.shape[1]))
            return (hidden[:, -1:, :],), options

        try:
            if self.model._forward_pre_hooks or self.model._forward_hooks or self.head._forward_pre_hooks or self.head._forward_hooks:
                raise ValueError('Existing model/head forward hooks conflict with the audited temporary hook lifecycle')
            handles.append(self.model.register_forward_pre_hook(forward_before, with_kwargs=True, prepend=True))
            handles.append(self.model.register_forward_hook(forward_after, with_kwargs=True, always_call=True))
            handles.append(self.head.register_forward_pre_hook(head_before, with_kwargs=True, prepend=True))
            output = self.native_generate(*args, **kwargs)
            if not positions:
                raise ValueError('Native generation never invoked the audited output head')
            failed = False
            return output
        finally:
            for handle in reversed(handles):
                handle.remove()
            self.generation_calls += 1
            self.forward_calls += len(positions)
            self.last_generation = {'status': 'failed' if failed else 'complete', 'native_hidden_sequence_lengths': positions,
                                    'head_sequence_lengths': [1]*len(positions), 'forward_calls': len(positions)}
            self.lock.release()

    def statistics(self):
        return {'generation_calls': self.generation_calls, 'forward_calls': self.forward_calls, 'last_generation': copy.deepcopy(self.last_generation)}

    def remove(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Cannot remove the adapter while generation is active')
        try:
            self.assert_installed()
            del self.model.generate
        finally:
            self.lock.release()


class LastTokenSDPARanker(ScalableSDPARanker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_token_adapter = None
        self.metadata.update(cache_namespace='crest-rankings-v4-json-byte-dfa-sdpa-last-token-logits-v1',
            last_token_runtime_protocol=RUNTIME_PROTOCOL, last_token_runtime_implementation_sha256=file_digest(__file__),
            last_token_logits_scope='Generation only, shared by ranking and continuation callers; no prompt truncation; separately calibrated',
            last_token_required_runtime=copy.deepcopy(SUPPORTED_RUNTIME))

    def load(self):
        super().load()
        if self._last_token_adapter is None:
            self._last_token_adapter = GenerationLastTokenLogits(self.model)
            self.metadata['generation_last_token_logits'] = copy.deepcopy(self._last_token_adapter.metadata)
        else:
            self._last_token_adapter.assert_installed()

    def statistics(self):
        result = super().statistics()
        result['last_token_logits'] = self._last_token_adapter.statistics() if self._last_token_adapter else None
        return result
