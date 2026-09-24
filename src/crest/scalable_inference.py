import functools
import json
import math
import time
from dataclasses import dataclass

from . import constrained_inference
from .artifacts import digest, file_digest
from .constrained_inference import audit_generation_config
from .inference import InvalidRanking, LocalRanker
from .records import messages_for, parse_ranking

OUTPUT_PROTOCOL = 'crest-json-byte-dfa-v4'
MAX_CANDIDATES = 100
MAX_K = 20
STATE_CACHE_SIZE = 512
JSON_ALPHABET = frozenset('[]", C0123456789')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _shape(candidate_count, k):
    _positive_integer(candidate_count, 'candidate_count')
    _positive_integer(k, 'k')
    if k > candidate_count or candidate_count > MAX_CANDIDATES or k > MAX_K:
        raise ValueError('Scalable diagnostic grammar requires K <= C <= 100 and K <= 20')


def _decode(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


class ByteLevelTokenInventory:
    def __init__(self, tokenizer, extra_special_ids=(), *, alphabet=JSON_ALPHABET):
        alphabet = frozenset(alphabet)
        if not alphabet or any(not isinstance(char, str) or len(char) != 1 or not 32 <= ord(char) <= 126 for char in alphabet):
            raise ValueError('The compositional token inventory requires a nonempty printable ASCII alphabet')
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        decoder = getattr(backend, 'decoder', None)
        try:
            decoder_config = json.loads(decoder.__getstate__())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError('A verifiable compositional ByteLevel tokenizer decoder is required') from error
        if not isinstance(decoder_config, dict) or decoder_config.get('type') != 'ByteLevel':
            raise ValueError('Only the compositional ByteLevel decoder is supported; no tokenizer fallback is inferred')
        if _decode(tokenizer, []) != '':
            raise ValueError('Tokenizer emits content for an empty output prefix')
        specials = set(getattr(tokenizer, 'all_special_ids', [])) | set(extra_special_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in specials):
            raise ValueError('Special-token IDs must be nonnegative integers')
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, dict) or not vocab or any(not isinstance(piece, str) or isinstance(index, bool) or not isinstance(index, int) or index < 0 for piece,index in vocab.items()):
            raise ValueError('Tokenizer must expose its complete string-to-token-ID vocabulary')
        if len(set(vocab.values())) != len(vocab):
            raise ValueError('Tokenizer vocabulary has ambiguous duplicate token IDs')
        raw_alphabet = (alphabet - {' '}) | ({'Ġ', ' '} if ' ' in alphabet else set())
        pieces = {}
        for raw,index in sorted(vocab.items(), key=lambda row:row[1]):
            if index in specials or not raw or not set(raw) <= raw_alphabet:
                continue
            expected = raw.replace('Ġ', ' ')
            decoded = _decode(tokenizer, [index])
            if decoded != expected:
                raise ValueError('ByteLevel token-piece rendering is not compositional ASCII')
            pieces[index] = decoded
        if not pieces:
            raise ValueError('Tokenizer has no eligible canonical-JSON pieces')
        self.tokenizer = tokenizer
        self.pieces = pieces
        self.special_ids = frozenset(specials)
        self.single_char_ids = {char:min(index for index,piece in pieces.items() if piece == char) for char in alphabet if any(piece == char for piece in pieces.values())}
        self.metadata = {'decoder':decoder_config, 'ascii_token_piece_count':len(pieces), 'ascii_token_pieces_sha256':digest(pieces), 'special_token_ids':sorted(specials), 'vocabulary_size':len(vocab), 'composition_policy':'ByteLevel raw-byte alphabet; each eligible ASCII token is decoded and checked without cleanup; emitted prefixes are checked compositionally'}
        if alphabet != JSON_ALPHABET:
            self.metadata['ascii_alphabet'] = ''.join(sorted(alphabet))


@dataclass(frozen=True)
class GrammarState:
    stage: str = 'start'
    used: int = 0
    digits: str = ''


class LazyRankingGrammar:
    def __init__(self, inventory, candidate_count, k, eos_token_id, max_new_tokens):
        _shape(candidate_count, k)
        _positive_integer(max_new_tokens, 'max_new_tokens')
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError('An integer native EOS token is required')
        if eos_token_id not in inventory.special_ids or eos_token_id in inventory.pieces:
            raise ValueError('The selected native EOS must be excluded from ranking content')
        self.inventory = inventory
        self.candidate_count = candidate_count
        self.k = k
        self.eos_token_id = eos_token_id
        self.initial_state = GrammarState()
        self.labels = tuple(f'C{index}' for index in range(candidate_count))
        self.prefix_masks = {}
        for index in range(candidate_count):
            digits = str(index)
            for length in range(1, len(digits)+1):
                prefix = digits[:length]
                self.prefix_masks[prefix] = self.prefix_masks.get(prefix, 0) | (1 << index)
        required = set('[]", C') | set(''.join(str(index) for index in range(candidate_count)))
        if not required <= set(inventory.single_char_ids):
            missing = ''.join(sorted(required-set(inventory.single_char_ids)))
            raise ValueError(f'Tokenizer lacks required single-character completion tokens: {missing!r}')
        lengths = sorted((len(json.dumps(label)) for label in self.labels), reverse=True)
        maximum_characters = 2 + sum(lengths[:k]) + 2*(k-1)
        if max_new_tokens < maximum_characters + 1:
            raise ValueError(f'max_new_tokens must be at least {maximum_characters + 1} to preserve every completable tokenization plus EOS')
        self._allowed_cached = functools.lru_cache(maxsize=STATE_CACHE_SIZE)(self._allowed_for_state)
        self.metadata = {'output_protocol':OUTPUT_PROTOCOL, 'candidate_count':candidate_count, 'k':k, 'valid_ranking_text_count':math.perm(candidate_count,k), 'enumerated_rankings':0, 'token_inventory':inventory.metadata, 'eos_token_id':eos_token_id, 'maximum_json_characters':maximum_characters, 'required_max_new_tokens_including_eos':maximum_characters+1, 'state_cache_capacity':STATE_CACHE_SIZE, 'serialization':"json.dumps(list(labels), ensure_ascii=True, separators=(', ', ': '))", 'language':'Every compositional ASCII tokenization of canonical JSON containing exactly K distinct candidate labels; may differ from canonical-encoding trie greedy choices'}
        for selected in (self.labels[:k], self.labels[-k:]):
            text = json.dumps(list(selected), ensure_ascii=True, separators=(', ', ': '))
            fallback = [inventory.single_char_ids[char] for char in text]
            if self._state_for_tokens(fallback).stage != 'done':
                raise ValueError('Single-character tokenizer completion could not be verified')
            preferred = inventory.tokenizer.encode(text, add_special_tokens=False)
            if not isinstance(preferred, list) or self._state_for_tokens(preferred).stage != 'done':
                raise ValueError('Native canonical tokenization cannot be represented by the output grammar')

    def _advance_character(self, state, char):
        stage, used, digits = state.stage, state.used, state.digits
        if stage == 'start' and char == '[':
            return GrammarState('quote', used)
        if stage == 'quote' and char == '"':
            return GrammarState('label_c', used)
        if stage == 'label_c' and char == 'C':
            return GrammarState('digits', used)
        if stage == 'digits':
            if char in '0123456789':
                prefix = digits + char
                if self.prefix_masks.get(prefix,0) & ~used:
                    return GrammarState('digits', used, prefix)
            elif char == '"' and digits:
                label = int(digits)
                if str(label) == digits and label < self.candidate_count and not used & (1 << label):
                    return GrammarState('after_label', used | (1 << label))
        if stage == 'after_label':
            if used.bit_count() == self.k and char == ']':
                return GrammarState('done', used)
            if used.bit_count() < self.k and char == ',':
                return GrammarState('space', used)
        if stage == 'space' and char == ' ':
            return GrammarState('quote', used)
        return None

    def _consume(self, state, piece):
        for char in piece:
            state = self._advance_character(state, char)
            if state is None:
                return None
        return state

    def _state_for_tokens(self, suffix_ids):
        state = self.initial_state
        rendered = []
        ids = tuple(suffix_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
            raise ValueError('Output token IDs must be nonnegative integers')
        for value in ids:
            piece = self.inventory.pieces.get(value)
            if piece is None:
                raise ValueError('Output prefix contains an unknown or special content token')
            state = self._consume(state, piece)
            if state is None:
                raise ValueError('Output token prefix is outside the distinct-label JSON grammar')
            rendered.append(piece)
        if _decode(self.inventory.tokenizer, ids) != ''.join(rendered):
            raise ValueError('Tokenizer changed output-piece rendering across a boundary')
        return state

    def _allowed_for_state(self, state):
        if state.stage == 'done':
            return (self.eos_token_id,)
        allowed = tuple(index for index,piece in self.inventory.pieces.items() if self._consume(state,piece) is not None)
        if not allowed:
            raise ValueError('No representable completion remains for this grammar prefix')
        return allowed

    def allowed_tokens(self, suffix_ids):
        return list(self._allowed_cached(self._state_for_tokens(suffix_ids)))

    def prefix_allowed_tokens_fn(self, prompt_token_count):
        _positive_integer(prompt_token_count, 'prompt_token_count')
        def allowed(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError('The scalable decoder accepts one greedy input sequence at a time')
            ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            if len(ids) < prompt_token_count:
                raise ValueError('Generation prefix is shorter than the prompt')
            return self.allowed_tokens(ids[prompt_token_count:])
        return allowed

    def validate_completed(self, generated_ids):
        ids = list(generated_ids)
        if not ids or isinstance(ids[-1], bool) or not isinstance(ids[-1], int) or ids[-1] != self.eos_token_id or self._state_for_tokens(ids[:-1]).stage != 'done':
            raise ValueError('Generation must end with exactly K distinct canonical labels followed by native EOS')
        return _decode(self.inventory.tokenizer, ids[:-1])


def build_ranking_grammar(tokenizer, candidate_count, k, eos_token_id, max_new_tokens, *, inventory=None):
    _shape(candidate_count, k)
    inventory = inventory or ByteLevelTokenInventory(tokenizer, (eos_token_id,))
    if inventory.tokenizer is not tokenizer:
        raise ValueError('Token inventory belongs to another tokenizer')
    return LazyRankingGrammar(inventory, candidate_count, k, eos_token_id, max_new_tokens)


class ScalableLocalRanker(LocalRanker):
    def __init__(self, model_path, revision, cache_path, device='cuda:0', dtype='bfloat16', max_input_tokens=4096, max_new_tokens=128):
        super().__init__(model_path, revision, cache_path, device, dtype, max_input_tokens, max_new_tokens)
        self.metadata.update({'output_protocol':OUTPUT_PROTOCOL, 'cache_namespace':'crest-rankings-v4-json-byte-dfa', 'scalable_inference_implementation_hash':file_digest(__file__), 'generation_audit_implementation_hash':file_digest(constrained_inference.__file__), 'output_constraint':{'kind':'lazy distinct-label canonical JSON byte-level state machine', 'maximum_candidates':MAX_CANDIDATES, 'maximum_k':MAX_K, 'rankings_enumerated':False, 'ranking_choice':'Native greedy generation after masking only tokens outside a completable grammar prefix; every legal token remains available', 'candidate_order':'Unchanged input order mapped to C0, C1, ...', 'repair':False, 'protocol_change':'All valid ASCII tokenizations are supported; ranking behavior is not asserted equivalent to the v3 canonical-tokenization trie and requires separate calibration'}})
        self._inventory = None
        self._grammars = {}
        self._generation_audited = False

    def load(self):
        super().load()
        if not self._generation_audited:
            defaults = self.model.generation_config.to_dict()
            audit = audit_generation_config(defaults, self.tokenizer.eos_token_id)
            if defaults.get('stop_strings') or defaults.get('max_time') is not None:
                raise ValueError('Native early stopping would invalidate the complete-output grammar contract')
            self._inventory = ByteLevelTokenInventory(self.tokenizer, audit['native_eos_token_ids'])
            self.metadata['scalable_generation'] = {**audit, 'native_generation_defaults':defaults, 'token_inventory':self._inventory.metadata, 'explicit_generation_overrides':{'do_sample':False,'num_beams':1,'max_new_tokens':self.metadata['max_new_tokens'],'pad_token_id':self.tokenizer.eos_token_id,'use_cache':True}, 'additional_override':'prefix_allowed_tokens_fn bound to the lazy distinct-label JSON grammar'}
            self._generation_audited = True

    def rank(self, decision, k, omitted=frozenset(), datamark=False):
        messages = messages_for(decision, k, omitted, datamark)
        _shape(len(decision['candidates']), k)
        labels = {f'C{index}':item for index,item in enumerate(decision['candidates'])}
        self.load()
        shape = (len(labels),k)
        if shape not in self._grammars:
            self._grammars[shape] = build_ranking_grammar(self.tokenizer,len(labels),k,self.metadata['scalable_generation']['selected_eos_token_id'],self.metadata['max_new_tokens'],inventory=self._inventory)
        grammar = self._grammars[shape]
        key = digest({'inference':self.metadata,'grammar':grammar.metadata,'messages':messages,'k':k,'candidates':decision['candidates']})
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return parse_ranking(json.dumps(cached['ranking']),decision['candidates'],k)
        import torch
        prompt = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs = self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(self.device)
        input_count = inputs['input_ids'].shape[1]
        if input_count > self.metadata['max_input_tokens']:
            raise ValueError('Prompt exceeds the configured context budget; no records are truncated')
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=self.metadata['max_new_tokens'],pad_token_id=self.tokenizer.eos_token_id,use_cache=True,prefix_allowed_tokens_fn=grammar.prefix_allowed_tokens_fn(input_count))
        generated = output[0,input_count:]
        generated_ids = generated.tolist()
        response = self.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        self.calls += 1
        self.input_tokens += input_count
        self.output_tokens += len(generated)
        try:
            canonical = grammar.validate_completed(generated_ids)
            if canonical != response:
                raise ValueError('Decoded response differs from the verified canonical output pieces')
            output_labels = parse_ranking(response,list(labels),k)
            ranking = [labels[label] for label in output_labels]
        except (ValueError,TypeError) as error:
            self.cache.failure(key,error,response,{'messages':messages,'k':k,'labels':labels,'inference':self.metadata,'grammar':grammar.metadata,'generated_token_ids':generated_ids,'raw_response_with_special_tokens':_decode(self.tokenizer,generated_ids)})
            raise InvalidRanking(f'Invalid scalable model ranking recorded in cache: {error}') from error
        self.cache.put(key,{'ranking':ranking,'response':response,'input_tokens':input_count,'output_tokens':len(generated),'seconds':time.monotonic()-started,'grammar':grammar.metadata,'output_protocol':OUTPUT_PROTOCOL})
        return ranking
