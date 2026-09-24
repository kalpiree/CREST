import functools
import json
import math
from dataclasses import dataclass

from .artifacts import digest, file_digest
from .scalable_inference import ByteLevelTokenInventory, _decode


OUTPUT_PROTOCOL = 'crest-keywords-json-byte-dfa-v1'
ASCII_ALPHABET = frozenset(chr(value) for value in range(32, 127))
STATE_CACHE_SIZE = 512


@dataclass(frozen=True)
class KeywordState:
    stage: str = 'start'
    used: int = 0
    node: int = 0


class KeywordGrammar:
    def __init__(self, inventory, keywords, count, eos_token_id, max_new_tokens):
        if not isinstance(keywords, (list, tuple)) or not keywords or any(not isinstance(word, str) or not word for word in keywords) or len(set(keywords)) != len(keywords):
            raise ValueError('Keyword vocabulary must contain distinct nonempty verbatim strings')
        if type(count) is not int or not 1 <= count <= len(keywords) or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('Keyword count and token budget must be explicit positive integers within the vocabulary')
        if type(eos_token_id) is not int or eos_token_id < 0 or eos_token_id not in inventory.special_ids or eos_token_id in inventory.pieces:
            raise ValueError('Native EOS must be a special token excluded from content')
        self.inventory, self.keywords, self.count = inventory, tuple(keywords), count
        self.eos_token_id, self.initial_state = eos_token_id, KeywordState()
        self.literals = tuple(json.dumps(word, ensure_ascii=True) for word in keywords)
        required = set('[], ') | set(''.join(self.literals))
        if not required <= set(inventory.single_char_ids):
            raise ValueError('Tokenizer lacks required single-character completion tokens: ' + repr(''.join(sorted(required - set(inventory.single_char_ids)))))
        maximum = 2 + sum(sorted(map(len, self.literals), reverse=True)[:count]) + 2 * (count - 1)
        if max_new_tokens < maximum + 1:
            raise ValueError(f'max_new_tokens must be at least {maximum + 1} to preserve every keyword tokenization plus EOS')
        self.children, self.masks, self.terminals = [{}], [0], {}
        for index, literal in enumerate(self.literals):
            node = 0
            self.masks[node] |= 1 << index
            for char in literal:
                if char not in self.children[node]:
                    self.children[node][char] = len(self.children)
                    self.children.append({}); self.masks.append(0)
                node = self.children[node][char]
                self.masks[node] |= 1 << index
            self.terminals[node] = index
        self.pieces_by_first = {}
        for token, piece in inventory.pieces.items():
            self.pieces_by_first.setdefault(piece[0], []).append((token, piece))
        self._allowed_cached = functools.lru_cache(maxsize=STATE_CACHE_SIZE)(self._allowed_for_state)
        self.metadata = {'output_protocol': OUTPUT_PROTOCOL, 'implementation_sha256': file_digest(__file__),
            'keywords': list(keywords), 'keywords_sha256': digest(list(keywords)), 'keyword_count': count,
            'vocabulary_size': len(keywords), 'valid_text_count': math.perm(len(keywords), count), 'enumerated_orderings': 0,
            'individual_literal_trie_nodes': len(self.children), 'token_inventory': inventory.metadata,
            'eos_token_id': eos_token_id, 'maximum_json_characters': maximum,
            'required_max_new_tokens_including_eos': maximum + 1, 'state_cache_capacity': STATE_CACHE_SIZE,
            'serialization': "json.dumps(list(keywords), ensure_ascii=True, separators=(', ', ': '))",
            'language': 'All compositional ASCII tokenizations of exactly count distinct verbatim supplied JSON string literals',
            'selection': 'Native generation settings unchanged; mask only tokens outside completable prefixes', 'repair': False}
        for word in self.keywords:
            text = json.dumps(word, ensure_ascii=True)
            ids = inventory.tokenizer.encode(text, add_special_tokens=False)
            if not isinstance(ids, list) or any(type(value) is not int or value not in inventory.pieces for value in ids) or ''.join(inventory.pieces[value] for value in ids) != text or _decode(inventory.tokenizer, ids) != text:
                raise ValueError('Native keyword literal tokenization is not representable compositionally')
        for selected in (self.keywords[:count], self.keywords[-count:]):
            text = json.dumps(list(selected), ensure_ascii=True, separators=(', ', ': '))
            preferred = inventory.tokenizer.encode(text, add_special_tokens=False)
            fallback = [inventory.single_char_ids[char] for char in text]
            if self._state_for_tokens(preferred).stage != 'done' or self._state_for_tokens(fallback).stage != 'done':
                raise ValueError('Native or character-wise keyword completion could not be verified')

    def _advance_character(self, state, char):
        stage, used, node = state.stage, state.used, state.node
        if stage == 'start' and char == '[':
            return KeywordState('word', used)
        if stage == 'word':
            child = self.children[node].get(char)
            if child is None or not self.masks[child] & ~used:
                return None
            if child in self.terminals:
                return KeywordState('after_word', used | (1 << self.terminals[child]))
            return KeywordState('word', used, child)
        if stage == 'after_word':
            if used.bit_count() == self.count and char == ']':
                return KeywordState('done', used)
            if used.bit_count() < self.count and char == ',':
                return KeywordState('space', used)
        if stage == 'space' and char == ' ':
            return KeywordState('word', used)
        return None

    def _consume(self, state, piece):
        for char in piece:
            state = self._advance_character(state, char)
            if state is None:
                return None
        return state

    def _state_for_tokens(self, suffix_ids):
        ids, state, rendered = tuple(suffix_ids), self.initial_state, []
        for token in ids:
            if type(token) is not int or token < 0:
                raise ValueError('Output token IDs must be nonnegative integers')
            piece = self.inventory.pieces.get(token)
            if piece is None:
                raise ValueError('Output contains an unknown or special content token')
            state = self._consume(state, piece)
            if state is None:
                raise ValueError('Output prefix is outside the distinct-keyword JSON grammar')
            rendered.append(piece)
        if _decode(self.inventory.tokenizer, ids) != ''.join(rendered):
            raise ValueError('Tokenizer changed compositional output rendering across a boundary')
        return state

    def _allowed_for_state(self, state):
        if state.stage == 'done':
            return (self.eos_token_id,)
        first = {'start': '[', 'space': ' ', 'after_word': ']' if state.used.bit_count() == self.count else ','}.get(state.stage)
        characters = first if first is not None else [char for char, child in self.children[state.node].items() if self.masks[child] & ~state.used]
        allowed = sorted(token for char in characters for token, piece in self.pieces_by_first.get(char, ()) if self._consume(state, piece) is not None)
        if not allowed:
            raise ValueError('No representable keyword completion remains')
        return tuple(allowed)

    def allowed_tokens(self, suffix_ids):
        return list(self._allowed_cached(self._state_for_tokens(suffix_ids)))

    def prefix_allowed_tokens_fn(self, prompt_token_count):
        if type(prompt_token_count) is not int or prompt_token_count < 1:
            raise ValueError('Positive prompt token count required')
        def allowed(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError('Keyword generation supports one input sequence')
            ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            if len(ids) < prompt_token_count:
                raise ValueError('Generation prefix is shorter than the prompt')
            return self.allowed_tokens(ids[prompt_token_count:])
        return allowed

    def validate_completed(self, generated_ids):
        ids = list(generated_ids)
        if not ids or type(ids[-1]) is not int or ids[-1] != self.eos_token_id or self._state_for_tokens(ids[:-1]).stage != 'done':
            raise ValueError('Generation must end with exactly the distinct keyword count followed by native EOS')
        return _decode(self.inventory.tokenizer, ids[:-1])


def build_keyword_grammar(tokenizer, keywords, count, eos_token_id, max_new_tokens, *, inventory=None):
    inventory = inventory or ByteLevelTokenInventory(tokenizer, (eos_token_id,), alphabet=ASCII_ALPHABET)
    if inventory.tokenizer is not tokenizer:
        raise ValueError('Token inventory belongs to another tokenizer')
    return KeywordGrammar(inventory, keywords, count, eos_token_id, max_new_tokens)
