import copy
import fcntl
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .records import parse_ranking, validate_decision

VERSION = 'textsimu-recatom-paper-pipeline-v1'
PAPER = 'https://arxiv.org/html/2409.11690v3'
GEMMA = 'google/gemma-1.1-2b-it'
QWEN = 'Qwen/Qwen2.5-7B-Instruct'
CONTENT_FIELDS = {'title', 'description', 'text', 'content', 'review', 'review_text', 'body', 'summary'}
KEYWORD_SELECTION_INSTRUCTION = ('Return only a JSON array of exactly 20 distinct strings selected verbatim from the supplied ranked_keywords list. '
    'Select exactly 20 even when fewer specific attributes are available. Prefer appealing, specific product attributes and avoid generic terms where possible; '
    'the exact count requirement takes precedence. Do not add, combine, rephrase, or repeat keywords. Do not include prose or Markdown.')


def explicit_preservation_request(request):

    request = copy.deepcopy(request)
    inputs = request.get('inputs', {})
    rule = inputs.get('preservation_rule')
    if not isinstance(rule, dict) or rule.get('kind') != 'retain_original_numeric_and_title_tokens':
        return request
    words = re.findall(r"\b[\w'-]+\b", inputs['original_content'])
    numeric = {word.casefold() if rule['casefold'] else word for word in words if any(c.isdecimal() for c in word)}
    inputs['required_verbatim_tokens'] = sorted(set(rule['required_title_tokens']) | numeric)


    chunks = list(dict.fromkeys(w for w in inputs['original_content'].split() if any(c.isdecimal() for c in w)))
    title = inputs.get('source_title') or ' '.join(rule['required_title_tokens'])
    prefix = title + '\n'
    if chunks:
        prefix += 'Source specifications: ' + ' '.join(chunks) + '\n'
    inputs['preserved_source_prefix'] = prefix + '\n'
    request['system'] += (' Begin with preserved_source_prefix verbatim, then write the rewritten product description. '
        'The prefix contains immutable source identity/specifications. Do not treat reference product facts as source facts.')
    if inputs.get('source_title'):
        request['system'] += (' Begin the returned description with this exact source product title: ' + json.dumps(inputs['source_title']) + '. '
            'Then describe only the source product from original_content. Reference content supplies writing style only: never copy its product names, prices, dates or specifications. '
            'Keep the original numeric expressions verbatim, including attached units and punctuation.')
    request['system'] += (' Every token in required_verbatim_tokens must occur in the returned description as a complete word token. '
        'Copy those tokens exactly; do not pluralize them, change units, spell out digits, split a token, or attach extra letters or digits. '
        'The title tokens identify the source product even if they were absent from original_content. '
        'Check that all required tokens are present before returning the description. The existing No abstention option, when offered, remains allowed.')
    return request


def vocabulary_complete_references(ranked, count, required_keywords, tokenize):


    if not 0 < required_keywords < count or len(ranked) < count:
        raise ValueError('Vocabulary completion requires more reference slots than required keywords')
    selected = list(ranked[:count])
    replacements = []
    def words(row):
        return set(tokenize(row['text']))
    sets = [words(row) for row in selected]
    for candidate in ranked[count:]:
        union = set().union(*sets)
        if len(union) >= required_keywords:
            break
        added = words(candidate)
        if added <= union:
            continue
        redundant = [i for i, value in enumerate(sets) if value <= set().union(*(sets[:i] + sets[i+1:]))]
        if not redundant:
            raise ValueError('Reference vocabulary invariant violated')
        index = max(redundant, key=lambda i: (-selected[i]['cosine'], selected[i]['item_id']))
        removed = selected[index]
        selected[index], sets[index] = candidate, added
        replacements.append({'removed_item_id': removed['item_id'], 'added_item_id': candidate['item_id'],
                             'vocabulary_after': len(set().union(*sets))})
    vocabulary = set().union(*sets)
    if len(vocabulary) < required_keywords:
        raise ValueError('Frozen reference corpus cannot supply the required keyword vocabulary')
    selected.sort(key=lambda row: (-row['cosine'], row['item_id']))
    return {'selected': selected, 'replacements': replacements, 'distinct_keywords': len(vocabulary),
            'policy': 'Nearest 50 retained when feasible; replace only vocabulary-redundant references to permit the frozen 40-candidate keyword stage',
            'victim_feedback_used': False, 'adaptation_not_original_textsimu': True}


class TextSimuFailure(RuntimeError):
    pass


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an explicit integer >= {minimum}')
    return value


def _sha(value, name, length=64):
    if not isinstance(value, str) or len(value) != length or any(char not in '0123456789abcdef' for char in value):
        raise ValueError(f'{name} must be a lowercase {length}-character digest')


def _resolved(value):
    if value is None or isinstance(value, str) and value.strip().lower() in {'', 'unknown', 'pending', 'tbd', 'unresolved'}:
        raise ValueError('All pipeline settings and provider identities must be resolved')
    if isinstance(value, dict):
        for child in value.values():
            _resolved(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _resolved(child)
    digest(value)


def _view(decision):
    validate_decision(decision)
    return {'user_id': decision['user_id'], 'candidates': list(decision['candidates']),
            'records': [{'id': record['id'], 'type': record['type'], 'fields': dict(record['fields'])} for record in decision['records']]}


def _vectors(values, count):
    if not isinstance(values, list) or len(values) != count or not values:
        raise ValueError('Embedding provider must return one vector per input text')
    dimension = len(values[0]) if isinstance(values[0], (list, tuple)) else 0
    if dimension < 1:
        raise ValueError('Embedding vectors must be nonempty')
    normalized = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != dimension or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError('Embedding dimensions and finite numeric values must agree')
        norm = math.hypot(*row)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('Cosine similarity requires nonzero finite embedding norms')
        normalized.append([x / norm for x in row])
    return normalized


class PinnedGemmaEmbeddings:
    def __init__(self, artifact):
        body = {key: value for key, value in artifact.items() if key != 'sha256'}
        if artifact.get('sha256') != digest(body) or artifact.get('status') != 'verified' or artifact.get('model_id') != GEMMA:
            raise ValueError('A verified, checksum-pinned original Gemma embedding artifact is required')
        _sha(artifact.get('revision'), 'Gemma revision', 40)
        _sha(artifact.get('implementation_sha256'), 'embedding implementation')
        parameters = artifact.get('embedding_parameters')
        if not isinstance(parameters, dict) or not {'pooling', 'max_input_tokens', 'truncation'} <= parameters.keys():
            raise ValueError('Embedding pooling and token handling must be explicitly pinned')
        _resolved(parameters)
        if not isinstance(artifact.get('vectors'), dict) or not artifact['vectors']:
            raise ValueError('Embedding artifact contains no text-digest vectors')
        for key, row in artifact['vectors'].items():
            _sha(key, 'embedded text digest')
            _vectors([row], 1)
        self.vectors = copy.deepcopy(artifact['vectors'])
        self.metadata = {key: copy.deepcopy(body[key]) for key in ('model_id', 'revision', 'embedding_parameters', 'implementation_sha256')}
        self.metadata.update(artifact_sha256=artifact['sha256'], text_identity='crest.artifacts.digest(text)')

    def __call__(self, texts):
        missing = [digest(text) for text in texts if digest(text) not in self.vectors]
        if missing:
            raise ValueError('Pinned Gemma embeddings are unavailable for requested text digests: ' + ', '.join(missing))
        return [list(self.vectors[digest(text)]) for text in texts]


def textrank_adjacent(documents, *, damping, tolerance, max_iterations, edge_weighting, self_edges):
    if isinstance(damping, bool) or not isinstance(damping, (float, int)) or not 0 < damping < 1:
        raise ValueError('TextRank damping must be explicitly between zero and one')
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('TextRank tolerance must be finite and positive')
    _integer(max_iterations, 'TextRank maximum iterations')
    if edge_weighting not in {'binary', 'cooccurrence_count'} or not isinstance(self_edges, bool):
        raise ValueError('Resolve TextRank adjacent-edge weighting and self-edge policy')
    graph = defaultdict(dict)
    for tokens in documents:
        if not isinstance(tokens, (list, tuple)) or any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError('Keyword tokenizer must return nonempty string tokens')
        for token in tokens:
            graph[token]
        for left, right in zip(tokens, tokens[1:]):
            if left == right and not self_edges:
                continue
            weight = 1 if edge_weighting == 'binary' else graph[left].get(right, 0) + 1
            graph[left][right] = weight
            graph[right][left] = weight
    nodes = sorted(graph)
    if not nodes:
        raise ValueError('Keyword graph is empty')
    n = len(nodes)
    scores = {node: 1 / n for node in nodes}
    strengths = {node: math.fsum(graph[node].values()) for node in nodes}
    for iteration in range(max_iterations):
        dangling = math.fsum(scores[node] for node in nodes if not strengths[node]) / n
        updated = {node: (1-damping)/n + damping*dangling for node in nodes}
        for left in nodes:
            if strengths[left]:
                for right, weight in graph[left].items():
                    updated[right] += damping*scores[left]*weight/strengths[left]
        error = math.fsum(abs(updated[node]-scores[node]) for node in nodes)
        scores = updated
        if error <= tolerance:
            return {'scores': [{'keyword': node, 'score': scores[node]} for node in sorted(nodes, key=lambda node: (-scores[node], node))],
                    'iterations': iteration+1, 'residual_l1': error, 'nodes': n,
                    'edge_count': sum(1 for left in nodes for right in graph[left] if left <= right),
                    'graph': 'undirected adjacent tokens within each reference; no cross-document edges'}
    raise ValueError('TextRank did not converge within its frozen iteration budget')


class TextSimuRecAtom:
    def __init__(self, frozen_settings, reference_manifest, embedding_provider, tokenize, attacker, victim_ranker, content_validator):
        if not isinstance(frozen_settings, dict) or set(frozen_settings) != {'status', 'parameters', 'provenance', 'sha256'} or frozen_settings['status'] != 'frozen':
            raise ValueError('A frozen settings/provenance envelope is required')
        if frozen_settings['sha256'] != digest({key: frozen_settings[key] for key in ('parameters', 'provenance')}):
            raise ValueError('Frozen settings checksum differs')
        config = frozen_settings['parameters']
        required = {'seed', 'text_field', 'affected_decisions', 'feedback_decisions', 'feedback_k', 'reference_count', 'keyword_count', 'persona_count', 'textrank', 'max_discussion_rounds', 'max_speaker_attempts', 'feedback_query_budget', 'feedback_policy', 'minimum_frequency_gain', 'maximum_output_characters', 'final_selection', 'target_selection_provenance'}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError('TextSimu settings have missing or unknown fields')
        _resolved(frozen_settings)
        if frozen_settings['provenance'].get('qwen_attacker_user_approved') is not True:
            raise ValueError('The material Qwen attacker substitution must have explicit user approval provenance')
        for key in ('feedback_k', 'max_discussion_rounds', 'max_speaker_attempts', 'feedback_query_budget', 'maximum_output_characters'):
            _integer(config[key], key)
        _integer(config['seed'], 'seed', 0)
        for key, expected in (('reference_count', 50), ('keyword_count', 20), ('persona_count', 5)):
            if type(config[key]) is not int or config[key] != expected:
                raise ValueError(f'Original {key}={expected} is required by this method-preserving adapter')
        for key in ('affected_decisions', 'feedback_decisions'):
            values = config[key]
            if not isinstance(values, list) or not values or any(type(index) is not int or index < 0 for index in values) or values != sorted(set(values)):
                raise ValueError(f'{key} must be a frozen nonempty sorted distinct index list')
        if config['feedback_query_budget'] < 2*len(config['feedback_decisions']):
            raise ValueError('Feedback budget must permit original observed feedback and the first summarized revision')
        gain = config['minimum_frequency_gain']
        if isinstance(gain, bool) or not isinstance(gain, (float, int)) or not math.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError('Resolve a finite recommendation-frequency gain stopping threshold in [0,1]')
        if config['text_field'] not in CONTENT_FIELDS or config['feedback_policy'] != 'after_each_accepted_revision' or config['final_selection'] != 'last_tested':
            raise ValueError('Only explicit content rewrites with feedback after accepted revisions and last-tested selection are implemented')
        if not isinstance(config['target_selection_provenance'], dict) or not config['target_selection_provenance']:
            raise ValueError('The externally selected target needs frozen popularity-selection provenance')
        tr = config['textrank']
        if not isinstance(tr, dict) or set(tr) != {'candidate_count', 'damping', 'tolerance', 'max_iterations', 'edge_weighting', 'self_edges'}:
            raise ValueError('TextRank graph, convergence and candidate keyword count must be explicit')
        _integer(tr['candidate_count'], 'TextRank candidate count', config['keyword_count'])
        textrank_adjacent([['validation']], **{key: value for key, value in tr.items() if key != 'candidate_count'})
        refs = {key: value for key, value in reference_manifest.items() if key != 'sha256'}
        if set(refs) != {'status', 'source_split', 'popularity_selection', 'items'} or refs['status'] != 'frozen' or refs['source_split'] not in {'train', 'development'} or reference_manifest.get('sha256') != digest(refs):
            raise ValueError('References require a pinned train/development popularity manifest; held-out sources are forbidden')
        if not isinstance(refs['popularity_selection'], dict) or not refs['popularity_selection']:
            raise ValueError('Popular-reference selection provenance is required')
        items = refs['items']
        if not isinstance(items, list) or len(items) < config['reference_count'] or any(not isinstance(row, dict) or set(row) != {'item_id', 'text'} or any(not isinstance(row[key], str) or not row[key] for key in row) for row in items) or len({row['item_id'] for row in items}) != len(items):
            raise ValueError('Reference items must have distinct IDs and existing nonempty text')
        self.providers = {'embeddings': embedding_provider, 'tokenizer': tokenize, 'attacker': attacker, 'victim': victim_ranker, 'content_validator': content_validator}
        for name, provider in self.providers.items():
            if not getattr(provider, 'metadata', None) or not (callable(getattr(provider, 'rank', None)) if name == 'victim' else callable(provider)):
                raise ValueError(f'A usable provider with frozen metadata is required: {name}')
        self.provider_metadata = copy.deepcopy({name: provider.metadata for name, provider in self.providers.items()})
        digest(self.provider_metadata)
        reference_model = self.provider_metadata['embeddings'].get('model_id')
        if reference_model == QWEN:
            if frozen_settings['provenance'].get('qwen_reference_encoder_user_approved') is not True:
                raise ValueError('Qwen reference embeddings require separate explicit user approval provenance')
        elif reference_model != GEMMA:
            raise ValueError('Reference embeddings require original Gemma or the explicitly approved Qwen substitution')
        for name, model in (('embeddings', reference_model), ('attacker', QWEN)):
            metadata = self.provider_metadata[name]
            if metadata.get('model_id') != model:
                raise ValueError(f'{name} must use {model}; no silent substitution is permitted')
            _sha(metadata.get('revision'), name+' revision', 40)
            _sha(metadata.get('implementation_sha256'), name+' implementation')
        embedding_parameters = self.provider_metadata['embeddings'].get('embedding_parameters')
        if not isinstance(embedding_parameters, dict) or not {'pooling', 'max_input_tokens', 'truncation'} <= embedding_parameters.keys():
            raise ValueError('Reference embedding pooling and tokenizer parameters are required')
        _resolved(embedding_parameters)
        if not self.provider_metadata['attacker'].get('generation_parameters') or not self.provider_metadata['content_validator'].get('rule'):
            raise ValueError('Attacker generation settings and content-preservation rule must be pinned')
        self.config = copy.deepcopy(config)
        self.settings = copy.deepcopy(frozen_settings)
        self.references = copy.deepcopy(reference_manifest)
        self.metadata = {'name': 'TextSimu-RecAtom', 'adapter_version': VERSION, 'paper': PAPER,
            'implementation_sha256': file_digest(__file__), 'settings_sha256': frozen_settings['sha256'],
            'settings': self.settings, 'reference_manifest_sha256': reference_manifest['sha256'], 'providers': self.provider_metadata,
            'is_full_reproduction': False, 'original_code_repository': None,
            'material_changes': ['User-approved pinned Qwen attacker replaces GPT-4o mini', 'Frozen victim inference replaces training-text poisoning', 'One target RecAtom content field and fixed candidate/decision schedules', 'Training/development references only', 'Explicit bounded speaker retries and feedback after each accepted revision; original budgets unspecified'],
            'feedback_unit': 'One logical observed-victim ranking request, including cache hits; no evaluator clean rankings or labels',
            'embedding_fallback': 'None; the explicitly frozen reference provider must be available',
            'reference_encoder': reference_model}
        if reference_model == QWEN:
            self.metadata['material_changes'].append('User-approved pinned Qwen embeddings replace the original Gemma reference encoder')

    def _providers_unchanged(self):
        if digest({name: provider.metadata for name, provider in self.providers.items()}) != digest(self.provider_metadata):
            raise ValueError('Provider metadata changed; finish loading and freeze providers before constructing TextSimu')

    def run(self, observed_decisions, target_item_id, record_id, original_text, output_dir, *, check=None, edited_item_id=None):
        config = self.config
        edited_item_id = target_item_id if edited_item_id is None else edited_item_id
        if not isinstance(edited_item_id, str) or not edited_item_id:
            raise ValueError('An explicit edited source item is required')
        if edited_item_id != target_item_id and self.settings['provenance'].get('multi_record_same_target_user_approved') is not True:
            raise ValueError('Editing a non-target source record requires explicit multiple-record same-target approval')
        if not isinstance(observed_decisions, list) or not observed_decisions or any(not isinstance(value, str) or not value for value in (target_item_id, record_id, original_text)):
            raise ValueError('Observed decisions and explicit target identity/original text are required')
        views = [_view(decision) for decision in observed_decisions]
        if max(config['affected_decisions']+config['feedback_decisions']) >= len(views):
            raise ValueError('Frozen attack/feedback schedule exceeds observed decisions')
        field = config['text_field']
        for index, decision in enumerate(views):
            matches = [row for row in decision['records'] if row['id'] == record_id]
            if index in config['affected_decisions'] and len(matches) != 1:
                raise ValueError('Target RecAtom is absent from an affected decision')
            for record in matches:
                if record['type'] != 'item_text' or record['fields'].get('item_id') != edited_item_id or record['fields'].get(field) != original_text:
                    raise ValueError('The edited identity, field, item reference or original text changed')
            if index in config['feedback_decisions'] and (target_item_id not in decision['candidates'] or config['feedback_k'] > len(decision['candidates'])):
                raise ValueError('Observed feedback requires the frozen target and valid K in fixed candidates')
        references = [row for row in self.references['items'] if row['item_id'] not in {target_item_id, edited_item_id}]
        if len(references) < config['reference_count']:
            raise ValueError('Too few eligible popular references after excluding the target')
        self._providers_unchanged()
        immutable = {'version': VERSION, 'generator': self.metadata, 'observed_sha256': digest(observed_decisions),
                     'record_field_order': [[list(row['fields']) for row in decision['records']] for decision in views],
                     'target_item_id': target_item_id, 'edited_item_id': edited_item_id, 'record_id': record_id, 'original_text_sha256': digest(original_text)}
        fingerprint = digest(immutable)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        trace = []
        check = check or (lambda: None)

        def changed(text, original=views):
            result = copy.deepcopy(original)
            for index in config['affected_decisions']:
                for row in result[index]['records']:
                    if row['id'] == record_id:
                        row['fields'][field] = text
            return result

        def validate_text(value):
            if not isinstance(value, str) or not value.strip() or len(value) > config['maximum_output_characters']:
                raise ValueError('Generated content is empty, malformed or exceeds the frozen character budget')
            if self.providers['content_validator'](original_text, value) is not True:
                raise ValueError('Generated content violates the declared preservation rule')
            return value

        with (directory/'.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = directory/'manifest.json'
            if manifest.exists():
                old = read_json(manifest)
                if old.get('fingerprint') != fingerprint or digest(old.get('immutable')) != fingerprint:
                    raise ValueError('TextSimu resume inputs/providers/settings changed')
            else:
                write_json(manifest, {'fingerprint': fingerprint, 'immutable': immutable})

            def stage(name, request, compute, validate=lambda value: value):
                self._providers_unchanged()
                identity = digest(name)
                intent, path = directory/'stages'/(identity+'.intent.json'), directory/'stages'/(identity+'.result.json')
                request = copy.deepcopy(request)
                request_hash = digest(request)
                if path.exists():
                    old = read_json(path)
                    body = {key: value for key, value in old.items() if key != 'sha256'}
                    if old.get('fingerprint') != fingerprint or old.get('name') != name or old.get('request_sha256') != request_hash or old.get('sha256') != digest(body):
                        raise ValueError('Corrupt or incompatible TextSimu stage checkpoint')
                    if not intent.exists() or read_json(intent) != {'fingerprint': fingerprint, 'name': name, 'request_sha256': request_hash, 'request': request}:
                        raise ValueError('TextSimu stage intent changed')
                    trace.append({'name': name, 'sha256': old['sha256'], 'status': old['status']})
                    if old['status'] != 'complete':
                        raise TextSimuFailure('Preserved failed TextSimu stage: '+str(name))
                    return copy.deepcopy(old['response'])
                if intent.exists():
                    raise TextSimuFailure('Uncertain interrupted provider call; refusing automatic duplicate: '+str(name))
                check()
                write_json(intent, {'fingerprint': fingerprint, 'name': name, 'request_sha256': request_hash, 'request': request})
                response = None
                raw_response = None
                try:
                    raw_response = compute(copy.deepcopy(request))
                    response = validate(copy.deepcopy(raw_response))
                    self._providers_unchanged()
                    digest(response)
                    body = {'fingerprint': fingerprint, 'name': name, 'request_sha256': request_hash, 'status': 'complete', 'response': response, 'raw_response': raw_response}
                except Exception as error:
                    try:
                        digest(raw_response)
                    except (TypeError, ValueError, OverflowError):
                        raw_response = {'invalid_type': type(raw_response).__name__, 'representation': repr(raw_response)}
                    body = {'fingerprint': fingerprint, 'name': name, 'request_sha256': request_hash, 'status': 'failed', 'response': None, 'raw_response': raw_response, 'error': {'type': type(error).__name__, 'message': str(error)}}
                    write_json(path, {**body, 'sha256': digest(body)})
                    raise TextSimuFailure('Failed TextSimu stage preserved: '+str(name)) from error
                record = {**body, 'sha256': digest(body)}
                write_json(path, record)
                trace.append({'name': name, 'sha256': record['sha256'], 'status': 'complete'})
                check()
                return copy.deepcopy(response)

            def llm(name, instruction, inputs, validate=validate_text, output_format='text'):
                request = {'stage': name, 'system': instruction, 'inputs': inputs,
                           'seed': int(digest({'seed': config['seed'], 'stage': name})[:15], 16), 'response_format': output_format}
                request = explicit_preservation_request(request)
                return stage(name, request, self.providers['attacker'], validate)

            vectors = stage('reference_embeddings', {'texts': [original_text]+[row['text'] for row in references]},
                            lambda request: self.providers['embeddings'](request['texts']), lambda values: _vectors(values, len(references)+1))
            similarities = [(math.fsum(a*b for a,b in zip(vectors[0], vector)), row) for row,vector in zip(references, vectors[1:])]
            retrieved = sorted(similarities, key=lambda value: (-value[0], value[1]['item_id']))
            ranked = [{'item_id': row['item_id'], 'text': row['text'], 'cosine': score} for score,row in retrieved]
            neighborhood = stage('reference_neighborhood', {'ranked': ranked, 'count': config['reference_count'],
                'required_keywords': config['textrank']['candidate_count']}, lambda r: vocabulary_complete_references(
                    r['ranked'], r['count'], r['required_keywords'], self.providers['tokenizer']))
            selected = neighborhood['selected']
            tokenized = stage('reference_tokenization', {'texts': [row['text'] for row in selected]}, lambda request: [self.providers['tokenizer'](text) for text in request['texts']])
            tr = stage('textrank', {'tokens': tokenized, 'settings': config['textrank']},
                       lambda request: textrank_adjacent(request['tokens'], **{key: value for key,value in request['settings'].items() if key != 'candidate_count'}))
            candidates = [row['keyword'] for row in tr['scores'][:config['textrank']['candidate_count']]]
            if len(candidates) < config['keyword_count']:
                raise ValueError('Reference graph has fewer than the required 20 keyword candidates')

            def keyword_response(raw):
                values = json.loads(raw)
                if not isinstance(values, list) or len(values) != config['keyword_count'] or any(not isinstance(word, str) for word in values) or len(set(values)) != len(values) or not set(values) <= set(candidates):
                    raise ValueError('Keyword selector must return exactly 20 distinct TextRank candidates')
                return {'raw_response': raw, 'keywords': values}

            keywords = llm('keyword_selection', KEYWORD_SELECTION_INSTRUCTION,
                           {'ranked_keywords': candidates, 'count': config['keyword_count']}, keyword_response, 'json_array_of_keyword_strings')['keywords']
            personas = []
            for index in range(config['persona_count']):
                def persona_response(raw):
                    if not isinstance(raw, str) or not raw.strip() or len(raw) > config['maximum_output_characters'] or raw.strip() in {value.strip() for value in personas}:
                        raise ValueError('Persona must be nonempty, distinct and within the frozen output budget')
                    return raw
                personas.append(llm(['persona', index], 'Create one sales-expert persona with distinctive customer preferences, different from the existing personas. Return only that persona description.',
                                    {'existing_personas': list(personas)}, persona_response))
            target_records = [row for decision in views for row in decision['records'] if row['type'] == 'item_text' and row['fields'].get('item_id') == target_item_id]
            common = {'original_content': original_text, 'reference_content': selected[0]['text'], 'keywords': keywords,
                      'source_title': next((row['fields'].get('title', '') for decision in views for row in decision['records'] if row['id'] == record_id), ''),
                      'source_item_id': edited_item_id, 'promotion_target_item_id': target_item_id,
                      'target_observed_content': target_records[0]['fields'].get(field, '') if target_records else '',
                      'objective': 'Rephrase the source product to favor the fixed promotion target through factual relevance or comparison; retain the source product identity and facts, and do not embed instructions addressed to the recommender'}
            drafts = [llm(['draft', index], 'Act as the supplied sales persona. Rephrase the product using the reference style and selected attributes while obeying the declared preservation rule. Return only rewritten content.',
                          {**common, 'persona': persona, 'preservation_rule': self.provider_metadata['content_validator']['rule']}) for index,persona in enumerate(personas)]
            summary = llm('summary', 'Combine the independent rewritten descriptions into one coherent product description while preserving the required original content. Return only that description.',
                          {**common, 'drafts': drafts, 'preservation_rule': self.provider_metadata['content_validator']['rule']})
            used = 0

            def feedback(name, text):
                nonlocal used
                if used+len(config['feedback_decisions']) > config['feedback_query_budget']:
                    raise ValueError('Feedback request would exceed the frozen budget')
                decisions = changed(text)
                count = 0
                for index in config['feedback_decisions']:
                    result = stage(['feedback', name, index], {'decision': decisions[index], 'k': config['feedback_k']},
                                   lambda request: self.providers['victim'].rank(request['decision'], request['k']),
                                   lambda value, index=index: parse_ranking(json.dumps(value), decisions[index]['candidates'], config['feedback_k']))
                    used += 1
                    count += target_item_id in result
                return {'target_recommendations': count, 'query_count': len(config['feedback_decisions']), 'frequency': count/len(config['feedback_decisions'])}

            original_feedback = feedback('original_observed', original_text)
            last_feedback = feedback('summary', summary)
            last_feedback['gain_over_original_observed'] = last_feedback['frequency']-original_feedback['frequency']
            last_text = summary
            history = [{'stage': 'independent_drafts', 'personas': personas, 'drafts': drafts}, {'stage': 'summary', 'content': summary, 'feedback': last_feedback}]
            stopped = 'discussion_round_limit'
            for round_index in range(config['max_discussion_rounds']):
                if last_feedback['gain_over_original_observed'] >= config['minimum_frequency_gain']:
                    stopped = 'frequency_gain_criterion'
                    break
                if used+len(config['feedback_decisions']) > config['feedback_query_budget']:
                    stopped = 'feedback_query_budget'
                    break
                rng = random.Random(digest({'seed': config['seed'], 'round': round_index}))
                proposed = None
                for attempt in range(config['max_speaker_attempts']):
                    speaker = rng.randrange(config['persona_count'])
                    def discussion_response(raw):
                        if isinstance(raw, str) and raw.strip().lower() == 'no':
                            return raw
                        return validate_text(raw)
                    response = llm(['discussion', round_index, attempt], 'Review the shared revision history and observed recommendation feedback from your persona. Reply No to abstain; otherwise return a revised product description using the references and keywords. Preserve the specified original content.',
                                   {**common, 'persona': personas[speaker], 'history': copy.deepcopy(history), 'last_tested_content': last_text,
                                    'last_feedback': last_feedback, 'preservation_rule': self.provider_metadata['content_validator']['rule']}, discussion_response)
                    history.append({'stage': 'discussion', 'round': round_index, 'attempt': attempt, 'speaker': speaker, 'content': response})
                    if response.strip().lower() != 'no':
                        proposed = response
                        break
                if proposed is None:
                    stopped = 'speaker_attempt_budget'
                    break
                last_text = proposed
                last_feedback = feedback(['discussion', round_index], last_text)
                last_feedback['gain_over_original_observed'] = last_feedback['frequency']-original_feedback['frequency']
                history.append({'stage': 'feedback', 'round': round_index, 'content': last_text, 'feedback': last_feedback})
                if last_feedback['gain_over_original_observed'] >= config['minimum_frequency_gain']:
                    stopped = 'frequency_gain_criterion'
                    break
            output_observed = changed(last_text, observed_decisions)
            actual = [index for index in config['affected_decisions'] if output_observed[index] != observed_decisions[index]]
            generator = {'name': 'TextSimu-RecAtom', 'implementation_sha256': file_digest(__file__), 'settings_sha256': self.settings['sha256']}
            result = {'generator': generator, 'attack_family': 'deceptive_text_rewriting', 'target_item_id': target_item_id, 'edited_item_id': edited_item_id,
                'record_id': record_id, 'text_field': field, 'affected_decisions': config['affected_decisions'], 'actual_affected_decisions': actual,
                'observed': output_observed, 'original_text': original_text, 'rewritten_text': last_text,
                'trace': {'fingerprint': fingerprint, 'stages': trace, 'retrieved_references': selected, 'reference_selection': neighborhood, 'keywords': keywords, 'personas': personas,
                          'history': history, 'original_observed_feedback': original_feedback, 'last_feedback': last_feedback, 'stopping_reason': stopped},
                'feedback_queries_used': used, 'feedback_query_budget': config['feedback_query_budget'],
                'attempt_status': 'applied' if actual else 'no_change', 'include_unsuccessful_attempts': True,
                'generation_metadata': self.metadata, 'is_full_reproduction': False}
            path = directory/'result.json'
            if path.exists():
                if read_json(path) != result:
                    raise ValueError('Final TextSimu result differs on identical resume')
            else:
                write_json(path, result)
            return result
