import copy
import fcntl
import json
import math
import re
import time
from decimal import Decimal
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .rq1_attacks import bundle_rewriting_results, rewriting_subsettings, _rounded, _fraction
from .rq1_protocol import load_pair_artifact
from .rq1_sequence_export import _catalog, _once, _readonly, _source, _observed_file
from .textsimu_attack import TextSimuRecAtom


VERSION = 'rq1-independent-multirecord-rewriting-v1'


class RewritingDeadline(BaseException):
    pass


def _frozen(value):
    if not isinstance(value, dict) or set(value) != {'status', 'parameters', 'provenance', 'sha256'} or value['status'] != 'frozen' or not isinstance(value['provenance'], dict) or not value['provenance'] or digest({k: value[k] for k in ('parameters', 'provenance')}) != value['sha256']:
        raise ValueError('A valid frozen settings envelope is required')
    return value['parameters']


def _runtime(value):
    p = _frozen(value)
    if set(p) != {'ranker', 'attacker', 'embeddings', 'tokenizer', 'preservation', 'popular_references', 'time_budget_seconds'}:
        raise ValueError('Runtime settings have missing or unknown fields')
    rank = p['ranker']
    if set(rank) != {'revision', 'model_identity_sha256', 'dtype', 'attention_implementation', 'max_input_tokens', 'max_new_tokens'} or rank['dtype'] != 'bfloat16' or rank['attention_implementation'] != 'sdpa_last_token':
        raise ValueError('Declare the pinned shared native last-token SDPA ranker')
    for key, size in (('revision', 40), ('model_identity_sha256', 64)):
        if not isinstance(rank[key], str) or len(rank[key]) != size or any(c not in '0123456789abcdef' for c in rank[key]):
            raise ValueError('Ranker revision/model identity must be pinned')
    for key in ('max_input_tokens', 'max_new_tokens'):
        if type(rank[key]) is not int or rank[key] < 1:
            raise ValueError('Positive ranker token budgets are required')
    if set(p['attacker']) != {'do_sample', 'num_beams', 'temperature', 'top_p', 'top_k', 'max_input_tokens', 'max_new_tokens', 'use_cache'}:
        raise ValueError('Declare all raw attacker generation parameters')
    attacker = p['attacker']
    if type(attacker['do_sample']) is not bool or type(attacker['num_beams']) is not int or attacker['num_beams'] != 1 or attacker['use_cache'] is not True:
        raise ValueError('Raw attacker requires explicit sampling, one beam and caching')
    if any(type(attacker[k]) is not int or attacker[k] < minimum for k, minimum in (('max_input_tokens', 1), ('max_new_tokens', 1), ('top_k', 0))):
        raise ValueError('Invalid raw attacker token or top-k budget')
    if any(type(attacker[k]) not in (int, float) or not math.isfinite(attacker[k]) for k in ('temperature', 'top_p')) or attacker['temperature'] <= 0 or not 0 < attacker['top_p'] <= 1:
        raise ValueError('Invalid finite raw attacker temperature/top-p')
    if set(p['embeddings']) != {'pooling', 'max_input_tokens', 'truncation', 'normalize', 'add_special_tokens', 'batch_size'}:
        raise ValueError('Declare all Qwen embedding parameters')
    embedding = p['embeddings']
    if embedding['pooling'] != 'masked_mean' or embedding['truncation'] is not False or embedding['normalize'] is not True or type(embedding['add_special_tokens']) is not bool or type(embedding['batch_size']) is not int or embedding['batch_size'] != 1 or type(embedding['max_input_tokens']) is not int or embedding['max_input_tokens'] < 1:
        raise ValueError('Only explicit normalized, untruncated, batch-one masked-mean embeddings are supported')
    if set(p['tokenizer']) != {'kind', 'casefold'} or p['tokenizer']['kind'] not in ('whitespace', 'regex_word') or type(p['tokenizer']['casefold']) is not bool:
        raise ValueError('Declare the literal keyword tokenizer')
    if set(p['preservation']) != {'kind', 'casefold'} or p['preservation']['kind'] != 'retain_original_numeric_and_title_tokens' or type(p['preservation']['casefold']) is not bool:
        raise ValueError('Declare the mechanical source-token preservation rule')
    refs = p['popular_references']
    if set(refs) != {'source_split', 'count', 'text_chars', 'minimum_description_tokens', 'order'} or refs['source_split'] != 'train' or refs['order'] != 'count_desc_item_id_asc':
        raise ValueError('Popular references require explicitly ranked training-only evidence')
    if any(type(refs[k]) is not int or refs[k] < minimum for k, minimum in (('count', 52), ('text_chars', 1), ('minimum_description_tokens', 2))) or type(p['time_budget_seconds']) is not int or p['time_budget_seconds'] < 1:
        raise ValueError('Invalid corpus, text or cumulative time budget')
    return copy.deepcopy(p)


def _code():
    root = Path(__file__).parent
    names = ('rq1_rewriting_export.py', 'rq1_sequence_export.py', 'rq1_attacks.py', 'rq1_protocol.py', 'textsimu_attack.py', 'qwen_attack_provider.py', 'qwen_embeddings.py', 'rq1_last_token_inference.py', 'rq1_inference.py', 'scalable_inference.py', 'inference.py', 'records.py', 'artifacts.py', 'keyword_inference.py', 'constrained_inference.py')
    return {name: file_digest(root / name) for name in names}


def _pinned(path, expected):
    if file_digest(path) != expected:
        raise ValueError(f'Pinned input changed: {path}')
    return read_json(path)


class LiteralTokenizer:
    def __init__(self, settings):
        self.settings = copy.deepcopy(settings)
        self.metadata = {'implementation_sha256': file_digest(__file__), 'settings': self.settings, 'regex': r"\b[\w'-]+\b" if settings['kind'] == 'regex_word' else None}

    def __call__(self, text):
        words = text.split() if self.settings['kind'] == 'whitespace' else re.findall(r"\b[\w'-]+\b", text)
        return [word.casefold() for word in words] if self.settings['casefold'] else words


class SourceTokenPreserver:
    def __init__(self, settings, source_title):
        self.casefold = settings['casefold']
        self.title_tokens = self._tokens(source_title)
        self.metadata = {'implementation_sha256': file_digest(__file__), 'settings': copy.deepcopy(settings),
                         'rule': {'kind': settings['kind'], 'required_title_tokens': sorted(self.title_tokens), 'casefold': self.casefold,
                                  'numeric_tokens': 'Retain every original description word token containing a decimal digit',
                                  'limitation': 'Mechanical token retention only; not a semantic factuality guarantee'}}

    def _tokens(self, text):
        values = re.findall(r"\b[\w'-]+\b", text)
        return {value.casefold() if self.casefold else value for value in values}

    def __call__(self, original, rewritten):
        required = self.title_tokens | {word for word in self._tokens(original) if any(c.isdecimal() for c in word)}
        return required <= self._tokens(rewritten)


class _ConfigurationOnlyProvider:
    def __init__(self, metadata):
        self.metadata = {'purpose': 'configuration_validation_only_no_model_calls', **metadata}

    def __call__(self, *args, **kwargs):
        raise RuntimeError('Configuration validation must never invoke a provider')

    def rank(self, *args, **kwargs):
        raise RuntimeError('Configuration validation must never invoke a ranker')


def prepare_rewriting(sequence_manifest_path, database_path, manifest_path, output_path, attack_settings, runtime_settings):
    runtime = _runtime(runtime_settings)
    rewriting_subsettings(attack_settings, 0, 'validation')
    for flag in ('qwen_attacker_user_approved', 'qwen_reference_encoder_user_approved', 'multi_record_same_target_user_approved'):
        if attack_settings['provenance'].get(flag) is not True:
            raise ValueError('All material Qwen/multiple-record substitutions require explicit approval provenance')
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sequence_path = Path(sequence_manifest_path).resolve()
        exported = read_json(sequence_path)
        if exported.get('kind') != 'rq1_sequence_export' or exported.get('status') not in ('complete', 'partially_blocked', 'blocked'):
            raise ValueError('A completed chronological reference export is required')
        planned = _pinned(exported['planned_sequences_path'], exported['planned_sequences_sha256'])
        if len(planned) != exported['planned_sequences']:
            raise ValueError('Planned sequence count differs')
        database, manifest_file = Path(database_path).resolve(), Path(manifest_path).resolve()
        if output == database.parent or database.parent in output.parents:
            raise ValueError('Write rewriting artifacts outside immutable prepared data')
        source_manifest, source, stat = _source(database, manifest_file, exported['dataset'])
        if any(source[k] != exported['source'][k] for k in source):
            raise ValueError('Rewriting corpus and sequences have different prepared source pins')
        refs = runtime['popular_references']
        projection = exported['frozen_sequence_settings']['parameters']['text_chars']
        if refs['text_chars'] != projection:
            raise ValueError('Reference text projection must match the explicitly frozen sequence projection')
        with _readonly(database) as connection:
            items = _catalog(connection, refs['text_chars'])
            counts = dict(connection.execute('SELECT item_id,COUNT(*) FROM events WHERE ts<? GROUP BY item_id', (source_manifest['boundaries_unix'][0],)))
        eligible = sorted((item for item in items if counts.get(item, 0) > 0 and len(items[item]['description'].split()) >= refs['minimum_description_tokens']), key=lambda item: (-counts[item], item))
        if len(eligible) < refs['count']:
            raise ValueError('Insufficient real training-interacted text items for the requested reference corpus; no padding or silent reduction')
        corpus = {'status': 'frozen', 'source_split': 'train', 'popularity_selection': {**refs, 'source_database_sha256': source['database_sha256'],
                  'source_manifest_sha256': source['manifest_sha256'], 'boundary_unix_exclusive': source_manifest['boundaries_unix'][0],
                  'count_definition': 'All source events, not evaluator outcomes', 'eligible_items': len(eligible),
                  'selected_counts': [[item, counts[item]] for item in eligible[:refs['count']]]},
                  'items': [{'item_id': item, 'text': items[item]['description']} for item in eligible[:refs['count']]]}
        corpus['sha256'] = digest(corpus)
        validation_metadata = {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'revision': runtime['ranker']['revision'], 'implementation_sha256': file_digest(__file__)}
        TextSimuRecAtom(rewriting_subsettings(attack_settings, 0, 'validation'), corpus,
                       _ConfigurationOnlyProvider({**validation_metadata, 'embedding_parameters': runtime['embeddings']}),
                       LiteralTokenizer(runtime['tokenizer']),
                       _ConfigurationOnlyProvider({**validation_metadata, 'generation_parameters': runtime['attacker']}),
                       _ConfigurationOnlyProvider(validation_metadata), SourceTokenPreserver(runtime['preservation'], ''))
        _once(output / 'popular-references.json', corpus)
        attempts = []
        multi = attack_settings['parameters']['multi_record']
        for row in planned:
            reference = _pinned(row['reference_path'], row['reference_sha256'])
            schedule = attack_settings['parameters']['affected_decisions']
            feedback = attack_settings['parameters']['feedback_decisions']
            if max(schedule + feedback) >= len(reference['clean']):
                raise ValueError('Attack or feedback schedule exceeds the frozen source sequence')
            if any(_rounded(Decimal(len(reference['clean'][i]['records'])) * _fraction(multi['nominal_intensity'], 'nominal_intensity'), multi['intensity_rounding']) != multi['record_count'] for i in schedule):
                raise ValueError('Declared editable-record count differs from rounded nominal intensity before generation')
            if any(attack_settings['parameters']['feedback_k'] > len(d['candidates']) for d in reference['clean']):
                raise ValueError('Feedback ranking length exceeds a frozen candidate set')
            selected = [reference['evaluation']['target'], *reference['protocol']['shared_distractor_ids'][:multi['record_count'] - 1]]
            if len(selected) != multi['record_count'] or len(set(selected)) != len(selected):
                raise ValueError('Every source sequence needs all prespecified existing shared editable records')
            records = []
            for item in selected:
                record_id = 'item/' + item
                occurrences = [r for d in reference['clean'] for r in d['records'] if r['id'] == record_id]
                if len(occurrences) != len(reference['clean']) or any(r['type'] != 'item_text' or r['fields']['item_id'] != item or len(r['fields'][attack_settings['parameters']['text_field']].split()) < 2 for r in occurrences):
                    raise ValueError('Selected editable identity lacks source text or full fixed-sequence presence')
                if len({digest(r) for r in occurrences}) != 1:
                    raise ValueError('Selected original record text/reference changes across its appearances')
                settings = rewriting_subsettings(attack_settings, reference['seed'], record_id)
                records.append({'record_id': record_id, 'edited_item_id': item, 'original_record_sha256': digest(occurrences[0]), 'settings_sha256': settings['sha256']})
            attempts.append({**row, 'records': records})
        if database.stat().st_size != stat.st_size or database.stat().st_mtime_ns != stat.st_mtime_ns or file_digest(manifest_file) != source['manifest_sha256']:
            raise ValueError('Prepared source changed during rewriting preparation')
        body = {'schema_version': 1, 'kind': VERSION, 'status': 'ready_to_generate', 'dataset': exported['dataset'], 'source': exported['source'],
                'sequence_manifest_path': str(sequence_path), 'sequence_manifest_sha256': file_digest(sequence_path),
                'attack_settings': attack_settings, 'runtime_settings': runtime_settings, 'planned_attempts': attempts,
                'popular_references_path': str(output / 'popular-references.json'), 'popular_references_sha256': file_digest(output / 'popular-references.json'),
                'code': _code(), 'failure_policy': 'Retain every planned attempt; any failed or uncertain construction holds publication of this entire family; valid no-change/ineffective attacks remain included',
                'is_full_reproduction': False}
        result = {**body, 'sha256': digest(body)}
        _once(output / 'prepared.json', result)
        return {'status': 'ready_to_generate', 'prepared_path': str(output / 'prepared.json'), 'prepared_sha256': file_digest(output / 'prepared.json'), 'planned_attempts': len(attempts)}


def run_rewriting(prepared_path, model_path, cache_path, device, invocation_budget_seconds, *, prepared_sha256, ranker_factory=None, embedding_factory=None, attacker_factory=None, clock=time.monotonic):
    entry_started = clock()
    from .qwen_attack_provider import QwenAttackProvider
    from .qwen_embeddings import QwenEmbeddings
    from .rq1_last_token_inference import LastTokenSDPARanker
    prepared_path = Path(prepared_path).resolve()
    prepared = _pinned(prepared_path, prepared_sha256)
    if prepared.get('sha256') != digest({k: v for k, v in prepared.items() if k != 'sha256'}) or prepared['code'] != _code():
        raise ValueError('Prepared rewriting settings or implementation fingerprint changed')
    runtime = _runtime(prepared['runtime_settings'])
    if isinstance(invocation_budget_seconds, bool) or not isinstance(invocation_budget_seconds, (int, float)) or not 0 < invocation_budget_seconds <= runtime['time_budget_seconds']:
        raise ValueError('Invocation budget must fit the frozen cumulative limit')
    if file_digest(Path(model_path) / 'model_identity.json') != runtime['ranker']['model_identity_sha256']:
        raise ValueError('Model snapshot identity differs from frozen runtime')
    _pinned(prepared['sequence_manifest_path'], prepared['sequence_manifest_sha256'])
    references = _pinned(prepared['popular_references_path'], prepared['popular_references_sha256'])
    for row in prepared['planned_attempts']:
        _pinned(row['reference_path'], row['reference_sha256'])
    output = prepared_path.parent
    immutable = {'prepared_sha256': prepared_sha256, 'model_path': str(Path(model_path).resolve()), 'cache_path': str(Path(cache_path).resolve()), 'device': device}
    fingerprint = digest(immutable)
    with (output / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'status.json'
        state = read_json(path) if path.exists() else {'status': 'ready', 'fingerprint': fingerprint, 'immutable': immutable, 'cumulative_seconds': 0.0, 'invocations': [], 'attempts': {row['key']: {'status': 'pending', 'completed_records': []} for row in prepared['planned_attempts']}}
        if path.exists() and (state.get('sha256') != digest({k: v for k, v in state.items() if k != 'sha256'}) or state['fingerprint'] != fingerprint):
            raise ValueError('Generation resume status, runtime or input pins changed')

        def save():
            state['sha256'] = digest({k: v for k, v in state.items() if k != 'sha256'})
            write_json(path, state)

        if state['status'] == 'complete':
            receipt = read_json(output / 'receipt.json')
            if receipt['sha256'] != digest({k: v for k, v in receipt.items() if k != 'sha256'}):
                raise ValueError('Completed generation receipt changed')
            for descriptor in _pinned(receipt['descriptors_path'], receipt['descriptors_sha256']):
                load_pair_artifact(descriptor)
            _pinned(receipt['observed_sequences_path'], receipt['observed_sequences_sha256'])
            return receipt
        if state['status'] == 'failed':
            raise RuntimeError('Preserved failed/uncertain construction requires review; no automatic retry')
        if state['invocations'] and state['invocations'][-1]['status'] == 'running':
            old = state['invocations'][-1]
            old['status'] = 'interrupted'
            state['cumulative_seconds'] += old['reserved_seconds']
        for intent in (output / 'generation').rglob('*.intent.json') if (output / 'generation').exists() else []:
            result = intent.with_name(intent.name.replace('.intent.json', '.result.json'))
            if not result.exists() or read_json(result).get('status') != 'complete':
                state.update(status='failed', error={'kind': 'uncertain_or_failed_stage', 'intent_path': str(intent)})
                save()
                raise RuntimeError('A failed or uncertain provider stage is preserved; no duplicate call or clean-input fallback')
        allowance = min(invocation_budget_seconds, runtime['time_budget_seconds'] - state['cumulative_seconds'])
        if allowance <= 0:
            state['status'] = 'budget_exhausted'
            save()
            return {'status': 'budget_exhausted', 'cumulative_seconds': state['cumulative_seconds']}
        started = entry_started
        invocation = {'status': 'running', 'reserved_seconds': allowance, 'started_unix': time.time()}
        state['invocations'].append(invocation)
        state['status'] = 'running'
        save()

        def check():
            if clock() - started >= allowance:
                raise RewritingDeadline()

        ranker = None
        try:
            check()
            rank = runtime['ranker']
            ranker = (ranker_factory or LastTokenSDPARanker)(str(model_path), rank['revision'], str(cache_path), device, rank['dtype'], rank['max_input_tokens'], rank['max_new_tokens'])
            ranker.load()
            check()
            if 'loaded_inference' in state and state['loaded_inference'] != ranker.metadata:
                raise ValueError('Loaded inference identity differs across generation resumes')
            state['loaded_inference'] = copy.deepcopy(ranker.metadata)
            embeddings = (embedding_factory or QwenEmbeddings)(ranker, embedding_parameters=runtime['embeddings'])
            attacker = (attacker_factory or QwenAttackProvider)(ranker, generation_parameters=runtime['attacker'])
            tokenizer = LiteralTokenizer(runtime['tokenizer'])
            for row in prepared['planned_attempts']:
                entry = state['attempts'][row['key']]
                if entry['status'] == 'complete':
                    continue
                check()
                reference = _pinned(row['reference_path'], row['reference_sha256'])
                results = []
                for record in row['records']:
                    directory = output / 'generation' / row['key'] / digest(record['record_id'])
                    settings = rewriting_subsettings(prepared['attack_settings'], reference['seed'], record['record_id'])
                    if settings['sha256'] != record['settings_sha256']:
                        raise ValueError('Per-record seed/settings derivation changed')
                    original = next(r for r in reference['clean'][0]['records'] if r['id'] == record['record_id'])
                    if digest(original) != record['original_record_sha256']:
                        raise ValueError('Original editable record changed')
                    validator = SourceTokenPreserver(runtime['preservation'], original['fields']['title'])
                    generator = TextSimuRecAtom(settings, references, embeddings, tokenizer, attacker, ranker, validator)
                    entry.update(status='running', active_record=record['record_id'])
                    save()
                    result = generator.run(reference['clean'], reference['evaluation']['target'], record['record_id'], original['fields'][settings['parameters']['text_field']], directory, check=check, edited_item_id=record['edited_item_id'])
                    results.append(result)
                    if record['record_id'] not in entry['completed_records']:
                        entry['completed_records'].append(record['record_id'])
                    save()
                bundle = bundle_rewriting_results(reference, results, prepared['attack_settings'], dataset=prepared['dataset'])
                descriptor = {'id': prepared['dataset'] + '/deceptive_text_rewriting/' + row['key'], 'dataset': prepared['dataset'], 'attack_family': 'deceptive_text_rewriting',
                              'split': row['split'], 'seed_group': row['seed_group'], 'sequence_seed': row['sequence_seed'], 'source_sequence_id': row['source_sequence_id'], 'source': prepared['source']}
                for name in ('pair', 'reference', 'audit'):
                    destination = output / 'pairs' / row['key'] / (name + '.json')
                    _once(destination, bundle[name])
                    descriptor[name + '_path'], descriptor[name + '_sha256'] = str(destination), file_digest(destination)
                load_pair_artifact(descriptor)
                entry.update(status='complete', active_record=None, descriptor=descriptor, attempt_status=bundle['audit']['attempt_status'])
                save()
            descriptors = [state['attempts'][row['key']]['descriptor'] for row in prepared['planned_attempts']]
            _once(output / 'descriptors.json', descriptors)
            _observed_file(output / 'observed-sequences.json', {'schema_version': 1, 'kind': 'rq1_observed_sequence_views', 'dataset': prepared['dataset'], 'source_manifest_sha256': prepared['source']['manifest_sha256'], 'sequence_export_sha256': prepared_sha256}, descriptors)
            body = {'status': 'complete', 'fingerprint': fingerprint, 'prepared_sha256': prepared_sha256, 'planned_attempts': len(descriptors),
                    'descriptors_path': str(output / 'descriptors.json'), 'descriptors_sha256': file_digest(output / 'descriptors.json'),
                    'observed_sequences_path': str(output / 'observed-sequences.json'), 'observed_sequences_sha256': file_digest(output / 'observed-sequences.json'),
                    'all_planned_attempts_included': True, 'is_full_reproduction': False}
            receipt = {**body, 'sha256': digest(body)}
            _once(output / 'receipt.json', receipt)
            state['status'] = 'complete'
            invocation['status'] = 'complete'
            return receipt
        except RewritingDeadline:
            state['status'] = 'partial'
            invocation['status'] = 'partial'
            return {'status': 'partial', 'fingerprint': fingerprint}
        except BaseException as error:
            state.update(status='failed', error={'type': type(error).__name__, 'message': str(error), 'policy': 'No skipped attempt, automatic retry, ranking repair or clean-input fallback'})
            for entry in state['attempts'].values():
                if entry['status'] == 'running':
                    entry.update(status='failed', error=copy.deepcopy(state['error']))
            invocation['status'] = 'failed'
            raise
        finally:
            elapsed = clock() - started
            invocation['elapsed_seconds'] = elapsed
            state['cumulative_seconds'] += elapsed
            if ranker is not None:
                state['last_ranker_statistics'] = ranker.statistics()
                ranker.cache.connection.close()
            save()
