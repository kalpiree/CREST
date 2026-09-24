import bisect
import copy
import fcntl
import json
import os
import random
import sqlite3
import tempfile
from collections import Counter, deque
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .records import validate_decision
from .rq1_attacks import freeze_attack_config, inject_instructions, insert_history
from .rq1_protocol import load_pair_artifact, provider_decisions, validate_attack_pair


VERSION = 'rq1-chronological-source-sequences-v1'
SPLITS = ('train', 'development', 'calibration', 'test')
FAMILIES = ('instruction_injection', 'interaction_history_manipulation', 'deceptive_text_rewriting')
BOTTOM90 = 'uniform_bottom90_train_interacted_text_eligible_excluding_sequence_positives'
PARAMETERS = {'schema_version', 'dataset', 'T', 'K', 'candidate_count', 'history_count', 'text_chars',
              'shared_distractors', 'repeated_user_pairs', 'window_selection', 'same_timestamp_positive',
              'target_selection', 'candidate_catalog', 'partition_policy', 'allocations'}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _settings(frozen):
    if not isinstance(frozen, dict) or set(frozen) != {'status', 'parameters', 'provenance', 'sha256'} or frozen['status'] != 'frozen':
        raise ValueError('A frozen sequence parameter/provenance envelope is required')
    if not isinstance(frozen['provenance'], dict) or not frozen['provenance'] or digest({k: frozen[k] for k in ('parameters', 'provenance')}) != frozen['sha256']:
        raise ValueError('Sequence settings fingerprint or provenance differs')
    p = frozen['parameters']
    if not isinstance(p, dict) or set(p) not in (PARAMETERS, PARAMETERS | {'structural_admissibility'}) or p['schema_version'] != 1 or isinstance(p['schema_version'], bool):
        raise ValueError('Sequence parameters have missing or unknown fields')
    if 'structural_admissibility' in p:
        rule = p['structural_admissibility']
        fixed = {'policy': 'bounded_history_construction_before_model_feedback',
                 'seed_policy': 'sha256_sequence_seed_draw_index', 'retain_rejected': 'source_recipe_and_reason'}
        if not isinstance(rule, dict) or set(rule) != {*fixed, 'max_draws_per_sequence'} or any(rule[k] != value for k, value in fixed.items()):
            raise ValueError('Resolve the explicit structural admissibility and candidate-retention policies')
        _integer(rule['max_draws_per_sequence'], 'max_draws_per_sequence', 1)
    if p['dataset'] not in ('amazon-2018-all-beauty', 'steam', 'movielens-1m'):
        raise ValueError('RQ1 supports the explicit Amazon All_Beauty and Steam editions')
    for name in ('T', 'K', 'candidate_count', 'history_count'):
        _integer(p[name], name, 1)
    for name in ('text_chars', 'shared_distractors', 'repeated_user_pairs'):
        _integer(p[name], name)
    if p['K'] > p['candidate_count'] or p['candidate_count'] < p['shared_distractors'] + 2 or 2 * p['repeated_user_pairs'] > p['T']:
        raise ValueError('Inconsistent candidate, rank or repeated-user counts')
    required = {'window_selection': 'uniform_distinct_user_cutoffs', 'same_timestamp_positive': 'lowest_event_id',
                'partition_policy': 'prepared_global_timestamp_cutoffs'}
    if any(p[k] != value for k, value in required.items()) or p['candidate_catalog'] not in ('full_static_metadata', 'text_eligible_static_metadata') or p['target_selection'] not in ('uniform_catalog_excluding_sequence_positives', BOTTOM90):
        raise ValueError('Unsupported explicit sequence sampling policy')
    if not isinstance(p['allocations'], list) or not p['allocations']:
        raise ValueError('Explicit sequence allocations are required')
    seeds, groups, roles = set(), set(), set()
    for row in p['allocations']:
        if not isinstance(row, dict) or set(row) != {'split', 'seed_group', 'seed_start', 'count'} or row['split'] not in ('calibration', 'test'):
            raise ValueError('Allocation requires calibration/test split, group, seed_start and count')
        for key in ('seed_group', 'seed_start', 'count'):
            _integer(row[key], key, 1 if key == 'count' else 0)
        group = (row['split'], row['seed_group'])
        if group in groups:
            raise ValueError('Duplicate split/seed-group allocation')
        groups.add(group)
        roles.add(row['split'])
        for seed in range(row['seed_start'], row['seed_start'] + row['count']):
            if seed in seeds:
                raise ValueError('Sequence seeds must be distinct across calibration and test allocations')
            seeds.add(seed)
    if roles != {'calibration', 'test'}:
        raise ValueError('Both calibration and test allocations are required')
    return copy.deepcopy(p)


def _once(path, payload):
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f'Output differs from immutable export: {path}')
    else:
        write_json(path, payload)


def _observed_file(path, header, descriptors):
    fd, temporary = tempfile.mkstemp(prefix='.observed-sequences.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(header, ensure_ascii=False)[:-1] + ', "sequences":[')
            for number, row in enumerate(descriptors):
                if number:
                    stream.write(',')
                view = {k: row[k] for k in ('id', 'split', 'seed_group', 'sequence_seed')}
                view['decisions'] = provider_decisions(read_json(row['pair_path']))
                json.dump(view, stream, ensure_ascii=False, allow_nan=False)
            stream.write(']}\n')
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if file_digest(path) != file_digest(temporary):
                raise ValueError('Observed provider views differ from immutable export')
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Prepared timestamps require a timezone')
    return parsed.timestamp()


def _readonly(path):
    return closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True))


def _source(database, manifest_path, dataset):
    manifest_hash = file_digest(manifest_path)
    manifest = read_json(manifest_path)
    expected = manifest.get('artifacts', {}).get('records.sqlite', {})
    if manifest.get('status') != 'complete' or manifest.get('dataset') != dataset or not expected.get('sha256'):
        raise ValueError('A complete matching prepared dataset manifest is required')
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Prepared database has uncheckpointed WAL data')
    database_hash = file_digest(database)
    if database_hash != expected['sha256'] or database.stat().st_size != expected.get('bytes'):
        raise ValueError('Prepared database checksum or size differs')
    boundaries = manifest.get('boundaries_unix')
    if not isinstance(boundaries, list) or len(boundaries) != 3 or any(isinstance(x, bool) or not isinstance(x, int) for x in boundaries) or boundaries != sorted(set(boundaries)):
        raise ValueError('Prepared split boundaries must be strictly increasing integer timestamps')
    return manifest, {'database_sha256': database_hash, 'manifest_sha256': manifest_hash}, database.stat()


def _catalog(connection, text_chars):
    result = {}
    for identity, original, payload in connection.execute('SELECT item_id,original_id,payload FROM items ORDER BY item_id'):
        item = json.loads(payload)
        if item.get('id') != identity or any(not isinstance(item.get(k), str) for k in ('title', 'description')):
            raise ValueError('Invalid canonical item metadata')
        result[identity] = {'id': identity, 'original_id': original, **{k: item[k][:text_chars] if text_chars else item[k] for k in ('title', 'description')}}
    return result


def _window_index(connection, output, immutable, p, manifest, items, progress):
    path = output / 'windows.sqlite'
    record = output / 'window-pool.json'
    key = digest(immutable)
    if record.exists():
        pool = read_json(record)
        if pool.get('source_fingerprint') != key or not path.exists() or file_digest(path) != pool.get('window_index_sha256'):
            raise ValueError('Window index is missing, changed or belongs to another source/configuration')
        return pool
    if path.exists():
        raise ValueError('Unmanifested completed window index; preserve it and use a new output directory')
    partial = output / 'windows.partial.sqlite'
    if partial.exists():
        with sqlite3.connect(partial) as old:
            value = old.execute('SELECT value FROM state WHERE key=?', ('source_fingerprint',)).fetchone()
            if value is None or json.loads(value[0]) != key:
                raise ValueError('Partial index belongs to different inputs')
        partial.unlink()
    index = sqlite3.connect(partial)
    counts, events, eligible = Counter(), Counter(), Counter()
    try:
        index.execute('CREATE TABLE state(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        index.execute('INSERT INTO state VALUES (?,?)', ('source_fingerprint', json.dumps(key)))
        index.execute('CREATE TABLE windows(split TEXT NOT NULL,ordinal INTEGER NOT NULL,user_id TEXT NOT NULL,cutoff INTEGER NOT NULL,event_id TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(split,ordinal),UNIQUE(user_id,cutoff))')
        index.commit()
        history, simultaneous = deque(maxlen=p['history_count']), deque(maxlen=p['history_count'])
        current_user = current_time = chosen_cutoff = None
        scanned = 0
        for identity, user, item, timestamp, positive, payload in connection.execute('SELECT event_id,user_id,item_id,ts,positive,payload FROM events ORDER BY user_id,ts,event_id'):
            scanned += 1
            if user != current_user:
                current_user, current_time, chosen_cutoff = user, timestamp, None
                history.clear()
                simultaneous.clear()
            elif timestamp != current_time:
                history.extend(simultaneous)
                simultaneous.clear()
                current_time, chosen_cutoff = timestamp, None
            event = json.loads(payload)
            if event.get('id') != identity or event.get('user_id') != user or event.get('item_id') != item or event.get('unix_timestamp') != timestamp or bool(event.get('positive')) != bool(positive) or _timestamp(event['timestamp']) != timestamp or item not in items:
                raise ValueError('Prepared event columns disagree with canonical payload')
            split = SPLITS[bisect.bisect_right(manifest['boundaries_unix'], timestamp)]
            events[split] += 1
            text_eligible = p['candidate_catalog'] != 'text_eligible_static_metadata' or len(items[item]['description'].split()) >= 2
            if positive and text_eligible and chosen_cutoff is None and len(history) >= p['history_count']:
                chosen_cutoff = identity
                eligible[split] += 1
                if split in ('calibration', 'test'):
                    window = {'user_id': user, 'positive': item, 'positive_event_id': identity, 'cutoff': event['timestamp'], 'cutoff_unix': timestamp, 'history': list(history), 'split': split}
                    index.execute('INSERT INTO windows VALUES (?,?,?,?,?,?)', (split, counts[split], user, timestamp, identity, json.dumps(window, ensure_ascii=False, separators=(',', ':'))))
                    counts[split] += 1
            simultaneous.append({'id': identity, 'item_id': item, 'timestamp': event['timestamp'], 'unix_timestamp': timestamp})
            if scanned % 100000 == 0:
                index.commit()
                progress({'stage': 'indexing_windows', 'events_scanned': scanned, 'eligible_windows': dict(eligible)})
        index.execute('CREATE INDEX windows_by_user ON windows(split,user_id,cutoff)')
        index.commit()
        users = {split: index.execute('SELECT COUNT(DISTINCT user_id) FROM windows WHERE split=?', (split,)).fetchone()[0] for split in ('calibration', 'test')}
        repeated = {split: index.execute('SELECT COUNT(*) FROM (SELECT user_id FROM windows WHERE split=? GROUP BY user_id HAVING COUNT(*)>=2)', (split,)).fetchone()[0] for split in ('calibration', 'test')}
    finally:
        index.close()
    partial.replace(path)
    pool = {'schema_version': 1, 'kind': 'rq1_chronological_window_index', 'dataset': p['dataset'], 'source_fingerprint': key,
            'window_index_path': str(path), 'window_index_sha256': file_digest(path), 'source_manifest_sha256': immutable['source']['manifest_sha256'],
            'source_database_sha256': immutable['source']['database_sha256'], 'boundaries_unix': manifest['boundaries_unix'],
            'window_counts': dict(counts), 'eligible_windows_all_splits': dict(eligible), 'source_event_counts': dict(events),
            'eligible_users': users, 'users_with_multiple_cutoffs': repeated, 'is_full_reproduction': False,
            'history_policy': 'Most recent real events strictly before each cutoff; earlier-split past events may remain historical context; no same-timestamp events enter history',
            'evaluation_partition': 'Calibration/test positive windows have disjoint prepared timestamp intervals; train/development positives are not evaluation cutoffs',
            'coverage_caveat': 'Chronological calibration/test partitions may differ in distribution; exchangeability and paper coverage are not asserted',
            'candidate_catalog_policy': p['candidate_catalog'],
            'candidate_catalog_items': sum(p['candidate_catalog'] != 'text_eligible_static_metadata' or len(item['description'].split()) >= 2 for item in items.values()),
            'source_catalog_items': len(items),
            'metadata_policy': 'Static source catalog; title/description literal prefixes only if explicit text_chars>0; historical availability is not established',
            'candidate_population_conditioning': 'Candidate items and source positive windows require at least two literal whitespace tokens in the projected description; real prior interactions are retained independently of text eligibility' if p['candidate_catalog'] == 'text_eligible_static_metadata' else 'All source catalog items remain eligible candidates'}
    _once(record, pool)
    return pool


def _reference(index, pool, items, p, frozen, split, seed, *, draw_seed=None):
    count = pool['window_counts'].get(split, 0)
    if count < p['T']:
        raise ValueError(f'Only {count} distinct eligible {split} user/cutoff windows for T={p["T"]}')
    rng = random.Random(seed if draw_seed is None else draw_seed)
    selected = {}
    if p['repeated_user_pairs']:
        users = [r[0] for r in index.execute('SELECT user_id FROM windows WHERE split=? GROUP BY user_id HAVING COUNT(*)>=2 ORDER BY user_id', (split,))]
        if len(users) < p['repeated_user_pairs']:
            raise ValueError('Insufficient users with two distinct cutoffs for the explicit anchor count')
        for user in rng.sample(users, p['repeated_user_pairs']):
            ordinals = [r[0] for r in index.execute('SELECT ordinal FROM windows WHERE split=? AND user_id=? ORDER BY cutoff', (split, user))]
            for ordinal in rng.sample(ordinals, 2):
                selected[ordinal] = None
    available = count - len(selected)
    needed = p['T'] - len(selected)
    draws = rng.sample(range(available), needed)
    blocked = sorted(selected)
    for value in draws:
        ordinal = value
        for excluded in blocked:
            if excluded > ordinal:
                break
            ordinal += 1
        selected[ordinal] = None
    windows = [json.loads(index.execute('SELECT payload FROM windows WHERE split=? AND ordinal=?', (split, ordinal)).fetchone()[0]) for ordinal in selected]
    windows.sort(key=lambda w: (w['cutoff_unix'], w['user_id'], w['positive_event_id']))
    positives = {w['positive'] for w in windows}
    catalog_ids = sorted(item for item in items if p['candidate_catalog'] != 'text_eligible_static_metadata' or len(items[item]['description'].split()) >= 2)
    catalog_pool = set(catalog_ids)
    target_policy = {'name': p['target_selection']}
    if p['target_selection'] == BOTTOM90:
        eligible = sorted((item for item in catalog_ids if items[item].get('train_count', 0) >= 1 and len(items[item]['description'].split()) >= 2), key=lambda item: (items[item]['train_count'], item))
        excluded = (len(eligible) + 9) // 10
        target_pool = eligible[:-excluded] if excluded else []
        target_policy.update(popularity_split='train', popularity_count='all real source events strictly before first prepared boundary',
                             eligible_population='train-interacted catalog items with at least two whitespace tokens in projected description',
                             cold_start='excluded', order='ascending (training_event_count, canonical_item_id)', exclusion='last ceil(0.10 * eligible_count) items',
                             eligible_items=len(eligible), excluded_top_items=excluded, target_pool_sha256=digest(target_pool),
                             shared_negative_eligibility='at least two whitespace tokens in projected description; exclude target and all sequence positives')
        shared_pool = {item for item in catalog_ids if len(items[item]['description'].split()) >= 2} - positives
    else:
        target_pool = catalog_ids
        shared_pool = catalog_pool - positives
    available_items = sorted(set(target_pool) - positives)
    if not available_items or len(shared_pool) < p['shared_distractors'] + 1 or len(catalog_ids) < p['candidate_count']:
        raise ValueError('Insufficient catalog items for the frozen target and shared candidates')
    target = rng.choice(available_items)
    shared = rng.sample(sorted(shared_pool - {target}), p['shared_distractors'])
    clean = []
    for window in windows:
        fixed = [window['positive'], target, *shared]
        candidates, included = list(fixed), set(fixed)
        while len(candidates) < p['candidate_count']:
            item = rng.choice(catalog_ids)
            if item not in included:
                candidates.append(item)
                included.add(item)
        rng.shuffle(candidates)
        records = [{'id': e['id'], 'type': 'interaction', 'fields': {'user_id': window['user_id'], 'item_id': e['item_id'], 'title': items[e['item_id']]['title'], 'timestamp': e['timestamp'], 'event': 'review'}} for e in window['history']]
        records.extend({'id': 'item/' + item, 'type': 'item_text', 'fields': {'item_id': item, 'title': items[item]['title'], 'description': items[item]['description']}} for item in candidates)
        decision = {'user_id': window['user_id'], 'candidates': candidates, 'records': records, 'cutoff': window['cutoff'], 'cutoff_unix': window['cutoff_unix']}
        validate_decision(decision)
        clean.append(decision)
    window_keys = [(w['user_id'], w['cutoff_unix'], w['positive_event_id']) for w in windows]
    source_id = p['dataset'] + '/' + split + '/' + digest(window_keys)
    reference = {'seed': seed, 'clean': clean, 'evaluation': {'positives': [w['positive'] for w in windows], 'target': target},
                 'protocol': {'name': VERSION, 'sequence_settings_sha256': frozen['sha256'], 'source_split': split,
                              'source_sequence_id': source_id, 'positive_event_ids': [w['positive_event_id'] for w in windows],
                              'target_selection': target_policy,
                              'window_keys_sha256': digest(window_keys), 'shared_distractor_ids': shared,
                              'requested_repeated_user_pairs': p['repeated_user_pairs'], 'distinct_users': len({w['user_id'] for w in windows}),
                              'distinct_user_cutoffs': len(window_keys), 'selection_frozen_before_model_feedback': True,
                              'sampling_across_sequences': 'Independent seeded draws from fixed split pool; windows may recur across sequences, never twice within a sequence',
                              'metadata_policy': pool['metadata_policy'], 'coverage_caveat': pool['coverage_caveat'],
                              'candidate_population_conditioning': pool['candidate_population_conditioning']}}
    return reference


def _admissible_reference(index, pool, items, p, frozen, split, seed, attack_settings, output, key, sources, progress):
    rule = p.get('structural_admissibility')
    if rule is None:
        reference = _reference(index, pool, items, p, frozen, split, seed)
        if reference['protocol']['source_sequence_id'] in sources:
            raise ValueError('Independent seeds selected an identical source sequence; no outcome-based resampling is allowed')
        return reference
    history = attack_settings.get('interaction_history_manipulation')
    if history is None:
        raise ValueError('Structural admissibility requires the exact frozen history-attack settings')
    directory = output / 'source-selection-ledger' / key
    for draw in range(rule['max_draws_per_sequence']):
        draw_seed = int(digest({'sequence_seed': seed, 'draw_index': draw})[:15], 16)
        reference = _reference(index, pool, items, p, frozen, split, seed, draw_seed=draw_seed)
        row = {'sequence_seed': seed, 'draw_index': draw, 'draw_seed': draw_seed, 'source_split': split,
               'source_sequence_id': reference['protocol']['source_sequence_id'],
               'source_windows': [{'user_id': decision['user_id'], 'cutoff_unix': decision['cutoff_unix'], 'positive_event_id': event}
                                  for decision, event in zip(reference['clean'], reference['protocol']['positive_event_ids'])],
               'target': reference['evaluation']['target'], 'shared_distractor_ids': reference['protocol']['shared_distractor_ids'],
               'candidate_reference_sha256': digest(reference), 'sequence_settings_sha256': frozen['sha256'],
               'history_settings_sha256': history['sha256'], 'selection_uses_model_feedback': False}
        try:
            if row['source_sequence_id'] in sources:
                raise ValueError('Duplicate source window sequence already accepted for another registered seed')
            insert_history(reference, history, items)
        except (ValueError, TypeError, KeyError) as error:
            row.update(status='structurally_rejected', reason={'type': type(error).__name__, 'message': str(error)})
            _once(directory / f'draw-{draw:05}.json', row)
            progress({'stage': 'source_admissibility', 'sequence_key': key, 'draw_index': draw, 'status': row['status'], 'reason': row['reason']})
            continue
        row['status'] = 'accepted_before_model_feedback'
        _once(directory / f'draw-{draw:05}.json', row)
        reference['protocol']['structural_admissibility'] = {
            **rule, 'accepted_draw_index': draw, 'rejected_candidates': draw, 'history_settings_sha256': history['sha256'],
            'accepted_recipe_path': str(directory / f'draw-{draw:05}.json'),
            'accepted_recipe_sha256': file_digest(directory / f'draw-{draw:05}.json'),
            'population_conditioning': 'First candidate satisfying the frozen chronological history construction, filler and suffix identity-reuse constraints; common to all attack families',
            'reconstruction': 'Replay _reference with pinned source/settings, logical sequence seed and retained draw_seed; candidate digest precedes this admissibility annotation',
            'coverage_caveat': 'Constraint-conditioned chronological calibration/test populations are not asserted exchangeable'}
        progress({'stage': 'source_admissibility', 'sequence_key': key, 'draw_index': draw, 'status': row['status']})
        return reference
    _once(directory / 'exhausted.json', {'status': 'structural_draw_budget_exhausted', 'sequence_seed': seed, 'attempted_candidates': rule['max_draws_per_sequence'], 'selection_uses_model_feedback': False})
    raise ValueError(f'No structurally admissible source sequence for {key} within the frozen draw budget; all candidate recipes retained')


def export_rq1_sequences(database_path, manifest_path, output_path, frozen_sequence_settings, attack_settings, *, progress=None):
    progress = progress or (lambda row: None)
    p = _settings(frozen_sequence_settings)
    if not isinstance(attack_settings, dict) or not attack_settings or not set(attack_settings) <= set(FAMILIES):
        raise ValueError('Explicit supported attack settings are required')
    for family, setting in attack_settings.items():
        if family == 'deceptive_text_rewriting':
            if not isinstance(setting, dict) or set(setting) != {'status', 'reason'} or setting['status'] != 'blocked' or not isinstance(setting['reason'], str) or not setting['reason']:
                raise ValueError('Rewriting must have an explicit blocked status/reason; its construction is a separate pipeline')
            continue
        checked = freeze_attack_config(setting.get('parameters'), setting.get('provenance'))
        if setting != checked or setting['parameters']['attack_family'] != family or setting['parameters']['dataset'] != p['dataset']:
            raise ValueError('Attack family/dataset/settings pin differs')
    if 'structural_admissibility' in p and 'interaction_history_manipulation' not in attack_settings:
        raise ValueError('Structural admissibility requires the exact frozen history-attack settings')
    database, manifest_path, output = (Path(v).resolve() for v in (database_path, manifest_path, output_path))
    if output == database.parent or database.parent in output.parents or output in (database, manifest_path):
        raise ValueError('Write sequence artifacts outside immutable prepared data')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, source, database_stat = _source(database, manifest_path, p['dataset'])
        immutable = {'version': VERSION, 'source': source, 'sequence_settings': frozen_sequence_settings, 'attack_settings': attack_settings,
                     'implementation_sha256': file_digest(__file__), 'attack_implementation_sha256': file_digest(Path(__file__).with_name('rq1_attacks.py')),
                     'protocol_implementation_sha256': file_digest(Path(__file__).with_name('rq1_protocol.py'))}
        _once(output / 'export-intent.json', immutable)
        intent_hash = file_digest(output / 'export-intent.json')
        with _readonly(database) as connection:
            stored = connection.execute('SELECT value FROM state WHERE key=?', ('fingerprint',)).fetchone()
            if stored is None or json.loads(stored[0]) != manifest['fingerprint']:
                raise ValueError('Prepared database and manifest fingerprints differ')
            items = _catalog(connection, p['text_chars'])
            if p['target_selection'] == BOTTOM90:
                counts = dict(connection.execute('SELECT item_id,COUNT(*) FROM events WHERE ts<? GROUP BY item_id', (manifest['boundaries_unix'][0],)))
                for identity, item in items.items():
                    item['train_count'] = counts.get(identity, 0)
            pool = _window_index(connection, output, immutable, p, manifest, items, progress)
        for allocation in p['allocations']:
            if pool['window_counts'].get(allocation['split'], 0) < max(p['T'], allocation['count']):
                raise ValueError('Requested allocation exceeds available distinct source windows; reduce count/T explicitly')
        source.update(pool_sha256=file_digest(output / 'window-pool.json'), source_split='prepared_chronological_calibration_and_test')
        planned = []
        with _readonly(output / 'windows.sqlite') as index:
            sources = set()
            for allocation in p['allocations']:
                for offset in range(allocation['count']):
                    seed = allocation['seed_start'] + offset
                    key = f'{allocation["split"]}-{allocation["seed_group"]}-{seed}'
                    reference = _admissible_reference(index, pool, items, p, frozen_sequence_settings, allocation['split'], seed, attack_settings, output, key, sources, progress)
                    source_id = reference['protocol']['source_sequence_id']
                    sources.add(source_id)
                    reference_path = output / 'references' / (key + '.json')
                    _once(reference_path, reference)
                    planned.append({'key': key, 'split': allocation['split'], 'seed_group': allocation['seed_group'], 'sequence_seed': seed, 'source_sequence_id': source_id,
                                    'reference_path': str(reference_path), 'reference_sha256': file_digest(reference_path)})
        _once(output / 'planned-sequences.json', planned)
        rows, outcomes, blocked = [], [], {}
        for family in sorted(attack_settings):
            if family == 'deceptive_text_rewriting':
                blocked[family] = {'reason': attack_settings[family]['reason'], 'planned_sequences': len(planned)}
                for attempt in planned:
                    outcome = {'id': p['dataset'] + '/' + family + '/' + attempt['key'], 'status': 'blocked', 'source_sequence_id': attempt['source_sequence_id'], 'reason': attack_settings[family]['reason'], 'pair_available_for_evaluation': False}
                    _once(output / 'pairs' / family / attempt['key'] / 'attempt.json', outcome)
                    outcomes.append(outcome)
                continue
            family_rows, failures = [], []
            for attempt in planned:
                reference = read_json(output / 'references' / (attempt['key'] + '.json'))
                identity = p['dataset'] + '/' + family + '/' + attempt['key']
                directory = output / 'pairs' / family / attempt['key']
                try:
                    builder = inject_instructions if family == 'instruction_injection' else insert_history
                    bundle = builder(reference, attack_settings[family]) if family == 'instruction_injection' else builder(reference, attack_settings[family], items)
                    validate_attack_pair(bundle['pair'], bundle['reference'], bundle['audit'], dataset=p['dataset'], attack_family=family, ingestion=bundle['ingestion'])
                    descriptor = {'id': identity, 'dataset': p['dataset'], 'attack_family': family, 'split': attempt['split'], 'seed_group': attempt['seed_group'],
                                  'sequence_seed': attempt['sequence_seed'], 'source_sequence_id': attempt['source_sequence_id'], 'source': source}
                    for name in ('pair', 'reference', 'audit', 'ingestion'):
                        if bundle[name] is None:
                            continue
                        path = directory / (name + '.json')
                        _once(path, bundle[name])
                        descriptor[name + '_path'], descriptor[name + '_sha256'] = str(path), file_digest(path)
                    load_pair_artifact(descriptor)
                    family_rows.append(descriptor)
                    outcome = {'id': identity, 'status': 'prepared', 'attempt_status': bundle['audit']['attempt_status'], 'source_sequence_id': attempt['source_sequence_id'], 'audit_sha256': descriptor['audit_sha256']}
                except (ValueError, TypeError, KeyError) as error:
                    outcome = {'id': identity, 'status': 'construction_failed', 'source_sequence_id': attempt['source_sequence_id'], 'error': {'type': type(error).__name__, 'message': str(error)}, 'pair_available_for_evaluation': False}
                    failures.append(identity)
                _once(directory / 'attempt.json', outcome)
                outcomes.append(outcome)
                progress({'stage': 'preparing_attacks', **outcome})
            if failures:
                blocked[family] = {'reason': 'Some pre-registered construction attempts failed; this entire family is excluded from runnable descriptors to prevent success filtering', 'failed_attempt_ids': failures, 'planned_sequences': len(planned)}
            else:
                rows.extend(family_rows)
        current_stat = database.stat()
        if current_stat.st_size != database_stat.st_size or current_stat.st_mtime_ns != database_stat.st_mtime_ns or file_digest(manifest_path) != source['manifest_sha256']:
            raise ValueError('Prepared source changed during export')
        _once(output / 'descriptors.json', rows)
        observed = {'schema_version': 1, 'kind': 'rq1_observed_sequence_views', 'dataset': p['dataset'], 'source_manifest_sha256': source['manifest_sha256'], 'sequence_export_sha256': intent_hash}
        _observed_file(output / 'observed-sequences.json', observed, rows)
        result = {'schema_version': 1, 'kind': 'rq1_sequence_export', 'status': 'complete' if not blocked else 'partially_blocked' if rows else 'blocked',
                  'dataset': p['dataset'], 'source': source, 'export_intent_sha256': intent_hash, 'frozen_sequence_settings': frozen_sequence_settings,
                  'planned_sequences': len(planned), 'prepared_descriptors': len(rows), 'blocked_families': blocked, 'attempts': outcomes,
                  'planned_sequences_path': str(output / 'planned-sequences.json'), 'planned_sequences_sha256': file_digest(output / 'planned-sequences.json'),
                  'descriptors_path': str(output / 'descriptors.json'), 'descriptors_sha256': file_digest(output / 'descriptors.json'),
                  'observed_sequences_path': str(output / 'observed-sequences.json'), 'observed_sequences_sha256': file_digest(output / 'observed-sequences.json'),
                  'pool_path': str(output / 'window-pool.json'), 'window_counts': pool['window_counts'], 'source_partition_caveat': pool['coverage_caveat'],
                  'source_selection_ledger': str(output / 'source-selection-ledger') if 'structural_admissibility' in p else None,
                  'is_full_reproduction': False, 'attack_protocol': 'Explicit zero-feedback development construction; no observed-ranking optimization or unimplemented rewriting is claimed'}
        _once(output / 'manifest.json', result)
        return result
