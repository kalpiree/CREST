import copy

INTENSITIES = (0.01, 0.025, 0.05, 0.10, 0.20)
LEVELS = (0.10, 0.25, 0.50, 0.75, 1.0)

def original_attack_settings(attack):
    from .artifacts import digest
    from .rq1_attacks import _config
    value = copy.deepcopy(attack)
    if value.get('status') != 'frozen' or value.get('sha256') != digest({key: value[key] for key in ('parameters', 'provenance')}):
        raise ValueError('Frozen attack settings fingerprint differs')
    if 'projection' in value['parameters']:
        original = value['provenance'].get('original_attack_settings')
        if not isinstance(original, dict) or original.get('sha256') != value['parameters'].get('source_attack_settings_sha256'):
            raise ValueError('Projected settings lack their unchanged original attack settings')
        value = copy.deepcopy(original)
    _config(value, 'instruction_injection')
    return value

def source_indices(reference, source_reference, source_schedule):

    from crest.artifacts import digest
    if reference['seed'] != source_reference['seed'] or reference['evaluation']['target'] != source_reference['evaluation']['target']:
        raise ValueError('Projected primary changed its original seed or target')
    by_digest = {}
    for index, decision in enumerate(source_reference['clean']):
        by_digest.setdefault(digest(decision), []).append(index)
    indices = []
    for decision in reference['clean']:
        choices = by_digest.get(digest(decision), [])
        if len(choices) != 1 or decision != source_reference['clean'][choices[0]]:
            raise ValueError('Primary projection must identify unique unchanged source decisions')
        indices.append(choices[0])
    if indices != sorted(set(indices)):
        raise ValueError('Primary projection must preserve chronological source order')
    if reference['evaluation']['positives'] != [source_reference['evaluation']['positives'][i] for i in indices]:
        raise ValueError('Projected primary positive labels differ from source indices')
    schedule = [new for new, old in enumerate(indices) if old in source_schedule]
    length = len(indices)
    if schedule != list(range(length // 2, length)):
        raise ValueError('Primary intensity reference requires first-half clean and second-half attacked decisions')
    return indices

def intensity_conditions(reference, attack, specification_sha256, *, source_reference=None, T=100):
    from crest.artifacts import file_digest
    from crest.rq1_attacks import _config, _finish, freeze_attack_config, inject_instructions
    from crest.rq1_protocol import validate_attack_pair
    attack = original_attack_settings(attack)
    original = source_reference if source_reference is not None else reference
    if len(reference['clean']) != T or len(original['clean']) not in (T, 100):
        raise ValueError('RQ2 needs matching primary T and the unchanged original T100 source realization')
    if any(len(d['candidates']) != 50 or len(d['records']) != 53 for d in reference['clean']):
        raise ValueError('Intensity retains C50 and 53 existing records per prompt')
    parameters = attack['parameters']
    if parameters['affected_decisions'] != list(range(len(original['clean']) // 2, len(original['clean']))) or parameters['recurrence_fraction'] != .5:
        raise ValueError('Original intensity attack must retain its frozen half-clean/half-attacked schedule')
    indices = source_indices(reference, original, parameters['affected_decisions'])
    mapping = {old: new for new, old in enumerate(indices)}
    results = []
    for intensity in INTENSITIES:
        values = copy.deepcopy(parameters)
        values['intensity'] = intensity
        provenance = {**attack['provenance'], 'rq2_preparation_specification_sha256': specification_sha256,
                      'changed_parameter': 'intensity_only', 'preparation_executor_sha256': file_digest(__file__)}
        full_settings = freeze_attack_config(values, provenance)
        full = inject_instructions(original, full_settings)
        if len(original['clean']) == T:
            bundle, frozen = full, full_settings
        else:
            values.update(affected_decisions=list(range(T // 2, T)), recurrence_fraction=.5)
            frozen = freeze_attack_config(values, {**provenance,
                'adaptation': 'Generate original T100 native intensity attack, including skipped-decision RNG draws; project the fixed matching primary clean-input indices',
                'original_full_settings_sha256': full_settings['sha256'], 'source_decision_indices': indices})
            operations = [dict(copy.deepcopy(op), decision_index=mapping[op['decision_index']])
                          for op in full['audit']['operations'] if op['decision_index'] in mapping]
            observed = [copy.deepcopy(full['pair']['observed'][i]) for i in indices]
            edited = {op['record_id'] for op in operations}
            details = {key: copy.deepcopy(full['pair']['attack'][key]) for key in ('edit_mode', 'target_identity_required_in_affected_decisions')}
            details.update(payloads_by_identity={k: copy.deepcopy(v) for k, v in full['pair']['attack']['payloads_by_identity'].items() if k in edited},
                intensity_projection={'source_T': 100, 'source_decision_indices': indices,
                    'rng_policy': 'Original native T100 generation before fixed input projection', 'no_outcome_selection': True})
            bundle = _finish(reference, observed, _config(frozen, 'instruction_injection'), operations,
                             full['audit']['permitted_edit_fields'], None, details)
        validation = validate_attack_pair(bundle['pair'], bundle['reference'], bundle['audit'],
            dataset='amazon-2018-all-beauty', attack_family='instruction_injection')
        diagnostics = {'intensity': intensity, 'source_T': len(original['clean']), 'T': T, 'source_decision_indices': indices,
            'affected_decisions': list(range(T // 2, T)), 'clean_decision_count': T // 2, 'attacked_decision_count': T // 2,
            'achieved': copy.deepcopy(bundle['pair']['attack']['achieved']), 'outcome_selection': False}
        results.append({'level': intensity, 'bundle': bundle, 'settings': frozen, 'diagnostics': diagnostics, 'validation': validation})
    return results

def schedule_order(seed, length):
    from crest.artifacts import digest
    return sorted(range(length), key=lambda index: digest({
        'protocol': 'rq2-recurrence-nested-dispersed-v1',
        'sequence_seed': seed, 'decision_index': index}))

def recurrence_conditions(reference, attack, specification_sha256, *, T=100):

    from crest.artifacts import digest, file_digest
    from crest.rq1_attacks import _config, _finish, freeze_attack_config, inject_instructions
    from crest.rq1_protocol import validate_attack_pair
    attack = original_attack_settings(attack)
    length = len(reference['clean'])
    if type(T) is not int or T <= 0 or T % 20:
        raise ValueError('Recurrence T must be a positive multiple of 20')
    if length != T or any(len(d['candidates']) != 50 or len(d['records']) != 53 for d in reference['clean']):
        raise ValueError('Recurrence cohort requires matching C50/T and 53 existing records per prompt')
    parameters = copy.deepcopy(attack['parameters'])
    parameters.update(intensity=0.05, recurrence_fraction=1.0,
                      affected_decisions=list(range(length)), record_selection='shared_identities',
                      recurrence_scope='each_manipulated_identity', minimum_reused_identities=3)
    provenance = {**attack['provenance'],
        'rq2_preparation_specification_sha256': specification_sha256,
        'recurrence_wrapper_sha256': file_digest(__file__),
        'selection': 'One v6 all-T shared-identity selection; zero model feedback; original attack seed and payload bank',
        'adaptation': 'Controlled three-identity cohort replaces the intensity sweep per-decision sampler'}
    full_settings = freeze_attack_config(parameters, provenance)
    full = inject_instructions(reference, full_settings)
    identities = full['pair']['evaluation']['attacked_ids']
    if len(identities) != 3 or set(full['pair']['attack']['achieved']['identity_appearance_counts'].values()) != {length}:
        raise ValueError('Exactly three editable identities must be present at every decision; no substitute cohort is selected')
    order = schedule_order(reference['seed'], length)
    results = []
    for level in LEVELS:
        schedule = sorted(order[:int(length * level)])
        parameters = copy.deepcopy(full_settings['parameters'])
        parameters.update(recurrence_fraction=level, affected_decisions=schedule)
        frozen = freeze_attack_config(parameters, {**provenance, 'all_T_selection_settings_sha256': full_settings['sha256'],
            'schedule_order': order, 'schedule_rule': 'SHA256 order by sequence seed and decision index; nested prefixes sorted chronologically'})
        observed = copy.deepcopy(reference['clean'])
        for index in schedule:
            observed[index] = copy.deepcopy(full['pair']['observed'][index])
        operations = [copy.deepcopy(op) for op in full['audit']['operations'] if op['decision_index'] in schedule]
        details = {key: copy.deepcopy(full['pair']['attack'][key]) for key in (
            'payloads_by_identity', 'edit_mode', 'target_identity_required_in_affected_decisions')}
        details['recurrence_construction'] = {'wrapper_sha256': file_digest(__file__),
            'all_T_pair_sha256': digest(full['pair']), 'fixed_record_ids': identities,
            'schedule_order': order, 'no_outcome_selection': True,
            'adaptation': provenance['adaptation']}
        bundle = _finish(reference, observed, _config(frozen, 'instruction_injection'), operations,
                         full['audit']['permitted_edit_fields'], None, details)
        validation = validate_attack_pair(bundle['pair'], bundle['reference'], bundle['audit'],
            dataset='amazon-2018-all-beauty', attack_family='instruction_injection')
        target = reference['evaluation']['target']
        eligible = [index for index, d in enumerate(reference['clean']) if target in d['candidates']]
        diagnostics = {'recurrence_fraction': level, 'affected_decisions': schedule, 'schedule_order': order,
            'fixed_record_ids': identities, 'distinct_manipulated_record_count': 3,
            'manipulated_appearance_count': len(operations),
            'identity_appearance_counts': {identity: len(schedule) for identity in identities},
            'target_eligible_decisions': eligible, 'target_eligibility_fraction': len(eligible) / length,
            'affected_target_eligible_decisions': sorted(set(eligible) & set(schedule)),
            'achieved_intensity_in_affected_prompts': 3 / 53,
            'clean_top_K_and_target_opportunity': 'Computed only by evaluator after clean rankings; never used for cohort or schedule selection'}
        results.append({'level': level, 'bundle': bundle, 'settings': frozen,
                        'diagnostics': diagnostics, 'validation': validation})
    return results

def timing_conditions(reference, attack, specification_sha256):
    from .artifacts import digest
    from .rq1_attacks import _config, _finish, freeze_attack_config
    from .rq1_protocol import validate_attack_pair
    length = len(reference['clean'])
    recurrence = recurrence_conditions(reference, attack, specification_sha256, T=length)
    full = recurrence[-1]
    dispersed = next(value for value in recurrence if value['level'] == 0.5)
    schedules = {'early': list(range(length // 2)), 'delayed': list(range(length // 2, length)),
                 'dispersed': dispersed['bundle']['pair']['evaluation']['affected_decisions']}
    results = []
    for name, schedule in schedules.items():
        parameters = copy.deepcopy(full['settings']['parameters'])
        parameters.update(recurrence_fraction=0.5, affected_decisions=schedule)
        setting = freeze_attack_config(parameters, {**full['settings']['provenance'], 'timing_schedule': name})
        observed = copy.deepcopy(reference['clean'])
        for index in schedule:
            observed[index] = copy.deepcopy(full['bundle']['pair']['observed'][index])
        operations = [copy.deepcopy(op) for op in full['bundle']['audit']['operations'] if op['decision_index'] in schedule]
        details = {key: copy.deepcopy(full['bundle']['pair']['attack'][key]) for key in
                   ('payloads_by_identity', 'edit_mode', 'target_identity_required_in_affected_decisions')}
        details['timing'] = name
        bundle = _finish(reference, observed, _config(setting, 'instruction_injection'), operations,
                         full['bundle']['audit']['permitted_edit_fields'], None, details)
        validate_attack_pair(bundle['pair'], bundle['reference'], bundle['audit'],
                             dataset='amazon-2018-all-beauty', attack_family='instruction_injection')
        if name == 'dispersed' and digest(observed) != digest(dispersed['bundle']['pair']['observed']):
            raise ValueError('Dispersed timing must reuse the 50-percent recurrence inputs')
        results.append({'level': name, 'bundle': bundle, 'settings': setting, 'diagnostics': {'affected_decisions': schedule}})
    return results
