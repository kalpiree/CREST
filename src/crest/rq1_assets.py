import copy
import fcntl
import hashlib
import heapq
import json
import random
import sqlite3
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .defenses.return_del import build_hop_graph, observed_views


DATASETS = {"amazon-2018-all-beauty", "steam", "movielens-1m"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an explicit integer >= {minimum}")


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    _require(parsed.tzinfo is not None, "Source timestamps must include a timezone")
    return parsed.timestamp()


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"), "An explicit SHA256 is required")
    return value


def _read_pinned(path, checksum):
    _sha(checksum)
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == checksum, "Observed input checksum mismatch")
    return json.loads(raw)


def _coverage(path, checksum, dataset, manifest_hash):
    wrapper = _read_pinned(path, checksum)
    _require(wrapper.get("schema_version") == 1 and wrapper.get("kind") == "rq1_observed_sequence_views" and
             wrapper.get("dataset") == dataset and wrapper.get("source_manifest_sha256") == manifest_hash,
             "Observed input must identify the same prepared dataset/manifest")
    _sha(wrapper.get("sequence_export_sha256"))
    _require(isinstance(wrapper.get("sequences"), list) and bool(wrapper["sequences"]), "Observed sequence coverage is empty")
    text_items, history_items, ids, max_history = {}, set(), set(), 0
    for row in wrapper["sequences"]:
        _require(isinstance(row.get("id"), str) and row["id"] and row["id"] not in ids, "Observed sequence IDs must be unique")
        ids.add(row["id"])
        _require(row.get("split") in ("calibration", "test"), "Observed sequence roles must be calibration/test")
        for key in ("seed_group", "sequence_seed"):
            _integer(row.get(key), key)
        for decision in row["decisions"]:
            _require(set(decision) == {"user_id", "candidates", "records"}, "Pass observed-only provider decisions, without labels or cutoff metadata")
            for record in decision["records"]:
                _require(set(record) == {"id", "type", "fields"}, "Provider record metadata is not permitted")
        for decision in observed_views(row["decisions"]):
            history = [r for r in decision["records"] if r["type"] == "interaction"]
            max_history = max(max_history, len(history))
            for record in history:
                item = record["fields"].get("item_id")
                _require(isinstance(item, str) and item, "History coverage needs trusted item references")
                history_items.add(item)
            for record in decision["records"]:
                if record["type"] != "item_text":
                    continue
                item = record["fields"].get("item_id")
                _require(isinstance(item, str) and item and record["id"] == "item/" + item,
                         "Cohort exporter supports canonical item/<item_id> record identities")
                _require(list(record["fields"]) == ["item_id", "title", "description"], "Canonical item text schema must be item_id/title/description")
                text_items[item] = record["id"]
    identity_content = {"text_items": dict(sorted(text_items.items())), "history_items": sorted(history_items), "max_history": max_history}
    return {**identity_content, "coverage_identity_sha256": digest(identity_content), "observed_input_sha256": checksum,
            "sequence_export_sha256": wrapper["sequence_export_sha256"], "sequence_count": len(ids)}


@contextmanager
def _prepared(database_path, manifest_path, observed_path, observed_sha256, output_path, source_split):
    _require(source_split in ("train", "development"), "Only train/development source splits are permitted")
    database, manifest_file, observed_file, output = (Path(p).resolve() for p in (database_path, manifest_path, observed_path, output_path))
    _require(not output.is_relative_to(database.parent) and output not in (observed_file, manifest_file),
             "Write assets outside immutable prepared inputs")
    manifest_hash = file_digest(manifest_file)
    manifest = read_json(manifest_file)
    expected = manifest.get("artifacts", {}).get("records.sqlite", {})
    _require(manifest.get("status") == "complete" and manifest.get("dataset") in DATASETS and expected.get("sha256"),
             "A complete supported prepared manifest with records.sqlite checksum is required")
    coverage = _coverage(observed_file, observed_sha256, manifest["dataset"], manifest_hash)
    wal = Path(str(database) + "-wal")
    _require(not wal.exists() or wal.stat().st_size == 0, "Prepared database has uncheckpointed WAL data")
    _require(database.stat().st_size == expected.get("bytes") and file_digest(database) == expected["sha256"],
             "Prepared database checksum or size differs from manifest")
    boundaries = manifest.get("boundaries_unix")
    _require(isinstance(boundaries, list) and len(boundaries) == 3 and all(type(v) is int for v in boundaries) and
             boundaries == sorted(boundaries), "Prepared chronological boundaries are invalid")
    lower, upper = (None, boundaries[0]) if source_split == "train" else tuple(boundaries[:2])
    _require(lower is None or lower < upper, "Requested source interval is empty")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        fingerprint = connection.execute("SELECT value FROM state WHERE key='fingerprint'").fetchone()
        _require(fingerprint is not None and json.loads(fingerprint[0]) == manifest["fingerprint"], "Prepared database fingerprint differs from manifest")
        items = {}
        for identity, payload in connection.execute("SELECT item_id,payload FROM items ORDER BY item_id"):
            item = json.loads(payload)
            _require(item.get("id") == identity and all(isinstance(item.get(k), str) for k in ("title", "description")),
                     "Prepared item payload does not match its identity/content columns")
            items[identity] = {"id": identity, "title": item["title"], "description": item["description"]}
        _require(bool(items), "Prepared catalog is empty")
        missing = (set(coverage["text_items"]) | set(coverage["history_items"])) - set(items)
        _require(not missing, "Coverage identities are absent from the real prepared catalog: " + str(sorted(missing)))
        evidence = {"source_database": str(database), "source_database_sha256": expected["sha256"], "source_database_bytes": expected["bytes"],
                    "source_manifest": str(manifest_file), "source_manifest_sha256": manifest_hash, "source_fingerprint": manifest["fingerprint"],
                    "source_split": source_split, "boundaries_unix": boundaries, "observed_path": str(observed_file),
                    "coverage": coverage, "export_code_sha256": file_digest(__file__),
                    "metadata_policy": "Literal prepared static catalog metadata; no historical metadata availability claim",
                    "evaluation_feedback_used": False}
        yield connection, manifest, items, coverage, lower, upper, evidence
        _require(file_digest(database) == expected["sha256"] and file_digest(manifest_file) == manifest_hash and
                 file_digest(observed_file) == observed_sha256, "Immutable inputs changed during read-only export")
        _require(not wal.exists() or wal.stat().st_size == 0, "Prepared database gained uncheckpointed WAL data")
    finally:
        connection.close()


def _events(connection, items, upper, lower=None):
    sql = "SELECT event_id,user_id,item_id,ts,positive,payload FROM events WHERE ts < ?"
    args = [upper]
    if lower is not None:
        sql += " AND ts >= ?"
        args.append(lower)
    sql += " ORDER BY user_id,ts,event_id"
    for identity, user, item, timestamp, positive, payload in connection.execute(sql, args):
        event = json.loads(payload)
        _require(event.get("id") == identity and event.get("user_id") == user and event.get("item_id") == item and
                 event.get("unix_timestamp") == timestamp and type(event.get("positive")) is bool and
                 bool(event["positive"]) == bool(positive) and positive in (0, 1) and item in items and
                 _timestamp(event["timestamp"]) == timestamp, "Prepared event columns and canonical payload disagree")
        _require(isinstance(user, str) and user and isinstance(identity, str) and identity, "Prepared user/event identity is empty")
        yield {"id": identity, "user_id": user, "item_id": items[item]["id"], "timestamp": event["timestamp"],
               "unix_timestamp": timestamp, "positive": event["positive"], "event": event["event"]}


def _guard_output(database_path, manifest_path, observed_path, output_path):
    database, manifest, observed, output = (Path(p).resolve() for p in (database_path, manifest_path, observed_path, output_path))
    _require(not output.is_relative_to(database.parent) and output not in (manifest, observed),
             "Write assets outside immutable prepared inputs")


@contextmanager
def _output_lock(output_path):
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield path


def _save(path, artifact, manifest_path):
    if path.exists():
        _require(read_json(path) == artifact, "Output already belongs to different immutable inputs/settings")
    else:
        write_json(path, artifact)
    descriptor = {"path": str(path), "sha256": file_digest(path), "manifest_path": str(Path(manifest_path).resolve()),
                  "manifest_sha256": artifact["source_manifest_sha256"]}
    descriptor_path = Path(str(path) + ".descriptor.json")
    if descriptor_path.exists():
        _require(read_json(descriptor_path) == descriptor, "Existing asset descriptor differs")
    else:
        write_json(descriptor_path, descriptor)
    return {"status": "complete", "kind": artifact["kind"], "descriptor": descriptor, "descriptor_path": str(descriptor_path),
            "coverage": artifact["coverage"], "statistics": artifact["statistics"]}


def export_return_graph(database_path, manifest_path, observed_path, observed_sha256, output_path, *, source_split,
                        history_scope, max_hops, history_cap, unknown_item_policy):
    _integer(max_hops, "max_hops", 1)
    _integer(history_cap, "history_cap")
    _require(history_scope in ("split_only", "prefix_through_split"), "Explicit graph history scope is required")
    _require(unknown_item_policy in ("error", "zero_support"), "Explicit unknown-item policy is required")
    _guard_output(database_path, manifest_path, observed_path, output_path)
    with _output_lock(output_path) as output:
        with _prepared(database_path, manifest_path, observed_path, observed_sha256, output, source_split) as source:
            connection, manifest, items, coverage, lower, upper, evidence = source
            _require(coverage["max_history"] <= max_hops + 1, "Observed history exceeds declared graph hop coverage")
            stats = {"source_events": 0, "retained_events": 0, "users": 0, "histories_capped": 0}
            trace = hashlib.sha256()
            def histories():
                history, current_user, source_count = deque(maxlen=history_cap or None), None, 0
                def flush(user):
                    stats["users"] += 1
                    stats["retained_events"] += len(history)
                    stats["histories_capped"] += source_count > len(history)
                    trace.update(json.dumps({"user_id": user, "events": list(history)}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
                    trace.update(b"\n")
                    return [event["item_id"] for event in history]
                for event in _events(connection, items, upper, lower if history_scope == "split_only" else None):
                    stats["source_events"] += 1
                    if current_user is not None and current_user != event["user_id"]:
                        yield flush(current_user)
                        history.clear()
                        source_count = 0
                    current_user = event["user_id"]
                    source_count += 1
                    history.append({"id": event["id"], "item_id": event["item_id"], "timestamp": event["timestamp"]})
                if current_user is not None:
                    yield flush(current_user)
            graph = build_hop_graph(histories(), source_split=source_split, source_manifest_sha256=evidence["source_manifest_sha256"], max_hops=max_hops)
            missing = sorted(set(coverage["history_items"]) - graph.item_ids)
            _require(unknown_item_policy == "zero_support" or not missing, "History items absent from source corpus: " + str(missing))
            graph_coverage = {"requested_history_item_count": len(coverage["history_items"]), "corpus_item_count": len(graph.item_ids),
                              "missing_item_ids": missing, "unknown_item_policy": unknown_item_policy,
                              "invented_graph_rows": 0, "maximum_observed_history": coverage["max_history"], "max_hops": max_hops}
            artifact = {"schema_version": 1, "kind": "rq1_return_hop_graph", "dataset": manifest["dataset"], "source_split": source_split,
                        "source_manifest_sha256": evidence["source_manifest_sha256"], "is_synthetic": False,
                        "max_hops": max_hops, "counts": [[gap, first, second, count] for (gap, first, second), count in sorted(graph.counts.items())],
                        "catalog_item_ids": sorted(graph.item_ids), "corpus_sha256": graph.corpus_sha256, "graph_sha256": graph.graph_sha256,
                        "coverage": graph_coverage, "statistics": dict(stats, graph_rows=len(graph.counts)),
                        "provenance": {**evidence, "settings": {"history_scope": history_scope, "history_cap": history_cap,
                            "max_hops": max_hops, "unknown_item_policy": unknown_item_policy}, "source_history_trace_sha256": trace.hexdigest(),
                            "history_rule": "Real events ordered by user, timestamp and event ID; all event types; history_cap keeps latest events (0 means all)",
                            "same_timestamp_rule": "Stable event-ID ordering within a timestamp; no artificial time differences are assigned",
                            "graph_core": graph.provenance(), "coverage_policy": "Only IDs present in retained source histories; absent IDs are reported and never added as fabricated graph rows"}}
        return _save(output, artifact, manifest_path)


def _windows(connection, items, lower, upper, history_count, stats):
    history = deque(maxlen=history_count)
    simultaneous, current_user, current_time = [], None, None
    def candidate():
        positives = [event for event in simultaneous if event["positive"]]
        if positives and len(history) == history_count and (lower is None or current_time >= lower):
            event = min(positives, key=lambda row: row["id"])
            stats["eligible_source_windows"] += 1
            return {"user_id": current_user, "cutoff": event["timestamp"], "cutoff_unix": current_time,
                    "source_positive_event_id": event["id"], "history": copy.deepcopy(list(history))}
        return None
    for event in _events(connection, items, upper):
        stats["source_events_scanned"] += 1
        if current_user is not None and (event["user_id"] != current_user or event["unix_timestamp"] != current_time):
            window = candidate()
            if window is not None:
                yield window
            if event["user_id"] != current_user:
                history.clear()
            else:
                history.extend(simultaneous)
            simultaneous.clear()
        current_user, current_time = event["user_id"], event["unix_timestamp"]
        simultaneous.append(event)
    if current_user is not None:
        window = candidate()
        if window is not None:
            yield window


def export_frequency_cohort(database_path, manifest_path, observed_path, observed_sha256, output_path, *, source_split,
                            cohort_size, candidate_count, recommendation_k, history_count, selection_seed, text_chars, candidate_policy):
    for name, value in (("cohort_size", cohort_size), ("candidate_count", candidate_count), ("recommendation_k", recommendation_k), ("history_count", history_count)):
        _integer(value, name, 1)
    for name, value in (("selection_seed", selection_seed), ("text_chars", text_chars)):
        _integer(value, name)
    _require(recommendation_k <= candidate_count, "recommendation_k exceeds candidate count")
    _require(candidate_policy == "cover_observed_identities", "Explicit cover_observed_identities candidate policy is required")
    _guard_output(database_path, manifest_path, observed_path, output_path)
    with _output_lock(output_path) as output:
        with _prepared(database_path, manifest_path, observed_path, observed_sha256, output, source_split) as source:
            connection, manifest, items, coverage, lower, upper, evidence = source
            required = sorted(coverage["text_items"])
            _require(bool(required), "No scored item-text identities were supplied for cohort coverage")
            _require(candidate_count <= len(items), "Candidate count exceeds actual catalog size")
            _require(len(required) <= cohort_size * candidate_count, "Insufficient cohort candidate slots for required identity coverage")
            stats = {"source_events_scanned": 0, "eligible_source_windows": 0}
            heap = []
            for index, window in enumerate(_windows(connection, items, lower, upper, history_count, stats)):
                key = int(digest({"seed": selection_seed, "user_id": window["user_id"], "cutoff_unix": window["cutoff_unix"],
                                  "source_positive_event_id": window["source_positive_event_id"]}), 16)
                entry = (-key, -index, window)
                if len(heap) < cohort_size:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
            _require(len(heap) == cohort_size, f"Insufficient real {source_split} chronological windows for requested cohort_size")
            selected = [row[2] for row in sorted(heap, key=lambda row: (-row[0], -row[1]))]
            required.sort(key=lambda item: (digest({"seed": selection_seed, "coverage_item": item}), item))
            catalog, decisions, windows = sorted(items), [], []
            for index, window in enumerate(selected):
                chosen = required[index * candidate_count:(index + 1) * candidate_count]
                rng = random.Random(digest({"seed": selection_seed, "window": window["source_positive_event_id"], "candidate_policy": candidate_policy}))
                chosen_set = set(chosen)
                chosen += rng.sample([item for item in catalog if item not in chosen_set], candidate_count - len(chosen))
                rng.shuffle(chosen)
                def text(item, field):
                    value = items[item][field]
                    return value if text_chars == 0 else value[:text_chars]
                records = [{"id": event["id"], "type": "interaction", "fields": {"user_id": window["user_id"], "item_id": event["item_id"],
                    "title": text(event["item_id"], "title"), "timestamp": event["timestamp"], "event": event["event"]}} for event in window["history"]]
                records += [{"id": "item/" + item, "type": "item_text", "fields": {"item_id": item, "title": text(item, "title"),
                            "description": text(item, "description")}} for item in chosen]
                decisions.append({"user_id": window["user_id"], "candidates": chosen, "records": records})
                windows.append({"user_id": window["user_id"], "cutoff": window["cutoff"], "cutoff_unix": window["cutoff_unix"],
                                "source_positive_event_id": window["source_positive_event_id"],
                                "history_event_ids": [event["id"] for event in window["history"]],
                                "history_max_timestamp": max(event["unix_timestamp"] for event in window["history"])})
            observed_views(decisions)
            present = {r["id"] for d in decisions for r in d["records"] if r["type"] == "item_text"}
            missing = sorted(set(coverage["text_items"].values()) - present)
            _require(not missing, "Exported cohort failed requested identity coverage")
            cohort_coverage = {"requested_item_count": len(required), "requested_record_ids": sorted(coverage["text_items"].values()),
                               "missing_record_ids": missing, "cohort_candidate_slots": cohort_size * candidate_count}
            artifact = {"schema_version": 1, "kind": "rq1_frequency_cohort", "dataset": manifest["dataset"], "source_split": source_split,
                        "source_manifest_sha256": evidence["source_manifest_sha256"], "is_synthetic": False, "decisions": decisions,
                        "coverage": cohort_coverage, "statistics": dict(stats, cohort_size=cohort_size, source_users=len({w["user_id"] for w in windows})),
                        "provenance": {**evidence, "settings": {"cohort_size": cohort_size, "candidate_count": candidate_count,
                            "recommendation_k": recommendation_k, "history_count": history_count, "selection_seed": selection_seed,
                            "text_chars": text_chars, "candidate_policy": candidate_policy}, "source_windows": windows,
                            "window_selection": "Smallest seeded SHA256 priorities across real positive user/cutoff windows; one lowest-event-ID positive per simultaneous group; no feedback or coverage-dependent user/window choice",
                            "history_rule": "Latest real events strictly before each cutoff, including earlier train events for development windows; simultaneous events excluded",
                            "candidate_rule": "Seeded partition of observed identity coverage, with remaining slots sampled from real static catalog and shuffled; source positive labels never force candidate membership",
                            "coverage_conditioning": "Observed item identities only; no observed payload text, evaluation positives, attack labels or model outputs",
                            "scoring_role": "Fixed train/development query cohort for unchanged RewriteDetection recommendation-frequency component"}}
        return _save(output, artifact, manifest_path)
