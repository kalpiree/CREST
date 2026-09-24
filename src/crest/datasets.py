import ast
import bisect
import copy
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import sqlite3
import ssl
import time
import urllib.request
import zipfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json


SPLITS = ("train", "development", "calibration", "test")


def emit_progress(stage, **details):
    print(json.dumps({"time": time.time(), "stage": stage, **details}, ensure_ascii=False), flush=True)


def parse_mapping(line):
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = ast.literal_eval(line)
    if not isinstance(value, dict):
        raise ValueError("record is not a mapping")
    return value


def _text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(_text(part) for part in value if part is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _identifier(value):
    if value is None or isinstance(value, (dict, list, bool)) or not str(value).strip():
        raise ValueError("missing identifier")
    return str(value).strip()


def _timestamp(value):
    if isinstance(value, bool):
        raise ValueError("invalid timestamp")
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            raise ValueError("invalid timestamp")
        result = int(number)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        result = int(parsed.replace(tzinfo=timezone.utc).timestamp()) if parsed.tzinfo is None else int(parsed.timestamp())
    else:
        raise ValueError("invalid timestamp")
    if not 0 <= result <= 4102444800:
        raise ValueError("timestamp outside supported 1970-2100 range")
    return result


def _canonical(dataset, kind, original):
    if kind == "user":
        original = hashlib.sha256(original.encode()).hexdigest()[:32]
    return f"{dataset}:{kind}:{original}"


def _json_lines(path, stats, role, start_line=0):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if number <= start_line:
                continue
            stats[f"{role}_rows_scanned"] += 1
            try:
                yield number, parse_mapping(line)
            except (ValueError, SyntaxError, TypeError) as error:
                stats[f"{role}_malformed"] += 1
                if stats[f"{role}_malformed"] <= 5:
                    emit_progress("malformed_record", role=role, source_line=number, error=str(error))


def _movie_lines(path, member):
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="latin-1") as stream:
                yield from enumerate(stream, 1)


def _item_rows(dataset, sources, stats):
    if dataset == "movielens-1m":
        for number, line in _movie_lines(sources["archive"], "ml-1m/movies.dat"):
            stats["metadata_rows_scanned"] += 1
            try:
                item, title, genres = line.rstrip("\r\n").split("::")
                yield number, {"id": item, "title": title, "genres": genres.split("|")}
            except ValueError:
                stats["metadata_malformed"] += 1
    else:
        yield from _json_lines(sources["metadata"], stats, "metadata")


def _normalize_item(dataset, number, row):
    original = _identifier(row.get("asin") if dataset == "amazon-2018-all-beauty" else row.get("id"))
    item = _canonical(dataset, "item", original)
    title = _text(row.get("title") or row.get("app_name")).strip()
    if not title:
        raise ValueError("missing title")
    selected = {}
    for field in ("description", "feature", "brand", "category", "categories", "genres", "tags", "publisher", "developer", "price", "release_date"):
        if row.get(field) is not None:
            selected[field] = copy.deepcopy(row[field])
    json.dumps(selected, allow_nan=False)
    description = _text(row.get("description"))
    if not description:
        description = "; ".join(f"{key}: {_text(value)}" for key, value in selected.items())
    return {
        "id": item,
        "original_id": original,
        "title": title,
        "description": description,
        "metadata": selected,
        "record_ids": {field: _canonical(dataset, "item_field", f"{original}/{field}") for field in sorted({"title", "description", *selected})},
        "source_row": number,
    }


def _event_rows(dataset, sources, stats, start_line=0):
    if dataset == "movielens-1m":
        for number, line in _movie_lines(sources["archive"], "ml-1m/ratings.dat"):
            if number <= start_line:
                continue
            stats["reviews_rows_scanned"] += 1
            try:
                user, item, rating, timestamp = line.rstrip("\r\n").split("::")
                yield number, {"user_id": user, "item_id": item, "rating": rating, "timestamp": timestamp}
            except ValueError:
                stats["reviews_malformed"] += 1
    else:
        yield from _json_lines(sources["reviews"], stats, "reviews", start_line)


def _normalize_event(dataset, number, row, positive_threshold):
    if dataset == "amazon-2018-all-beauty":
        user = _identifier(row.get("reviewerID"))
        item = _identifier(row.get("asin"))
        timestamp = _timestamp(row.get("unixReviewTime"))
        rating = float(row["overall"])
        text, summary = _text(row.get("reviewText")), _text(row.get("summary"))
        positive = rating >= positive_threshold
    elif dataset == "steam":
        user = _identifier(row.get("username"))
        item = _identifier(row.get("product_id"))
        timestamp = _timestamp(row.get("date"))
        rating = None
        text, summary = _text(row.get("text")), ""
        positive = True
    elif dataset == "movielens-1m":
        user = _identifier(row.get("user_id"))
        item = _identifier(row.get("item_id"))
        timestamp = _timestamp(row.get("timestamp"))
        rating = float(row["rating"])
        text = summary = ""
        positive = rating >= positive_threshold
    else:
        raise ValueError("unsupported dataset edition")
    if rating is not None and (not math.isfinite(rating) or not 1 <= rating <= 5):
        raise ValueError("rating outside 1-5")
    return {
        "id": _canonical(dataset, "interaction", f"{number:012d}"),
        "review_id": _canonical(dataset, "review", f"{number:012d}") if dataset != "movielens-1m" else None,
        "original_event_id": _text(row.get("review_id")) or None,
        "user_id": _canonical(dataset, "user", user),
        "original_user_id": user,
        "item_id": _canonical(dataset, "item", item),
        "original_item_id": item,
        "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
        "unix_timestamp": timestamp,
        "rating": rating,
        "positive": positive,
        "text": text,
        "summary": summary,
        "source_row": number,
        "event": "rating" if dataset == "movielens-1m" else "review",
    }


def validate_archive(path, expected_bytes=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty raw archive: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Raw size differs from the verified edition: {path}")
    if path.name.endswith(".gz") or path.name.endswith(".gz.part"):
        with gzip.open(path, "rb") as stream:
            for _ in iter(lambda: stream.read(1024 * 1024), b""):
                pass
    elif path.name.endswith(".zip") or path.name.endswith(".zip.part"):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failed: {bad}")
    else:
        raise ValueError("Expected a gzip or ZIP archive")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_digest(path), "archive_integrity": "verified"}


def download_file(spec, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        result = validate_archive(destination, spec.get("expected_bytes"))
        if spec.get("sha256") and result["sha256"] != spec["sha256"]:
            raise ValueError("Existing source checksum differs from the pinned archive")
        return {**result, "url": spec["url"], "reused": True}
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    expected = spec.get("expected_bytes")
    if expected is not None and offset > expected:
        raise ValueError(f"Partial archive is larger than the expected source: {partial}")
    if offset != expected:
        headers = {"User-Agent": "CREST-research-dataset-preparation/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(spec["url"], headers=headers)
        import certifi
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            append = offset > 0 and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned an unexpected resume range")
            received = offset if append else 0
            with partial.open("ab" if append else "wb") as stream:
                last_report = time.monotonic()
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                    received += len(chunk)
                    if time.monotonic() - last_report >= 10:
                        emit_progress("downloading", filename=destination.name, bytes=received, expected_bytes=expected)
                        last_report = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
    result = validate_archive(partial, expected)
    if spec.get("sha256") and result["sha256"] != spec["sha256"]:
        raise ValueError("Downloaded source checksum differs from the pinned archive")
    os.replace(partial, destination)
    result["path"] = str(destination.resolve())
    return {**result, "url": spec["url"], "reused": False}


def _state(connection, key, value=None):
    if value is None:
        found = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(found[0]) if found else None
    connection.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _open_database(path):
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS items (item_id TEXT PRIMARY KEY, original_id TEXT NOT NULL, payload TEXT NOT NULL, raw_payload TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, item_id TEXT NOT NULL, ts INTEGER NOT NULL, positive INTEGER NOT NULL, payload TEXT NOT NULL)")
    return connection


def _ingest(connection, dataset, sources, settings):
    progress = _state(connection, "ingestion") or {"metadata_done": False, "events_done": False, "last_event_line": 0, "stats": {}}
    stats = Counter(progress["stats"])
    if not progress["metadata_done"]:
        connection.execute("DELETE FROM items")
        for key in list(stats):
            if key.startswith("metadata_"):
                del stats[key]
        for number, row in _item_rows(dataset, sources, stats):
            try:
                item = _normalize_item(dataset, number, row)
                result = connection.execute("INSERT OR IGNORE INTO items VALUES (?,?,?,?)", (item["id"], item["original_id"], json.dumps(item, ensure_ascii=False, sort_keys=True), json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)))
                stats["metadata_accepted" if result.rowcount else "metadata_duplicate_ids"] += 1
            except (ValueError, TypeError, KeyError):
                stats["metadata_invalid_fields"] += 1
        progress.update(metadata_done=True, stats=dict(stats))
        _state(connection, "ingestion", progress)
        connection.commit()
        emit_progress("metadata_ingested", dataset=dataset, statistics=dict(stats))
    item_ids = {row[0] for row in connection.execute("SELECT item_id FROM items")}
    if not item_ids:
        raise ValueError("No valid metadata items")
    if not progress["events_done"]:
        last_line = progress["last_event_line"]
        for number, row in _event_rows(dataset, sources, stats, last_line):
            last_line = number
            try:
                event = _normalize_event(dataset, number, row, settings["positive_rating_threshold"])
                if event["item_id"] not in item_ids:
                    stats["reviews_missing_metadata"] += 1
                else:
                    inserted = connection.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)", (event["id"], event["user_id"], event["item_id"], event["unix_timestamp"], int(event["positive"]), json.dumps(event, ensure_ascii=False, sort_keys=True)))
                    if inserted.rowcount:
                        stats["reviews_accepted"] += 1
                        stats["positive_events"] += int(event["positive"])
                        stats["nonempty_review_texts"] += bool(event["text"])
            except (ValueError, TypeError, KeyError, OverflowError):
                stats["reviews_invalid_fields"] += 1
            if number - progress["last_event_line"] >= 10000:
                progress.update(last_event_line=last_line, stats=dict(stats))
                _state(connection, "ingestion", progress)
                connection.commit()
                emit_progress("events_ingesting", dataset=dataset, source_line=number, accepted=stats["reviews_accepted"], malformed=stats["reviews_malformed"])
        progress.update(last_event_line=last_line, events_done=True, stats=dict(stats))
        _state(connection, "ingestion", progress)
        connection.commit()
    emit_progress("indexing", dataset=dataset, accepted=stats["reviews_accepted"])
    connection.execute("CREATE INDEX IF NOT EXISTS events_by_time ON events(ts)")
    connection.execute("CREATE INDEX IF NOT EXISTS events_by_user_time ON events(user_id,ts,event_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS events_by_item_time ON events(item_id,ts)")
    connection.commit()
    return dict(stats)


def _boundaries(connection, quantiles):
    if len(quantiles) != 3 or not 0 < quantiles[0] < quantiles[1] < quantiles[2] < 1:
        raise ValueError("Provide three strictly increasing split quantiles in (0,1)")
    count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if count < 4:
        raise ValueError("At least four valid interactions are needed for chronological splits")
    values = [connection.execute("SELECT ts FROM events ORDER BY ts LIMIT 1 OFFSET ?", (min(count - 1, int(count * q)),)).fetchone()[0] for q in quantiles]
    return values


def _write_pools(connection, output, dataset, settings, provenance, boundaries):
    streams = {}
    paths = {}
    counts = Counter()
    items = {item: json.loads(payload) for item, payload in connection.execute("SELECT item_id,payload FROM items ORDER BY item_id")}
    examples_db = output / "example_index.sqlite"
    index = sqlite3.connect(examples_db)
    index.execute("CREATE TABLE IF NOT EXISTS examples (event_id TEXT PRIMARY KEY, split TEXT NOT NULL, user_id TEXT NOT NULL, cutoff INTEGER NOT NULL)")
    index.execute("DELETE FROM examples")
    try:
        for split in SPLITS:
            path = output / f"{split}.pool.json"
            paths[split] = path
            stream = path.with_suffix(path.suffix + ".partial").open("w", encoding="utf-8")
            streams[split] = stream
            prefix = {"schema_version": 2, "kind": "crest_full_source_pool", "dataset": dataset, "split": split, "is_full_reproduction": False, "items": items, "provenance": {**provenance, "split": split}}
            encoded = json.dumps(prefix, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write(encoded[:-1] + ',"examples":[')
        current_user = None
        current_timestamp = None
        history = deque(maxlen=settings["max_history_export"])
        simultaneous = deque(maxlen=settings["max_history_export"])
        chosen = {}

        def flush_examples():
            for split, example in chosen.items():
                if counts[split]:
                    streams[split].write(",")
                streams[split].write(json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                index.execute("INSERT INTO examples VALUES (?,?,?,?)", (example["positive_event_id"], split, example["user_id"], example["cutoff_unix"]))
                counts[split] += 1

        for number, (user, timestamp, payload) in enumerate(connection.execute("SELECT user_id,ts,payload FROM events ORDER BY user_id,ts,event_id"), 1):
            if user != current_user:
                if current_user is not None:
                    flush_examples()
                current_user, current_timestamp = user, timestamp
                history.clear()
                simultaneous.clear()
                chosen.clear()
            elif timestamp != current_timestamp:
                history.extend(simultaneous)
                simultaneous.clear()
                current_timestamp = timestamp
            event = json.loads(payload)
            split = SPLITS[bisect.bisect_right(boundaries, timestamp)]
            if event["positive"] and len(history) >= settings["min_history"]:
                chosen[split] = {
                    "user_id": user,
                    "original_user_id": event["original_user_id"],
                    "history": list(history),
                    "positive": event["item_id"],
                    "positive_original_id": event["original_item_id"],
                    "positive_event_id": event["id"],
                    "cutoff": event["timestamp"],
                    "cutoff_unix": timestamp,
                    "split": split,
                }
            simultaneous.append(event)
            if number % 100000 == 0:
                index.commit()
                emit_progress("exporting_pools", dataset=dataset, events=number, examples=dict(counts))
        if current_user is not None:
            flush_examples()
        index.commit()
        for split, stream in streams.items():
            stream.write("]}\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(paths[split].with_suffix(paths[split].suffix + ".partial"), paths[split])
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        index.close()
    return {split: counts[split] for split in SPLITS}


def prepare_dataset(dataset, sources, output_dir, settings, source_info=None):
    if dataset not in {"steam", "amazon-2018-all-beauty", "movielens-1m"}:
        raise ValueError("Unsupported explicit dataset edition")
    if settings["min_history"] < 1 or settings["max_history_export"] < settings["min_history"]:
        raise ValueError("Invalid history settings")
    if not 1 <= settings["positive_rating_threshold"] <= 5:
        raise ValueError("Positive rating threshold must lie in 1-5")
    source_info = source_info or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_manifest = {}
        for role, path in sorted(sources.items()):
            emit_progress("hashing_raw_source", dataset=dataset, role=role)
            path = Path(path)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty source: {path}")
            source_manifest[role] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_digest(path)}
            expected_sha = source_info.get("files", {}).get(role, {}).get("sha256")
            if expected_sha and source_manifest[role]["sha256"] != expected_sha:
                raise ValueError(f"Raw source checksum differs from the pinned edition: {path}")
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError(f"ZIP CRC failed: {bad}")
        fingerprint = digest({"dataset": dataset, "source_contents": {key: {"bytes": value["bytes"], "sha256": value["sha256"]} for key, value in source_manifest.items()}, "settings": settings, "source_info": source_info, "code_sha256": file_digest(__file__)})
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous.get("fingerprint") != fingerprint:
                raise ValueError("Prepared directory belongs to different source contents/settings/code; choose a new directory")
            if previous.get("status") == "complete":
                for name, info in previous["artifacts"].items():
                    if not (output / name).is_file() or file_digest(output / name) != info["sha256"]:
                        raise ValueError(f"Prepared artifact is missing or changed: {name}")
                emit_progress("preparation_reused", dataset=dataset)
                return previous
        manifest = {"status": "preparing", "fingerprint": fingerprint, "dataset": dataset, "sources": source_manifest, "settings": settings, "source_info": source_info, "is_full_reproduction": False}
        write_json(manifest_path, manifest)
        connection = _open_database(output / "records.sqlite")
        try:
            previous_fingerprint = _state(connection, "fingerprint")
            if previous_fingerprint is not None and previous_fingerprint != fingerprint:
                raise ValueError("Intermediate SQLite database has a different fingerprint")
            _state(connection, "fingerprint", fingerprint)
            connection.commit()
            stats = _ingest(connection, dataset, sources, settings)
            boundaries = _boundaries(connection, settings["split_quantiles"])
            provenance = {"fingerprint": fingerprint, "sources": source_manifest, "edition": source_info.get("edition", dataset), "preprocessing": settings, "timestamp_boundaries": boundaries, "split_rule": "timestamp<first boundary is train; equal-boundary timestamps go to the later split; same timestamp never straddles splits", "positive_definition": "review occurrence as implicit positive, not sentiment" if dataset == "steam" else f"rating >= {settings['positive_rating_threshold']}", "history_rule": "Only events strictly before the positive event timestamp; simultaneous events excluded; full event table preserved", "retrieval_rule": "Tune/retrieve only from train/development pools, never calibration/test targets or future review text", "rq3_exchangeability": "Not established for chronological calibration/test windows; these pools evaluate temporal shift. RQ3 needs independent draws from the same fixed sequence generator.", "metadata_time_limit": "Metadata is a static source snapshot; no historical availability guarantee", "candidate_catalog_policy": "Every valid metadata item is retained in each pool regardless of first interaction or release date; experiment manifests must declare candidate eligibility, and no historical catalog claim is supported", "source_scope": "Complete raw archives scanned without prefix or user sampling; export histories have an explicit cap", "statistics": stats}
            counts = _write_pools(connection, output, dataset, settings, provenance, boundaries)
            stats["users"] = connection.execute("SELECT COUNT(DISTINCT user_id) FROM events").fetchone()[0]
            stats["interacted_items"] = connection.execute("SELECT COUNT(DISTINCT item_id) FROM events").fetchone()[0]
            stats["metadata_items"] = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            stats["events"] = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            if dataset == "movielens-1m":
                with zipfile.ZipFile(sources["archive"]) as archive:
                    readme = archive.read("ml-1m/README")
                    (output / "SOURCE_README.txt").write_bytes(readme)
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
            artifacts = {}
            for path in sorted(output.iterdir()):
                if path.name in {"records.sqlite", "example_index.sqlite", "SOURCE_README.txt"} or path.name.endswith(".pool.json"):
                    artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
            manifest.update(status="complete", statistics=stats, split_examples=counts, ready_for_sequence_generation=all(counts.values()), empty_splits=[name for name, count in counts.items() if count == 0], boundaries_unix=boundaries, artifacts=artifacts)
            write_json(manifest_path, manifest)
            emit_progress("preparation_complete", dataset=dataset, statistics=stats, split_examples=counts)
            return manifest
        except Exception as error:
            connection.close()
            manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            write_json(manifest_path, manifest)
            raise


def resolve_sources(catalog, dataset, raw_dir, overrides=None):
    definition = catalog["datasets"][dataset]
    overrides = overrides or {}
    result = {}
    for role, spec in definition["files"].items():
        key = f"{dataset}:{role}"
        path = Path(overrides[key]) if key in overrides else Path(raw_dir) / definition["raw_subdir"] / spec["filename"]
        result[role] = path
    return result
