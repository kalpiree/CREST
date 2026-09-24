import argparse
import copy
import importlib.util
import json
from pathlib import Path

from crest import rq1_sequence_export as source
from crest.artifacts import digest, file_digest, read_json
from crest.controls import PAPER_PROTOCOL, seal


NAMESPACE = "crest-rq3-beauty-injection-independent-blocks-v1-20260910"


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("crest_release_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_seed(block, role, index):
    return int(digest({"namespace": NAMESPACE, "block": block, "role": role, "index": index})[:15], 16)


def injection_projection(projector):
    native = projector.decision_map

    def mapping(reference, source_id):
        try:
            return native(reference, source_id)
        except ValueError as error:
            if str(error) != "Source lacks a repeated user in the attacked half; retain this failure without choosing another sequence":
                raise
        identity = {"policy_sha256": digest(projector.POLICY), "reference_sha256": digest(reference),
                    "source_sequence_id": source_id, "sequence_seed": reference["seed"]}
        order = lambda i: (digest({**identity, "decision_index": i}), i)
        selected = sorted(sorted(range(50), key=order)[:10] + sorted(range(50, 100), key=order)[:10])
        return seal({
            **identity, "old_indices": selected,
            "old_to_new": {str(old): new for new, old in enumerate(selected)},
            "anchor_user": None, "anchor_old_indices": [],
            "selection_reads": ["reference", "descriptor_identity"],
            "scheduled_affected_decisions": list(range(10, 20)), "T": 20, "C": 50, "K": 5,
            "adaptation": "Balanced source halves without requiring a repeated attacked user; no replacement draw.",
            "rq3_injection_no_anchor_fallback": True,
        })

    projector.decision_map = mapping
    return projector


def write_bundle(preparation, directory, bundle, descriptor):
    descriptor = copy.deepcopy(descriptor)
    for name in ("pair", "reference", "audit", "ingestion"):
        if bundle.get(name) is not None:
            path = directory / (name + ".json")
            preparation.once(path, bundle[name])
            descriptor[name + "_path"] = str(path)
            descriptor[name + "_sha256"] = file_digest(path)
    source.load_pair_artifact(descriptor)
    return descriptor


def prepare(experiment_path, output):
    preparation = helper("prepare_experiment")
    experiment, settings = preparation.experiment(experiment_path)
    if settings["dataset"] != "amazon-2018-all-beauty" or settings["mode"] != "source-projection":
        raise ValueError("RQ3 requires the Beauty source-projection experiment")
    output = Path(output).resolve()
    options = settings["options"]
    frozen = settings["sequence_settings"]
    parameters = source._settings(frozen)
    if parameters["T"] != 100 or parameters["K"] != 10:
        raise ValueError("RQ3 preserves the original T100/K10 source construction")
    database = Path(settings["prepared_data"]["database"]["path"])
    manifest_path = Path(settings["prepared_data"]["manifest"]["path"])
    manifest, source_pins, _ = source._source(database, manifest_path, parameters["dataset"])
    pool_path = experiment / "sequences/window-pool.json"
    pool = read_json(pool_path)
    index_path = experiment / "sequences/windows.sqlite"
    if (
        pool["window_index_sha256"] != file_digest(index_path)
        or pool["source_manifest_sha256"] != source_pins["manifest_sha256"]
        or pool["source_database_sha256"] != source_pins["database_sha256"]
    ):
        raise ValueError("Window pool and prepared dataset do not match")
    source_pins.update(pool_sha256=file_digest(pool_path), source_split="rq3_shared_heldout_test_pool")
    configuration = {
        "dataset": parameters["dataset"], "attack_family": "instruction_injection",
        **{key: PAPER_PROTOCOL[key] for key in ("N", "T", "C", "K")},
        "alpha": 0.10, "eta": 0.10,
        **{key: options[key] for key in ("beta", "penalty", "beam_width", "max_depth")},
        "methods": ["CREST"],
    }
    preparation.once(output / "preparation.json", seal({
        "protocol": PAPER_PROTOCOL, "config": configuration,
        "source_identity": settings["source_identity"], "source": source_pins,
        "allocation_namespace": NAMESPACE,
        "sampling": "Independent block/role/index seeds from the same held-out test pool, with no across-sequence deduplication or outcome-based replacement.",
        "projection": "T100 to T20, ten decisions from each half; use a repeated-user anchor when present and balanced hash ordering otherwise.",
    }))
    with source._readonly(database) as connection:
        state = connection.execute("SELECT value FROM state WHERE key=?", ("fingerprint",)).fetchone()
        if state is None or json.loads(state[0]) != manifest["fingerprint"]:
            raise ValueError("Prepared database fingerprint changed")
        items = source._catalog(connection, parameters["text_chars"])
        counts = dict(connection.execute("SELECT item_id, COUNT(*) FROM events WHERE ts<? GROUP BY item_id", (manifest["boundaries_unix"][0],)))
        for identity, item in items.items():
            item["train_count"] = counts.get(identity, 0)
    projector = injection_projection(helper("project_sequences"))
    attack = settings["attack_settings"]["instruction_injection"]
    with source._readonly(index_path) as index:
        for block in range(PAPER_PROTOCOL["blocks"]):
            unit = {"kind": "crest-control-unit-v1", "block": block, "config": configuration,
                    "calibration": [], "test": []}
            for role, count in (("calibration", PAPER_PROTOCOL["N"]), ("test", PAPER_PROTOCOL["tests_per_block"])):
                for position in range(count):
                    seed = independent_seed(block, role, position)
                    key = f"block-{block:02d}-{role}-{position:03d}"
                    reference = source._reference(index, pool, items, parameters, frozen, "test", seed)
                    original = source.inject_instructions(reference, attack)
                    descriptor = {
                        "id": "crest-rq3/original/" + key, "dataset": parameters["dataset"],
                        "attack_family": "instruction_injection", "split": role, "seed_group": block,
                        "sequence_seed": seed, "source_sequence_id": reference["protocol"]["source_sequence_id"],
                        "source": source_pins,
                    }
                    directory = output / "draws" / key
                    descriptor = write_bundle(preparation, directory / "original", original, descriptor)
                    bundle = source.load_pair_artifact(descriptor)
                    mapping = projector.decision_map(bundle["reference"], descriptor["source_sequence_id"])
                    projected_settings = projector.family_settings(bundle["pair"]["attack"], "instruction_injection")
                    projected = projector.project_bundle(bundle, mapping, projected_settings)
                    descriptor.update(id="crest-rq3/" + key, source_sequence_id=projected["source_sequence_id"])
                    final = write_bundle(preparation, directory / "projected", projected, descriptor)
                    unit[role].append({"id": final["id"], "seed_group": block, "pair": projected["pair"]})
            preparation.once(output / "units" / f"block-{block:02d}.json", seal(unit))
            print(f"Prepared block {block + 1}/{PAPER_PROTOCOL['blocks']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare independent RQ3 calibration and test blocks from Beauty's held-out pool.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.experiment, args.output)


if __name__ == "__main__":
    main()
