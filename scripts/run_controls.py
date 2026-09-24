import argparse
import fcntl
import time
from pathlib import Path

from crest.artifacts import digest, read_json, write_json
from crest.calibration import calibrate
from crest.controls import _native_stage, _verify_seal
from crest.experiments import implementation_digest
from crest.metrics import gmax
from crest.rq1_execution import observed_view
from crest.screening import screen
from crest.selection import select


class InvocationLimit(RuntimeError):
    pass


def run_block(unit, ranker, output, deadline):
    _verify_seal(unit, "calibration block")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = unit["config"]
    immutable = {
        "config": config, "ranker": ranker.metadata, "implementation_sha256": implementation_digest(),
        **{role: [{"id": item["id"], "sha256": digest(item)} for item in unit[role]]
           for role in ("calibration", "test")},
    }
    fingerprint = digest(immutable)
    manifest = {"fingerprint": fingerprint, "immutable": immutable}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Run directory belongs to different inputs or model/runtime settings")
    write_json(manifest_path, manifest)

    def rank(decisions, omitted=frozenset()):
        result = []
        for decision in observed_view(decisions):
            if time.monotonic() >= deadline:
                raise InvocationLimit("Invocation budget reached; restart with the same command to resume")
            result.append(ranker.rank(decision, config["K"], omitted))
        return result

    def stage(name, function):
        path = output / "stages" / (digest(name) + ".json")
        if path.exists():
            value = read_json(path)
            if name[0] == "method":
                if (value.get("name") != name or value.get("fingerprint") != fingerprint
                    or value.get("result_sha256") != digest(value.get("result"))
                    or value.get("status") != "complete"
                    or value.get("result", {}).get("status") not in ("ok", "infeasible")):
                    raise ValueError("Saved method stage is incomplete or corrupt")
                return value["result"]
            return _native_stage(output, name, fingerprint)[0]
        if time.monotonic() >= deadline:
            raise InvocationLimit("Invocation budget reached; restart with the same command to resume")
        result = function()
        write_json(path, {"name": name, "fingerprint": fingerprint, "status": "complete",
                          "result": result, "result_sha256": digest(result)})
        return result

    clean, screened = {}, {}
    for item in [*unit["calibration"], *unit["test"]]:
        identity, pair = item["id"], item["pair"]
        clean[identity] = stage(["clean", identity], lambda pair=pair: {"status": "ok", "rankings": rank(pair["clean"])})

        def screen_pair(pair=pair):
            decisions = observed_view(pair["observed"])
            value = screen([[record["id"] for record in decision["records"]] for decision in decisions],
                           lambda omissions: rank(decisions, omissions), config["K"], config["beta"],
                           config["penalty"], config["beam_width"], config["max_depth"])
            return {"status": "ok", **value}

        screened[identity] = stage(["screen", identity], screen_pair)
    scores = [gmax(screened[item["id"]]["rankings"], clean[item["id"]]["rankings"]) for item in unit["calibration"]]
    threshold = stage(["calibration"], lambda: {"status": "ok", "scores": scores,
                                               **calibrate(scores, config["alpha"], config["T"])})
    for item in unit["test"]:
        stage(["method", "CREST", item["id"]], lambda item=item: select(
            screened[item["id"]]["rankings"], [d["candidates"] for d in item["pair"]["observed"]],
            config["K"], config["eta"], threshold["h_alpha"]))
    write_json(output / "status.json", {"status": "complete", "fingerprint": fingerprint})


def main():
    parser = argparse.ArgumentParser(description="Run CREST once per independent RQ3 calibration block; subsequent alpha/eta replay is CPU-only.")
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block", type=int, choices=range(20))
    parser.add_argument("--budget-seconds", type=float, default=3600)
    args = parser.parse_args()
    if not 0 < args.budget_seconds < float("inf"):
        parser.error("--budget-seconds must be finite and positive")
    identity = read_json(args.model / "model_identity.json")
    if (identity.get("status") != "verified" or identity.get("repo_id") != "Qwen/Qwen2.5-7B-Instruct"
        or identity.get("revision") != "a09a35458c702b33eeacc393d103063234e8bc28"):
        raise ValueError("RQ3 requires the pinned Qwen2.5-7B-Instruct checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".controls.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from crest.rq1_last_token_inference import LastTokenSDPARanker
        ranker = LastTokenSDPARanker(args.model, identity["revision"], args.output / "queries.sqlite",
                                    device=args.device, dtype="bfloat16", max_input_tokens=12288, max_new_tokens=256)
        ranker.load()
        deadline = time.monotonic() + args.budget_seconds
        blocks = range(20) if args.block is None else [args.block]
        try:
            for block in blocks:
                run_block(read_json(args.units / f"block-{block:02d}.json"), ranker,
                          args.output / f"block-{block:02d}", deadline)
                print(f"Completed block {block + 1}/20", flush=True)
        except InvocationLimit as error:
            print(str(error), flush=True)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
