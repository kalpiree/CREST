import argparse
import json
from pathlib import Path

from .artifacts import read_json, write_json
from .data import prepare_steam


def main():
    parser = argparse.ArgumentParser(prog="python -m crest")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-steam")
    prep.add_argument("--games", required=True)
    prep.add_argument("--reviews", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--max-reviews", type=int, default=100000)
    prep.add_argument("--min-history", type=int, default=1)
    prep.add_argument("--max-examples", type=int, default=1000)
    prep.add_argument("--text-chars", type=int, default=240)
    pilot = commands.add_parser("run-pilot")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--pool", required=True)
    pilot.add_argument("--model", required=True)
    pilot.add_argument("--revision", required=True)
    pilot.add_argument("--device", default="cuda:0")
    pilot.add_argument("--cache", required=True)
    pilot.add_argument("--output", required=True)
    smoke = commands.add_parser("model-smoke")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--revision", required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--cache", required=True)
    smoke.add_argument("--output", required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare-steam":
        result = prepare_steam(args.games, args.reviews, args.output, args.max_reviews, args.min_history, args.max_examples, args.text_chars)
    elif args.command == "status":
        result = read_json(Path(args.output) / "status.json")
    else:
        from .inference import LocalRanker

        if args.command == "run-pilot":
            from .experiments import run_pilot

            config = read_json(args.config)
            ranker = LocalRanker(args.model, args.revision, args.cache, args.device, config["dtype"], config["max_input_tokens"], config["max_new_tokens"])
            result = run_pilot(config, read_json(args.pool), ranker, args.output)
        else:
            ranker = LocalRanker(args.model, args.revision, args.cache, args.device)
            decision = {"user_id": "smoke-user", "candidates": ["book-a", "game-b", "film-c"], "records": [
                {"id": "history/smoke", "type": "interaction", "fields": {"preference": "Enjoys strategy games"}},
                {"id": "item/book-a", "type": "item_text", "fields": {"item_id": "book-a", "description": "A gardening book"}},
                {"id": "item/game-b", "type": "item_text", "fields": {"item_id": "game-b", "description": "A turn-based strategy game"}},
                {"id": "item/film-c", "type": "item_text", "fields": {"item_id": "film-c", "description": "A comedy film"}},
            ]}
            first = ranker.rank(decision, 2)
            second = ranker.rank(decision, 2)
            if first != second:
                raise RuntimeError("Cached ranking changed")
            result = {"status": "smoke_only", "ranking": first, "statistics": ranker.statistics()}
            write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
