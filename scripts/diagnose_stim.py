"""Inspect a pending batch's API errors without loading policy/PRM weights or updating."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-positions", type=int, default=8)
    args = parser.parse_args()
    if args.max_positions < 1:
        raise ValueError("max-positions must be positive")
    import torch
    from transformers import AutoTokenizer
    from src.rewardmaxxing.config import TrainConfig
    from src.rewardmaxxing.policy import Rollout
    from src.rewardmaxxing.stim import build_scorer, select_positions
    run = Path(args.run_dir)
    cfg = TrainConfig.load(run/"runner-config.json")
    # Trusted pending file from this run; no policy/PRM weights are loaded.
    pending = torch.load(run/"pending.pt", map_location="cpu", weights_only=False)
    if pending["fingerprint"] != cfg.fingerprint():
        raise ValueError("Pending batch does not match run configuration")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision, use_fast=True)
    scorer = build_scorer(cfg.stim, tokenizer)
    observations = []
    try:
        for data in pending["records"]:
            record = Rollout.from_dict(data)
            for position in select_positions(record,tokenizer,cfg.stim):
                item = scorer.score_position(record,position)
                observations.append(vars(item))
                print(json.dumps(vars(item)), flush=True)
                if item.status == "query_failure" or len(observations) >= args.max_positions:
                    return
    finally:
        print(json.dumps({"pending_batch":pending["step"],
            "status":dict(Counter(o["status"] for o in observations)), "cache":scorer.client.stats}), flush=True)
        scorer.close()


if __name__ == "__main__":
    main()
