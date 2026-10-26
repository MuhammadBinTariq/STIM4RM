"""Usage: python -m src.train --config configs/rewardmaxxing.resolved.json"""
import argparse
import json
from pathlib import Path
import torch

from .rewardmaxxing.checkpoints import atomic_json, load_training_state, seed_all
from .rewardmaxxing.config import TrainConfig
from .rewardmaxxing.data import batches, load_gsm8k
from .rewardmaxxing.policy import Policy
from .rewardmaxxing.rewards import OutcomeReward, VersaReward
from .rewardmaxxing.stim import build_scorer
from .rewardmaxxing.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Checkpoint directory; only use trusted local checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monitor-stim", action="store_true",
                        help="Measure STIM in the lambda=0 arm without changing its loss")
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config)
    for revision in (cfg.model_revision, cfg.dataset_revision, cfg.prm_revision,
                     cfg.prm_base_revision, cfg.stim.index_tokenizer_revision):
        if revision == "main":
            raise ValueError("Resolve revisions with python -m src.prepare before training")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. CPU tensor/gradient checks: python tests/smoke_test.py")
    output = Path(cfg.output_dir)
    if not args.resume and (output / "latest.json").exists():
        raise FileExistsError("Run already exists. Resume explicitly or choose a new output directory")
    if args.resume:
        latest = json.loads((output / "latest.json").read_text())["checkpoint"]
        if (output / latest).resolve() != Path(args.resume).resolve():
            raise ValueError("Resume must use this output directory's latest checkpoint")
    seed_all(cfg.seed)
    examples, _, _, manifest = load_gsm8k(cfg)
    if args.resume:
        previous = json.loads((Path(args.resume) / "data_manifest.json").read_text())
        if previous != manifest:
            raise ValueError("Dataset/split manifest changed during resume")
    policy = Policy.load(cfg, Path(args.resume)/"adapter" if args.resume else None, args.device)
    optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=0.0)
    reward = VersaReward(cfg, args.device) if cfg.reward == "versa" else OutcomeReward()
    scorer = build_scorer(cfg.stim, policy.tokenizer) if cfg.stim_lambda > 0 or args.monitor_stim else None
    try:
        start = load_training_state(args.resume, optimizer, cfg) if args.resume else 0
        atomic_json(output / "config.json", cfg.to_dict())
        atomic_json(output / "data_manifest.json", manifest)
        train(policy, reward, scorer, optimizer, cfg, batches(examples, cfg), manifest, start)
    finally:
        if scorer:
            scorer.close()


if __name__ == "__main__":
    main()
