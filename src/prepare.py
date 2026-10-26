"""Resolve every Hub revision once, before sharing configs across experimental arms."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

from .rewardmaxxing.checkpoints import atomic_json
from .rewardmaxxing.config import TrainConfig


def resolve(cfg):
    api = HfApi()
    cfg.model_revision = api.model_info(cfg.model_id, revision=cfg.model_revision).sha
    cfg.prm_revision = api.model_info(cfg.prm_id, revision=cfg.prm_revision).sha
    adapter = json.loads(Path(hf_hub_download(cfg.prm_id, "adapter_config.json",
                                            revision=cfg.prm_revision)).read_text())
    cfg.prm_base_revision = api.model_info(adapter["base_model_name_or_path"],
                                          revision=cfg.prm_base_revision).sha
    cfg.dataset_revision = api.dataset_info("openai/gsm8k", revision=cfg.dataset_revision).sha
    cfg.stim.index_tokenizer_revision = api.model_info(cfg.stim.index_tokenizer,
                                           revision=cfg.stim.index_tokenizer_revision).sha
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Resolved configuration exists; reuse it or select another output")
    cfg = resolve(TrainConfig.load(args.config))
    atomic_json(args.output, cfg.to_dict())
    print(f"Pinned revisions written to {args.output}")


if __name__ == "__main__":
    main()
