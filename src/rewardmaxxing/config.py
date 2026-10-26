from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class StimConfig:
    index: str = "v4_olmo-mix-1124_llama"
    index_tokenizer: str = "hf-internal-testing/llama-tokenizer"
    index_tokenizer_revision: str = "main"
    mode: str = "retokenized"
    acknowledge_proxy: bool = False
    corpus_scope: str = "OLMo-mix-1124 stage-one corpus; excludes later training"
    cache_path: str = "cache/infinigram.sqlite"
    cache_backup: str | None = None
    top_k: int = 8
    max_positions: int = 4
    max_context: int = 32
    min_probability: float = 0.3
    min_support: int = 5
    min_distinct_counts: int = 3
    workers: int = 4
    attempts: int = 3
    timeout: float = 10.0
    max_failure_fraction: float = 0.0


@dataclass
class TrainConfig:
    model_id: str = "allenai/OLMo-2-0425-1B-Instruct"
    model_revision: str = "main"
    dataset_revision: str = "main"
    prm_id: str = "UW-Madison-Lee-Lab/VersaPRM"
    prm_revision: str = "main"
    prm_base_revision: str = "main"
    reward: str = "versa"
    quantize_prm: bool = True
    seed: int = 42
    split_seed: int = 1729
    train_size: int = 512
    dev_size: int = 128
    epochs: int = 1
    prompts_per_batch: int = 2
    group_size: int = 4
    generation_batch_size: int = 2
    max_prompt_tokens: int = 768
    max_new_tokens: int = 256
    learning_rate: float = 2e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True
    head_chunk_size: int = 32
    clip_epsilon: float = 0.2
    beta: float = 0.04
    advantage_floor: float = 0.05
    stim_lambda: float = 0.1
    penalty: str = "additive"
    max_grad_norm: float = 1.0
    save_every: int = 20
    keep_checkpoints: int = 2
    max_steps: int | None = None
    output_dir: str = "runs/rewardmaxxing"
    stim: StimConfig = field(default_factory=StimConfig)

    def validate(self):
        if self.group_size < 2:
            raise ValueError("GRPO requires at least two responses per prompt")
        for key in ("prompts_per_batch", "generation_batch_size", "max_new_tokens",
                    "max_prompt_tokens", "train_size", "dev_size", "epochs",
                    "head_chunk_size", "save_every", "keep_checkpoints", "lora_rank"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reward not in {"versa", "outcome"}:
            raise ValueError("reward must be versa or outcome (separate ablation)")
        if self.penalty not in {"additive", "multiplicative"}:
            raise ValueError("Unknown penalty")
        if self.stim_lambda < 0 or self.beta < 0 or self.advantage_floor <= 0:
            raise ValueError("Invalid objective coefficients")
        if self.penalty == "multiplicative" and self.stim_lambda > 1:
            raise ValueError("Multiplicative lambda must be <= 1")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0,1)")
        if self.stim.mode not in {"native", "retokenized"}:
            raise ValueError("Unknown tokenizer compatibility mode")
        if self.stim_lambda and self.stim.mode == "retokenized" and not self.stim.acknowledge_proxy:
            raise ValueError("Set acknowledge_proxy=true to explicitly accept retokenized, partial-corpus STIM")
        if not 3 <= self.stim.top_k <= 20 or self.stim.max_context < 1:
            raise ValueError("Invalid STIM candidate/context budget")
        if self.stim.max_positions < 1 or self.stim.workers < 1 or self.stim.attempts < 1:
            raise ValueError("Invalid STIM request budget")
        if not 0 <= self.stim.min_probability <= 1 or not 0 <= self.stim.max_failure_fraction <= 1:
            raise ValueError("Invalid STIM threshold")
        if self.stim.min_support < 1 or not 2 <= self.stim.min_distinct_counts <= self.stim.top_k:
            raise ValueError("Invalid frequency support thresholds")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["stim"] = StimConfig(**data.get("stim", {}))
        return cls(**data).validate()

    def fingerprint(self):
        data = self.to_dict()
        # These affect storage/stopping, not the stochastic training trajectory.
        for key in ("output_dir", "max_steps", "save_every", "keep_checkpoints"):
            data.pop(key)
        data["stim"].pop("cache_path")
        data["stim"].pop("cache_backup")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
