from dataclasses import dataclass, field
import hashlib
import json
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .objective import token_loss


@dataclass
class Rollout:
    sample_id: str
    group_id: int
    question: str
    answer: str
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    truncated: bool = False
    old_log_probs: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    top_probs: torch.Tensor | None = None
    step_scores: list[float] = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self):
        return {key: (value.tolist() if isinstance(value, torch.Tensor) else value)
                for key, value in vars(self).items()}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key, dtype in (("old_log_probs", torch.float32), ("top_ids", torch.long),
                           ("top_probs", torch.float32)):
            if data.get(key) is not None:
                data[key] = torch.tensor(data[key], dtype=dtype)
        return cls(**data)


def tokenizer_fingerprint(tokenizer):
    # Includes normalization, segmentation, IDs and special-token handling.
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for auditable boundaries")
    raw = tokenizer.backend_tokenizer.to_str()
    return hashlib.sha256(raw.encode()).hexdigest()


def tokenization_signature(tokenizer):
    """Core segmentation identity, excluding chat templates and added special tokens."""
    raw = json.loads(tokenizer.backend_tokenizer.to_str())
    fields = {k: raw.get(k) for k in ("model", "normalizer", "pre_tokenizer", "decoder")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class Policy:
    """Single PEFT backbone; adapter-disabled inference is the frozen reference.

    Supported architecture contract: decoder at base.model, frozen linear LM head.
    Calling the decoder directly avoids allocating [batch, sequence, vocab] logits.
    """
    def __init__(self, model, tokenizer=None, head_chunk_size=32):
        self.model, self.tokenizer = model, tokenizer
        self.base = model.get_base_model()
        self.decoder = self.base.model
        self.head = self.base.get_output_embeddings()
        self.head_chunk_size = head_chunk_size
        if any(p.requires_grad for p in self.head.parameters()):
            raise ValueError("LM head must remain frozen for shared-reference scoring")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, cfg, adapter_path=None, device="cuda", trainable=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                                  use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                                   torch_dtype=dtype, attn_implementation="sdpa")
        if base.config.model_type not in {"olmo2", "olmo", "llama"}:
            raise ValueError("This backend is tested for OLMo/OLMo2/Llama decoders only")
        if adapter_path:
            model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable)
        else:
            model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                                  r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                  lora_dropout=0.0, target_modules=cfg.lora_targets, bias="none"))
        model.to(device)
        if cfg.gradient_checkpointing and trainable:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        return cls(model, tokenizer, cfg.head_chunk_size)

    def _hidden(self, record):
        if not record.prompt_ids or not record.completion_ids:
            raise ValueError("Empty prompt or completion")
        ids = torch.tensor([record.prompt_ids + record.completion_ids], device=self.device)
        # Each training microbatch contains one unpadded sequence.
        states = self.decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                              use_cache=False, return_dict=True).last_hidden_state[0]
        start = len(record.prompt_ids) - 1
        return states[start:start + len(record.completion_ids)]

    @torch.no_grad()
    def score_rollouts(self, records, top_k):
        self.model.eval()
        for record in records:
            hidden = self._hidden(record)
            log_probs, candidate_ids, candidate_probs = [], [], []
            actions = torch.tensor(record.completion_ids, device=self.device)
            for start in range(0, len(actions), self.head_chunk_size):
                end = start + self.head_chunk_size
                logits = self.head(hidden[start:end]).float()
                lp = F.log_softmax(logits, dim=-1)
                top = lp.topk(top_k, dim=-1)
                log_probs.append(lp.gather(-1, actions[start:end, None]).squeeze(-1).cpu())
                candidate_ids.append(top.indices.cpu())
                candidate_probs.append(top.values.exp().cpu())
            record.old_log_probs = torch.cat(log_probs)
            record.top_ids = torch.cat(candidate_ids)
            record.top_probs = torch.cat(candidate_probs)

    def loss_for_record(self, record, advantages, epsilon, beta, normalizer):
        """Return one microbatch loss; caller backpropagates before next record."""
        if record.old_log_probs is None or advantages.shape != (len(record.completion_ids),):
            raise ValueError("Missing old probabilities or misaligned advantages")
        if normalizer <= 0:
            raise ValueError("normalizer must be positive")
        self.model.eval()
        if beta:
            with self.model.disable_adapter(), torch.no_grad():
                ref_hidden = self._hidden(record).detach()
        else:
            ref_hidden = None
        self.model.train()
        hidden = self._hidden(record)
        actions = torch.tensor(record.completion_ids, device=self.device)
        old = record.old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        loss = hidden.new_zeros((), dtype=torch.float32)
        stats = {"kl_sum": 0.0, "ratio_sum": 0.0, "clip_count": 0.0}
        for start in range(0, len(actions), self.head_chunk_size):
            end = min(start + self.head_chunk_size, len(actions))
            # Explicit arguments avoid late-binding slice bugs during recomputation.
            def calculate(h, rh, a, o, adv):
                logits = self.head(h)
                with torch.no_grad():
                    reference = self.head(rh) if beta else logits.detach()
                losses, _, kl, ratio = token_loss(logits, reference, a, o, adv, epsilon, beta)
                return losses.sum(), kl.detach().sum(), ratio.detach().sum(), (
                    (ratio.detach() - 1).abs() > epsilon).float().sum()
            rh = ref_hidden[start:end] if beta else hidden[start:end].detach()
            total, kl, ratio, clipped = checkpoint(calculate, hidden[start:end], rh,
                actions[start:end], old[start:end], advantages[start:end], use_reentrant=False)
            loss = loss + total / normalizer
            stats["kl_sum"] += kl.item()
            stats["ratio_sum"] += ratio.item()
            stats["clip_count"] += clipped.item()
        return loss, stats

    @torch.no_grad()
    def kl_for_record(self, record):
        from .objective import forward_kl
        self.model.eval()
        with self.model.disable_adapter():
            ref = self._hidden(record)
        current = self._hidden(record)
        values = []
        for start in range(0, len(current), self.head_chunk_size):
            end = start + self.head_chunk_size
            values.append(forward_kl(self.head(current[start:end]), self.head(ref[start:end])).cpu())
        return torch.cat(values)

    def generate(self, examples, cfg, step=0, greedy=False):
        from transformers import GenerationConfig
        self.model.eval()
        rows = []
        for group, example in enumerate(examples):
            text = example["question"] + "\nSolve step by step. End with #### <answer>."
            if self.tokenizer.chat_template:
                ids = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                         add_generation_prompt=True, tokenize=True)
            else:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > cfg.max_prompt_tokens:
                raise ValueError(f"Prompt {example['id']} exceeds max_prompt_tokens; increase limit explicitly")
            if len(ids) + cfg.max_new_tokens > self.base.config.max_position_embeddings:
                raise ValueError("Prompt + response exceed model context window")
            for member in range(1 if greedy else cfg.group_size):
                rows.append((group, member, example, ids))
        records = []
        eos = self.tokenizer.eos_token_id
        generation = GenerationConfig(do_sample=not greedy, max_new_tokens=cfg.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=eos,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True,
            **({"temperature": 1.0, "top_k": 0, "top_p": 1.0} if not greedy else {}))
        for start in range(0, len(rows), cfg.generation_batch_size):
            chunk = rows[start:start + cfg.generation_batch_size]
            batch = self.tokenizer.pad({"input_ids": [row[3] for row in chunk]},
                                       padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self.model.generate(**batch, generation_config=generation)
            for row, full in zip(chunk, output):
                completion = full[batch["input_ids"].shape[1]:].tolist()
                ended = eos in completion if eos is not None else False
                if ended:
                    completion = completion[:completion.index(eos) + 1]
                group, member, example, ids = row
                records.append(Rollout(f"{step}:{example['id']}:{member}", group,
                    example["question"], example["answer"], ids, completion,
                    self.tokenizer.decode(completion, skip_special_tokens=True),
                    truncated=not ended and len(completion) == cfg.max_new_tokens))
        return records
