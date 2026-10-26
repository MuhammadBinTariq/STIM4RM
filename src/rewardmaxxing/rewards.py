from fractions import Fraction
import math
import re
import torch


NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
NUMERIC = re.compile(rf"({NUMBER})(?:\s*/\s*({NUMBER}))?")


def parse_answer(text, require_marker=True):
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    elif require_marker:
        return None
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    match = NUMERIC.fullmatch(text)
    if match is None:
        return None
    try:
        numerator = Fraction(match[1].replace(",", ""))
        denominator = Fraction(match[2].replace(",", "")) if match[2] else 1
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return None


def exact_match(prediction, reference):
    actual = parse_answer(prediction)
    expected = parse_answer(reference, require_marker=False)
    if expected is None:
        raise ValueError(f"Unparseable reference answer: {reference!r}")
    return actual is not None and actual == expected


class OutcomeReward:
    """Explicit ablation; never silently substituted for VersaPRM."""
    def score(self, records):
        for record in records:
            record.reward = float(exact_match(record.text, record.answer))
            record.step_scores = []


class VersaReward:
    def __init__(self, cfg, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftConfig, PeftModel
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.prm_id, revision=cfg.prm_revision)
        adapter = PeftConfig.from_pretrained(cfg.prm_id, revision=cfg.prm_revision)
        kwargs = {"torch_dtype": torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                  else torch.float32}
        if cfg.quantize_prm:
            if not device.startswith("cuda"):
                raise ValueError("4-bit VersaPRM requires CUDA; use unquantized CPU explicitly")
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=kwargs["torch_dtype"]), device_map={"": device})
        base = AutoModelForCausalLM.from_pretrained(adapter.base_model_name_or_path,
                                                   revision=cfg.prm_base_revision, **kwargs)
        self.model = PeftModel.from_pretrained(base, cfg.prm_id, revision=cfg.prm_revision)
        if not cfg.quantize_prm:
            self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.separator = " \n\n\n\n"
        self.separator_id = 23535
        if self.tokenizer.encode(self.separator, add_special_tokens=False) != [self.separator_id]:
            raise ValueError("VersaPRM delimiter contract changed")
        if [self.tokenizer.encode(v, add_special_tokens=False) for v in ["-", "+"]] != [[12], [10]]:
            raise ValueError("VersaPRM label token contract changed")

    @torch.inference_mode()
    def score(self, records):
        for record in records:
            steps = [s.strip() for s in record.text.splitlines() if s.strip()]
            if not steps:
                record.step_scores, record.reward = [], 0.0
                continue
            text = record.question + " \n\n" + self.separator.join(steps) + self.separator
            ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)
            base = self.model.get_base_model()
            if ids.shape[1] > base.config.max_position_embeddings:
                raise ValueError("PRM context overflow; refusing silent truncation")
            # Injected LoRA layers remain active when calling the decoder directly.
            hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, return_dict=True).last_hidden_state[0]
            selected = hidden[ids[0] == self.separator_id]
            if selected.shape[0] != len(steps):
                raise ValueError("PRM delimiter count does not match reasoning-step count")
            head = base.get_output_embeddings()
            # Only project onto +/- labels; no full vocabulary tensor is needed.
            logits = torch.nn.functional.linear(selected, head.weight[[12, 10]],
                        head.bias[[12, 10]] if head.bias is not None else None)
            values = logits.float().softmax(-1)[:, 1].cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite PRM reward")
            record.step_scores = values
            record.reward = sum(values) / len(values)
