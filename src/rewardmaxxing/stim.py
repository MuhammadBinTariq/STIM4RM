"""Sparse, capped-context STIM-local. Missing observations remain explicit."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import Counter
import hashlib
import math
import re
import torch

from .cache import HTTPTransport, InfiniGramClient, QueryFailure, SQLiteCache
from .policy import tokenization_signature, tokenizer_fingerprint


RELATIONS = {"remaining", "left", "each", "per", "total", "difference", "twice",
             "half", "more", "less", "altogether", "combined", "times", "divided"}
SPAN = re.compile(r"\d+(?:[.,]\d+)*(?:/\d+)?|[+\-*/=%]|[A-Za-z]+")
# Deliberately small, reviewable whitelist, not a semantic theorem detector.
IDENTITIES = [re.compile(r"\bx\s*\+\s*0\s*=\s*x\b"),
              re.compile(r"\bx\s*\*\s*1\s*=\s*x\b"),
              re.compile(r"\ba\^2\s*\+\s*b\^2\s*=\s*c\^2\b")]


def rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for pos in order[i:j]:
            ranks[pos] = (i + j - 1) / 2
        i = j
    return ranks


def spearman(x, y):
    if len(x) != len(y) or len(x) < 3 or not all(math.isfinite(v) for v in [*x, *y]):
        return None
    x, y = rank(x), rank(y)
    mx, my = sum(x)/len(x), sum(y)/len(y)
    dx, dy = [v-mx for v in x], [v-my for v in y]
    denominator = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return max(-1.0, min(1.0, sum(a*b for a, b in zip(dx, dy))/denominator)) if denominator else None


@dataclass
class Observation:
    sample_id: str
    position: int
    status: str
    rho: float | None = None
    support: int | None = None
    context_length: int | None = None
    error: str | None = None


def select_positions(record, tokenizer, config, audit=False):
    ids = record.completion_ids
    eligible = [t for t, v in enumerate(ids) if v not in tokenizer.all_special_ids]
    if audit:
        # Policy-independent random positions within each trace (not confidence gating).
        order = sorted(eligible, key=lambda t: hashlib.sha256(
            f"audit-v1:{record.sample_id}:{t}".encode()).digest())
        return sorted(order[:config.max_positions])
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if tokenized["input_ids"] != ids:
        # Never guess char->token alignment after a lossy decode/encode round trip.
        return []
    excluded = [(m.start(), m.end()) for pattern in IDENTITIES for m in pattern.finditer(text)]
    spans = [m for m in SPAN.finditer(text) if m.group().lower() in RELATIONS
             or re.search(r"\d|[+\-*/=%]", m.group())]
    candidates = []
    for t in eligible:
        if record.old_log_probs[t].exp().item() < config.min_probability:
            continue
        start, end = tokenized["offset_mapping"][t]
        if start == end or any(start < b and end > a for a, b in excluded):
            continue
        for span_id, span in enumerate(spans):
            if start < span.end() and end > span.start():
                step = text[:start].count("\n")
                candidates.append((record.old_log_probs[t].item(), t, span_id, step))
                break
    candidates.sort(reverse=True)
    chosen, used_spans, used_steps = [], set(), set()
    for diversify in (True, False):
        for _, t, span_id, step in candidates:
            if t in chosen or span_id in used_spans or (diversify and step in used_steps):
                continue
            chosen.append(t)
            used_spans.add(span_id)
            used_steps.add(step)
            if len(chosen) >= config.max_positions:
                return sorted(chosen)
    return sorted(chosen)


class StimScorer:
    def __init__(self, config, policy_tokenizer, index_tokenizer, client):
        self.config, self.policy_tokenizer = config, policy_tokenizer
        self.index_tokenizer, self.client = index_tokenizer, client
        if config.mode == "retokenized" and not config.acknowledge_proxy:
            raise ValueError("Retokenized proxy requires explicit acknowledgement")
        if config.mode == "native" and tokenization_signature(policy_tokenizer) != tokenization_signature(index_tokenizer):
            raise ValueError("Native STIM requires identical segmentation and vocabulary")
        self.executor = ThreadPoolExecutor(max_workers=config.workers)

    def _map(self, record, position):
        candidates = record.top_ids[position].tolist()
        if self.config.mode == "native":
            context = record.completion_ids[max(0, position-self.config.max_context):position]
            return context, [record.completion_ids[position]], [[v] for v in candidates]
        tok, index_tok = self.policy_tokenizer, self.index_tokenizer
        prefix_ids = record.completion_ids[:position]
        if any(v in tok.all_special_ids for v in prefix_ids):
            return None
        prefix = tok.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        context_ids = index_tok.encode(prefix, add_special_tokens=False)
        if index_tok.decode(context_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != prefix:
            return None
        tails = []
        for v in [record.completion_ids[position], *candidates]:
            if v in tok.all_special_ids:
                return None
            full = tok.decode(prefix_ids + [v], skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
            if not full.startswith(prefix) or full == prefix:
                return None
            encoded = index_tok.encode(full, add_special_tokens=False)
            if encoded[:len(context_ids)] != context_ids:
                return None  # Candidate merges with prefix under index tokenization.
            if index_tok.decode(encoded, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != full:
                return None
            tail = encoded[len(context_ids):]
            if not tail:
                return None
            tails.append(tail)
        if len({tuple(t) for t in tails[1:]}) != len(candidates):
            return None
        return context_ids[-self.config.max_context:], tails[0], tails[1:]

    def score_position(self, record, position):
        def observation(status, **kw):
            return Observation(record.sample_id, position, status, **kw)
        mapped = self._map(record, position)
        if mapped is None:
            return observation("incompatible_boundary")
        context, target, alternatives = mapped
        if not context:
            return observation("no_context")
        try:
            if self.client.count(target) == 0:
                return observation("unseen_target")
            # Exact counts on nested suffix sequences are monotone.
            low, high = 0, len(context)
            while low < high:
                mid = (low + high + 1) // 2
                if self.client.count(context[-mid:] + target) > 0:
                    low = mid
                else:
                    high = mid - 1
            if low == 0:
                return observation("no_supported_context")
            window = context[-low:]
            support = self.client.count(window)
            if support < self.config.min_support:
                return observation("low_support", support=support, context_length=low)
            counts = [self.client.count(window + tail) for tail in alternatives]
            if len(set(counts)) < self.config.min_distinct_counts:
                return observation("tied_counts", support=support, context_length=low)
            rho = spearman(record.top_probs[position].tolist(), counts)
            if rho is None:
                return observation("undefined_rho", support=support, context_length=low)
            return observation("ok", rho=rho, support=support, context_length=low)
        except QueryFailure as exc:
            return observation("query_failure", error=str(exc)[:500])

    def score(self, records, audit=False, positions=None):
        length = max(len(r.completion_ids) for r in records)
        penalty = torch.zeros((len(records), length), dtype=torch.float32)
        futures = []
        for row, record in enumerate(records):
            selected = positions[row] if positions is not None else select_positions(
                record, self.policy_tokenizer, self.config, audit)
            for pos in selected:
                futures.append((row, self.executor.submit(self.score_position, record, pos)))
        observations = []
        for row, future in futures:
            result = future.result()
            observations.append(result)
            if result.status == "ok":
                penalty[row, result.position] = max(0.0, result.rho)
        failures = sum(o.status == "query_failure" for o in observations)
        if failures / max(1, len(observations)) > self.config.max_failure_fraction:
            causes = Counter(o.error for o in observations if o.status == "query_failure")
            detail = "; ".join(f"{n} position(s): {message}" for message,n in causes.most_common(3))
            error = QueryFailure(f"STIM incomplete: {failures}/{len(observations)} selected token positions failed. "
                                 f"Resume pending batch. Causes: {detail}")
            error.observations = [vars(o) for o in observations]
            raise error
        return penalty, observations

    def close(self):
        self.executor.shutdown(wait=True)
        if self.config.cache_backup:
            self.client.cache.snapshot(self.config.cache_backup)
        self.client.close()


def build_scorer(config, policy_tokenizer):
    from transformers import AutoTokenizer
    # Public index->tokenizer contracts; unknown/private indexes require an extension.
    if config.index.endswith("_llama") or config.index == "v4_rpj_llama_s4":
        expected = "hf-internal-testing/llama-tokenizer"
    elif config.index == "v4_dolmasample_olmo":
        expected = "allenai/OLMo-7B"
    else:
        raise ValueError("Unregistered index: establish corpus and tokenizer contract first")
    if config.index_tokenizer != expected:
        raise ValueError(f"Index requires tokenizer {expected}")
    tokenizer = AutoTokenizer.from_pretrained(config.index_tokenizer,
        revision=config.index_tokenizer_revision, use_fast=True)
    cache = SQLiteCache(config.cache_path)
    if config.cache_backup:
        cache.restore_if_empty(config.cache_backup)
    client = InfiniGramClient(cache, config.index, tokenizer_fingerprint(tokenizer),
                             HTTPTransport(config.timeout, config.attempts))
    try:
        return StimScorer(config, policy_tokenizer, tokenizer, client)
    except Exception:
        client.close()
        raise
