from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import torch

from .checkpoints import atomic_json, rng_state, restore_rng, save_checkpoint
from .objective import group_advantages, shape_advantages
from .policy import Rollout
from .rewards import exact_match


def update(policy, records, penalty, optimizer, cfg):
    lengths = [len(r.completion_ids) for r in records]
    if penalty.shape != (len(records), max(lengths)):
        raise ValueError("Penalty tensor must align exactly with rollout token positions")
    if len({r.sample_id for r in records}) != len(records):
        raise ValueError("Duplicate rollout ID")
    groups = torch.tensor([r.group_id for r in records])
    for group in groups.unique():
        if (groups == group).sum() != cfg.group_size:
            raise ValueError("Incorrect rollout group cardinality")
    rewards = torch.tensor([r.reward for r in records], dtype=torch.float32)
    advantages = group_advantages(rewards, groups, cfg.advantage_floor)
    shaped = shape_advantages(advantages, penalty, cfg.stim_lambda, cfg.penalty)
    normalizer = sum(lengths)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    sums = {"kl_sum": 0.0, "ratio_sum": 0.0, "clip_count": 0.0}
    try:
        for row, record in enumerate(records):
            loss, stats = policy.loss_for_record(record, shaped[row, :lengths[row]],
                                                cfg.clip_epsilon, cfg.beta, normalizer)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy loss")
            loss.backward()
            total_loss += loss.detach().item()
            for key in sums:
                sums[key] += stats[key]
        parameters = [p for p in policy.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {"loss": total_loss, "gradient_norm": grad_norm.item(),
            "kl": sums["kl_sum"] / normalizer, "ratio": sums["ratio_sum"] / normalizer,
            "clip_fraction": sums["clip_count"] / normalizer,
            "task_reward": rewards.mean().item(), "advantage_abs": advantages.abs().mean().item(),
            "penalty_mean_all_tokens": sum(penalty[i, :n].sum().item() for i, n in enumerate(lengths))/normalizer,
            "correct": sum(exact_match(r.text, r.answer) for r in records)/len(records),
            "mean_length": normalizer/len(records),
            "truncated_fraction": sum(r.truncated for r in records)/len(records), "tokens": normalizer}


def save_pending(path, step, cfg, records):
    temporary = Path(str(path) + ".part")
    torch.save({"step": step, "fingerprint": cfg.fingerprint(),
                "records": [r.to_dict() for r in records], "rng": rng_state()}, temporary)
    os.replace(temporary, path)


def train(policy, reward_model, scorer, optimizer, cfg, batches, manifest, start_step=0):
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "pending.pt"
    limit = min(len(batches), cfg.max_steps or len(batches))
    last_saved = start_step
    if pending.exists():
        prior = torch.load(pending, weights_only=False, map_location="cpu")
        if prior["fingerprint"] != cfg.fingerprint():
            raise ValueError("Pending batch belongs to a different configuration")
        if prior["step"] > start_step:
            # Hard interruption between periodic checkpoints: replay from saved RNG
            # instead of applying a newer rollout to an older policy.
            pending.unlink()
    if start_step == 0 and not (output / "step-000000").exists():
        save_checkpoint(output, policy, optimizer, cfg, 0, manifest)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for step in range(start_step, limit):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            observations = []
            cache_before = dict(scorer.client.stats) if scorer else {}
            try:
                if pending.exists():
                    state = torch.load(pending, weights_only=False, map_location="cpu")
                    if state["step"] == step and state["fingerprint"] == cfg.fingerprint():
                        records = [Rollout.from_dict(r) for r in state["records"]]
                        restore_rng(state["rng"])
                    elif state["step"] < step:
                        pending.unlink()  # A completed batch after crash between save/unlink.
                        records = None
                    else:
                        raise ValueError("Pending batch does not match resumed checkpoint")
                else:
                    records = None
                if records is None:
                    records = policy.generate(batches[step], cfg, step)
                    policy.score_rollouts(records, cfg.stim.top_k)
                    # CPU HTTP scoring overlaps the frozen GPU PRM pass.
                    scoring = executor.submit(scorer.score, records) if scorer else None
                    reward_model.score(records)
                    save_pending(pending, step, cfg, records)
                else:
                    scoring = executor.submit(scorer.score, records) if scorer else None
                scored_at = time.perf_counter()
                if scoring:
                    penalty, observations = scoring.result()
                else:
                    penalty = torch.zeros((len(records), max(len(r.completion_ids) for r in records)))
                counts = Counter(o.status for o in observations)
                if scorer and not counts["ok"] and cfg.stim_lambda:
                    raise ValueError(f"No valid STIM observations in batch: {dict(counts)}. Inspect coverage before training.")
            except Exception as exc:
                # No update yet: persist the exact policy required by pending old logps.
                if step != last_saved:
                    save_checkpoint(output, policy, optimizer, cfg, step, manifest)
                    last_saved = step
                if scorer and cfg.stim.cache_backup:
                    scorer.client.cache.snapshot(cfg.stim.cache_backup)
                atomic_json(output / "failures" / f"batch-{step:06d}-{time.time_ns()}.json", {
                    "completed_updates":step, "error_type":type(exc).__name__, "error":str(exc),
                    "stim_observations":getattr(exc, "observations", []),
                    "cache":{k:scorer.client.stats[k]-cache_before[k] for k in cache_before} if scorer else {},
                    "pending_saved":pending.exists()})
                raise
            update_started = time.perf_counter()
            metrics = update(policy, records, penalty, optimizer, cfg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            finished = time.perf_counter()
            successful = [o.rho for o in observations if o.status == "ok"]
            metrics.update(step=step+1, elapsed_seconds=finished-started,
                rollout_prm_seconds=scored_at-started, stim_wait_seconds=update_started-scored_at,
                update_seconds=finished-update_started, stim_status=dict(counts),
                stim_valid_fraction=len(successful)/len(observations) if observations else None,
                stim_rho=sum(successful)/len(successful) if successful else None,
                stim_positive=sum(max(0, x) for x in successful)/len(successful) if successful else None,
                cache={k: scorer.client.stats[k]-cache_before[k] for k in cache_before} if scorer else {},
                cuda_peak_gb=torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0,
                cuda_allocated_gb=torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0,
                cuda_reserved_gb=torch.cuda.memory_reserved()/1e9 if torch.cuda.is_available() else 0,
                cuda_peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9 if torch.cuda.is_available() else 0)
            # Persist token IDs and coverage, so the training trajectory can be audited.
            atomic_json(output / "rollouts" / f"step-{step+1:06d}.json", {
                "records": [r.to_dict() for r in records],
                "stim": [vars(o) for o in observations]})
            if (step+1) % cfg.save_every == 0 or step+1 == limit:
                save_checkpoint(output, policy, optimizer, cfg, step+1, manifest)
                last_saved = step+1
                if scorer and cfg.stim.cache_backup:
                    scorer.client.cache.snapshot(cfg.stim.cache_backup)
            pending.unlink(missing_ok=True)
            metrics["persistence_seconds"] = time.perf_counter() - finished
            metrics["elapsed_seconds"] = time.perf_counter() - started
            metrics["unix_time"] = time.time()
            cache = metrics["cache"]
            lookups = sum(cache.get(k, 0) for k in ("hits", "misses", "shared"))
            metrics["cache_hit_rate"] = cache.get("hits", 0)/lookups if lookups else None
            metrics["request_mean_seconds"] = (cache.get("request_seconds", 0)/cache["misses"]
                                                if cache.get("misses") else None)
            atomic_json(output / "metrics" / f"step-{step+1:06d}.json", metrics)
            print(json.dumps(metrics, allow_nan=False), flush=True)
    return limit
