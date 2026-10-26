"""Greedy held-out evaluation and paired, template-clustered comparisons."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .rewardmaxxing.checkpoints import atomic_json, seed_all
from .rewardmaxxing.config import TrainConfig
from .rewardmaxxing.data import load_gsm8k, question_hash
from .rewardmaxxing.metrics import TrainingOverlap, paired_comparison, summarize, mean
from .rewardmaxxing.policy import Policy
from .rewardmaxxing.rewards import VersaReward, exact_match, parse_answer
from .rewardmaxxing.stim import build_scorer


def load_twins(path, originals):
    """Accept official GSM-Symbolic JSONL / GSM-Plus JSON(L), never guess new labels."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    lookup = {question_hash(r["question"]):r for r in originals}
    result = []
    for number, row in enumerate(data):
        source = row.get("original_question", row.get("seed_question"))
        if source is None or question_hash(source) not in lookup:
            raise ValueError("Twin missing a verified original GSM8K test question")
        parent = lookup[question_hash(source)]
        if parse_answer(row["answer"], require_marker=False) is None:
            raise ValueError("Twin has no valid recomputed reference answer")
        family = "gsm_symbolic" if "original_question" in row else "gsm_plus"
        category = row.get("perturbation_type", row.get("subset", "symbolic"))
        result.append({"id":f"{family}:{parent['id']}:{row.get('instance', number)}:{category}",
            "question":row["question"], "answer":row["answer"], "original_id":parent["id"],
            "split": f"{family}/{category}"})
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate twin IDs")
    return result


def run(args):
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    cfg = TrainConfig.load(checkpoint/"config.json" if checkpoint else args.config)
    seed_all(cfg.seed)
    train, dev, test, manifest = load_gsm8k(cfg)
    if checkpoint:
        stored_manifest = json.loads((checkpoint/"data_manifest.json").read_text())
        if stored_manifest != manifest:
            raise ValueError("Evaluation data manifest differs from training checkpoint")
    source = test if args.split == "test" else dev
    examples = [{**r, "original_id":r["id"], "split":"gsm8k"} for r in source]
    if args.limit:
        examples = examples[:args.limit]
    if args.twins:
        if args.split != "test":
            raise ValueError("Public test-derived twins cannot enter development evaluation")
        allowed = {r["id"] for r in examples}
        for path in args.twins:
            examples.extend(r for r in load_twins(path, test) if r["original_id"] in allowed)
    if len({e["id"] for e in examples}) != len(examples):
        raise ValueError("Evaluation IDs overlap across input files")
    if args.paired_only:
        if not args.twins:
            raise ValueError("paired-only requires twins")
        parents = {e["original_id"] for e in examples if e["split"] != "gsm8k"}
        examples = [e for e in examples if e["original_id"] in parents]
        if not examples:
            raise ValueError("No paired originals/twins selected")
    evaluation_cfg = replace(cfg, max_new_tokens=args.max_new_tokens)
    policy = Policy.load(cfg, checkpoint/"adapter" if checkpoint else None, args.device, trainable=False)
    reward = None if args.skip_prm else VersaReward(cfg, args.device)
    audit_cfg = replace(cfg.stim, top_k=20, max_positions=args.audit_positions, min_probability=0.0)
    scorer = build_scorer(audit_cfg, policy.tokenizer) if args.stim_audit else None
    overlap = TrainingOverlap([r["answer"] for r in train])
    metadata = {"seed":cfg.seed, "model_id":cfg.model_id, "model_revision":cfg.model_revision,
        "dataset_revision":cfg.dataset_revision, "eval_max_new_tokens":args.max_new_tokens,
        "training_manifest_hash":hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        "evaluation_input_hash":hashlib.sha256(json.dumps(examples,sort_keys=True).encode()).hexdigest(),
        "prm_id":cfg.prm_id, "prm_revision":cfg.prm_revision, "prm_base_revision":cfg.prm_base_revision,
        "training_protocol":replace(cfg, stim_lambda=0.0).fingerprint(),
        "checkpoint_name":checkpoint.name if checkpoint else "initial",
        "reward":cfg.reward, "lambda":cfg.stim_lambda if checkpoint else None,
        "checkpoint":str(checkpoint) if checkpoint else "initial",
        "split":args.split, "stim_audit":vars(audit_cfg) if args.stim_audit else None,
        "metric_tokenizer":"lowercase regex words + arithmetic operators; no stemming",
        "decoding":"greedy", "prm_quantized":cfg.quantize_prm, "prm_scored":not args.skip_prm}
    # Cache locations do not define the audit metric.
    if metadata["stim_audit"]:
        metadata["stim_audit"] = {k:v for k,v in metadata["stim_audit"].items()
                                  if k not in {"cache_path","cache_backup"}}
    output = Path(args.output)
    partial = output.with_suffix(".partial.json")
    rows = []
    if output.exists():
        raise FileExistsError("Evaluation output exists; choose a new path")
    if partial.exists():
        prior = json.loads(partial.read_text())
        if prior["metadata"] != metadata or prior["example_ids"] != [e["id"] for e in examples]:
            raise ValueError("Partial evaluation configuration/input set differs")
        rows = prior["rows"]
    try:
        for example in examples[len(rows):]:
            record = policy.generate([example], evaluation_cfg, greedy=True)[0]
            # Identical identifiers across checkpoints for independent audit position sampling.
            record.sample_id = example["id"]
            policy.score_rollouts([record], audit_cfg.top_k)
            if reward:
                reward.score([record])
            observations = scorer.score([record], audit=True)[1] if scorer else []
            values = [o.rho for o in observations if o.status == "ok"]
            row = {"id":example["id"], "original_id":example["original_id"], "split":example["split"],
                "question":example["question"], "reference":example["answer"], "response":record.text,
                "correct":int(exact_match(record.text, record.answer)),
                "format_valid":int(parse_answer(record.text) is not None),
                "prm_reward":record.reward if reward else None, "prm_steps":record.step_scores,
                "kl":policy.kl_for_record(record).mean().item(),
                "length":len(record.completion_ids), "truncated":int(record.truncated),
                "stim_rho":mean(values), "stim_positive":mean([max(0,v) for v in values]),
                "stim_valid_fraction":len(values)/len(observations) if observations else None,
                "stim_status":dict(Counter(o.status for o in observations)),
                "stim_observations":[vars(o) for o in observations],
                **overlap.score(record.text, record.question)}
            rows.append(row)
            atomic_json(partial, {"metadata":metadata, "example_ids":[e["id"] for e in examples], "rows":rows})
            print(f"Evaluated {len(rows)}/{len(examples)}", flush=True)
        atomic_json(output, {"metadata":metadata, "summary":summarize(rows), "rows":rows})
        partial.unlink(missing_ok=True)
    finally:
        if scorer:
            scorer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("run")
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--config", help="Evaluate the untouched initial policy")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--split", choices=["test","dev"], default="test")
    evaluate.add_argument("--twins", nargs="*")
    evaluate.add_argument("--paired-only", action="store_true", help="Evaluate only supplied twins and their GSM8K parents")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument("--stim-audit", action="store_true")
    evaluate.add_argument("--audit-positions", type=int, default=8)
    evaluate.add_argument("--skip-prm", action="store_true", help="Explicitly omit expensive PRM diagnostics")
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", nargs="+", required=True)
    compare.add_argument("--rewardmaxxing", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap", type=int, default=2000)
    score = commands.add_parser("score-prm", help="PRM-only process over saved generations; no policy in VRAM")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--config", required=True, help="Pinned run config.json; checkpoint weights are not needed")
    score.add_argument("--device", default="cuda")
    score.add_argument("--unquantized", action="store_true", help="Explicit PRM calibration pass; requires more memory")
    curve = commands.add_parser("frontier", help="Aggregate checkpoint proxy/gold reward points")
    curve.add_argument("--inputs", nargs="+", required=True)
    curve.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        if args.audit_positions <= 0 or args.max_new_tokens <= 0:
            raise ValueError("Evaluation lengths must be positive")
        run(args)
    elif args.command == "compare":
        baseline = [json.loads(Path(p).read_text()) for p in args.baseline]
        treatment = [json.loads(Path(p).read_text()) for p in args.rewardmaxxing]
        atomic_json(args.output, paired_comparison(baseline, treatment, args.bootstrap))
    elif args.command == "score-prm":
        from .rewardmaxxing.posthoc import score_prm_file
        score_prm_file(args.input, args.output, args.config, args.device, args.unquantized)
    else:
        from .rewardmaxxing.posthoc import frontier
        atomic_json(args.output, frontier(args.inputs))


if __name__ == "__main__":
    main()
