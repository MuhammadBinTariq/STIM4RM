"""PRM-only scoring of immutable saved responses; never loads the policy."""
import hashlib
import json
from pathlib import Path

from .checkpoints import atomic_json
from .config import TrainConfig
from .metrics import summarize
from .policy import Rollout
from .rewards import VersaReward, exact_match


def score_prm_file(input_path, output_path, config_path, device="cuda", unquantized=False,
                   reward_factory=VersaReward):
    source, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    payload = source.read_bytes()
    original = json.loads(payload)
    cfg = TrainConfig.load(config_path)
    for key in ("prm_id","prm_revision","prm_base_revision"):
        if original["metadata"][key] != getattr(cfg, key) or getattr(cfg, key) == "main":
            raise ValueError(f"PRM provenance mismatch or unresolved revision: {key}")
    if unquantized:
        cfg.quantize_prm = False
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {**original["metadata"], "prm_scored":True, "prm_quantized":cfg.quantize_prm}
    partial = output.with_suffix(".partial.json")
    rows = []
    if partial.exists():
        prior = json.loads(partial.read_text())
        if prior["input_sha256"] != digest or prior["metadata"] != metadata:
            raise ValueError("Partial scoring source/protocol changed")
        rows = prior["rows"]
    reward = reward_factory(cfg, device)
    for row in original["rows"][len(rows):]:
        record = Rollout(row["id"], 0, row["question"], row["reference"], [], [],
                         row["response"], bool(row["truncated"]))
        reward.score([record])
        rows.append({**row, "prm_reward":record.reward, "prm_steps":record.step_scores,
                     "correct":int(exact_match(record.text, record.answer))})
        atomic_json(partial, {"input_sha256":digest, "metadata":metadata, "rows":rows})
        print(f"PRM scored {len(rows)}/{len(original['rows'])}", flush=True)
    atomic_json(output, {"metadata":metadata, "input_sha256":digest,
                        "summary":summarize(rows), "rows":rows})
    partial.unlink(missing_ok=True)


def frontier(input_paths):
    points, protocols = [], {}
    for path in input_paths:
        data = json.loads(Path(path).read_text())
        meta = data["metadata"]
        if not meta["prm_scored"]:
            raise ValueError("Frontier requires PRM-scored responses")
        key = meta["seed"]
        protocol = {k:meta[k] for k in ("evaluation_input_hash","eval_max_new_tokens","split",
            "decoding","prm_id","prm_revision","prm_base_revision","prm_quantized","training_protocol")}
        if key in protocols and protocols[key] != protocol:
            raise ValueError("Frontier points within a seed use different data or scoring protocols")
        protocols[key] = protocol
        points.append({"source":str(path), "seed":meta["seed"], "lambda":meta["lambda"],
                       "checkpoint":meta["checkpoint_name"], "splits":summarize(data["rows"])})
    return {"points":points, "note":"Descriptive proxy/gold points, not causal proof or confidence intervals."}
