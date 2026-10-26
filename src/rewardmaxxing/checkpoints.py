import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_replace(source, destination):
    """Retry transient Windows locks without falling back to a non-atomic copy."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            # Access denied / sharing violation / lock violation. Other errors
            # (including POSIX permission errors) must surface immediately.
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.8))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    atomic_replace(temporary, path)


def save_checkpoint(directory, policy, optimizer, cfg, step, manifest):
    directory = Path(directory)
    destination = directory / f"step-{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    temporary = directory / f".step-{step:06d}-{uuid.uuid4().hex}.part"
    temporary.mkdir(parents=True)
    policy.model.save_pretrained(temporary / "adapter", safe_serialization=True)
    if policy.tokenizer is not None:
        policy.tokenizer.save_pretrained(temporary / "tokenizer")
    atomic_json(temporary / "config.json", cfg.to_dict())
    atomic_json(temporary / "data_manifest.json", manifest)
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng_state(), "step": step,
                "config_fingerprint": cfg.fingerprint()}, temporary / "training_state.pt")
    atomic_replace(temporary, destination)
    atomic_json(directory / "latest.json", {"checkpoint": destination.name, "step": step})
    # Only remove recognized checkpoint directories created inside this run directory.
    old = sorted(directory.glob("step-[0-9][0-9][0-9][0-9][0-9][0-9]"))
    root = directory.resolve()
    for path in old[:-cfg.keep_checkpoints]:
        resolved = path.resolve()
        if path.is_symlink() or resolved.parent != root:
            raise ValueError("Unsafe checkpoint retention target")
        if (path / "training_state.pt").is_file():
            shutil.rmtree(path)
    return destination


def load_training_state(path, optimizer, cfg):
    # Only load checkpoints produced by this run; torch optimizer/RNG files use pickle.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["config_fingerprint"] != cfg.fingerprint():
        raise ValueError("Resume configuration changes training semantics")
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["step"]
