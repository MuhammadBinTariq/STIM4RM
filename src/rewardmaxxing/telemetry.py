"""Local telemetry only; no credentials, external tracking service, or GPU tensors."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time


def tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file() and not file.is_symlink():
                total += file.stat().st_size
        except FileNotFoundError:
            pass  # Concurrent checkpoint retention.
    return total


def storage_sample(run_dir, cache_path):
    run_dir, cache_path = Path(run_dir), Path(cache_path)
    checkpoints = sorted(run_dir.glob("step-[0-9][0-9][0-9][0-9][0-9][0-9]"))
    return {
        "run_bytes": tree_bytes(run_dir),
        "checkpoint_bytes": {p.name:tree_bytes(p) for p in checkpoints},
        "adapter_bytes": {p.name:tree_bytes(p/"adapter") for p in checkpoints},
        "optimizer_bytes": {p.name:(p/"training_state.pt").stat().st_size
                            for p in checkpoints if (p/"training_state.pt").exists()},
        "cache_bytes": sum(p.stat().st_size for p in cache_path.parent.glob(cache_path.name+"*") if p.is_file()),
        "local_disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "unexpected_full_model_files": [str(p) for cp in checkpoints for p in cp.rglob("*")
            if p.is_file() and (p.name.startswith("pytorch_model") or p.name.startswith("model-")
                               or p.name == "model.safetensors")]}


def gpu_sample():
    try:
        raw = subprocess.check_output(["nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"], text=True, timeout=4)
        keys = ["index","gpu_util_percent","memory_util_percent","used_mib","total_mib","power_w"]
        return [dict(zip(keys, [float(v.strip()) if v.strip().replace(".","",1).isdigit() else v.strip()
                               for v in line.split(",")])) for line in raw.strip().splitlines()]
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error":str(exc)}


@contextmanager
def monitor(path, run_dir, cache_path, interval=5):
    """Sample nvidia-smi every 5 s and directory sizes every 60 s, tied to child lifetime."""
    stop = threading.Event()
    def sample_loop():
        last_storage = 0.0
        with Path(path).open("a", encoding="utf-8") as stream:
            while not stop.is_set():
                now = time.monotonic()
                row = {"unix_time":time.time(), "gpu":gpu_sample()}
                if now-last_storage >= 60:
                    try:
                        row["storage"] = storage_sample(run_dir, cache_path)
                    except OSError as exc:
                        row["storage_error"] = str(exc)
                    last_storage = now
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                stop.wait(interval)
    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=10)
