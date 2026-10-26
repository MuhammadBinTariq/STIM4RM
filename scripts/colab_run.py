"""Run one matched experimental arm on Colab; never assume a particular GPU.

Invoke from the repository root after installing requirements-v2.txt.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(command):
    print("Running:", " ".join(map(str,command)), flush=True)
    subprocess.run(list(map(str,command)), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--output-root", required=True, help="Persistent run directory, normally Google Drive")
    parser.add_argument("--arm", choices=["baseline","rewardmaxxing"], required=True)
    parser.add_argument("--phase", choices=["pilot","train","evaluate"], default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--train-steps", type=int, help="Cumulative update limit; resume to 64,128,256 for frontier evaluation")
    parser.add_argument("--stim-lambda", type=float, help="Treatment coefficient; use separate output roots for different lambdas")
    parser.add_argument("--monitor-stim", action="store_true", help="Also score baseline rollouts, with no penalty")
    parser.add_argument("--skip-prm", action="store_true", help="Evaluation only: score PRM later in a separate process")
    parser.add_argument("--split", choices=["test","dev"], default="test")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--eval-output", help="Evaluation filename within the run directory")
    parser.add_argument("--twins", nargs="*")
    parser.add_argument("--paired-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.pilot_steps <= 0:
        raise ValueError("pilot-steps must be positive")
    # Run the offline CPU test in its own process before any pretrained weights.
    run([sys.executable, "tests/smoke_test.py"])
    import torch
    from src.rewardmaxxing.checkpoints import atomic_json
    from src.rewardmaxxing.config import TrainConfig
    from src.rewardmaxxing.telemetry import monitor, storage_sample
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab")
    gpu = torch.cuda.get_device_properties(0)
    hardware = {"gpu":gpu.name, "vram_gib":gpu.total_memory/2**30,
                "torch":torch.__version__, "cuda":torch.version.cuda,
                "bf16":torch.cuda.is_bf16_supported()}
    if gpu.total_memory < 20*2**30 or not hardware["bf16"]:
        raise RuntimeError(f"Default paired VersaPRM recipe requires >=20 GiB and BF16: {hardware}. Use A100/L4; no silent recipe changes.")
    cfg = TrainConfig.load(args.resolved_config)
    if args.stim_lambda is not None:
        cfg = replace(cfg, stim_lambda=args.stim_lambda)
    if args.arm == "rewardmaxxing" and cfg.stim_lambda <= 0:
        raise ValueError("RewardMaxxing requires a positive penalty in resolved config")
    root = Path(args.output_root).resolve()
    run_dir = root / f"{args.arm}-seed{args.seed}"
    cfg = replace(cfg, seed=args.seed, output_dir=str(run_dir),
                  stim_lambda=0.0 if args.arm == "baseline" else cfg.stim_lambda,
                  max_steps=args.pilot_steps if args.phase == "pilot" else args.train_steps)
    cfg.stim.cache_path = "/content/rewardmaxxing-cache/infinigram.sqlite"
    cfg.stim.cache_backup = str(root / "cache-snapshot.sqlite")
    cfg.validate()
    config_path = run_dir / "runner-config.json"
    if config_path.exists() and TrainConfig.load(config_path).fingerprint() != cfg.fingerprint():
        raise ValueError("Run semantics changed; use a different output-root (especially for lambda sweeps)")
    atomic_json(config_path, cfg.to_dict())
    atomic_json(run_dir / "hardware.json", hardware)
    freeze = subprocess.check_output([sys.executable,"-m","pip","freeze","--all"], text=True)
    (run_dir / "environment.txt").write_text(freeze, encoding="utf-8")
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git","status","--porcelain"], cwd=ROOT, text=True).strip()
        source = {"commit":commit, "dirty":bool(dirty), "status":dirty}
    except (OSError, subprocess.CalledProcessError):
        manifest = ROOT / "source-manifest.json"
        source = {"zip_manifest":json.loads(manifest.read_text()) if manifest.exists() else None}
    invocation = str(time.time_ns())
    atomic_json(run_dir/"invocations"/f"{invocation}.json",
                {"arguments":vars(args), "source":source, "hardware":hardware, "environment":freeze})
    latest_path = run_dir / "latest.json"
    latest = run_dir / json.loads(latest_path.read_text())["checkpoint"] if latest_path.exists() else None
    if args.phase == "evaluate":
        if latest is None:
            raise FileNotFoundError("No trained checkpoint exists")
        filename = args.eval_output or f"evaluation-{args.split}-{latest.name}.json"
        if Path(filename).name != filename:
            raise ValueError("eval-output must be a filename")
        command = [sys.executable,"-m","src.evaluate","run","--checkpoint",latest,
                   "--output",run_dir/filename,"--stim-audit","--split",args.split]
        if args.skip_prm:
            command.append("--skip-prm")
        if args.eval_limit:
            command.extend(["--limit",str(args.eval_limit)])
        if args.twins:
            command.extend(["--twins",*args.twins])
        if args.paired_only:
            command.append("--paired-only")
        with monitor(run_dir/"system.jsonl", run_dir, cfg.stim.cache_path):
            run(command)
    else:
        command = [sys.executable,"-m","src.train","--config",config_path]
        if latest:
            command.extend(["--resume",latest])
        if args.monitor_stim:
            command.append("--monitor-stim")
        try:
            with monitor(run_dir/"system.jsonl", run_dir, cfg.stim.cache_path):
                run(command)
        finally:
            atomic_json(run_dir/"storage.json", storage_sample(run_dir, cfg.stim.cache_path))
        metric_files = sorted((run_dir/"metrics").glob("step-*.json"))
        metrics = [json.loads(p.read_text()) for p in metric_files]
        if metrics:
            coverage = [m["stim_valid_fraction"] for m in metrics if m["stim_valid_fraction"] is not None]
            summary = {"updates":len(metrics), "mean_seconds_per_update":sum(m["elapsed_seconds"] for m in metrics)/len(metrics),
                "max_cuda_allocated_gb":max(m["cuda_peak_gb"] for m in metrics),
                "mean_stim_wait_seconds":sum(m["stim_wait_seconds"] for m in metrics)/len(metrics),
                "mean_persistence_seconds":sum(m.get("persistence_seconds",0) for m in metrics)/len(metrics),
                "cache_hit_rate":sum(m["cache"].get("hits",0) for m in metrics)/max(1,
                    sum(sum(m["cache"].get(k,0) for k in ("hits","misses","shared")) for m in metrics)),
                "mean_stim_valid_fraction":sum(coverage)/len(coverage) if coverage else None}
            atomic_json(run_dir/"pilot-summary.json", summary)
            print(json.dumps(summary, indent=2))
            print("Use measured timings and coverage to decide the equal update budget for both arms.")


if __name__ == "__main__":
    main()
