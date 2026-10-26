"""Download a small, revision-pinned official GSM-Symbolic evaluation panel."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def select_instances(rows, subset, instances):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["original_id"]].append(dict(row))
    selected = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda r:r["instance"])
        if len({r["instance"] for r in candidates}) != len(candidates):
            raise ValueError("Duplicate instance IDs")
        if len(candidates) < instances:
            raise ValueError("Requested more instances than available")
        selected.extend({**r, "subset":subset} for r in candidates[:instances])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subsets", nargs="+", choices=["main","p1","p2"], default=["main","p1"])
    parser.add_argument("--instances", type=int, default=1, help="Fixed smallest instance IDs per parent, never selected by outcomes")
    args = parser.parse_args()
    if args.instances < 1 or len(set(args.subsets)) != len(args.subsets):
        raise ValueError("Invalid instance count or duplicate subsets")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Use a new output directory; preserve existing pinned evaluation files")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info("apple/GSM-Symbolic", revision=args.revision).sha
    files = {}
    for subset in args.subsets:
        rows = select_instances(load_dataset("apple/GSM-Symbolic", subset, split="test",
                                            revision=revision), subset, args.instances)
        data = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows).encode("utf-8")
        path = destination/f"{subset}.jsonl"
        path.write_bytes(data)
        files[path.name] = {"sha256":hashlib.sha256(data).hexdigest(), "rows":len(rows)}
    (destination/"manifest.json").write_text(json.dumps({"source":"apple/GSM-Symbolic",
        "revision":revision, "instances_per_parent":args.instances, "files":files}, indent=2))
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
