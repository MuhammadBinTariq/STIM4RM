"""Export local per-update metrics to CSV; no external tracking account needed."""
import argparse
import csv
import json
from pathlib import Path


def flatten(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [flatten(json.loads(p.read_text())) for p in sorted((Path(args.run_dir)/"metrics").glob("step-*.json"))]
    if not rows:
        raise ValueError("No per-step metrics found")
    keys = ["step"] + sorted(set().union(*(row.keys() for row in rows)) - {"step"})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} updates to {path}")


if __name__ == "__main__":
    main()
