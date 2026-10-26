"""Package only the v2 source, tests and docs; never include models, caches or data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT/p for p in ["README.md", "LICENSE", "requirements-v2.txt", "src/__init__.py",
             "src/train.py", "src/evaluate.py", "src/prepare.py",
             "tests/smoke_test.py", "notebooks/RewardMaxxing_v2.ipynb"]]
    for folder, pattern in [("src/rewardmaxxing", "*.py"), ("scripts", "*.py"),
                            ("configs", "*.json"), ("docs", "*.md")]:
        files.extend(sorted((ROOT/folder).glob(pattern)))
    destination = ROOT / "dist" / "rewardmaxxing-v2.zip"
    destination.parent.mkdir(exist_ok=True)
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(ROOT).as_posix())
        archive.writestr("source-manifest.json", json.dumps(hashes, indent=2))
    print(f"Created {destination} ({destination.stat().st_size:,} bytes; {len(files)} files)")


if __name__ == "__main__":
    main()
