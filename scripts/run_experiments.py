"""Run the assignment's controlled ablations without using test data for training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIRS = [ROOT / "data" / "test" / f"test{i}" for i in range(5)]
BASE = ["--num-images", "10000", "--model", "captcha_resnet", "--batch-size", "64", "--learning-rate", "0.001", "--weight-decay", "0.0001"]


def experiment_specs(study: str) -> list[tuple[str, list[str]]]:
    studies = {
        "size": [("10k", []), ("20k", ["--num-images", "20000"]), ("100k", ["--num-images", "100000"])],
        "augmentation": [("no-augmentation", ["--no-augment"]), ("augmentation", ["--augment"])],
        "weight-decay": [("wd-0", ["--weight-decay", "0"]), ("wd-1e-4", []), ("wd-1e-3", ["--weight-decay", "0.001"])],
        "batch-size": [("batch-32", ["--batch-size", "32"]), ("batch-64", []), ("batch-128", ["--batch-size", "128"])],
        "learning-rate": [("lr-1e-4", ["--learning-rate", "0.0001"]), ("lr-1e-3", []), ("lr-1e-2", ["--learning-rate", "0.01"])],
        "backbone": [("resnet18", ["--model", "resnet18"]), ("resnet34", ["--model", "resnet34"])],
        "pooling": [("global-average-pooling", ["--model", "resnet18"]), ("spatial-flatten", ["--model", "resnet18_spatial"])],
    }
    if study == "all":
        return [(f"{name}-{run_name}", overrides) for name, entries in studies.items() for run_name, overrides in entries]
    return [(run_name, overrides) for run_name, overrides in studies[study]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "size", "augmentation", "weight-decay", "batch-size", "learning-rate", "backbone", "pooling"), default="all")
    parser.add_argument("--data-dir", default="data/training.zip")
    parser.add_argument("--val-data-dir", default="data/validation.zip")
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def newest_config(run_root: Path, previous: set[Path]) -> Path:
    configs = set(run_root.glob("*/config.json")) - previous
    if len(configs) != 1:
        raise RuntimeError(f"expected one newly created run config under {run_root}, found {len(configs)}")
    return configs.pop()


def main():
    args = parse_args()
    output_root = (ROOT / args.output_dir).resolve()
    experiments = experiment_specs(args.study)
    if args.dry_run:
        for name, overrides in experiments:
            print(name, [*BASE, *overrides])
        return

    runs = []
    # Complete every train/validation run first. Test metrics cannot affect
    # configuration choices or checkpoint selection.
    for name, overrides in experiments:
        run_root = output_root / name
        previous = set(run_root.glob("*/config.json"))
        command = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--data-dir", str((ROOT / args.data_dir).resolve()),
            "--val-data-dir", str((ROOT / args.val_data_dir).resolve()),
            "--output-dir", str(run_root), "--epochs", str(args.epochs),
            "--workers", str(args.workers), "--seed", str(args.seed),
            *BASE, *overrides,
        ]
        # Use the requested epoch count and one canonical seed across variants.
        print(f"\n=== Training {name} ===", flush=True)
        start = time.perf_counter()
        subprocess.run(command, cwd=ROOT, check=True)
        elapsed = time.perf_counter() - start
        config_path = newest_config(run_root, previous)
        runs.append({
            "name": name,
            "config": str(config_path.relative_to(ROOT)),
            "checkpoint": str((config_path.parent / "checkpoints" / "best.pt").relative_to(ROOT)),
            "training_seconds": elapsed,
        })

    results = []
    for run in runs:
        config_path = Path(run["config"])
        output_json = config_path.parent / "ablation_test_results.json"
        command = [
            sys.executable, str(ROOT / "scripts" / "evaluate.py"),
            "--checkpoint", run["checkpoint"], "--data", *map(str, TEST_DIRS),
            "--device", "auto", "--workers", str(args.workers), "--output-json", str(output_json),
        ]
        print(f"\n=== Evaluation {run['name']} on fixed test sets ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads(output_json.read_text(encoding="utf-8"))
        per_set = report["results"]
        count = sum(result["num_images"] for result in per_set.values())
        aggregate = {
            "num_images": count,
            "digit_acc": sum(result["digit_acc"] * result["num_images"] for result in per_set.values()) / count,
            "image_acc": sum(result["image_acc"] * result["num_images"] for result in per_set.values()) / count,
        }
        results.append({**run, "test_results": per_set, "aggregate": aggregate})

    report_path = output_root / f"{args.study}_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"study": args.study, "seed": args.seed, "runs": results}, indent=2) + "\n", encoding="utf-8")
    if args.study == "all":
        for study in ("size", "augmentation", "weight-decay", "batch-size", "learning-rate", "backbone", "pooling"):
            group = [run for run in results if run["name"].startswith(f"{study}-")]
            (output_root / f"{study}_summary.json").write_text(
                json.dumps({"study": study, "seed": args.seed, "runs": group}, indent=2) + "\n",
                encoding="utf-8",
            )
    print(f"\nSaved experiment summary: {report_path}")


if __name__ == "__main__":
    main()
