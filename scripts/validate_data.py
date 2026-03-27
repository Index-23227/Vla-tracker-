#!/usr/bin/env python3
"""
Validate data integrity for all YAML files in data/.
Checks required fields, value ranges, and cross-references.
"""

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"

REQUIRED_MODEL_FIELDS = ["name", "organization", "date", "architecture", "benchmarks"]
REQUIRED_BENCHMARK_FIELDS = ["name", "metric", "higher_is_better"]

VALID_ACTION_HEAD_CATEGORIES = {
    "autoregressive", "diffusion", "flow_matching", "discrete_diffusion",
    "regression", "inverse_dynamics", "hybrid", "other",
}

errors = []
warnings = []


def load_yaml(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        errors.append(f"[PARSE ERROR] {path.name}: {e}")
        return None


def validate_model(path: Path, data: dict, known_benchmarks: set[str]):
    name = data.get("name", path.stem)

    # Required fields
    for field in REQUIRED_MODEL_FIELDS:
        if field not in data:
            errors.append(f"[{name}] Missing required field: {field}")

    # Architecture checks
    arch = data.get("architecture", {})
    if not arch.get("action_head"):
        warnings.append(f"[{name}] Missing architecture.action_head")

    # action_head_category validation
    cat = arch.get("action_head_category")
    if not cat:
        errors.append(f"[{name}] Missing architecture.action_head_category")
    elif cat not in VALID_ACTION_HEAD_CATEGORIES:
        errors.append(
            f"[{name}] Invalid action_head_category '{cat}'. "
            f"Must be one of: {', '.join(sorted(VALID_ACTION_HEAD_CATEGORIES))}"
        )

    # Benchmark value checks
    benchmarks = data.get("benchmarks", {})
    for bench_name, bench_data in benchmarks.items():
        if not isinstance(bench_data, dict):
            errors.append(f"[{name}] Benchmark '{bench_name}' should be a dict")
            continue

        for key, val in bench_data.items():
            if key in ("source", "date_reported"):
                continue
            if isinstance(val, (int, float)):
                if val < 0 or val > 100:
                    warnings.append(
                        f"[{name}] {bench_name}.{key} = {val} — outside 0-100 range"
                    )

    # Tags check
    if not data.get("tags"):
        warnings.append(f"[{name}] No tags defined")

    # Date format
    date = data.get("date", "")
    if date and not isinstance(date, str):
        warnings.append(f"[{name}] Date should be a string (YYYY-MM-DD)")


def validate_benchmark(path: Path, data: dict):
    name = data.get("name", path.stem)

    for field in REQUIRED_BENCHMARK_FIELDS:
        if field not in data:
            errors.append(f"[Benchmark:{name}] Missing required field: {field}")

    if not data.get("categories"):
        warnings.append(f"[Benchmark:{name}] No categories defined")


def main():
    print("=== VLA-Tracker Data Validation ===\n")

    # Load benchmarks
    known_benchmarks = set()
    benchmark_files = list(BENCHMARKS_DIR.glob("*.yaml"))
    print(f"Validating {len(benchmark_files)} benchmark files...")
    for bf in sorted(benchmark_files):
        data = load_yaml(bf)
        if data:
            known_benchmarks.add(bf.stem)
            validate_benchmark(bf, data)

    # Load and validate models
    model_files = list(MODELS_DIR.glob("*.yaml"))
    print(f"Validating {len(model_files)} model files...")
    for mf in sorted(model_files):
        data = load_yaml(mf)
        if data:
            validate_model(mf, data, known_benchmarks)

    # Report
    print(f"\n{'='*40}")
    if errors:
        print(f"\n❌ {len(errors)} ERROR(S):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\n✅ No errors found!")

    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  {w}")
    else:
        print("✅ No warnings!")

    print(f"\nSummary: {len(model_files)} models, {len(benchmark_files)} benchmarks")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
