#!/usr/bin/env python3
"""
Validate data integrity for all YAML files in data/.
Checks required fields, value ranges, and cross-references.
"""

import json
import sys
from pathlib import Path

import yaml

try:
    from jsonschema import Draft7Validator
except ImportError:  # keep the rest of the checks usable without the dep
    Draft7Validator = None


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"
MODEL_SCHEMA_FILE = ROOT / "schemas" / "model.schema.json"


def load_model_schema_validator():
    """Compile schemas/model.schema.json, which until 2026-08 was enforced
    nowhere and had drifted to 147 failing models. It now matches the data
    exactly, so violations are errors — in particular a nested score dict
    inside a benchmark block, which the builder cannot see."""
    if Draft7Validator is None or not MODEL_SCHEMA_FILE.exists():
        return None
    with open(MODEL_SCHEMA_FILE, "r", encoding="utf-8") as f:
        return Draft7Validator(json.load(f))

REQUIRED_MODEL_FIELDS = ["name", "organization", "date", "architecture", "benchmarks"]
REQUIRED_BENCHMARK_FIELDS = ["name", "metric", "higher_is_better"]

VALID_ACTION_HEAD_CATEGORIES = {
    "autoregressive", "diffusion", "flow_matching", "discrete_diffusion",
    "regression", "inverse_dynamics", "hybrid", "other",
}

# Keys build_leaderboard.py accepts as a block's paper-declared headline average.
# Must stay in sync with the compute_*_avg functions there.
HEADLINE_KEYS = {
    "libero": ("libero_5_suite_avg", "libero_avg"),
    "calvin": ("calvin_abc_d_avg_len", "calvin_avg"),
    "simpler_env": ("simpler_avg",),
    "rlbench": ("rlbench_avg", "rlbench_18tasks", "average"),
    "robocasa": ("robocasa_avg", "average"),
    "robotwin_v1": ("average", "robotwin_v1_avg"),
    "robotwin_v2": ("robotwin_v2_avg", "average"),
    "metaworld": ("metaworld_avg", "ml45_avg", "metaworld_mt50_avg", "average"),
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

    # Guessed-average check.
    #
    # When a ranked benchmark block declares no headline average, the builder
    # falls back to the mean of whatever numeric fields are present. That is
    # right for per-task score lists but wrong when the fields are separate eval
    # conditions -- it silently produced a SimplerEnv score that mixed a paper's
    # own headline with its sub-aggregates, and an RLBench score averaged with
    # COLOSSEUM and MemoryBench. Few fields plus a wide spread is the signature
    # of conditions rather than tasks, so flag those for a human to declare.
    for bench_name, keys in HEADLINE_KEYS.items():
        block = benchmarks.get(bench_name)
        if not isinstance(block, dict):
            continue
        nums = {k: v for k, v in block.items()
                if k not in ("source", "date_reported", "eval_condition")
                and isinstance(v, (int, float))}
        if len(nums) < 2 or len(nums) > 3 or any(k in nums for k in keys):
            continue
        spread = max(nums.values()) - min(nums.values())
        if spread >= 20:
            warnings.append(
                f"[{name}] {bench_name} has no headline key "
                f"({'/'.join(keys)}); its average is the mean of "
                f"{', '.join(nums)} which span {spread:.1f} points"
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
    schema_validator = load_model_schema_validator()
    if schema_validator is None:
        warnings.append("jsonschema unavailable — schemas/model.schema.json NOT enforced")

    model_files = list(MODELS_DIR.glob("*.yaml"))
    print(f"Validating {len(model_files)} model files...")
    for mf in sorted(model_files):
        data = load_yaml(mf)
        if data:
            validate_model(mf, data, known_benchmarks)
            if schema_validator is not None:
                for err in schema_validator.iter_errors(data):
                    where = "/".join(str(x) for x in err.absolute_path) or "<root>"
                    errors.append(f"[{data.get('name', mf.stem)}] schema: {where}: {err.message[:120]}")

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
