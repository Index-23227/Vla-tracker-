#!/usr/bin/env python3
"""
Validate that documentation counts match actual data.
Checks README.md and blueprint against real YAML/JSON data.

Run after adding/removing models or benchmarks:
    python scripts/validate_counts.py
"""

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"
LEADERBOARD = ROOT / "data" / "leaderboard.json"
README = ROOT / "README.md"
BLUEPRINT = ROOT / "physical-ai-dashboard-blueprint.md"

errors = []
warnings = []


def count_yaml_files(directory: Path) -> int:
    return len(list(directory.glob("*.yaml")))


def count_benchmark_models(leaderboard: dict, avg_key: str) -> int:
    """Count models that have a non-null average for a given benchmark."""
    return sum(1 for m in leaderboard["models"] if m.get(avg_key) is not None)


def check_readme_counts(leaderboard: dict):
    """Check that README.md counts match actual data."""
    if not README.exists():
        warnings.append("README.md not found, skipping count checks")
        return

    text = README.read_text(encoding="utf-8")

    # --- YAML file count ---
    actual_model_yamls = count_yaml_files(MODELS_DIR)
    match = re.search(r"models/.*?(\d+)\s+YAML files", text)
    if match:
        stated = int(match.group(1))
        if stated != actual_model_yamls:
            errors.append(
                f"README: model YAML file count is {stated}, actual is {actual_model_yamls}"
            )

    # --- Benchmark model counts ---
    benchmark_checks = {
        "LIBERO": ("libero_avg", r"\[LIBERO\].*?\|\s*(\d+)\s*\|"),
        "CALVIN": ("calvin_avg", r"\[CALVIN\].*?\|\s*(\d+)\s*\|"),
        "SimplerEnv": ("simpler_avg", r"\[SimplerEnv\].*?\|\s*(\d+)\s*\|"),
        "RLBench": ("rlbench_avg", r"\[RLBench\].*?\|\s*(\d+)\s*\|"),
    }

    for bench_name, (avg_key, pattern) in benchmark_checks.items():
        actual = count_benchmark_models(leaderboard, avg_key)
        match = re.search(pattern, text)
        if match:
            stated = int(match.group(1))
            if stated != actual:
                errors.append(
                    f"README: {bench_name} model count is {stated}, actual is {actual}"
                )

    # --- RoboTwin (combines v1 + v2) ---
    rt_v1 = count_benchmark_models(leaderboard, "robotwin_v1_avg")
    rt_v2 = count_benchmark_models(leaderboard, "robotwin_v2_avg")
    rt_total = len(set(
        m["name"] for m in leaderboard["models"]
        if m.get("robotwin_v1_avg") is not None or m.get("robotwin_v2_avg") is not None
    ))
    match = re.search(r"\[RoboTwin.*?\].*?\|\s*(\d+)\s*\|", text)
    if match:
        stated = int(match.group(1))
        if stated != rt_total:
            errors.append(
                f"README: RoboTwin model count is {stated}, actual is {rt_total}"
            )

    # --- LIBERO leaderboard heading count ---
    match = re.search(r"LIBERO Leaderboard \((\d+) models\)", text)
    if match:
        stated = int(match.group(1))
        actual = count_benchmark_models(leaderboard, "libero_avg")
        if stated != actual:
            errors.append(
                f"README: LIBERO leaderboard heading says {stated} models, actual is {actual}"
            )

    # --- Total models count (line like "84 models") ---
    match = re.search(r"Tracking\s+\*\*(\d+)\*\*\s+VLA models", text)
    if match:
        stated = int(match.group(1))
        actual = leaderboard["num_models"]
        if stated != actual:
            errors.append(
                f"README: total model count is {stated}, actual is {actual}"
            )


def check_blueprint_counts(leaderboard: dict):
    """Check that blueprint counts match actual data."""
    if not BLUEPRINT.exists():
        warnings.append("Blueprint file not found, skipping")
        return

    text = BLUEPRINT.read_text(encoding="utf-8")

    benchmark_checks = {
        "LIBERO": "libero_avg",
        "CALVIN": "calvin_avg",
        "SimplerEnv": "simpler_avg",
        "RLBench": "rlbench_avg",
    }

    for bench_name, avg_key in benchmark_checks.items():
        actual = count_benchmark_models(leaderboard, avg_key)
        # Match patterns like "| LIBERO | 4 suites | 25+ |" or "| LIBERO | 4 suites | 40 |"
        pattern = rf"\|\s*{bench_name}\s*\|[^|]*\|\s*(\d+)\+?\s*\|"
        match = re.search(pattern, text)
        if match:
            stated = int(match.group(1))
            if stated != actual:
                errors.append(
                    f"Blueprint: {bench_name} model count is {stated}, actual is {actual}"
                )


def check_leaderboard_freshness(leaderboard: dict):
    """Warn if leaderboard.json may be stale vs YAML files."""
    actual_model_yamls = count_yaml_files(MODELS_DIR)
    leaderboard_count = leaderboard["num_models"]
    if actual_model_yamls != leaderboard_count:
        warnings.append(
            f"leaderboard.json has {leaderboard_count} models but there are "
            f"{actual_model_yamls} YAML files. Run: python scripts/build_leaderboard.py"
        )


def main():
    print("=== VLA-Tracker Count Validation ===\n")

    # Load leaderboard
    if not LEADERBOARD.exists():
        print("ERROR: leaderboard.json not found. Run build_leaderboard.py first.")
        return 1

    with open(LEADERBOARD, "r", encoding="utf-8") as f:
        leaderboard = json.load(f)

    print(f"Leaderboard: {leaderboard['num_models']} models")
    print(f"Model YAMLs: {count_yaml_files(MODELS_DIR)} files")
    print(f"Benchmark YAMLs: {count_yaml_files(BENCHMARKS_DIR)} files\n")

    check_leaderboard_freshness(leaderboard)
    check_readme_counts(leaderboard)
    check_blueprint_counts(leaderboard)

    # Report
    print(f"{'=' * 50}")
    if errors:
        print(f"\n ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n No count inconsistencies found!")

    if warnings:
        print(f"\n WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    print()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
