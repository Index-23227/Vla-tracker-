#!/usr/bin/env python3
"""
Restore missing model YAML files from a previous leaderboard.json.
Reads the old leaderboard and generates YAML stubs for models
that have no corresponding YAML file in data/models/.
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"

# Benchmark key mapping: leaderboard avg key -> YAML section name
BENCH_AVG_TO_SECTION = {
    "libero_avg": "libero",
    "calvin_avg": "calvin",
    "simpler_avg": "simpler_env",
    "robotwin_v1_avg": "robotwin_v1",
    "robotwin_v2_avg": "robotwin_v2",
    "rlbench_avg": "rlbench",
    "robocasa_avg": "robocasa",
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace("*", "").replace(".", "")


def model_to_yaml(entry: dict) -> dict:
    """Convert a leaderboard entry back to YAML-compatible dict."""
    doc = {
        "name": entry["name"],
        "organization": entry.get("organization", "Unknown"),
        "date": entry.get("date"),
        "paper_url": entry.get("paper_url"),
        "code_url": entry.get("code_url"),
        "venue": entry.get("venue"),
        "open_source": entry.get("open_source", False),
        "model_type": entry.get("model_type"),
        "tags": entry.get("tags", []),
        "architecture": entry.get("architecture", {}),
    }
    if entry.get("inference_hz"):
        doc["inference_hz"] = entry["inference_hz"]

    # Rebuild benchmarks section from detailed scores
    benchmarks = {}
    raw_benchmarks = entry.get("benchmarks", {})
    for bench_name, scores in raw_benchmarks.items():
        if scores:
            benchmarks[bench_name] = dict(scores)

    # Add eval conditions
    eval_conds = entry.get("eval_conditions", {})
    for bench_name, cond in eval_conds.items():
        if bench_name in benchmarks:
            benchmarks[bench_name]["eval_condition"] = cond

    if benchmarks:
        doc["benchmarks"] = benchmarks

    # Remove None values
    return {k: v for k, v in doc.items() if v is not None}


def main():
    if len(sys.argv) < 2:
        print("Usage: python restore_missing_models.py <old_leaderboard.json>")
        return 1

    old_path = Path(sys.argv[1])
    with open(old_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    # Find existing model names
    existing_names = set()
    for yf in MODELS_DIR.glob("*.yaml"):
        data = yaml.safe_load(open(yf, encoding="utf-8"))
        if data:
            existing_names.add(data.get("name", ""))

    restored = 0
    for entry in old_data["models"]:
        name = entry["name"]
        if name in existing_names:
            continue

        slug = slugify(name)
        out_path = MODELS_DIR / f"{slug}.yaml"
        if out_path.exists():
            continue

        doc = model_to_yaml(entry)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"  Restored: {name} -> {out_path.name}")
        restored += 1

    print(f"\nRestored {restored} model YAML files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
