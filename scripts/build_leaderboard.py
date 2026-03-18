#!/usr/bin/env python3
"""
Build leaderboard.json from model YAML files.
Reads all data/models/*.yaml and generates a unified leaderboard JSON.
"""

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"
OUTPUT_FILE = ROOT / "data" / "leaderboard.json"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_models() -> list[dict]:
    models = []
    for yaml_file in sorted(MODELS_DIR.glob("*.yaml")):
        data = load_yaml(yaml_file)
        data["_file"] = yaml_file.name
        models.append(data)
    return models


def load_all_benchmarks() -> dict[str, dict]:
    benchmarks = {}
    for yaml_file in sorted(BENCHMARKS_DIR.glob("*.yaml")):
        data = load_yaml(yaml_file)
        key = yaml_file.stem
        benchmarks[key] = data
    return benchmarks


def compute_libero_avg(benchmarks: dict) -> float | None:
    libero = benchmarks.get("libero")
    if not libero:
        return None
    scores = []
    for key in ["libero_spatial", "libero_object", "libero_goal", "libero_long"]:
        val = libero.get(key)
        if val is not None and isinstance(val, (int, float)):
            scores.append(val)
    if len(scores) == 4:
        return round(sum(scores) / 4, 2)
    return None


def build_leaderboard(models: list[dict], benchmarks_meta: dict[str, dict]) -> dict:
    leaderboard_entries = []

    for model in models:
        entry = {
            "name": model["name"],
            "organization": model.get("organization", "Unknown"),
            "date": model.get("date"),
            "paper_url": model.get("paper_url"),
            "code_url": model.get("code_url"),
            "open_source": model.get("open_source", False),
            "tags": model.get("tags", []),
            "architecture": {
                "action_head": model.get("architecture", {}).get("action_head", "unknown"),
                "parameters": model.get("architecture", {}).get("parameters", "unknown"),
            },
            "benchmarks": {},
        }

        model_benchmarks = model.get("benchmarks", {})

        # Process LIBERO
        if "libero" in model_benchmarks:
            libero = model_benchmarks["libero"]
            entry["benchmarks"]["libero"] = {
                k: v for k, v in libero.items()
                if k not in ("source", "date_reported") and isinstance(v, (int, float))
            }

        # Process CALVIN
        if "calvin" in model_benchmarks:
            calvin = model_benchmarks["calvin"]
            entry["benchmarks"]["calvin"] = {
                k: v for k, v in calvin.items()
                if k not in ("source", "date_reported") and isinstance(v, (int, float))
            }

        # Process SimplerEnv
        if "simpler_env" in model_benchmarks:
            simpler = model_benchmarks["simpler_env"]
            entry["benchmarks"]["simpler_env"] = {
                k: v for k, v in simpler.items()
                if k not in ("source", "date_reported") and isinstance(v, (int, float))
            }

        # Compute LIBERO average
        libero_avg = compute_libero_avg(model_benchmarks)
        if libero_avg is not None:
            entry["libero_avg"] = libero_avg

        leaderboard_entries.append(entry)

    # Sort by LIBERO average (descending), models without LIBERO go to end
    leaderboard_entries.sort(
        key=lambda x: x.get("libero_avg", -1),
        reverse=True,
    )

    # Add ranks
    for i, entry in enumerate(leaderboard_entries):
        entry["rank"] = i + 1

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "num_models": len(leaderboard_entries),
        "primary_benchmark": "libero",
        "benchmarks_available": list(benchmarks_meta.keys()),
        "models": leaderboard_entries,
    }


def main():
    print("Loading models...")
    models = load_all_models()
    print(f"  Found {len(models)} models")

    print("Loading benchmarks...")
    benchmarks_meta = load_all_benchmarks()
    print(f"  Found {len(benchmarks_meta)} benchmarks")

    print("Building leaderboard...")
    leaderboard = build_leaderboard(models, benchmarks_meta)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Leaderboard written to {OUTPUT_FILE}")
    print(f"  {leaderboard['num_models']} models ranked")

    # Print top 5
    print("\nTop 5 (LIBERO average):")
    for entry in leaderboard["models"][:5]:
        avg = entry.get("libero_avg", "N/A")
        print(f"  #{entry['rank']} {entry['name']}: {avg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
