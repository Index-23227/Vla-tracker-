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
REVIEWS_FILE = ROOT / "data" / "paper_reviews.json"
OUTPUT_FILE = ROOT / "data" / "leaderboard.json"
DASHBOARD_COPY = ROOT / "dashboard" / "src" / "data" / "leaderboard.json"

METADATA_KEYS = ("source", "date_reported", "eval_condition")


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


def load_reviews() -> dict[str, dict]:
    """Load paper reviews indexed by model name."""
    if not REVIEWS_FILE.exists():
        return {}
    with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["model_name"]: p for p in data.get("papers", []) if p.get("venue")}


def load_all_benchmarks() -> dict[str, dict]:
    benchmarks = {}
    for yaml_file in sorted(BENCHMARKS_DIR.glob("*.yaml")):
        data = load_yaml(yaml_file)
        key = yaml_file.stem
        benchmarks[key] = data
    return benchmarks


def extract_scores(bench_data: dict) -> dict:
    """Extract numeric scores and metadata from a benchmark entry."""
    scores = {}
    meta = {}
    for k, v in bench_data.items():
        if k in METADATA_KEYS:
            meta[k] = v
        elif isinstance(v, (int, float)):
            scores[k] = v
    return scores, meta


def compute_libero_avg(benchmarks: dict) -> float | None:
    libero = benchmarks.get("libero")
    if not libero:
        return None
    # Check for pre-computed suite average
    suite_avg = libero.get("libero_5_suite_avg")
    if suite_avg is not None and isinstance(suite_avg, (int, float)):
        return round(suite_avg, 2)
    # Compute from individual suite scores
    scores = []
    for key in ["libero_spatial", "libero_object", "libero_goal", "libero_long"]:
        val = libero.get(key)
        if val is not None and isinstance(val, (int, float)):
            scores.append(val)
    if len(scores) == 4:
        return round(sum(scores) / 4, 2)
    # Fallback: pre-computed libero_avg in YAML
    fallback = libero.get("libero_avg")
    if fallback is not None and isinstance(fallback, (int, float)):
        return round(fallback, 2)
    return None


def compute_calvin_avg(benchmarks: dict) -> float | None:
    calvin = benchmarks.get("calvin")
    if not calvin:
        return None
    val = calvin.get("calvin_abc_d_avg_len")
    if val is not None and isinstance(val, (int, float)):
        return round(val, 2)
    # Fallback: pre-computed calvin_avg in YAML
    fallback = calvin.get("calvin_avg")
    if fallback is not None and isinstance(fallback, (int, float)):
        return round(fallback, 2)
    return None


def compute_simpler_avg(benchmarks: dict) -> float | None:
    simpler = benchmarks.get("simpler_env")
    if not simpler:
        return None
    scores = [v for k, v in simpler.items()
              if k not in METADATA_KEYS and isinstance(v, (int, float))]
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def compute_robotwin_v1_avg(benchmarks: dict) -> float | None:
    robotwin = benchmarks.get("robotwin_v1")
    if not robotwin:
        return None
    # Use pre-computed average if available
    avg = robotwin.get("average")
    if avg is not None and isinstance(avg, (int, float)):
        return round(avg, 2)
    scores = [v for k, v in robotwin.items()
              if k not in METADATA_KEYS and isinstance(v, (int, float))]
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def compute_robotwin_v2_avg(benchmarks: dict) -> float | None:
    robotwin_v2 = benchmarks.get("robotwin_v2")
    if not robotwin_v2:
        return None
    # Use pre-computed average if available
    avg = robotwin_v2.get("robotwin_v2_avg")
    if avg is not None and isinstance(avg, (int, float)):
        return round(avg, 2)
    scores = [v for k, v in robotwin_v2.items()
              if k not in METADATA_KEYS and isinstance(v, (int, float))]
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def compute_rlbench_avg(benchmarks: dict) -> float | None:
    rlbench = benchmarks.get("rlbench")
    if not rlbench:
        return None
    avg = rlbench.get("rlbench_avg")
    if avg is not None and isinstance(avg, (int, float)):
        return round(avg, 2)
    scores = [v for k, v in rlbench.items()
              if k not in METADATA_KEYS and isinstance(v, (int, float))]
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None



def compute_robocasa_avg(benchmarks: dict) -> float | None:
    robocasa = benchmarks.get("robocasa")
    if not robocasa:
        return None
    avg = robocasa.get("robocasa_avg")
    if avg is not None and isinstance(avg, (int, float)):
        return round(avg, 2)
    scores = [v for k, v in robocasa.items()
              if k not in METADATA_KEYS and isinstance(v, (int, float))]
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def build_leaderboard(models: list[dict], benchmarks_meta: dict[str, dict],
                      reviews: dict[str, dict] | None = None) -> dict:
    leaderboard_entries = []

    for model in models:
        entry = {
            "name": model["name"],
            "organization": model.get("organization", "Unknown"),
            "date": model.get("date"),
            "paper_url": model.get("paper_url"),
            "code_url": model.get("code_url"),
            "venue": model.get("venue"),
            "open_source": model.get("open_source", False),
            "tags": model.get("tags", []),
            "architecture": {
                "backbone": model.get("architecture", {}).get("backbone"),
                "llm": model.get("architecture", {}).get("llm"),
                "action_head": model.get("architecture", {}).get("action_head", "unknown"),
                "parameters": model.get("architecture", {}).get("parameters", "unknown"),
            },
            "model_type": model.get("model_type"),
            "inference_hz": model.get("inference_hz"),
            "benchmarks": {},
            "eval_conditions": {},
        }

        model_benchmarks = model.get("benchmarks", {})

        # Process each benchmark
        for bench_name in ["libero", "calvin", "simpler_env", "rlbench", "metaworld", "robotwin_v1", "robotwin_v2", "robocasa"]:
            if bench_name in model_benchmarks:
                scores, meta = extract_scores(model_benchmarks[bench_name])
                if scores:
                    entry["benchmarks"][bench_name] = scores
                if meta.get("eval_condition"):
                    entry["eval_conditions"][bench_name] = meta["eval_condition"]

        # Also carry over eval_conditions from model YAML
        model_eval = model.get("eval_conditions", {})
        for bench_name, cond in model_eval.items():
            if isinstance(cond, str):
                entry["eval_conditions"][bench_name] = cond

        # Compute averages per benchmark
        libero_avg = compute_libero_avg(model_benchmarks)
        if libero_avg is not None:
            entry["libero_avg"] = libero_avg

        calvin_avg = compute_calvin_avg(model_benchmarks)
        if calvin_avg is not None:
            entry["calvin_avg"] = calvin_avg

        simpler_avg = compute_simpler_avg(model_benchmarks)
        if simpler_avg is not None:
            entry["simpler_avg"] = simpler_avg

        robotwin_v1_avg = compute_robotwin_v1_avg(model_benchmarks)
        if robotwin_v1_avg is not None:
            entry["robotwin_v1_avg"] = robotwin_v1_avg

        robotwin_v2_avg = compute_robotwin_v2_avg(model_benchmarks)
        if robotwin_v2_avg is not None:
            entry["robotwin_v2_avg"] = robotwin_v2_avg

        rlbench_avg = compute_rlbench_avg(model_benchmarks)
        if rlbench_avg is not None:
            entry["rlbench_avg"] = rlbench_avg

        robocasa_avg = compute_robocasa_avg(model_benchmarks)
        if robocasa_avg is not None:
            entry["robocasa_avg"] = robocasa_avg

        # Merge peer-review data if available
        if reviews and model["name"] in reviews:
            rev = reviews[model["name"]]
            entry["peer_review"] = {
                "venue": rev.get("venue"),
                "decision": rev.get("decision"),
                "review_avg": rev.get("review_avg"),
                "confidence_avg": rev.get("confidence_avg"),
                "num_reviews": rev.get("num_reviews", 0),
                "openreview_url": rev.get("openreview_url"),
            }

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

    print("Loading reviews...")
    reviews = load_reviews()
    print(f"  Found reviews for {len(reviews)} models")

    print("Building leaderboard...")
    leaderboard = build_leaderboard(models, benchmarks_meta, reviews)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Leaderboard written to {OUTPUT_FILE}")

    # Sync to dashboard
    if DASHBOARD_COPY.parent.exists():
        import shutil
        shutil.copy2(OUTPUT_FILE, DASHBOARD_COPY)
        print(f"Synced to {DASHBOARD_COPY}")

    print(f"  {leaderboard['num_models']} models ranked")

    # Print top 5
    print("\nTop 5 (LIBERO average):")
    for entry in leaderboard["models"][:5]:
        avg = entry.get("libero_avg", "N/A")
        date = entry.get("date", "?")
        print(f"  #{entry['rank']} {entry['name']} ({date}): {avg}")

    # Print CALVIN rankings
    calvin_models = [e for e in leaderboard["models"] if e.get("calvin_avg")]
    if calvin_models:
        calvin_models.sort(key=lambda x: x["calvin_avg"], reverse=True)
        print("\nTop CALVIN models:")
        for e in calvin_models[:5]:
            print(f"  {e['name']}: {e['calvin_avg']} avg len")

    return 0


if __name__ == "__main__":
    sys.exit(main())
