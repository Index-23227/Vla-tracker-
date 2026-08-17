#!/usr/bin/env python3
"""
Build leaderboard.json from model YAML files.
Reads all data/models/*.yaml and generates a unified leaderboard JSON.
"""

import copy
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"
REVIEWS_FILE = ROOT / "data" / "paper_reviews.json"
AI_REVIEWS_DIR = ROOT / "data" / "ai_reviews"
OUTPUT_FILE = ROOT / "data" / "leaderboard.json"
DASHBOARD_COPY = ROOT / "dashboard" / "src" / "data" / "leaderboard.json"
PUBLIC_DIR = ROOT / "dashboard" / "public"
REVIEWS_BUNDLE = PUBLIC_DIR / "reviews.json"
REVIEWS_DIR = PUBLIC_DIR / "reviews"

METADATA_KEYS = ("source", "date_reported", "eval_condition")

CANONICAL_BENCHMARKS = ("libero", "calvin", "simpler_env", "rlbench", "metaworld",
                        "robotwin_v1", "robotwin_v2", "robocasa", "real_world")

# LIBERO robustness suites are written both as their own top-level block and as
# libero_plus_* / libero_pro_* keys inside benchmarks.libero. Only the nested
# form ever reached consumers, so the block form was silently discarded.
LIBERO_VARIANTS = ("libero_plus", "libero_pro")


# Evaluation results that were written at the YAML root instead of under
# benchmarks:, where nothing reads them. Engineering metrics that also live at
# the root (latency, inference_speedup, verifier_metrics) are deliberately not
# listed -- they are not benchmark scores.
ROOT_RESULT_BLOCKS = ("real_world", "real_world_results", "ood_robustness",
                      "one_shot", "cross_task", "vlabench")


def hoist_root_result_blocks(model: dict) -> dict:
    """Move stray root-level result blocks into the model's benchmarks map."""
    benchmarks = dict(model.get("benchmarks") or {})
    for key in ROOT_RESULT_BLOCKS:
        block = model.get(key)
        if isinstance(block, dict) and key not in benchmarks:
            benchmarks[key] = block
    return benchmarks


def fold_libero_variants(model_benchmarks: dict) -> dict:
    """Merge top-level libero_plus / libero_pro blocks into benchmarks.libero."""
    if not isinstance(model_benchmarks, dict):
        return {}
    if not any(v in model_benchmarks for v in LIBERO_VARIANTS):
        return model_benchmarks

    folded = dict(model_benchmarks)
    libero = dict(folded.get("libero") or {})
    for variant in LIBERO_VARIANTS:
        block = folded.pop(variant, None)
        if not isinstance(block, dict):
            continue
        for key, val in block.items():
            if key in METADATA_KEYS:
                # Keep the variant's provenance without clobbering LIBERO's own.
                libero.setdefault(f"{variant}_{key}", val)
            elif isinstance(val, (int, float)):
                libero.setdefault(key if key.startswith(variant) else f"{variant}_{key}", val)
    folded["libero"] = libero
    return folded


def review_slug(name: str) -> str:
    """Filesystem- and URL-safe key for a per-model review file.

    Must stay in sync with reviewSlug() in dashboard/src/lib/reviews.js.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


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


def load_ai_reviews(model_names: list[str] | None = None) -> dict[str, str]:
    """Load AI-generated reviews indexed by model name.

    Review files are named after the model with spaces replaced by underscores
    (the convention check_reviews.py enforces), so resolve each file against the
    real model names rather than a hand-maintained list. A hardcoded map silently
    dropped every model whose name contained a space and wasn't listed.
    """
    reviews = {}
    if not AI_REVIEWS_DIR.exists():
        return reviews

    # Stems that differ from the model name by more than space→underscore.
    SPECIAL = {"pi_star_0.6": "pi*0.6"}
    by_stem = {name.replace(" ", "_"): name for name in (model_names or [])}

    for md_file in sorted(AI_REVIEWS_DIR.glob("*.md")):
        stem = md_file.stem
        model_name = SPECIAL.get(stem) or by_stem.get(stem, stem)
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        verified = "VERIFIED: pdf" in content
        reviews[model_name] = {
            "content": content,
            "verified": verified,
        }
    return reviews


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
    # Compute from individual suite scores. The long-horizon suite is written
    # both as libero_long and libero_10 (its official name) across papers; a
    # model using only the latter had to declare libero_avg by hand or it
    # dropped off the leaderboard entirely.
    scores = []
    for keys in [("libero_spatial",), ("libero_object",), ("libero_goal",),
                 ("libero_long", "libero_10")]:
        for key in keys:
            val = libero.get(key)
            if isinstance(val, (int, float)):
                scores.append(val)
                break
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


def declared(block: dict, *keys) -> float | None:
    """First numeric value among `keys`, the paper-reported headline average."""
    for key in keys:
        val = block.get(key)
        if isinstance(val, (int, float)):
            return round(val, 2)
    return None


def mean_of_parts(block: dict) -> float | None:
    """Mean of every numeric score in a benchmark block.

    Only reached when the block declares no headline average. Other *_avg keys
    are left in deliberately: a key like colosseum_all_variations_avg summarizes
    a sibling eval condition, not the other keys here, so dropping it would
    silently swing the model toward whichever condition it reports in full.
    """
    scores = [v for k, v in block.items()
              if k not in METADATA_KEYS and isinstance(v, (int, float))]
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def compute_simpler_avg(benchmarks: dict) -> float | None:
    simpler = benchmarks.get("simpler_env")
    if not simpler:
        return None
    # Prefer the paper's own headline average, as every other benchmark here
    # does. Averaging it in with its own components inflated TBD-VLA from its
    # reported 77.7 to 82.09 and moved it up the SimplerEnv ranking.
    return declared(simpler, "simpler_avg") or mean_of_parts(simpler)


def compute_robotwin_v1_avg(benchmarks: dict) -> float | None:
    robotwin = benchmarks.get("robotwin_v1")
    if not robotwin:
        return None
    return declared(robotwin, "average", "robotwin_v1_avg") or mean_of_parts(robotwin)


def compute_robotwin_v2_avg(benchmarks: dict) -> float | None:
    robotwin_v2 = benchmarks.get("robotwin_v2")
    if not robotwin_v2:
        return None
    return declared(robotwin_v2, "robotwin_v2_avg", "average") or mean_of_parts(robotwin_v2)


def compute_metaworld_avg(benchmarks: dict) -> float | None:
    metaworld = benchmarks.get("metaworld")
    if not metaworld:
        return None
    # Papers headline this as metaworld_avg, ml45_avg or metaworld_mt50_avg.
    return declared(metaworld, "metaworld_avg", "ml45_avg", "metaworld_mt50_avg",
                    "average") or mean_of_parts(metaworld)


def compute_rlbench_avg(benchmarks: dict) -> float | None:
    rlbench = benchmarks.get("rlbench")
    if not rlbench:
        return None
    # rlbench_18tasks is the standard 18-task protocol and is how four models
    # headline their score; without it the mean swept in COLOSSEUM / GemBench /
    # MemoryBench suites that happen to share the block.
    return (declared(rlbench, "rlbench_avg", "rlbench_18tasks", "average")
            or mean_of_parts(rlbench))



def compute_robocasa_avg(benchmarks: dict) -> float | None:
    robocasa = benchmarks.get("robocasa")
    if not robocasa:
        return None
    return declared(robocasa, "robocasa_avg", "average") or mean_of_parts(robocasa)


def build_leaderboard(models: list[dict], benchmarks_meta: dict[str, dict],
                      reviews: dict[str, dict] | None = None,
                      ai_reviews: dict[str, dict] | None = None) -> dict:
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
                "action_head_category": model.get("architecture", {}).get("action_head_category", "other"),
                "parameters": model.get("architecture", {}).get("parameters", "unknown"),
            },
            "model_type": model.get("model_type"),
            "inference_hz": model.get("inference_hz"),
            "benchmarks": {},
            "eval_conditions": {},
        }

        model_benchmarks = fold_libero_variants(hoist_root_result_blocks(model))

        # Carry every scored block, not just a hardcoded list. The old list left
        # out real_world — which has its own definition file and appears in
        # benchmarks_available — so 59 models' real-robot numbers never reached
        # leaderboard.json at all, along with driving, maniskill and a dozen
        # other blocks written into the YAMLs.
        ordered = [b for b in CANONICAL_BENCHMARKS if b in model_benchmarks]
        ordered += [b for b in model_benchmarks if b not in CANONICAL_BENCHMARKS]
        for bench_name in ordered:
            block = model_benchmarks[bench_name]
            if not isinstance(block, dict):
                continue
            scores, meta = extract_scores(block)
            if scores:
                entry["benchmarks"][bench_name] = scores
            if meta.get("eval_condition"):
                entry["eval_conditions"][bench_name] = meta["eval_condition"]

        # Also carry over eval_conditions from model YAML. 22 models write these
        # as structured dicts ({setting, base_model, note}); the old str-only
        # check silently dropped every one of them. Flatten dicts to the display
        # string the dashboard's classifier and tooltips expect.
        model_eval = model.get("eval_conditions", {})
        for bench_name, cond in model_eval.items():
            if isinstance(cond, str):
                entry["eval_conditions"][bench_name] = cond
            elif isinstance(cond, dict):
                parts = [str(cond[k]) for k in ("setting", "base_model") if cond.get(k)]
                parts += [f"{k}: {v}" for k, v in cond.items()
                          if k not in ("setting", "base_model") and isinstance(v, (str, int, float))]
                if parts:
                    entry["eval_conditions"][bench_name] = "; ".join(parts)

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

        metaworld_avg = compute_metaworld_avg(model_benchmarks)
        if metaworld_avg is not None:
            entry["metaworld_avg"] = metaworld_avg

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

        # Merge AI review if available
        if ai_reviews and model["name"] in ai_reviews:
            entry["ai_review"] = ai_reviews[model["name"]]

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

    print("Loading AI reviews...")
    ai_reviews = load_ai_reviews([m.get("name") for m in models if m.get("name")])
    print(f"  Found {len(ai_reviews)} AI reviews")

    print("Building leaderboard...")
    leaderboard = build_leaderboard(models, benchmarks_meta, reviews, ai_reviews)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Leaderboard written to {OUTPUT_FILE}")

    # Sync to dashboard.
    # The dashboard copy is statically imported by App.jsx, so it lands in the
    # main JS bundle. Review bodies are ~92% of the payload, which every visitor
    # would download just to see the leaderboard. Strip them here and emit them
    # as separately fetched assets; data/leaderboard.json (the public API) keeps
    # the full content.
    if DASHBOARD_COPY.parent.exists():
        slim = copy.deepcopy(leaderboard)
        for entry in slim["models"]:
            review = entry.get("ai_review")
            if review:
                entry["ai_review"] = {"verified": review.get("verified", False)}

        with open(DASHBOARD_COPY, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)
        print(f"Synced slim copy to {DASHBOARD_COPY}")

        if PUBLIC_DIR.exists():
            bodies = {
                entry["name"]: entry["ai_review"]["content"]
                for entry in leaderboard["models"]
                if entry.get("ai_review", {}).get("content")
            }
            with open(REVIEWS_BUNDLE, "w", encoding="utf-8") as f:
                json.dump(bodies, f, ensure_ascii=False)

            REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
            for name, content in bodies.items():
                (REVIEWS_DIR / f"{review_slug(name)}.md").write_text(
                    content, encoding="utf-8"
                )
            print(f"  {len(bodies)} review bodies → {REVIEWS_BUNDLE.name} + {REVIEWS_DIR.name}/")

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
