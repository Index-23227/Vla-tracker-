#!/usr/bin/env python3
"""
Auto-sync hardcoded counts in README.md and blueprint with actual data.
Fixes the counts that validate_counts.py reports as mismatches.

Run after adding/removing models:
    python scripts/sync_counts.py

Dry-run (show changes without writing):
    python scripts/sync_counts.py --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
LEADERBOARD = ROOT / "data" / "leaderboard.json"
README = ROOT / "README.md"
BLUEPRINT = ROOT / "physical-ai-dashboard-blueprint.md"
AI_REVIEWS_DIR = ROOT / "data" / "ai_reviews"


def count_yaml_files(directory: Path) -> int:
    return len(list(directory.glob("*.yaml")))


def count_benchmark_models(leaderboard: dict, avg_key: str) -> int:
    return sum(1 for m in leaderboard["models"] if m.get(avg_key) is not None)


def count_reviews() -> tuple[int, int]:
    """Return (total_reviews, pdf_verified)."""
    if not AI_REVIEWS_DIR.exists():
        return 0, 0
    total = 0
    verified = 0
    for md in AI_REVIEWS_DIR.glob("*.md"):
        total += 1
        content = md.read_text(encoding="utf-8")
        if "VERIFIED: pdf" in content:
            verified += 1
    return total, verified


def sync_readme(leaderboard: dict, dry_run: bool) -> list[str]:
    """Sync README.md counts. Returns list of changes made."""
    if not README.exists():
        return []

    text = README.read_text(encoding="utf-8")
    original = text
    changes = []
    num_models = leaderboard["num_models"]
    num_reviews, num_verified = count_reviews()
    num_libero = count_benchmark_models(leaderboard, "libero_avg")
    num_yamls = count_yaml_files(MODELS_DIR)

    # Badge: Models-XX
    new_text, n = re.subn(
        r"(Models-)\d+(-)",
        rf"\g<1>{num_models}\2",
        text,
    )
    if n and new_text != text:
        changes.append(f"Models badge: → {num_models}")
        text = new_text

    # Badge: Paper_Reviews-XX
    new_text, n = re.subn(
        r"(Paper_Reviews-)\d+(-)",
        rf"\g<1>{num_reviews}\2",
        text,
    )
    if n and new_text != text:
        changes.append(f"Reviews badge: → {num_reviews}")
        text = new_text

    # Inline: **XX VLA models**
    new_text, n = re.subn(
        r"\*\*\d+ VLA models\*\*",
        f"**{num_models} VLA models**",
        text,
    )
    if n and new_text != text:
        changes.append(f"Inline model count: → {num_models}")
        text = new_text

    # Inline: **XX AI paper reviews**
    new_text, n = re.subn(
        r"\*\*\d+ AI paper reviews\*\*",
        f"**{num_reviews} AI paper reviews**",
        text,
    )
    if n and new_text != text:
        changes.append(f"Inline reviews count: → {num_reviews}")
        text = new_text

    # Paper Reviews section: **XX AI-generated seminar-style**
    new_text, n = re.subn(
        r"\*\*\d+ AI-generated seminar-style paper reviews\*\*",
        f"**{num_reviews} AI-generated seminar-style paper reviews**",
        text,
    )
    if n and new_text != text:
        changes.append(f"Reviews section count: → {num_reviews}")
        text = new_text

    # PDF verified count: **XX reviews are PDF-verified**
    new_text, n = re.subn(
        r"\*\*\d+ reviews are PDF-verified\*\*",
        f"**{num_verified} reviews are PDF-verified**",
        text,
    )
    if n and new_text != text:
        changes.append(f"PDF verified count: → {num_verified}")
        text = new_text

    # Tracked Benchmarks table: LIBERO model count
    for bench_name, avg_key in [
        ("LIBERO", "libero_avg"), ("CALVIN", "calvin_avg"),
        ("SimplerEnv", "simpler_avg"), ("RLBench", "rlbench_avg"),
        ("RoboCasa", "robocasa_avg"),
    ]:
        actual = count_benchmark_models(leaderboard, avg_key)
        pattern = rf"(\[{bench_name}\][^\n]*?\|\s*)\d+(\s*\|)"
        new_text, n = re.subn(pattern, rf"\g<1>{actual}\2", text)
        if n and new_text != text:
            changes.append(f"Benchmark table {bench_name}: → {actual}")
            text = new_text

    # RoboTwin (unique models across v1+v2)
    rt_total = len(set(
        m["name"] for m in leaderboard["models"]
        if m.get("robotwin_v1_avg") is not None or m.get("robotwin_v2_avg") is not None
    ))
    pattern = r"(\[RoboTwin[^\]]*\][^\n]*?\|\s*)\d+(\s*\|)"
    new_text, n = re.subn(pattern, rf"\g<1>{rt_total}\2", text)
    if n and new_text != text:
        changes.append(f"Benchmark table RoboTwin: → {rt_total}")
        text = new_text

    # LIBERO Leaderboard heading: (XX models)
    new_text, n = re.subn(
        r"LIBERO Leaderboard \(\d+ models\)",
        f"LIBERO Leaderboard ({num_libero} models)",
        text,
    )
    if n and new_text != text:
        changes.append(f"LIBERO leaderboard heading: → {num_libero}")
        text = new_text

    # Project Structure: ## model YAML files
    new_text, n = re.subn(
        r"(\d+) model YAML files",
        f"{num_yamls} model YAML files",
        text,
    )
    if n and new_text != text:
        changes.append(f"Project structure YAML count: → {num_yamls}")
        text = new_text

    # Project Structure: ## AI-generated paper reviews
    new_text, n = re.subn(
        r"(\d+) AI-generated paper reviews \(markdown\)",
        f"{num_reviews} AI-generated paper reviews (markdown)",
        text,
    )
    if n and new_text != text:
        changes.append(f"Project structure reviews count: → {num_reviews}")
        text = new_text

    # Individual model files (## files)
    new_text, n = re.subn(
        r"Individual model files \(\d+ files\)",
        f"Individual model files ({num_yamls} files)",
        text,
    )
    if n and new_text != text:
        changes.append(f"Machine-readable YAML count: → {num_yamls}")
        text = new_text

    # AI reviews in limitations: (XX/XX are PDF-verified)
    new_text, n = re.subn(
        r"\d+/\d+ are PDF-verified",
        f"{num_verified}/{num_reviews} are PDF-verified",
        text,
    )
    if n and new_text != text:
        changes.append(f"Limitations PDF count: → {num_verified}/{num_reviews}")
        text = new_text

    if text != original:
        if not dry_run:
            README.write_text(text, encoding="utf-8")
        return changes
    return []


def sync_blueprint(leaderboard: dict, dry_run: bool) -> list[str]:
    """Sync blueprint counts."""
    if not BLUEPRINT.exists():
        return []

    text = BLUEPRINT.read_text(encoding="utf-8")
    original = text
    changes = []

    for bench_name, avg_key in [
        ("LIBERO", "libero_avg"), ("CALVIN", "calvin_avg"),
        ("SimplerEnv", "simpler_avg"), ("RLBench", "rlbench_avg"),
    ]:
        actual = count_benchmark_models(leaderboard, avg_key)
        pattern = rf"(\|\s*{bench_name}\s*\|[^|]*\|\s*)\d+\+?(\s*\|)"
        new_text, n = re.subn(pattern, rf"\g<1>{actual}\2", text)
        if n and new_text != text:
            changes.append(f"Blueprint {bench_name}: → {actual}")
            text = new_text

    if text != original:
        if not dry_run:
            BLUEPRINT.write_text(text, encoding="utf-8")
        return changes
    return []


def main():
    parser = argparse.ArgumentParser(description="Auto-sync documentation counts")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    print("=== VLA-Tracker Count Sync ===\n")

    if not LEADERBOARD.exists():
        print("ERROR: leaderboard.json not found. Run build_leaderboard.py first.")
        return 1

    with open(LEADERBOARD) as f:
        leaderboard = json.load(f)

    num_models = leaderboard["num_models"]
    num_reviews, num_verified = count_reviews()
    print(f"  Models: {num_models}")
    print(f"  Reviews: {num_reviews} ({num_verified} PDF-verified)")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'WRITE'}\n")

    all_changes = []

    readme_changes = sync_readme(leaderboard, args.dry_run)
    if readme_changes:
        print("README.md changes:")
        for c in readme_changes:
            print(f"  {c}")
        all_changes.extend(readme_changes)

    blueprint_changes = sync_blueprint(leaderboard, args.dry_run)
    if blueprint_changes:
        print("\nBlueprint changes:")
        for c in blueprint_changes:
            print(f"  {c}")
        all_changes.extend(blueprint_changes)

    if not all_changes:
        print("All counts are already in sync!")
    else:
        print(f"\n{'Would update' if args.dry_run else 'Updated'} {len(all_changes)} count(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
