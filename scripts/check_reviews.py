#!/usr/bin/env python3
"""
Check for models missing AI reviews and prepare review tasks.
Used by session-start hook and auto-track workflow to maintain review coverage.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
REVIEWS_DIR = ROOT / "data" / "ai_reviews"
REVIEWS_JSON = ROOT / "data" / "paper_reviews.json"


def sanitize_filename(name: str) -> str:
    """Convert model name to safe filename (matching existing convention)."""
    # Known name -> filename mappings for special cases
    SPECIAL = {
        "3D Diffuser Actor": "3D_Diffuser_Actor",
        "Diffusion Policy": "Diffusion_Policy",
        "Gemini Robotics": "Gemini_Robotics",
        "Mobility VLA": "Mobility_VLA",
        "pi*0.6": "pi_star_0.6",
    }
    if name in SPECIAL:
        return SPECIAL[name]
    return name.replace(" ", "_")


def get_models_from_yaml() -> list[dict]:
    """Load all model names and paper URLs from YAML files."""
    import yaml

    models = []
    for yf in sorted(MODELS_DIR.glob("*.yaml")):
        with open(yf) as f:
            data = yaml.safe_load(f)
        models.append({
            "name": data.get("name", yf.stem),
            "paper_url": data.get("paper_url", ""),
            "arxiv_id": "",
            "yaml_file": yf.name,
        })
        # Extract arxiv ID
        url = data.get("paper_url") or ""
        if url and "arxiv.org/abs/" in url:
            models[-1]["arxiv_id"] = url.split("arxiv.org/abs/")[-1].split("v")[0]
    return models


def get_existing_reviews() -> set[str]:
    """Get set of model names that already have AI reviews."""
    existing = set()
    if REVIEWS_DIR.exists():
        for md in REVIEWS_DIR.glob("*.md"):
            existing.add(md.stem)
    return existing


def check_review_coverage():
    """Check which models are missing AI reviews."""
    models = get_models_from_yaml()
    existing = get_existing_reviews()

    missing = []
    needs_update = []

    for m in models:
        fname = sanitize_filename(m["name"])
        review_path = REVIEWS_DIR / f"{fname}.md"

        if not review_path.exists():
            missing.append(m)
        else:
            # Check if review has been verified against PDF
            with open(review_path) as f:
                content = f.read()
            if "<!-- VERIFIED: pdf -->" not in content:
                needs_update.append(m)

    return {
        "total_models": len(models),
        "reviewed": len(models) - len(missing),
        "missing": missing,
        "needs_pdf_verification": needs_update,
    }


def update_paper_reviews_json(models_with_reviews: list[str]):
    """Ensure paper_reviews.json references all existing AI reviews."""
    if not REVIEWS_JSON.exists():
        return

    with open(REVIEWS_JSON) as f:
        data = json.load(f)

    updated = 0
    for paper in data.get("papers", []):
        fname = sanitize_filename(paper["model_name"])
        review_path = REVIEWS_DIR / f"{fname}.md"
        if review_path.exists() and "ai_review" not in paper:
            paper["ai_review"] = {
                "source": "ai_generated",
                "review_path": str(review_path.relative_to(ROOT)),
                "generated_by": "claude-opus-4-6",
            }
            updated += 1

    if updated:
        data["num_ai_reviewed"] = sum(
            1 for p in data["papers"] if "ai_review" in p
        )
        with open(REVIEWS_JSON, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Updated {updated} entries in paper_reviews.json")


def main():
    result = check_review_coverage()

    print(f"\n📝 AI Review Coverage: {result['reviewed']}/{result['total_models']} models")

    if result["missing"]:
        print(f"\n⚠️  {len(result['missing'])} models missing AI reviews:")
        for m in result["missing"][:10]:
            arxiv = f" (arxiv:{m['arxiv_id']})" if m["arxiv_id"] else ""
            print(f"    - {m['name']}{arxiv}")
        if len(result["missing"]) > 10:
            print(f"    ... and {len(result['missing']) - 10} more")

    unverified = len(result["needs_pdf_verification"])
    if unverified:
        print(f"\n📋 {unverified} reviews need PDF verification")
        print("    Run: claude 'Read and verify AI reviews against paper PDFs'")

    if not result["missing"] and unverified == 0:
        print("  ✅ All models reviewed and verified!")

    # Auto-update paper_reviews.json
    update_paper_reviews_json([])

    return 0 if not result["missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
