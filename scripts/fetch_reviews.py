#!/usr/bin/env python3
"""
Fetch peer-review data for VLA models from multiple venue datasets on HuggingFace.

Supported venues:
  - ICLR 2026 (davidheineman/iclr-2026)       → full reviews
  - ICLR 2025 (babytreecc/ICLR2025review)      → full reviews
  - NeurIPS 2025 (davidheineman/neurips-2025)   → accept list only
  - CoLM 2025 (davidheineman/colm-2025)         → accept list only

Matching: model YAML paper_url → arXiv ID → fuzzy title match against venue data.
Output: data/paper_reviews.json
"""

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import yaml

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package required. Install with: pip install datasets")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
OUTPUT_FILE = ROOT / "data" / "paper_reviews.json"

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def extract_arxiv_id(url: str | None) -> str | None:
    """Extract arXiv ID from a URL like https://arxiv.org/abs/2511.15669"""
    if not url:
        return None
    m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', url)
    return m.group(1) if m else None


def normalize_title(title: str) -> str:
    """Lowercase, strip whitespace/punctuation for fuzzy comparison."""
    t = re.sub(r'[^\w\s]', '', title.lower())
    return ' '.join(t.split())


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def load_models() -> list[dict]:
    """Load all model YAMLs and return list with name, full_name, paper_url."""
    models = []
    for yaml_file in sorted(MODELS_DIR.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        models.append({
            "name": data.get("name", yaml_file.stem),
            "full_name": data.get("full_name", ""),
            "paper_url": data.get("paper_url"),
            "arxiv_id": extract_arxiv_id(data.get("paper_url")),
            "_file": yaml_file.name,
        })
    return models


# ──────────────────────────────────────────────────────────────
# Venue parsers
# ──────────────────────────────────────────────────────────────

class VenueParser:
    """Base class for venue-specific dataset parsers."""
    name: str = ""
    hf_dataset: str = ""
    hf_split: str = "main"
    has_reviews: bool = False

    def load(self):
        print(f"  Loading {self.name} from {self.hf_dataset}...")
        self.ds = load_dataset(self.hf_dataset, split=self.hf_split)
        print(f"    {len(self.ds)} papers loaded")

    def build_index(self) -> dict[str, list]:
        """Return {normalized_title: [row_indices]}."""
        idx = {}
        for i, row in enumerate(self.ds):
            t = normalize_title(self.get_title(row))
            idx.setdefault(t, []).append(i)
        return idx

    def get_title(self, row) -> str:
        raise NotImplementedError

    def parse_paper(self, row) -> dict:
        raise NotImplementedError


class ICLR2026Parser(VenueParser):
    name = "ICLR 2026"
    hf_dataset = "davidheineman/iclr-2026"
    hf_split = "main"
    has_reviews = True

    def get_title(self, row) -> str:
        return row.get("title", "")

    def parse_paper(self, row) -> dict:
        reviews_raw = row.get("reviews", []) or []
        reviews = []
        ratings = []
        confidences = []
        for r in reviews_raw:
            rating = r.get("rating")
            confidence = r.get("confidence")
            if rating is not None:
                ratings.append(rating)
            if confidence is not None:
                confidences.append(confidence)
            reviews.append({
                "rating": rating,
                "confidence": confidence,
                "summary": (r.get("summary") or "")[:500],
                "strengths": (r.get("strengths") or "")[:500],
                "weaknesses": (r.get("weaknesses") or "")[:500],
            })
        return {
            "venue": self.name,
            "decision": row.get("venue_type"),  # poster, spotlight, oral, reject
            "openreview_url": row.get("url"),
            "num_reviews": len(reviews),
            "review_avg": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "confidence_avg": round(sum(confidences) / len(confidences), 2) if confidences else None,
            "reviews": reviews,
        }


class ICLR2025Parser(VenueParser):
    name = "ICLR 2025"
    hf_dataset = "babytreecc/ICLR2025review"
    hf_split = "train"
    has_reviews = True

    def get_title(self, row) -> str:
        return row.get("title", "")

    def parse_paper(self, row) -> dict:
        # Ratings come as semicolon-separated string like "5;6;6;8"
        rating_str = row.get("rating", "")
        ratings = []
        if isinstance(rating_str, str) and rating_str:
            ratings = [int(x) for x in rating_str.split(";") if x.strip().isdigit()]

        confidence_str = row.get("confidence", "")
        confidences = []
        if isinstance(confidence_str, str) and confidence_str:
            confidences = [int(x) for x in confidence_str.split(";") if x.strip().isdigit()]

        # Parse individual reviews
        reviews_raw = row.get("Review", []) or []
        reviews = []
        for r in reviews_raw:
            rev_rating = r.get("rating", {}).get("value") if isinstance(r.get("rating"), dict) else r.get("rating")
            rev_confidence = r.get("confidence", {}).get("value") if isinstance(r.get("confidence"), dict) else r.get("confidence")
            rev_summary = r.get("summary", {}).get("value", "") if isinstance(r.get("summary"), dict) else (r.get("summary") or "")
            rev_strengths = r.get("strengths", {}).get("value", "") if isinstance(r.get("strengths"), dict) else (r.get("strengths") or "")
            rev_weaknesses = r.get("weaknesses", {}).get("value", "") if isinstance(r.get("weaknesses"), dict) else (r.get("weaknesses") or "")
            reviews.append({
                "rating": rev_rating,
                "confidence": rev_confidence,
                "summary": rev_summary[:500],
                "strengths": rev_strengths[:500],
                "weaknesses": rev_weaknesses[:500],
            })

        return {
            "venue": self.name,
            "decision": row.get("status"),
            "openreview_url": None,
            "num_reviews": len(reviews) or len(ratings),
            "review_avg": round(row["rating_avg"], 2) if row.get("rating_avg") else (
                round(sum(ratings) / len(ratings), 2) if ratings else None
            ),
            "confidence_avg": round(row["confidence_avg"], 2) if row.get("confidence_avg") else (
                round(sum(confidences) / len(confidences), 2) if confidences else None
            ),
            "reviews": reviews,
        }


class NeurIPS2025Parser(VenueParser):
    name = "NeurIPS 2025"
    hf_dataset = "davidheineman/neurips-2025"
    hf_split = "main"
    has_reviews = False

    def get_title(self, row) -> str:
        return row.get("name", "")

    def parse_paper(self, row) -> dict:
        return {
            "venue": self.name,
            "decision": "accepted",  # only accepted papers in dataset
            "openreview_url": row.get("virtualsite_url"),
            "num_reviews": 0,
            "review_avg": None,
            "confidence_avg": None,
            "reviews": [],
        }


class CoLM2025Parser(VenueParser):
    name = "CoLM 2025"
    hf_dataset = "davidheineman/colm-2025"
    hf_split = "main"
    has_reviews = False

    def get_title(self, row) -> str:
        return row.get("title", "")

    def parse_paper(self, row) -> dict:
        return {
            "venue": self.name,
            "decision": "accepted",
            "openreview_url": row.get("openreview_url"),
            "num_reviews": 0,
            "review_avg": None,
            "confidence_avg": None,
            "reviews": [],
        }


ALL_PARSERS = [
    ICLR2026Parser(),
    ICLR2025Parser(),
    NeurIPS2025Parser(),
    CoLM2025Parser(),
]

# ──────────────────────────────────────────────────────────────
# Matching engine
# ──────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.80


def match_models_to_venues(models: list[dict], parsers: list[VenueParser]) -> list[dict]:
    """Match model papers against venue datasets by title similarity."""
    results = []
    matched_count = 0

    # Build title indices for all venues
    venue_indices = []
    for parser in parsers:
        try:
            parser.load()
            idx = parser.build_index()
            venue_indices.append((parser, idx))
        except Exception as e:
            print(f"  WARNING: Failed to load {parser.name}: {e}")

    for model in models:
        model_title = model.get("full_name") or model.get("name") or ""
        if not model_title:
            continue

        norm_model_title = normalize_title(model_title)
        best_match = None
        best_score = 0.0

        for parser, idx in venue_indices:
            # Try exact normalized match first
            if norm_model_title in idx:
                row_i = idx[norm_model_title][0]
                row = parser.ds[row_i]
                best_match = parser.parse_paper(row)
                best_match["match_score"] = 1.0
                best_score = 1.0
                break

            # Fuzzy match
            for norm_t, row_indices in idx.items():
                sim = title_similarity(model_title, norm_t)  # compare against un-normalized for SequenceMatcher
                if sim > best_score:
                    best_score = sim
                    row = parser.ds[row_indices[0]]
                    best_match = parser.parse_paper(row)
                    best_match["match_score"] = round(sim, 3)

        entry = {
            "model_name": model["name"],
            "paper_url": model["paper_url"],
            "arxiv_id": model["arxiv_id"],
        }

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            entry.update(best_match)
            matched_count += 1
            print(f"  ✓ {model['name']} → {best_match['venue']} (score={best_score:.2f}, decision={best_match.get('decision')})")
        else:
            entry.update({
                "venue": None,
                "decision": None,
                "num_reviews": 0,
                "review_avg": None,
                "confidence_avg": None,
                "reviews": [],
                "match_score": round(best_score, 3) if best_score > 0 else 0,
            })
            if best_score > 0.5:
                print(f"  ~ {model['name']} → near miss (score={best_score:.2f})")

        results.append(entry)

    print(f"\n  Matched {matched_count}/{len(models)} models to venue papers")
    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("Loading model YAMLs...")
    models = load_models()
    print(f"  Found {len(models)} models")

    print("\nMatching against venue datasets...")
    results = match_models_to_venues(models, ALL_PARSERS)

    # Write output
    output = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "venues_searched": [p.name for p in ALL_PARSERS],
        "num_models": len(models),
        "num_matched": sum(1 for r in results if r.get("venue")),
        "papers": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nReview data written to {OUTPUT_FILE}")

    # Summary
    venue_counts = {}
    for r in results:
        v = r.get("venue")
        if v:
            venue_counts[v] = venue_counts.get(v, 0) + 1
    print("\nMatches by venue:")
    for v, c in sorted(venue_counts.items()):
        print(f"  {v}: {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
