#!/usr/bin/env python3
"""
Scan arXiv for new VLA (Vision-Language-Action) papers.

Uses the arXiv API to search for recent papers matching VLA-related keywords.
Filters against already-tracked models and outputs candidates as JSON.

Usage:
    python scripts/scan_arxiv.py [--days 14] [--output candidates.json]
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"

ARXIV_API = "http://export.arxiv.org/api/query"

# Search queries targeting VLA papers
SEARCH_QUERIES = [
    '"vision-language-action"',
    '"VLA" AND "robot"',
    '"vision language action" AND "manipulation"',
    '"VLA" AND ("LIBERO" OR "CALVIN" OR "SimplerEnv" OR "RoboTwin")',
    '"action chunking" AND "vision language"',
    '"robotic manipulation" AND ("flow matching" OR "diffusion policy")',
]

# Keywords that indicate a paper is VLA-related (in title or abstract)
VLA_KEYWORDS = [
    r"\bVLA\b",
    r"vision.language.action",
    r"robotic.manipulation",
    r"action.prediction",
    r"action.chunking",
    r"flow.matching.*(robot|action|manipulat)",
    r"diffusion.policy",
    r"(robot|manipulat).*(language|vision).*(model|policy)",
    r"\bLIBERO\b",
    r"\bCALVIN\b",
    r"\bSimpler.?Env\b",
    r"\bRoboTwin\b",
    r"\bOpen.?X\b",
]

# Benchmark keywords to detect in abstract
BENCHMARK_PATTERNS = {
    "libero": r"\bLIBERO\b",
    "calvin": r"\bCALVIN\b",
    "simpler_env": r"\bSimpler.?Env\b",
    "robotwin": r"\bRoboTwin\b",
    "rlbench": r"\bRLBench\b",
    "metaworld": r"\bMeta.?World\b",
}

# Venue patterns in comments
VENUE_PATTERNS = [
    (r"ICLR\s*20\d{2}", "ICLR"),
    (r"NeurIPS\s*20\d{2}", "NeurIPS"),
    (r"ICML\s*20\d{2}", "ICML"),
    (r"CVPR\s*20\d{2}", "CVPR"),
    (r"ICCV\s*20\d{2}", "ICCV"),
    (r"ECCV\s*20\d{2}", "ECCV"),
    (r"CoRL\s*20\d{2}", "CoRL"),
    (r"RSS\s*20\d{2}", "RSS"),
    (r"ICRA\s*20\d{2}", "ICRA"),
    (r"IROS\s*20\d{2}", "IROS"),
]

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def get_existing_paper_urls() -> set[str]:
    """Load all paper URLs from existing model YAMLs."""
    import yaml

    urls = set()
    names = set()
    for yaml_file in MODELS_DIR.glob("*.yaml"):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        if data.get("paper_url"):
            urls.add(data["paper_url"])
            # Also extract arXiv ID
            m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", data["paper_url"])
            if m:
                urls.add(m.group(1))
        if data.get("name"):
            names.add(data["name"].lower().strip())
    return urls, names


def query_arxiv(search_query: str, max_results: int = 50, start: int = 0) -> str:
    """Query arXiv API and return XML response."""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VLA-Tracker/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  [WARN] Failed to query arXiv: {e}", file=sys.stderr)
                return ""


def parse_arxiv_response(xml_text: str) -> list[dict]:
    """Parse arXiv API XML response into a list of paper dicts."""
    if not xml_text:
        return []

    papers = []
    root = ET.fromstring(xml_text)

    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id_raw = entry.findtext(f"{ATOM_NS}id", "")
        arxiv_id = arxiv_id_raw.split("/abs/")[-1].split("v")[0] if "/abs/" in arxiv_id_raw else ""

        title = entry.findtext(f"{ATOM_NS}title", "").strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)

        summary = entry.findtext(f"{ATOM_NS}summary", "").strip().replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary)

        published = entry.findtext(f"{ATOM_NS}published", "")
        updated = entry.findtext(f"{ATOM_NS}updated", "")

        authors = []
        for author_el in entry.findall(f"{ATOM_NS}author"):
            name = author_el.findtext(f"{ATOM_NS}name", "")
            affil = author_el.findtext(f"{ARXIV_NS}affiliation", "")
            authors.append({"name": name, "affiliation": affil})

        comment = entry.findtext(f"{ARXIV_NS}comment", "") or ""
        categories = [
            cat.get("term", "")
            for cat in entry.findall(f"{ATOM_NS}category")
        ]

        # Extract PDF link
        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "published": published,
            "updated": updated,
            "comment": comment,
            "categories": categories,
            "pdf_url": pdf_url,
            "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
        })

    return papers


def is_vla_related(paper: dict) -> bool:
    """Check if a paper is VLA-related based on title and abstract."""
    text = f"{paper['title']} {paper['summary']}"
    for pattern in VLA_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def extract_venue(paper: dict) -> str | None:
    """Try to detect venue from comment field."""
    comment = paper.get("comment", "")
    text = f"{comment} {paper.get('title', '')}"
    for pattern, venue_name in VENUE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def detect_benchmarks(paper: dict) -> list[str]:
    """Detect which benchmarks are mentioned in the abstract."""
    text = f"{paper['title']} {paper['summary']}"
    found = []
    for bench_name, pattern in BENCHMARK_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(bench_name)
    return found


def extract_model_name(paper: dict) -> str | None:
    """Try to extract the model name from the title."""
    title = paper["title"]

    # Pattern: "ModelName: Description"
    m = re.match(r"^([A-Z][A-Za-z0-9\-\.]+(?:\s*[\-\+]+)?)\s*[:]\s", title)
    if m:
        return m.group(1).strip()

    # Pattern: "ModelName — Description" or "ModelName - Description"
    m = re.match(r"^([A-Z][A-Za-z0-9\-\.]+)\s*[—–\-]\s", title)
    if m:
        return m.group(1).strip()

    return None


def compute_relevance_score(paper: dict) -> float:
    """Compute a relevance score for ranking candidates."""
    score = 0.0
    text = f"{paper['title']} {paper['summary']}"

    # Direct VLA mention
    if re.search(r"\bVLA\b", text):
        score += 3.0
    if re.search(r"vision.language.action", text, re.IGNORECASE):
        score += 3.0

    # Benchmark mentions
    benchmarks = detect_benchmarks(paper)
    score += len(benchmarks) * 1.5

    # Has a clear model name
    if extract_model_name(paper):
        score += 1.0

    # Accepted at venue
    if extract_venue(paper):
        score += 2.0

    # Code available mention
    if re.search(r"github\.com|code.*(avail|releas)|open.sourc", text, re.IGNORECASE):
        score += 1.0

    # Robotics-specific categories
    for cat in paper.get("categories", []):
        if cat in ("cs.RO", "cs.AI", "cs.CV", "cs.LG"):
            score += 0.5

    return score


def filter_by_date(papers: list[dict], days: int) -> list[dict]:
    """Filter papers to those published within the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for p in papers:
        try:
            pub_date = datetime.fromisoformat(p["published"].replace("Z", "+00:00"))
            if pub_date >= cutoff:
                filtered.append(p)
        except (ValueError, TypeError):
            filtered.append(p)  # Keep if date parsing fails
    return filtered


def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by arXiv ID."""
    seen = set()
    unique = []
    for p in papers:
        aid = p["arxiv_id"]
        if aid and aid not in seen:
            seen.add(aid)
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Scan arXiv for new VLA papers")
    parser.add_argument("--days", type=int, default=14, help="Look back N days (default: 14)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--min-score", type=float, default=3.0, help="Minimum relevance score")
    parser.add_argument("--max-results", type=int, default=50, help="Max results per query")
    args = parser.parse_args()

    print(f"Scanning arXiv for VLA papers (last {args.days} days)...")

    # Load existing models
    existing_urls, existing_names = get_existing_paper_urls()
    print(f"  {len(existing_urls)} existing paper URLs/IDs tracked")
    print(f"  {len(existing_names)} existing model names tracked")

    # Query arXiv with multiple search terms
    all_papers = []
    for i, query in enumerate(SEARCH_QUERIES):
        print(f"  Query {i+1}/{len(SEARCH_QUERIES)}: {query[:60]}...")
        xml = query_arxiv(f"all:{query}", max_results=args.max_results)
        papers = parse_arxiv_response(xml)
        all_papers.extend(papers)
        time.sleep(3)  # arXiv rate limiting: 1 request per 3 seconds

    # Deduplicate
    all_papers = deduplicate(all_papers)
    print(f"  {len(all_papers)} unique papers found")

    # Filter by date
    all_papers = filter_by_date(all_papers, args.days)
    print(f"  {len(all_papers)} papers within last {args.days} days")

    # Filter to VLA-related only
    vla_papers = [p for p in all_papers if is_vla_related(p)]
    print(f"  {len(vla_papers)} VLA-related papers")

    # Filter out already-tracked papers
    new_papers = []
    for p in vla_papers:
        arxiv_id = p["arxiv_id"]
        paper_url = p["paper_url"]
        model_name = extract_model_name(p)

        if arxiv_id in existing_urls or paper_url in existing_urls:
            continue
        if model_name and model_name.lower().strip() in existing_names:
            continue
        new_papers.append(p)

    print(f"  {len(new_papers)} new (untracked) papers")

    # Score and rank candidates
    candidates = []
    for p in new_papers:
        score = compute_relevance_score(p)
        if score < args.min_score:
            continue

        model_name = extract_model_name(p) or "[needs manual extraction]"
        venue = extract_venue(p)
        benchmarks = detect_benchmarks(p)

        # Extract first author organization
        orgs = list({a["affiliation"] for a in p["authors"] if a["affiliation"]})

        candidates.append({
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "model_name": model_name,
            "paper_url": p["paper_url"],
            "published": p["published"][:10],
            "venue": venue,
            "organizations": orgs[:3],
            "benchmarks_mentioned": benchmarks,
            "relevance_score": round(score, 1),
            "authors_first3": [a["name"] for a in p["authors"][:3]],
            "abstract_snippet": p["summary"][:300] + "...",
        })

    # Sort by relevance score
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

    print(f"\n{'='*60}")
    print(f"Found {len(candidates)} new VLA paper candidates:")
    print(f"{'='*60}")

    for i, c in enumerate(candidates[:20]):
        print(f"\n  [{i+1}] {c['model_name']} (score={c['relevance_score']})")
        print(f"      {c['title'][:80]}")
        print(f"      arXiv: {c['arxiv_id']}  |  Date: {c['published']}")
        if c["venue"]:
            print(f"      Venue: {c['venue']}")
        if c["benchmarks_mentioned"]:
            print(f"      Benchmarks: {', '.join(c['benchmarks_mentioned'])}")

    # Output JSON
    output_data = {
        "scan_date": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.days,
        "total_queried": len(all_papers),
        "vla_related": len(vla_papers),
        "new_candidates": len(candidates),
        "candidates": candidates,
    }

    output_path = args.output or str(ROOT / "data" / "scan_candidates.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")
    return 0 if candidates else 1


if __name__ == "__main__":
    sys.exit(main())
