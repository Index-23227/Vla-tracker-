#!/usr/bin/env python3
"""
Scan arXiv and Semantic Scholar for new VLA (Vision-Language-Action) papers.

Uses the arXiv API as primary source, with Semantic Scholar API as fallback.
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

ARXIV_API = "https://export.arxiv.org/api/query"

# Semantic Scholar API (no auth required, 100 requests/5min for unauthenticated)
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Search queries targeting VLA and video-based robot learning papers (arXiv format)
SEARCH_QUERIES = [
    '"vision-language-action"',
    '"VLA" AND "robot"',
    '"vision language action" AND "manipulation"',
    '"VLA" AND ("LIBERO" OR "CALVIN" OR "SimplerEnv" OR "RoboTwin")',
    '"action chunking" AND "vision language"',
    '"robotic manipulation" AND ("flow matching" OR "diffusion policy")',
    # Video-based action models (must generate actions, not just video)
    '"video prediction" AND "action" AND ("robot" OR "manipulation")',
    '"inverse dynamics" AND "video" AND ("robot" OR "manipulation")',
    '"subgoal" AND "video" AND ("robot" OR "manipulation" OR "action")',
]

# Semantic Scholar search queries (simpler keyword format)
S2_SEARCH_QUERIES = [
    "vision language action robot",
    "VLA robotic manipulation",
    "vision language action LIBERO CALVIN",
    "action chunking vision language model",
    "robotic manipulation flow matching diffusion policy",
    # Video-based action models
    "video prediction action robot manipulation",
    "inverse dynamics video robot manipulation",
    "subgoal image generation robot action policy",
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
    # Video-based action models (predict actions via video)
    r"video.predict.*(action|manipulat|control|policy)",
    r"inverse.dynamics.*(video|visual)",
    r"subgoal.*(image|video).*(action|policy|manipulat)",
    r"video.*(action.model|action.predict|action.generat)",
]

# Benchmark keywords to detect in abstract
BENCHMARK_PATTERNS = {
    "libero": r"\bLIBERO\b",
    "calvin": r"\bCALVIN\b",
    "simpler_env": r"\bSimpler.?Env\b",
    "robotwin": r"\bRoboTwin\b",
    "rlbench": r"\bRLBench\b",
    "metaworld": r"\bMeta.?World\b",
    "robocasa": r"\bRoboCasa\b",
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


def get_existing_paper_urls() -> tuple[set[str], set[str]]:
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


# ── arXiv API ──────────────────────────────────────────────


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
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [WARN] Failed to parse arXiv response: {e}", file=sys.stderr)
        return []

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


def scan_arxiv(max_results: int) -> list[dict]:
    """Scan arXiv for VLA papers. Returns list of paper dicts."""
    all_papers = []
    arxiv_ok = False
    for i, query in enumerate(SEARCH_QUERIES):
        print(f"  arXiv query {i+1}/{len(SEARCH_QUERIES)}: {query[:60]}...")
        xml = query_arxiv(f"all:{query}", max_results=max_results)
        papers = parse_arxiv_response(xml)
        if papers:
            arxiv_ok = True
        all_papers.extend(papers)
        time.sleep(3)  # arXiv rate limiting: 1 request per 3 seconds

    if not arxiv_ok:
        print("  [WARN] arXiv API returned no results for any query", file=sys.stderr)

    return all_papers, arxiv_ok


# ── Semantic Scholar API ───────────────────────────────────


def query_semantic_scholar(query: str, limit: int = 50, year_from: str | None = None) -> list[dict]:
    """Query Semantic Scholar API and return list of paper dicts."""
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,externalIds,title,abstract,authors,year,publicationDate,venue,openAccessPdf,citationCount",
    }
    if year_from:
        params["year"] = f"{year_from}-"

    url = f"{S2_API}?{urllib.parse.urlencode(params)}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VLA-Tracker/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("data", [])
        except Exception as e:
            if attempt < 2:
                wait = 3 * (attempt + 1)
                # Handle rate limiting
                if "429" in str(e):
                    wait = 10 * (attempt + 1)
                time.sleep(wait)
            else:
                print(f"  [WARN] Failed to query Semantic Scholar: {e}", file=sys.stderr)
                return []


def s2_paper_to_dict(paper: dict) -> dict:
    """Convert a Semantic Scholar paper response to our standard dict format."""
    arxiv_id = ""
    external_ids = paper.get("externalIds") or {}
    if external_ids.get("ArXiv"):
        arxiv_id = external_ids["ArXiv"]

    title = (paper.get("title") or "").strip()
    summary = (paper.get("abstract") or "").strip()

    # Parse publication date
    pub_date = paper.get("publicationDate") or ""
    if not pub_date and paper.get("year"):
        pub_date = f"{paper['year']}-01-01"

    authors = []
    for author in (paper.get("authors") or []):
        authors.append({
            "name": author.get("name", ""),
            "affiliation": "",
        })

    venue = paper.get("venue") or ""

    paper_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    if not paper_url and paper.get("paperId"):
        paper_url = f"https://www.semanticscholar.org/paper/{paper['paperId']}"

    pdf_url = ""
    if paper.get("openAccessPdf"):
        pdf_url = paper["openAccessPdf"].get("url", "")

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "authors": authors,
        "published": pub_date,
        "updated": pub_date,
        "comment": venue,
        "categories": [],
        "pdf_url": pdf_url,
        "paper_url": paper_url,
    }


def scan_semantic_scholar(days: int) -> list[dict]:
    """Scan Semantic Scholar for VLA papers. Returns list of paper dicts."""
    print("\n  Falling back to Semantic Scholar API...")
    all_papers = []

    # Compute year cutoff for S2 API
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    year_from = str(cutoff_date.year)

    for i, query in enumerate(S2_SEARCH_QUERIES):
        print(f"  S2 query {i+1}/{len(S2_SEARCH_QUERIES)}: {query[:60]}...")
        results = query_semantic_scholar(query, limit=50, year_from=year_from)
        for r in results:
            paper = s2_paper_to_dict(r)
            if paper["title"]:  # Skip empty results
                all_papers.append(paper)
        time.sleep(1)  # S2 rate limiting

    return all_papers


# ── Shared utilities ───────────────────────────────────────


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
            date_str = p["published"]
            if not date_str:
                filtered.append(p)
                continue
            # Handle various date formats
            date_str = date_str.replace("Z", "+00:00")
            if len(date_str) == 10:  # YYYY-MM-DD
                date_str += "T00:00:00+00:00"
            pub_date = datetime.fromisoformat(date_str)
            if pub_date >= cutoff:
                filtered.append(p)
        except (ValueError, TypeError):
            filtered.append(p)  # Keep if date parsing fails
    return filtered


def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by arXiv ID or title."""
    seen_ids = set()
    seen_titles = set()
    unique = []
    for p in papers:
        aid = p["arxiv_id"]
        title_key = p["title"].lower().strip()

        if aid:
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
        elif title_key:
            if title_key in seen_titles:
                continue
        seen_titles.add(title_key)
        unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Scan arXiv for new VLA papers")
    parser.add_argument("--days", type=int, default=14, help="Look back N days (default: 14)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--min-score", type=float, default=3.0, help="Minimum relevance score")
    parser.add_argument("--max-results", type=int, default=50, help="Max results per query")
    parser.add_argument("--source", choices=["auto", "arxiv", "s2"], default="auto",
                        help="Paper source: arxiv, s2 (Semantic Scholar), or auto (try arXiv first, fallback to S2)")
    args = parser.parse_args()

    print(f"Scanning for VLA papers (last {args.days} days)...")

    # Load existing models
    existing_urls, existing_names = get_existing_paper_urls()
    print(f"  {len(existing_urls)} existing paper URLs/IDs tracked")
    print(f"  {len(existing_names)} existing model names tracked")

    # Query papers from selected source
    source_used = "none"
    all_papers = []

    if args.source in ("auto", "arxiv"):
        arxiv_papers, arxiv_ok = scan_arxiv(args.max_results)
        all_papers.extend(arxiv_papers)
        if arxiv_ok:
            source_used = "arxiv"

    if args.source == "s2" or (args.source == "auto" and not all_papers):
        s2_papers = scan_semantic_scholar(args.days)
        all_papers.extend(s2_papers)
        source_used = "s2" if not all_papers else f"{source_used}+s2" if source_used != "none" else "s2"

    print(f"  Source: {source_used}")

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
            "abstract_snippet": p["summary"][:300] + "..." if p["summary"] else "",
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
        "source": source_used,
        "total_queried": len(all_papers),
        "vla_related": len(vla_papers),
        "new_candidates": len(candidates),
        "candidates": candidates,
    }

    output_path = args.output or str(ROOT / "data" / "scan_candidates.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_path}")
    # Always exit 0 — no new papers is not an error
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        # Write empty results so downstream steps don't fail
        output_path = ROOT / "data" / "scan_candidates.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(output_path, "w") as f:
            _json.dump({"scan_date": "", "lookback_days": 0, "total_queried": 0,
                         "vla_related": 0, "new_candidates": 0, "candidates": []}, f)
        sys.exit(0)
