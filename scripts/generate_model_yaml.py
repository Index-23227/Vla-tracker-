#!/usr/bin/env python3
"""
Generate draft model YAML files from scan candidates.

Reads data/scan_candidates.json (output of scan_arxiv.py) and creates
draft YAML files in data/models/ for human review.

Usage:
    python scripts/generate_model_yaml.py [--input candidates.json] [--top 5]
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"


def sanitize_filename(name: str) -> str:
    """Convert model name to a safe filename."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def infer_tags(candidate: dict) -> list[str]:
    """Infer tags from candidate metadata."""
    tags = []
    abstract = candidate.get("abstract_snippet", "").lower()

    # Architecture tags
    if "flow matching" in abstract or "flow-matching" in abstract:
        tags.append("flow-matching")
    if "diffusion" in abstract:
        tags.append("diffusion")
    if "autoregressive" in abstract:
        tags.append("autoregressive")
    if "reinforcement learning" in abstract or " RL " in abstract:
        tags.append("RL")
    if "chain-of-thought" in abstract or "reasoning" in abstract:
        tags.append("reasoning")

    # Scale tags
    if re.search(r"\b\d+[bB]\b", abstract):
        m = re.search(r"\b(\d+\.?\d*)[bB]\b", abstract)
        if m:
            tags.append(f"{m.group(1)}B")

    # Property tags
    if "open-source" in abstract or "open source" in abstract:
        tags.append("open-source")
    if "cross-embodiment" in abstract:
        tags.append("cross-embodiment")
    if "dexterous" in abstract:
        tags.append("dexterous")
    if "bimanual" in abstract or "dual-arm" in abstract:
        tags.append("bimanual")

    # Benchmark tags
    for bench in candidate.get("benchmarks_mentioned", []):
        tags.append(bench)

    # Venue tag
    if candidate.get("venue"):
        tags.append(candidate["venue"].replace(" ", "-"))

    return tags


def infer_action_head(candidate: dict) -> str:
    """Try to infer action head type from abstract."""
    abstract = candidate.get("abstract_snippet", "").lower()

    if "flow matching" in abstract:
        return "flow matching"
    if "diffusion transformer" in abstract:
        return "diffusion transformer"
    if "diffusion policy" in abstract:
        return "diffusion policy"
    if "diffusion" in abstract:
        return "diffusion"
    if "autoregressive" in abstract:
        return "autoregressive"
    if "FAST" in candidate.get("abstract_snippet", ""):
        return "FAST tokenizer + autoregressive"
    return "TODO: check paper"


def generate_yaml_content(candidate: dict) -> str:
    """Generate YAML content string for a candidate."""
    model_name = candidate["model_name"]
    if model_name == "[needs manual extraction]":
        model_name = "TODO_MODEL_NAME"

    title = candidate["title"]
    orgs = " / ".join(candidate.get("organizations", [])) or "TODO: check paper"
    date = candidate["published"]
    paper_url = candidate["paper_url"]
    venue = candidate.get("venue") or "null"
    if venue != "null":
        venue = f'"{venue}"'

    tags = infer_tags(candidate)
    tags_str = ", ".join(f"{t}" for t in tags)
    action_head = infer_action_head(candidate)
    benchmarks_mentioned = candidate.get("benchmarks_mentioned", [])

    # Build benchmark section
    bench_lines = []
    for bench in benchmarks_mentioned:
        if bench == "libero":
            bench_lines.append(f"""  libero:
    libero_spatial: TODO
    libero_object: TODO
    libero_goal: TODO
    libero_long: TODO""")
        elif bench == "calvin":
            bench_lines.append(f"""  calvin:
    calvin_abc_d_avg_len: TODO""")
        elif bench == "simpler_env":
            bench_lines.append(f"""  simpler_env:
    google_robot_pick_coke_can: TODO""")
        elif bench == "robotwin":
            bench_lines.append(f"""  robotwin:
    robotwin_easy: TODO
    robotwin_hard: TODO""")
        else:
            bench_lines.append(f"""  {bench}:
    TODO: TODO""")

    if not bench_lines:
        bench_lines.append("""  libero:
    TODO: "check paper for benchmark scores" """)

    benchmarks_yaml = "\n".join(bench_lines)

    # Eval conditions
    eval_lines = []
    for bench in benchmarks_mentioned:
        eval_lines.append(f"  {bench}: \"fine-tuned\"")
    eval_yaml = "\n".join(eval_lines) if eval_lines else '  TODO: "check paper"'

    yaml_content = f"""# DRAFT - Auto-generated from arXiv scan. Needs human review.
# Paper: {title}
# arXiv: {candidate['arxiv_id']}

name: {model_name}
full_name: "{title}"
organization: "{orgs}"
date: "{date}"
paper_url: {paper_url}
code_url: null  # TODO: check for GitHub repo
venue: {venue}
open_source: false  # TODO: verify

architecture:
  action_head: "{action_head}"
  parameters: "TODO"

key_innovation: >
  TODO: Read paper and summarize key contribution in 2-3 sentences.

benchmarks:
{benchmarks_yaml}

eval_conditions:
{eval_yaml}

tags: [{tags_str}]
"""
    return yaml_content


def main():
    parser = argparse.ArgumentParser(description="Generate draft model YAMLs from scan candidates")
    parser.add_argument("--input", type=str, default=None, help="Input candidates JSON")
    parser.add_argument("--top", type=int, default=5, help="Generate for top N candidates")
    parser.add_argument("--dry-run", action="store_true", help="Print YAML without writing files")
    args = parser.parse_args()

    input_path = args.input or str(ROOT / "data" / "scan_candidates.json")

    if not Path(input_path).exists():
        print(f"Warning: {input_path} not found. No candidates to process.")
        return 0

    with open(input_path, "r") as f:
        data = json.load(f)

    candidates = data.get("candidates", [])
    if not candidates:
        print("No candidates found.")
        return 0

    top_candidates = candidates[:args.top]
    print(f"Generating YAML drafts for top {len(top_candidates)} candidates...\n")

    generated = []
    for candidate in top_candidates:
        model_name = candidate["model_name"]
        filename = sanitize_filename(model_name) + ".yaml"
        filepath = MODELS_DIR / filename

        # Skip if file already exists
        if filepath.exists():
            print(f"  [SKIP] {filename} already exists")
            continue

        yaml_content = generate_yaml_content(candidate)

        if args.dry_run:
            print(f"--- {filename} ---")
            print(yaml_content)
            print()
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            print(f"  [CREATED] {filepath.relative_to(ROOT)}")
            generated.append(filename)

    if generated:
        print(f"\n{len(generated)} draft YAML files created.")
        print("Please review and fill in TODO fields before committing.")
    elif not args.dry_run:
        print("\nNo new files generated (all candidates already tracked).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
