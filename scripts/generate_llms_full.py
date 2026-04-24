#!/usr/bin/env python3
"""
Generate llms-full.txt for AI agent consumption.
Contains complete model data, rankings, and summaries in plain text.
Deployed to GitHub Pages at /llms-full.txt
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "data" / "leaderboard.json"
OUTPUT = ROOT / "dashboard" / "public" / "llms-full.txt"


def main():
    with open(LEADERBOARD) as f:
        data = json.load(f)

    models = data["models"]
    lines = []

    # Header
    lines.append("# VLA-Tracker: Complete Model Database")
    lines.append(f"# Generated: {data['generated_at']}")
    lines.append(f"# Models: {data['num_models']}")
    lines.append(f"# Benchmarks: {', '.join(data['benchmarks_available'])}")
    lines.append(f"# Source: https://github.com/HyeongjinKim/Vla-tracker-")
    lines.append(f"# Dashboard: https://hyeongjinkim.github.io/Vla-tracker-/")
    lines.append("")
    lines.append("=" * 80)
    lines.append("SECTION 1: RANKINGS")
    lines.append("=" * 80)
    lines.append("")

    # LIBERO rankings
    lines.append("## LIBERO Benchmark Rankings (sorted by average)")
    lines.append("")
    libero = [m for m in models if m.get("libero_avg")]
    for i, m in enumerate(libero, 1):
        bench = m.get("benchmarks", {}).get("libero", {})
        spatial = bench.get("libero_spatial", "—")
        obj = bench.get("libero_object", "—")
        goal = bench.get("libero_goal", "—")
        long = bench.get("libero_long", "—")
        lines.append(f"  {i:2d}. {m['name']:25s} avg={m['libero_avg']:5.1f}  "
                      f"spatial={spatial}  object={obj}  goal={goal}  long={long}  "
                      f"org={m['organization']}  date={m.get('date','?')}")
    lines.append("")

    # CALVIN rankings
    lines.append("## CALVIN Benchmark Rankings (sorted by avg length)")
    lines.append("")
    calvin = sorted([m for m in models if m.get("calvin_avg")],
                    key=lambda x: x["calvin_avg"], reverse=True)
    for i, m in enumerate(calvin, 1):
        lines.append(f"  {i:2d}. {m['name']:25s} avg_len={m['calvin_avg']:4.2f}  "
                      f"org={m['organization']}  date={m.get('date','?')}")
    lines.append("")

    # SimplerEnv rankings
    lines.append("## SimplerEnv Rankings")
    lines.append("")
    simpler = sorted([m for m in models if m.get("simpler_avg")],
                     key=lambda x: x["simpler_avg"], reverse=True)
    for i, m in enumerate(simpler, 1):
        lines.append(f"  {i:2d}. {m['name']:25s} avg={m['simpler_avg']:5.1f}  "
                      f"org={m['organization']}")
    lines.append("")

    # RoboTwin rankings
    for ver in ["robotwin_v1", "robotwin_v2"]:
        key = f"{ver}_avg"
        ranked = sorted([m for m in models if m.get(key)],
                        key=lambda x: x[key], reverse=True)
        if ranked:
            lines.append(f"## {ver.replace('_',' ').title()} Rankings")
            lines.append("")
            for i, m in enumerate(ranked, 1):
                lines.append(f"  {i:2d}. {m['name']:25s} avg={m[key]:5.1f}")
            lines.append("")

    # RoboCasa / RLBench rankings (previously missing)
    for bench in ["robocasa", "rlbench"]:
        key = f"{bench}_avg"
        ranked = sorted([m for m in models if m.get(key)],
                        key=lambda x: x[key], reverse=True)
        if ranked:
            lines.append(f"## {bench.replace('_',' ').title()} Rankings")
            lines.append("")
            for i, m in enumerate(ranked, 1):
                lines.append(f"  {i:2d}. {m['name']:25s} avg={m[key]:5.1f}  "
                              f"org={m.get('organization', '?')}")
            lines.append("")

    lines.append("=" * 80)
    lines.append("SECTION 2: MODEL DETAILS")
    lines.append("=" * 80)
    lines.append("")

    # All models with full details
    for m in models:
        lines.append("-" * 60)
        lines.append(f"MODEL: {m['name']}")
        lines.append(f"  Organization: {m.get('organization', '?')}")
        lines.append(f"  Date: {m.get('date', '?')}")
        lines.append(f"  Paper: {m.get('paper_url', '—')}")
        lines.append(f"  Code: {m.get('code_url', '—')}")
        lines.append(f"  Venue: {m.get('venue', 'Preprint')}")
        lines.append(f"  Open Source: {m.get('open_source', False)}")

        arch = m.get("architecture", {})
        lines.append(f"  Architecture:")
        lines.append(f"    Action Head: {arch.get('action_head', '?')}")
        lines.append(f"    Backbone: {arch.get('backbone', '?')}")
        lines.append(f"    LLM: {arch.get('llm', '?')}")
        lines.append(f"    Parameters: {arch.get('parameters', '?')}")

        if m.get("inference_hz"):
            lines.append(f"  Inference: {m['inference_hz']} Hz")

        if m.get("tags"):
            lines.append(f"  Tags: {', '.join(m['tags'])}")

        # Benchmark scores
        benchmarks = m.get("benchmarks", {})
        if benchmarks:
            lines.append(f"  Benchmarks:")
            for bname, scores in benchmarks.items():
                avg_key = f"{bname}_avg"
                avg = m.get(avg_key)
                score_str = ", ".join(f"{k}={v}" for k, v in scores.items()
                                     if isinstance(v, (int, float)))
                avg_str = f" (avg={avg})" if avg else ""
                lines.append(f"    {bname}{avg_str}: {score_str}")

        # Eval conditions
        evals = m.get("eval_conditions", {})
        if evals:
            lines.append(f"  Eval Conditions: {evals}")

        # Peer review
        pr = m.get("peer_review")
        if pr:
            lines.append(f"  Peer Review:")
            lines.append(f"    Venue: {pr.get('venue')}, Decision: {pr.get('decision')}")
            lines.append(f"    Review Avg: {pr.get('review_avg')}, Confidence: {pr.get('confidence_avg')}")

        # AI review summary (just first line)
        ai = m.get("ai_review")
        if ai:
            content = ai.get("content", "")
            # Extract one-line summary
            for line in content.split("\n"):
                if "한 줄 요약" in line or line.startswith("> "):
                    summary = line.lstrip("> ").replace("**한 줄 요약**:", "").strip()
                    if summary and len(summary) > 20:
                        lines.append(f"  AI Review Summary: {summary[:300]}")
                        break
            lines.append(f"  AI Review: {'PDF-verified' if ai.get('verified') else 'Abstract-based'}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("SECTION 3: BENCHMARK DEFINITIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("LIBERO: 4 manipulation suites (Spatial, Object, Goal, Long-horizon)")
    lines.append("  Score range: 0-100% success rate. Higher is better.")
    lines.append("  Primary ranking metric for VLA-Tracker.")
    lines.append("")
    lines.append("CALVIN: Long-horizon language-conditioned manipulation (ABC→D setting)")
    lines.append("  Score: Average number of tasks completed in sequence (0-5). Higher is better.")
    lines.append("")
    lines.append("SimplerEnv: Sim-to-real transfer evaluation")
    lines.append("  Score range: 0-100% success rate. Higher is better.")
    lines.append("")
    lines.append("RoboTwin v1/v2: Bimanual dual-arm manipulation (50+ tasks)")
    lines.append("  Score range: 0-100% success rate. Higher is better.")
    lines.append("")
    lines.append("RLBench: Diverse manipulation tasks (18 tasks)")
    lines.append("  Score range: 0-100% success rate. Higher is better.")
    lines.append("")
    lines.append("RoboCasa: Home robot manipulation tasks")
    lines.append("  Score range: 0-100% success rate. Higher is better.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF FILE")
    lines.append(f"Total: {len(models)} models, {len(libero)} with LIBERO scores")
    lines.append("=" * 80)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated {OUTPUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
