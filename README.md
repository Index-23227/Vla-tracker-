<div align="center">

# VLA-Tracker

### The most comprehensive benchmark tracker for Vision-Language-Action models

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Visit-blue?style=for-the-badge)](https://hyeongjinkim.github.io/Vla-tracker-/)
[![Models](https://img.shields.io/badge/Models-121-purple?style=flat-square)](data/models/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-8-green?style=flat-square)](data/benchmarks/)
[![AI Reviews](https://img.shields.io/badge/Paper_Reviews-83-orange?style=flat-square)](data/ai_reviews/)
[![Auto-Track](https://img.shields.io/badge/Auto--Scan-arXiv_weekly-red?style=flat-square)](.github/workflows/auto-track.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![llms.txt](https://img.shields.io/badge/llms.txt-available-brightgreen?style=flat-square)](https://hyeongjinkim.github.io/Vla-tracker-/llms.txt)

**121 VLA models** · **8 benchmarks** · **83 AI paper reviews** · **Auto-updated from arXiv**

[Live Dashboard](https://hyeongjinkim.github.io/Vla-tracker-/) · [API (JSON)](https://hyeongjinkim.github.io/Vla-tracker-/leaderboard.json) · [llms.txt](https://hyeongjinkim.github.io/Vla-tracker-/llms.txt) · [Paper Reviews](#paper-reviews)

</div>

---

## Why VLA-Tracker?

The VLA (Vision-Language-Action) field is moving fast — new models every week, each claiming SOTA on different benchmarks with different eval conditions. **No one can keep track.**

VLA-Tracker solves this by:

| Feature | Static Awesome-Lists | VLA-Tracker |
|---------|---------------------|-------------|
| Benchmark scores | Rarely included | **Normalized across 8 benchmarks** |
| New paper discovery | Manual updates | **Auto-scanned from arXiv twice weekly** |
| Data validation | None | **CI checks on every PR** |
| Interactive exploration | None | **Full React dashboard with filters** |
| Paper reviews | None | **70 AI-generated seminar-style reviews** |
| Machine-readable | None | **JSON API + llms.txt for AI agents** |

---

## Current SOTA Rankings

### LIBERO (Primary Benchmark)

| Rank | Model | Avg | Date | Venue | Action Head |
|------|-------|-----|------|-------|-------------|
| 1 | **PLD** | **99.0** | Oct 2025 | ICLR 2026 | Residual RL |
| 2 | **SimpleVLA-RL** | **98.8** | Sep 2025 | ICLR 2026 | AR + GRPO RL |
| 3 | **X-VLA** | **97.8** | Oct 2025 | ICLR 2026 | Flow matching + soft prompts |
| 4 | Fast-WAM | 97.6 | Mar 2026 | — | Flow + video DiT |
| 5 | VLA-Thinker | 97.5 | Mar 2026 | — | AR + visual CoT + RL |
| 6 | DreamVLA | 97.2 | Jul 2025 | — | Inverse dynamics |
| 7 | DeepThinkVLA | 97.1 | Nov 2025 | ICLR 2026 | Hybrid attention |
| 8 | AtomicVLA | 96.6 | Mar 2026 | CVPR 2026 | Flow + SG-MoE |
| 9 | MemoryVLA | 96.5 | Aug 2025 | ICLR 2026 | Diffusion transformer |
| 10 | dVLA | 96.4 | Sep 2025 | ICLR 2026 | Discrete diffusion |

> Full leaderboard with 36 LIBERO models → [Dashboard](https://hyeongjinkim.github.io/Vla-tracker-/) or [JSON](data/leaderboard.json)

### Other Benchmarks (Top 3)

| Benchmark | #1 | #2 | #3 |
|-----------|-----|-----|-----|
| **CALVIN** (avg len) | LingBot-VLA (4.50) | UD-VLA (4.50) | DreamVLA (4.44) |
| **SimplerEnv** (avg) | InstructVLA (80.3) | SpatialVLA (78.2) | X-VLA (78.1) |
| **RoboTwin v1** (avg) | Fast-WAM (91.8) | SimpleVLA-RL (70.4) | LingBot-VLA (61.5) |
| **RoboTwin v2** (avg) | Motus (87.8) | GigaWorld-Policy (86.0) | X-VLA (72.5) |
| **RoboCasa** (avg) | FLARE (70.1) | DiT4DiT (50.8) | — |

---

## Paper Reviews

VLA-Tracker includes **83 AI-generated seminar-style paper reviews** for every tracked model. Each review covers:

1. One-line summary
2. Background & motivation
3. Methodology deep-dive (with equations and Q&A)
4. Experimental results (exact numbers from papers)
5. Ablation analysis
6. Related work comparison tables
7. Limitations & attribution issues
8. Overall assessment with ratings
9. Expected tough questions for seminars

**30 reviews are PDF-verified** — numbers cross-checked against actual paper tables.

Browse reviews: [Dashboard Reviews Tab](https://hyeongjinkim.github.io/Vla-tracker-/) · [Markdown files](data/ai_reviews/)

---

## For AI Agents & Developers

### Machine-Readable Data

| Format | URL | Description |
|--------|-----|-------------|
| **llms.txt** | [/llms.txt](https://hyeongjinkim.github.io/Vla-tracker-/llms.txt) | Concise overview for AI agents ([standard](https://llmstxt.org/)) |
| **llms-full.txt** | [/llms-full.txt](https://hyeongjinkim.github.io/Vla-tracker-/llms-full.txt) | Complete model database (1700+ lines) |
| **JSON API** | [/leaderboard.json](https://hyeongjinkim.github.io/Vla-tracker-/leaderboard.json) | Structured JSON with all scores + AI reviews |
| **YAML** | [data/models/](data/models/) | Individual model files (121 files) |

### Quick Start (Python)

```python
import json, urllib.request

# Fetch latest leaderboard
url = "https://hyeongjinkim.github.io/Vla-tracker-/leaderboard.json"
data = json.loads(urllib.request.urlopen(url).read())

# Top 5 LIBERO models
for m in data["models"][:5]:
    print(f"{m['name']}: {m.get('libero_avg', 'N/A')}")
```

### Quick Start (YAML)

```python
import yaml

with open("data/models/pi0_5.yaml") as f:
    model = yaml.safe_load(f)

print(model["benchmarks"]["libero"]["libero_long"])  # 72.8
print(model["architecture"]["action_head"])           # flow matching
```

---

## Dashboard

The interactive dashboard provides:

- **Leaderboard** — Sort by any benchmark, filter by action head / backbone / eval condition
- **Compare** — Radar charts comparing models side-by-side
- **Reviews** — AI-generated paper reviews with search & filter
- **Lineage** — Model family tree visualization
- **Efficiency** — Performance vs. parameters / inference speed
- **Coverage** — Heatmap of which models report which benchmarks
- **Architecture** — Breakdown by action head type, VLM backbone

```bash
# Run locally
cd dashboard && npm install && npm run dev
```

Or visit the **[live dashboard](https://hyeongjinkim.github.io/Vla-tracker-/)**.

---

## Tracked Benchmarks

| Benchmark | Tasks | Models | Focus | Venue |
|-----------|-------|--------|-------|-------|
| [LIBERO](data/benchmarks/libero.yaml) | 4 suites | 39 | Manipulation generalization | NeurIPS 2023 |
| [CALVIN](data/benchmarks/calvin.yaml) | ABC→D | 18 | Long-horizon, language | RA-L 2022 |
| [SimplerEnv](data/benchmarks/simpler_env.yaml) | 5 tasks | 16 | Sim-to-real transfer | NeurIPS 2024 |
| [RoboTwin v1/v2](data/benchmarks/robotwin.yaml) | 50+ tasks | 11 | Bimanual manipulation | CVPR 2025 |
| [RLBench](data/benchmarks/rlbench.yaml) | 18 tasks | 3 | Diverse manipulation | RA-L 2020 |
| [RoboCasa](data/benchmarks/robocasa.yaml) | Various | 5 | Home robot tasks | RSS 2024 |
| [Meta-World](data/benchmarks/metaworld.yaml) | ML-10/45 | — | Multi-task dexterity | CoRL 2020 |

---

## Automation

| Workflow | Schedule | Description |
|----------|----------|-------------|
| [auto-track.yml](.github/workflows/auto-track.yml) | Wed & Sat 10:00 UTC | Scans arXiv + Semantic Scholar, creates PRs with draft YAMLs |
| [weekly-analysis.yml](.github/workflows/weekly-analysis.yml) | Mon 9:00 UTC | Validates data, builds leaderboard, generates llms-full.txt, deploys dashboard |
| [validate-pr.yml](.github/workflows/validate-pr.yml) | On PR | Validates YAML integrity and count consistency |

---

## Project Structure

```
├── data/
│   ├── models/              # 121 model YAML files (source of truth)
│   ├── benchmarks/          # 8 benchmark definitions
│   ├── ai_reviews/          # 83 AI-generated paper reviews (markdown)
│   ├── leaderboard.json     # Auto-generated unified leaderboard
│   └── paper_reviews.json   # Venue peer-review data (ICLR, NeurIPS, CoLM)
├── scripts/
│   ├── scan_arxiv.py        # Paper scanner (arXiv + Semantic Scholar)
│   ├── build_leaderboard.py # YAML → JSON leaderboard builder
│   ├── generate_llms_full.py # Generates llms-full.txt for AI agents
│   ├── check_reviews.py     # AI review coverage tracker
│   └── validate_data.py     # Data integrity checks
├── dashboard/               # React + Tailwind + Recharts interactive dashboard
│   └── public/
│       ├── llms.txt         # AI agent discovery file
│       └── llms-full.txt    # Complete model database for AI
└── .github/workflows/       # CI/CD automation (3 workflows)
```

---

## Contributing

The easiest way to contribute is to **add a new model**:

1. Create `data/models/your_model.yaml` (see existing files or [schema](schemas/model.schema.json))
2. Run `python scripts/validate_data.py` to verify
3. Run `python scripts/build_leaderboard.py` to check rankings
4. Submit a PR — CI will validate automatically

We also welcome:
- Benchmark score corrections (with paper citations)
- New benchmark integrations
- Dashboard improvements

---

## Limitations

- Benchmark numbers are from papers — evaluation conditions vary (fine-tuned vs. zero-shot)
- Not all models report on all benchmarks (fair comparison is hard)
- AI reviews are generated by Claude and may contain inaccuracies (30/83 are PDF-verified)
- Real-world performance ≠ benchmark performance

---

## Citation

```bibtex
@misc{vla-tracker-2026,
  title={VLA-Tracker: Benchmark Dashboard for Vision-Language-Action Models},
  author={Hyeongjin Kim},
  year={2026},
  url={https://github.com/HyeongjinKim/Vla-tracker-}
}
```

---

<div align="center">

Built with data from the VLA research community.

[Star this repo](https://github.com/HyeongjinKim/Vla-tracker-) to stay updated · [Live Dashboard](https://hyeongjinkim.github.io/Vla-tracker-/) · [llms.txt](https://hyeongjinkim.github.io/Vla-tracker-/llms.txt)

</div>
