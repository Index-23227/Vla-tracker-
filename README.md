<div align="center">

# VLA-Tracker

### The most comprehensive benchmark tracker for Vision-Language-Action models

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Visit-blue?style=for-the-badge)](https://hyeongjinkim.github.io/Vla-tracker-/)
[![Models](https://img.shields.io/badge/Models-220-purple?style=flat-square)](data/models/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-8-green?style=flat-square)](data/benchmarks/)
[![AI Reviews](https://img.shields.io/badge/Paper_Reviews-216-orange?style=flat-square)](data/ai_reviews/)
[![Auto-Track](https://img.shields.io/badge/Auto--Scan-arXiv_weekly-red?style=flat-square)](.github/workflows/auto-track.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![llms.txt](https://img.shields.io/badge/llms.txt-available-brightgreen?style=flat-square)](https://hyeongjinkim.github.io/Vla-tracker-/llms.txt)

**220 VLA models** · **8 benchmarks** · **216 AI paper reviews** · **Auto-updated from arXiv**

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
| Paper reviews | None | **216 AI-generated seminar-style reviews** |
| Machine-readable | None | **JSON API + llms.txt for AI agents** |

---

## Current SOTA Rankings

### LIBERO (Primary Benchmark)

| Rank | Model | Avg | Date | Venue | Action Head |
|------|-------|-----|------|-------|-------------|
| 1 | **LaST-R1** | **99.8** | Apr 2026 | — | AR latent CoT + parallel discrete action tokens |
| 2 | **QuoVLA** | **99.6** | May 2026 | — | Flow matching + 8-bit quantized prefix (pi0.5 base) |
| 3 | **PriorVLA** | **99.05** | May 2026 | — | Dual-expert flow matching (pi0.5 backbone) |
| 4 | PLD | 99.0 | Oct 2025 | ICLR 2026 | Residual RL |
| 5 | **MPCoT** | **98.85** | Jun 2026 | arXiv | OFT parallel-decoding + multi-path latent CoT |
| 6 | SimpleVLA-RL | 98.83 | Sep 2025 | ICLR 2026 | AR + GRPO RL |
| 7 | DualCoT-VLA | 98.8 | Mar 2026 | — | Flow matching + parallel CoT |
| 8 | SnapFlow | 98.75 | Apr 2026 | — | One-step flow distillation (pi0.5 base) |
| 9 | **3DThinkVLA** | **98.65** | Jun 2026 | arXiv | OFT regression + 3D-thinking-guided co-training |
| 10 | IntentVLA | 98.62 | May 2026 | — | DiT flow-matching + VGGT short-horizon intent |
| 11 | AttenA+ | 98.6 | May 2026 | arXiv | Inverse-velocity loss reweighting (plug-in) |
| 12 | DiT4DiT | 98.55 | Mar 2026 | — | Cosmos DiT + flow-matching action |
| 13 | G3T-AML | 98.55 | May 2026 | arXiv | 16-layer DiT on learned action manifold |

> Full leaderboard with 106 LIBERO models → [Dashboard](https://hyeongjinkim.github.io/Vla-tracker-/) or [JSON](data/leaderboard.json)

### Other Benchmarks (Top 3)

| Benchmark | #1 | #2 | #3 |
|-----------|-----|-----|-----|
| **CALVIN** (avg len) | MPCoT (4.92) | MMaDA-VLA (4.78) | UD-VLA (4.64) |
| **SimplerEnv** (avg) | GTA-VLA (81.2) | InstructVLA (80.3) | Retrieve-then-Steer (79.5) |
| **RoboTwin v1** (avg) | Fast-WAM (91.8) | Qwen-VLA (86.7) | FineVLA (84.7) |
| **RoboTwin v2** (avg) | MotuBrain (96.0) | STARRY (93.6) | AttenA+ (92.5) |
| **RoboCasa** (avg) | StereoPolicy (75.6) | DIAL (70.2) | UniT (66.0) |

---

## Paper Reviews

VLA-Tracker includes **216 AI-generated seminar-style paper reviews** for every tracked model. Each review covers:

1. One-line summary
2. Background & motivation
3. Methodology deep-dive (with equations and Q&A)
4. Experimental results (exact numbers from papers)
5. Ablation analysis
6. Related work comparison tables
7. Limitations & attribution issues
8. Overall assessment with ratings
9. Expected tough questions for seminars

**216 reviews are PDF-verified** — numbers cross-checked against actual paper tables.

Browse reviews: [Dashboard Reviews Tab](https://hyeongjinkim.github.io/Vla-tracker-/) · [Markdown files](data/ai_reviews/)

---

## For AI Agents & Developers

### Machine-Readable Data

| Format | URL | Description |
|--------|-----|-------------|
| **llms.txt** | [/llms.txt](https://hyeongjinkim.github.io/Vla-tracker-/llms.txt) | Concise overview for AI agents ([standard](https://llmstxt.org/)) |
| **llms-full.txt** | [/llms-full.txt](https://hyeongjinkim.github.io/Vla-tracker-/llms-full.txt) | Complete model database (4500+ lines) |
| **JSON API** | [/leaderboard.json](https://hyeongjinkim.github.io/Vla-tracker-/leaderboard.json) | Structured JSON with all scores + AI reviews |
| **YAML** | [data/models/](data/models/) | Individual model files (220 files) |

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
| [LIBERO](data/benchmarks/libero.yaml) | 4 suites | 106 | Manipulation generalization | NeurIPS 2023 |
| [CALVIN](data/benchmarks/calvin.yaml) | ABC→D | 29 | Long-horizon, language | RA-L 2022 |
| [SimplerEnv](data/benchmarks/simpler_env.yaml) | 5 tasks | 37 | Sim-to-real transfer | NeurIPS 2024 |
| [RoboTwin v1/v2](data/benchmarks/robotwin.yaml) | 50+ tasks | 37 | Bimanual manipulation | CVPR 2025 |
| [RLBench](data/benchmarks/rlbench.yaml) | 18 tasks | 6 | Diverse manipulation | RA-L 2020 |
| [RoboCasa](data/benchmarks/robocasa.yaml) | Various | 18 | Home robot tasks | RSS 2024 |
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
│   ├── models/              # 220 model YAML files (source of truth)
│   ├── benchmarks/          # 8 benchmark definitions
│   ├── ai_reviews/          # 216 AI-generated paper reviews (markdown)
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
- AI reviews are generated by Claude and may contain inaccuracies (216/216 are PDF-verified)
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
