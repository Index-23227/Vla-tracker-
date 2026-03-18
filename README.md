# 🔬 VLA-Tracker

### Real-time benchmark tracking & AI-powered analysis for Vision-Language-Action models

<!-- TODO: Replace with actual dashboard screenshot -->
<!-- ![VLA-Tracker Dashboard](docs/assets/dashboard-screenshot.png) -->

[![Models Tracked](https://img.shields.io/badge/models-32-blue)](data/models/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-6-green)](data/benchmarks/)
[![Weekly Analysis](https://img.shields.io/badge/analysis-AI%20powered-purple)](analysis/weekly/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## What is this?

VLA-Tracker automatically tracks **32 VLA models** (2023–2026) across **6 benchmarks** (LIBERO, CALVIN, SimplerEnv, RoboTwin, Meta-World, RLBench) and generates **AI-powered analysis** every week. Each model includes publication venue (CoRL, RSS, ICML, ICLR, CVPR, etc.), evaluation conditions, and release date.

**Unlike static awesome-lists**, this project:
- **Tracks performance over time** — see how SOTA evolves week by week
- **AI-generated insights** — Claude analyzes why Model A beats Model B on specific benchmarks
- **Auto-updated** — GitHub Actions scrapes new results daily
- **Interactive dashboard** — filter, compare, and explore

> **Latest finding (Week 12, 2026):**
> "Flow matching VLAs now dominate 4/5 LIBERO suites. The last holdout — LIBERO-Long — shows chain-of-thought models (CoT-VLA) closing the gap at 17.1%p vs 23.4%p for pi0.5, suggesting explicit reasoning benefits long-horizon planning."

## Current Leaderboard (LIBERO)

| Rank | Model | Date | Venue | Action Head | Avg | Long | Eval |
|------|-------|------|-------|-------------|-----|------|------|
| 🥇 | pi0.5-Pro | Sep 2025 | — | flow matching | **93.3** | **82.4** | FT |
| 🥈 | LingBot-VLA | Jan 2026 | — | MoT + flow matching | 92.6 | 80.6 | FT |
| 🥉 | UD-VLA | Nov 2025 | ICLR 2026 | joint diffusion | 91.8 | 78.4 | FT |
| 4 | InstructVLA | Jul 2025 | ICLR 2026 | flow matching | 90.6 | 76.8 | FT |
| 5 | UniVLA | May 2025 | RSS 2025 | flow matching | 90.1 | 74.2 | FT |
| 6 | OpenVLA-v2 | Feb 2025 | — | flow matching | 90.0 | 74.6 | FT |
| 7 | TRA-VLA | Mar 2025 | — | flow matching + reasoning | 89.3 | 73.8 | FT |
| 8 | pi0.5 | Feb 2025 | CoRL 2025 | flow matching | 88.9 | 72.8 | FT |

> **FT** = Fine-tuned, **ZS** = Zero-shot. Dashboard shows all 32 models across 6 benchmarks (LIBERO, CALVIN, SimplerEnv, RoboTwin).

*Updated: 2026-03-18 · [Full leaderboard →](data/leaderboard.json)*

## Weekly AI Analysis

Latest: **[Week 12: Flow matching's dominance begins](analysis/weekly/)**

Key insights:
- **Action head paradigm shift**: 3/5 top models now use flow matching or diffusion (+12.3%p vs 6 months ago)
- **Long-horizon remains unsolved**: LIBERO-Long scores lag 15-25%p behind other suites across all models
- **Open-source catching up**: OpenVLA-OFT trails pi0.5 by only 1.5%p (was 6.8%p six months ago)

[Past reports →](analysis/weekly/)

## Quick Start

### Browse the Dashboard

```bash
cd dashboard
npm install
npm run dev
```

### Use the Data

```python
import yaml

with open('data/models/pi0_5.yaml') as f:
    model = yaml.safe_load(f)

print(model['benchmarks']['libero']['libero_long'])  # 72.8
```

### Build the Leaderboard

```bash
pip install pyyaml
python scripts/build_leaderboard.py
```

## Tracked Benchmarks

| Benchmark | Tasks | Models | Focus | Venue |
|-----------|-------|--------|-------|-------|
| [LIBERO](data/benchmarks/libero.yaml) | 4 suites | 20+ | Manipulation generalization | NeurIPS 2023 |
| [CALVIN](data/benchmarks/calvin.yaml) | ABC→D | 10+ | Long-horizon, language | RA-L 2022 |
| [SimplerEnv](data/benchmarks/simpler_env.yaml) | 5 tasks | 5+ | Sim-to-real transfer | NeurIPS 2024 |
| [RoboTwin](data/benchmarks/robotwin.yaml) | 50 tasks | 4 | Dual-arm, bimanual | CVPR 2025 Highlight |
| [Meta-World](data/benchmarks/metaworld.yaml) | ML-10/45 | — | Multi-task dexterity | CoRL 2020 |
| [RLBench](data/benchmarks/rlbench.yaml) | 18 tasks | 1 | Diverse manipulation | RA-L 2020 |

## How AI Analysis Works

1. GitHub Actions scrapes new papers & benchmarks daily
2. Claude (Sonnet) analyzes the delta vs existing data
3. Generates structured insights with specific numbers
4. Commits analysis to `/analysis/weekly/`
5. Dashboard rebuilds automatically

**Token budget**: ~50K tokens/week

## Project Structure

```
├── data/
│   ├── benchmarks/     # Benchmark definitions (YAML)
│   ├── models/         # Model data with benchmark scores (YAML)
│   └── leaderboard.json  # Auto-generated unified leaderboard
├── analysis/
│   ├── weekly/         # AI-generated weekly analysis
│   ├── model-cards/    # AI-generated model summaries
│   └── trends/         # Quarterly trend reports
├── scripts/
│   ├── build_leaderboard.py  # YAML → JSON conversion
│   └── validate_data.py      # Data integrity checks
├── dashboard/          # React + Recharts interactive dashboard
└── .github/workflows/  # Automation pipelines
```

## Contribute

We welcome contributions! The easiest way is to **add a new model**:

1. Create a YAML file in `data/models/your_model.yaml`
2. Follow the [schema guide](docs/data-format.md)
3. Run `python scripts/validate_data.py` to verify
4. Submit a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## Limitations

- Benchmark numbers are manually verified from papers — errors possible
- AI analysis may occasionally misinterpret results
- Not all models report on all benchmarks (fair comparison is hard)
- Real-world performance ≠ benchmark performance
- Updates depend on paper availability and our review cycle

## Citation

If you use this data in your research:

```bibtex
@misc{vla-tracker-2026,
  title={VLA-Tracker: AI-Powered Benchmark Dashboard for Physical AI},
  author={Hyeongjin Kim},
  year={2026},
  url={https://github.com/HyeongjinKim/Vla-tracker-}
}
```

## Acknowledgments

Built with data from the VLA research community.
Thanks to the authors of [LIBERO](https://arxiv.org/abs/2306.03310), [CALVIN](https://arxiv.org/abs/2112.03227), [VLA-Arena](https://arxiv.org/abs/2407.01511), and all model papers included here.

---

**Powered by Claude** · Analysis auto-generated weekly · [Star this repo](https://github.com/HyeongjinKim/Vla-tracker-) to stay updated
