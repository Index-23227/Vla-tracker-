# VLA-Tracker

### Real-time benchmark tracking for Vision-Language-Action models

[![Models Tracked](https://img.shields.io/badge/models-84-blue)](data/models/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-9-green)](data/benchmarks/)
[![Auto-Track](https://img.shields.io/badge/paper%20scan-arXiv%20%2B%20Semantic%20Scholar-orange)](.github/workflows/auto-track.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## What is this?

VLA-Tracker tracks **84 VLA models** (2023-2026) across **9 benchmarks** (LIBERO, CALVIN, SimplerEnv, RoboTwin v1/v2, Meta-World, RLBench, ManiSkill, VLABench, RoboCasa). New papers are auto-detected from arXiv and Semantic Scholar via GitHub Actions.

**Unlike static awesome-lists**, this project:
- **Tracks performance across multiple benchmarks** with normalized, comparable scores
- **Auto-discovers new papers** via arXiv + Semantic Scholar scanning (twice weekly)
- **Validates data integrity** on every PR with automated CI checks
- **Interactive dashboard** to filter, compare, and explore models

## Leaderboard

### LIBERO (Top 10)

| Rank | Model | Date | Venue | Action Head | Avg | Eval |
|------|-------|------|-------|-------------|-----|------|
| 1 | **PLD** | Nov 2025 | ICLR 2026 | residual RL (probe → learn → distill) | **99.0** | FT |
| 2 | **SimpleVLA-RL** | Sep 2025 | ICLR 2026 | autoregressive + GRPO RL | **98.8** | FT |
| 3 | **X-VLA** | Oct 2025 | ICLR 2026 | autoregressive w/ soft prompts | **97.8** | FT |
| 4 | Fast-WAM | Mar 2026 | - | flow matching + video DiT | 97.6 | FT |
| 5 | VLA-Thinker | Mar 2026 | - | autoregressive + visual CoT + GRPO RL | 97.5 | FT |
| 6 | DreamVLA | Jul 2025 | - | inverse dynamics from world knowledge | 97.2 | FT |
| 7 | AtomicVLA | Mar 2026 | CVPR 2026 | flow matching + SG-MoE | 96.6 | FT |
| 8 | MemoryVLA | Aug 2025 | ICLR 2026 | diffusion transformer | 96.5 | FT |
| 9 | dVLA | Sep 2025 | - | discrete diffusion + multimodal CoT | 96.4 | FT |
| 10 | GST-VLA | Mar 2026 | - | flow matching + Depth-Aware CoT | 96.3 | FT |

> **FT** = Fine-tuned | [Full leaderboard (JSON) ->](data/leaderboard.json)

### Other Benchmarks (Top 3)

| Benchmark | #1 | #2 | #3 |
|-----------|-----|-----|-----|
| **CALVIN** (avg len) | pi0.5-Pro (4.6) | LingBot-VLA (4.5) | UD-VLA (4.5) |
| **SimplerEnv** (avg) | OpenVLA-v2 (82.3) | InstructVLA (80.3) | SpatialVLA (78.2) |
| **RoboTwin v1** (avg) | SimpleVLA-RL (70.4) | pi0.5-Pro (65.5) | LingBot-VLA (61.5) |
| **RoboTwin v2** (avg) | X-VLA (72.5) | SimpleVLA-RL (68.8) | - |

## All Tracked Models

<details>
<summary>84 models (click to expand)</summary>

3D Diffuser Actor, AtomicVLA, AVDC, CogACT, CoT-VLA, DexVLA, Diffusion Policy, dVLA, E0, FALCON, FAST-VLA, FAST-WAM, FLARE, FLOWER, GR-1, GR-2, GR00T-N1, GR00T-N1.5, GR00T-N1.6, GR00T-N1.7, GR00T-N2, GST-VLA, HPT, HybridVLA, InstructVLA, LingBot-VLA, MemoryVLA, NanoVLA, Octo, OpenVLA, OpenVLA-OFT, OpenVLA-v2, pi0, pi0.5, pi0.5-Pro, pi0.6, pi0-FAST, RDT-1B, RoboVLM, RT-2-X, SimpleVLA-RL, SmolVLA, Sparse-VLA, SpatialVLA, SuSIE, TempoFit, TRA-VLA, TraceVLA, UD-VLA, UniPI, UniVLA, VLA-Thinker, VLASER, X-VLA, *and more...*

</details>

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

### Scan for New Papers

```bash
# Auto mode: tries arXiv first, falls back to Semantic Scholar
python scripts/scan_arxiv.py --days 14

# Force Semantic Scholar only
python scripts/scan_arxiv.py --source s2 --days 30

# Generate draft YAMLs for top candidates
python scripts/generate_model_yaml.py --top 5
```

## Tracked Benchmarks

| Benchmark | Tasks | Models Tracked | Focus | Venue |
|-----------|-------|----------------|-------|-------|
| [LIBERO](data/benchmarks/libero.yaml) | 4 suites | 25+ | Manipulation generalization | NeurIPS 2023 |
| [CALVIN](data/benchmarks/calvin.yaml) | ABC->D | 12+ | Long-horizon, language | RA-L 2022 |
| [SimplerEnv](data/benchmarks/simpler_env.yaml) | 5 tasks | 8+ | Sim-to-real transfer | NeurIPS 2024 |
| [RoboTwin v1/v2](data/benchmarks/robotwin.yaml) | 50 tasks | 5+ | Dual-arm, bimanual | CVPR 2025 Highlight |
| [Meta-World](data/benchmarks/metaworld.yaml) | ML-10/45 | - | Multi-task dexterity | CoRL 2020 |
| [RLBench](data/benchmarks/rlbench.yaml) | 18 tasks | 1 | Diverse manipulation | RA-L 2020 |

## Automation

Three GitHub Actions workflows keep the data up to date:

| Workflow | Schedule | Description |
|----------|----------|-------------|
| [auto-track.yml](.github/workflows/auto-track.yml) | Wed & Sat 10:00 UTC | Scans arXiv + Semantic Scholar for new VLA papers, creates PRs with draft YAMLs |
| [weekly-analysis.yml](.github/workflows/weekly-analysis.yml) | Mon 9:00 UTC | Validates data, builds leaderboard, deploys dashboard |
| [validate-pr.yml](.github/workflows/validate-pr.yml) | On PR | Validates YAML integrity and rebuilds leaderboard as dry run |

## Project Structure

```
├── data/
│   ├── benchmarks/          # Benchmark definitions (6 YAML files)
│   ├── models/              # Model data with benchmark scores (37 YAML files)
│   ├── leaderboard.json     # Auto-generated unified leaderboard
│   └── scan_candidates.json # Latest arXiv/S2 scan results
├── scripts/
│   ├── scan_arxiv.py        # Paper scanner (arXiv + Semantic Scholar)
│   ├── generate_model_yaml.py  # Draft YAML generator from scan results
│   ├── build_leaderboard.py # YAML -> JSON leaderboard builder
│   └── validate_data.py     # Data integrity checks
├── dashboard/               # React + Recharts interactive dashboard
└── .github/workflows/       # CI/CD automation (3 workflows)
```

## Contributing

The easiest way to contribute is to **add a new model**:

1. Create `data/models/your_model.yaml` (see existing files for format)
2. Run `python scripts/validate_data.py` to verify
3. Run `python scripts/build_leaderboard.py` to check rankings
4. Submit a PR — CI will validate automatically

## Limitations

- Benchmark numbers are manually verified from papers — errors possible
- Not all models report on all benchmarks (fair comparison is hard)
- Real-world performance != benchmark performance
- Evaluation conditions (fine-tuned vs zero-shot) vary across papers

## Citation

```bibtex
@misc{vla-tracker-2026,
  title={VLA-Tracker: Benchmark Dashboard for Vision-Language-Action Models},
  author={Hyeongjin Kim},
  year={2026},
  url={https://github.com/HyeongjinKim/Vla-tracker-}
}
```

## Acknowledgments

Built with data from the VLA research community.
Thanks to the authors of [LIBERO](https://arxiv.org/abs/2306.03310), [CALVIN](https://arxiv.org/abs/2112.03227), and all model papers tracked here.

---

[Star this repo](https://github.com/HyeongjinKim/Vla-tracker-) to stay updated
