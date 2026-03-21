# VLA-Tracker

### Real-time benchmark tracking for Vision-Language-Action models

[![Models Tracked](https://img.shields.io/badge/models-77-blue)](data/models/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-8-green)](data/benchmarks/)
[![Auto-Track](https://img.shields.io/badge/paper%20scan-arXiv%20%2B%20Semantic%20Scholar-orange)](.github/workflows/auto-track.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## What is this?

VLA-Tracker tracks **77 VLA models** (2023-2026) across **8 benchmarks** (LIBERO, CALVIN, SimplerEnv, RoboTwin v1/v2, Meta-World, RLBench, RoboCasa, Real-World). New papers are auto-detected from arXiv and Semantic Scholar via GitHub Actions.

**Unlike static awesome-lists**, this project:
- **Tracks performance across multiple benchmarks** with normalized, comparable scores
- **Auto-discovers new papers** via arXiv + Semantic Scholar scanning (twice weekly)
- **Validates data integrity** on every PR with automated CI checks
- **Interactive dashboard** to filter, compare, and explore models

## Leaderboard

### LIBERO (Top 10)

| Rank | Model | Date | Venue | Action Head | Avg | Eval |
|------|-------|------|-------|-------------|-----|------|
| 1 | **PLD** | Oct 2025 | ICLR 2026 | residual RL (probe → learn → distill) | **99.0** | FT |
| 2 | **SimpleVLA-RL** | Sep 2025 | ICLR 2026 | autoregressive + GRPO RL | **98.8** | FT |
| 3 | **X-VLA** | Oct 2025 | ICLR 2026 | autoregressive w/ soft prompts | **97.8** | FT |
| 4 | Fast-WAM | Mar 2026 | - | flow matching + video DiT | 97.6 | FT |
| 5 | VLA-Thinker | Mar 2026 | - | autoregressive + visual CoT + GRPO RL | 97.5 | FT |
| 6 | DreamVLA | Jul 2025 | - | inverse dynamics from world knowledge | 97.2 | FT |
| 7 | AtomicVLA | Mar 2026 | CVPR 2026 | flow matching + SG-MoE | 96.6 | FT |
| 8 | MemoryVLA | Aug 2025 | ICLR 2026 | diffusion transformer | 96.5 | FT |
| 9 | dVLA | Sep 2025 | ICLR 2026 | discrete diffusion + multimodal CoT | 96.4 | FT |
| 10 | GST-VLA | Mar 2026 | - | flow matching + Depth-Aware CoT | 96.3 | FT |

> **FT** = Fine-tuned | [Full leaderboard (JSON) ->](data/leaderboard.json)

### Other Benchmarks (Top 3)

| Benchmark | #1 | #2 | #3 |
|-----------|-----|-----|-----|
| **CALVIN** (avg len) | LingBot-VLA (4.5) | UD-VLA (4.5) | DreamVLA (4.44) |
| **SimplerEnv** (avg) | InstructVLA (80.3) | SpatialVLA (78.2) | X-VLA (78.05) |
| **RoboTwin v1** (avg) | Fast-WAM (91.8) | SimpleVLA-RL (70.4) | LingBot-VLA (61.5) |
| **RoboTwin v2** (avg) | Motus (87.84) | GigaWorld-Policy (86.0) | X-VLA (72.5) |

## All Tracked Models

<details>
<summary>77 models (click to expand)</summary>

3D Diffuser Actor, AtomicVLA, AVDC, CogACT, CoT-VLA, DexVLA, Diffusion Policy, dVLA, E0, FALCON, FAST-WAM, FLARE, FLOWER, GR-1, GR-2, GR00T-N1, GR00T-N1.5, GR00T-N1.6, GR00T-N1.7, GR00T-N2, GST-VLA, HPT, HybridVLA, InstructVLA, LingBot-VLA, MemoryVLA, NanoVLA, Octo, OpenVLA, OpenVLA-OFT, pi0, pi0.5, pi*0.6, pi0-FAST, RDT-1B, RoboVLM, RT-2-X, SimpleVLA-RL, SmolVLA, SpatialVLA, SuSIE, TraceVLA, UD-VLA, UniPI, UniVLA, VLA-Thinker, VLASER, X-VLA, *and more...*

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
| [LIBERO](data/benchmarks/libero.yaml) | 4 suites | 36 | Manipulation generalization | NeurIPS 2023 |
| [CALVIN](data/benchmarks/calvin.yaml) | ABC->D | 18 | Long-horizon, language | RA-L 2022 |
| [SimplerEnv](data/benchmarks/simpler_env.yaml) | 5 tasks | 15 | Sim-to-real transfer | NeurIPS 2024 |
| [RoboTwin v1/v2](data/benchmarks/robotwin.yaml) | 50 tasks | 11 | Dual-arm, bimanual | CVPR 2025 Highlight |
| [Meta-World](data/benchmarks/metaworld.yaml) | ML-10/45 | - | Multi-task dexterity | CoRL 2020 |
| [RLBench](data/benchmarks/rlbench.yaml) | 18 tasks | 3 | Diverse manipulation | RA-L 2020 |
| [RoboCasa](data/benchmarks/robocasa.yaml) | Various | 5 | Home robot tasks | RSS 2024 |

### LIBERO Leaderboard (36 models)

| Rank | Model | Organization | LIBERO Avg |
|------|-------|-------------|------------|
| 1 | PLD | NVIDIA GEAR Lab | 99.0 |
| 2 | SimpleVLA-RL | Shanghai Jiao Tong University / Peking University / Shanghai AI Lab / HKU | 98.83 |
| 3 | X-VLA | Tsinghua University AIR / Shanghai AI Lab | 97.8 |
| 4 | Fast-WAM | Tsinghua University (IIIS) / Galaxea AI | 97.6 |
| 5 | VLA-Thinker | University of Central Florida / University of Wurzburg / USC / NVIDIA Research | 97.45 |
| 6 | DreamVLA | Shanghai AI Lab / Fudan University / NUS | 97.2 |
| 7 | DeepThinkVLA | Huazhong University / Tsinghua University / Renmin University | 97.05 |
| 8 | AtomicVLA | Sun Yat-sen University / Peng Cheng Laboratory / Yinwang Intelligent Technology | 96.6 |
| 9 | MemoryVLA | PKU / Tsinghua University / Shanghai AI Lab | 96.5 |
| 10 | dVLA | Midea Group | 96.4 |
| 11 | GST-VLA | Yeungnam University / KAIST | 96.33 |
| 12 | DD-VLA | PKU / Shanghai AI Lab | 96.3 |
| 13 | pi*0.6 | Physical Intelligence | 93.1 |
| 14 | LingBot-VLA | Ant Group / Robbyant | 92.55 |
| 15 | UD-VLA | Tsinghua University / Shanghai AI Lab | 91.75 |
| 16 | ECoT | UC Berkeley / Stanford / University of Warsaw | 90.8 |
| 17 | InstructVLA | Shanghai AI Lab (InternRobotics) | 90.6 |
| 18 | UniVLA | HKU / OpenDriveLab / AgiBot | 90.1 |
| 19 | FLARE | NVIDIA | 89.2 |
| 20 | pi0.5 | Physical Intelligence | 88.88 |
| 21 | SpatialVLA | Shanghai AI Lab / Tsinghua University | 88.05 |
| 22 | DexVLA | ByteDance / Tsinghua University | 87.85 |
| 23 | OpenVLA-OFT | Stanford / UC Berkeley / TRI | 87.35 |
| 24 | CoT-VLA | UC Berkeley | 86.32 |
| 25 | pi0-FAST | Physical Intelligence | 86.1 |
| 26 | CogACT | Microsoft Research / Tsinghua University | 85.9 |
| 27 | GR00T-N1 | NVIDIA | 85.75 |
| 28 | HybridVLA | ByteDance | 85.75 |
| 29 | pi0 | Physical Intelligence | 84.75 |
| 30 | RoboVLM | Tsinghua University / ByteDance Research / CASIA / SJTU / NUS | 84.05 |
| 31 | RDT-1B | Tsinghua University / Shanghai Qi Zhi Institute | 82.5 |
| 32 | SmolVLA | Hugging Face | 79.5 |
| 33 | HPT | MIT / Meta FAIR | 77.5 |
| 34 | OpenVLA | Stanford / UC Berkeley / TRI / Google DeepMind / PI / MIT | 74.5 |
| 35 | Octo | UC Berkeley / Stanford / CMU / Google DeepMind | 67.17 |
| 36 | Diffusion Policy | Columbia University / MIT / TRI | 66.75 |

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
│   ├── benchmarks/          # Benchmark definitions (10 YAML files)
│   ├── models/              # Model data with benchmark scores (77 YAML files)
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
