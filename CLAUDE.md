# CLAUDE.md — VLA-Tracker Maintenance Guide

## Project Overview

VLA-Tracker tracks Vision-Language-Action (VLA) models and their benchmark scores.
Data lives in YAML files, gets compiled to `leaderboard.json`, and is displayed via a React dashboard.

## Architecture

```
data/models/*.yaml       → Source of truth (one YAML per model)
data/benchmarks/*.yaml   → Benchmark definitions
scripts/build_leaderboard.py → Compiles YAMLs → data/leaderboard.json
dashboard/               → React + Vite app that reads leaderboard.json
```

## Critical Rules: Adding or Removing Models

When you add, remove, or modify a model YAML file, you MUST follow this checklist:

### 1. Update leaderboard.json
```bash
python scripts/build_leaderboard.py
```

### 2. Run validation
```bash
python scripts/validate_data.py       # Data integrity
python scripts/validate_counts.py     # Count consistency
```

### 3. Update documentation counts
The following files contain hardcoded counts that MUST match actual data:

- **README.md** line ~104: Benchmark table `Models Tracked` column
- **README.md** line ~111: LIBERO leaderboard heading `(N models)`
- **README.md** line ~171: Model YAML file count `(N YAML files)`
- **README.md** line ~14: Total models tracked `**N** VLA models`
- **physical-ai-dashboard-blueprint.md** line ~377: Benchmark table counts

### 4. Update LIBERO leaderboard table
If the model has LIBERO scores, add/update it in the README.md LIBERO leaderboard table.
Keep the table sorted by LIBERO Avg descending.

## Benchmarks & Their Average Keys

| Benchmark   | leaderboard.json key | YAML section    |
|-------------|---------------------|-----------------|
| LIBERO      | `libero_avg`        | `benchmarks.libero` |
| CALVIN      | `calvin_avg`        | `benchmarks.calvin` |
| SimplerEnv  | `simpler_avg`       | `benchmarks.simpler_env` |
| RoboTwin v1 | `robotwin_v1_avg`   | `benchmarks.robotwin_v1` |
| RoboTwin v2 | `robotwin_v2_avg`   | `benchmarks.robotwin_v2` |
| RLBench     | `rlbench_avg`       | `benchmarks.rlbench` |
| ManiSkill   | `maniskill_avg`     | `benchmarks.maniskill` |
| VLABench    | `vlabench_avg`      | `benchmarks.vlabench` |
| RoboCasa    | `robocasa_avg`      | `benchmarks.robocasa` |

## Dashboard Hardcoded Values

These values in the dashboard should stay consistent with the data:

- `CoverageHeatmap.jsx` line ~53: `.slice(0, 40)` — default visible models
- `AnalysisDashboard.jsx` line ~139: `.slice(0, 40)` — heatmap top N
- `AnalysisDashboard.jsx` line ~410: label text "Top 40 models"

## Common Workflows

### Add a new model
```bash
# 1. Create YAML from template
python scripts/generate_model_yaml.py --top 1

# 2. Edit the generated YAML with actual scores

# 3. Rebuild & validate
python scripts/build_leaderboard.py
python scripts/validate_counts.py

# 4. Update README counts + leaderboard table
```

### Scan for new papers
```bash
python scripts/scan_arxiv.py --days 7
python scripts/generate_model_yaml.py --input data/scan_candidates.json --top 10
```

## Code Style
- Python: standard library preferred, type hints used
- Dashboard: React + Tailwind CSS + Recharts
- YAML: 2-space indent, benchmark scores as flat key-value pairs
