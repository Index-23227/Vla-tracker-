# CLAUDE.md — VLA-Tracker Maintenance Guide

## Project Overview

VLA-Tracker tracks Vision-Language-Action (VLA) models and their benchmark scores.
Data lives in YAML files, gets compiled to `leaderboard.json`, and is displayed via a React dashboard.

## Quick Start (Agent Onboarding)

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd dashboard && npm install && cd ..

# 2. Build leaderboard from YAML sources
python scripts/build_leaderboard.py

# 3. Validate everything
python scripts/validate_data.py       # Data integrity
python scripts/validate_counts.py     # Count consistency

# 4. (Optional) Start dashboard dev server
cd dashboard && npm run dev
```

## Architecture

```
data/models/*.yaml              → Source of truth (one YAML per model)
data/benchmarks/*.yaml          → Benchmark definitions
schemas/model.schema.json       → JSON Schema for model YAMLs
schemas/benchmark.schema.json   → JSON Schema for benchmark YAMLs
scripts/build_leaderboard.py    → Compiles YAMLs → data/leaderboard.json
dashboard/                      → React + Vite app that reads leaderboard.json
```

## Data Schemas

Model and benchmark YAML files have formal JSON Schemas in `schemas/`. Use them for:
- **Validation**: Verify YAML structure before committing
- **Generation**: Agents can read the schema to produce correct YAML without guessing
- **IDE support**: Auto-completion and inline validation in editors that support JSON Schema

### Model YAML Required Fields
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model identifier (matches filename) |
| `organization` | string | Lab/company name |
| `date` | string | `YYYY-MM-DD` publication date |
| `architecture` | object | Must contain `action_head` |
| `benchmarks` | object | Keyed by benchmark name (can be `{}`) |

### Model YAML Optional Fields
`full_name`, `paper_url`, `code_url`, `venue`, `open_source` (bool), `model_type`, `inference_hz`, `tags` (array), `training`, `eval_conditions`

See `schemas/model.schema.json` for the complete specification with types, constraints, and examples.

## Scripts Reference

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `build_leaderboard.py` | Compile YAMLs → JSON | (none) |
| `validate_data.py` | Check YAML integrity | (none) |
| `validate_counts.py` | Check doc count sync | (none) |
| `generate_model_yaml.py` | Create YAML stubs | `--input FILE`, `--top N`, `--dry-run` |
| `scan_arxiv.py` | Find new VLA papers | `--days N`, `--source {arxiv,s2,auto}`, `--min-score F` |
| `restore_missing_models.py` | Recover from backup | `<path_to_old_leaderboard.json>` |

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
| RoboCasa    | `robocasa_avg`      | `benchmarks.robocasa` |

### Average Computation Rules
- **LIBERO**: `libero_5_suite_avg` → mean of 4 suites → `libero_avg` fallback
- **CALVIN**: `calvin_abc_d_avg_len` → `calvin_avg` fallback
- **Others**: mean of all numeric fields (excluding `source`, `date_reported`, `eval_condition`)
- **Ranking**: Primary sort by `libero_avg` descending; models without LIBERO sorted to end

## Dashboard Hardcoded Values

These values in the dashboard should stay consistent with the data:

- `CoverageHeatmap.jsx` line ~53: `.slice(0, 40)` — default visible models
- `AnalysisDashboard.jsx` line ~139: `.slice(0, 40)` — heatmap top N
- `AnalysisDashboard.jsx` line ~410: label text "Top 40 models"

## Common Workflows

### Add a new model
```bash
# 1. Create YAML from template (or manually using schemas/model.schema.json as reference)
python scripts/generate_model_yaml.py --top 1

# 2. Edit the generated YAML with actual scores

# 3. Rebuild & validate
python scripts/build_leaderboard.py
python scripts/validate_data.py
python scripts/validate_counts.py

# 4. Update README counts + leaderboard table
```

### Scan for new papers
```bash
python scripts/scan_arxiv.py --days 7
python scripts/generate_model_yaml.py --input data/scan_candidates.json --top 10
```

### Full rebuild from scratch
```bash
pip install -r requirements.txt
python scripts/build_leaderboard.py
python scripts/validate_data.py
python scripts/validate_counts.py
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `leaderboard.json` has wrong model count | Run `python scripts/build_leaderboard.py` |
| `validate_counts.py` reports mismatches | Update hardcoded counts in README.md / blueprint |
| YAML parse errors | Check 2-space indent, no tabs, scores are numbers not strings |
| Model missing from dashboard | Verify YAML exists, rebuild leaderboard, check dashboard reads updated JSON |
| arXiv scan returns 0 results | Check internet access; try `--source arxiv` or `--source s2` |

## Code Style
- Python: standard library preferred, type hints used
- Dashboard: React + Tailwind CSS + Recharts
- YAML: 2-space indent, benchmark scores as flat key-value pairs
