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
| `check_reviews.py` | Check AI review coverage | (none) |
| `generate_llms_full.py` | Generate llms-full.txt | (none) |
| `push_to_hf.py` | Push dataset to HuggingFace | `--repo NAME`, `--dry-run` |
| `sync_counts.py` | Auto-fix README/blueprint counts | `--dry-run` |
| `apply_action_head_category.py` | Classify action heads | (none) |

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
The following files contain hardcoded counts that MUST match actual data.
`sync_counts.py` auto-updates most of these — run it after any model add/remove.

- **README.md** line ~15: Total models tracked `**N** VLA models · **M** benchmarks · **K** AI paper reviews` (auto-synced)
- **README.md** line ~59: LIBERO leaderboard heading `Full leaderboard with N LIBERO models` (auto-synced)
- **README.md** line ~65-70: "Other Benchmarks (Top 3)" table — verify top-3 per benchmark matches current leaderboard (NOT auto-synced; must be updated manually after benchmark score changes)
- **README.md** line ~181: Model YAML file count `# N model YAML files` (auto-synced)
- **physical-ai-dashboard-blueprint.md** line ~377: Benchmark table counts (auto-synced)

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

### Add a new model (FULL PIPELINE)

When a new model is discovered (via auto-track or manually), the **complete pipeline** must run:

```bash
# 1. Create YAML (auto-track does this, or manually)
python scripts/generate_model_yaml.py --top 1
# → action_head_category is auto-inferred from abstract

# 2. Read the paper and fill in:
#    - Exact benchmark scores (from paper tables)
#    - Architecture details (backbone, LLM, parameters)
#    - Organization, venue, open_source status
#    - Verify action_head_category is correct

# 3. Generate AI review (read paper PDF first!)
#    Place review in data/ai_reviews/{ModelName}.md
#    Add <!-- VERIFIED: pdf --> tag if numbers are from paper
#    Add <!-- VERIFIED: abstract-only --> if only abstract was read

# 4. Rebuild everything
python scripts/build_leaderboard.py    # YAML → JSON
python scripts/validate_data.py        # Check integrity + action_head_category
python scripts/sync_counts.py          # Auto-fix README/blueprint counts
python scripts/generate_llms_full.py   # Regenerate llms-full.txt

# 5. Validate
python scripts/validate_counts.py      # Verify all counts match
python scripts/check_reviews.py        # Check review coverage
```

**What the session-start hook checks automatically:**
- Builds leaderboard, validates data, syncs counts, checks review coverage
- If new models lack reviews, it will alert you

### Scan for new papers
```bash
python scripts/scan_arxiv.py --days 7
python scripts/generate_model_yaml.py --input data/scan_candidates.json --top 10
```

### AI Paper Reviews

AI-generated seminar-style reviews are stored in `data/ai_reviews/`. Each review follows a structured format (1-12 sections) suitable for academic seminars.

```bash
# Check review coverage (which models are missing reviews)
python scripts/check_reviews.py

# Reviews are markdown files in data/ai_reviews/
# Referenced from data/paper_reviews.json (ai_review field)
```

**Review verification workflow:**
1. Read the paper PDF (via arxiv)
2. Verify/correct experimental numbers, ablation tables, architecture details
3. Add `<!-- VERIFIED: pdf -->` tag at end of review file when verified
4. Reviews without this tag are flagged by `check_reviews.py`

**When adding a new model:**
- After creating the YAML, also generate an AI review for the paper
- Read the paper PDF for accurate numbers
- Place review in `data/ai_reviews/{ModelName}.md`

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
