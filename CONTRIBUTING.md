# Contributing to VLA-Tracker

Thanks for your interest in contributing! Here's how you can help.

## Adding a New Model (Easiest Contribution)

1. Create a new YAML file: `data/models/your_model.yaml`
2. Use this template:

```yaml
name: ModelName
version: "1.0"
organization: "Lab / Company"
paper_url: https://arxiv.org/abs/XXXX.XXXXX
code_url: https://github.com/org/repo  # or null
date: "YYYY-MM-DD"
architecture:
  backbone: "Vision encoder description"
  llm: "Language model used"
  action_head: "autoregressive / diffusion / flow matching / hybrid"
  action_space: "continuous / discrete"
  parameters: "7B"
training:
  dataset: "Dataset name"
  compute: "Hardware description"
benchmarks:
  libero:
    libero_spatial: 0.0
    libero_object: 0.0
    libero_goal: 0.0
    libero_long: 0.0
    source: "Table X, paper name"
    date_reported: "YYYY-MM-DD"
open_source: true
tags: [tag1, tag2]
```

3. Validate: `python scripts/validate_data.py`
4. Submit a pull request

## Adding Benchmark Results

If a model already exists but is missing results for a benchmark:

1. Edit the model's YAML in `data/models/`
2. Add the benchmark section with scores and source
3. Always include `source` and `date_reported` fields

## Reporting Errors

If you find incorrect benchmark numbers:

1. Open an issue with:
   - Which model and benchmark
   - The correct number
   - Source (paper table/figure reference)

## Development Setup

```bash
# Data scripts
pip install pyyaml
python scripts/validate_data.py
python scripts/build_leaderboard.py

# Dashboard
cd dashboard
npm install
npm run dev
```

## Guidelines

- All benchmark numbers must come from published papers or official reports
- Include source references for every number
- Use consistent YAML formatting
- Run validation before submitting PRs
