# Adding a New Model

## Step 1: Create YAML File

Create `data/models/your_model_name.yaml` (use lowercase, underscores).

Copy the template from [CONTRIBUTING.md](../CONTRIBUTING.md) and fill in the details.

## Step 2: Fill in Benchmark Data

For each benchmark your model was evaluated on, add a section under `benchmarks:`.

Currently supported benchmarks:
- `libero` — LIBERO (Spatial, Object, Goal, Long)
- `calvin` — CALVIN ABC→D
- `simpler_env` — SimplerEnv
- `metaworld` — Meta-World
- `rlbench` — RLBench

## Step 3: Validate

```bash
python scripts/validate_data.py
```

This checks:
- Required fields are present
- Scores are in valid range (0-100)
- YAML syntax is correct

## Step 4: Build Leaderboard

```bash
python scripts/build_leaderboard.py
```

Verify your model appears in the output.

## Step 5: Submit PR

Commit your YAML file and the updated `leaderboard.json`, then submit a pull request.
