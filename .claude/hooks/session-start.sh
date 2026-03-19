#!/bin/bash
set -e

echo "=== VLA-Tracker: Setting up agent environment ==="

# 1. Python dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null

# 2. Node dependencies (dashboard)
echo "Installing dashboard dependencies..."
cd dashboard && npm install --prefer-offline --no-audit --silent 2>/dev/null && cd ..

# 3. Build leaderboard from source YAML
echo "Building leaderboard.json..."
python scripts/build_leaderboard.py > /dev/null 2>&1

# 4. Validate
echo "Validating data..."
python scripts/validate_data.py > /dev/null 2>&1

# 5. Summary
YAML_COUNT=$(ls -1 data/models/*.yaml 2>/dev/null | wc -l)
echo ""
echo "Setup complete:"
echo "  Models: $YAML_COUNT YAML files"
echo "  Python: $(python --version 2>&1)"
echo "  Node:   $(node --version 2>&1)"
echo ""
echo "Ready. See CLAUDE.md for workflows and schemas/ for data specs."
