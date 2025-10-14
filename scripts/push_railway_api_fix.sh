#!/usr/bin/env bash
set -euo pipefail

git add railway_production_api.py
git commit -m "Align Railway API with UI endpoint expectations"
git push
