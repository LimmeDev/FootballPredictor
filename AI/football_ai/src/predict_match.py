"""Predict outcome probabilities for a single fixture.

Usage
-----
$ python predict_match.py path/to/fixture.json
The JSON file **must** contain exactly the same feature names as the model
was trained on (see `data/training_features.parquet`).  A quick way to create
it is:

>>> import pandas as pd
>>> sample = pd.read_parquet('data/training_features.parquet').iloc[[0]]
>>> sample.to_json('fixture.json', orient='records')
>>> # edit the numbers/team-specific columns as needed

The script prints a JSON dict with probabilities for HomeWin / Draw / AwayWin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "football_predictor.txt"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_match.py fixture.json", file=sys.stderr)
        sys.exit(1)

    fixture_path = Path(sys.argv[1])
    if not fixture_path.exists():
        print(f"File not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    booster = lgb.Booster(model_file=str(MODEL_PATH))

    features = pd.read_json(fixture_path, orient="records")
    probs = booster.predict(features)[0]

    out = {
        "HomeWin": float(probs[0]),
        "Draw": float(probs[1]),
        "AwayWin": float(probs[2]),
    }
    print(json.dumps(out, indent=2)) 