"""Predict outcome probabilities for a single fixture.

Usage
-----
1. Legacy – provide a JSON file that already contains a full feature row:

$ python predict_match.py path/to/fixture.json

2. Convenience – just pass HomeTeam AwayTeam (and optional date).  The
   script will look up the most recent head-to-head in the historical
   feature table and reuse its engineered features.

$ python predict_match.py "Real Madrid" "Chelsea" 2026-05-21

If no direct H2H meeting exists in the dataset, the script falls back to
picking each team's most recent league match (same home/away side).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "football_predictor.txt"
FEATS_PATH = PROJECT_ROOT / "data" / "training_features.parquet"


def _prepare_features_from_h2h(home: str, away: str) -> pd.DataFrame:
    """Get the most recent *engineered* feature row for *home vs away*.

    Falls back to each team's most recent match if the exact pairing is
    missing.  Drops non-model columns and returns a 1-row DataFrame.
    """

    df = pd.read_parquet(FEATS_PATH)

    # try exact most-recent head-to-head first
    mask = (df["HomeTeam"].str.lower() == home.lower()) & (
        df["AwayTeam"].str.lower() == away.lower()
    )
    cand = df[mask].sort_values("Date", ascending=False)

    if cand.empty:
        # fallback → most recent home match of *home* vs anyone
        cand_home = (
            df[df["HomeTeam"].str.lower() == home.lower()]
            .sort_values("Date", ascending=False)
            .head(1)
        )
        cand_away = (
            df[df["AwayTeam"].str.lower() == away.lower()]
            .sort_values("Date", ascending=False)
            .head(1)
        )
        if cand_home.empty or cand_away.empty:
            raise ValueError("Cannot locate recent matches for supplied teams.")

        # merge the two rows by taking home-side engineered columns from
        # cand_home and away-side columns from cand_away
        home_cols = [c for c in df.columns if c.startswith("Home_")]
        away_cols = [c for c in df.columns if c.startswith("Away_")]
        merged = pd.DataFrame({})
        for col in home_cols:
            merged[col] = cand_home.iloc[0][col]
        for col in away_cols:
            merged[col] = cand_away.iloc[0][col]
        cand = merged

    row = cand.head(1).copy()

    # make sure columns set matches model's feature names exactly
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    needed = booster.feature_name()
    row = row[[c for c in row.columns if c in needed]]
    # re-order to model order
    row = row[needed]
    return row


def _predict(df_row: pd.DataFrame):
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    probs = booster.predict(df_row)[0]
    return {
        "HomeWin": float(probs[0]),
        "Draw": float(probs[1]),
        "AwayWin": float(probs[2]),
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # legacy JSON-file mode
        fixture_path = Path(sys.argv[1])
        if not fixture_path.exists():
            print(f"File not found: {fixture_path}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_json(fixture_path, orient="records")

        booster = lgb.Booster(model_file=str(MODEL_PATH))
        # align columns just in case
        df = df[booster.feature_name()]
        result = _predict(df)
    elif len(sys.argv) >= 3:
        # shorthand: team names supplied directly
        home_team = sys.argv[1]
        away_team = sys.argv[2]
        try:
            df_row = _prepare_features_from_h2h(home_team, away_team)
            result = _predict(df_row)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Usage:\n"
            "  python predict_match.py fixture.json\n"
            "  python predict_match.py \"Home Team\" \"Away Team\" [YYYY-MM-DD]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps(result, indent=2)) 