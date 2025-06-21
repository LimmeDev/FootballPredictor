"""Monte-Carlo tournament simulator (group + knockout).

This is a minimal, **extensible** scaffold.  Edit `FIXTURE_JSON` to point to a
file that describes your tournament schedule and group/knockout bracket.

Example fixture schema (list of dicts):
[
  {"stage": "Group A", "home": "Arsenal",  "away": "Bayern",   "date": "2026-06-12"},
  {"stage": "Group A", "home": "Inter",    "away": "Ajax",      "date": "2026-06-12"},
  ...
]
Each record must contain whatever columns `predict_match.py` needs to compute
features.

Run
----
$ python simulate_tournament.py --n 10000 --fixture path/to/worldcup.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import choices

import numpy as np
import pandas as pd

from predict_match import MODEL_PATH, Path as _Path
from predict_match import lgb, pd as _pd  # reuse imports
from predict_match import PROJECT_ROOT  # noqa: E402
from predict_match import lgb


def load_model():
    return lgb.Booster(model_file=str(MODEL_PATH))


def predict_probs(model, features: pd.DataFrame):
    return model.predict(features)[0]


def simulate_fixture(df_fixture: pd.DataFrame, model) -> dict[str, int]:
    """Return champion tally dict after one Monte-Carlo run.

    For league-format tournaments, champion is team with most points.
    Knock-out logic is left as an exercise (see README next-steps).
    """
    points: dict[str, int] = {t: 0 for t in pd.unique(df_fixture[["home", "away"]].values.ravel())}

    for _, match in df_fixture.iterrows():
        p = predict_probs(model, match.to_frame().T)
        outcome = choices([0, 1, 2], weights=p, k=1)[0]
        if outcome == 0:
            points[match.home] += 3
        elif outcome == 1:
            points[match.home] += 1
            points[match.away] += 1
        else:
            points[match.away] += 3
    champion = max(points, key=points.get)
    return champion


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=Path, help="Path to fixture JSON list")
    ap.add_argument("--n", type=int, default=10000, help="Number of Monte-Carlo iterations")
    args = ap.parse_args(argv)

    df_fixture = pd.read_json(args.fixture)
    model = load_model()

    tally: dict[str, int] = {}
    for _ in range(args.n):
        champ = simulate_fixture(df_fixture, model)
        tally[champ] = tally.get(champ, 0) + 1

    probs = {k: v / args.n for k, v in sorted(tally.items(), key=lambda x: -x[1])}
    print(json.dumps(probs, indent=2))


if __name__ == "__main__":
    main() 