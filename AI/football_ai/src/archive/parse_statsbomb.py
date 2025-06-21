"""Aggregate StatsBomb Open-Data event JSON files into per-match features.

Generated file
--------------
`data/statsbomb_features.parquet`

Columns (subset)
----------------
Date, HomeTeam, AwayTeam,
Home_SB_xG, Away_SB_xG,
Home_SB_Shots, Away_SB_Shots,
Home_SB_ShotOnTarget, Away_SB_ShotOnTarget,
Home_SB_PassCmpPct, Away_SB_PassCmpPct
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SB_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
OUTFILE = PROCESSED_DIR / "statsbomb_features.parquet"


def _iter_match_json_paths() -> list[tuple[Path, Path]]:
    """Yield (matches_path, events_path) pairs."""
    matches_root = RAW_SB_DIR / "matches"
    events_root = RAW_SB_DIR / "events"
    for comp_dir in matches_root.iterdir():
        if not comp_dir.is_dir():
            continue
        for season_file in comp_dir.glob("*.json"):
            # season_file relative parts: matches/<comp>/<season>.json
            with season_file.open("r", encoding="utf-8") as fh:
                season_matches = json.load(fh)
            for m in season_matches:
                match_id = m["match_id"]
                event_path = events_root / f"{match_id}.json"
                if event_path.exists():
                    yield m, event_path


def _aggregate_single(match_meta: dict, events_path: Path) -> dict:
    home_name = match_meta["home_team"]["home_team_name"]
    away_name = match_meta["away_team"]["away_team_name"]
    date = pd.to_datetime(match_meta["match_date"])

    stats = {
        "HomeTeam": home_name,
        "AwayTeam": away_name,
        "Date": date,
    }

    per_team = defaultdict(lambda: defaultdict(float))

    with events_path.open("r", encoding="utf-8") as fh:
        events = json.load(fh)

    for ev in events:
        team = ev["team"]["name"]
        typ = ev["type"]["name"]
        if typ == "Shot":
            per_team[team]["shots"] += 1
            xg = ev["shot"].get("statsbomb_xg", 0.0) or 0.0
            per_team[team]["xg"] += xg
            outcome = ev["shot"]["outcome"]["name"]
            if outcome in ("Goal", "Saved"):
                per_team[team]["shots_on_target"] += 1
        elif typ == "Pass":
            per_team[team]["passes_total"] += 1
            completed = ev["pass"].get("outcome") is None
            if completed:
                per_team[team]["passes_completed"] += 1

    # helper to safely get
    def g(t: str, k: str) -> float:
        return per_team[t].get(k, 0.0)

    for prefix, team in (("Home", home_name), ("Away", away_name)):
        shots = g(team, "shots")
        sot = g(team, "shots_on_target")
        xg = g(team, "xg")
        passes_total = g(team, "passes_total")
        passes_completed = g(team, "passes_completed")
        pass_cmp_pct = (
            passes_completed / passes_total * 100 if passes_total > 0 else None
        )
        stats[f"{prefix}_SB_xG"] = xg
        stats[f"{prefix}_SB_Shots"] = shots
        stats[f"{prefix}_SB_ShotOnTarget"] = sot
        stats[f"{prefix}_SB_PassCmpPct"] = pass_cmp_pct

    return stats


if __name__ == "__main__":
    rows: list[dict] = []
    for match_meta, events_path in tqdm(_iter_match_json_paths(), desc="StatsBomb matches"):
        try:
            rows.append(_aggregate_single(match_meta, events_path))
        except Exception as exc:
            print(f"[parse_statsbomb] Failed for {events_path.name}: {exc}")
    if not rows:
        print("[parse_statsbomb] No matches processed.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    # remove duplicates just in case
    df = df.drop_duplicates(subset=["HomeTeam", "AwayTeam", "Date"])
    df.to_parquet(OUTFILE, index=False)
    print(f"[parse_statsbomb] Wrote {len(df)} rows to {OUTFILE.relative_to(PROJECT_ROOT)}") 