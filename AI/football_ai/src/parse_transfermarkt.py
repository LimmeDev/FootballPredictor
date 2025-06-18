"""Download and aggregate Transfermarkt squad data → season-level features.

Data source (public)
--------------------
The open dataset maintained at:
  https://raw.githubusercontent.com/ishandutta2007/Transfermarkt-datasets/master/output/teams.csv

That CSV contains (club_code, season, squad_size, age, total_value) among others.
It is small (~3 MB) and needs no credentials.

This script downloads the CSV (if missing), computes per-team-season
  • AvgAge          (years)
  • SquadValueEUR   (million €)
Then saves `data/transfermarkt_features.parquet` for merging into the main
feature table.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
OUTFILE = PROCESSED_DIR / "transfermarkt_features.parquet"
TMP_CSV = PROCESSED_DIR / "transfermarkt_teams.csv"

CSV_URL = (
    "https://raw.githubusercontent.com/ishandutta2007/Transfermarkt-datasets/"
    "master/output/teams.csv"
)


def _download_csv() -> Path:
    if TMP_CSV.exists():
        return TMP_CSV
    print("[parse_transfermarkt] downloading squad CSV …")
    resp = requests.get(CSV_URL, timeout=30)
    if resp.status_code != 200:
        print(f"[parse_transfermarkt] download failed (status={resp.status_code}); skipping Transfermarkt features.")
        return None
    TMP_CSV.write_bytes(resp.content)
    return TMP_CSV


def _aggregate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expected columns: club_name, season, avg_age, squad_size, total_value
    required = {"club_name", "season", "avg_age", "total_value"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")

    # Convert value like "€477.60m" → 477.6
    def _parse_value(val: str) -> float | None:
        if isinstance(val, (int, float)):
            return float(val) / 1e6
        if not isinstance(val, str) or not val.startswith("€"):
            return None
        val = val[1:]
        mult = 1.0
        if val.endswith("k"):
            mult = 1e-3
            val = val[:-1]
        elif val.endswith("m"):
            mult = 1.0
            val = val[:-1]
        try:
            return float(val.replace(",", "")) * mult
        except ValueError:
            return None

    df["SquadValue_mEUR"] = df["total_value"].map(_parse_value)
    df = df.rename(columns={"avg_age": "SquadAvgAge", "club_name": "Team"})

    # Keep only needed columns
    df = df[["Team", "season", "SquadAvgAge", "SquadValue_mEUR"]]
    df = df.dropna(subset=["SquadAvgAge", "SquadValue_mEUR"])
    # Some seasons use e.g. "2009-10" – take the ending year as numeric
    df["SeasonEnd"] = df["season"].str.split("-").str[-1].astype(int)
    df = df.drop(columns=["season"])
    return df


if __name__ == "__main__":
    csv_file = _download_csv()
    if csv_file is None:
        sys.exit(0)

    df = _aggregate(csv_file)
    df.to_parquet(OUTFILE, index=False)
    print(
        f"[parse_transfermarkt] wrote {len(df):,} team-season rows → {OUTFILE.relative_to(PROJECT_ROOT)}"
    ) 