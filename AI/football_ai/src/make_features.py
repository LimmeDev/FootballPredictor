"""Convert raw match CSVs into a single feature table for model training.

This first-pass feature set is intentionally simple so newcomers can follow the
code.  Accuracy will improve as you add richer features (Elo ratings, rolling
xG, betting odds, etc.).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

OUTFILE = PROCESSED_DIR / "training_features.parquet"
SB_PARQUET = PROCESSED_DIR / "statsbomb_features.parquet"
TM_PARQUET = PROCESSED_DIR / "transfermarkt_features.parquet"


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _load_football_data() -> pd.DataFrame:
    """Load all Football-Data league CSVs found in *data/raw/*.csv*"""
    files = glob.glob(str(RAW_DIR / "*.csv"))
    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as err:
            print("[warn] Couldn't read", fp, ":", err)
            continue
        if "FTR" not in df.columns:
            # not a match results file
            continue
        df["SourceFile"] = os.path.basename(fp)
        frames.append(df)
    if not frames:
        raise RuntimeError("No Football-Data CSVs found – did you run download_data.py?")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Convert Date column (varies in format across seasons)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _encode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"H": 0, "D": 1, "A": 2}
    df["Result"] = df["FTR"].map(mapping)
    return df


def _add_rolling_means(df: pd.DataFrame, numeric_cols: list[str], window: int = 5) -> pd.DataFrame:
    """Attach rolling-mean statistics for the last *window* matches per team."""
    df = df.copy()
    for col in numeric_cols:
        # Home perspective
        df[f"Home_{col}_roll{window}"] = (
            df.groupby("HomeTeam")[col]
            .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
        )
        # Away perspective
        df[f"Away_{col}_roll{window}"] = (
            df.groupby("AwayTeam")[col]
            .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
        )
    return df


def build_feature_table() -> pd.DataFrame:
    raw = _load_football_data()
    raw = _basic_clean(raw)
    raw = _encode_outcome(raw)

    # ------------------------------------------------------------------
    # 1. Basic numeric + rolling features (5 + 10 match windows)
    # ------------------------------------------------------------------
    numeric_cols = [
        "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    ]
    df = _add_rolling_means(raw, numeric_cols, window=5)
    df = _add_rolling_means(df, numeric_cols, window=10)
    df = _add_rolling_means(df, numeric_cols, window=15)

    # ------------------------------------------------------------------
    # 2. Bookmaker odds → implied probabilities
    # ------------------------------------------------------------------
    odds_cols = [c for c in df.columns if c.endswith("H") or c.endswith("D") or c.endswith("A")]
    # keep only numeric odds columns (e.g. B365H, BW A etc.)
    odds_cols = [c for c in odds_cols if df[c].dtype != "O"]
    for col in odds_cols:
        prob_col = col + "_impP"
        df[prob_col] = 1 / df[col]

    # Example: create bookmaker average implied probs across providers
    prov_prefixes = set(c[:-1] for c in odds_cols if c[-1] in ("H", "D", "A"))
    for pref in prov_prefixes:
        cols = [f"{pref}{s}" for s in ("H", "D", "A") if f"{pref}{s}_impP" in df]
        if len(cols) == 3:
            df[f"{pref}_overround"] = df[[c + "_impP" for c in cols]].sum(axis=1)
            for c in cols:
                df[c + "_normP"] = df[c + "_impP"] / df[f"{pref}_overround"]

    # ------------------------------------------------------------------
    # 3. Simple Elo rating
    # ------------------------------------------------------------------
    def compute_elo(df_: pd.DataFrame, k: float = 40.0, start: float = 1500.0):
        elo_home, elo_away = [], []
        ratings: dict[str, float] = {}
        for h, a, r in zip(df_["HomeTeam"], df_["AwayTeam"], df_["Result"]):
            rh = ratings.get(h, start)
            ra = ratings.get(a, start)
            elo_home.append(rh)
            elo_away.append(ra)

            # expected probs using logistic curve
            exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
            # outcome: 1=home win, 0.5=draw, 0=away win
            score_h = 1 if r == 0 else 0.5 if r == 1 else 0
            score_a = 1 - score_h

            ratings[h] = rh + k * (score_h - exp_h)
            ratings[a] = ra + k * (score_a - (1 - exp_h))

        df_["Home_Elo"] = elo_home
        df_["Away_Elo"] = elo_away
        df_["Elo_Diff"] = df_["Home_Elo"] - df_["Away_Elo"]
        return df_

    df = df.sort_values("Date").reset_index(drop=True)
    df = compute_elo(df)

    # ------------------------------------------------------------------
    # 4. Final feature selection
    # ------------------------------------------------------------------
    feature_cols = []
    # rolling numeric features
    feature_cols += [c for c in df.columns if c.startswith("Home_") or c.startswith("Away_")]
    # odds implied probabilities
    feature_cols += [c for c in df.columns if c.endswith("_normP")]

    # Rest days helper inside main function
    def add_rest_days(df_: pd.DataFrame) -> pd.DataFrame:
        for side, team_col in (("Home", "HomeTeam"), ("Away", "AwayTeam")):
            prev_date = df_.groupby(team_col)["Date"].shift()
            df_[f"{side}_RestDays"] = (
                (df_["Date"] - prev_date).dt.days.clip(lower=0).fillna(7)
            )
        return df_

    df = add_rest_days(df)

    # elo features
    feature_cols += ["Home_Elo", "Away_Elo", "Elo_Diff"]
    # rest days
    feature_cols += ["Home_RestDays", "Away_RestDays"]

    feature_cols = list(dict.fromkeys(feature_cols))

    # ------------------------------------------------------------------
    #  StatsBomb enriched features (xG, pass %, etc.)
    # ------------------------------------------------------------------
    if SB_PARQUET.exists():
        sb = pd.read_parquet(SB_PARQUET)
        sb["Date"] = pd.to_datetime(sb["Date"])
        df = df.merge(
            sb,
            on=["HomeTeam", "AwayTeam", "Date"],
            how="left",
            suffixes=("", ""),
        )
        # Add any new SB prefixed columns to feature list automatically
        for col in sb.columns:
            if col.startswith("Home_SB_") or col.startswith("Away_SB_"):
                if col not in feature_cols:
                    feature_cols.append(col)

    features = df[feature_cols + ["Result", "Date"]]

    # ------------------------------------------------------------------
    #  Transfermarkt squad features (AvgAge, SquadValue)
    # ------------------------------------------------------------------
    if TM_PARQUET.exists():
        tm = pd.read_parquet(TM_PARQUET)
        # derive season end year for matches (assuming July–June season)
        df_season_end = features["Date"].dt.year.where(features["Date"].dt.month >= 7, features["Date"].dt.year - 1) + 1
        features["SeasonEnd"] = df_season_end

        for side, team_col in (("Home", "HomeTeam"), ("Away", "AwayTeam")):
            features = features.merge(
                tm.rename(columns={"Team": team_col}),
                on=[team_col, "SeasonEnd"],
                how="left",
                suffixes=("", f"_{side}"),
            )
            # rename merged columns to side-specific
            for col in ("SquadAvgAge", "SquadValue_mEUR"):
                features.rename(columns={col: f"{side}_{col}"}, inplace=True)

        # Derived diffs
        features["Age_Diff"] = features["Home_SquadAvgAge"] - features["Away_SquadAvgAge"]
        features["Value_Diff_mEUR"] = features["Home_SquadValue_mEUR"] - features["Away_SquadValue_mEUR"]

        # update feature_cols
        feature_cols += [
            "Home_SquadAvgAge",
            "Away_SquadAvgAge",
            "Home_SquadValue_mEUR",
            "Away_SquadValue_mEUR",
            "Age_Diff",
            "Value_Diff_mEUR",
        ]

        # cleanup
        features = features[feature_cols + ["Result", "Date"]]

    # remove helper column if exists
    if "SeasonEnd" in features.columns:
        features = features.drop(columns=["SeasonEnd"])

    # ---------------------------------------------------------------
    #  Persist team identifiers so downstream scripts (predict_match)
    #  can reconstruct fixtures using only team names, without
    #  re-engineering features.
    # ---------------------------------------------------------------
    keep_cols = feature_cols + ["Result", "Date", "HomeTeam", "AwayTeam"]
    features = df[keep_cols]

    return features


if __name__ == "__main__":
    print("[make_features] Building feature table …")
    feats = build_feature_table()
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUTFILE, index=False)
    print(f"[make_features] Saved → {OUTFILE.relative_to(PROJECT_ROOT)}  (rows={len(feats):,})") 