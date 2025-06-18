"""Download open-access football datasets used by the project.

Datasets fetched
----------------
1. StatsBomb open-data (event-level JSON)  – via shallow `git clone` (~200 MB).
2. FiveThirtyEight SPI match ratings       – single CSV (<5 MB).
3. Football-Data.co.uk league CSVs         – ~1 MB per season.

All files are stored under `data/raw/` relative to the project root.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _run(cmd: list[str]):
    """Run a shell command; exit on failure."""
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(res.returncode)


# -----------------------------------------------------------------------------
# Downloaders
# -----------------------------------------------------------------------------

def download_statsbomb() -> None:
    """Clone or update the StatsBomb open-data repository."""
    repo_url = "https://github.com/statsbomb/open-data.git"
    dst = RAW_DIR / "statsbomb"

    if dst.exists():
        print("[statsbomb] Repository already exists – pulling latest commits …")
        _run(["git", "-C", str(dst), "pull", "--ff-only"])
    else:
        print("[statsbomb] Cloning repository … (~200 MB on first run)")
        _run(["git", "clone", "--depth", "1", repo_url, str(dst)])


def download_538() -> None:
    """Fetch the latest FiveThirtyEight club-level match ratings."""
    url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
    dest = RAW_DIR / "spi_matches.csv"

    if dest.exists():
        print("[538] spi_matches.csv already present – skipping download")
        return

    print("[538] Downloading spi_matches.csv …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[538] Saved → {dest.relative_to(PROJECT_ROOT)}  ({dest.stat().st_size/1_048_576:.2f} MB)")


def download_football_data(
    seasons: range = range(9394, 2425 + 1),
    leagues: tuple[str, ...] = (
        # England + Scotland
        "E0", "E1", "E2", "E3", "E4", "SC0", "SC1", "SC2",
        # Spain, Germany, Italy, France, Portugal, Netherlands
        "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2", "P1", "N1",
        # Others
        "B1", "T1", "G1", "A1", "M1", "L1",
    ),
) -> None:
    """Download match-stat CSVs from Football-Data.co.uk.

    Parameters
    ----------
    seasons : range
        Football-Data encodes seasons as YYZZ → 1415 == 2014-2015.
    leagues : tuple[str]
        League code(s); ``E0`` is English Premier League.
    """
    base_url = "https://www.football-data.co.uk/mmz4281/{season:04d}/{league}.csv"

    for season in seasons:
        for league in leagues:
            url = base_url.format(season=season, league=league)
            dest = RAW_DIR / f"{league}_{season}.csv"
            if dest.exists():
                continue
            print(f"[football-data] GET {url}")
            resp = requests.get(url, timeout=30)
            # Some season/league combos 404; ignore small files (<1 KB)
            if resp.status_code == 200 and len(resp.content) > 1024:
                dest.write_bytes(resp.content)
                size_mb = dest.stat().st_size / 1_048_576
                print(f"[football-data] Saved → {dest.name} ({size_mb:.2f} MB)")
            else:
                print("[football-data] Not available, skipping …")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("==> Data directory:", RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        download_statsbomb()
    except FileNotFoundError:
        print("[warning] git executable not found – please install git and rerun for StatsBomb data.")

    download_538()
    download_football_data()

    print("✓ All downloads complete") 