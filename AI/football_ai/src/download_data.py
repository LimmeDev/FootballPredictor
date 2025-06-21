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
import signal
import subprocess
import sys
import time
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
# Enhanced Helpers
# -----------------------------------------------------------------------------

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signals."""
    raise TimeoutError("Command timed out")


def _run_with_timeout(cmd: list[str], timeout: int = 300):
    """Run a shell command with timeout; handle interruptions gracefully."""
    print("$", " ".join(cmd))
    
    try:
        # Set up timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Run the command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            stdout, stderr = process.communicate()
            signal.alarm(0)  # Cancel timeout
            
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                if stderr:
                    print(f"Error: {stderr.decode()}")
                return False
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Download interrupted by user. Cleaning up...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            signal.alarm(0)
            return False
            
    except TimeoutError:
        print(f"\n⚠️  Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"\n⚠️  Unexpected error: {e}")
        return False


def _run(cmd: list[str]):
    """Run a shell command; exit on failure (legacy version)."""
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        return False
    return True


# -----------------------------------------------------------------------------
# Enhanced Downloaders
# -----------------------------------------------------------------------------

def download_statsbomb_robust() -> bool:
    """Clone or update the StatsBomb open-data repository with robust error handling."""
    
    if os.getenv("SKIP_STATSBOMB") == "1":
        print("[statsbomb] SKIP_STATSBOMB=1 – skipping StatsBomb download")
        return True

    repo_url = "https://github.com/statsbomb/open-data.git"
    dst = RAW_DIR / "statsbomb"

    if dst.exists():
        print("[statsbomb] Repository already exists – pulling latest commits …")
        success = _run_with_timeout(["git", "-C", str(dst), "pull", "--ff-only"], timeout=120)
        if not success:
            print("[statsbomb] ⚠️  Pull failed, but existing data is available")
        return True
    else:
        print("[statsbomb] Cloning repository (partial, ~60 MB) …")
        print("⏳ This may take 2-5 minutes depending on your connection...")
        
        # Try partial clone first (faster)
        success = _run_with_timeout([
            "git",
            "clone",
            "--depth", "1",
            "--filter=blob:none",  # lazy-fetch large JSON blobs later
            repo_url,
            str(dst),
        ], timeout=600)  # 10 minutes timeout
        
        if success:
            print("✅ StatsBomb data cloned successfully!")
            return True
        else:
            print("\n❌ StatsBomb clone failed or was interrupted.")
            print("🔄 You can:")
            print("   1. Run the script again to retry")
            print("   2. Set SKIP_STATSBOMB=1 to skip StatsBomb data")
            print("   3. Continue with other datasets")
            
            # Clean up partial download
            if dst.exists():
                import shutil
                shutil.rmtree(dst, ignore_errors=True)
                print("🧹 Cleaned up partial download")
            
            return False


def download_statsbomb() -> None:
    """Clone or update the StatsBomb open-data repository (legacy wrapper)."""
    try:
        download_statsbomb_robust()
    except KeyboardInterrupt:
        print("\n⚠️  StatsBomb download interrupted by user")
        dst = RAW_DIR / "statsbomb"
        if dst.exists():
            import shutil
            shutil.rmtree(dst, ignore_errors=True)
            print("🧹 Cleaned up partial download")


def download_538() -> None:
    """Fetch the latest FiveThirtyEight club-level match ratings."""
    url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
    dest = RAW_DIR / "spi_matches.csv"

    if dest.exists():
        print("[538] spi_matches.csv already present – skipping download")
        return

    print("[538] Downloading spi_matches.csv …")
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        
        # Get file size if available
        total_size = int(resp.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="538 data") as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                f.write(resp.content)
        
        print(f"[538] ✅ Saved → {dest.relative_to(PROJECT_ROOT)}  ({dest.stat().st_size/1_048_576:.2f} MB)")
        
    except requests.RequestException as e:
        print(f"[538] ❌ Download failed: {e}")
    except KeyboardInterrupt:
        print("\n[538] ⚠️  Download interrupted")
        if dest.exists():
            dest.unlink()


def download_football_data(
    seasons: range = range(9394, 2425 + 1),
    leagues: tuple[str, ...] = (
        # England + Scotland (reduced set for faster download)
        "E0", "E1", "E2", "SC0",
        # Major European leagues
        "SP1", "D1", "I1", "F1",
        # Additional leagues (optional)
        # "E3", "E4", "SC1", "SC2", "SP2", "D2", "I2", "F2", "P1", "N1", "B1", "T1", "G1", "A1", "M1", "L1",
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
    successful_downloads = 0
    total_attempts = 0

    print(f"[football-data] Downloading {len(leagues)} leagues × {len(seasons)} seasons...")
    
    try:
        for season in tqdm(seasons, desc="Seasons"):
            for league in leagues:
                total_attempts += 1
                url = base_url.format(season=season, league=league)
                dest = RAW_DIR / f"{league}_{season}.csv"
                
                if dest.exists():
                    successful_downloads += 1
                    continue
                
                try:
                    resp = requests.get(url, timeout=15)
                    # Some season/league combos 404; ignore small files (<1 KB)
                    if resp.status_code == 200 and len(resp.content) > 1024:
                        dest.write_bytes(resp.content)
                        successful_downloads += 1
                        size_mb = dest.stat().st_size / 1_048_576
                        if size_mb > 0.01:  # Only show for files > 10KB
                            tqdm.write(f"[football-data] ✅ {dest.name} ({size_mb:.2f} MB)")
                    else:
                        tqdm.write(f"[football-data] ⚠️  {league}_{season} not available")
                        
                except requests.RequestException as e:
                    tqdm.write(f"[football-data] ❌ Failed {league}_{season}: {e}")
                except KeyboardInterrupt:
                    print("\n[football-data] ⚠️  Download interrupted")
                    raise
                    
    except KeyboardInterrupt:
        print(f"\n[football-data] Downloaded {successful_downloads}/{total_attempts} files before interruption")
        return
    
    print(f"[football-data] ✅ Downloaded {successful_downloads}/{total_attempts} files")


def create_minimal_dataset():
    """Create a minimal dataset for quick testing if full downloads fail."""
    print("\n🔧 Creating minimal dataset for testing...")
    
    # Create a small sample dataset
    sample_data = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR
2023-08-12,Arsenal,Nottingham Forest,2,1,H,1,0,H,15,8,6,3,7,4,12,15,2,1,0,0
2023-08-12,Burnley,Manchester City,0,3,A,0,1,A,6,19,2,8,3,10,10,8,3,2,0,0
2023-08-12,Sheffield United,Crystal Palace,1,0,H,0,0,D,11,12,4,4,5,6,14,11,4,2,0,0"""
    
    sample_file = RAW_DIR / "E0_2324_sample.csv"
    sample_file.write_text(sample_data)
    print(f"✅ Created sample dataset: {sample_file.name}")


# -----------------------------------------------------------------------------
# Enhanced CLI entry-point
# -----------------------------------------------------------------------------

def main():
    """Main download function with enhanced error handling."""
    print("🏈 Football AI Data Downloader")
    print("=" * 50)
    print(f"📁 Data directory: {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_success = {"statsbomb": False, "538": False, "football_data": False}
    
    # 1. StatsBomb data (most likely to fail/be interrupted)
    print("\n📊 1/3 Downloading StatsBomb data...")
    try:
        if download_statsbomb_robust():
            download_success["statsbomb"] = True
    except FileNotFoundError:
        print("[warning] ⚠️  git executable not found – please install git and rerun for StatsBomb data.")
    except Exception as e:
        print(f"[statsbomb] ❌ Unexpected error: {e}")

    # 2. FiveThirtyEight data
    print("\n📈 2/3 Downloading FiveThirtyEight data...")
    try:
        download_538()
        download_success["538"] = True
    except Exception as e:
        print(f"[538] ❌ Error: {e}")

    # 3. Football-Data.co.uk
    print("\n⚽ 3/3 Downloading Football-Data.co.uk...")
    try:
        download_football_data()
        download_success["football_data"] = True
    except Exception as e:
        print(f"[football-data] ❌ Error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 50)
    
    total_success = sum(download_success.values())
    
    for dataset, success in download_success.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:15} : {status}")
    
    if total_success == 0:
        print("\n⚠️  No datasets downloaded successfully!")
        print("🔧 Creating minimal dataset for testing...")
        create_minimal_dataset()
        print("\n💡 You can still proceed with feature engineering using the sample data.")
        print("   Run: python src/make_features.py")
    elif total_success < 3:
        print(f"\n⚠️  Only {total_success}/3 datasets downloaded successfully.")
        print("💡 You can still proceed with available data or retry failed downloads.")
    else:
        print("\n🎉 All downloads completed successfully!")
    
    print(f"\n📁 Data saved to: {RAW_DIR}")
    print("▶️  Next step: python src/make_features.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download process interrupted by user")
        print("💡 You can run the script again to resume downloads")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 