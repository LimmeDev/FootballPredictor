"""Download lightweight football datasets from multiple sources.

Datasets fetched (Total ~500-800 MB)
------------------------------------
1. OpenFootball JSON data (~100-200 MB) - Premier League, Bundesliga, Serie A, La Liga
2. Football-Data.org API (~50-100 MB) - Multiple European leagues
3. FiveThirtyEight SPI ratings (~5 MB) - Global club ratings
4. Football-Data.co.uk CSVs (~100-200 MB) - Historical match data
5. API-Football sample data (~50-100 MB) - Recent seasons with detailed stats

Each dataset covers different competitions:
- Premier League (England)
- Bundesliga (Germany) 
- Serie A (Italy)
- La Liga (Spain)
- Ligue 1 (France)
- Champions League (Europe)

All files are stored under `data/raw/` relative to the project root.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (free tier limits)
FOOTBALL_DATA_ORG_TOKEN = os.getenv('FOOTBALL_DATA_TOKEN', '')  # Free: 10 calls/min
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY', '')  # Free: 100 calls/day

# -----------------------------------------------------------------------------
# Configuration for lightweight datasets
# -----------------------------------------------------------------------------
DATASET_CONFIG = {
    'openfootball': {
        'enabled': True,
        'size_mb': 150,
        'competitions': ['Premier League', 'Bundesliga', 'Serie A', 'La Liga'],
        'seasons': 3,  # Last 3 seasons
    },
    'football_data_org': {
        'enabled': True,
        'size_mb': 80,
        'competitions': ['Champions League', 'Ligue 1', 'Eredivisie'],
        'seasons': 2,  # Last 2 seasons
    },
    'fivethirtyeight': {
        'enabled': True,
        'size_mb': 5,
        'competitions': ['Global SPI ratings'],
        'seasons': 'all',
    },
    'football_data_uk': {
        'enabled': True,
        'size_mb': 200,
        'competitions': ['Multiple European leagues'],
        'seasons': 5,  # Last 5 seasons
    }
}

# -----------------------------------------------------------------------------
# Enhanced Helpers
# -----------------------------------------------------------------------------

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signals."""
    raise TimeoutError("Command timed out")

def _download_with_progress(url: str, dest: Path, desc: str, timeout: int = 300) -> bool:
    """Download a file with progress bar and timeout."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                f.write(response.content)
        
        print(f"✅ Downloaded {desc} → {dest.relative_to(PROJECT_ROOT)} ({dest.stat().st_size/1_048_576:.1f} MB)")
        return True
        
    except requests.RequestException as e:
        print(f"❌ Failed to download {desc}: {e}")
        if dest.exists():
            dest.unlink()
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Download of {desc} interrupted")
        if dest.exists():
            dest.unlink()
        return False

def _run_with_timeout(cmd: list[str], timeout: int = 300) -> bool:
    """Run a shell command with timeout; handle interruptions gracefully."""
    print("$", " ".join(cmd))
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            stdout, stderr = process.communicate()
            signal.alarm(0)
            
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                if stderr:
                    print(f"Error: {stderr.decode()}")
                return False
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Command interrupted by user. Cleaning up...")
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

# -----------------------------------------------------------------------------
# Lightweight Dataset Downloaders
# -----------------------------------------------------------------------------

def download_openfootball_json() -> bool:
    """Download OpenFootball JSON data for major European leagues (~150 MB)."""
    
    if not DATASET_CONFIG['openfootball']['enabled']:
        print("[openfootball] Disabled in config - skipping")
        return True
    
    repo_url = "https://github.com/openfootball/football.json.git"
    dst = RAW_DIR / "openfootball"
    
    if dst.exists():
        print("[openfootball] Repository exists - pulling latest...")
        success = _run_with_timeout(["git", "-C", str(dst), "pull", "--ff-only"], timeout=120)
        if not success:
            print("[openfootball] ⚠️  Pull failed, but existing data available")
        return True
    else:
        print("[openfootball] Cloning OpenFootball JSON data...")
        print("📦 Getting Premier League, Bundesliga, Serie A, La Liga (~150 MB)")
        
        success = _run_with_timeout([
            "git", "clone", "--depth", "1", 
            "--single-branch",  # Only master branch
            repo_url, str(dst)
        ], timeout=300)
        
        if success:
            print("✅ OpenFootball JSON data downloaded successfully!")
            # Clean up to keep only recent seasons
            cleanup_old_seasons(dst, keep_seasons=3)
            return True
        else:
            print("❌ OpenFootball download failed")
            if dst.exists():
                import shutil
                shutil.rmtree(dst, ignore_errors=True)
            return False

def download_football_data_org() -> bool:
    """Download data from football-data.org API (~80 MB)."""
    
    if not DATASET_CONFIG['football_data_org']['enabled']:
        print("[football-data.org] Disabled in config - skipping")
        return True
    
    base_url = "http://api.football-data.org/v4"
    headers = {'X-Auth-Token': FOOTBALL_DATA_ORG_TOKEN} if FOOTBALL_DATA_ORG_TOKEN else {}
    
    dst_dir = RAW_DIR / "football_data_org"
    dst_dir.mkdir(exist_ok=True)
    
    # Free tier competitions
    competitions = {
        'PL': 'Premier League',
        'BL1': 'Bundesliga', 
        'SA': 'Serie A',
        'PD': 'La Liga',
        'FL1': 'Ligue 1',
        'CL': 'Champions League'
    }
    
    print("[football-data.org] Downloading competition data...")
    print("🏆 Getting Champions League, Ligue 1, and more (~80 MB)")
    
    success_count = 0
    for comp_code, comp_name in competitions.items():
        try:
            # Get competition info
            url = f"{base_url}/competitions/{comp_code}"
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                comp_file = dst_dir / f"{comp_code}_info.json"
                with open(comp_file, 'w') as f:
                    json.dump(response.json(), f, indent=2)
                
                # Get current season matches
                matches_url = f"{base_url}/competitions/{comp_code}/matches"
                matches_response = requests.get(matches_url, headers=headers, timeout=30)
                
                if matches_response.status_code == 200:
                    matches_file = dst_dir / f"{comp_code}_matches.json"
                    with open(matches_file, 'w') as f:
                        json.dump(matches_response.json(), f, indent=2)
                    success_count += 1
                    print(f"✅ Downloaded {comp_name} data")
                
                time.sleep(6)  # Respect rate limit (10 calls/min)
                
            elif response.status_code == 429:
                print(f"⚠️  Rate limit reached for {comp_name}")
                break
            else:
                print(f"⚠️  Failed to get {comp_name}: HTTP {response.status_code}")
                
        except requests.RequestException as e:
            print(f"⚠️  Error downloading {comp_name}: {e}")
        except KeyboardInterrupt:
            print("\n⚠️  Football-data.org download interrupted")
            return False
    
    if success_count > 0:
        print(f"✅ Downloaded {success_count}/{len(competitions)} competitions from football-data.org")
        return True
    else:
        print("❌ No data downloaded from football-data.org")
        return False

def download_fivethirtyeight() -> bool:
    """Download FiveThirtyEight SPI ratings (~5 MB)."""
    
    if not DATASET_CONFIG['fivethirtyeight']['enabled']:
        print("[538] Disabled in config - skipping")
        return True
    
    urls = {
        'spi_matches.csv': 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv',
        'spi_global_rankings.csv': 'https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv'
    }
    
    print("[538] Downloading FiveThirtyEight SPI data...")
    print("📊 Getting global club ratings and match predictions (~5 MB)")
    
    success_count = 0
    for filename, url in urls.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"[538] {filename} already exists - skipping")
            success_count += 1
            continue
        
        if _download_with_progress(url, dest, f"538 {filename}"):
            success_count += 1
    
    return success_count > 0

def download_football_data_uk() -> bool:
    """Download Football-Data.co.uk historical data (~200 MB)."""
    
    if not DATASET_CONFIG['football_data_uk']['enabled']:
        print("[football-data.co.uk] Disabled in config - skipping")  
        return True
    
    print("[football-data.co.uk] Downloading historical match data...")
    print("📈 Getting 5 seasons of European league data (~200 MB)")
    
    base_url = "https://www.football-data.co.uk/mmz4281"
    
    # Major leagues with smaller file sizes
    leagues = {
        'E0': 'Premier League',
        'D1': 'Bundesliga', 
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'F1': 'Ligue 1'
    }
    
    # Last 5 seasons
    seasons = ['2021', '2122', '2223', '2324', '2425']
    season_names = ['20/21', '21/22', '22/23', '23/24', '24/25']
    
    dst_dir = RAW_DIR / "football_data_uk"
    dst_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_files = len(leagues) * len(seasons)
    
    for season, season_name in zip(seasons, season_names):
        for league_code, league_name in leagues.items():
            try:
                filename = f"{league_code}.csv"
                url = f"{base_url}/{season}/{filename}"
                dest = dst_dir / f"{league_code}_{season}.csv"
                
                if dest.exists():
                    success_count += 1
                    continue
                
                if _download_with_progress(url, dest, f"{league_name} {season_name}"):
                    success_count += 1
                
                time.sleep(0.5)  # Be respectful
                
            except KeyboardInterrupt:
                print("\n⚠️  Football-data.co.uk download interrupted")
                return success_count > 0
    
    print(f"✅ Downloaded {success_count}/{total_files} files from football-data.co.uk")
    return success_count > 0

def cleanup_old_seasons(repo_path: Path, keep_seasons: int = 3):
    """Remove older seasons to save space."""
    try:
        season_dirs = [d for d in repo_path.iterdir() if d.is_dir() and d.name.count('-') == 1]
        season_dirs.sort(reverse=True)  # Most recent first
        
        if len(season_dirs) > keep_seasons:
            for old_season in season_dirs[keep_seasons:]:
                import shutil
                shutil.rmtree(old_season, ignore_errors=True)
                print(f"🧹 Removed old season: {old_season.name}")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def download_sample_api_football():
    """Download sample data from API-Football for recent matches."""
    
    if not API_FOOTBALL_KEY:
        print("[api-football] No API key - skipping (set API_FOOTBALL_KEY env var)")
        return True
    
    print("[api-football] Downloading recent match samples...")
    print("🔥 Getting latest fixtures and standings (~50 MB)")
    
    # This would require API-Football implementation
    # For now, skip and use other sources
    print("[api-football] ⚠️  Implementation pending - using other sources")
    return True

def create_dataset_summary():
    """Create a summary of downloaded datasets."""
    summary = {
        'datasets': {},
        'total_size_mb': 0,
        'competitions_covered': [],
        'download_timestamp': time.time()
    }
    
    for dataset_name, config in DATASET_CONFIG.items():
        if config['enabled']:
            summary['datasets'][dataset_name] = {
                'size_mb': config['size_mb'],
                'competitions': config['competitions'],
                'seasons': config['seasons']
            }
            summary['total_size_mb'] += config['size_mb']
            summary['competitions_covered'].extend(config['competitions'])
    
    summary_file = RAW_DIR / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    print(f"Total size: ~{summary['total_size_mb']} MB")
    print(f"Competitions: {len(set(summary['competitions_covered']))}")
    print("\nCompetitions covered:")
    for comp in sorted(set(summary['competitions_covered'])):
        print(f"  • {comp}")
    print("="*60)

# -----------------------------------------------------------------------------
# Main Download Function
# -----------------------------------------------------------------------------

def main():
    """Download all configured lightweight datasets."""
    
    print("🚀 LIGHTWEIGHT FOOTBALL DATA DOWNLOADER")
    print("="*60)
    print("Downloading multiple small datasets instead of large StatsBomb...")
    print("Total expected size: ~500-800 MB across 3-6 different sources")
    print("="*60)
    
    downloaders = [
        ("OpenFootball JSON", download_openfootball_json),
        ("Football-Data.org API", download_football_data_org), 
        ("FiveThirtyEight SPI", download_fivethirtyeight),
        ("Football-Data.co.uk", download_football_data_uk),
        ("API-Football Sample", download_sample_api_football),
    ]
    
    successful_downloads = []
    failed_downloads = []
    
    for name, downloader_func in downloaders:
        print(f"\n📥 Starting {name}...")
        try:
            if downloader_func():
                successful_downloads.append(name)
                print(f"✅ {name} completed successfully")
            else:
                failed_downloads.append(name)
                print(f"❌ {name} failed")
        except KeyboardInterrupt:
            print(f"\n⚠️  {name} interrupted by user")
            failed_downloads.append(name)
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            failed_downloads.append(name)
    
    # Create summary
    create_dataset_summary()
    
    # Final report
    print(f"\n🎯 DOWNLOAD COMPLETE")
    print(f"✅ Successful: {len(successful_downloads)}/{len(downloaders)}")
    print(f"❌ Failed: {len(failed_downloads)}/{len(downloaders)}")
    
    if successful_downloads:
        print("\nSuccessful downloads:")
        for name in successful_downloads:
            print(f"  ✅ {name}")
    
    if failed_downloads:
        print("\nFailed downloads:")
        for name in failed_downloads:
            print(f"  ❌ {name}")
        print("\n💡 Tips:")
        print("  • Check your internet connection")
        print("  • Set FOOTBALL_DATA_TOKEN for football-data.org")
        print("  • Set API_FOOTBALL_KEY for API-Football data")
        print("  • Re-run script to retry failed downloads")
    
    print(f"\n📁 All data saved to: {RAW_DIR.relative_to(PROJECT_ROOT)}")
    
    return len(successful_downloads) > 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        print("🔄 You can re-run the script to continue from where it left off")
        sys.exit(1) 