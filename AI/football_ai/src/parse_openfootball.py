"""Parse OpenFootball JSON data into standardized features.

Generated file
--------------
`data/openfootball_features.parquet`

Columns (subset)
----------------
Date, HomeTeam, AwayTeam, League, Season,
Home_Goals, Away_Goals, Result,
Home_xG_Est, Away_xG_Est (estimated from match stats)
"""
from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OPENFOOTBALL_DIR = PROJECT_ROOT / "data" / "raw" / "openfootball"
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
OUTFILE = PROCESSED_DIR / "openfootball_features.parquet"

# League mappings for standardization
LEAGUE_MAPPINGS = {
    'en.1': {'name': 'Premier League', 'country': 'England'},
    'de.1': {'name': 'Bundesliga', 'country': 'Germany'},
    'it.1': {'name': 'Serie A', 'country': 'Italy'},
    'es.1': {'name': 'La Liga', 'country': 'Spain'},
    'fr.1': {'name': 'Ligue 1', 'country': 'France'},
    'pt.1': {'name': 'Primeira Liga', 'country': 'Portugal'},
    'nl.1': {'name': 'Eredivisie', 'country': 'Netherlands'},
}


def _find_league_files() -> List[Path]:
    """Find all league JSON files in the OpenFootball data."""
    league_files = []
    
    if not RAW_OPENFOOTBALL_DIR.exists():
        print(f"OpenFootball directory not found: {RAW_OPENFOOTBALL_DIR}")
        return league_files
    
    # Look for season directories (e.g., 2023-24, 2022-23)
    for season_dir in RAW_OPENFOOTBALL_DIR.iterdir():
        if season_dir.is_dir() and '-' in season_dir.name:
            # Look for league files (e.g., en.1.json, de.1.json)
            for league_file in season_dir.glob("*.json"):
                if any(league_code in league_file.name for league_code in LEAGUE_MAPPINGS.keys()):
                    league_files.append(league_file)
    
    return sorted(league_files)


def _parse_league_file(file_path: Path) -> List[Dict]:
    """Parse a single league JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'matches' not in data:
            print(f"No matches found in {file_path.name}")
            return []
        
        # Extract league info from filename and path
        season = file_path.parent.name  # e.g., "2023-24"
        league_code = file_path.stem    # e.g., "en.1"
        
        league_info = LEAGUE_MAPPINGS.get(league_code, {
            'name': league_code.upper(),
            'country': 'Unknown'
        })
        
        matches = []
        for match in data['matches']:
            try:
                match_data = {
                    'Date': pd.to_datetime(match.get('date')),
                    'HomeTeam': match.get('team1', ''),
                    'AwayTeam': match.get('team2', ''),
                    'League': league_info['name'],
                    'Country': league_info['country'],
                    'Season': season,
                    'Round': match.get('round', ''),
                }
                
                # Handle different score formats
                score = match.get('score', {})
                if isinstance(score, dict):
                    ft_score = score.get('ft', [None, None])
                    ht_score = score.get('ht', [None, None])
                    
                    if ft_score and len(ft_score) >= 2:
                        match_data['Home_Goals'] = ft_score[0]
                        match_data['Away_Goals'] = ft_score[1]
                        
                        # Determine result
                        if ft_score[0] > ft_score[1]:
                            match_data['Result'] = 'H'
                        elif ft_score[0] < ft_score[1]:
                            match_data['Result'] = 'A'
                        else:
                            match_data['Result'] = 'D'
                    
                    if ht_score and len(ht_score) >= 2:
                        match_data['Home_HT_Goals'] = ht_score[0]
                        match_data['Away_HT_Goals'] = ht_score[1]
                
                # Estimate xG based on goals (simple approximation)
                if match_data.get('Home_Goals') is not None:
                    match_data['Home_xG_Est'] = estimate_xg_from_goals(match_data['Home_Goals'])
                    match_data['Away_xG_Est'] = estimate_xg_from_goals(match_data['Away_Goals'])
                
                # Add additional metadata
                match_data['Source'] = 'OpenFootball'
                match_data['MatchID'] = f"{league_code}_{season}_{len(matches)}"
                
                matches.append(match_data)
                
            except Exception as e:
                print(f"Error parsing match in {file_path.name}: {e}")
                continue
        
        return matches
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def estimate_xg_from_goals(goals: int) -> float:
    """Simple xG estimation based on actual goals scored."""
    if goals is None:
        return 0.0
    
    # Basic estimation: assume some randomness around actual goals
    # In reality, xG would be calculated from shot data
    base_xg = goals * 0.85  # Slightly lower than actual goals
    
    # Add some variation based on typical xG patterns
    if goals == 0:
        return 0.3  # Even 0 goal games usually have some xG
    elif goals == 1:
        return 0.9 + (base_xg - 0.85)
    elif goals >= 4:
        return goals * 0.9  # High-scoring games tend to overperform xG
    else:
        return base_xg


def _standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize team names across different sources."""
    
    # Common team name variations
    name_mappings = {
        # Premier League
        'Manchester United': 'Man United',
        'Manchester City': 'Man City',
        'Newcastle United': 'Newcastle',
        'Tottenham Hotspur': 'Tottenham',
        'Brighton & Hove Albion': 'Brighton',
        'West Ham United': 'West Ham',
        
        # Bundesliga
        'Bayern Munich': 'Bayern München',
        'Borussia Dortmund': 'Dortmund',
        'Borussia Mönchengladbach': 'Gladbach',
        'Eintracht Frankfurt': 'Frankfurt',
        
        # Serie A
        'AC Milan': 'Milan',
        'Inter Milan': 'Inter',
        'AS Roma': 'Roma',
        'Juventus': 'Juventus',
        
        # La Liga  
        'Real Madrid': 'Real Madrid',
        'FC Barcelona': 'Barcelona',
        'Atlético Madrid': 'Atletico Madrid',
    }
    
    for col in ['HomeTeam', 'AwayTeam']:
        df[col] = df[col].replace(name_mappings)
    
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for machine learning."""
    
    # Goal difference
    df['Goal_Difference'] = df['Home_Goals'] - df['Away_Goals']
    
    # Total goals
    df['Total_Goals'] = df['Home_Goals'] + df['Away_Goals']
    
    # High/low scoring games
    df['High_Scoring'] = (df['Total_Goals'] >= 3).astype(int)
    df['Low_Scoring'] = (df['Total_Goals'] <= 1).astype(int)
    
    # Win margins
    df['Home_Win_Margin'] = df.apply(
        lambda x: x['Goal_Difference'] if x['Result'] == 'H' else 0, axis=1
    )
    df['Away_Win_Margin'] = df.apply(
        lambda x: abs(x['Goal_Difference']) if x['Result'] == 'A' else 0, axis=1
    )
    
    # Competitive match indicator
    df['Competitive'] = (abs(df['Goal_Difference']) <= 1).astype(int)
    
    # Season progress (approximate)
    df['Season_Progress'] = df.groupby(['League', 'Season']).cumcount() / df.groupby(['League', 'Season']).transform('size')
    
    return df


def main():
    """Parse OpenFootball data and create features."""
    
    print("🔍 Parsing OpenFootball JSON data...")
    
    league_files = _find_league_files()
    
    if not league_files:
        print("❌ No OpenFootball league files found!")
        print(f"Expected location: {RAW_OPENFOOTBALL_DIR}")
        print("Run 'python src/download_data.py' first to download data.")
        return False
    
    print(f"📁 Found {len(league_files)} league files")
    
    all_matches = []
    
    for file_path in tqdm(league_files, desc="Parsing league files"):
        matches = _parse_league_file(file_path)
        all_matches.extend(matches)
        print(f"✅ Parsed {len(matches)} matches from {file_path.name}")
    
    if not all_matches:
        print("❌ No matches were successfully parsed!")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(all_matches)
    print(f"📊 Total matches parsed: {len(df)}")
    
    # Data cleaning and standardization
    print("🧹 Cleaning and standardizing data...")
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    df = _standardize_team_names(df)
    df = _add_derived_features(df)
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam', 'League'])
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"🔄 Removed {initial_count - final_count} duplicate matches")
    
    # Summary statistics
    print("\n📈 Dataset Summary:")
    print(f"  • Total matches: {len(df):,}")
    print(f"  • Leagues: {df['League'].nunique()}")
    print(f"  • Seasons: {df['Season'].nunique()}")
    print(f"  • Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  • Countries: {', '.join(df['Country'].unique())}")
    
    # League breakdown
    print("\n🏆 Matches by League:")
    league_counts = df['League'].value_counts()
    for league, count in league_counts.items():
        print(f"  • {league}: {count:,}")
    
    # Save to parquet
    df.to_parquet(OUTFILE, index=False)
    print(f"\n✅ Saved features to: {OUTFILE.relative_to(PROJECT_ROOT)}")
    print(f"📊 Features shape: {df.shape}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 