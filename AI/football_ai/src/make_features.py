"""Combine multiple football datasets into ML-ready features.

This script processes data from multiple lightweight sources:
1. OpenFootball JSON data (Premier League, Bundesliga, Serie A, La Liga, etc.)
2. Football-Data.org API data (additional competitions)
3. FiveThirtyEight SPI ratings (global club ratings)
4. Football-Data.co.uk historical data (detailed match statistics)

Generated files
---------------
`data/combined_features.parquet` - Final ML-ready dataset
`data/feature_summary.json` - Dataset statistics and metadata
"""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import parsers
from parse_openfootball import main as parse_openfootball

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Output files
COMBINED_FEATURES_FILE = PROCESSED_DIR / "combined_features.parquet"
FEATURE_SUMMARY_FILE = PROCESSED_DIR / "feature_summary.json"

def load_openfootball_data() -> Optional[pd.DataFrame]:
    """Load parsed OpenFootball data."""
    openfootball_file = PROCESSED_DIR / "openfootball_features.parquet"
    
    if not openfootball_file.exists():
        print("📦 OpenFootball data not found, parsing...")
        if parse_openfootball():
            return pd.read_parquet(openfootball_file)
        else:
            print("❌ Failed to parse OpenFootball data")
            return None
    else:
        print("✅ Loading existing OpenFootball data...")
        return pd.read_parquet(openfootball_file)

def load_football_data_org() -> Optional[pd.DataFrame]:
    """Load and parse Football-Data.org API data."""
    api_dir = RAW_DIR / "football_data_org"
    
    if not api_dir.exists():
        print("⚠️  Football-Data.org API data not found")
        return None
    
    print("📊 Processing Football-Data.org API data...")
    
    matches_data = []
    
    # Process match files
    for match_file in api_dir.glob("*_matches.json"):
        try:
            with open(match_file, 'r') as f:
                data = json.load(f)
            
            if 'matches' not in data:
                continue
            
            comp_code = match_file.stem.replace('_matches', '')
            
            for match in data['matches']:
                try:
                    match_data = {
                        'Date': pd.to_datetime(match.get('utcDate')),
                        'HomeTeam': match['homeTeam']['name'],
                        'AwayTeam': match['awayTeam']['name'],
                        'League': _get_competition_name(comp_code),
                        'Country': _get_country_from_competition(comp_code),
                        'Season': match.get('season', {}).get('startYear', ''),
                        'Matchday': match.get('matchday', ''),
                        'Status': match.get('status', ''),
                        'Source': 'Football-Data.org'
                    }
                    
                    # Score information
                    score = match.get('score', {})
                    ft_score = score.get('fullTime', {})
                    ht_score = score.get('halfTime', {})
                    
                    if ft_score.get('home') is not None:
                        match_data['Home_Goals'] = ft_score['home']
                        match_data['Away_Goals'] = ft_score['away']
                        
                        # Determine result
                        if ft_score['home'] > ft_score['away']:
                            match_data['Result'] = 'H'
                        elif ft_score['home'] < ft_score['away']:
                            match_data['Result'] = 'A'
                        else:
                            match_data['Result'] = 'D'
                    
                    if ht_score.get('home') is not None:
                        match_data['Home_HT_Goals'] = ht_score['home']
                        match_data['Away_HT_Goals'] = ht_score['away']
                    
                    matches_data.append(match_data)
                    
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"⚠️  Error processing {match_file.name}: {e}")
            continue
    
    if matches_data:
        df = pd.DataFrame(matches_data)
        # Remove incomplete matches
        df = df.dropna(subset=['Home_Goals', 'Away_Goals'])
        print(f"✅ Loaded {len(df)} matches from Football-Data.org")
        return df
    else:
        print("⚠️  No Football-Data.org matches found")
        return None

def load_fivethirtyeight_data() -> Optional[pd.DataFrame]:
    """Load FiveThirtyEight SPI data."""
    spi_matches_file = RAW_DIR / "spi_matches.csv"
    spi_rankings_file = RAW_DIR / "spi_global_rankings.csv"
    
    data_frames = []
    
    if spi_matches_file.exists():
        print("📈 Loading FiveThirtyEight SPI matches...")
        df_matches = pd.read_csv(spi_matches_file)
        
        # Standardize column names
        df_matches = df_matches.rename(columns={
            'date': 'Date',
            'team1': 'HomeTeam', 
            'team2': 'AwayTeam',
            'league': 'League',
            'score1': 'Home_Goals',
            'score2': 'Away_Goals',
            'xg1': 'Home_xG_538',
            'xg2': 'Away_xG_538',
            'prob1': 'Home_Win_Prob_538',
            'prob2': 'Away_Win_Prob_538',
            'probtie': 'Draw_Prob_538',
            'importance1': 'Home_Importance_538',
            'importance2': 'Away_Importance_538'
        })
        
        df_matches['Date'] = pd.to_datetime(df_matches['Date'])
        df_matches['Source'] = 'FiveThirtyEight'
        
        # Determine result
        df_matches['Result'] = df_matches.apply(
            lambda x: 'H' if x['Home_Goals'] > x['Away_Goals'] 
                     else 'A' if x['Home_Goals'] < x['Away_Goals'] 
                     else 'D', axis=1
        )
        
        # Add country information
        df_matches['Country'] = df_matches['League'].map(_get_country_from_league_538)
        
        data_frames.append(df_matches)
        print(f"✅ Loaded {len(df_matches)} matches from 538 SPI")
    
    if spi_rankings_file.exists():
        print("🏆 Loading FiveThirtyEight SPI rankings...")
        # Rankings data can be used later for team strength features
        pass
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        print("⚠️  No FiveThirtyEight data found")
        return None

def load_football_data_uk() -> Optional[pd.DataFrame]:
    """Load Football-Data.co.uk historical data."""
    uk_dir = RAW_DIR / "football_data_uk"
    
    if not uk_dir.exists():
        print("⚠️  Football-Data.co.uk data not found")
        return None
    
    print("🇬🇧 Loading Football-Data.co.uk historical data...")
    
    all_matches = []
    
    for csv_file in uk_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            # Extract league and season from filename
            filename_parts = csv_file.stem.split('_')
            league_code = filename_parts[0]
            season_code = filename_parts[1] if len(filename_parts) > 1 else 'unknown'
            
            # Map league codes to names
            league_name = _get_league_name_from_code(league_code)
            country = _get_country_from_league_code(league_code)
            
            # Standardize column names (Football-Data.co.uk format)
            if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
                df = df.rename(columns={
                    'FTHG': 'Home_Goals',
                    'FTAG': 'Away_Goals',
                    'FTR': 'Result',
                    'HTHG': 'Home_HT_Goals', 
                    'HTAG': 'Away_HT_Goals',
                    'HTR': 'HT_Result',
                    'HS': 'Home_Shots',
                    'AS': 'Away_Shots',
                    'HST': 'Home_Shots_Target',
                    'AST': 'Away_Shots_Target',
                    'HC': 'Home_Corners',
                    'AC': 'Away_Corners',
                    'HF': 'Home_Fouls',
                    'AF': 'Away_Fouls',
                    'HY': 'Home_Yellow',
                    'AY': 'Away_Yellow',
                    'HR': 'Home_Red',
                    'AR': 'Away_Red'
                })
                
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
                df['League'] = league_name
                df['Country'] = country
                df['Season'] = _format_season_from_code(season_code)
                df['Source'] = 'Football-Data.co.uk'
                
                # Clean data
                df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'Home_Goals', 'Away_Goals'])
                
                all_matches.append(df)
                print(f"✅ Loaded {len(df)} matches from {csv_file.name}")
        
        except Exception as e:
            print(f"⚠️  Error loading {csv_file.name}: {e}")
            continue
    
    if all_matches:
        combined_df = pd.concat(all_matches, ignore_index=True)
        print(f"🎯 Total Football-Data.co.uk matches: {len(combined_df)}")
        return combined_df
    else:
        print("❌ No Football-Data.co.uk data loaded")
        return None

def _get_competition_name(comp_code: str) -> str:
    """Map competition codes to names."""
    mapping = {
        'PL': 'Premier League',
        'BL1': 'Bundesliga',
        'SA': 'Serie A', 
        'PD': 'La Liga',
        'FL1': 'Ligue 1',
        'CL': 'Champions League',
        'DED': 'Eredivisie'
    }
    return mapping.get(comp_code, comp_code)

def _get_country_from_competition(comp_code: str) -> str:
    """Map competition codes to countries."""
    mapping = {
        'PL': 'England',
        'BL1': 'Germany',
        'SA': 'Italy',
        'PD': 'Spain', 
        'FL1': 'France',
        'CL': 'Europe',
        'DED': 'Netherlands'
    }
    return mapping.get(comp_code, 'Unknown')

def _get_country_from_league_538(league: str) -> str:
    """Map 538 league names to countries."""
    if pd.isna(league):
        return 'Unknown'
    
    league_lower = league.lower()
    if 'premier league' in league_lower or 'england' in league_lower:
        return 'England'
    elif 'bundesliga' in league_lower or 'germany' in league_lower:
        return 'Germany'
    elif 'serie a' in league_lower or 'italy' in league_lower:
        return 'Italy'
    elif 'la liga' in league_lower or 'spain' in league_lower:
        return 'Spain'
    elif 'ligue 1' in league_lower or 'france' in league_lower:
        return 'France'
    elif 'champions' in league_lower:
        return 'Europe'
    else:
        return 'Other'

def _get_league_name_from_code(code: str) -> str:
    """Map Football-Data.co.uk league codes to names."""
    mapping = {
        'E0': 'Premier League',
        'E1': 'Championship',
        'E2': 'League One',
        'E3': 'League Two',
        'D1': 'Bundesliga',
        'D2': '2. Bundesliga',
        'I1': 'Serie A',
        'I2': 'Serie B', 
        'SP1': 'La Liga',
        'SP2': 'Segunda División',
        'F1': 'Ligue 1',
        'F2': 'Ligue 2'
    }
    return mapping.get(code, code)

def _get_country_from_league_code(code: str) -> str:
    """Map Football-Data.co.uk league codes to countries."""
    if code.startswith('E'):
        return 'England'
    elif code.startswith('D'):
        return 'Germany'
    elif code.startswith('I'):
        return 'Italy'
    elif code.startswith('SP'):
        return 'Spain'
    elif code.startswith('F'):
        return 'France'
    else:
        return 'Unknown'

def _format_season_from_code(season_code: str) -> str:
    """Format season code to readable format."""
    if len(season_code) == 4:
        year1 = '20' + season_code[:2]
        year2 = '20' + season_code[2:]
        return f"{year1}-{year2}"
    else:
        return season_code

def combine_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple datasets with common schema."""
    
    if not datasets:
        raise ValueError("No datasets provided")
    
    print(f"🔗 Combining {len(datasets)} datasets...")
    
    # Get common columns across all datasets
    common_cols = set(datasets[0].columns)
    for df in datasets[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    print(f"📊 Common columns: {len(common_cols)}")
    
    # Ensure all datasets have minimum required columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Home_Goals', 'Away_Goals', 'Result', 'League', 'Source']
    missing_required = set(required_cols) - common_cols
    
    if missing_required:
        print(f"⚠️  Missing required columns: {missing_required}")
        # Add missing columns with default values where possible
        for df in datasets:
            for col in missing_required:
                if col not in df.columns:
                    if col == 'Result':
                        df[col] = df.apply(
                            lambda x: 'H' if x.get('Home_Goals', 0) > x.get('Away_Goals', 0)
                                     else 'A' if x.get('Home_Goals', 0) < x.get('Away_Goals', 0)
                                     else 'D', axis=1
                        )
                    else:
                        df[col] = 'Unknown'
        
        common_cols.update(required_cols)
    
    # Select common columns and combine
    combined_datasets = []
    for i, df in enumerate(datasets):
        df_subset = df[list(common_cols)].copy()
        print(f"Dataset {i+1}: {len(df_subset)} matches")
        combined_datasets.append(df_subset)
    
    combined_df = pd.concat(combined_datasets, ignore_index=True)
    
    # Remove duplicates based on key columns
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=['Date', 'HomeTeam', 'AwayTeam', 'League'], 
        keep='first'
    )
    final_count = len(combined_df)
    
    if initial_count != final_count:
        print(f"🔄 Removed {initial_count - final_count} duplicate matches")
    
    return combined_df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced ML features."""
    
    print("🔬 Adding advanced features...")
    
    # Basic derived features
    df['Goal_Difference'] = df['Home_Goals'] - df['Away_Goals']
    df['Total_Goals'] = df['Home_Goals'] + df['Away_Goals']
    
    # Match characteristics
    df['High_Scoring'] = (df['Total_Goals'] >= 3).astype(int)
    df['Low_Scoring'] = (df['Total_Goals'] <= 1).astype(int)
    df['Competitive'] = (abs(df['Goal_Difference']) <= 1).astype(int)
    
    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Weekend'] = (df['DayOfWeek'].isin([5, 6])).astype(int)
    
    # Rolling form features (last 5 matches)
    print("📈 Computing team form features...")
    
    for team_col in ['HomeTeam', 'AwayTeam']:
        prefix = 'Home' if team_col == 'HomeTeam' else 'Away'
        
        # Team-specific statistics
        team_stats = df.groupby([team_col, 'League']).apply(
            lambda x: compute_team_rolling_stats(x, team_col, prefix)
        ).reset_index(level=[0, 1], drop=True)
        
        # Merge back to main dataframe
        df = df.join(team_stats, how='left')
    
    # League-specific features
    print("🏆 Adding league-specific features...")
    league_stats = df.groupby('League').agg({
        'Total_Goals': 'mean',
        'Home_Goals': 'mean', 
        'Away_Goals': 'mean',
        'Competitive': 'mean'
    }).add_suffix('_League_Avg')
    
    df = df.merge(league_stats, left_on='League', right_index=True, how='left')
    
    return df

def compute_team_rolling_stats(team_df: pd.DataFrame, team_col: str, prefix: str) -> pd.Series:
    """Compute rolling statistics for a team."""
    
    team_df = team_df.sort_values('Date')
    
    # Goals scored/conceded when playing home/away
    if prefix == 'Home':
        goals_for = team_df['Home_Goals'].rolling(5, min_periods=1).mean()
        goals_against = team_df['Away_Goals'].rolling(5, min_periods=1).mean()
        wins = (team_df['Result'] == 'H').rolling(5, min_periods=1).sum()
    else:
        goals_for = team_df['Away_Goals'].rolling(5, min_periods=1).mean()
        goals_against = team_df['Home_Goals'].rolling(5, min_periods=1).mean()
        wins = (team_df['Result'] == 'A').rolling(5, min_periods=1).sum()
    
    return pd.Series({
        f'{prefix}_Goals_For_L5': goals_for,
        f'{prefix}_Goals_Against_L5': goals_against,
        f'{prefix}_Wins_L5': wins,
        f'{prefix}_Form_L5': wins / 5  # Win percentage
    }, index=team_df.index)

def create_feature_summary(df: pd.DataFrame) -> Dict:
    """Create a summary of the final dataset."""
    
    summary = {
        'dataset_info': {
            'total_matches': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'leagues': df['League'].nunique(),
            'countries': df['Country'].nunique() if 'Country' in df.columns else 0,
            'seasons': df['Season'].nunique() if 'Season' in df.columns else 0,
            'features': len(df.columns)
        },
        'sources': df['Source'].value_counts().to_dict(),
        'leagues': df['League'].value_counts().to_dict(),
        'countries': df['Country'].value_counts().to_dict() if 'Country' in df.columns else {},
        'feature_types': {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object']).columns),
            'datetime': len(df.select_dtypes(include=['datetime64']).columns)
        },
        'match_outcomes': df['Result'].value_counts().to_dict(),
        'goals_stats': {
            'avg_total_goals': float(df['Total_Goals'].mean()),
            'avg_home_goals': float(df['Home_Goals'].mean()),
            'avg_away_goals': float(df['Away_Goals'].mean()),
            'high_scoring_pct': float(df['High_Scoring'].mean() * 100),
            'low_scoring_pct': float(df['Low_Scoring'].mean() * 100)
        }
    }
    
    return summary

def main():
    """Main feature engineering pipeline."""
    
    print("🚀 LIGHTWEIGHT FOOTBALL FEATURE ENGINEERING")
    print("=" * 60)
    print("Processing multiple small datasets instead of large StatsBomb...")
    print("=" * 60)
    
    datasets = []
    
    # Load all available datasets
    print("\n📥 Loading datasets...")
    
    # 1. OpenFootball data
    openfootball_df = load_openfootball_data()
    if openfootball_df is not None:
        datasets.append(openfootball_df)
    
    # 2. Football-Data.org API data
    api_df = load_football_data_org()
    if api_df is not None:
        datasets.append(api_df)
    
    # 3. FiveThirtyEight data
    fte_df = load_fivethirtyeight_data()
    if fte_df is not None:
        datasets.append(fte_df)
    
    # 4. Football-Data.co.uk data
    uk_df = load_football_data_uk()
    if uk_df is not None:
        datasets.append(uk_df)
    
    if not datasets:
        print("❌ No datasets loaded! Run 'python src/download_data.py' first.")
        return False
    
    print(f"\n🎯 Successfully loaded {len(datasets)} datasets")
    
    # Combine datasets
    try:
        combined_df = combine_datasets(datasets)
        print(f"📊 Combined dataset: {len(combined_df)} matches")
    except Exception as e:
        print(f"❌ Error combining datasets: {e}")
        return False
    
    # Add advanced features
    try:
        final_df = add_advanced_features(combined_df)
        print(f"🔬 Final dataset: {final_df.shape}")
    except Exception as e:
        print(f"❌ Error adding features: {e}")
        return False
    
    # Data validation
    print("\n🔍 Data validation...")
    null_counts = final_df.isnull().sum()
    if null_counts.sum() > 0:
        print("⚠️  Found null values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  • {col}: {count}")
    
    # Remove rows with missing critical data
    critical_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Home_Goals', 'Away_Goals', 'Result']
    initial_count = len(final_df)
    final_df = final_df.dropna(subset=critical_cols)
    final_count = len(final_df)
    
    if initial_count != final_count:
        print(f"🧹 Removed {initial_count - final_count} rows with missing critical data")
    
    # Save final dataset
    final_df.to_parquet(COMBINED_FEATURES_FILE, index=False)
    print(f"\n✅ Saved combined features: {COMBINED_FEATURES_FILE.relative_to(PROJECT_ROOT)}")
    
    # Create and save summary
    summary = create_feature_summary(final_df)
    with open(FEATURE_SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n📈 DATASET SUMMARY")
    print("=" * 40)
    print(f"Total matches: {summary['dataset_info']['total_matches']:,}")
    print(f"Date range: {summary['dataset_info']['date_range']['start']} to {summary['dataset_info']['date_range']['end']}")
    print(f"Leagues: {summary['dataset_info']['leagues']}")
    print(f"Countries: {summary['dataset_info']['countries']}")
    print(f"Features: {summary['dataset_info']['features']}")
    
    print(f"\nData sources:")
    for source, count in summary['sources'].items():
        print(f"  • {source}: {count:,} matches")
    
    print(f"\nTop leagues:")
    for league, count in list(summary['leagues'].items())[:5]:
        print(f"  • {league}: {count:,} matches")
    
    print(f"\nMatch outcomes:")
    total_matches = sum(summary['match_outcomes'].values())
    for outcome, count in summary['match_outcomes'].items():
        pct = count / total_matches * 100
        outcome_name = {'H': 'Home wins', 'A': 'Away wins', 'D': 'Draws'}[outcome]
        print(f"  • {outcome_name}: {count:,} ({pct:.1f}%)")
    
    print(f"\nGoals statistics:")
    goals_stats = summary['goals_stats'] 
    print(f"  • Average total goals: {goals_stats['avg_total_goals']:.2f}")
    print(f"  • Average home goals: {goals_stats['avg_home_goals']:.2f}")
    print(f"  • Average away goals: {goals_stats['avg_away_goals']:.2f}")
    print(f"  • High scoring games: {goals_stats['high_scoring_pct']:.1f}%")
    print(f"  • Low scoring games: {goals_stats['low_scoring_pct']:.1f}%")
    
    print(f"\n📁 Summary saved: {FEATURE_SUMMARY_FILE.relative_to(PROJECT_ROOT)}")
    print("▶️  Next step: python src/train_model.py")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 