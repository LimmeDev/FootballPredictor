#!/usr/bin/env python3
"""
Advanced Feature Engineering for Realistic Football Predictions
Incorporates team strength, form, injuries, head-to-head, UEFA coefficients
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Input files
OPENFOOTBALL_FILE = PROCESSED_DIR / "openfootball_features.parquet"
TEAM_STRENGTH_FILE = RAW_DIR / "team_strength.json"
UEFA_COEFF_FILE = RAW_DIR / "uefa_coefficients.json"
HISTORICAL_FILE = RAW_DIR / "historical_results.json"
INJURY_FILE = RAW_DIR / "injury_data.json"
FORM_FILE = RAW_DIR / "team_form.json"

# Output file
ADVANCED_FEATURES_FILE = PROCESSED_DIR / "advanced_features.parquet"
FEATURE_SUMMARY_FILE = PROCESSED_DIR / "advanced_feature_summary.json"

def load_base_data() -> Optional[pd.DataFrame]:
    """Load the base OpenFootball data."""
    if OPENFOOTBALL_FILE.exists():
        print("✅ Loading base OpenFootball data...")
        return pd.read_parquet(OPENFOOTBALL_FILE)
    else:
        print("❌ Base data not found! Run download and parse first.")
        return None

def load_team_strength() -> Dict:
    """Load team strength data."""
    if TEAM_STRENGTH_FILE.exists():
        with open(TEAM_STRENGTH_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_uefa_coefficients() -> Dict:
    """Load UEFA coefficients."""
    if UEFA_COEFF_FILE.exists():
        with open(UEFA_COEFF_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_historical_results() -> List[Dict]:
    """Load historical head-to-head results."""
    if HISTORICAL_FILE.exists():
        with open(HISTORICAL_FILE, 'r') as f:
            return json.load(f)
    return []

def load_injury_data() -> Dict:
    """Load current injury data."""
    if INJURY_FILE.exists():
        with open(INJURY_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_form_data() -> Dict:
    """Load team form data."""
    if FORM_FILE.exists():
        with open(FORM_FILE, 'r') as f:
            return json.load(f)
    return {}

def add_team_strength_features(df: pd.DataFrame, team_strength: Dict) -> pd.DataFrame:
    """Add team strength features."""
    print("🏆 Adding team strength features...")
    
    # Initialize with default values
    df['Home_Market_Value'] = 400  # Default market value in millions
    df['Away_Market_Value'] = 400
    df['Home_Avg_Age'] = 26.0
    df['Away_Avg_Age'] = 26.0
    df['Home_Tier'] = 2  # Default tier (1=top, 2=mid, 3=lower)
    df['Away_Tier'] = 2
    
    # Fill with actual data where available
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        if home_team in team_strength:
            df.at[idx, 'Home_Market_Value'] = team_strength[home_team]['market_value']
            df.at[idx, 'Home_Avg_Age'] = team_strength[home_team]['avg_age']
            df.at[idx, 'Home_Tier'] = team_strength[home_team]['tier']
        
        if away_team in team_strength:
            df.at[idx, 'Away_Market_Value'] = team_strength[away_team]['market_value']
            df.at[idx, 'Away_Avg_Age'] = team_strength[away_team]['avg_age']
            df.at[idx, 'Away_Tier'] = team_strength[away_team]['tier']
    
    # Derived strength features
    df['Market_Value_Ratio'] = df['Home_Market_Value'] / df['Away_Market_Value']
    df['Age_Difference'] = df['Home_Avg_Age'] - df['Away_Avg_Age']
    df['Tier_Advantage'] = df['Away_Tier'] - df['Home_Tier']  # Positive = home advantage
    
    return df

def add_uefa_coefficient_features(df: pd.DataFrame, uefa_coeffs: Dict) -> pd.DataFrame:
    """Add UEFA coefficient features."""
    print("🇪🇺 Adding UEFA coefficient features...")
    
    # Default UEFA coefficient for teams not in database
    df['Home_UEFA_Coeff'] = 30.0  # Average coefficient
    df['Away_UEFA_Coeff'] = 30.0
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        if home_team in uefa_coeffs:
            df.at[idx, 'Home_UEFA_Coeff'] = uefa_coeffs[home_team]
        
        if away_team in uefa_coeffs:
            df.at[idx, 'Away_UEFA_Coeff'] = uefa_coeffs[away_team]
    
    # Derived UEFA features
    df['UEFA_Coeff_Difference'] = df['Home_UEFA_Coeff'] - df['Away_UEFA_Coeff']
    df['UEFA_Quality_Match'] = (df['Home_UEFA_Coeff'] + df['Away_UEFA_Coeff']) / 2
    
    return df

def add_form_features(df: pd.DataFrame, form_data: Dict) -> pd.DataFrame:
    """Add recent form features."""
    print("📈 Adding team form features...")
    
    # Default form values
    df['Home_Form_Points'] = 1.5  # Average points per game
    df['Away_Form_Points'] = 1.5
    df['Home_Last_5_Points'] = 7.5  # Average 1.5 * 5
    df['Away_Last_5_Points'] = 7.5
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        if home_team in form_data:
            df.at[idx, 'Home_Form_Points'] = form_data[home_team]['points_per_game']
            df.at[idx, 'Home_Last_5_Points'] = sum(form_data[home_team]['last_5'])
        
        if away_team in form_data:
            df.at[idx, 'Away_Form_Points'] = form_data[away_team]['points_per_game']
            df.at[idx, 'Away_Last_5_Points'] = sum(form_data[away_team]['last_5'])
    
    # Derived form features
    df['Form_Difference'] = df['Home_Form_Points'] - df['Away_Form_Points']
    df['Recent_Form_Diff'] = df['Home_Last_5_Points'] - df['Away_Last_5_Points']
    
    return df

def add_injury_features(df: pd.DataFrame, injury_data: Dict) -> pd.DataFrame:
    """Add injury/availability features."""
    print("🏥 Adding injury/availability features...")
    
    # Default injury values
    df['Home_Injured_Count'] = 1  # Average injuries
    df['Away_Injured_Count'] = 1
    df['Home_Suspended_Count'] = 0
    df['Away_Suspended_Count'] = 0
    df['Home_Key_Players_Out'] = 0
    df['Away_Key_Players_Out'] = 0
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        if home_team in injury_data:
            df.at[idx, 'Home_Injured_Count'] = injury_data[home_team]['injured']
            df.at[idx, 'Home_Suspended_Count'] = injury_data[home_team]['suspended']
            df.at[idx, 'Home_Key_Players_Out'] = len(injury_data[home_team]['key_players_out'])
        
        if away_team in injury_data:
            df.at[idx, 'Away_Injured_Count'] = injury_data[away_team]['injured']
            df.at[idx, 'Away_Suspended_Count'] = injury_data[away_team]['suspended']
            df.at[idx, 'Away_Key_Players_Out'] = len(injury_data[away_team]['key_players_out'])
    
    # Derived injury features
    df['Injury_Advantage'] = df['Away_Injured_Count'] - df['Home_Injured_Count']
    df['Availability_Score'] = (
        (5 - df['Home_Injured_Count'] - df['Home_Suspended_Count'] - df['Home_Key_Players_Out']) -
        (5 - df['Away_Injured_Count'] - df['Away_Suspended_Count'] - df['Away_Key_Players_Out'])
    )
    
    return df

def add_historical_h2h_features(df: pd.DataFrame, historical_data: List[Dict]) -> pd.DataFrame:
    """Add historical head-to-head features."""
    print("📊 Adding head-to-head historical features...")
    
    # Create head-to-head lookup
    h2h_lookup = {}
    for record in historical_data:
        key = (record['home_team'], record['away_team'])
        if key not in h2h_lookup:
            h2h_lookup[key] = []
        h2h_lookup[key].append(record)
    
    # Default H2H values
    df['H2H_Home_Wins'] = 1  # Default even record
    df['H2H_Draws'] = 1
    df['H2H_Away_Wins'] = 1
    df['H2H_Home_Win_Rate'] = 0.33
    df['H2H_Recent_Form'] = 0  # 0 = even, positive = home advantage
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Check both directions for H2H
        key = (home_team, away_team)
        reverse_key = (away_team, home_team)
        
        h2h_records = h2h_lookup.get(key, []) + h2h_lookup.get(reverse_key, [])
        
        if h2h_records:
            home_wins = sum(1 for r in h2h_records if r['result'] == 2)
            draws = sum(1 for r in h2h_records if r['result'] == 1)
            away_wins = sum(1 for r in h2h_records if r['result'] == 0)
            total_games = len(h2h_records)
            
            df.at[idx, 'H2H_Home_Wins'] = home_wins
            df.at[idx, 'H2H_Draws'] = draws
            df.at[idx, 'H2H_Away_Wins'] = away_wins
            df.at[idx, 'H2H_Home_Win_Rate'] = home_wins / total_games if total_games > 0 else 0.33
            
            # Recent form (last 3 games)
            recent_results = [r['result'] for r in sorted(h2h_records, key=lambda x: x['games_ago'])[:3]]
            home_recent = sum(1 for r in recent_results if r == 2)
            away_recent = sum(1 for r in recent_results if r == 0)
            df.at[idx, 'H2H_Recent_Form'] = home_recent - away_recent
    
    return df

def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competition-specific features."""
    print("🏆 Adding competition features...")
    
    # Competition importance/pressure
    competition_weights = {
        'Champions League': 1.0,
        'Europa League': 0.8,
        'Premier League': 0.9,
        'La Liga': 0.9,
        'Bundesliga': 0.9,
        'Serie A': 0.9,
        'Ligue 1': 0.8,
        'FA Cup': 0.6,
        'Copa del Rey': 0.6,
        'DFB-Pokal': 0.6,
    }
    
    df['Competition_Weight'] = df['League'].map(competition_weights).fillna(0.7)
    
    # Home advantage varies by competition
    df['Home_Advantage'] = np.where(
        df['League'].isin(['Champions League', 'Europa League']), 
        1.2,  # Less home advantage in European competitions
        1.4   # Standard home advantage
    )
    
    # Derby/rivalry matches (simplified)
    rivalry_pairs = [
        ('Manchester City', 'Manchester United'),
        ('Liverpool', 'Everton'),
        ('Arsenal', 'Tottenham'),
        ('Barcelona', 'Real Madrid'),
        ('Milan', 'Inter'),
        ('Bayern München', 'Borussia Dortmund'),
    ]
    
    df['Is_Derby'] = 0
    for home, away in rivalry_pairs:
        mask = ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) | \
               ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
        df.loc[mask, 'Is_Derby'] = 1
    
    return df

def add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add situational context features."""
    print("⚽ Adding situational features...")
    
    # Convert date to datetime if it isn't already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Season timing
    df['Days_Since_Season_Start'] = (df['Date'] - df['Date'].groupby(df['Season']).transform('min')).dt.days
    df['Season_Stage'] = pd.cut(df['Days_Since_Season_Start'], 
                               bins=[0, 60, 120, 180, 365], 
                               labels=[0, 1, 2, 3])  # Early, Mid-Early, Mid-Late, Late
    
    # Match importance (simplified)
    df['Match_Importance'] = 1.0
    
    # European competition dates
    df['Is_European_Competition'] = df['League'].isin(['Champions League', 'Europa League', 'Conference League']).astype(int)
    
    # Weekend vs midweek
    df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)  # Saturday, Sunday
    
    return df

def main():
    """Create advanced features for realistic predictions."""
    print("🚀 ADVANCED FEATURE ENGINEERING")
    print("=" * 60)
    print("🎯 Creating realistic football prediction features...")
    print("📊 Incorporating: Team strength, form, injuries, H2H, UEFA coefficients")
    print("=" * 60)
    
    # Load base data
    df = load_base_data()
    if df is None:
        return False
    
    print(f"📈 Starting with {len(df)} matches")
    
    # Load all auxiliary data
    team_strength = load_team_strength()
    uefa_coeffs = load_uefa_coefficients()
    historical_data = load_historical_results()
    injury_data = load_injury_data()
    form_data = load_form_data()
    
    print(f"🗃️  Loaded {len(team_strength)} team strength records")
    print(f"🇪🇺 Loaded {len(uefa_coeffs)} UEFA coefficients")
    print(f"📊 Loaded {len(historical_data)} historical results")
    print(f"🏥 Loaded {len(injury_data)} injury reports")
    print(f"📈 Loaded {len(form_data)} form records")
    
    # Add all advanced features
    df = add_team_strength_features(df, team_strength)
    df = add_uefa_coefficient_features(df, uefa_coeffs)
    df = add_form_features(df, form_data)
    df = add_injury_features(df, injury_data)
    df = add_historical_h2h_features(df, historical_data)
    df = add_competition_features(df)
    df = add_situational_features(df)
    
    # Clean data
    print("🧹 Cleaning and finalizing data...")
    df = df.dropna(subset=['Home_Goals', 'Away_Goals', 'Result'])
    df = df[df['Result'].isin(['H', 'D', 'A'])]
    
    # Result encoding for ML
    result_mapping = {'H': 0, 'D': 1, 'A': 2}
    df['Result_Encoded'] = df['Result'].map(result_mapping)
    
    print(f"📊 Final dataset: {len(df)} matches")
    print(f"📈 Total features: {len(df.columns)}")
    
    # Save advanced features
    df.to_parquet(ADVANCED_FEATURES_FILE, index=False)
    print(f"✅ Saved to: {ADVANCED_FEATURES_FILE}")
    
    # Create feature summary
    feature_categories = {
        'basic': ['Home_Goals', 'Away_Goals', 'Total_Goals', 'Goal_Difference'],
        'team_strength': ['Market_Value_Ratio', 'Age_Difference', 'Tier_Advantage'],
        'uefa': ['UEFA_Coeff_Difference', 'UEFA_Quality_Match'],
        'form': ['Form_Difference', 'Recent_Form_Diff'],
        'injuries': ['Injury_Advantage', 'Availability_Score'],
        'historical': ['H2H_Home_Win_Rate', 'H2H_Recent_Form'],
        'competition': ['Competition_Weight', 'Home_Advantage', 'Is_Derby'],
        'situational': ['Season_Stage', 'Is_European_Competition', 'Is_Weekend']
    }
    
    summary = {
        'total_matches': len(df),
        'total_features': len(df.columns),
        'feature_categories': feature_categories,
        'target_distribution': df['Result'].value_counts().to_dict(),
        'leagues': df['League'].unique().tolist(),
        'date_range': {
            'start': df['Date'].min().isoformat(),
            'end': df['Date'].max().isoformat()
        }
    }
    
    with open(FEATURE_SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n📊 ADVANCED FEATURES SUMMARY:")
    print(f"  • Total matches: {summary['total_matches']:,}")
    print(f"  • Total features: {summary['total_features']}")
    print(f"  • Feature categories: {len(feature_categories)}")
    print(f"  • Leagues: {len(summary['leagues'])}")
    print(f"  • Results: {summary['target_distribution']}")
    
    print("\n🎉 Advanced feature engineering complete!")
    print("Ready for realistic model training with proper features.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 