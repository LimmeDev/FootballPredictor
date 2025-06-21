#!/usr/bin/env python3
"""
Simple Feature Creation - Using only OpenFootball data
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Input/Output files
OPENFOOTBALL_FILE = PROCESSED_DIR / "openfootball_features.parquet"
COMBINED_FEATURES_FILE = PROCESSED_DIR / "combined_features.parquet"
FEATURE_SUMMARY_FILE = PROCESSED_DIR / "feature_summary.json"

def load_openfootball_data() -> pd.DataFrame:
    """Load existing OpenFootball data."""
    if OPENFOOTBALL_FILE.exists():
        print("✅ Loading OpenFootball data...")
        return pd.read_parquet(OPENFOOTBALL_FILE)
    else:
        print("❌ OpenFootball data not found! Run download_data.py first.")
        return None

def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features."""
    print("🔬 Adding simple features...")
    
    # Basic goal statistics
    df['Goal_Difference'] = df['Home_Goals'] - df['Away_Goals']
    df['Total_Goals'] = df['Home_Goals'] + df['Away_Goals']
    df['High_Scoring'] = (df['Total_Goals'] >= 3).astype(int)
    df['Low_Scoring'] = (df['Total_Goals'] <= 1).astype(int)
    
    # Result encoding for ML
    result_mapping = {'H': 0, 'D': 1, 'A': 2}
    df['Result_Encoded'] = df['Result'].map(result_mapping)
    
    # Simple team encoding (this is basic, could be improved)
    df['HomeTeam_Encoded'] = pd.Categorical(df['HomeTeam']).codes
    df['AwayTeam_Encoded'] = pd.Categorical(df['AwayTeam']).codes
    df['League_Encoded'] = pd.Categorical(df['League']).codes
    
    # Date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    
    print(f"✅ Added {len(df.columns)} features")
    return df

def create_summary(df: pd.DataFrame) -> Dict:
    """Create feature summary."""
    summary = {
        'total_matches': len(df),
        'features': list(df.columns),
        'feature_count': len(df.columns),
        'leagues': df['League'].unique().tolist(),
        'date_range': {
            'start': df['Date'].min().isoformat(),
            'end': df['Date'].max().isoformat()
        },
        'target_distribution': df['Result'].value_counts().to_dict()
    }
    return summary

def main():
    """Simple feature creation using only OpenFootball data."""
    print("🚀 SIMPLE FOOTBALL FEATURE CREATION")
    print("=" * 50)
    
    # Load OpenFootball data
    df = load_openfootball_data()
    if df is None:
        return False
    
    print(f"📊 Loaded {len(df)} matches from OpenFootball")
    
    # Add simple features
    df = add_simple_features(df)
    
    # Clean data
    print("🧹 Cleaning data...")
    df = df.dropna(subset=['Home_Goals', 'Away_Goals', 'Result'])
    
    # Remove incomplete matches
    df = df[df['Result'].isin(['H', 'D', 'A'])]
    
    print(f"📊 Final dataset: {len(df)} matches")
    print(f"📈 Features: {len(df.columns)}")
    
    # Save features
    df.to_parquet(COMBINED_FEATURES_FILE, index=False)
    print(f"✅ Saved to: {COMBINED_FEATURES_FILE}")
    
    # Create summary
    summary = create_summary(df)
    with open(FEATURE_SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n📊 DATASET SUMMARY:")
    print(f"  • Total matches: {summary['total_matches']:,}")
    print(f"  • Features: {summary['feature_count']}")
    print(f"  • Leagues: {len(summary['leagues'])}")
    print(f"  • Results: {summary['target_distribution']}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 