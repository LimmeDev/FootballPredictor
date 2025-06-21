#!/usr/bin/env python3
"""
ENHANCED DATA COLLECTOR
Collect more football data from multiple sources for better training
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

class EnhancedDataCollector:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        print("🚀 Enhanced Football Data Collector")
    
    def collect_extended_data(self):
        """Collect extended football data."""
        print("📊 Collecting extended data...")
        
        # Load existing data
        try:
            df = pd.read_parquet(DATA_DIR / "mega_features.parquet")
            print(f"  �� Loaded {len(df)} existing matches")
        except:
            print("  ⚠️  No existing data found, creating new dataset")
            df = pd.DataFrame()
        
        # Generate additional synthetic matches for more training data
        print("🧬 Generating additional training data...")
        
        teams = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Manchester United',
                'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham',
                'Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla', 'Valencia',
                'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
                'Juventus', 'AC Milan', 'Inter Milan', 'Roma', 'Napoli',
                'PSG', 'Monaco', 'Lyon', 'Marseille', 'Ajax', 'PSV']
        
        additional_matches = []
        
        for i in range(5000):  # Generate 5000 additional matches
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate realistic match data
            home_strength = np.random.normal(1400, 200)
            away_strength = np.random.normal(1400, 200)
            
            # Home advantage
            home_strength += 50
            
            strength_diff = home_strength - away_strength
            
            # Determine result based on strength difference
            prob_home = 1 / (1 + np.exp(-strength_diff / 200))
            prob_away = 1 / (1 + np.exp(strength_diff / 200))
            prob_draw = 1 - prob_home - prob_away
            
            result_prob = np.random.random()
            if result_prob < prob_home:
                result = 'H'
                result_encoded = 0
                home_goals = np.random.poisson(2.1)
                away_goals = np.random.poisson(1.3)
            elif result_prob < prob_home + prob_draw:
                result = 'D'
                result_encoded = 1
                home_goals = np.random.poisson(1.5)
                away_goals = home_goals
            else:
                result = 'A'
                result_encoded = 2
                home_goals = np.random.poisson(1.2)
                away_goals = np.random.poisson(2.2)
            
            match_data = {
                'Date': datetime.now() - pd.Timedelta(days=np.random.randint(0, 1000)),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Home_Goals': max(0, home_goals),
                'Away_Goals': max(0, away_goals),
                'Result': result,
                'Result_Encoded': result_encoded,
                'League': np.random.choice(['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']),
                'Country': np.random.choice(['England', 'Spain', 'Germany', 'Italy', 'France']),
                'Source': 'Enhanced_Synthetic',
                
                # Advanced features
                'Home_ELO': home_strength,
                'Away_ELO': away_strength,
                'Home_Attack_Rating': home_strength * (0.9 + np.random.random() * 0.2),
                'Away_Attack_Rating': away_strength * (0.9 + np.random.random() * 0.2),
                'Home_Defense_Rating': home_strength * (0.9 + np.random.random() * 0.2),
                'Away_Defense_Rating': away_strength * (0.9 + np.random.random() * 0.2),
                'Home_Form_PPG': 1.0 + np.random.random() * 2.0,
                'Away_Form_PPG': 1.0 + np.random.random() * 2.0,
                'Home_Market_Value': 50 + np.random.exponential(100),
                'Away_Market_Value': 50 + np.random.exponential(100),
                'Home_Avg_Age': 24 + np.random.normal(2, 1),
                'Away_Avg_Age': 24 + np.random.normal(2, 1),
                'Is_Derby': 1 if np.random.random() < 0.05 else 0,
                'Is_Weekend': 1 if np.random.random() < 0.6 else 0,
                'Temperature': 5 + np.random.random() * 30,
                'Is_Rainy': 1 if np.random.random() < 0.2 else 0,
                'Home_Confidence': 0.3 + np.random.random() * 0.7,
                'Away_Confidence': 0.3 + np.random.random() * 0.7,
                'Competition_Importance': np.random.randint(1, 6),
                'Stakes': np.random.randint(1, 6),
                'Home_Key_Players_Available': 0.7 + np.random.random() * 0.3,
                'Away_Key_Players_Available': 0.7 + np.random.random() * 0.3,
            }
            
            additional_matches.append(match_data)
        
        # Create DataFrame
        new_df = pd.DataFrame(additional_matches)
        
        # Add interaction features
        new_df['ELO_Difference'] = new_df['Home_ELO'] - new_df['Away_ELO']
        new_df['ELO_Sum'] = new_df['Home_ELO'] + new_df['Away_ELO']
        new_df['ELO_Ratio'] = new_df['Home_ELO'] / (new_df['Away_ELO'] + 1e-8)
        new_df['Form_Difference'] = new_df['Home_Form_PPG'] - new_df['Away_Form_PPG']
        new_df['Market_Value_Ratio'] = new_df['Home_Market_Value'] / (new_df['Away_Market_Value'] + 1e-8)
        new_df['Age_Difference'] = new_df['Home_Avg_Age'] - new_df['Away_Avg_Age']
        new_df['Confidence_Difference'] = new_df['Home_Confidence'] - new_df['Away_Confidence']
        
        # Combine with existing data
        if not df.empty:
            # Align columns
            common_cols = set(df.columns) & set(new_df.columns)
            df = df[list(common_cols)]
            new_df = new_df[list(common_cols)]
            combined_df = pd.concat([df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Save enhanced dataset
        output_file = DATA_DIR / "mega_enhanced_features.parquet"
        combined_df.to_parquet(output_file, index=False)
        
        print(f"✅ Enhanced dataset created!")
        print(f"📊 Total matches: {len(combined_df):,}")
        print(f"🔢 Total features: {len(combined_df.columns)}")
        print(f"💾 Saved to: {output_file}")
        
        return True

def main():
    collector = EnhancedDataCollector()
    return collector.collect_extended_data()

if __name__ == "__main__":
    main()
