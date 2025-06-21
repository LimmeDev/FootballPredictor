#!/usr/bin/env python3
"""
Realistic Football Match Predictor for VM
Uses the anti-overfitting trained model
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from pathlib import Path
import json
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "realistic_xgboost_model.json"
METADATA_FILE = MODELS_DIR / "realistic_model_metadata.json"
DATA_FILE = PROJECT_ROOT / "data" / "combined_features.parquet"

def load_realistic_model():
    """Load the realistic trained model."""
    if not MODEL_FILE.exists():
        print("❌ Realistic model not found! Run train_realistic.py first.")
        return None, None
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(str(MODEL_FILE))
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def create_realistic_match_features(home_team, away_team, league="Premier League"):
    """Create realistic match features for prediction."""
    
    # Load training data to get realistic feature ranges
    df = pd.read_parquet(DATA_FILE)
    
    # Create varied, realistic match scenarios
    scenarios = [
        # High-scoring scenario
        {
            'Home_Goals': random.randint(2, 4),
            'Away_Goals': random.randint(1, 3),
            'Home_xG_Est': random.uniform(1.5, 2.8),
            'Away_xG_Est': random.uniform(0.8, 2.2),
            'scenario': 'High-scoring'
        },
        # Defensive scenario
        {
            'Home_Goals': random.randint(0, 1),
            'Away_Goals': random.randint(0, 1),
            'Home_xG_Est': random.uniform(0.3, 1.2),
            'Away_xG_Est': random.uniform(0.3, 1.2),
            'scenario': 'Defensive'
        },
        # Balanced scenario
        {
            'Home_Goals': random.randint(1, 2),
            'Away_Goals': random.randint(1, 2),
            'Home_xG_Est': random.uniform(0.8, 1.8),
            'Away_xG_Est': random.uniform(0.8, 1.8),
            'scenario': 'Balanced'
        }
    ]
    
    # Randomly select a scenario
    scenario = random.choice(scenarios)
    
    # Create match features
    home_goals = scenario['Home_Goals']
    away_goals = scenario['Away_Goals']
    
    features = {
        'Home_Goals': [home_goals],
        'Away_Goals': [away_goals],
        'Home_HT_Goals': [max(0, home_goals - random.randint(0, 2))],
        'Away_HT_Goals': [max(0, away_goals - random.randint(0, 2))],
        'Home_xG_Est': [scenario['Home_xG_Est']],
        'Away_xG_Est': [scenario['Away_xG_Est']],
        'Goal_Difference': [home_goals - away_goals],
        'Total_Goals': [home_goals + away_goals],
        'High_Scoring': [1 if (home_goals + away_goals) >= 3 else 0],
        'Low_Scoring': [1 if (home_goals + away_goals) <= 1 else 0],
        'Home_Win_Margin': [home_goals - away_goals if home_goals > away_goals else 0],
        'Away_Win_Margin': [away_goals - home_goals if away_goals > home_goals else 0],
        'Competitive': [1 if abs(home_goals - away_goals) <= 1 else 0],
        'Season_Progress': [random.uniform(0.1, 0.9)],
        'HomeTeam_Encoded': [abs(hash(home_team)) % 200],
        'AwayTeam_Encoded': [abs(hash(away_team)) % 200],
        'League_Encoded': [abs(hash(league)) % 20],
        'Year': [2024],
        'Month': [random.choice([8, 9, 10, 11, 12, 1, 2, 3, 4, 5])],  # Football season months
        'Day_of_Week': [random.randint(0, 6)]
    }
    
    return pd.DataFrame(features), scenario['scenario']

def predict_realistic_match(model, home_team, away_team, league="Premier League", num_simulations=5):
    """Predict match with multiple realistic scenarios."""
    print(f"⚽ Predicting: {home_team} vs {away_team}")
    print(f"🏆 League: {league}")
    print(f"🎲 Running {num_simulations} scenario simulations...")
    
    all_predictions = []
    scenarios = []
    
    # Run multiple simulations
    for i in range(num_simulations):
        X_pred, scenario = create_realistic_match_features(home_team, away_team, league)
        
        # Make prediction
        dtest = xgb.DMatrix(X_pred)
        probabilities = model.predict(dtest)[0]
        
        all_predictions.append(probabilities)
        scenarios.append(scenario)
    
    # Average the predictions
    avg_probabilities = np.mean(all_predictions, axis=0)
    
    # Map to outcomes
    outcomes = ['Home Win', 'Draw', 'Away Win']
    predicted_class = np.argmax(avg_probabilities)
    predicted_outcome = outcomes[predicted_class]
    
    print(f"\n📊 AVERAGED PREDICTION RESULTS:")
    print(f"🥇 Most Likely: {predicted_outcome} ({avg_probabilities[predicted_class]:.3f})")
    print(f"\n📈 Average Probabilities:")
    for i, (outcome, prob) in enumerate(zip(outcomes, avg_probabilities)):
        emoji = "🥇" if i == predicted_class else "⚪"
        confidence = "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low"
        print(f"   {emoji} {outcome}: {prob:.3f} ({prob*100:.1f}%) - {confidence} confidence")
    
    # Show individual scenario results
    print(f"\n🎯 Individual Scenario Results:")
    for i, (pred, scenario) in enumerate(zip(all_predictions, scenarios)):
        best_outcome = outcomes[np.argmax(pred)]
        confidence = np.max(pred)
        print(f"   Scenario {i+1} ({scenario}): {best_outcome} ({confidence:.3f})")
    
    return predicted_outcome, avg_probabilities

def main():
    """Main realistic prediction function."""
    parser = argparse.ArgumentParser(description="Realistic football match prediction")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name") 
    parser.add_argument("--league", default="Premier League", help="League name")
    parser.add_argument("--simulations", type=int, default=5, help="Number of scenario simulations")
    args = parser.parse_args()
    
    print("🏈 REALISTIC VM FOOTBALL PREDICTION")
    print("=" * 60)
    
    # Load model
    model, metadata = load_realistic_model()
    if model is None:
        return False
    
    print(f"🤖 Model: {metadata['model_type']}")
    print(f"🎯 Accuracy: {metadata['accuracy']:.3f}")
    print(f"⚡ Device: {metadata['device']}")
    print(f"🛡️  Regularization: {metadata['regularization']}")
    print("=" * 60)
    
    # Make prediction
    try:
        prediction, probabilities = predict_realistic_match(
            model, args.home, args.away, args.league, args.simulations
        )
        print(f"\n🎉 Realistic prediction complete!")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 