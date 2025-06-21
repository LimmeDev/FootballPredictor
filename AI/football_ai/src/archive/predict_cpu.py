#!/usr/bin/env python3
"""
Simple Prediction Script for CPU-trained XGBoost Model
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "cpu_xgboost_model.json"
METADATA_FILE = MODELS_DIR / "cpu_model_metadata.json"

def load_model_and_metadata():
    """Load the trained model and its metadata."""
    if not MODEL_FILE.exists():
        print("❌ Model not found! Run train_cpu.py first.")
        return None, None
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(str(MODEL_FILE))
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def create_sample_match_data(home_team, away_team, league="Premier League"):
    """Create sample match data for prediction."""
    # Create data in the same order as training
    sample_data = {
        'Home_Goals': [1],  # Example values
        'Away_Goals': [1],
        'Home_HT_Goals': [0],
        'Away_HT_Goals': [0],
        'Home_xG_Est': [1.2],
        'Away_xG_Est': [0.8],
        'Goal_Difference': [0],
        'Total_Goals': [2],
        'High_Scoring': [0],
        'Low_Scoring': [0],
        'HomeTeam_Encoded': [hash(home_team) % 100],
        'AwayTeam_Encoded': [hash(away_team) % 100],
        'League_Encoded': [hash(league) % 10],
        'Competitive': [1],
        'Season_Progress': [0.5],
        'Home_Win_Margin': [0],
        'Away_Win_Margin': [0],
        'Year': [2024],
        'Month': [12],
        'Day_of_Week': [0]
    }
    
    df = pd.DataFrame(sample_data)
    return df

def predict_match(model, home_team, away_team, league="Premier League"):
    """Predict match outcome."""
    print(f"⚽ Predicting: {home_team} vs {away_team}")
    print(f"🏆 League: {league}")
    
    # Create sample data
    X = create_sample_match_data(home_team, away_team, league)
    
    # Make prediction
    dtest = xgb.DMatrix(X)
    probabilities = model.predict(dtest)[0]  # Get first (and only) prediction
    
    # Map to outcomes
    outcomes = ['Home Win', 'Draw', 'Away Win']
    predicted_class = np.argmax(probabilities)
    predicted_outcome = outcomes[predicted_class]
    
    print(f"\n📊 PREDICTION RESULTS:")
    print(f"🥇 Most Likely: {predicted_outcome} ({probabilities[predicted_class]:.3f})")
    print(f"\n📈 All Probabilities:")
    for i, (outcome, prob) in enumerate(zip(outcomes, probabilities)):
        emoji = "🥇" if i == predicted_class else "⚪"
        print(f"   {emoji} {outcome}: {prob:.3f} ({prob*100:.1f}%)")
    
    return predicted_outcome, probabilities

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict football match outcomes")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name") 
    parser.add_argument("--league", default="Premier League", help="League name")
    args = parser.parse_args()
    
    print("🏈 FOOTBALL MATCH PREDICTION")
    print("=" * 40)
    
    # Load model
    model, metadata = load_model_and_metadata()
    if model is None:
        return False
    
    print(f"🤖 Model: {metadata['model_type']}")
    print(f"🎯 Training Accuracy: {metadata['accuracy']:.3f}")
    print(f"⚡ Device: {metadata['device']}")
    print("=" * 40)
    
    # Make prediction
    prediction, probabilities = predict_match(model, args.home, args.away, args.league)
    
    print(f"\n🎉 Prediction Complete!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 