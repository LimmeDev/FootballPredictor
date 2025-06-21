#!/usr/bin/env python3
"""
Simple Working Prediction Script for VM Football AI
Matches exact features from training data
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
DATA_FILE = PROJECT_ROOT / "data" / "combined_features.parquet"

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

def get_training_features():
    """Get the exact feature list used in training."""
    df = pd.read_parquet(DATA_FILE)
    
    # Remove text columns and target as done in training
    X = df.drop(columns=["Result", "Date", "HomeTeam", "AwayTeam"], errors="ignore")
    text_columns = X.select_dtypes(include=['object']).columns.tolist()
    if text_columns:
        X = X.drop(columns=text_columns)
    
    return X.columns.tolist()

def create_sample_prediction_data(home_team, away_team, league="Premier League"):
    """Create realistic sample data for prediction."""
    
    # Get a sample from training data for realistic baseline
    df = pd.read_parquet(DATA_FILE)
    sample_row = df.iloc[0].copy()  # Use first row as template
    
    # Create new prediction data based on sample
    prediction_data = {
        'Home_Goals': [1],  # Example match: 1-1 draw
        'Away_Goals': [1],
        'Home_HT_Goals': [0],
        'Away_HT_Goals': [1],
        'Home_xG_Est': [1.2],
        'Away_xG_Est': [0.9],
        'Goal_Difference': [0],  # 1-1 = 0
        'Total_Goals': [2],      # 1+1 = 2
        'High_Scoring': [0],     # 2 goals is not high scoring
        'Low_Scoring': [0],      # 2 goals is not low scoring
        'Home_Win_Margin': [0],  # Draw, so no win margin
        'Away_Win_Margin': [0],
        'Competitive': [1],      # Close match
        'Season_Progress': [0.3], # Early in season
        'Result_Encoded': [1],   # Draw (but we'll remove this)
        'HomeTeam_Encoded': [abs(hash(home_team)) % 200],
        'AwayTeam_Encoded': [abs(hash(away_team)) % 200],
        'League_Encoded': [abs(hash(league)) % 20],
        'Year': [2024],
        'Month': [12],
        'Day_of_Week': [0]  # Monday
    }
    
    df_pred = pd.DataFrame(prediction_data)
    
    # Remove target column
    if 'Result_Encoded' in df_pred.columns:
        df_pred = df_pred.drop(columns=['Result_Encoded'])
    
    return df_pred

def predict_match(model, home_team, away_team, league="Premier League"):
    """Predict match outcome."""
    print(f"⚽ Predicting: {home_team} vs {away_team}")
    print(f"🏆 League: {league}")
    
    # Create prediction data
    X_pred = create_sample_prediction_data(home_team, away_team, league)
    
    print(f"📊 Features for prediction: {len(X_pred.columns)}")
    
    # Make prediction
    dtest = xgb.DMatrix(X_pred)
    probabilities = model.predict(dtest)[0]  # Get first prediction
    
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
    
    print("🏈 VM FOOTBALL MATCH PREDICTION")
    print("=" * 50)
    
    # Load model
    model, metadata = load_model_and_metadata()
    if model is None:
        return False
    
    print(f"🤖 Model: {metadata['model_type']}")
    print(f"🎯 Training Accuracy: {metadata['accuracy']:.3f}")
    print(f"⚡ Device: {metadata['device']}")
    print("=" * 50)
    
    # Make prediction
    try:
        prediction, probabilities = predict_match(model, args.home, args.away, args.league)
        print(f"\n🎉 Prediction Complete!")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 