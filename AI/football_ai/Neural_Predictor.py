#!/usr/bin/env python3
"""
PYTORCH NEURAL NETWORK PREDICTOR
Live predictions with confidence intervals and probabilities

Usage: python Neural_Predictor.py --predict "Man City" "Liverpool"
       python Neural_Predictor.py --scenarios "Arsenal" "Chelsea"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "mega_features.parquet"
MODEL_FILE = PROJECT_ROOT / "models" / "best_neural_network.pth"

class FootballNet(nn.Module):
    """Deep Neural Network for Football Prediction"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], num_classes=3, dropout=0.3):
        super(FootballNet, self).__init__()
        
        # Build dynamic architecture
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NeuralPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.team_features = {}
        
        print(f"🧠 PyTorch Neural Network Predictor")
        print(f"💻 Device: {self.device}")
        print("=" * 60)
    
    def load_model(self):
        """Load the trained neural network model."""
        if not MODEL_FILE.exists():
            print(f"❌ Model file not found: {MODEL_FILE}")
            print("💡 Please train the model first: python Neural_Trainer.py")
            return False
        
        print("📦 Loading trained model...")
        checkpoint = torch.load(MODEL_FILE, map_location=self.device)
        
        # Determine architecture
        input_size = checkpoint['input_size']
        
        if input_size > 150:
            hidden_sizes = [512, 256, 128, 64, 32]
        elif input_size > 100:
            hidden_sizes = [256, 128, 64, 32]
        else:
            hidden_sizes = [128, 64, 32]
        
        # Create and load model
        self.model = FootballNet(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=3,
            dropout=0.3
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler = checkpoint['scaler']
        
        print(f"✅ Model loaded successfully!")
        print(f"🏆 Best accuracy: {checkpoint['best_accuracy']:.2f}%")
        print(f"🔧 Architecture: {input_size} → {' → '.join(map(str, hidden_sizes))} → 3")
        
        return True
    
    def load_team_data(self):
        """Load and process team feature data."""
        print("📊 Loading team features...")
        df = pd.read_parquet(DATA_FILE)
        
        # Exclude data leakage features
        exclude_cols = [
            'Home_Goals', 'Away_Goals', 'Total_Goals', 'Goal_Difference',
            'Home_Win_Margin', 'Away_Win_Margin', 'Home_xG_Est', 'Away_xG_Est',
            'Result', 'Result_Encoded', 'Date', 'HomeTeam', 'AwayTeam', 
            'League', 'Country', 'Season', 'Round', 'Source', 'MatchID',
            'Home_Formation', 'Away_Formation', 'Weather'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        
        # Create team feature profiles
        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            if home_team not in self.team_features:
                # Extract home team features (features starting with 'Home_')
                home_features = {}
                for col in feature_cols:
                    if col.startswith('Home_'):
                        base_feature = col.replace('Home_', '')
                        home_features[base_feature] = row[col] if pd.notna(row[col]) else 0
                self.team_features[home_team] = home_features
            
            if away_team not in self.team_features:
                # Extract away team features (features starting with 'Away_')
                away_features = {}
                for col in feature_cols:
                    if col.startswith('Away_'):
                        base_feature = col.replace('Away_', '')
                        away_features[base_feature] = row[col] if pd.notna(row[col]) else 0
                self.team_features[away_team] = away_features
        
        self.feature_columns = feature_cols
        print(f"✅ Loaded {len(self.team_features)} teams with {len(feature_cols)} features")
        
        return True
    
    def create_match_features(self, home_team, away_team, scenario="Balanced"):
        """Create feature vector for a match between two teams."""
        if home_team not in self.team_features or away_team not in self.team_features:
            available_teams = list(self.team_features.keys())[:20]  # Show first 20
            print(f"❌ Team not found. Available teams include: {', '.join(available_teams)}...")
            return None
        
        # Start with template from actual data
        template_row = pd.read_parquet(DATA_FILE).iloc[0].copy()
        
        # Get team features
        home_features = self.team_features[home_team]
        away_features = self.team_features[away_team]
        
        # Apply scenario modifications
        scenario_mods = self.get_scenario_modifications(scenario)
        
        # Build feature vector
        feature_vector = []
        for col in self.feature_columns:
            if col.startswith('Home_'):
                base_feature = col.replace('Home_', '')
                value = home_features.get(base_feature, 0)
                
                # Apply scenario modifications
                if base_feature in scenario_mods:
                    value *= scenario_mods[base_feature]
                
                feature_vector.append(value)
                
            elif col.startswith('Away_'):
                base_feature = col.replace('Away_', '')
                value = away_features.get(base_feature, 0)
                
                # Apply scenario modifications
                if base_feature in scenario_mods:
                    value *= scenario_mods[base_feature]
                
                feature_vector.append(value)
            else:
                # Generic features
                feature_vector.append(template_row[col] if pd.notna(template_row[col]) else 0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def get_scenario_modifications(self, scenario):
        """Get feature modifications for different scenarios."""
        scenarios = {
            "Balanced": {},
            "High-scoring": {
                "Goals_Per_Game": 1.3,
                "Attack_Strength": 1.2,
                "xG_Per_Game": 1.25
            },
            "Defensive": {
                "Goals_Per_Game": 0.7,
                "Defense_Strength": 1.3,
                "Goals_Conceded_Per_Game": 0.8
            },
            "Home_Advantage": {
                "Points_Per_Game": 1.2,
                "Win_Rate": 1.15
            },
            "European_Night": {
                "UEFA_Coefficient": 1.1,
                "Competition_Weight": 1.2
            }
        }
        
        return scenarios.get(scenario, {})
    
    def predict_match(self, home_team, away_team, scenarios=None):
        """Predict match outcome using neural network."""
        if scenarios is None:
            scenarios = ["Balanced"]
        
        print(f"\n⚽ MATCH PREDICTION: {home_team} vs {away_team}")
        print("=" * 60)
        
        all_predictions = []
        
        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario}")
            
            # Create features
            features = self.create_match_features(home_team, away_team, scenario)
            if features is None:
                return None
            
            # Normalize features
            features_scaled = self.scaler.transform(features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                confidence = torch.max(F.softmax(outputs, dim=1)).item()
            
            # Format results
            home_prob = probabilities[2] * 100  # Away Win is actually Home Win in our encoding
            draw_prob = probabilities[1] * 100  # Draw
            away_prob = probabilities[0] * 100  # Home Win is actually Away Win
            
            print(f"🏠 {home_team} Win: {home_prob:5.1f}%")
            print(f"🤝 Draw:           {draw_prob:5.1f}%")
            print(f"✈️  {away_team} Win: {away_prob:5.1f}%")
            print(f"🎯 Confidence:     {confidence*100:5.1f}%")
            
            all_predictions.append({
                'scenario': scenario,
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob,
                'confidence': confidence * 100
            })
        
        # Ensemble prediction if multiple scenarios
        if len(scenarios) > 1:
            print(f"\n🎲 ENSEMBLE PREDICTION (Average of {len(scenarios)} scenarios)")
            print("=" * 60)
            
            avg_home = np.mean([p['home_prob'] for p in all_predictions])
            avg_draw = np.mean([p['draw_prob'] for p in all_predictions])
            avg_away = np.mean([p['away_prob'] for p in all_predictions])
            avg_conf = np.mean([p['confidence'] for p in all_predictions])
            
            print(f"🏠 {home_team} Win: {avg_home:5.1f}%")
            print(f"🤝 Draw:           {avg_draw:5.1f}%")
            print(f"✈️  {away_team} Win: {avg_away:5.1f}%")
            print(f"🎯 Avg Confidence: {avg_conf:5.1f}%")
            
            # Determine most likely outcome
            outcomes = [
                (avg_home, f"{home_team} Win"),
                (avg_draw, "Draw"),
                (avg_away, f"{away_team} Win")
            ]
            
            most_likely = max(outcomes, key=lambda x: x[0])
            
            print(f"\n🏆 MOST LIKELY: {most_likely[1]} ({most_likely[0]:.1f}%)")
        
        return all_predictions
    
    def run_prediction(self, home_team, away_team, use_scenarios=False):
        """Run full prediction pipeline."""
        # Load model and data
        if not self.load_model():
            return False
        
        if not self.load_team_data():
            return False
        
        # Set scenarios
        if use_scenarios:
            scenarios = ["Balanced", "High-scoring", "Defensive", "Home_Advantage", "European_Night"]
        else:
            scenarios = ["Balanced"]
        
        # Make predictions
        predictions = self.predict_match(home_team, away_team, scenarios)
        
        if predictions is None:
            return False
        
        print(f"\n✅ Prediction complete!")
        return True

def main():
    """Main function."""
    print("🚀 Neural Network Football Predictor")
    print("Train the model first with: python Neural_Trainer.py --5")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 